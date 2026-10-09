"""Core analytics engine: modelling, route clustering, scenario simulation, recommendations, KPIs."""
import numpy as np, pandas as pd, json, joblib
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.cluster import KMeans
from geo import *
from prep import load_clean, flag_outliers

NUM = ["distance_km", "Units", "log_sales", "month"]
CAT = ["Ship Mode", "Region", "Factory"]
FACTORY_LIST = list(FACTORIES)

DEFAULTS = dict(
    logistics_cost=0.10,   # $ per unit per 1,000 km (ASSUMPTION - not in dataset)
    cost_premium=0.03,     # production-cost premium at a non-native factory (ASSUMPTION)
    w_speed=0.5,           # optimisation priority: 0 = pure profit, 1 = pure speed
    min_score_gain=0.03,   # minimum composite-score gain to recommend a move
    max_profit_loss=0.01,  # guardrail: max tolerated profit loss (fraction of baseline GP)
    min_orders=30,         # guardrail: minimum historical orders in a scope before recommending
    max_load_increase=25,  # guardrail: max % volume a receiving factory may absorb from one move
)
SPEED_WEIGHTS = dict(lead=0.35, dist=0.45, risk=0.20)

# ------------------------------------------------------------------ modelling
def feat(df):
    X = df.copy()
    X["log_sales"] = np.log1p(X["Sales"])
    return X[NUM + CAT]

def make_pipe(model):
    pre = ColumnTransformer([("n", StandardScaler(), NUM),
                             ("c", OneHotEncoder(handle_unknown="ignore"), CAT)])
    return Pipeline([("pre", pre), ("m", model)])

def evaluate(models, Xtr, ytr, Xte, yte):
    rows = {}
    for name, m in models.items():
        p = make_pipe(m).fit(Xtr, ytr)
        pr = p.predict(Xte)
        cv = cross_val_score(make_pipe(m), Xtr, ytr, cv=5, scoring="neg_root_mean_squared_error")
        rows[name] = dict(RMSE=float(np.sqrt(mean_squared_error(yte, pr))),
                          MAE=float(mean_absolute_error(yte, pr)),
                          R2=float(r2_score(yte, pr)), CV_RMSE=float(-cv.mean()))
    return rows

def build(path, seed=42):
    raw = load_clean(path)
    raw["outlier"] = flag_outliers(raw)
    df = raw[~raw.outlier].copy().reset_index(drop=True)
    X, y = feat(df), df["lead_time"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=seed)
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=200, min_samples_leaf=20, n_jobs=-1, random_state=seed),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=150, max_depth=3, learning_rate=0.05,
                                                       subsample=0.8, random_state=seed),
    }
    metrics = evaluate(models, Xtr, ytr, Xte, yte)
    # temporal robustness check: train on orders before Oct-2025, test on Q4-2025
    cut = pd.Timestamp("2025-10-01"); tr, te = df["Order Date"] < cut, df["Order Date"] >= cut
    temporal = evaluate(models, X[tr], y[tr], X[te], y[te])
    # naive benchmark: predict mean lead time of each ship mode
    mm = ytr.groupby(Xtr["Ship Mode"]).mean()
    naive = float(np.sqrt(mean_squared_error(yte, Xte["Ship Mode"].map(mm))))
    base_rmse = float(np.sqrt(mean_squared_error(yte, np.full(len(yte), ytr.mean()))))
    # selection: lowest RMSE, but prefer the simpler/interpretable model if within 1%
    order = ["Linear Regression", "Random Forest", "Gradient Boosting"]
    # accuracy = mean of random-split RMSE and time-holdout RMSE; parsimony: simplest model within 1% of best
    acc = {n: (metrics[n]["RMSE"] + temporal[n]["RMSE"])/2 for n in order}
    best = min(acc.values())
    chosen = next(n for n in order if acc[n] <= best*1.01)
    final = make_pipe(models[chosen]).fit(X, y)           # refit on all clean data
    # feature importance / coefficients (for interpretability)
    names = final.named_steps["pre"].get_feature_names_out()
    est = final.named_steps["m"]
    imp = est.coef_ if hasattr(est, "coef_") else est.feature_importances_
    importance = pd.Series(imp, index=names).sort_values(key=abs, ascending=False)
    return dict(raw=raw, df=df, model=final, metrics=metrics, temporal=temporal, chosen=chosen,
                naive_rmse=naive, mean_rmse=base_rmse, importance=importance,
                route_stats=route_stats(df))

# ------------------------------------------------------------------ route analysis
def route_stats(df):
    g = df.groupby(["Factory", "Region", "Ship Mode"]).agg(
        n=("lead_time", "size"), lead=("lead_time", "mean"), std=("lead_time", "std"),
        dist=("distance_km", "mean"), gp=("Gross Profit", "mean"), units=("Units", "sum")).reset_index()
    mode_mean = df.groupby("Ship Mode").lead_time.mean()
    g["excess_lead"] = g["lead"] - g["Ship Mode"].map(mode_mean)
    g["std"] = g["std"].fillna(0)
    return g

def cluster_routes(rs, k=3, min_n=30, seed=42):
    r = rs[rs.n >= min_n].copy()
    Z = StandardScaler().fit_transform(np.c_[r.excess_lead, r["std"], np.log(r.n), r.dist])
    r["cluster"] = KMeans(k, n_init=10, random_state=seed).fit_predict(Z)
    order = r.groupby("cluster").excess_lead.mean().sort_values()
    lab = {order.index[0]: "Fast / reliable", order.index[-1]: "Consistently slow"}
    r["cluster_label"] = r.cluster.map(lambda c: lab.get(c, "Typical"))
    return r

def congestion(df):
    t = df.groupby(["Product Name", "Region"]).agg(orders=("lead_time", "size"), lead=("lead_time", "mean"),
                                                   units=("Units", "sum")).reset_index()
    mm = df.groupby("Ship Mode").lead_time.mean()
    df = df.assign(excess=df.lead_time - df["Ship Mode"].map(mm))
    t["excess_lead"] = t.set_index(["Product Name", "Region"]).index.map(
        df.groupby(["Product Name", "Region"]).excess.mean())
    t["volume_share"] = t.orders / t.orders.sum()
    t["congested"] = (t.volume_share >= t.volume_share.quantile(.6)) & (t.excess_lead > 0)
    return t

# ------------------------------------------------------------------ scenario simulation
def _swap(scope, factory):
    s = scope.copy()
    s["Factory"] = factory
    f = FACTORIES[factory]
    s["distance_km"] = haversine(f[0], f[1], s.clat, s.clon)
    return s

def _rs_index(art):
    """Cached lookups: (factory, region, mode) -> (n, std); mode -> global std."""
    if "_rsd" not in art:
        rs, df = art["route_stats"], art["df"]
        art["_rsd"] = {(r.Factory, r.Region, r["Ship Mode"]): (int(r.n), float(r["std"])) for _, r in rs.iterrows()}
        art["_gsd"] = df.groupby("Ship Mode").lead_time.std().to_dict()
    return art["_rsd"], art["_gsd"]

def _volatility(art, factory, region, mode, k=20):
    """Shrunk route volatility (std of lead time) and number of supporting orders."""
    rsd, gsd = _rs_index(art)
    gl = gsd[mode]
    if (factory, region, mode) not in rsd: return gl, 0
    n, s = rsd[(factory, region, mode)]
    return (n*s + k*gl)/(n + k), n

def scenario_table(art, product, region=None, mode=None, **p):
    """Evaluate all five factories for a product within an (optional) region / ship-mode filter."""
    P = {**DEFAULTS, **p}
    df, model, rs = art["df"], art["model"], art["route_stats"]
    scope = df[df["Product Name"] == product]
    if region and region != "All": scope = scope[scope.Region == region]
    if mode and mode != "All": scope = scope[scope["Ship Mode"] == mode]
    if scope.empty: return pd.DataFrame()
    native = PRODUCT_FACTORY[product]
    fac_units = df.groupby("Factory").Units.sum()
    base_pred = model.predict(feat(scope)).mean()
    base_dist = scope.distance_km.mean()
    # blended volatility of the scope (weighted by region x mode mix)
    mix = scope.groupby(["Region", "Ship Mode"]).size()
    def vol(fac):
        v = sum(_volatility(art, fac, r, m)[0]*c for (r, m), c in mix.items())/mix.sum(); return v
    base_vol = vol(native)
    # evidence behind each candidate (orders that factory already shipped to this region/mode mix)
    rows = []
    for f in FACTORY_LIST:
        s = _swap(scope, f) if f != native else scope
        pred = model.predict(feat(s)).mean()
        dist = s.distance_km.mean()
        d_km = (s.distance_km.values - scope.distance_km.values)
        logistics = P["logistics_cost"] * float((scope.Units.values * d_km).sum())/1000
        premium = 0.0 if f == native else P["cost_premium"] * float(scope.Cost.sum())
        profit_delta = -(logistics + premium)
        base_gp = float(scope["Gross Profit"].sum())
        profit_pct = profit_delta/base_gp if base_gp else 0
        v = vol(f)
        support = sum(_volatility(art, f, r, m)[1]*c for (r, m), c in mix.items())/mix.sum()
        # Scenario Confidence = evidence behind the candidate route x model reliability (R2), 0-100
        conf = (support/(support + 30)) * max(art["metrics"][art["chosen"]]["R2"], 0) * 100
        load_increase = 0 if f == native else float(scope.Units.sum())/float(fac_units[f])
        lead_n = np.clip(.5 + (base_pred - pred)/2.0, 0, 1)
        dist_n = np.clip(.5 + (base_dist - dist)/3000, 0, 1)
        risk_n = np.clip(.5 + (base_vol - v)/2.0, 0, 1)
        profit_n = np.clip(.5 + profit_pct/0.20, 0, 1)
        speed = SPEED_WEIGHTS["lead"]*lead_n + SPEED_WEIGHTS["dist"]*dist_n + SPEED_WEIGHTS["risk"]*risk_n
        score = P["w_speed"]*speed + (1 - P["w_speed"])*profit_n
        rows.append(dict(Factory=f, Current=(f == native), orders=len(scope),
                         pred_lead=pred, lead_change_days=pred - base_pred,
                         lead_change_pct=(pred - base_pred)/base_pred*100 if base_pred else 0,
                         avg_distance_km=dist, distance_change_km=dist - base_dist,
                         volatility=v, risk_change=v - base_vol,
                         profit_delta=profit_delta, profit_pct=profit_pct*100,
                         load_increase_pct=load_increase*100, confidence=conf,
                         speed_index=speed, score=score))
    t = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    t["rank"] = np.arange(1, len(t) + 1)
    t["flag"] = t.apply(lambda r: flags(r, P), axis=1)
    return t

def flags(r, P):
    out = []
    if r.Current: return "Current assignment"
    if r.profit_pct < -P["max_profit_loss"]*100: out.append("Profit impact alert")
    if r.distance_change_km > 0: out.append("Longer shipping distance")
    if r.load_increase_pct > P["max_load_increase"]: out.append(f"Capacity strain (>{P['max_load_increase']}% volume added)")
    if r.confidence < 40: out.append("Low confidence")
    if not out: return "OK"
    return "; ".join(out)

def best_option(t, P=None):
    P = {**DEFAULTS, **(P or {})}
    if t.empty: return None
    cur = t[t.Current].iloc[0]
    if cur.orders < P["min_orders"]: return None
    cand = t[(~t.Current) & (t.score >= cur.score + P["min_score_gain"]) &
             (t.profit_pct >= -P["max_profit_loss"]*100) &
             (t.load_increase_pct <= P["max_load_increase"])]
    return None if cand.empty else cand.sort_values("score", ascending=False).iloc[0]

def recommend_all(art, by_mode=False, mode=None, **p):
    """Top reassignment per product x region (x ship mode). `mode` restricts to one ship mode."""
    P = {**DEFAULTS, **p}
    df = art["df"]; out = []
    if mode and mode != "All":
        df = df[df["Ship Mode"] == mode]
    keys = ["Product Name", "Region"] + (["Ship Mode"] if by_mode else [])
    for k, g in df.groupby(keys):
        prod, reg = k[0], k[1]; mode_k = k[2] if by_mode else None
        t = scenario_table(art, prod, reg, mode_k or (mode if mode and mode != "All" else None), **p)
        if t.empty: continue
        cur = t[t.Current].iloc[0]; b = best_option(t, P)
        rec = dict(Product=prod, Region=reg, Mode=(k[2] if by_mode else (mode or "All")), orders=len(g),
                   units=int(g.Units.sum()), current_factory=cur.Factory,
                   current_lead=cur.pred_lead, current_dist=cur.avg_distance_km,
                   baseline_gp=float(g["Gross Profit"].sum()))
        if b is not None:
            rec.update(recommended_factory=b.Factory, new_lead=b.pred_lead, new_dist=b.avg_distance_km,
                       lead_change_pct=b.lead_change_pct, distance_change_km=b.distance_change_km,
                       profit_delta=b.profit_delta, profit_pct=b.profit_pct, score=b.score,
                       score_gain=b.score - cur.score, confidence=b.confidence, flag=b.flag, reassign=True)
        else:
            why = "Insufficient data" if cur.orders < P["min_orders"] else "Keep current"
            rec.update(recommended_factory=cur.Factory, new_lead=cur.pred_lead, new_dist=cur.avg_distance_km,
                       lead_change_pct=0, distance_change_km=0, profit_delta=0, profit_pct=0,
                       score=cur.score, score_gain=0, confidence=cur.confidence, flag=why, reassign=False)
        out.append(rec)
    return pd.DataFrame(out)

def kpis(art, rec, **p):
    """Four project KPIs, computed over the recommendation set."""
    P = {**DEFAULTS, **p}
    r = rec[rec.reassign]
    out = dict(coverage_scopes=len(r)/len(rec)*100 if len(rec) else 0,
               coverage_volume=rec[rec.reassign].units.sum()/rec.units.sum()*100 if len(rec) else 0,
               products_covered=r["Product"].nunique(), products_total=rec["Product"].nunique())
    if r.empty:
        out.update(lead_reduction_pct=0, profit_stability=100, confidence=0, net_profit_delta=0, avg_dist_reduction_km=0)
        return out
    w = r.orders
    out["lead_reduction_pct"] = float(-(r.lead_change_pct*w).sum()/w.sum())
    out["avg_dist_reduction_km"] = float(-(r.distance_change_km*w).sum()/w.sum())
    out["confidence"] = float((r.confidence*w).sum()/w.sum())
    out["net_profit_delta"] = float(r.profit_delta.sum())
    # stability: share of recommendations that stay profit-neutral-or-better under assumption stress
    stable = 0
    for _, x in r.iterrows():
        ok = True
        for lc, cp in [(P["logistics_cost"]*0.5, P["cost_premium"]), (P["logistics_cost"], P["cost_premium"] + 0.02),
                       (P["logistics_cost"]*0.5, P["cost_premium"] + 0.02)]:
            t = scenario_table(art, x.Product, x.Region, None if x.Mode == "All" else x.Mode,
                               **{**P, "logistics_cost": lc, "cost_premium": cp})
            m = t[t.Factory == x.recommended_factory]
            if m.empty or m.profit_pct.iloc[0] < -P["max_profit_loss"]*100: ok = False; break
        stable += ok
    out["profit_stability"] = stable/len(r)*100
    return out

def load_shift(art, rec):
    """Cumulative factory volume (units) before / after applying all recommended reassignments."""
    before = art["df"].groupby("Factory").Units.sum().reindex(FACTORY_LIST).fillna(0)
    after = before.copy()
    for _, x in rec[rec.reassign].iterrows():
        after[x.current_factory] -= x.units; after[x.recommended_factory] += x.units
    t = pd.DataFrame({"units_before": before, "units_after": after})
    t["change_pct"] = (t.units_after/t.units_before - 1)*100
    return t.reset_index(names="Factory")
