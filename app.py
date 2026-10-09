"""
Factory Reallocation & Shipping Optimization Recommendation System
Nassau Candy Distributor - Streamlit dashboard

Run:  streamlit run app.py
"""
import os, sys
import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine import (DEFAULTS, FACTORY_LIST, build, scenario_table, recommend_all, kpis, load_shift,
                    route_stats, cluster_routes, congestion)
from geo import FACTORIES, PRODUCT_FACTORY

HERE = os.path.dirname(os.path.abspath(__file__))
ART_PATH = os.path.join(HERE, "artifacts", "artifacts.joblib")
CSV_PATH = os.path.join(HERE, "data", "Nassau_Candy_Distributor.csv")
BLUE, ORANGE, GREEN, GREY, RED = "#1F4E79", "#E07B39", "#2E8B57", "#B0B0B0", "#C0392B"

st.set_page_config(page_title="Nassau Candy | Factory Optimization", page_icon="🍬", layout="wide")


# ------------------------------------------------------------------ data / model loading
@st.cache_resource(show_spinner="Loading data and model ...")
def load_artifacts():
    if os.path.exists(ART_PATH):
        try:
            return joblib.load(ART_PATH)
        except Exception:                       # e.g. library-version mismatch -> retrain below
            pass
    art = build(CSV_PATH)                       # first run: train from raw CSV
    os.makedirs(os.path.dirname(ART_PATH), exist_ok=True)
    joblib.dump(art, ART_PATH)
    return art


@st.cache_data(show_spinner="Simulating every product x region scenario ...")
def cached_recommendations(_art, mode, w_speed, logistics_cost, cost_premium, min_orders):
    return recommend_all(_art, mode=mode, w_speed=w_speed, logistics_cost=logistics_cost,
                         cost_premium=cost_premium, min_orders=min_orders)


@st.cache_data(show_spinner=False)
def cached_kpis(_art, rec_key, mode, w_speed, logistics_cost, cost_premium, min_orders):
    rec = cached_recommendations(_art, mode, w_speed, logistics_cost, cost_premium, min_orders)
    return kpis(_art, rec, w_speed=w_speed, logistics_cost=logistics_cost,
                cost_premium=cost_premium, min_orders=min_orders)


art = load_artifacts()
df = art["df"]
PRODUCTS = sorted(df["Product Name"].unique(), key=lambda p: -len(df[df["Product Name"] == p]))
REGIONS = ["All"] + sorted(df["Region"].unique())
MODES = ["All", "Same Day", "First Class", "Second Class", "Standard Class"]

# ------------------------------------------------------------------ sidebar
st.sidebar.title("🍬 Controls")
product = st.sidebar.selectbox("Product", PRODUCTS, help="Product whose factory assignment you want to explore.")
region = st.sidebar.selectbox("Destination region", REGIONS)
mode = st.sidebar.selectbox("Ship mode", MODES)
st.sidebar.markdown("**Optimization priority**")
priority = st.sidebar.slider("Profit  ⟵  ⟶  Speed", 0, 100, 50, help="0 = rank factories purely by profit impact, "
                             "100 = purely by speed (lead time, distance and route risk).")
w_speed = priority / 100

with st.sidebar.expander("Advanced assumptions"):
    st.caption("Shipping cost and factory cost premiums are NOT in the dataset, so they are explicit, "
               "adjustable assumptions.")
    logistics_cost = st.number_input("Logistics cost ($ per unit per 1,000 km)", 0.0, 2.0, DEFAULTS["logistics_cost"], 0.01)
    cost_premium = st.slider("Production-cost premium at a new factory (%)", 0, 20, int(DEFAULTS["cost_premium"]*100)) / 100
    min_orders = st.slider("Minimum historical orders to recommend a move", 5, 200, DEFAULTS["min_orders"])
params = dict(w_speed=w_speed, logistics_cost=logistics_cost, cost_premium=cost_premium, min_orders=min_orders)

st.sidebar.markdown("---")
st.sidebar.caption(f"Model: **{art['chosen']}**  |  R² = {art['metrics'][art['chosen']]['R2']:.2f}  |  "
                   f"MAE = {art['metrics'][art['chosen']]['MAE']:.2f} days")

# ------------------------------------------------------------------ header + KPIs
st.title("Factory Reallocation & Shipping Optimization")
st.caption("Decision-intelligence system for Nassau Candy Distributor - simulate factory-product reassignments, "
           "quantify the impact before execution and get ranked recommendations.")

rec_all = cached_recommendations(art, mode, **params)
k = cached_kpis(art, "k", mode, **params)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Lead Time Reduction", f"{k['lead_reduction_pct']:.2f}%", help="Order-weighted predicted lead-time change on recommended moves.")
c2.metric("Profit Impact Stability", f"{k['profit_stability']:.0f}%",
          help="Share of recommendations that keep profit within tolerance when cost assumptions are stressed.")
c3.metric("Scenario Confidence", f"{k['confidence']:.0f}/100", help="Evidence behind each scenario x model reliability.")
c4.metric("Recommendation Coverage", f"{k['coverage_volume']:.0f}% of orders",
          f"{k['products_covered']:.0f} of {k['products_total']:.0f} products", delta_color="off")
c5, c6, c7 = st.columns(3)
c5.metric("Avg. shipping distance saved", f"{k['avg_dist_reduction_km']:,.0f} km")
c6.metric("Net profit impact (history)", f"${k['net_profit_delta']:,.0f}")
c7.metric("Moves recommended", f"{int(rec_all.reassign.sum())} of {len(rec_all)} product-region scopes")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["🏭 Factory Simulator", "🔀 What-If Analysis", "⭐ Recommendations",
                                        "⚠️ Risk & Impact", "📚 Model & Data"])

# ================================================================== helpers
mode_arg = None if mode == "All" else mode
region_arg = None if region == "All" else region
native = PRODUCT_FACTORY[product]
table = scenario_table(art, product, region, mode_arg, **params)


def best_row(t):
    cands = t[(~t.Current) & (t.score > t[t.Current].score.iloc[0] + DEFAULTS["min_score_gain"]) &
              (t.profit_pct >= -DEFAULTS["max_profit_loss"]*100) &
              (t.load_increase_pct <= DEFAULTS["max_load_increase"])]
    if t.empty or t[t.Current].orders.iloc[0] < min_orders or cands.empty:
        return None
    return cands.iloc[0]


def pretty(t):
    o = t[["rank", "Factory", "Current", "pred_lead", "lead_change_pct", "avg_distance_km", "distance_change_km",
           "volatility", "profit_pct", "confidence", "score", "flag"]].copy()
    o.columns = ["Rank", "Factory", "Current?", "Pred. lead (d)", "Lead Δ %", "Avg dist (km)", "Dist Δ (km)",
                 "Route volatility (d)", "Profit Δ %", "Confidence", "Score", "Flags"]
    o["Current?"] = o["Current?"].map({True: "✔", False: ""})
    return o.round(2)


# ================================================================== TAB 1 - SIMULATOR
with tab1:
    st.subheader(f"{product}")
    if table.empty:
        st.warning("No historical orders for this combination - try another region or ship mode.")
    else:
        n_orders = int(table.orders.iloc[0])
        st.caption(f"Currently produced at **{native}**. Simulation uses {n_orders:,} historical orders "
                   f"(region: {region}, ship mode: {mode}).")
        if n_orders < min_orders:
            st.warning(f"Only {n_orders} historical orders - below the minimum of {min_orders}. "
                       "Results are shown for exploration but no recommendation will be made.")
        b = best_row(table)
        colors = [GREEN if (b is not None and f == b.Factory) else (ORANGE if cur else GREY)
                  for f, cur in zip(table.Factory, table.Current)]
        a, bcol = st.columns(2)
        fig = go.Figure(go.Bar(x=table.Factory, y=table.pred_lead, marker_color=colors,
                               text=table.pred_lead.round(2), textposition="outside"))
        fig.update_layout(title="Predicted lead time by factory (days)", yaxis_title="days", height=360,
                          margin=dict(t=50, b=10))
        lo, hi = table.pred_lead.min(), table.pred_lead.max()
        fig.update_yaxes(range=[max(0, lo - 0.6), hi + 0.6])
        a.plotly_chart(fig)
        fig = go.Figure(go.Bar(x=table.Factory, y=table.avg_distance_km, marker_color=colors,
                               text=table.avg_distance_km.round(0), textposition="outside"))
        fig.update_layout(title="Average shipping distance (km)", yaxis_title="km", height=360, margin=dict(t=50, b=10))
        bcol.plotly_chart(fig)
        st.caption("🟧 current factory   🟩 recommended factory   ⬜ other candidates. "
                   "Note: the y-axis of the lead-time chart is zoomed - differences are small (see Model & Data).")
        if b is not None:
            st.success(f"**Recommendation:** move *{product}* from **{native}** to **{b.Factory}** "
                       f"({b.distance_change_km:,.0f} km shipping distance, {b.profit_pct:+.1f}% profit, "
                       f"lead time {b.lead_change_pct:+.2f}%, confidence {b.confidence:.0f}/100).")
        else:
            st.info("**No reassignment recommended** for this selection under the current priority, guardrails and assumptions.")
        st.dataframe(pretty(table), hide_index=True)

        # map
        cust = df[df["Product Name"] == product]
        if region_arg: cust = cust[cust.Region == region_arg]
        if mode_arg: cust = cust[cust["Ship Mode"] == mode_arg]
        cs = cust.groupby("State/Province").agg(lat=("clat", "first"), lon=("clon", "first"), n=("Units", "size")).reset_index()
        m = go.Figure()
        m.add_trace(go.Scattergeo(lat=cs.lat, lon=cs.lon, text=cs["State/Province"] + ": " + cs.n.astype(str) + " orders",
                                  mode="markers", marker=dict(size=(cs.n**0.5*3).clip(4, 30), color="#9DB9D6", line=dict(width=1, color=BLUE)),
                                  name="Customer states"))
        m.add_trace(go.Scattergeo(lat=[v[0] for v in FACTORIES.values()], lon=[v[1] for v in FACTORIES.values()],
                                  text=list(FACTORIES), mode="markers+text", textposition="top center",
                                  marker=dict(size=14, symbol="star", color=ORANGE, line=dict(width=1, color="black")), name="Factories"))
        if b is not None:
            a_, b_ = FACTORIES[native], FACTORIES[b.Factory]
            m.add_trace(go.Scattergeo(lat=[a_[0], b_[0]], lon=[a_[1], b_[1]], mode="lines",
                                      line=dict(width=3, color=RED, dash="dash"), name="Recommended move"))
        m.update_geos(scope="north america", showland=True, landcolor="#F4F4F4", showcountries=True, lataxis_range=[23, 58], lonaxis_range=[-130, -60])
        m.update_layout(height=430, margin=dict(t=10, b=0, l=0, r=0), legend=dict(orientation="h"))
        st.plotly_chart(m)

# ================================================================== TAB 2 - WHAT-IF
with tab2:
    st.subheader("Compare current vs. alternative assignment")
    if table.empty:
        st.warning("No data for this selection.")
    else:
        b = best_row(table)
        options = [f for f in FACTORY_LIST if f != native]
        default_idx = options.index(b.Factory) if b is not None else 0
        alt = st.selectbox("Alternative factory", options, index=default_idx,
                           help="Defaults to the recommended factory when one exists.")
        cur_r = table[table.Current].iloc[0]; alt_r = table[table.Factory == alt].iloc[0]
        x1, x2, x3, x4 = st.columns(4)
        x1.metric("Predicted lead time", f"{alt_r.pred_lead:.2f} d", f"{alt_r.lead_change_days:+.3f} d", delta_color="inverse")
        x2.metric("Shipping distance", f"{alt_r.avg_distance_km:,.0f} km", f"{alt_r.distance_change_km:+,.0f} km", delta_color="inverse")
        x3.metric("Profit impact (history)", f"${alt_r.profit_delta:,.0f}", f"{alt_r.profit_pct:+.1f}%")
        x4.metric("Route volatility", f"{alt_r.volatility:.2f} d", f"{alt_r.risk_change:+.2f} d", delta_color="inverse")
        if alt_r.flag not in ("OK",):
            st.warning(f"Flags: {alt_r.flag}")

        # lead time by ship mode & region: current vs alternative
        rows = []
        for md in MODES[1:]:
            t = scenario_table(art, product, region, md, **params)
            if t.empty: continue
            rows.append(dict(Mode=md, Current=t[t.Current].pred_lead.iloc[0], Alternative=t[t.Factory == alt].pred_lead.iloc[0],
                             CurrentKm=t[t.Current].avg_distance_km.iloc[0], AlternativeKm=t[t.Factory == alt].avg_distance_km.iloc[0]))
        rows_r = []
        for rg in REGIONS[1:]:
            t = scenario_table(art, product, rg, mode_arg, **params)
            if t.empty: continue
            rows_r.append(dict(Region=rg, Current=t[t.Current].avg_distance_km.iloc[0], Alternative=t[t.Factory == alt].avg_distance_km.iloc[0]))
        g1, g2 = st.columns(2)
        if rows:
            d1 = pd.DataFrame(rows).melt("Mode", ["Current", "Alternative"], var_name="Assignment", value_name="Predicted lead (days)")
            f1 = px.bar(d1, x="Mode", y="Predicted lead (days)", color="Assignment", barmode="group",
                        color_discrete_map={"Current": GREY, "Alternative": GREEN}, title="Lead time by ship mode")
            f1.update_layout(height=340, margin=dict(t=50, b=10)); g1.plotly_chart(f1)
        if rows_r:
            d2 = pd.DataFrame(rows_r).melt("Region", ["Current", "Alternative"], var_name="Assignment", value_name="Avg distance (km)")
            f2 = px.bar(d2, x="Region", y="Avg distance (km)", color="Assignment", barmode="group",
                        color_discrete_map={"Current": GREY, "Alternative": GREEN}, title="Shipping distance by destination region")
            f2.update_layout(height=340, margin=dict(t=50, b=10)); g2.plotly_chart(f2)
        st.info("Interpretation: in the historical data, lead time is driven almost entirely by **ship mode**; moving production "
                "shortens physical distance (and logistics cost) far more than it changes predicted lead time.")

# ================================================================== TAB 3 - RECOMMENDATIONS
with tab3:
    st.subheader("Ranked factory reassignment recommendations")
    rr = rec_all[rec_all.reassign].copy()
    if region_arg: rr = rr[rr.Region == region_arg]
    topn = st.slider("Show top N", 1, max(1, min(30, len(rec_all))), min(10, max(1, len(rr))) if len(rr) else 1)
    if rr.empty:
        st.info("No reassignments pass the guardrails for the current filters/assumptions.")
    else:
        rr = rr.sort_values("score_gain", ascending=False).head(topn)
        show = rr[["Product", "Region", "Mode", "current_factory", "recommended_factory", "orders", "lead_change_pct",
                   "distance_change_km", "profit_delta", "profit_pct", "confidence", "score_gain"]].copy()
        show.columns = ["Product", "Region", "Ship mode", "From", "To", "Orders", "Lead Δ %", "Dist Δ (km)", "Profit Δ ($)",
                        "Profit Δ %", "Confidence", "Score gain"]
        st.dataframe(show.round(2), hide_index=True)
        st.download_button("⬇ Download all recommendations (CSV)", rec_all.to_csv(index=False).encode(),
                           "factory_reallocation_recommendations.csv", "text/csv")
        lab = rr.Product.str.replace("Wonka Bar - ", "").str.replace("Wonka Bar -", "") + " | " + rr.Region
        fig = px.bar(x=-rr.distance_change_km, y=lab, orientation="h", color=rr.profit_pct,
                     color_continuous_scale="Greens", labels={"x": "Shipping distance saved (km)", "y": "", "color": "Profit Δ %"},
                     title="Expected efficiency gains")
        fig.update_layout(height=max(320, 40*len(rr)), yaxis=dict(autorange="reversed"), margin=dict(t=50, b=10))
        st.plotly_chart(fig)
    st.markdown("**Not recommended / insufficient evidence**")
    rest = rec_all[~rec_all.reassign]
    if region_arg: rest = rest[rest.Region == region_arg]
    st.dataframe(rest[["Product", "Region", "Mode", "current_factory", "orders", "flag"]]
                 .rename(columns={"current_factory": "Current factory", "orders": "Orders", "flag": "Reason"}),
                 hide_index=True)

# ================================================================== TAB 4 - RISK
with tab4:
    st.subheader("Risk & impact panel")
    r1, r2 = st.columns(2)
    ls = load_shift(art, rec_all)
    fig = px.bar(ls, x="Factory", y="change_pct", title="Factory volume change if ALL recommendations are applied",
                 labels={"change_pct": "% change in units"})
    fig.update_traces(marker_color=[RED if v > DEFAULTS["max_load_increase"] else BLUE for v in ls.change_pct])
    fig.add_hline(y=DEFAULTS["max_load_increase"], line_dash="dot", line_color=RED)
    fig.update_layout(height=340, margin=dict(t=50, b=10)); r1.plotly_chart(fig)
    over = ls[ls.change_pct > DEFAULTS["max_load_increase"]]
    with r2:
        st.markdown("**High-risk reassignment warnings**")
        if len(over):
            for _, x in over.iterrows():
                st.warning(f"**{x.Factory}** would absorb **{x.change_pct:+.0f}%** more volume if every move is executed at once. "
                           "Phase the rollout and confirm capacity first.")
        else:
            st.success("No factory exceeds the capacity-strain threshold after all recommended moves.")
        st.markdown("**Profit impact alerts**")
        neg = rec_all[(rec_all.reassign) & (rec_all.profit_pct < 0)]
        if neg.empty:
            st.success("All recommended moves are profit-positive under the current assumptions.")
        else:
            st.error(f"{len(neg)} recommended move(s) lose profit.")
        low = rec_all[(rec_all.reassign) & (rec_all.confidence < 40)]
        if len(low): st.warning(f"{len(low)} recommended move(s) have low confidence (<40).")
        st.metric("Profit Impact Stability", f"{k['profit_stability']:.0f}%")

    st.markdown("**Scenario flags - every alternative factory for the selected product**")
    if not table.empty:
        st.dataframe(pretty(table)[["Factory", "Profit Δ %", "Dist Δ (km)", "Confidence", "Flags"]], hide_index=True)

    st.markdown("**Sensitivity of the recommendation set to cost assumptions**")
    if st.button("Run sensitivity analysis (3 x 3 grid)"):
        sens = []
        for lc in [0.05, 0.10, 0.20]:
            for cp in [0.0, 0.03, 0.08]:
                r_ = cached_recommendations(art, mode, w_speed, lc, cp, min_orders)
                sens.append({"Logistics $/unit/1000km": lc, "Production premium": f"{cp:.0%}",
                             "Moves recommended": int(r_.reassign.sum()),
                             "Net profit impact ($)": round(float(r_[r_.reassign].profit_delta.sum()))})
        st.dataframe(pd.DataFrame(sens), hide_index=True)
    else:
        st.caption("Click to re-run the recommendation engine under 9 combinations of logistics cost and production premium.")

# ================================================================== TAB 5 - MODEL & DATA
with tab5:
    st.subheader("Predictive model evaluation")
    met = pd.DataFrame(art["metrics"]).T[["RMSE", "MAE", "R2", "CV_RMSE"]].round(3)
    tmp = pd.DataFrame(art["temporal"]).T[["RMSE", "R2"]].round(3).add_prefix("Time-holdout ")
    st.dataframe(pd.concat([met, tmp], axis=1))
    st.caption(f"Selected model: **{art['chosen']}** (simplest model within 1% of the best average RMSE). "
               f"Naive benchmark (mean lead time per ship mode) RMSE = {art['naive_rmse']:.3f}; "
               f"predicting the global mean gives {art['mean_rmse']:.3f}.")
    imp = art["importance"].head(10)[::-1]
    fig = px.bar(x=imp.values, y=[i.split("__")[1] for i in imp.index], orientation="h",
                 title="Top model coefficients / importances", labels={"x": "", "y": ""})
    fig.update_layout(height=340, margin=dict(t=50, b=10)); st.plotly_chart(fig)

    st.subheader("Route & product clustering")
    cl = cluster_routes(art["route_stats"])
    fig = px.scatter(cl, x="excess_lead", y="std", size="n", color="cluster_label", hover_data=["Factory", "Region", "Ship Mode"],
                     color_discrete_map={"Fast / reliable": GREEN, "Typical": GREY, "Consistently slow": RED},
                     labels={"excess_lead": "Excess lead vs ship-mode mean (days)", "std": "Lead-time std (days)"})
    fig.update_layout(height=380, margin=dict(t=20, b=10)); st.plotly_chart(fig)
    cg = congestion(df)
    st.markdown("**Congested region-product combinations** (high volume and above-average lead time)")
    st.dataframe(cg[cg.congested].sort_values("orders", ascending=False)[["Product Name", "Region", "orders", "excess_lead"]].round(3), hide_index=True)

    st.subheader("Data-quality note")
    raw = art["raw"]
    st.markdown(
        f"The raw `Ship Date − Order Date` gap ranges from **{int(raw.raw_gap_days.min()):,} to {int(raw.raw_gap_days.max()):,} days** "
        "(a date-shifting artifact: orders were moved to 2024-25 but ship dates were shifted by whole years). "
        "The gaps form three tight clusters exactly 365 days apart that match the year in each Order ID, so the whole-year offset "
        "was removed to recover a realistic lead time (0-11 days, mean "
        f"{raw.lead_time.mean():.1f}). After correction, lead time lines up with ship mode as expected.")
    st.caption(f"{len(raw):,} rows loaded - {int(raw.outlier.sum())} extreme outliers excluded from modelling.")
