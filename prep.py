"""Data preparation: cleaning, lead-time reconstruction, geo features."""
import pandas as pd, numpy as np
from geo import *

OFFSET_BASE = 904      # smallest raw gap observed
YEAR_STEP = 365

def load_clean(path):
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    df["Order Date"] = pd.to_datetime(df["Order Date"], format="%d-%m-%Y")
    df["Ship Date"] = pd.to_datetime(df["Ship Date"], format="%d-%m-%Y")
    df["raw_gap_days"] = (df["Ship Date"] - df["Order Date"]).dt.days
    # Date-shift artifact: raw gap = true lead time + 904 + 365*k  (k = 0,1,2)
    df["shift_k"] = ((df["raw_gap_days"] - OFFSET_BASE) // YEAR_STEP).clip(0, 2)
    df["lead_time"] = df["raw_gap_days"] - OFFSET_BASE - YEAR_STEP*df["shift_k"]
    df["Product Name"] = df["Product Name"].str.strip()
    df["Factory"] = df["Product Name"].map(PRODUCT_FACTORY)
    df["clat"] = df["State/Province"].map(lambda s: STATE_CENTROIDS[s][0])
    df["clon"] = df["State/Province"].map(lambda s: STATE_CENTROIDS[s][1])
    df["flat"] = df["Factory"].map(lambda f: FACTORIES[f][0])
    df["flon"] = df["Factory"].map(lambda f: FACTORIES[f][1])
    df["distance_km"] = haversine(df.flat, df.flon, df.clat, df.clon)
    df["margin"] = df["Gross Profit"] / df["Sales"]
    df["month"] = df["Order Date"].dt.month
    return df

def flag_outliers(df):
    """Extreme (3xIQR) outliers on lead time within ship mode; extreme Units/Sales."""
    out = pd.Series(False, index=df.index)
    for _, g in df.groupby("Ship Mode"):
        q1, q3 = g.lead_time.quantile([.25, .75]); i = q3 - q1
        out |= df.index.isin(g.index[(g.lead_time > q3 + 3*i) | (g.lead_time < q1 - 3*i)])
    for c in ["Units", "Sales"]:
        q1, q3 = df[c].quantile([.25, .75]); i = q3 - q1
        out |= df[c] > q3 + 3*i
    return out
