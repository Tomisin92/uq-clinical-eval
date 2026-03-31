"""
preprocess.py
-------------
MIMIC-IV data extraction and preprocessing pipeline.

Expects the following MIMIC-IV files (compressed .csv.gz) in:
  data/hosp/   <- admissions.csv.gz, patients.csv.gz, labevents.csv.gz
  data/icu/    <- icustays.csv.gz, chartevents.csv.gz

MIMIC-IV changes from MIMIC-III:
  - Folder structure: hosp/ and icu/ subdirectories
  - Files are .csv.gz (pandas reads these directly)
  - All column names are lowercase  (hadm_id not HADM_ID)
  - patients.csv: no DOB — uses anchor_age + anchor_year instead
  - admissions.csv: hospital_expire_flag still present
  - icustays.csv: stay_id replaces icustay_id
  - chartevents.csv: stay_id column (not icustay_id)
  - labevents.csv: charttime column (same as before)
  - MIMIC-IV covers ~2008-2019 → use 2008-2016 train, 2017-2019 shift

Outputs:
  - data/processed/mortality_features.csv
  - data/processed/readmission_30d_features.csv
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

BASE_DIR  = os.path.dirname(__file__)
HOSP_DIR  = os.path.join(BASE_DIR, "hosp")
ICU_DIR   = os.path.join(BASE_DIR, "icu")
PROC_DIR  = os.path.join(BASE_DIR, "processed")
os.makedirs(PROC_DIR, exist_ok=True)

# ── Vital sign ItemIDs (MIMIC-IV MetaVision IDs only) ───────────────────────
VITAL_ITEMIDS = {
    "heart_rate":    [220045],
    "sbp":           [220179, 220050],
    "dbp":           [220180, 220051],
    "mbp":           [220052, 220181],
    "resp_rate":     [220210, 224690],
    "temp_c":        [223761, 226329],
    "spo2":          [220277],
    "glucose_chart": [225664, 220621, 226537],
}

# ── Lab ItemIDs (hosp/labevents) ─────────────────────────────────────────────
LAB_ITEMIDS = {
    "bun":         [51006],
    "creatinine":  [50912],
    "sodium":      [50983],
    "potassium":   [50971],
    "bicarbonate": [50882],
    "hemoglobin":  [51222],
    "wbc":         [51301],
    "platelets":   [51265],
    "lactate":     [50813],
    "alt":         [50861],
    "ast":         [50878],
    "bilirubin":   [50885],
}

STATS = ["mean", "min", "max", "std"]


def load_admissions():
    adm = pd.read_csv(
        os.path.join(HOSP_DIR, "admissions.csv.gz"),
        low_memory=False,
        parse_dates=["admittime", "dischtime", "deathtime"],
    )
    pat = pd.read_csv(
        os.path.join(HOSP_DIR, "patients.csv.gz"),
        low_memory=False,
    )
    icu = pd.read_csv(
        os.path.join(ICU_DIR, "icustays.csv.gz"),
        low_memory=False,
        parse_dates=["intime", "outtime"],
    )

    # Adult ICU stays >= 24 hours
    icu["los_hours"] = (icu["outtime"] - icu["intime"]).dt.total_seconds() / 3600
    icu = icu[icu["los_hours"] >= 24].copy()

    # Merge admissions
    df = icu.merge(
        adm[["hadm_id", "admittime", "dischtime", "deathtime",
             "hospital_expire_flag", "admission_type"]],
        on="hadm_id", how="left",
    )

    # Merge patients — MIMIC-IV uses anchor_age/anchor_year instead of DOB
    df = df.merge(
        pat[["subject_id", "anchor_age", "anchor_year", "gender"]],
        on="subject_id", how="left",
    )

    # Compute age at admission
    df["admit_year"] = df["admittime"].dt.year
    df["age"] = df["anchor_age"] + (df["admit_year"] - df["anchor_year"])
    df = df[(df["age"] >= 18) & (df["age"] < 120)].copy()

    # Mortality label
    df["mortality"] = df["hospital_expire_flag"].astype(int)

    # 30-day readmission label
    df = df.sort_values(["subject_id", "admittime"])
    df["next_admittime"] = df.groupby("subject_id")["admittime"].shift(-1)
    df["days_to_next"]   = (df["next_admittime"] - df["dischtime"]).dt.days
    df["readmission_30d"] = (
        (df["days_to_next"] >= 0) & (df["days_to_next"] <= 30)
    ).astype(int)
    # Exclude patients who died (cannot be readmitted)
    df["readmission_30d"] = np.where(df["mortality"] == 1, np.nan, df["readmission_30d"])

    return df


def aggregate_vitals(stay_ids, intime_map):
    """Extract first-24h vital signs from icu/chartevents.csv.gz — chunked."""
    print("  Loading chartevents (may take a while)...")
    all_item_ids = [iid for ids in VITAL_ITEMIDS.values() for iid in ids]

    chunks = []
    for chunk in pd.read_csv(
        os.path.join(ICU_DIR, "chartevents.csv.gz"),
        usecols=["stay_id", "itemid", "charttime", "valuenum", "warning"],
        parse_dates=["charttime"],
        chunksize=1_000_000,
        low_memory=False,
    ):
        chunk = chunk[
            (chunk["stay_id"].isin(stay_ids)) &
            (chunk["itemid"].isin(all_item_ids)) &
            (chunk["warning"].isna() | (chunk["warning"] == 0))
        ].copy()
        if not chunk.empty:
            chunks.append(chunk)

    if not chunks:
        raise RuntimeError("No chartevents rows matched. Check stay_ids and itemids.")
    charts = pd.concat(chunks, ignore_index=True)

    charts["intime"]   = charts["stay_id"].map(intime_map)
    charts["hours_in"] = (charts["charttime"] - charts["intime"]).dt.total_seconds() / 3600
    charts = charts[(charts["hours_in"] >= 0) & (charts["hours_in"] <= 24)]

    item_to_feat = {iid: feat for feat, ids in VITAL_ITEMIDS.items() for iid in ids}
    charts["feature"] = charts["itemid"].map(item_to_feat)

    agg = (
        charts.groupby(["stay_id", "feature"])["valuenum"]
        .agg(STATS)
        .unstack("feature")
    )
    agg.columns = [f"{feat}_{stat}" for stat, feat in agg.columns]
    agg.reset_index(inplace=True)
    return agg


def aggregate_labs(hadm_ids, admittime_map):
    """Extract first-24h lab values from hosp/labevents.csv.gz — chunked."""
    print("  Loading labevents (may take a while)...")
    all_item_ids = [iid for ids in LAB_ITEMIDS.values() for iid in ids]

    # ── FIXED: read in chunks of 1M rows to avoid out-of-memory error ────────
    chunks = []
    for chunk in pd.read_csv(
        os.path.join(HOSP_DIR, "labevents.csv.gz"),
        usecols=["hadm_id", "itemid", "charttime", "valuenum"],
        parse_dates=["charttime"],
        chunksize=1_000_000,       # <-- KEY FIX: was loading entire file at once
        low_memory=False,
        on_bad_lines='skip',
    ):
        chunk = chunk[
            (chunk["hadm_id"].isin(hadm_ids)) &
            (chunk["itemid"].isin(all_item_ids))
        ].copy()
        if not chunk.empty:
            chunks.append(chunk)

    if not chunks:
        raise RuntimeError("No labevents rows matched. Check hadm_ids and itemids.")
    labs = pd.concat(chunks, ignore_index=True)

    labs["admittime"] = labs["hadm_id"].map(admittime_map)
    labs["hours_in"]  = (labs["charttime"] - labs["admittime"]).dt.total_seconds() / 3600
    labs = labs[(labs["hours_in"] >= 0) & (labs["hours_in"] <= 24)]

    item_to_feat = {iid: feat for feat, ids in LAB_ITEMIDS.items() for iid in ids}
    labs["feature"] = labs["itemid"].map(item_to_feat)

    agg = (
        labs.groupby(["hadm_id", "feature"])["valuenum"]
        .agg(STATS)
        .unstack("feature")
    )
    agg.columns = [f"{feat}_{stat}" for stat, feat in agg.columns]
    agg.reset_index(inplace=True)
    return agg


def build_feature_matrix(df, vital_agg, lab_agg):
    """Merge all features into a flat feature matrix."""
    feat = df[[
        "stay_id", "hadm_id", "subject_id", "admit_year",
        "mortality", "readmission_30d",
        "age", "los_hours",
    ]].copy()

    feat["gender_male"] = (df["gender"] == "M").astype(int)

    adm_dummies = pd.get_dummies(
        df["admission_type"], prefix="admtype", drop_first=True
    )
    feat = pd.concat([feat.reset_index(drop=True), adm_dummies.reset_index(drop=True)], axis=1)

    feat = feat.merge(vital_agg, on="stay_id", how="left")
    feat = feat.merge(lab_agg,   on="hadm_id", how="left")

    return feat


def impute_and_scale(X_train, X_val, X_shift):
    """Fit imputer and scaler on training data only."""
    imputer = SimpleImputer(strategy="median")
    scaler  = StandardScaler()

    X_train_p = scaler.fit_transform(imputer.fit_transform(X_train))
    X_val_p   = scaler.transform(imputer.transform(X_val))
    X_shift_p = scaler.transform(imputer.transform(X_shift))

    return X_train_p, X_val_p, X_shift_p, imputer, scaler


def make_splits(feat, label_col,
                train_years=(2008, 2016),
                shift_years=(2017, 2019),
                val_frac=0.15, cal_frac=0.15, seed=42):
    """
    Time-based train/val/cal/shift split.
    cal = calibration set for conformal prediction (carved from training data).
    """
    rng = np.random.default_rng(seed)

    train_mask = (feat["admit_year"] >= train_years[0]) & (feat["admit_year"] <= train_years[1])
    shift_mask = (feat["admit_year"] >= shift_years[0]) & (feat["admit_year"] <= shift_years[1])

    exclude_cols = [
        "stay_id", "hadm_id", "subject_id", "admit_year",
        "mortality", "readmission_30d",
    ]
    feature_cols = [c for c in feat.columns if c not in exclude_cols]

    train_df = feat[train_mask].dropna(subset=[label_col]).reset_index(drop=True)
    shift_df = feat[shift_mask].dropna(subset=[label_col]).reset_index(drop=True)

    n     = len(train_df)
    idx   = rng.permutation(n)
    n_val = int(n * val_frac)
    n_cal = int(n * cal_frac)

    val_idx   = idx[:n_val]
    cal_idx   = idx[n_val:n_val + n_cal]
    train_idx = idx[n_val + n_cal:]

    X_tr  = train_df.iloc[train_idx][feature_cols].values.astype(np.float32)
    X_val = train_df.iloc[val_idx  ][feature_cols].values.astype(np.float32)
    X_cal = train_df.iloc[cal_idx  ][feature_cols].values.astype(np.float32)
    X_sh  = shift_df[feature_cols].values.astype(np.float32)

    y_tr  = train_df.iloc[train_idx][label_col].values.astype(np.float32)
    y_val = train_df.iloc[val_idx  ][label_col].values.astype(np.float32)
    y_cal = train_df.iloc[cal_idx  ][label_col].values.astype(np.float32)
    y_sh  = shift_df[label_col].values.astype(np.float32)

    # Fit on training partition only; apply same transform to cal + shift
    X_tr, X_val, X_sh, imp, scl = impute_and_scale(X_tr, X_val, X_sh)
    X_cal = scl.transform(imp.transform(X_cal))

    return {
        "X_train": X_tr,   "y_train": y_tr,
        "X_val":   X_val,  "y_val":   y_val,
        "X_cal":   X_cal,  "y_cal":   y_cal,
        "X_shift": X_sh,   "y_shift": y_sh,
        "feature_cols": feature_cols,
        "imputer": imp,    "scaler":  scl,
    }


def run():
    print("Loading admissions...")
    df = load_admissions()

    stay_ids      = set(df["stay_id"].dropna().astype(int))
    hadm_ids      = set(df["hadm_id"].dropna().astype(int))
    intime_map    = df.dropna(subset=["stay_id"]).set_index("stay_id")["intime"].to_dict()
    admittime_map = df.dropna(subset=["hadm_id"]).set_index("hadm_id")["admittime"].to_dict()

    print("Aggregating vitals...")
    vital_agg = aggregate_vitals(stay_ids, intime_map)

    print("Aggregating labs...")
    lab_agg = aggregate_labs(hadm_ids, admittime_map)

    print("Building feature matrix...")
    feat = build_feature_matrix(df, vital_agg, lab_agg)

    for label in ["mortality", "readmission_30d"]:
        out_path = os.path.join(PROC_DIR, f"{label}_features.csv")
        feat.to_csv(out_path, index=False)
        print(f"  Saved {out_path}  (shape {feat.shape})")

    print("Done.")


if __name__ == "__main__":
    run()