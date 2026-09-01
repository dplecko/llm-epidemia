"""Generalized version of build.py (this folder): same cleaning / recoding /
imputation, for an arbitrary BRFSS survey year, downloaded directly (no local
XPT path needed) exactly like build.py does for 2023. Output goes to
data/clean/ (not datasets/brfss/data/, unlike build.py) so it sits next to
the other cleaned datasets eval.py reads.

Usage: python datasets/brfss/build_temporal.py --year 2015
"""

import argparse
import os
import tempfile
import zipfile

import numpy as np
import pandas as pd
import requests

BRFSS_URL = "https://www.cdc.gov/brfss/annual_data/{year}/files/LLCP{year}XPT.zip"

# ---------------------------------------------------------------------------
# State FIPS -> name map. Copied verbatim from datasets/brfss/build.py.
# Stable across all BRFSS years used here.
# ---------------------------------------------------------------------------
STATE_MAP = {
    1: "Alabama", 2: "Alaska", 4: "Arizona", 5: "Arkansas", 6: "California", 8: "Colorado",
    9: "Connecticut", 10: "Delaware", 11: "District of Columbia", 12: "Florida", 13: "Georgia",
    15: "Hawaii", 16: "Idaho", 17: "Illinois", 18: "Indiana", 19: "Iowa", 20: "Kansas",
    22: "Louisiana", 23: "Maine", 24: "Maryland", 25: "Massachusetts", 26: "Michigan",
    27: "Minnesota", 28: "Mississippi", 29: "Missouri", 30: "Montana", 31: "Nebraska",
    32: "Nevada", 33: "New Hampshire", 34: "New Jersey", 35: "New Mexico", 36: "New York",
    37: "North Carolina", 38: "North Dakota", 39: "Ohio", 40: "Oklahoma", 41: "Oregon",
    44: "Rhode Island", 45: "South Carolina", 46: "South Dakota", 47: "Tennessee",
    48: "Texas", 49: "Utah", 50: "Vermont", 51: "Virginia", 53: "Washington",
    54: "West Virginia", 55: "Wisconsin", 56: "Wyoming", 66: "Guam", 72: "Puerto Rico",
    78: "Virgin Islands"
}

# ---------------------------------------------------------------------------
# Per-field column resolution: (candidate source columns in priority order).
# Verified against CDC codebooks / "Calculated Variables" docs for 2015,
# 2018, 2020, 2023 (see NOTES.md).
# ---------------------------------------------------------------------------
CANDIDATES = {
    "_STATE_COL":  ["_STATE"],
    "SEX_COL":     ["_SEX", "SEXVAR", "SEX1", "SEX"],
    "IMPRACE_COL": ["_IMPRACE"],              # 6-cat, present 2018/2020/2023, not 2015
    "RACE8_COL":   ["_RACE"],                 # 8-cat fallback, present all years
    "AGE_COL":     ["_AGE80"],
    "EDUCAG_COL":  ["_EDUCAG"],
    "INCOMG1_COL": ["_INCOMG1"],               # 7-cat, 2021+
    "INCOMG_COL":  ["_INCOMG"],                # 5-cat, pre-2021
    "SMOKER_COL":  ["_SMOKER3"],
    "BMI_COL":     ["_BMI5"],
    "EXERCISE_COL":["EXERANY2"],
    "MENT14_COL":  ["_MENT14D"],
    "DIABETES_COL":["DIABETE4", "DIABETE3"],
    "HIGHBP_COL":  ["BPHIGH6", "BPHIGH5", "BPHIGH4"],
    "ASTHMA_COL":  ["ASTHMA3"],
    "CHOL_COL":    ["TOLDHI3", "TOLDHI2"],
    "HEART_COL":   ["CVDINFR4"],
    "STROKE_COL":  ["CVDSTRK3"],
    "DEPR_COL":    ["ADDEPEV3", "ADDEPEV2"],
    "DEAF_COL":    ["DEAF"],
    "BLIND_COL":   ["BLIND"],
    "WEIGHT_COL":  ["_LLCPWT"],
}


def resolve(cols, candidates):
    for c in candidates:
        if c in cols:
            return c
    return None


def recode_chunk(df, resolved, year, verbose=False):
    """Apply all the semantic recodes to one raw chunk, returning only the
    slim output columns. Keeping this per-chunk (instead of concatenating
    all ~300 raw columns first) keeps peak memory low enough to fit the
    sandbox's cgroup memory limit on a full-size (~440-490k row) BRFSS file.
    `verbose` gates the one-time NOTE/WARNING prints (pass True for the
    first chunk only).
    """
    if verbose:
        missing = [k for k, v in resolved.items() if v is None
                   and k not in ("IMPRACE_COL", "RACE8_COL", "INCOMG1_COL", "INCOMG_COL")]
        print(f"[{year}] resolved columns: { {k: v for k, v in resolved.items() if v} }")
        if missing:
            print(f"[{year}] WARNING: no source column found for: {missing} -> output will be all-NaN")

    out = pd.DataFrame(index=df.index)

    out["state"] = df[resolved["_STATE_COL"]].map(STATE_MAP).astype("category")

    sex_col = resolved["SEX_COL"]
    out["sex"] = df[sex_col].map({1.0: "Male", 2.0: "Female"}).astype("category") if sex_col else np.nan

    # --- race: prefer 6-cat _IMPRACE (2018/2020/2023); else collapse 8-cat _RACE (2015) ---
    if resolved["IMPRACE_COL"]:
        out["race"] = df[resolved["IMPRACE_COL"]].map({
            1.0: "White", 2.0: "Black", 3.0: "Asian", 4.0: "AIAN", 5.0: "Hispanic", 6.0: "Other"
        }).astype("category")
    elif resolved["RACE8_COL"]:
        # _RACE: 1 White,2 Black,3 AIAN,4 Asian,5 NHPI,6 Other,7 Multiracial,8 Hispanic,9 DK/Refused
        # Collapsed to the same 6 output categories as _IMPRACE (NHPI & Multiracial -> Other).
        out["race"] = df[resolved["RACE8_COL"]].map({
            1.0: "White", 2.0: "Black", 3.0: "AIAN", 4.0: "Asian",
            5.0: "Other", 6.0: "Other", 7.0: "Other", 8.0: "Hispanic"
        }).astype("category")
        if verbose:
            print(f"[{year}] NOTE: race derived from 8-category _RACE (no _IMPRACE this year); "
                  f"NHPI & Multiracial collapsed into 'Other'.")
    else:
        out["race"] = np.nan

    out["age"] = df[resolved["AGE_COL"]] if resolved["AGE_COL"] else np.nan
    age_bins = [18, 25, 35, 45, 55, 65, 75, 80, 81]
    age_labels = ["18-24 years", "25-34 years", "35-44 years", "45-54 years",
                  "55-64 years", "65-74 years", "75-79 years", "80+ years"]
    out["age_group"] = pd.Categorical(
        pd.cut(out["age"], bins=age_bins, labels=age_labels, right=False), ordered=True
    )

    educag_col = resolved["EDUCAG_COL"]
    out["education"] = df[educag_col].map({
        1.0: "no high school", 2.0: "high school", 3.0: "some college", 4.0: "college graduate", 9.0: np.nan
    }).astype("category") if educag_col else np.nan

    # --- income: 7-cat _INCOMG1 (2021+) else 5-cat _INCOMG (coarser; "$50k+" not split further) ---
    if resolved["INCOMG1_COL"]:
        out["income"] = df[resolved["INCOMG1_COL"]].map({
            1.0: "<$15k", 2.0: "$15–25k", 3.0: "$25–35k", 4.0: "$35–50k",
            5.0: "$50–100k", 6.0: "$100k-200k", 7.0: ">$200k", 9.0: np.nan
        }).astype("category")
    elif resolved["INCOMG_COL"]:
        out["income"] = df[resolved["INCOMG_COL"]].map({
            1.0: "<$15k", 2.0: "$15–25k", 3.0: "$25–35k", 4.0: "$35–50k", 5.0: ">$50k", 9.0: np.nan
        }).astype("category")
        if verbose:
            print(f"[{year}] NOTE: income uses coarser 5-category _INCOMG scale "
                  f"(single '>$50k' bucket instead of 2023's split $50-100k/$100-200k/>$200k).")
    else:
        out["income"] = np.nan

    smoker_col = resolved["SMOKER_COL"]
    out["smoker"] = df[smoker_col].map({1.0: "Yes", 2.0: "Yes", 3.0: "No", 4.0: "No", 9.0: np.nan}).astype("category") if smoker_col else np.nan

    bmi_col = resolved["BMI_COL"]
    out["bmi"] = df[bmi_col] / 100.0 if bmi_col else np.nan

    ex_col = resolved["EXERCISE_COL"]
    out["exercise_monthly"] = df[ex_col].map({1.0: "Yes", 2.0: "No", 7.0: "No", 9.0: "No"}).astype("category") if ex_col else np.nan

    ment_col = resolved["MENT14_COL"]
    if ment_col:
        out["poor_mental_health"] = df[ment_col].map({1.0: "No", 2.0: "Yes", 3.0: "Yes", 9.0: np.nan}).astype("category")
    else:
        out["poor_mental_health"] = np.nan
        if verbose:
            print(f"[{year}] NOTE: _MENT14D not present this year -> poor_mental_health is all-NaN.")

    dia_col = resolved["DIABETES_COL"]
    out["diabetes"] = df[dia_col].map({1.0: "Yes", 2.0: "Yes", 3.0: "No", 4.0: "No", 7.0: "No", 9.0: "No"}).astype("category") if dia_col else np.nan

    bp_col = resolved["HIGHBP_COL"]
    if bp_col:
        out["high_bp"] = df[bp_col].map({1.0: "Yes", 2.0: "Yes", 3.0: "No", 4.0: "No", 7.0: np.nan, 9.0: np.nan}).astype("category")
    else:
        out["high_bp"] = np.nan
        if verbose:
            print(f"[{year}] NOTE: High Blood Pressure module (BPHIGH*) not asked this year -> high_bp is all-NaN.")

    asthma_col = resolved["ASTHMA_COL"]
    out["asthma"] = df[asthma_col].map({1.0: "Yes", 2.0: "No", 4.0: "No", 7.0: "No", 9.0: "No"}).astype("category") if asthma_col else np.nan

    chol_col = resolved["CHOL_COL"]
    if chol_col:
        out["cholesterol"] = df[chol_col].map({1.0: "Yes", 2.0: "No", 7.0: "No", 9.0: "No"}).astype("category")
    else:
        out["cholesterol"] = np.nan
        if verbose:
            print(f"[{year}] NOTE: Cholesterol Awareness module (TOLDHI*) not asked this year -> cholesterol is all-NaN.")

    heart_col = resolved["HEART_COL"]
    out["heart_attack"] = df[heart_col].map({1.0: "Yes", 2.0: "No", 7.0: "No", 9.0: "No"}).astype("category") if heart_col else np.nan

    stroke_col = resolved["STROKE_COL"]
    out["stroke"] = df[stroke_col].map({1.0: "Yes", 2.0: "No", 7.0: "No", 9.0: "No"}).astype("category") if stroke_col else np.nan

    depr_col = resolved["DEPR_COL"]
    out["depression"] = df[depr_col].map({1.0: "Yes", 2.0: "No", 7.0: "No", 9.0: "No"}).astype("category") if depr_col else np.nan

    deaf_col = resolved["DEAF_COL"]
    if deaf_col:
        out["deaf"] = df[deaf_col].map({1.0: "Yes", 2.0: "No", 7.0: "No", 9.0: "No"}).astype("category")
    else:
        out["deaf"] = np.nan
        if verbose:
            print(f"[{year}] NOTE: DEAF not asked this year -> deaf is all-NaN.")

    blind_col = resolved["BLIND_COL"]
    out["blind"] = df[blind_col].map({1.0: "Yes", 2.0: "No", 7.0: "No", 9.0: "No"}).astype("category") if blind_col else np.nan

    weight_col = resolved["WEIGHT_COL"]
    out["weight"] = df[weight_col] if weight_col else np.nan

    out = out.reset_index(drop=True)

    col_order = ["state", "sex", "race", "age", "age_group", "education", "income",
                 "smoker", "bmi", "exercise_monthly", "poor_mental_health", "diabetes",
                 "high_bp", "asthma", "cholesterol", "heart_attack", "stroke",
                 "depression", "deaf", "blind", "weight"]
    return out[col_order]


def download_xpt(year, tmp_dir):
    """Same download as datasets/brfss/build.py: stream the CDC zip to disk,
    unzip, and locate the .XPT entry (its filename has a trailing space)."""
    url = BRFSS_URL.format(year=year)
    zip_path = os.path.join(tmp_dir, f"LLCP{year}XPT.zip")
    print(f"[{year}] downloading {url} ...")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(zip_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        entry = zip_ref.namelist()[0]  # e.g. "LLCP2015.XPT " (trailing space)
        zip_ref.extractall(tmp_dir)
    return os.path.join(tmp_dir, entry)


def extract_slim(xpt_path, year, chunksize=50_000):
    """Phase A: stream the raw XPT in chunks, recoding each chunk down to
    the ~20 slim output columns immediately (instead of ever holding the
    full ~300-column raw file in memory), and concatenate the slim result.
    Returns the un-imputed slim DataFrame.
    """
    print(f"[{year}] streaming {xpt_path} in chunks of {chunksize} ...")
    resolved = None
    slim_chunks = []
    n_rows = 0
    for i, chunk in enumerate(pd.read_sas(xpt_path, format="xport", encoding="latin1", chunksize=chunksize)):
        if resolved is None:
            resolved = {k: resolve(set(chunk.columns), v) for k, v in CANDIDATES.items()}
        slim_chunks.append(recode_chunk(chunk, resolved, year, verbose=(i == 0)))
        n_rows += len(chunk)
        if i % 4 == 0:
            print(f"[{year}]   ... {n_rows} rows processed")
    out = pd.concat(slim_chunks, ignore_index=True)
    del slim_chunks

    # Re-cast to category after concat: per-chunk .astype("category") calls
    # can end up with different category sets per chunk, which makes
    # pd.concat silently fall back to plain object/str dtype.
    numeric_cols = {"age", "bmi", "weight"}
    for c in out.columns:
        if c not in numeric_cols:
            ordered = (c == "age_group")
            out[c] = out[c].astype("category")
            if ordered:
                out[c] = out[c].cat.as_ordered()

    print(f"[{year}] slim shape: {out.shape}")
    return out


def impute_and_save(out, year, out_path, do_impute=True, mice_iters=5, random_state=0):
    """Phase B: MICE-impute the slim (already recoded) DataFrame and save."""
    col_order = list(out.columns)
    all_nan_cols = [c for c in col_order if out[c].isna().all()]
    impute_cols = [c for c in col_order if c not in all_nan_cols]

    if do_impute:
        import miceforest as mf
        print(f"[{year}] running MICE imputation on {len(impute_cols)} columns "
              f"(skipping fully-missing: {all_nan_cols}) ...")
        sub = out[impute_cols].copy()
        # mean_match_candidates=0: use the fitted model's direct prediction
        # instead of KD-tree nearest-neighbor mean-matching. The latter
        # (miceforest's default) computes an odds-ratio transform that
        # divides by zero -- and crashes with "data must be finite" -- for
        # our low-prevalence binary columns (stroke, heart_attack, deaf,
        # blind all have ~3-6% "Yes"), independent of survey year.
        kds = mf.ImputationKernel(sub, mean_match_candidates=0,
                                   save_all_iterations_data=True, random_state=random_state)
        kds.mice(mice_iters)
        sub_imputed = kds.complete_data()
        for c in impute_cols:
            out[c] = sub_imputed[c]
        print(f"[{year}] imputation done.")
    else:
        print(f"[{year}] skipping imputation (--no-impute).")

    out.to_parquet(out_path, index=False)
    print(f"[{year}] wrote {out_path}  shape={out.shape}")
    return out


def build(year, out_path, do_impute=True, mice_iters=5, random_state=0):
    with tempfile.TemporaryDirectory() as tmp_dir:
        xpt_path = download_xpt(year, tmp_dir)
        out = extract_slim(xpt_path, year)
    return impute_and_save(out, year, out_path, do_impute=do_impute, mice_iters=mice_iters, random_state=random_state)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--out", type=str, default=None, help="Output parquet path (default: data/clean/brfss_<year>.parquet)")
    ap.add_argument("--no-impute", action="store_true", help="Skip MICE imputation (faster, for debugging)")
    args = ap.parse_args()
    out_path = args.out or f"data/clean/brfss_{args.year}.parquet"
    build(args.year, out_path, do_impute=not args.no_impute)
