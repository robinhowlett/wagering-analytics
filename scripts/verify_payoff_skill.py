"""Verify how much skill the OLS payoff model adds above a naive Stern baseline.

WA-T1.2 / WA-T1.3 verification.

For each vertical bet type:
  - Naive baseline:  log(payoff) ~ log(stern_fair)         (one predictor + intercept)
  - Pre-race full:   current spec minus post-race fav_* features
  - Original full:   current spec including post-race features (for comparison)

All fits use a YEAR-STRATIFIED holdout: train on race_date < 2016, test on
race_date in 2016-2017. Reports R² for each model on train and test.
"""

import argparse
import logging

import numpy as np
import pandas as pd
import psycopg2
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

DSN = "host=localhost port=5434 dbname=handycapper user=handycapper password=handycapper"

VERTICAL_POSITIONS = {"EXACTA": 2, "TRIFECTA": 3}
# Note: SUPERFECTA omitted — race_probabilities.wagering_position only goes 1-3
# in this dataset, so we'd need to join by finish_position (matching how
# fit_payoff_models.py handles it). The 2-3 position bet types already answer
# the skill-vs-naive question for WA-T1.2.

SQL = """
SELECT
    ehr.id,
    ehr.bet_type,
    ehr.actual_payoff,
    ehr.stern_fair,
    ehr.harville_fair,
    ehr.pool_size,
    ehr.field_size,
    ehr.hhi,
    ehr.surface,
    ehr.fav_finish_pos,
    EXTRACT(YEAR FROM r.date)::int AS yr,
    rp1.odds AS odds_1,
    rp2.odds AS odds_2,
    rp3.odds AS odds_3,
    rp4.odds AS odds_4
FROM exotic_harville_ratios ehr
JOIN races r ON r.id = ehr.race_id
LEFT JOIN race_probabilities rp1 ON rp1.race_id = ehr.race_id AND rp1.wagering_position = 1
LEFT JOIN race_probabilities rp2 ON rp2.race_id = ehr.race_id AND rp2.wagering_position = 2
LEFT JOIN race_probabilities rp3 ON rp3.race_id = ehr.race_id AND rp3.wagering_position = 3
LEFT JOIN race_probabilities rp4 ON rp4.race_id = ehr.race_id AND rp4.wagering_position = 4
WHERE ehr.bet_type = %(bt)s
  AND ehr.actual_payoff > 0
  AND ehr.stern_fair > 0
  AND ehr.pool_size > 0
  AND rp1.odds IS NOT NULL
"""


def load(conn, bt: str) -> pd.DataFrame:
    log.info("Loading %s...", bt)
    df = pd.read_sql(SQL, conn, params={"bt": bt})
    log.info("  %d rows", len(df))

    df["log_payoff"] = np.log(df["actual_payoff"].astype(float))
    df["log_stern"]  = np.log(df["stern_fair"].astype(float))
    df["log_harville"] = np.log(df["harville_fair"].astype(float).clip(lower=0.01))
    df["log_pool"]   = np.log(df["pool_size"].astype(float))

    n_pos = VERTICAL_POSITIONS[bt]
    for i in range(1, n_pos + 1):
        col = f"odds_{i}"
        df[f"log_{col}"] = np.log(df[col].astype(float).clip(lower=0.1) + 1)

    df["fav_in_combo"] = df["fav_finish_pos"].between(1, n_pos).astype(float)
    df["fav_won"]      = (df["fav_finish_pos"] == 1).astype(float)
    df["fav_second"]   = (df["fav_finish_pos"] == 2).astype(float)
    df["fav_third"]    = (df["fav_finish_pos"] == 3).astype(float)
    df["fav_fourth"]   = (df["fav_finish_pos"] == 4).astype(float)

    df = pd.get_dummies(df, columns=["surface"], drop_first=False, dtype=float)
    for surf in ["surface_T", "surface_S"]:
        if surf not in df.columns:
            df[surf] = 0.0

    odds_cols = [f"log_odds_{i}" for i in range(1, n_pos + 1)]
    df = df.dropna(subset=["log_payoff", "log_stern", "log_pool", "yr"] + odds_cols)
    log.info("  %d rows after cleaning", len(df))
    return df


def fit_eval(X_train, y_train, X_test, y_test):
    m = LinearRegression().fit(X_train, y_train)
    return {
        "r2_train": r2_score(y_train, m.predict(X_train)),
        "r2_test":  r2_score(y_test,  m.predict(X_test)),
        "rmse_train": float(np.sqrt(mean_squared_error(y_train, m.predict(X_train)))),
        "rmse_test":  float(np.sqrt(mean_squared_error(y_test,  m.predict(X_test)))),
        "n_train": len(y_train),
        "n_test":  len(y_test),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/payoff_skill_audit.txt")
    args = ap.parse_args()

    conn = psycopg2.connect(DSN)

    fh = open(args.out, "w")
    fh.write("WA-T1.2 / WA-T1.3 verification — payoff model skill above naive Stern baseline\n")
    fh.write("=" * 80 + "\n")
    fh.write("Year-stratified split: train < 2016, test 2016-2017\n")
    fh.write(f"Underlying stern_fair recomputed with k=0.87 ({pd.Timestamp.now().date()})\n\n")

    for bt in ["EXACTA", "TRIFECTA", "SUPERFECTA"]:
        log.info("=== %s ===", bt)
        df = load(conn, bt)

        train = df[df["yr"] < 2016]
        test  = df[df["yr"].isin([2016, 2017])]
        log.info("  train: %d (years %d-%d)  test: %d (years %d-%d)",
                 len(train), int(train["yr"].min()), int(train["yr"].max()),
                 len(test), int(test["yr"].min()), int(test["yr"].max()))

        if len(test) < 1000:
            log.warning("  too few test rows, skipping")
            continue

        n_pos = VERTICAL_POSITIONS[bt]
        odds_cols = [f"log_odds_{i}" for i in range(1, n_pos + 1)]

        # 1. Naive Harville baseline (current Harville fair value, k=1)
        Xb_train_h = train[["log_harville"]].values
        Xb_test_h  = test[["log_harville"]].values
        naive_h = fit_eval(Xb_train_h, train["log_payoff"].values,
                            Xb_test_h,  test["log_payoff"].values)

        # 2. Naive Stern baseline (with newly calibrated k=0.87)
        Xb_train_s = train[["log_stern"]].values
        Xb_test_s  = test[["log_stern"]].values
        naive_s = fit_eval(Xb_train_s, train["log_payoff"].values,
                            Xb_test_s,  test["log_payoff"].values)

        # 3. Pre-race full model: stern_fair + odds + pool/field/hhi/surface (NO fav_* post-race features)
        pre_cols = ["log_stern", "log_pool", "field_size", "hhi", "surface_T", "surface_S"] + odds_cols
        X_pre_train = train[pre_cols].values
        X_pre_test  = test[pre_cols].values
        pre_full = fit_eval(X_pre_train, train["log_payoff"].values,
                             X_pre_test,  test["log_payoff"].values)

        # 4. Original full model (includes post-race fav features) — replicates fit_payoff_models.py spec
        full_cols = pre_cols + ["fav_in_combo", "fav_won", "fav_second", "fav_third", "fav_fourth"]
        X_full_train = train[full_cols].values
        X_full_test  = test[full_cols].values
        original_full = fit_eval(X_full_train, train["log_payoff"].values,
                                  X_full_test,  test["log_payoff"].values)

        block = (
            f"\n--- {bt}  (n_train={len(train)}, n_test={len(test)}) ---\n"
            f"  1. naive Harville  (1 predictor):           R²_train={naive_h['r2_train']:.4f}  R²_test={naive_h['r2_test']:.4f}\n"
            f"  2. naive Stern k=0.87 (1 predictor):        R²_train={naive_s['r2_train']:.4f}  R²_test={naive_s['r2_test']:.4f}\n"
            f"  3. pre-race full  (no fav_* features):      R²_train={pre_full['r2_train']:.4f}  R²_test={pre_full['r2_test']:.4f}\n"
            f"  4. original full  (includes fav_* leakage): R²_train={original_full['r2_train']:.4f}  R²_test={original_full['r2_test']:.4f}\n"
            f"  ΔR² (3 - 2) [skill above Stern, pre-race]:    {pre_full['r2_test']  - naive_s['r2_test']:+.4f}\n"
            f"  ΔR² (4 - 3) [post-race feature leakage gain]: {original_full['r2_test'] - pre_full['r2_test']:+.4f}\n"
            f"  ΔR² (2 - 1) [Stern-vs-Harville baseline]:     {naive_s['r2_test']  - naive_h['r2_test']:+.4f}\n"
        )
        log.info(block)
        fh.write(block)

    fh.close()
    log.info("Report written to %s", args.out)
    conn.close()


if __name__ == "__main__":
    main()
