"""Verify Pick 3 / Pick 4 payoff model skill above a naive parlay baseline.

WA-T1.3 horizontal verification.

For each bet type:
  - Naive parlay baseline:  log(payoff) ~ sum(log(odds_leg_i + 1))
                            (literally the parlay odds — what an accumulator would pay)
  - Pre-race full:           naive features + log_pool + avg_hhi + avg_field_size
  - Original full:           pre-race full + bad_fav_legs (post-race)

Year-stratified split (train < 2016, test 2016-2017).

Excludes carryover races (carryover > 0) — those have inflated payoffs that
the OLS model can't sensibly project from leg odds alone.
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

DSN = "host=127.0.0.1 port=5434 dbname=handycapper user=handycapper password=handycapper"

HORIZONTAL_LEGS = {"PICK_3": 3, "PICK_4": 4}

SQL = """
WITH leg_data AS (
    SELECT
        erl.exotic_id,
        erl.leg_number,
        s.odds         AS winner_odds,
        rm.hhi         AS leg_hhi,
        rm.field_size  AS leg_field_size,
        COALESCE((rm.finish_choice_ranks[1] != 1)::int, 0) AS bad_fav_leg
    FROM exotic_race_legs erl
    JOIN starters s
        ON s.race_id = erl.race_id AND s.official_position = 1
    JOIN race_metrics rm
        ON rm.race_id = erl.race_id
)
SELECT
    e.id,
    e.payoff / NULLIF(e.unit, 0) AS actual_payoff,
    e.pool        AS pool_size,
    EXTRACT(YEAR FROM r.date)::int AS yr,
    MAX(ld.winner_odds)    FILTER (WHERE ld.leg_number = 1) AS odds_leg1,
    MAX(ld.winner_odds)    FILTER (WHERE ld.leg_number = 2) AS odds_leg2,
    MAX(ld.winner_odds)    FILTER (WHERE ld.leg_number = 3) AS odds_leg3,
    MAX(ld.winner_odds)    FILTER (WHERE ld.leg_number = 4) AS odds_leg4,
    AVG(ld.leg_hhi)        AS avg_hhi,
    AVG(ld.leg_field_size) AS avg_field_size,
    SUM(ld.bad_fav_leg)    AS bad_fav_legs
FROM exotics e
JOIN exotic_race_legs erl1 ON erl1.exotic_id = e.id AND erl1.leg_number = 1
JOIN races r ON r.id = erl1.race_id
JOIN leg_data ld ON ld.exotic_id = e.id
WHERE e.bet_type   = %(bet_type)s
  AND e.pool_type  = 'STANDARD'
  AND e.payoff     > 0
  AND e.pool       > 0
  AND (e.carryover IS NULL OR e.carryover = 0)  -- exclude carryover (rare for P3/P4 anyway)
GROUP BY e.id, e.payoff, e.pool, r.date
HAVING COUNT(ld.leg_number) = %(n_legs)s
"""


def load(conn, bt: str) -> pd.DataFrame:
    n_legs = HORIZONTAL_LEGS[bt]
    log.info("Loading %s (%d legs)...", bt, n_legs)

    with conn.cursor() as cur:
        cur.execute("SET work_mem = '32MB'")
        cur.execute("SET max_parallel_workers_per_gather = 0")

    df = pd.read_sql(SQL, conn, params={"bet_type": bt, "n_legs": n_legs})
    log.info("  %d rows", len(df))

    df["actual_payoff"] = pd.to_numeric(df["actual_payoff"], errors="coerce")
    df["pool_size"] = pd.to_numeric(df["pool_size"], errors="coerce")
    df["log_payoff"] = np.log(df["actual_payoff"].clip(lower=0.01))
    df["log_pool"]   = np.log(df["pool_size"].clip(lower=1))

    leg_cols = []
    for i in range(1, n_legs + 1):
        col = f"odds_leg{i}"
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[f"log_{col}"] = np.log(df[col].clip(lower=0.1) + 1)
        leg_cols.append(f"log_{col}")

    # Naive parlay baseline: sum of log(odds+1) across legs = log of parlay odds product
    df["log_parlay"] = df[leg_cols].sum(axis=1)

    df = df.dropna(subset=["log_payoff", "log_pool", "log_parlay"] + leg_cols)
    log.info("  %d rows after cleaning", len(df))
    return df


def fit_eval(X_train, y_train, X_test, y_test):
    m = LinearRegression().fit(X_train, y_train)
    return {
        "r2_train": r2_score(y_train, m.predict(X_train)),
        "r2_test":  r2_score(y_test,  m.predict(X_test)),
        "rmse_test": float(np.sqrt(mean_squared_error(y_test, m.predict(X_test)))),
        "n_train": len(y_train),
        "n_test":  len(y_test),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/payoff_skill_horizontal_audit.txt")
    args = ap.parse_args()

    conn = psycopg2.connect(DSN)

    fh = open(args.out, "w")
    fh.write("WA-T1.3 horizontal verification — Pick 3 / Pick 4 OLS skill above naive parlay\n")
    fh.write("=" * 80 + "\n")
    fh.write("Year-stratified split: train < 2016, test 2016-2017\n")
    fh.write("Excludes carryover races. Joins via official_position = 1.\n\n")

    for bt in ["PICK_3", "PICK_4"]:
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

        n_legs = HORIZONTAL_LEGS[bt]
        leg_cols = [f"log_odds_leg{i}" for i in range(1, n_legs + 1)]

        # 1. Naive parlay (single predictor: sum of log(leg_odds+1))
        Xn_train = train[["log_parlay"]].values
        Xn_test  = test[["log_parlay"]].values
        naive = fit_eval(Xn_train, train["log_payoff"].values,
                          Xn_test,  test["log_payoff"].values)

        # 2. Per-leg odds (n_legs predictors instead of one — lets coefficients differ per leg)
        Xl_train = train[leg_cols].values
        Xl_test  = test[leg_cols].values
        per_leg = fit_eval(Xl_train, train["log_payoff"].values,
                            Xl_test,  test["log_payoff"].values)

        # 3. Pre-race full (per-leg odds + pool/hhi/field — NO bad_fav_legs)
        pre_cols = leg_cols + ["log_pool", "avg_hhi", "avg_field_size"]
        Xp_train = train[pre_cols].values
        Xp_test  = test[pre_cols].values
        pre_full = fit_eval(Xp_train, train["log_payoff"].values,
                             Xp_test,  test["log_payoff"].values)

        # 4. Original full (pre-race + bad_fav_legs post-race feature)
        full_cols = pre_cols + ["bad_fav_legs"]
        Xf_train = train[full_cols].values
        Xf_test  = test[full_cols].values
        original_full = fit_eval(Xf_train, train["log_payoff"].values,
                                  Xf_test,  test["log_payoff"].values)

        # bad_fav_legs coefficient in original full model
        from sklearn.linear_model import LinearRegression
        m_full = LinearRegression().fit(Xf_train, train["log_payoff"].values)
        bad_fav_coef = m_full.coef_[full_cols.index("bad_fav_legs")]

        block = (
            f"\n--- {bt}  (n_train={len(train)}, n_test={len(test)}) ---\n"
            f"  1. naive parlay (log_parlay = Σ log(leg_odds+1)):  R²_train={naive['r2_train']:.4f}  R²_test={naive['r2_test']:.4f}\n"
            f"  2. per-leg odds (separate β per leg):              R²_train={per_leg['r2_train']:.4f}  R²_test={per_leg['r2_test']:.4f}\n"
            f"  3. pre-race full (per-leg + pool/hhi/field):       R²_train={pre_full['r2_train']:.4f}  R²_test={pre_full['r2_test']:.4f}\n"
            f"  4. original full (+ bad_fav_legs post-race):       R²_train={original_full['r2_train']:.4f}  R²_test={original_full['r2_test']:.4f}\n"
            f"  ΔR² (3 - 1) [skill above naive parlay]:            {pre_full['r2_test']  - naive['r2_test']:+.4f}\n"
            f"  ΔR² (4 - 3) [bad_fav_legs post-race contribution]: {original_full['r2_test'] - pre_full['r2_test']:+.4f}\n"
            f"  bad_fav_legs coefficient (full model):              {bad_fav_coef:+.4f}\n"
        )
        log.info(block)
        fh.write(block)

    fh.close()
    log.info("Report written to %s", args.out)
    conn.close()


if __name__ == "__main__":
    main()
