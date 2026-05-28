"""Verify Pick 5 / Pick 6 payoff model with carryover-aware modeling.

Carryover-corrected re-verification of WA-T1.3 for the longer horizontals.

Pari-mutuel carryover (pool_type = STANDARD, carryover > 0) is +EV — yesterday's
stranded pool joins today's. Race-day-sim should be able to model these days,
not exclude them. We INCLUDE carryover rows and add log_carryover as a feature.

JACKPOT pool_type is excluded — different product (single-unique-winner rule).

Models tested per bet type:
  1. naive parlay         — Σ log(odds_leg+1)  + intercept
  2. + log_carryover      — does carryover money change predicted payoff?
  3. pre-race full        — per-leg odds + log_pool + carryover + hhi/field
  4. original full        — pre-race full + bad_fav_legs (post-race)

Year-stratified split: train < 2016, test 2016-2017.
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

HORIZONTAL_LEGS = {"PICK_5": 5, "PICK_6": 6}

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
    COALESCE(e.carryover, 0) AS carryover_amt,
    EXTRACT(YEAR FROM r.date)::int AS yr,
    MAX(ld.winner_odds)    FILTER (WHERE ld.leg_number = 1) AS odds_leg1,
    MAX(ld.winner_odds)    FILTER (WHERE ld.leg_number = 2) AS odds_leg2,
    MAX(ld.winner_odds)    FILTER (WHERE ld.leg_number = 3) AS odds_leg3,
    MAX(ld.winner_odds)    FILTER (WHERE ld.leg_number = 4) AS odds_leg4,
    MAX(ld.winner_odds)    FILTER (WHERE ld.leg_number = 5) AS odds_leg5,
    MAX(ld.winner_odds)    FILTER (WHERE ld.leg_number = 6) AS odds_leg6,
    AVG(ld.leg_hhi)        AS avg_hhi,
    AVG(ld.leg_field_size) AS avg_field_size,
    SUM(ld.bad_fav_leg)    AS bad_fav_legs
FROM exotics e
JOIN exotic_race_legs erl1 ON erl1.exotic_id = e.id AND erl1.leg_number = 1
JOIN races r ON r.id = erl1.race_id
JOIN leg_data ld ON ld.exotic_id = e.id
WHERE e.bet_type   = %(bet_type)s
  AND e.pool_type  = 'STANDARD'   -- exclude JACKPOT (different product)
  AND e.payoff     > 0
  AND e.pool       > 0
GROUP BY e.id, e.payoff, e.pool, e.carryover, r.date
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
    df["pool_size"]     = pd.to_numeric(df["pool_size"], errors="coerce")
    df["carryover_amt"] = pd.to_numeric(df["carryover_amt"], errors="coerce").fillna(0)
    df["log_payoff"]    = np.log(df["actual_payoff"].clip(lower=0.01))
    df["log_pool"]      = np.log(df["pool_size"].clip(lower=1))
    # log1p so carryover=0 maps to 0 (no transform discontinuity)
    df["log_carryover"] = np.log1p(df["carryover_amt"].clip(lower=0))

    leg_cols = []
    for i in range(1, n_legs + 1):
        col = f"odds_leg{i}"
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[f"log_{col}"] = np.log(df[col].clip(lower=0.1) + 1)
        leg_cols.append(f"log_{col}")

    df["log_parlay"] = df[leg_cols].sum(axis=1)

    df = df.dropna(subset=["log_payoff", "log_pool", "log_parlay"] + leg_cols)
    n_carryover = (df["carryover_amt"] > 0).sum()
    log.info("  %d rows after cleaning  (%d with carryover > 0, %.1f%%)",
             len(df), n_carryover, 100.0 * n_carryover / len(df))
    return df


def fit_eval(X_train, y_train, X_test, y_test):
    m = LinearRegression().fit(X_train, y_train)
    return {
        "r2_train":  r2_score(y_train, m.predict(X_train)),
        "r2_test":   r2_score(y_test,  m.predict(X_test)),
        "rmse_test": float(np.sqrt(mean_squared_error(y_test, m.predict(X_test)))),
        "n_train":   len(y_train),
        "n_test":    len(y_test),
        "model":     m,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/payoff_skill_pick56_audit.txt")
    args = ap.parse_args()

    conn = psycopg2.connect(DSN)

    fh = open(args.out, "w")
    fh.write("WA-T1.3 Pick 5 / Pick 6 verification (carryover-aware)\n")
    fh.write("=" * 80 + "\n")
    fh.write("Year-stratified split: train < 2016, test 2016-2017\n")
    fh.write("INCLUDES carryover rows; excludes pool_type = JACKPOT\n\n")

    for bt in ["PICK_5", "PICK_6"]:
        log.info("=== %s ===", bt)
        df = load(conn, bt)

        train = df[df["yr"] < 2016]
        test  = df[df["yr"].isin([2016, 2017])]
        log.info("  train: %d  test: %d", len(train), len(test))

        if len(train) < 1000 or len(test) < 500:
            log.warning("  insufficient sample, skipping")
            continue

        n_legs = HORIZONTAL_LEGS[bt]
        leg_cols = [f"log_odds_leg{i}" for i in range(1, n_legs + 1)]

        # 1. Naive parlay
        Xn_train = train[["log_parlay"]].values
        Xn_test  = test[["log_parlay"]].values
        naive = fit_eval(Xn_train, train["log_payoff"].values,
                          Xn_test,  test["log_payoff"].values)

        # 2. + log_carryover
        Xc_train = train[["log_parlay", "log_carryover"]].values
        Xc_test  = test[["log_parlay", "log_carryover"]].values
        with_carry = fit_eval(Xc_train, train["log_payoff"].values,
                               Xc_test,  test["log_payoff"].values)
        carry_coef = with_carry["model"].coef_[1]

        # 3. pre-race full (per-leg odds + pool + carryover + hhi/field)
        pre_cols = leg_cols + ["log_pool", "log_carryover", "avg_hhi", "avg_field_size"]
        Xp_train = train[pre_cols].values
        Xp_test  = test[pre_cols].values
        pre_full = fit_eval(Xp_train, train["log_payoff"].values,
                             Xp_test,  test["log_payoff"].values)

        # 4. + bad_fav_legs (post-race)
        full_cols = pre_cols + ["bad_fav_legs"]
        Xf_train = train[full_cols].values
        Xf_test  = test[full_cols].values
        original_full = fit_eval(Xf_train, train["log_payoff"].values,
                                  Xf_test,  test["log_payoff"].values)
        bad_fav_coef = original_full["model"].coef_[full_cols.index("bad_fav_legs")]
        carry_in_full = original_full["model"].coef_[full_cols.index("log_carryover")]

        block = (
            f"\n--- {bt}  (n_train={len(train)}, n_test={len(test)}) ---\n"
            f"  1. naive parlay:                                R²_train={naive['r2_train']:.4f}  R²_test={naive['r2_test']:.4f}\n"
            f"  2. + log_carryover (2 predictors):              R²_train={with_carry['r2_train']:.4f}  R²_test={with_carry['r2_test']:.4f}\n"
            f"  3. pre-race full (legs+pool+carry+hhi+field):   R²_train={pre_full['r2_train']:.4f}  R²_test={pre_full['r2_test']:.4f}\n"
            f"  4. + bad_fav_legs (post-race):                  R²_train={original_full['r2_train']:.4f}  R²_test={original_full['r2_test']:.4f}\n"
            f"  ΔR² (2 - 1) [carryover contribution]:           {with_carry['r2_test']  - naive['r2_test']:+.4f}\n"
            f"  ΔR² (3 - 2) [pool/hhi/field contribution]:      {pre_full['r2_test']    - with_carry['r2_test']:+.4f}\n"
            f"  ΔR² (4 - 3) [bad_fav_legs contribution]:        {original_full['r2_test'] - pre_full['r2_test']:+.4f}\n"
            f"  log_carryover coef (with_carry model):           {carry_coef:+.4f}\n"
            f"  log_carryover coef (full model):                 {carry_in_full:+.4f}\n"
            f"  bad_fav_legs coef (full model):                  {bad_fav_coef:+.4f}\n"
        )
        log.info(block)
        fh.write(block)

    fh.close()
    log.info("Report written to %s", args.out)
    conn.close()


if __name__ == "__main__":
    main()
