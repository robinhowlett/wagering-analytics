"""
AN1 Phase 6: Fit log-linear payoff prediction models for all exotic bet types.

For each bet type, fits:
    log(actual_payoff) ~ log(finisher_odds...) + log(pool_size) + field_size
                       + hhi + fav_position_flags + surface + interactions

Verticals  (Exacta, Trifecta, Superfecta, Hi5, Quinella):  OLS or Ridge
Horizontals (Daily Double, Pick 3-6):                       OLS or Ridge

Outputs:
    rkm/models/payoff_{BET_TYPE}.pkl   — fitted model (joblib)
    rkm/models/payoff_coefficients.json — human-readable coefficients + fit stats

Usage:
    python scripts/fit_payoff_models.py [--bet-type TRIFECTA] [--holdout 0.20]
"""

import argparse
import json
import logging
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
import psycopg2

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).parent.parent / "models"
MODEL_DIR.mkdir(exist_ok=True)

# Bet types that use OLS (large samples, well-behaved).
# Superfecta, Hi5 use Ridge due to sparser long-tail combinations.
OLS_TYPES   = {"EXACTA", "TRIFECTA", "DAILY_DOUBLE", "PICK_3", "PICK_4"}
RIDGE_TYPES = {"SUPERFECTA", "HI_5", "QUINELLA", "PICK_5", "PICK_6"}

VERTICAL_POSITIONS = {
    "EXACTA":    2,
    "QUINELLA":  2,
    "TRIFECTA":  3,
    "SUPERFECTA": 4,
    "HI_5":      5,
}

HORIZONTAL_LEGS = {
    "DAILY_DOUBLE": 2,
    "PICK_3": 3,
    "PICK_4": 4,
    "PICK_5": 5,
    "PICK_6": 6,
}

# Right-tail winsorization quantile applied to actual_payoff before log-transform.
# Even after log, a handful of extreme jackpot-style payoffs dominate OLS.
WINSOR_PCT = 0.995


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

VERTICAL_SQL = """
SELECT
    ehr.id,
    ehr.actual_payoff,
    ehr.pool_size,
    ehr.field_size,
    ehr.hhi,
    ehr.surface,
    ehr.fav_finish_pos,
    ehr.finish_choice_1,
    ehr.finish_choice_2,
    ehr.finish_choice_3,
    ehr.finish_choice_4,
    -- Odds of each ordered finisher.
    -- Positions 1-3 join via race_probabilities.wagering_position
    -- (a WPS-payout-attribution column that correctly handles coupled
    -- entries for top-3 finishers). Positions 4-5 are NOT in
    -- wagering_position (it is structurally 1-3 only — a WPS pool
    -- pays only top-3), so we resolve those via starters.official_position
    -- joined to race_probabilities by starter_id. SUPERFECTA and HI_5
    -- depend on this; EXACTA/TRIFECTA only need rp1-rp3.
    rp1.odds AS odds_1,
    rp2.odds AS odds_2,
    rp3.odds AS odds_3,
    rp4.odds AS odds_4,
    rp5.odds AS odds_5
FROM exotic_harville_ratios ehr
LEFT JOIN race_probabilities rp1
    ON rp1.race_id = ehr.race_id AND rp1.wagering_position = 1
LEFT JOIN race_probabilities rp2
    ON rp2.race_id = ehr.race_id AND rp2.wagering_position = 2
LEFT JOIN race_probabilities rp3
    ON rp3.race_id = ehr.race_id AND rp3.wagering_position = 3
LEFT JOIN starters s4
    ON s4.race_id = ehr.race_id AND s4.official_position = 4
LEFT JOIN race_probabilities rp4
    ON rp4.starter_id = s4.id
LEFT JOIN starters s5
    ON s5.race_id = ehr.race_id AND s5.official_position = 5
LEFT JOIN race_probabilities rp5
    ON rp5.starter_id = s5.id
WHERE ehr.bet_type = %(bet_type)s
  AND ehr.actual_payoff > 0
  AND ehr.pool_size    > 0
  AND rp1.odds IS NOT NULL
"""

HORIZONTAL_SQL = """
WITH leg_winners AS (
    SELECT
        erl.exotic_id,
        erl.leg_number,
        s.odds         AS winner_odds,
        rm.hhi         AS leg_hhi,
        rm.field_size  AS leg_field_size,
        -- Was there a bad fav in this leg? (choice 1 missed board)
        COALESCE((rm.finish_choice_ranks[1] != 1)::int, 0) AS bad_fav_leg
    FROM exotic_race_legs erl
    JOIN starters s
        ON s.race_id = erl.race_id AND s.finish_position = 1
    JOIN race_metrics rm ON rm.race_id = erl.race_id
)
SELECT
    e.id,
    e.payoff / NULLIF(e.unit, 0) AS actual_payoff,
    e.pool        AS pool_size,
    -- Pivot leg data (up to 6 legs)
    MAX(lw.winner_odds)    FILTER (WHERE lw.leg_number = 1) AS odds_leg1,
    MAX(lw.winner_odds)    FILTER (WHERE lw.leg_number = 2) AS odds_leg2,
    MAX(lw.winner_odds)    FILTER (WHERE lw.leg_number = 3) AS odds_leg3,
    MAX(lw.winner_odds)    FILTER (WHERE lw.leg_number = 4) AS odds_leg4,
    MAX(lw.winner_odds)    FILTER (WHERE lw.leg_number = 5) AS odds_leg5,
    MAX(lw.winner_odds)    FILTER (WHERE lw.leg_number = 6) AS odds_leg6,
    AVG(lw.leg_hhi)        AS avg_hhi,
    AVG(lw.leg_field_size) AS avg_field_size,
    SUM(lw.bad_fav_leg)    AS bad_fav_legs
FROM exotics e
JOIN leg_winners lw ON lw.exotic_id = e.id
WHERE e.bet_type   = %(bet_type)s
  AND e.pool_type  = 'STANDARD'
  AND e.payoff     > 0
  AND e.pool       > 0
GROUP BY e.id, e.payoff, e.pool
HAVING COUNT(lw.leg_number) = %(n_legs)s
"""


def load_vertical(conn, bet_type: str) -> pd.DataFrame:
    log.info("Loading %s data...", bet_type)
    df = pd.read_sql(VERTICAL_SQL, conn, params={"bet_type": bet_type})
    n_pos = VERTICAL_POSITIONS[bet_type]

    # Winsorize the right tail before logging. Even after log-transform a
    # handful of $10K+ trifectas can pull OLS coefficients meaningfully.
    # Cap at the 99.5th percentile (one-sided — small payoffs aren't outliers
    # in the same way; the lower-bound clip handles structural zeros below).
    cap = df["actual_payoff"].quantile(WINSOR_PCT)
    n_capped = int((df["actual_payoff"] > cap).sum())
    if n_capped > 0:
        log.info("  winsorizing %d/%d (%.2f%%) payoffs > $%.2f (P%g)",
                 n_capped, len(df), 100.0 * n_capped / len(df), cap, WINSOR_PCT * 100)
        df["actual_payoff"] = df["actual_payoff"].clip(upper=cap)
    df["log_payoff"] = np.log(df["actual_payoff"])
    df["log_pool"]   = np.log(df["pool_size"])

    for i in range(1, n_pos + 1):
        col = f"odds_{i}"
        df[f"log_{col}"] = np.log(df[col].clip(lower=0.1) + 1)

    # Favorite position flags
    df["fav_in_combo"] = (df["fav_finish_pos"].between(1, n_pos)).astype(int)
    df["fav_won"]      = (df["fav_finish_pos"] == 1).astype(int)
    df["fav_second"]   = (df["fav_finish_pos"] == 2).astype(int)
    df["fav_third"]    = (df["fav_finish_pos"] == 3).astype(int) if n_pos >= 3 else 0
    df["fav_fourth"]   = (df["fav_finish_pos"] == 4).astype(int) if n_pos >= 4 else 0

    # Surface dummies (drop 'D' as reference)
    df = pd.get_dummies(df, columns=["surface"], drop_first=False, dtype=float)
    for surf in ["surface_T", "surface_S"]:
        if surf not in df.columns:
            df[surf] = 0.0

    # Drop rows missing any of the position-odds we need for this bet type.
    # SUPERFECTA needs odds_1..odds_4; HI_5 needs odds_1..odds_5. The
    # rp4/rp5 joins via starters.official_position can yield NULL when
    # finish data is incomplete (≪1% of rows).
    required = ["log_payoff", "log_pool"] + [f"log_odds_{i}" for i in range(1, n_pos + 1)]
    before = len(df)
    df = df.dropna(subset=required)
    log.info("  %d rows after cleaning (dropped %d for missing position odds).",
             len(df), before - len(df))
    return df


def load_horizontal(conn, bet_type: str) -> pd.DataFrame:
    n_legs = HORIZONTAL_LEGS[bet_type]
    log.info("Loading %s data (%d legs)...", bet_type, n_legs)

    # Reduce memory pressure on the remote DB
    with conn.cursor() as cur:
        cur.execute("SET work_mem = '16MB'")
        cur.execute("SET max_parallel_workers_per_gather = 0")

    # Pick 6: exclude carryover days
    extra = ""
    if bet_type == "PICK_6":
        extra = " AND (e.carryover IS NULL OR e.carryover = 0)"

    sql = HORIZONTAL_SQL.replace("WHERE e.bet_type", extra + "\nWHERE e.bet_type")
    df  = pd.read_sql(sql, conn, params={"bet_type": bet_type, "n_legs": n_legs})

    # Ensure numeric types
    df["actual_payoff"] = pd.to_numeric(df["actual_payoff"], errors="coerce")
    df["pool_size"] = pd.to_numeric(df["pool_size"], errors="coerce")
    # Winsorize the right tail before logging — see load_vertical for rationale.
    cap = df["actual_payoff"].dropna().quantile(WINSOR_PCT)
    n_capped = int((df["actual_payoff"] > cap).sum())
    if n_capped > 0:
        log.info("  winsorizing %d/%d (%.2f%%) payoffs > $%.2f (P%g)",
                 n_capped, len(df), 100.0 * n_capped / len(df), cap, WINSOR_PCT * 100)
        df["actual_payoff"] = df["actual_payoff"].clip(upper=cap)
    df["log_payoff"] = np.log(df["actual_payoff"].clip(lower=0.01))
    df["log_pool"]   = np.log(df["pool_size"].clip(lower=1))

    for i in range(1, n_legs + 1):
        col = f"odds_leg{i}"
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[f"log_{col}"] = np.log(df[col].clip(lower=0.1) + 1)

    drop_cols = ["log_payoff", "log_pool"] + [f"log_odds_leg{i}" for i in range(1, n_legs + 1)]
    df = df.dropna(subset=drop_cols)
    log.info("  %d rows after cleaning.", len(df))
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Feature matrix construction
# ─────────────────────────────────────────────────────────────────────────────

def build_vertical_features(df: pd.DataFrame, bet_type: str) -> tuple[np.ndarray, np.ndarray]:
    n_pos = VERTICAL_POSITIONS[bet_type]
    cols  = [f"log_odds_{i}" for i in range(1, n_pos + 1)]
    cols += ["log_pool", "field_size", "hhi",
             "fav_in_combo", "fav_won", "fav_second", "fav_third", "fav_fourth",
             "surface_T", "surface_S"]

    # Interactions: log_odds_1 × fav_second / fav_third
    # (captures "price on top, fav underneath" overlay)
    df = df.copy()
    df["log_odds1_x_fav_second"] = df["log_odds_1"] * df["fav_second"]
    df["log_odds1_x_fav_third"]  = df["log_odds_1"] * df["fav_third"]
    cols += ["log_odds1_x_fav_second", "log_odds1_x_fav_third"]

    X = sm.add_constant(df[cols].values.astype(float))
    y = df["log_payoff"].values
    return X, y


def build_horizontal_features(df: pd.DataFrame, bet_type: str) -> tuple[np.ndarray, np.ndarray]:
    n_legs = HORIZONTAL_LEGS[bet_type]
    cols   = [f"log_odds_leg{i}" for i in range(1, n_legs + 1)]
    cols  += ["log_pool", "avg_hhi", "avg_field_size", "bad_fav_legs"]

    X = sm.add_constant(df[cols].values.astype(float))
    y = df["log_payoff"].values
    return X, y


# ─────────────────────────────────────────────────────────────────────────────
# Fitting
# ─────────────────────────────────────────────────────────────────────────────

def fit_ols(X_train, y_train):
    model = sm.OLS(y_train, X_train).fit()
    return model


def fit_ridge(X_train, y_train):
    alphas = np.logspace(-3, 4, 50)
    cv     = RidgeCV(alphas=alphas, cv=5, scoring="neg_mean_squared_error")
    cv.fit(X_train[:, 1:], y_train)   # skip the intercept column for sklearn
    log.info("  Ridge α selected: %.4f", cv.alpha_)
    model = Ridge(alpha=cv.alpha_).fit(X_train[:, 1:], y_train)
    return model, cv.alpha_


def evaluate(model, X_test, y_test, is_ols=True) -> dict:
    if is_ols:
        y_pred = model.predict(X_test)
    else:
        y_pred = model.predict(X_test[:, 1:])

    rmse    = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2      = float(r2_score(y_test, y_pred))
    # Back-transform: RMSE in log space → median factor error in raw space
    med_fac = float(np.exp(np.median(np.abs(y_pred - y_test))))

    return {"rmse_log": round(rmse, 4), "r2": round(r2, 4),
            "median_factor_error": round(med_fac, 3)}


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run_bet_type(conn, bet_type: str, holdout: float):
    is_horiz = bet_type in HORIZONTAL_LEGS
    use_ols  = bet_type in OLS_TYPES

    if is_horiz:
        df = load_horizontal(conn, bet_type)
    else:
        df = load_vertical(conn, bet_type)

    if len(df) == 0:
        log.warning(f"No data for {bet_type}, skipping.")
        return None

    if is_horiz:
        X, y = build_horizontal_features(df, bet_type)
        feat_names = ["const"] + [f"log_odds_leg{i}" for i in range(1, HORIZONTAL_LEGS[bet_type]+1)] \
                   + ["log_pool", "avg_hhi", "avg_field_size", "bad_fav_legs"]
    else:
        X, y = build_vertical_features(df, bet_type)
        n_pos = VERTICAL_POSITIONS[bet_type]
        feat_names = ["const"] + [f"log_odds_{i}" for i in range(1, n_pos+1)] \
                   + ["log_pool", "field_size", "hhi",
                      "fav_in_combo", "fav_won", "fav_second", "fav_third", "fav_fourth",
                      "surface_T", "surface_S",
                      "log_odds1_x_fav_second", "log_odds1_x_fav_third"]

    if len(df) < 1000:
        log.warning("  Only %d rows for %s — skipping (insufficient data).", len(df), bet_type)
        return

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=holdout, random_state=42
    )
    log.info("  Train: %d  Test: %d", len(y_train), len(y_test))

    if use_ols:
        model   = fit_ols(X_train, y_train)
        metrics = evaluate(model, X_test, y_test, is_ols=True)
        coefs   = dict(zip(feat_names, model.params.tolist()))
        pvals   = dict(zip(feat_names, model.pvalues.tolist()))
        alpha   = None
        r2_train = float(model.rsquared)
        log.info("  R²=%.4f  RMSE(log)=%.4f  MedianFactorErr=%.3f",
                 metrics["r2"], metrics["rmse_log"], metrics["median_factor_error"])
    else:
        model, alpha = fit_ridge(X_train, y_train)
        metrics      = evaluate(model, X_test, y_test, is_ols=False)
        coefs        = dict(zip(feat_names[1:], model.coef_.tolist()))
        coefs["const"] = float(model.intercept_)
        pvals        = {}
        r2_train     = float(r2_score(y_train, model.predict(X_train[:, 1:])))
        log.info("  R²(train)=%.4f  R²(test)=%.4f  MedianFactorErr=%.3f",
                 r2_train, metrics["r2"], metrics["median_factor_error"])

    # Save model
    out_path = MODEL_DIR / f"payoff_{bet_type}.pkl"
    joblib.dump({"model": model, "bet_type": bet_type, "is_ols": use_ols,
                 "feat_names": feat_names, "alpha": alpha}, out_path)
    log.info("  Saved → %s", out_path)

    return {
        "bet_type":        bet_type,
        "n_train":         int(len(y_train)),
        "n_test":          int(len(y_test)),
        "r2_train":        r2_train,
        "r2_test":         metrics["r2"],
        "rmse_log":        metrics["rmse_log"],
        "median_factor_error": metrics["median_factor_error"],
        "ridge_alpha":     alpha,
        "coefficients":    coefs,
        "p_values":        pvals,
    }


def main():
    parser = argparse.ArgumentParser(description="Fit AN1 Phase 6 payoff prediction models.")
    parser.add_argument("--bet-type", default=None,
                        help="Fit only this bet type (default: all)")
    parser.add_argument("--holdout", type=float, default=0.20,
                        help="Test holdout fraction (default 0.20)")
    args = parser.parse_args()

    # from rkm.db import connect_raw
    conn = psycopg2.connect("host=localhost port=5434 dbname=handycapper user=handycapper password=handycapper")

    all_types = list(VERTICAL_POSITIONS.keys()) + list(HORIZONTAL_LEGS.keys())
    targets   = [args.bet_type] if args.bet_type else all_types

    all_results = {}
    for bt in targets:
        log.info("=== %s ===", bt)
        result = run_bet_type(conn, bt, args.holdout)
        if result:
            all_results[bt] = result

    # Write consolidated coefficients file
    out_json = MODEL_DIR / "payoff_coefficients.json"
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2)
    log.info("Coefficients written → %s", out_json)

    conn.close()


if __name__ == "__main__":
    main()
