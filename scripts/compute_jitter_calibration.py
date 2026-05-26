"""
AN1 Phase 5: Calibrate log-normal jitter parameters for horizontal leg
odds projection.

When simulating a Pick 3/4/5/6, only leg 1 has near-final closing odds.
Later legs are projected using a log-normal perturbation whose sigma
increases with leg position. This script derives those sigma values from
the empirical variance of closing odds across legs of the same sequence.

The logic: in an efficient market, if leg 1 odds are your best estimate of
true probability, the spread of leg 2+ winner odds around the parlay's
implied fair price captures how much additional uncertainty existed in those
legs at bet-construction time.

Output:
    rkm/models/jitter_calibration.json  — {leg_position: log_normal_sigma}

Usage:
    python scripts/compute_jitter_calibration.py
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).parent.parent / "models"
MODEL_DIR.mkdir(exist_ok=True)

SQL = """
WITH leg_data AS (
    SELECT
        erl.exotic_id,
        erl.leg_number,
        e.bet_type,
        -- Closing odds of the actual winner in this leg
        rp_winner.odds  AS winner_odds,
        -- Log-odds of the winner
        LN(rp_winner.odds + 1) AS log_winner_odds,
        -- Median log-odds of all runners in this leg (field centre of gravity)
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY LN(rp.odds + 1)) AS median_log_odds,
        COUNT(rp.starter_id) AS field_size,
        rm.hhi
    FROM exotic_race_legs erl
    JOIN exotics e      ON e.id = erl.exotic_id
    JOIN race_probabilities rp
        ON rp.race_id = erl.race_id
    JOIN race_probabilities rp_winner
        ON rp_winner.race_id        = erl.race_id
        AND rp_winner.wagering_position = 1
    JOIN race_metrics rm ON rm.race_id = erl.race_id
    WHERE e.bet_type IN ('DAILY_DOUBLE','PICK_3','PICK_4','PICK_5','PICK_6')
      AND e.pool_type = 'STANDARD'
      AND rp_winner.odds IS NOT NULL
      AND rp_winner.odds > 0
    GROUP BY erl.exotic_id, erl.leg_number, e.bet_type,
             rp_winner.odds, rm.hhi
),
-- Anchor each sequence: leg 1 log-winner-odds is the "known" reference
anchored AS (
    SELECT
        ld.*,
        FIRST_VALUE(ld.log_winner_odds) OVER (
            PARTITION BY ld.exotic_id ORDER BY ld.leg_number
        ) AS leg1_log_odds
    FROM leg_data ld
)
SELECT
    leg_number,
    bet_type,
    COUNT(*)                               AS n,
    AVG(log_winner_odds)                   AS mean_log_odds,
    STDDEV(log_winner_odds)                AS stddev_log_odds,
    -- Deviation from leg 1 odds (the "surprise" in each subsequent leg)
    STDDEV(log_winner_odds - leg1_log_odds) AS stddev_deviation_from_leg1,
    PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY log_winner_odds) AS p25_log_odds,
    PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY log_winner_odds) AS p75_log_odds
FROM anchored
GROUP BY leg_number, bet_type
ORDER BY bet_type, leg_number
"""


def main():
    import psycopg2

    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    # from rkm.db import connect_raw

    conn = psycopg2.connect("host=localhost port=5434 dbname=handycapper user=handycapper password=handycapper")
    log.info("Running jitter calibration query...")
    df = pd.read_sql(SQL, conn)
    conn.close()

    log.info("Results by bet_type and leg_number:")
    log.info("\n%s", df.to_string(index=False))

    # Build the sigma table: for each leg_position, take the weighted average
    # stddev across bet types (weighted by sample size n).
    # Leg 1 gets sigma=0 (closing odds are known).
    sigma_by_leg: dict[int, float] = {1: 0.0}

    for leg_pos in sorted(df["leg_number"].unique()):
        if leg_pos == 1:
            continue
        subset = df[df["leg_number"] == leg_pos]
        if len(subset) == 0:
            continue
        # Weighted average of stddev_deviation_from_leg1 across bet types
        total_n   = subset["n"].sum()
        weighted_sigma = (subset["stddev_deviation_from_leg1"] * subset["n"]).sum() / total_n
        sigma_by_leg[int(leg_pos)] = round(float(weighted_sigma), 4)
        log.info("  Leg %d: sigma=%.4f  (n=%d)", leg_pos, weighted_sigma, total_n)

    # Also store per-bet-type breakdown for reference
    detail: dict[str, dict] = {}
    for _, row in df.iterrows():
        bt  = row["bet_type"]
        leg = int(row["leg_number"])
        detail.setdefault(bt, {})[leg] = {
            "n":                   int(row["n"]),
            "mean_log_odds":       round(float(row["mean_log_odds"]), 4),
            "stddev_log_odds":     round(float(row["stddev_log_odds"]), 4),
            "stddev_from_leg1":    round(float(row["stddev_deviation_from_leg1"]), 4)
            if row["stddev_deviation_from_leg1"] is not None else None,
        }

    out = {
        "sigma_by_leg_position": sigma_by_leg,
        "note": (
            "sigma_by_leg_position[N] is the log-normal standard deviation to apply "
            "when projecting closing odds for leg N of a horizontal sequence. "
            "Leg 1 = 0.0 (closing odds are known). Higher legs = more uncertainty."
        ),
        "detail_by_bet_type": detail,
    }

    out_path = MODEL_DIR / "jitter_calibration.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    log.info("Jitter calibration saved → %s", out_path)


if __name__ == "__main__":
    main()
