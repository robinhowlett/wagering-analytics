"""
AN2 Phase 1/3: Compute trainer A/E profiles across 6 dimensions.

For each trainer with sufficient data, compute their historical A/E in
each decision context:
  1. FTS (first-time starter) maiden record
  2. Claim (first 3 starts after claiming a horse)
  3. Class drop (dropping 30%+ in purse)
  4. Layoff (returning off 90+ days)
  5. Surface switch (changing surface from prior start)
  6. Jockey upgrade (switching to a 5%+ better jockey)

All computations are aggregate (full career in date range), not
point-in-time. Point-in-time requires window functions that are better
done in a materialized view or at query time.

Output:
    Writes to handycapper.trainer_ae_profiles table (creates if not exists).

Usage:
    python scripts/compute_trainer_profiles.py
"""

import logging
import os

import psycopg2
import psycopg2.extras

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

DATE_RANGE = ("2005-01-01", "2017-12-31")
MIN_STARTS = 30


def get_conn():
    return psycopg2.connect(
        host=os.environ.get("WA_DB_HOST", "localhost"),
        port=os.environ.get("WA_DB_PORT", "5432"),
        dbname=os.environ.get("WA_DB_NAME", "handycapper"),
        user=os.environ.get("WA_DB_USER", "handycapper"),
        password=os.environ.get("WA_DB_PASSWORD", "handycapper"),
    )


CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS handycapper.trainer_ae_profiles (
    trainer_key VARCHAR(200) PRIMARY KEY,
    trainer_last VARCHAR(100),
    trainer_first VARCHAR(100),
    total_starts INTEGER,
    overall_win_pct NUMERIC(5,4),
    overall_ae NUMERIC(5,3),
    fts_starts INTEGER,
    fts_ae NUMERIC(5,3),
    claim_starts INTEGER,
    claim_ae NUMERIC(5,3),
    drop_starts INTEGER,
    drop_ae NUMERIC(5,3),
    layoff_starts INTEGER,
    layoff_ae NUMERIC(5,3),
    switch_starts INTEGER,
    switch_ae NUMERIC(5,3),
    jock_upgrade_starts INTEGER,
    jock_upgrade_ae NUMERIC(5,3),
    computed_at TIMESTAMP DEFAULT NOW()
);
"""

SQL_OVERALL = """
SELECT s.trainer_last, s.trainer_first,
    COUNT(*) as n,
    SUM(CASE WHEN s.official_position = 1 THEN 1 ELSE 0 END) as wins,
    SUM(1.0 / (s.odds + 1)) as expected
FROM handycapper.starters s
JOIN handycapper.races r ON r.id = s.race_id
WHERE r.breed = 'TB'
  AND r.date BETWEEN %s AND %s
  AND r.number_of_runners >= 5
  AND s.odds IS NOT NULL AND s.odds > 0
GROUP BY s.trainer_last, s.trainer_first
HAVING COUNT(*) >= %s
"""

SQL_FTS = """
SELECT s.trainer_last, s.trainer_first,
    COUNT(*) as n,
    SUM(CASE WHEN s.official_position = 1 THEN 1 ELSE 0 END) as wins,
    SUM(1.0 / (s.odds + 1)) as expected
FROM handycapper.starters s
JOIN handycapper.races r ON r.id = s.race_id
WHERE s.last_raced_date IS NULL
  AND r.type LIKE '%%MAIDEN%%'
  AND r.breed = 'TB'
  AND r.date BETWEEN %s AND %s
  AND r.number_of_runners >= 5
  AND s.odds IS NOT NULL AND s.odds > 0
GROUP BY s.trainer_last, s.trainer_first
HAVING COUNT(*) >= 10
"""

SQL_CLAIM = """
WITH claims AS (
    SELECT s.horse, r.date as claim_date
    FROM handycapper.starters s
    JOIN handycapper.races r ON r.id = s.race_id
    WHERE s.claimed = true AND r.breed = 'TB'
      AND r.surface = 'Dirt' AND r.track_condition = 'Fast'
      AND r.date BETWEEN %s AND %s
      AND s.horse IS NOT NULL
),
-- Each post-claim start belongs to the MOST RECENT prior claim within 180d.
-- Without this dedupe, a horse claimed twice in close succession had its
-- subsequent starts double-counted (one row per claim_date * race within
-- both 180d windows). Pick the latest matching claim_date per (horse, start).
post_claim_per_start AS (
    SELECT s.id AS starter_id, c.horse, c.claim_date,
        s.trainer_last, s.trainer_first,
        s.official_position, s.odds, r.date AS race_date,
        ROW_NUMBER() OVER (
            PARTITION BY s.id
            ORDER BY c.claim_date DESC
        ) AS claim_rank
    FROM claims c
    JOIN handycapper.starters s ON s.horse = c.horse
    JOIN handycapper.races r ON r.id = s.race_id
    WHERE r.date > c.claim_date
      AND r.date <= c.claim_date + interval '180 days'
      AND r.breed = 'TB' AND r.surface = 'Dirt' AND r.track_condition = 'Fast'
      AND r.number_of_runners >= 5
      AND s.odds IS NOT NULL AND s.odds > 0
),
post_claim AS (
    SELECT horse, claim_date, trainer_last, trainer_first,
        official_position, odds,
        ROW_NUMBER() OVER (PARTITION BY horse, claim_date ORDER BY race_date) as rn
    FROM post_claim_per_start
    WHERE claim_rank = 1
)
SELECT trainer_last, trainer_first,
    COUNT(*) as n,
    SUM(CASE WHEN official_position = 1 THEN 1 ELSE 0 END) as wins,
    SUM(1.0 / (odds + 1)) as expected
FROM post_claim
WHERE rn <= 3
GROUP BY trainer_last, trainer_first
HAVING COUNT(*) >= 10
"""

SQL_DROP = """
WITH class_moves AS (
    SELECT s.trainer_last, s.trainer_first,
        s.official_position, s.odds, r.purse,
        LAG(r.purse) OVER (PARTITION BY s.horse ORDER BY r.date) as prev_purse
    FROM handycapper.starters s
    JOIN handycapper.races r ON r.id = s.race_id
    WHERE r.breed = 'TB' AND r.surface = 'Dirt' AND r.track_condition = 'Fast'
      AND r.date BETWEEN %s AND %s
      AND r.number_of_runners >= 5
      AND s.horse IS NOT NULL
      AND s.odds IS NOT NULL AND s.odds > 0
)
SELECT trainer_last, trainer_first,
    COUNT(*) as n,
    SUM(CASE WHEN official_position = 1 THEN 1 ELSE 0 END) as wins,
    SUM(1.0 / (odds + 1)) as expected
FROM class_moves
WHERE prev_purse IS NOT NULL AND purse < prev_purse * 0.7
GROUP BY trainer_last, trainer_first
HAVING COUNT(*) >= 10
"""

SQL_LAYOFF = """
SELECT s.trainer_last, s.trainer_first,
    COUNT(*) as n,
    SUM(CASE WHEN s.official_position = 1 THEN 1 ELSE 0 END) as wins,
    SUM(1.0 / (s.odds + 1)) as expected
FROM handycapper.starters s
JOIN handycapper.races r ON r.id = s.race_id
WHERE r.breed = 'TB' AND r.surface = 'Dirt' AND r.track_condition = 'Fast'
  AND r.date BETWEEN %s AND %s
  AND r.number_of_runners >= 5
  AND s.last_raced_date IS NOT NULL
  AND (r.date - s.last_raced_date) >= 90
  AND s.odds IS NOT NULL AND s.odds > 0
GROUP BY s.trainer_last, s.trainer_first
HAVING COUNT(*) >= 10
"""

# SURFACE_SWITCH is intentionally NOT filtered to a single surface (unlike
# DROP/LAYOFF/CLAIM which constrain to Dirt/Fast). A switch event spans two
# surfaces by definition, so a per-surface filter would discard the signal.
# The track-condition filter is also dropped for the same reason: a switch
# to/from turf doesn't have a "Fast" surface analogue. This means the SWITCH
# A/E baseline differs from the other dimensions; consumers comparing them
# should treat SWITCH as measured against a broader population.
SQL_SWITCH = """
WITH sequential AS (
    SELECT s.trainer_last, s.trainer_first,
        s.official_position, s.odds, r.surface,
        LAG(r.surface) OVER (PARTITION BY s.horse ORDER BY r.date) as prev_surface
    FROM handycapper.starters s
    JOIN handycapper.races r ON r.id = s.race_id
    WHERE r.breed = 'TB'
      AND r.date BETWEEN %s AND %s
      AND r.number_of_runners >= 5
      AND s.horse IS NOT NULL
      AND s.odds IS NOT NULL AND s.odds > 0
      AND r.surface IN ('Dirt', 'Turf', 'Synthetic')
)
SELECT trainer_last, trainer_first,
    COUNT(*) as n,
    SUM(CASE WHEN official_position = 1 THEN 1 ELSE 0 END) as wins,
    SUM(1.0 / (odds + 1)) as expected
FROM sequential
WHERE prev_surface IS NOT NULL AND surface != prev_surface
GROUP BY trainer_last, trainer_first
HAVING COUNT(*) >= 10
"""


def safe_ae(wins, expected):
    if expected and expected > 0:
        return round(float(wins) / float(expected), 3)
    return None


def run_dimension(conn, name: str, sql: str, params: tuple) -> dict:
    """Run a single dimension query with logging and timing."""
    import time
    cur = conn.cursor()
    log.info(f"  Computing {name}...")
    t0 = time.time()
    cur.execute(sql, params)
    rows = cur.fetchall()
    elapsed = time.time() - t0
    result = {}
    for row in rows:
        key = f"{row[0]}|{row[1]}"
        result[key] = (row[2], row[3], row[4])
    log.info(f"  {name}: {len(result)} trainers in {elapsed:.1f}s")
    cur.close()
    return result


def main():
    conn = get_conn()
    cur = conn.cursor()

    log.info("Creating trainer_ae_profiles table...")
    cur.execute(CREATE_TABLE)
    conn.commit()
    cur.close()

    # Phase 1: Overall stats (lightweight)
    log.info("Phase 1/6: Overall trainer stats...")
    overall_data = run_dimension(conn, "overall", SQL_OVERALL,
                                 (DATE_RANGE[0], DATE_RANGE[1], MIN_STARTS))
    # Convert to richer dict
    cur = conn.cursor()
    cur.execute(SQL_OVERALL, (DATE_RANGE[0], DATE_RANGE[1], MIN_STARTS))
    overall = {
        f"{row[0]}|{row[1]}": {"last": row[0], "first": row[1], "n": row[2], "wins": row[3], "exp": row[4]}
        for row in cur.fetchall()
    }
    cur.close()
    log.info(f"  {len(overall)} trainers with {MIN_STARTS}+ starts")

    # Phase 2-6: Each dimension independently (resumable — if one fails, others still computed)
    dimensions = {}

    for name, sql, params in [
        ("FTS", SQL_FTS, (DATE_RANGE[0], DATE_RANGE[1])),
        ("claim", SQL_CLAIM, (DATE_RANGE[0], DATE_RANGE[1])),
        ("class_drop", SQL_DROP, (DATE_RANGE[0], DATE_RANGE[1])),
        ("layoff", SQL_LAYOFF, (DATE_RANGE[0], DATE_RANGE[1])),
        ("surface_switch", SQL_SWITCH, (DATE_RANGE[0], DATE_RANGE[1])),
    ]:
        try:
            result = run_dimension(conn, name, sql, params)
            for key, vals in result.items():
                dimensions.setdefault(key, {})[name] = vals
        except Exception as e:
            log.error(f"  {name} FAILED: {e}")
            log.info("  Reconnecting and continuing with remaining dimensions...")
            try:
                conn.close()
            except Exception:
                pass
            conn = get_conn()

    # Phase 7: Write results
    log.info("Writing profiles to DB...")
    cur = conn.cursor()
    batch = []
    for key, ov in overall.items():
        dims = dimensions.get(key, {})
        fts = dims.get("FTS", (0, 0, 0))
        claim = dims.get("claim", (0, 0, 0))
        drop = dims.get("class_drop", (0, 0, 0))
        layoff = dims.get("layoff", (0, 0, 0))
        switch = dims.get("surface_switch", (0, 0, 0))

        batch.append((
            key,
            ov["last"], ov["first"],
            ov["n"],
            round(float(ov["wins"]) / ov["n"], 4) if ov["n"] else None,
            safe_ae(ov["wins"], ov["exp"]),
            fts[0], safe_ae(fts[1], fts[2]),
            claim[0], safe_ae(claim[1], claim[2]),
            drop[0], safe_ae(drop[1], drop[2]),
            layoff[0], safe_ae(layoff[1], layoff[2]),
            switch[0], safe_ae(switch[1], switch[2]),
            0, None,  # jock_upgrade — requires separate complex query
        ))

    cur.execute("DELETE FROM handycapper.trainer_ae_profiles")
    psycopg2.extras.execute_values(
        cur,
        """
        INSERT INTO handycapper.trainer_ae_profiles (
            trainer_key, trainer_last, trainer_first,
            total_starts, overall_win_pct, overall_ae,
            fts_starts, fts_ae,
            claim_starts, claim_ae,
            drop_starts, drop_ae,
            layoff_starts, layoff_ae,
            switch_starts, switch_ae,
            jock_upgrade_starts, jock_upgrade_ae
        ) VALUES %s
        """,
        batch,
        page_size=5000,
    )
    conn.commit()
    log.info(f"Done. Written {len(batch)} trainer profiles.")
    cur.close()
    conn.close()


if __name__ == "__main__":
    main()
