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
      AND r.date BETWEEN %s AND %s
      AND s.horse IS NOT NULL
),
post_claim AS (
    SELECT c.horse, c.claim_date,
        s.trainer_last, s.trainer_first,
        s.official_position, s.odds,
        ROW_NUMBER() OVER (PARTITION BY c.horse, c.claim_date ORDER BY r.date) as rn
    FROM claims c
    JOIN handycapper.starters s ON s.horse = c.horse
    JOIN handycapper.races r ON r.id = s.race_id
    WHERE r.date > c.claim_date
      AND r.date <= c.claim_date + interval '180 days'
      AND r.breed = 'TB' AND r.number_of_runners >= 5
      AND s.odds IS NOT NULL AND s.odds > 0
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


def main():
    conn = get_conn()
    cur = conn.cursor()

    log.info("Creating trainer_ae_profiles table...")
    cur.execute(CREATE_TABLE)
    conn.commit()

    log.info("Computing overall trainer stats...")
    cur.execute(SQL_OVERALL, (DATE_RANGE[0], DATE_RANGE[1], MIN_STARTS))
    overall = {
        f"{row[0]}|{row[1]}": {"last": row[0], "first": row[1], "n": row[2], "wins": row[3], "exp": row[4]}
        for row in cur.fetchall()
    }
    log.info(f"  {len(overall)} trainers with {MIN_STARTS}+ starts")

    dimensions = {}

    log.info("Computing FTS dimension...")
    cur.execute(SQL_FTS, (DATE_RANGE[0], DATE_RANGE[1]))
    for row in cur.fetchall():
        key = f"{row[0]}|{row[1]}"
        dimensions.setdefault(key, {})["fts"] = (row[2], row[3], row[4])

    log.info("Computing claim dimension...")
    cur.execute(SQL_CLAIM, (DATE_RANGE[0], DATE_RANGE[1]))
    for row in cur.fetchall():
        key = f"{row[0]}|{row[1]}"
        dimensions.setdefault(key, {})["claim"] = (row[2], row[3], row[4])

    log.info("Computing class drop dimension...")
    cur.execute(SQL_DROP, (DATE_RANGE[0], DATE_RANGE[1]))
    for row in cur.fetchall():
        key = f"{row[0]}|{row[1]}"
        dimensions.setdefault(key, {})["drop"] = (row[2], row[3], row[4])

    log.info("Computing layoff dimension...")
    cur.execute(SQL_LAYOFF, (DATE_RANGE[0], DATE_RANGE[1]))
    for row in cur.fetchall():
        key = f"{row[0]}|{row[1]}"
        dimensions.setdefault(key, {})["layoff"] = (row[2], row[3], row[4])

    log.info("Computing surface switch dimension...")
    cur.execute(SQL_SWITCH, (DATE_RANGE[0], DATE_RANGE[1]))
    for row in cur.fetchall():
        key = f"{row[0]}|{row[1]}"
        dimensions.setdefault(key, {})["switch"] = (row[2], row[3], row[4])

    log.info("Building profiles and writing to DB...")
    batch = []
    for key, ov in overall.items():
        dims = dimensions.get(key, {})
        fts = dims.get("fts", (0, 0, 0))
        claim = dims.get("claim", (0, 0, 0))
        drop = dims.get("drop", (0, 0, 0))
        layoff = dims.get("layoff", (0, 0, 0))
        switch = dims.get("switch", (0, 0, 0))

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
            0, None,  # jock_upgrade placeholder — requires more complex query
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
    log.info(f"Written {len(batch)} trainer profiles.")
    cur.close()
    conn.close()


if __name__ == "__main__":
    main()
