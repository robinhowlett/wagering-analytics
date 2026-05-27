"""
AN2 Phase 2: Compute WCMI (Wisdom of Crowd Market Index) for every race.

WCMI adapts Shannon's Entropy to betting markets, producing a 0-1 score
that measures how much information the market has collectively absorbed:
  - WCMI → 0: maximum entropy, all runners same price, crowd knows nothing
  - WCMI → 1: minimum entropy, one horse at prohibitive odds, outcome "known"

Based on Matekus (2016) via Sports Trader Blog / FlatStats.

Formula:
    p_i = implied probability of runner i (from normalized odds)
    H = -sum(p_i * log_n(p_i))   where n = number of runners
    WCMI = 1 - H

Thresholds (Matekus):
    < 0.13: uninformed market — opportunity for informed bettor
    > 0.20: well-informed market — model edge is smaller

Output:
    Writes to handycapper.race_wcmi table (creates if not exists).

Usage:
    python scripts/compute_wcmi.py
"""

import logging
import os

import numpy as np
import psycopg2
import psycopg2.extras

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

BATCH = 50_000


def get_conn():
    return psycopg2.connect(
        host=os.environ.get("WA_DB_HOST", "localhost"),
        port=os.environ.get("WA_DB_PORT", "5432"),
        dbname=os.environ.get("WA_DB_NAME", "handycapper"),
        user=os.environ.get("WA_DB_USER", "handycapper"),
        password=os.environ.get("WA_DB_PASSWORD", "handycapper"),
    )


def compute_wcmi(implied_probs: np.ndarray) -> float:
    """
    Compute WCMI from an array of implied probabilities (pre-normalized).

    Returns value in [0, 1]. Returns 0 if all runners are equal price.
    """
    p = implied_probs / implied_probs.sum()
    n = len(p)
    if n < 2:
        return 0.0
    p = p[p > 0]
    h = -np.sum(p * np.log(p) / np.log(n))
    return float(1.0 - h)


def process_year(year: int, conn) -> int:
    """Process a single year. Returns number of races computed."""
    cur = conn.cursor()
    cur.execute("""
        SELECT r.id as race_id, array_agg(s.odds ORDER BY s.pp) as odds_arr
        FROM handycapper.races r
        JOIN handycapper.starters s ON s.race_id = r.id
        WHERE r.breed = 'TB'
          AND EXTRACT(YEAR FROM r.date) = %s
          AND r.number_of_runners >= 3
          AND s.odds IS NOT NULL AND s.odds > 0
          AND r.id NOT IN (SELECT race_id FROM handycapper.race_wcmi)
        GROUP BY r.id
        HAVING COUNT(*) >= 3
    """, (year,))

    rows = cur.fetchall()
    if not rows:
        cur.close()
        return 0

    batch = []
    for race_id, odds_arr in rows:
        odds = np.array([float(o) for o in odds_arr if o and float(o) > 0])
        if len(odds) < 2:
            continue
        implied = 1.0 / (odds + 1.0)
        wcmi = compute_wcmi(implied)
        normalized = implied / implied.sum()
        max_prob = float(normalized.max())
        batch.append((race_id, round(wcmi, 4), len(odds), round(max_prob, 4)))

    if batch:
        _write_batch(cur, batch)
        conn.commit()

    cur.close()
    return len(batch)


def main():
    conn = get_conn()
    cur = conn.cursor()

    log.info("Creating race_wcmi table if not exists...")
    cur.execute("""
        CREATE TABLE IF NOT EXISTS handycapper.race_wcmi (
            race_id BIGINT PRIMARY KEY,
            wcmi NUMERIC(5,4),
            n_runners SMALLINT,
            max_implied_prob NUMERIC(5,4),
            computed_at TIMESTAMP DEFAULT NOW()
        );
    """)
    conn.commit()
    cur.close()

    log.info("Processing year by year (1999-2017)...")
    total = 0
    for year in range(1999, 2018):
        n = process_year(year, conn)
        total += n
        log.info(f"  {year}: {n} races (cumulative: {total})")

    log.info(f"Done. Total: {total} races.")
    conn.close()


def _write_batch(cur, batch):
    psycopg2.extras.execute_values(
        cur,
        """
        INSERT INTO handycapper.race_wcmi (race_id, wcmi, n_runners, max_implied_prob)
        VALUES %s
        ON CONFLICT (race_id) DO UPDATE SET
            wcmi = EXCLUDED.wcmi,
            n_runners = EXCLUDED.n_runners,
            max_implied_prob = EXCLUDED.max_implied_prob,
            computed_at = NOW()
        """,
        batch,
        page_size=BATCH,
    )


if __name__ == "__main__":
    main()
