"""
Populate exotic_harville_ratios.stern_fair using the empirically calibrated
Stern k=0.81. Reads the full race probability vector per race from
race_probabilities, then computes Stern-corrected Harville probability for
each exotic result's finishing combination.

Run after AN1 Phase 1 has populated exotic_harville_ratios.

Usage:
    python scripts/populate_stern_fair.py
"""

import csv
import io
import logging
import sys
from pathlib import Path

import numpy as np
import psycopg2
import psycopg2.extras

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
# from rkm.db import connect, connect_raw  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

STERN_K = 0.86  # calibrated 2026-05-27 via calibrate_stern_k.py (MLE on 81K clean races,
                # using official_position; LL surface is very flat so 0.86-0.88 are nearly equivalent)
BATCH   = 50_000


def stern_harville_prob(
    win_probs: np.ndarray,
    finisher_indices: list[int],
    k: float = STERN_K,
) -> float:
    """
    Stern-corrected Harville probability for an ordered finish sequence.

    win_probs:        normalized win probabilities for all runners (sums to 1)
    finisher_indices: 0-based indices into win_probs, in finishing order
    k:                Stern exponent (0.86 calibrated 2026-05-27 on TB racing 1991-2017)

    Returns the joint probability of the sequence occurring.
    """
    remaining = np.ones(len(win_probs), dtype=bool)
    prob = 1.0
    for idx in finisher_indices:
        p_k = np.where(remaining, np.power(win_probs, k), 0.0)
        denom = p_k.sum()
        if denom < 1e-12:
            return 0.0
        prob *= p_k[idx] / denom
        remaining[idx] = False
    return float(prob)


def fetch_race_probs(conn) -> dict[int, dict[int, float]]:
    """
    Load race_probabilities into memory: {race_id: {starter_id: win_prob}}.
    Also build {race_id: {starter_id: wagering_position}} for position lookups.
    Returns (probs_by_race, pos_by_race).
    """
    log.info("Loading race_probabilities...")
    cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    cur.execute("""
        SELECT race_id, starter_id, win_prob, wagering_position
        FROM   race_probabilities
        WHERE  win_prob IS NOT NULL
    """)
    probs_by_race: dict[int, dict[int, float]] = {}
    pos_by_race:   dict[int, dict[int, int]]   = {}
    for row in cur:
        rid, sid, wp, pos = row["race_id"], row["starter_id"], row["win_prob"], row["wagering_position"]
        probs_by_race.setdefault(rid, {})[sid] = float(wp)
        if pos is not None:
            pos_by_race.setdefault(rid, {})[int(pos)] = sid
    cur.close()
    log.info("Loaded %d races into memory.", len(probs_by_race))
    return probs_by_race, pos_by_race


def fetch_ehr_rows(conn, recompute_all: bool = False):
    """
    Yield rows from exotic_harville_ratios needing stern_fair.

    By default only refreshes NULL rows. Pass recompute_all=True to refresh
    every row (e.g., after changing STERN_K).
    """
    where_filter = "ehr.actual_payoff IS NOT NULL"
    if not recompute_all:
        where_filter = "ehr.stern_fair IS NULL AND " + where_filter

    cur = conn.cursor(name="ehr_cursor", cursor_factory=psycopg2.extras.DictCursor)
    cur.execute(f"""
        SELECT
            ehr.id,
            ehr.race_id,
            ehr.bet_type,
            ehr.actual_payoff,
            ehr.pool_size,
            ehr.harville_fair,
            -- We need the starter_ids for each finishing position
            rp1.starter_id AS sid_1,
            rp2.starter_id AS sid_2,
            rp3.starter_id AS sid_3,
            rp4.starter_id AS sid_4
        FROM exotic_harville_ratios ehr
        LEFT JOIN race_probabilities rp1
            ON rp1.race_id = ehr.race_id AND rp1.wagering_position = 1
        LEFT JOIN race_probabilities rp2
            ON rp2.race_id = ehr.race_id AND rp2.wagering_position = 2
        LEFT JOIN race_probabilities rp3
            ON rp3.race_id = ehr.race_id AND rp3.wagering_position = 3
        LEFT JOIN race_probabilities rp4
            ON rp4.race_id = ehr.race_id AND rp4.wagering_position = 4
        WHERE {where_filter}
    """)
    return cur


def compute_stern_fair(
    race_id:    int,
    bet_type:   str,
    sids:       list[int | None],
    probs_by_race: dict[int, dict[int, float]],
    takeout_map: dict[tuple, float],
    track: str,
) -> float | None:
    """
    Compute the Stern-corrected fair price for one exotic result.
    Returns None if race data is missing.
    """
    race_probs = probs_by_race.get(race_id)
    if not race_probs:
        return None

    # Build ordered probability vector for all runners
    all_probs = np.array(list(race_probs.values()))
    sid_list  = list(race_probs.keys())

    # Map each finisher's starter_id → index in sid_list
    finisher_indices = []
    for sid in sids:
        if sid is None:
            break
        if sid not in race_probs:
            return None
        finisher_indices.append(sid_list.index(sid))

    if not finisher_indices:
        return None

    stern_prob = stern_harville_prob(all_probs, finisher_indices)
    if stern_prob <= 0:
        return None

    takeout = takeout_map.get((track, bet_type), takeout_map.get((None, bet_type), 0.20))
    return round((1.0 - takeout) / stern_prob, 4)


def load_takeout_map(conn) -> dict[tuple, float]:
    """Load takeout_rates into a {(track, bet_type): rate} dict."""
    cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    cur.execute("""
        SELECT track, bet_type, takeout_rate
        FROM   takeout_rates
        WHERE  pool_type = 'STANDARD'
        ORDER  BY effective_date DESC
    """)
    tmap: dict[tuple, float] = {}
    for row in cur:
        key = (row["track"], row["bet_type"])
        if key not in tmap:          # keep most recent rate
            tmap[key] = float(row["takeout_rate"])
    cur.close()
    return tmap


def load_track_by_race(conn) -> dict[int, str]:
    """Load {race_id: track} for joining takeout rates."""
    cur = conn.cursor()
    cur.execute("SELECT id, track FROM races")
    result = {row[0]: row[1] for row in cur}
    cur.close()
    return result


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--recompute-all", action="store_true",
                    help="Refresh stern_fair for every row, not just NULLs (use after changing STERN_K).")
    args = ap.parse_args()

    conn = psycopg2.connect("host=localhost port=5434 dbname=handycapper user=handycapper password=handycapper")
    conn.autocommit = False

    takeout_map    = load_takeout_map(conn)
    track_by_race  = load_track_by_race(conn)
    probs_by_race, _ = fetch_race_probs(conn)

    log.info("Computing stern_fair values (k=%.3f, recompute_all=%s)...", STERN_K, args.recompute_all)
    cur    = fetch_ehr_rows(conn, recompute_all=args.recompute_all)
    write = psycopg2.connect("host=localhost port=5434 dbname=handycapper user=handycapper password=handycapper")
    write.autocommit = False

    batch_data: list[tuple] = []
    total = 0

    for row in cur:
        ehr_id  = row["id"]
        race_id = row["race_id"]
        bet_type = row["bet_type"]
        track    = track_by_race.get(race_id, "")

        # Determine how many positions this bet type uses
        n_pos = {
            "EXACTA": 2, "QUINELLA": 2,
            "TRIFECTA": 3,
            "SUPERFECTA": 4,
            "HI_5": 5,
        }.get(bet_type, 0)

        if n_pos == 0:
            continue  # horizontal bets handled separately

        sids = [row[f"sid_{i+1}"] for i in range(n_pos)]
        stern_fair = compute_stern_fair(
            race_id, bet_type, sids, probs_by_race, takeout_map, track
        )

        if stern_fair is not None:
            batch_data.append((stern_fair, ehr_id))

        if len(batch_data) >= BATCH:
            _flush(write, batch_data)
            total += len(batch_data)
            log.info("  Updated %d rows so far...", total)
            batch_data = []

    if batch_data:
        _flush(write, batch_data)
        total += len(batch_data)

    write.commit()
    log.info("Done. Populated stern_fair for %d rows.", total)
    cur.close()
    conn.close()
    write.close()


def _flush(conn, batch: list[tuple]):
    buf = io.StringIO()
    writer = csv.writer(buf)
    for stern_fair, ehr_id in batch:
        writer.writerow([stern_fair, ehr_id])
    buf.seek(0)

    cur = conn.cursor()
    cur.execute("CREATE TEMP TABLE IF NOT EXISTS _stern_update (stern_fair numeric, id bigint) ON COMMIT DELETE ROWS")
    cur.copy_expert("COPY _stern_update FROM STDIN CSV", buf)
    cur.execute("""
        UPDATE exotic_harville_ratios ehr
        SET    stern_fair = u.stern_fair
        FROM   _stern_update u
        WHERE  ehr.id = u.id
    """)
    conn.commit()
    cur.close()


if __name__ == "__main__":
    main()
