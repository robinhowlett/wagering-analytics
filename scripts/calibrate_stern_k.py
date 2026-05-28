"""Calibrate Stern-Harville k against actual top-3 ordering frequencies.

Reads race_probabilities + starters.finish_position; for each clean race
(no coupled entries, no DH/DQ in top 3, complete top 3 with prob data),
computes log-likelihood of the observed top-3 sequence under Stern-Harville
parameterized by k. Grid-searches k to find the MLE.

Phase 1: global k.
Phase 2: segmented by field size (5-7, 8-10, 11+).

This is a verification step for audit finding WA-T1.1.
"""

import logging
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import psycopg2
import psycopg2.extras

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

DSN = "host=localhost port=5434 dbname=handycapper user=handycapper password=handycapper"
K_GRID = np.arange(0.50, 1.21, 0.02)


def stern_top3_logprob(win_probs: np.ndarray, top3_indices: list[int], k: float) -> float:
    """Log-prob of observed top-3 sequence under Stern-Harville with exponent k."""
    p_k = np.power(win_probs, k)
    remaining = np.ones(len(win_probs), dtype=bool)
    log_prob = 0.0
    for idx in top3_indices:
        denom = p_k[remaining].sum()
        if denom <= 0:
            return float("-inf")
        log_prob += np.log(p_k[idx]) - np.log(denom)
        remaining[idx] = False
    return log_prob


def load_clean_races(conn, limit: int | None = None):
    """Yield (race_id, field_size, surface, distance, win_probs, top3_indices).

    Filters out races with:
      - any coupled entries (s.entry = TRUE)
      - DH in top 3 (position_dead_heat among top 3)
      - DQ in top 3
      - incomplete top 3 (missing finish_position 1, 2, or 3)
      - any starter in top 3 missing win_prob
    """
    log.info("Loading clean races (this may take a few minutes)...")

    sample_clause = ""
    if limit is not None:
        sample_clause = f"AND rp.race_id IN (SELECT id FROM races ORDER BY id LIMIT {limit})"

    # Use official_position, not finish_position — official is what the tote pays
    # against (post-DQ, post-objection). bad_races excludes any race where a DQ
    # or DH affects the top 3 of the OFFICIAL order.
    sql = f"""
    WITH bad_races AS (
        SELECT DISTINCT s.race_id
        FROM starters s
        WHERE s.entry = TRUE
           OR (s.position_dead_heat = TRUE AND s.official_position <= 3)
           OR (s.disqualified = TRUE AND s.official_position <= 3)
    )
    SELECT
        rp.race_id, rp.starter_id, rp.win_prob,
        s.official_position AS finish_position,
        r.surface, r.feet AS distance_feet
    FROM race_probabilities rp
    JOIN starters s
      ON s.id = rp.starter_id AND s.race_id = rp.race_id
    JOIN races r
      ON r.id = rp.race_id
    WHERE rp.win_prob IS NOT NULL
      AND rp.race_id NOT IN (SELECT race_id FROM bad_races)
      {sample_clause}
    ORDER BY rp.race_id
    """

    cur = conn.cursor(name="races_cursor", cursor_factory=psycopg2.extras.DictCursor)
    cur.itersize = 50_000
    cur.execute(sql)

    current_race_id = None
    current_rows = []

    def emit(race_rows):
        if not race_rows:
            return None
        rid = race_rows[0]["race_id"]
        surface = race_rows[0]["surface"]
        distance = race_rows[0]["distance_feet"]
        probs = []
        sids = []
        finish = {}  # finish_pos -> index in probs
        for i, row in enumerate(race_rows):
            probs.append(float(row["win_prob"]))
            sids.append(row["starter_id"])
            fp = row["finish_position"]
            if fp in (1, 2, 3):
                finish[fp] = i
        if len(finish) != 3:
            return None
        if len(probs) < 5:  # very small fields are dropped
            return None
        # Renormalize win_probs (in case they don't sum to 1 exactly)
        probs_arr = np.array(probs, dtype=float)
        s = probs_arr.sum()
        if s <= 0:
            return None
        probs_arr /= s
        top3_idx = [finish[1], finish[2], finish[3]]
        # Sanity: probs of top 3 must be positive
        if any(probs_arr[i] <= 0 for i in top3_idx):
            return None
        return (rid, len(probs), surface, distance, probs_arr, top3_idx)

    n_yielded = 0
    n_skipped = 0
    for row in cur:
        if row["race_id"] != current_race_id:
            if current_rows:
                rec = emit(current_rows)
                if rec is not None:
                    yield rec
                    n_yielded += 1
                else:
                    n_skipped += 1
            current_race_id = row["race_id"]
            current_rows = []
        current_rows.append(row)

    if current_rows:
        rec = emit(current_rows)
        if rec is not None:
            yield rec
            n_yielded += 1
        else:
            n_skipped += 1

    cur.close()
    log.info("Yielded %d clean races, skipped %d (incomplete top-3 or tiny field).", n_yielded, n_skipped)


def grid_search(records: list[tuple], k_grid: np.ndarray) -> tuple[float, dict]:
    """Sum log-likelihood across records for each k. Return best k and full curve."""
    log.info("Grid-searching k over %d races × %d k values...", len(records), len(k_grid))
    ll_by_k = {}
    for k in k_grid:
        total = 0.0
        n_valid = 0
        for (_rid, _n, _surf, _dist, probs, top3) in records:
            lp = stern_top3_logprob(probs, top3, float(k))
            if np.isfinite(lp):
                total += lp
                n_valid += 1
        ll_by_k[float(k)] = (total, n_valid)
    best_k = max(ll_by_k, key=lambda k: ll_by_k[k][0])
    return best_k, ll_by_k


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-races", type=int, default=None,
                    help="Sample first N race_ids only (testing).")
    ap.add_argument("--out", default="stern_k_calibration.txt")
    args = ap.parse_args()

    conn = psycopg2.connect(DSN)
    # named/server-side cursors require a live transaction (default mode)

    records = []
    for rec in load_clean_races(conn, limit=args.limit_races):
        records.append(rec)

    log.info("Total clean races materialized: %d", len(records))

    # Phase 1: Global
    log.info("--- Phase 1: Global k ---")
    best_k, curve = grid_search(records, K_GRID)
    log.info("Global best k = %.3f", best_k)
    log.info("Log-likelihood curve (k -> total LL, n_valid):")
    for k in sorted(curve):
        ll, n = curve[k]
        marker = " <-- best" if k == best_k else ""
        log.info("  k=%.2f  ll=%.1f  n=%d%s", k, ll, n, marker)

    # Phase 2: Segmented by field size
    log.info("--- Phase 2: Segmented by field size ---")
    bins = {
        "5-7":   [r for r in records if 5 <= r[1] <= 7],
        "8-10":  [r for r in records if 8 <= r[1] <= 10],
        "11+":   [r for r in records if r[1] >= 11],
    }
    seg_results = {}
    for label, recs in bins.items():
        log.info("Bin %s: %d races", label, len(recs))
        if len(recs) < 1000:
            log.warning("  too few races in bin %s, skipping", label)
            continue
        best, curve_seg = grid_search(recs, K_GRID)
        seg_results[label] = (best, curve_seg)
        log.info("  best k = %.3f", best)

    # Write report
    out_path = Path(args.out)
    with out_path.open("w") as fh:
        fh.write(f"Stern k calibration — {len(records):,} clean races\n")
        fh.write("=" * 60 + "\n\n")
        fh.write(f"Global best k = {best_k:.3f}\n\n")
        fh.write("Global LL curve:\n")
        for k in sorted(curve):
            ll, n = curve[k]
            fh.write(f"  k={k:.2f}  ll={ll:.1f}  n={n}\n")
        fh.write("\n")
        fh.write("By field size:\n")
        for label, (best, curve_seg) in seg_results.items():
            fh.write(f"  {label}: best k = {best:.3f}  (n={len(bins[label])})\n")
            for k in sorted(curve_seg):
                ll, n = curve_seg[k]
                fh.write(f"    k={k:.2f}  ll={ll:.1f}\n")

    log.info("Report written to %s", out_path.resolve())
    conn.close()


if __name__ == "__main__":
    main()
