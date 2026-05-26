# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

Wagering Analytics computes fair value benchmarks for pari-mutuel exotic wagers and identifies systematic mispricings. It sits between the performance measurement layer (RKM) and the application layer (Bet Doctor/Redboarders).

The core output: for any exotic result (exacta, trifecta, superfecta, pick 3/4/5/6), predict what it SHOULD pay given the odds of the finishers, then compare to what it ACTUALLY paid. The ratio reveals where the crowd over/under-bets.

## Architecture

```
handycapper DB
    ├── race_probabilities (V003 materialized view — win probs from odds)
    ├── exotics (bet_type, payoff, unit, pool, winning_numbers)
    ├── exotic_race_legs (maps horizontal legs to races)
    ├── takeout_rates (552 rates across 75 tracks, all bet types)
    └── exotic_harville_ratios (computed by AN1 — 2.9M rows)
         ↓
    scripts/
    ├── populate_stern_fair.py → updates stern_fair column (k=0.81)
    ├── compute_jitter_calibration.py → models/jitter_calibration.json
    └── fit_payoff_models.py → models/payoff_*.pkl
```

## Key Design Decisions

- **Payoffs normalized to per-dollar**: `e.payoff / e.unit` (exactas are often per $2, tris per $1 or $0.50)
- **Takeout per track per bet type**: 552 rates from trktkout.pdf. Default 0.21 (exacta) / 0.24 (trifecta) where track-specific unavailable.
- **Stern k = 0.81 globally**: minimal variation by field size. Single parameter sufficient.
- **Winner identification for horizontals**: use `starters.finish_position = 1` not `race_probabilities.wagering_position = 1` (the latter is NULL for ~7.5% of races)
- **Scripts use psycopg2** (not psycopg3) for compatibility with pandas read_sql and cursor_factory

## Database Tables

| Table | Owner | Purpose |
|---|---|---|
| `exotic_harville_ratios` | This project | 2.9M rows: actual payoff vs Harville/Stern fair value per result |
| `takeout_rates` | This project (shared) | 552 takeout rates across 75 tracks |
| `race_probabilities` | V003 migration | Normalized win probs from tote odds |
| `race_metrics` | V003 migration | HHI, field size, finish_choice_ranks per race |
| `exotic_race_legs` | V002 migration | Maps Pick N legs to individual races |

## Running

```bash
source .venv/bin/activate
python scripts/populate_stern_fair.py        # ~3 min — writes to DB (updates stern_fair column)
python scripts/compute_jitter_calibration.py # ~30 sec — reads DB, writes models/jitter_calibration.json
python scripts/fit_payoff_models.py          # ~2 min — reads DB, writes models/payoff_*.pkl + .json
```

Scripts must run in order. `populate_stern_fair.py` requires `exotic_harville_ratios` to already be populated (done via SQL in the AN1 analysis session).

**What writes where:**
- `populate_stern_fair.py` → UPDATEs `exotic_harville_ratios.stern_fair` column in PostgreSQL
- `compute_jitter_calibration.py` → writes `models/jitter_calibration.json` (local file only)
- `fit_payoff_models.py` → writes `models/payoff_*.pkl` + `models/payoff_coefficients.json` (local files only)

The model JSON files are copied into [race-day-sim](https://github.com/robinhowlett/race-day-sim)/models/ for use during blinded simulations.

## Database

Requires PostgreSQL with the `handycapper` schema. Configure via environment or use defaults:

```bash
export WA_DB_HOST=localhost
export WA_DB_PORT=5432
export WA_DB_NAME=handycapper
export WA_DB_USER=handycapper
export WA_DB_PASSWORD=handycapper
```

## Specs

- `docs/specs/exotic-payoff-analysis.md` — full AN1 specification (5 phases)
- `docs/specs/itp-wagering-framework.md` — ITP's professional wagering principles
