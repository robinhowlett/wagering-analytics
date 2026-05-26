# Wagering Analytics

Empirical analysis of pari-mutuel exotic wagering pools — where the crowd is systematically wrong and how to quantify it.

## What This Does

Computes fair value for every exotic wager result in a 1.8M+ race database using Harville/Stern probability models, then identifies systematic mispricings by the betting public. Produces calibrated models that predict expected payoffs from odds + field structure.

## Key Findings

**Vertical Exotics:**
- "Price over favorite" exactas/trifectas are 15-21% overlays (systematically undervalued by the crowd)
- "Beat the favorite" combinations (fav off board) are 15-25% underlays (over-bet by the crowd)
- Stern k ≈ 0.81 confirmed empirically — Harville overestimates the favorite's placing probability

**Horizontal Exotics:**
- Daily Doubles: efficient (no structural edge)
- Pick 3: situational value (avg premium 21%, median at parity)
- Pick 4/5: typically underpay vs synthetic parlay
- Pick 6: lottery (R² = 0.27, unpredictable from odds alone)

**Payoff Prediction Models:**

| Pool | R² | Median Error | Actionable? |
|---|---|---|---|
| Exacta | 0.90 | 17% | Yes — can reliably identify overlays |
| Trifecta | 0.88 | 27% | Yes |
| Superfecta | 0.70 | 59% | Somewhat |
| Daily Double | 0.84 | 23% | Yes |
| Pick 3 | 0.68 | 41% | Situational |
| Pick 4 | 0.56 | 67% | Marginal |
| Pick 5/6 | 0.27-0.38 | 145-215% | No — too noisy |

## Pipeline

```
handycapper DB (race_probabilities, exotics, takeout_rates)
    ↓
1. populate_stern_fair.py    → stern_fair column in exotic_harville_ratios
2. compute_jitter_calibration.py → models/jitter_calibration.json
3. fit_payoff_models.py      → models/payoff_*.pkl + payoff_coefficients.json
```

Depends on:
- V002 + V003 migrations applied (from `redboarders/db/migrations/`)
- `exotic_harville_ratios` table populated (by AN1 SQL in the current session)

## Theoretical Foundations

| Source | Contribution |
|---|---|
| **Harville (1973)** | Conditional probability model for ordered finishes from win probabilities |
| **Stern (1992)** | Power correction (k < 1) that reduces favorite's placing probability — empirically confirmed at k ≈ 0.81 |
| **ITP (Inside the Pylons)** | "Price on top, favorite underneath" is the systematically undervalued structure; the crowd either backs the fav on top or excludes entirely |
| **Benter (1994)** | Market combination — the public's probability vector is useful but biased; model corrections add value |

## Models Directory

```
models/
├── payoff_EXACTA.pkl           OLS model: log(payoff) ~ log(odds) + field + fav_position
├── payoff_TRIFECTA.pkl         Same for trifecta (3 finish positions)
├── payoff_SUPERFECTA.pkl       Same for superfecta (4 finish positions)
├── payoff_DAILY_DOUBLE.pkl     Horizontal: 2-leg model
├── payoff_PICK_3.pkl           Horizontal: 3-leg model
├── payoff_PICK_4.pkl           Horizontal: 4-leg model
├── payoff_PICK_5.pkl           Horizontal: 5-leg model (low R²)
├── payoff_PICK_6.pkl           Horizontal: 6-leg model (low R²)
├── payoff_coefficients.json    All model coefficients in readable form
└── jitter_calibration.json     Per-leg-position odds uncertainty (σ) for simulation
```

## Setup

```bash
# Requires Python 3.11+, access to handycapper Postgres
python -m venv .venv && source .venv/bin/activate
pip install psycopg2-binary pandas numpy statsmodels scikit-learn joblib

# Configure database (defaults shown)
export WA_DB_HOST=localhost
export WA_DB_PORT=5432
export WA_DB_NAME=handycapper
export WA_DB_USER=handycapper
export WA_DB_PASSWORD=handycapper

# Run in order
python scripts/populate_stern_fair.py        # writes to DB (stern_fair column)
python scripts/compute_jitter_calibration.py # writes models/jitter_calibration.json
python scripts/fit_payoff_models.py          # writes models/payoff_*.pkl + .json
```

Model output files (`models/`) are used by [race-day-sim](https://github.com/robinhowlett/race-day-sim) for quantitative overlay estimation during blinded simulations.

## Related Projects

- [rkm](https://github.com/robinhowlett/rkm) — velocity curve model (measures horse performance)
- [pdf-importer](https://github.com/robinhowlett/pdf-importer) — loads Equibase PDFs into PostgreSQL
- [race-day-sim](https://github.com/robinhowlett/race-day-sim) — blinded backtesting (consumes model outputs)
- [redboarders](https://github.com/robinhowlett/redboarders) — Bet Doctor + Redboarders game (application layer)
