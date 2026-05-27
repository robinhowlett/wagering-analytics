# Market Bias Analysis (AN2): Pre-Race Group-Level Mispricings

## Purpose

AN1 (exotic-payoff-analysis) answers a post-race question: "given who finished where, was the exotic payoff fair?" It calibrates Stern-Harville and produces payoff prediction models.

AN2 answers a pre-race question: **"given observable characteristics BEFORE the race, does the market systematically misprice certain types of horses?"**

These are complementary. AN1 tells you which exotic *structures* are overlaid (trifectas with longshots on top). AN2 tells you which *horses within the field* the crowd gets wrong, before you even construct the ticket.

---

## Core Metrics

### A/E (Actual / Expected)

```
A/E = actual_winners / sum(implied_probability_from_odds)
    = actual_winners / sum(1 / (SP + 1))
```

- A/E > 1.0 → group wins MORE than the market expects → systematically underbet
- A/E < 1.0 → group wins LESS than the market expects → systematically overbet
- Neutral ≈ 0.80-0.82 (reflects takeout — a perfectly-priced group still shows A/E < 1)

**Relative A/E** is more useful than absolute: if the overall population A/E = 0.80, a group at 0.83 is underbet by ~4% relative to baseline, even though absolute A/E < 1.

### Impact Value (IV)

```
IV = (group_winners / total_winners) / (group_runners / total_runners)
   = group_win_rate / overall_win_rate
```

- IV > 1.0 → group wins more than proportional share (ability signal)
- IV < 1.0 → group wins less than proportional share
- IV measures ABILITY. A/E measures MARKET ERROR. They are independent.

A high-IV, neutral-A/E group means: genuinely better horses, correctly priced.
A neutral-IV, high-A/E group means: average horses that the crowd underestimates.
The combination identifies where edge exists AND where the market fails to price it.

### WCMI (Wisdom of Crowd Market Index)

```
WCMI = 1 - H_normalized
     = 1 - (-sum(p_i * log_n(p_i)))
where:
  p_i = implied probability of runner i (from normalized odds)
  n = number of runners in the race
  H_normalized = Shannon entropy normalized to [0, 1]
```

- WCMI → 0: maximum entropy, all runners equal price, crowd knows nothing
- WCMI → 1: minimum entropy, one runner at prohibitive odds, outcome "known"
- WCMI < 0.13: uninformed market — model edge is maximized
- WCMI 0.13-0.20: moderately informed — model adds value at margins
- WCMI > 0.20: well-informed — model edge is minimal

WCMI is computed per-race and becomes a feature that informs bet sizing and race selection.

---

## Phase Structure

### Phase 1: A/E Tables by Factor

For each factor from research-plan Items 3-12, compute A/E segmented by the factor's values. Output: a lookup table of relative A/E by factor level.

| Factor | Segmentation | Expected finding |
|---|---|---|
| Carried weight | 5 weight buckets | Top weights overbet (A/E lower) |
| Post position | PP 1-12, by track × zone | Extreme PPs at biased tracks mispriced |
| Medication change | First Lasix, first blinkers, no change | First-time Lasix underbet? |
| Trainer change | Claimed in last 180 days vs not | Claimed horses underbet? |
| Track condition | Fast/off, by horse's condition history | Mud specialists underbet on off tracks? |
| Jockey allowance | 0 / 3 / 5 / 7 lb | 5lb apprentices underbet |
| Surface switch | First-time turf, first-time dirt | Untested-on-surface underbet? |
| Trainer first-out record | Tier by FTS win rate | High FTS trainers underbet in maidens? |
| Jockey track record | Win % at meet vs career | Hot jockeys at this track underbet? |
| Days since last race | Layoff buckets | Freshened horses over/underbet? |

**Critical design choice:** Each factor must be computable from PRE-RACE data only. No future information leakage. For "trainer first-out record," this means computing the record AS OF the race date (point-in-time), not career-total.

### Phase 2: WCMI Computation

Compute WCMI for every race in the database (1999-2017, where odds data exists). Store as a column on the `race_metrics` table or a new `race_wcmi` table.

Additionally, compute WCMI by race category to validate the hypothesis:
- Maiden races should have lower WCMI than open claiming
- Stakes races should have higher WCMI (more public information)
- Races with first-time starters should have lower WCMI

### Phase 3: IV Tables for Limited-Form Contexts

For races where individual horse curves don't exist (maidens, first-time starters), compute IV by:

| Context | Segmentation | Use case |
|---|---|---|
| Trainer × surface × zone | FTS on dirt sprint vs FTS on turf route | Which trainers win fresh on which surface? |
| Trainer × race type | MSW vs MCL maiden debut | Different trainers target different maiden types |
| Jockey × track × meet | Win rate this meet vs historical | Who's riding hot here right now? |
| Sire × surface × zone | First-crop sire stats by surface/distance | Does this sire produce dirt sprinters or turf routers? |
| Gender × race type | Colts vs fillies in maiden races by type | Where does gender predict best? |

These are all point-in-time computations (backward-looking from race date). They serve as the **substitute signal** for Item 10 (limited-form race assessment).

### Phase 4: Composite Edge Score

Combine individual factor A/Es into a single pre-race "market bias score" per starter:

```
edge_score = sum(relative_A/E_adjustment for each applicable factor)
```

For example, a horse that is:
- Carrying light weight (relative A/E +2%)
- Drawing post 1 at a biased track (relative A/E -1% — the market already knows)
- First-time Lasix (relative A/E +3%)
- Claimed last race (relative A/E +2%)

Would get a composite edge_score of +6% — meaning the market historically misprices this profile by 6% in aggregate. This becomes an input to the Value calculation in race-day-sim.

---

## Confidence and Sample Size

Following FlatStats' Exp/Archie framework:

**Exp (Expected Wins):** The sum of implied probabilities for all runners in a group. Once Exp ≥ 5.0, the sample is large enough that the A/E figure is meaningful. Below 5.0, the result could easily be random.

**Archie (Chi-squared significance):** Tests whether the observed A/E deviates from 1.0 by more than chance. Archie ≥ 3.0 (medium confidence) or ≥ 5.0 (strong confidence) indicates the bias is real, not noise.

```
Archie = (A - E)² / E
where A = actual winners, E = expected winners from odds
```

All A/E tables should include both metrics. Factors with Exp < 5 or Archie < 3 are flagged as insufficient evidence.

---

## Relationship to Existing Architecture

```
┌─────────────────────┐     ┌─────────────────────────────┐
│   RKM (physics)     │     │  wagering-analytics         │
│                     │     │                             │
│  velocity curves    │     │  AN1: exotic fair value     │
│  normalization      │     │    (Stern k, payoff OLS)    │
│  current form       │     │                             │
│                     │     │  AN2: market bias  ← NEW    │
│  → individual       │     │    (A/E, IV, WCMI tables)   │
│    horse rating     │     │    → group-level edge       │
└────────┬────────────┘     └──────────────┬──────────────┘
         │                                  │
         └──────────┬───────────────────────┘
                    ↓
         ┌─────────────────────┐
         │  race-day-sim       │
         │                     │
         │  rating + bias →    │
         │  probability →      │
         │  overlay →          │
         │  bet construction   │
         └─────────────────────┘
```

### What changes in wagering-analytics:

1. **New scripts:** `scripts/compute_ae_tables.py`, `scripts/compute_wcmi.py`, `scripts/compute_iv_tables.py`
2. **New output files:** `models/ae_tables.json`, `models/wcmi_summary.json`, `models/iv_tables.json`
3. **New DB tables (optional):** `race_wcmi` (per-race WCMI), `factor_ae` (A/E by factor × level)

### What changes in race-day-sim:

1. **Consult A/E tables during handicapping:** When assessing a horse with first-time Lasix at an inside post with an apprentice jockey, the Market Bias Layer quantifies the aggregate historical edge from those characteristics.
2. **Use WCMI for race selection and sizing:** Low-WCMI races get larger allocations; high-WCMI races get smaller ones or are passed.
3. **Use IV tables for maiden/FTS assessment:** Replace "no data = pass" with "this trainer's first-out IV on this surface is 1.45 → include on tickets."

### What does NOT change:

- AN1 exotic fair value models (still valid)
- RKM velocity curves (still pure physics)
- The blinder protocol (A/E tables are historical aggregates, not post-race data)
- Payoff prediction models (still valid — they predict what the exotic SHOULD pay)

---

## Output Format

### ae_tables.json
```json
{
  "weight_carried": {
    "metadata": {"n_races": 645917, "date_range": "2000-2017", "archie": 12.4},
    "levels": {
      "<=114": {"ae": 0.814, "iv": 0.882, "n": 395800, "relative_ae": 1.018},
      "115-117": {"ae": 0.800, "iv": 0.966, "n": 1086337, "relative_ae": 1.000},
      "118-120": {"ae": 0.801, "iv": 1.009, "n": 2079761, "relative_ae": 1.001},
      "121-123": {"ae": 0.797, "iv": 1.030, "n": 1265497, "relative_ae": 0.996},
      "124+": {"ae": 0.789, "iv": 1.080, "n": 337369, "relative_ae": 0.986}
    }
  },
  "jockey_allowance": { ... },
  "post_position": { ... },
  "first_time_lasix": { ... }
}
```

### race_wcmi (DB table or per-race computation)
```sql
CREATE TABLE handycapper.race_wcmi (
    race_id BIGINT PRIMARY KEY REFERENCES races(id),
    wcmi NUMERIC(5,4),
    n_runners SMALLINT,
    max_implied_prob NUMERIC(5,4),
    computed_at TIMESTAMP DEFAULT NOW()
);
```

---

## Execution Priority

| Phase | Depends on | Effort | Blocks |
|---|---|---|---|
| 1: A/E tables | Research Items 3-9 complete | Medium | race-day-sim value assessment |
| 2: WCMI | Odds data (1999+) | Low | bet sizing, race selection |
| 3: IV tables | Research Items 10, 12 | High (point-in-time) | maiden race assessment |
| 4: Composite score | Phases 1-3 | Low (aggregation) | automated bet construction |

Phase 2 (WCMI) can be computed immediately — it only needs odds data. Phases 1 and 3 should wait for the remaining research items to determine which factors have meaningful A/E deviations worth including.
