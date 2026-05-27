# Exotic Payoff Analysis: Empirical Calibration of Fair Value Models

## Purpose

Before we can simulate a race day with confidence, we need to know whether Harville's formula is actually a useful benchmark for fair value — and where it systematically lies. The hypothesis, supported by decades of academic literature and ITP's practitioner framework alike, is that the betting public concentrates their money in predictable ways: favorites are over-backed in exotics, chalk-involving combinations consistently underpay relative to Harville's theoretical price, and when favorites miss the board the payoffs spike in ways that Harville cannot fully explain. This analysis proves or disproves that hypothesis against our own 1MM+ race database, derives empirical correction factors for the Harville model, and produces the calibration data that will feed both the simulation spec and the odds-jitter model for horizontal legs.

---

## Foundational Theory

### Harville Formula

Given a win probability vector `p_1, p_2, ..., p_n` for an n-horse field, the Harville probability of horse A winning and horse B finishing second is:

```
P(A wins, B 2nd) = p_A × p_B / (1 - p_A)
```

And for horse C finishing third:

```
P(A wins, B 2nd, C 3rd) = p_A × [p_B / (1 - p_A)] × [p_C / (1 - p_A - p_B)]
```

The fair trifecta price for a $1 wager on A-B-C is then `1 / P(A wins, B 2nd, C 3rd)`, before takeout. Actual payoff below this price means the combination was over-bet; above it means it was under-bet.

### Harville's Known Bias

Harville overestimates the probability of the race favorite finishing second or third. The intuition: a horse that ran hard enough to lead and win has depleted energy reserves that a horse saving ground in third does not share. The favorite's conditional probability of placing, given that it did not win, is lower than the Harville formula implies. Stern (1992) proposed a correction:

```
p_i^adjusted = p_i^k / Σ(p_j^k)     where k ≈ 0.81 for thoroughbreds
```

Rather than accepting 0.81 as given, this analysis derives our own empirically calibrated `k` parameter — segmented by field size, surface, and track tier — from the actual finishing distributions in our data.

### Win Probabilities

All win probabilities are derived from `race_probabilities` (V003 view), which normalizes final tote odds using the standard overround correction:

```
p_i = (1 / (odds_i + 1)) / Σ(1 / (odds_j + 1))
```

This is the public's implied probability vector — a useful starting point even though our RKM model will later supplement it. For the purpose of this analysis, the public's own probability vector is what we test Harville against, since that is the vector the public implicitly assumed when constructing their tickets.

---

## Data Available

| Table / View | Relevant Fields |
|---|---|
| `race_probabilities` | `race_id`, `starter_id`, `program`, `odds`, `choice`, `win_prob`, `finish_position`, `wagering_position` |
| `race_metrics` | `race_id`, `field_size`, `hhi`, `fav_prob`, `finish_choice_ranks`, `finish_order` |
| `exotics` | `race_id`, `bet_type`, `winning_numbers`, `payoff`, `pool` |
| `exotic_race_legs` | Maps Pick 3/4/5/6 legs to individual race ids |
| `races` | `track`, `date`, `surface`, `type`, `furlongs`, `number_of_runners` |
| `takeout_rates` | Per-track, per-pool-type takeout percentages |

The `finish_choice_ranks` array on `race_metrics` (populated by V003) encodes the choice rank of each finisher in order — e.g., `{3, 1, 5}` means the 3rd choice won, the favorite ran 2nd, and the 5th choice ran 3rd. This is the primary join key for all payoff ratio analyses.

---

## Phase 1: Harville Baseline — Computing Fair Value for Every Exotic Result

For every vertical exotic result in the database (Exacta, Trifecta, Superfecta) with a non-null payoff, compute the Harville-implied fair price and the ratio of actual payoff to fair price.

### Exacta

```sql
WITH win_probs AS (
    SELECT race_id, program, choice, win_prob,
           wagering_position
    FROM race_probabilities
    WHERE wagering_position IS NOT NULL
),
exacta_results AS (
    SELECT
        e.race_id,
        e.payoff,
        e.pool,
        e.winning_numbers,
        wp1.win_prob AS p_win,
        wp1.choice AS winner_choice,
        wp2.win_prob AS p_second,
        wp2.choice AS second_choice,
        -- Harville P(A wins, B 2nd)
        wp1.win_prob * wp2.win_prob / (1 - wp1.win_prob) AS harville_prob,
        -- Fair price before takeout
        (1 - tr.rate) / (wp1.win_prob * wp2.win_prob / (1 - wp1.win_prob)) AS harville_fair_price
    FROM exotics e
    JOIN races r ON r.id = e.race_id
    JOIN win_probs wp1 ON wp1.race_id = e.race_id AND wp1.wagering_position = 1
    JOIN win_probs wp2 ON wp2.race_id = e.race_id AND wp2.wagering_position = 2
    JOIN takeout_rates tr ON tr.track = r.track AND tr.bet_type = 'EXACTA'
    WHERE e.bet_type = 'EXACTA'
      AND e.payoff IS NOT NULL
      AND wp1.win_prob < 1  -- avoid degenerate cases
)
SELECT
    race_id,
    winner_choice,
    second_choice,
    payoff,
    harville_fair_price,
    payoff / harville_fair_price AS payoff_ratio,  -- >1 = overlay, <1 = underlay
    pool
FROM exacta_results;
```

The output table `exotic_harville_ratios` stores one row per exotic result with the payoff ratio. Trifecta and Superfecta follow the same pattern with chained conditional probabilities.

### Output Table

```sql
CREATE TABLE exotic_harville_ratios (
    race_id         BIGINT,
    bet_type        TEXT,          -- EXACTA, TRIFECTA, SUPERFECTA
    winning_numbers TEXT,
    finish_choice_1 INT,           -- choice rank of winner
    finish_choice_2 INT,           -- choice rank of 2nd
    finish_choice_3 INT,           -- choice rank of 3rd (NULL for exacta)
    finish_choice_4 INT,           -- choice rank of 4th (NULL for tri/exacta)
    fav_on_board    BOOLEAN,       -- did the favorite finish in the money?
    fav_finish_pos  INT,           -- what position did the favorite actually finish?
    actual_payoff   NUMERIC,
    harville_fair   NUMERIC,
    stern_fair      NUMERIC,       -- populated in Phase 3
    payoff_ratio    NUMERIC,       -- actual / harville_fair
    pool_size       NUMERIC,
    field_size      INT,
    hhi             NUMERIC,
    track           TEXT,
    surface         TEXT,
    race_date       DATE
);
```

---

## Phase 2: Payoff Ratio Matrix — Systematic Bias by Choice Rank Combination

With `exotic_harville_ratios` populated, aggregate to reveal the systematic over/underpayment pattern by who finished where. The key insight will be visible in these matrices: combinations involving the favorite in favored positions will cluster below 1.0 (underlay); combinations where the favorite misses or finishes deep will cluster above 1.0 (overlay).

### Exacta Matrix

```sql
SELECT
    finish_choice_1 AS winner_choice,
    finish_choice_2 AS second_choice,
    COUNT(*)        AS instances,
    AVG(payoff_ratio)                                          AS avg_ratio,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY payoff_ratio) AS median_ratio,
    PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY payoff_ratio) AS p25_ratio,
    PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY payoff_ratio) AS p75_ratio,
    AVG(actual_payoff)                                         AS avg_actual_payoff,
    AVG(harville_fair)                                         AS avg_harville_fair
FROM exotic_harville_ratios
WHERE bet_type = 'EXACTA'
  AND finish_choice_1 <= 6
  AND finish_choice_2 <= 6
GROUP BY 1, 2
ORDER BY 1, 2;
```

The same structure applies for the Trifecta 3-way matrix (winner × 2nd × 3rd), though cells thin out quickly beyond choice rank 5. Aggregate choice ranks 5+ into a single "price" bucket to maintain sample size.

### Favorite Position Segmentation

The more analytically interesting cut segments not by the winner's choice rank but by where the favorite finished:

```sql
SELECT
    fav_finish_pos,
    CASE
        WHEN fav_finish_pos = 1 THEN 'Favorite wins'
        WHEN fav_finish_pos = 2 THEN 'Favorite 2nd'
        WHEN fav_finish_pos = 3 THEN 'Favorite 3rd'
        WHEN fav_finish_pos = 4 THEN 'Favorite 4th'
        ELSE 'Favorite off board'
    END AS fav_scenario,
    bet_type,
    COUNT(*)        AS instances,
    AVG(payoff_ratio)   AS avg_ratio,
    AVG(actual_payoff)  AS avg_actual,
    AVG(harville_fair)  AS avg_harville_fair,
    AVG(pool_size)      AS avg_pool
FROM exotic_harville_ratios
GROUP BY 1, 2, 3
ORDER BY 3, 1;
```

This produces the core table that answers the most important question: what happens to the payoff ratio at each level of favorite involvement? We expect a monotonic relationship — the deeper the favorite finishes, the higher the payoff ratio. But the magnitude and shape of that relationship is what the data will tell us.

### HHI Interaction

Field competitiveness interacts with the favorite-position effect: in a chalky race (high HHI), the public's ticket concentration is more severe and the overlay when the favorite misses will be larger. In a spread race (low HHI), the public is already somewhat distributed and the effect attenuates. Cross-tab payoff ratios against HHI quintiles within each fav_scenario bucket.

---

## Phase 3: Stern Parameter Calibration

The Stern formula replaces Harville's conditional probabilities with a power-transformed version:

```
P(horse i finishes kth | horses 1..k-1 have already finished) =
    p_i^k / Σ_remaining p_j^k
```

where `k` is the Stern exponent. When `k=1`, this reduces to Harville. When `k<1`, it shrinks the probability difference between the favorite and the field in conditional positions — exactly the correction we need.

To calibrate `k` empirically: for every trifecta result, compute the Stern fair price across a grid of k values from 0.5 to 1.2, find the k that minimizes the mean squared log error between predicted and actual payoff across all results. Do this segmented by field size (≤7, 8-10, 11+), surface (dirt/turf/synthetic), and track tier (graded stakes, ungraded stakes, claiming, maiden).

```sql
-- Calibration query structure (run iteratively via Python for k grid search)
SELECT
    field_size_bucket,
    surface,
    race_type_bucket,
    k_value,
    AVG((LN(actual_payoff) - LN(stern_fair_price(k_value)))^2) AS mse_log
FROM exotic_harville_ratios
GROUP BY 1, 2, 3, 4
ORDER BY 1, 2, 3, 4;
```

The output is a `stern_calibration` table storing the optimal k per segment. These values replace the fixed 0.81 in all downstream fair-value computations.

---

## Phase 4: Horizontal Exotic Analysis

Horizontal exotics (Daily Double, Pick 3/4/5/6) are theoretically just parlays. The fair price for a Pick 3 covering horses A, B, C in legs 1, 2, 3 is:

```
fair_price = 1 / (p_A × p_B × p_C) × (1 - takeout)
```

The actual pool price should approximate this for an efficient market. The deviation — the pool-to-parlay premium — tells us how much the horizontal pool over- or under-prices combinations relative to the product of the individual leg win probabilities.

### Parlay Premium Computation

```sql
WITH legs AS (
    SELECT
        erl.exotic_id,
        erl.race_id,
        erl.leg_number,
        rp.win_prob AS winner_prob
    FROM exotic_race_legs erl
    JOIN race_probabilities rp
        ON rp.race_id = erl.race_id
        AND rp.wagering_position = 1   -- the actual winner of that leg
),
sequence_parlay AS (
    SELECT
        exotic_id,
        EXP(SUM(LN(winner_prob))) AS parlay_win_prob,  -- product of leg probabilities
        COUNT(*) AS legs
    FROM legs
    GROUP BY exotic_id
)
SELECT
    e.bet_type,
    sp.legs,
    e.payoff,
    (1 - tr.rate) / sp.parlay_win_prob AS parlay_fair_price,
    e.payoff / ((1 - tr.rate) / sp.parlay_win_prob) AS pool_to_parlay_ratio,
    e.pool
FROM exotics e
JOIN sequence_parlay sp ON sp.exotic_id = e.id
JOIN exotic_race_legs erl_first ON erl_first.exotic_id = e.id AND erl_first.leg_number = 1
JOIN races r ON r.id = erl_first.race_id
JOIN takeout_rates tr ON tr.track = r.track AND tr.bet_type = e.bet_type
WHERE e.payoff IS NOT NULL;
```

A pool-to-parlay ratio above 1.0 means the horizontal paid more than the synthetic parlay — the pool was inefficient in favor of the bettor. Below 1.0 means the pool was over-bought on that combination (the crowd singled the winning sequence).

### Sequence-Level Favorite Involvement

The ITP thesis predicts that sequences with a bad favorite in one or more legs produce the largest pool-to-parlay premiums, because the public over-concentrates on the favorite in those legs and the pool under-bets the remaining combinations. Test this by computing the average pool-to-parlay ratio segmented by how many legs in the sequence contained a "bad favorite" (where the favorite's choice rank is 1, HHI > 0.20, and the favorite ultimately missed the board).

---

## Phase 5: Odds Jitter Calibration for Horizontal Simulation

When simulating a horizontal wager, the first leg has near-final closing odds available, but later legs do not. We model the uncertainty as a log-normal jitter applied to the closing odds, with standard deviation increasing by leg position in the sequence.

The calibration comes from the distribution of payoff ratios within each leg position: the variance in the payoff ratio for leg-2 winners relative to leg-1 winners, holding the final odds constant, reflects how much additional uncertainty existed in leg 2 at the time of bet construction. That variance, translated back through the odds-probability mapping, gives us the standard deviation of the log-normal jitter parameter per leg position.

```sql
SELECT
    erl.leg_number,
    e.bet_type,
    COUNT(*)                                                               AS instances,
    STDDEV(LN(rp.odds))                                                    AS log_odds_stddev,
    STDDEV(LN(winner_payoff.payoff / parlay_fair_price))                   AS payoff_ratio_stddev
FROM exotic_race_legs erl
JOIN exotics e ON e.id = erl.exotic_id
JOIN race_probabilities rp ON rp.race_id = erl.race_id AND rp.wagering_position = 1
-- [join parlay fair price from Phase 4 subquery]
GROUP BY 1, 2
ORDER BY 2, 1;
```

The result is a jitter calibration table: `leg_position → log_normal_sigma`, to be used in the simulation's pre-race odds projection model.

---

## Phase 6: Payoff Prediction Model

Given the odds of each finisher and the pool size, predict what the exotic payoff would be before the race is run. This closes the loop on the simulation: rather than only comparing actual payoffs to Harville fair after the fact, we can estimate expected payoffs prospectively as part of bet construction.

### Why Log-Linear Regression

Payoffs are multiplicative — doubling all three horses' odds should roughly quadruple the trifecta payoff, so log-space is the natural domain. The residuals around Harville fair are approximately log-normal (confirmed by the AN1 payoff ratio distributions), making OLS on log-transformed variables the right estimator. With 1.07M trifecta rows the coefficients will be precise to multiple decimal places.

### Model Taxonomy — One Model Per Bet Type

Each bet type gets its own fitted model because the crowd's behaviour, pool dynamics, and feature dimensionality all differ. All share the same log-linear structure; they differ in the number of finisher/leg terms and which covariates apply.

#### Verticals (single race, ordered finish)

| Bet Type | Finisher Terms | Notes |
|---|---|---|
| Exacta | `log(winner_odds)`, `log(second_odds)` | 2 terms; largest sample (~1.09M rows) |
| Trifecta | + `log(third_odds)` | 3 terms; ~1.07M rows |
| Superfecta | + `log(fourth_odds)` | 4 terms; thinner but still 100K+ |
| Hi-5 | + `log(fifth_odds)` | 5 terms; smallest sample, use ridge |
| Quinella | `log(odds_A + odds_B)` | Unordered pair; Harville baseline differs |

All vertical models share the same structural covariates beyond finisher odds: `log_pool`, `field_size`, `hhi`, `surface`, `fav_in_combo`, `fav_position` (categorical: won/2nd/3rd/4th/off-board), and the key interaction `log_winner_odds:fav_position`.

Superfecta and Hi-5 have sparse coverage of extreme-price combinations. Use ridge regression (`sklearn.linear_model.Ridge`) with λ selected by 5-fold cross-validation rather than OLS, to avoid overfitting on the long tail.

#### Horizontals (multi-race, winners only)

| Bet Type | Leg Terms | Notes |
|---|---|---|
| Daily Double | `log(leg1_winner_odds)`, `log(leg2_winner_odds)` | Near-efficient; EV signal weak |
| Pick 3 | 3 leg winner odds | Situational overlay confirmed in AN1 |
| Pick 4 | 4 leg winner odds | Structural underlay vs parlay; include |
| Pick 5 | 5 leg winner odds | Same pattern as Pick 4 |
| Pick 6 | 6 leg winner odds | All-up races only (no carryover days) |

Horizontal models replace `surface` with per-leg HHI values (`hhi_leg1` … `hhi_legN`) and add a `bad_fav_legs` count (number of legs where choice 1 was predicted vulnerable and missed the board). Pool size for horizontals is the combined sequence pool. Pick 6 is fit only on races where `exotics.carryover = 0` or `carryover IS NULL`; carryover Pick 6 pools are too distorted to model cleanly and are excluded.

### Model Specification

One model per bet type, fit separately. For Trifecta (the reference case):

```python
import statsmodels.formula.api as smf
import pandas as pd
import numpy as np

# Load from exotic_harville_ratios
df = pd.read_sql("""
    SELECT
        ehr.actual_payoff,
        ehr.pool_size,
        ehr.field_size,
        ehr.hhi,
        ehr.surface,
        -- Odds of each finisher (join back to race_probabilities)
        rp1.odds  AS winner_odds,
        rp2.odds  AS second_odds,
        rp3.odds  AS third_odds,
        ehr.finish_choice_1,
        ehr.finish_choice_2,
        ehr.finish_choice_3,
        ehr.fav_finish_pos,
        -- Derived
        (ehr.finish_choice_1 = 1 OR ehr.finish_choice_2 = 1 OR ehr.finish_choice_3 = 1)::int
                  AS fav_in_combo,
        ehr.finish_choice_1 = 1 AS fav_won,
        ehr.finish_choice_2 = 1 AS fav_second,
        ehr.finish_choice_3 = 1 AS fav_third
    FROM exotic_harville_ratios ehr
    -- [join race_probabilities for odds of each finisher]
    WHERE ehr.bet_type = 'TRIFECTA'
      AND ehr.actual_payoff IS NOT NULL
      AND ehr.pool_size > 0
""", conn)

df['log_payoff']      = np.log(df['actual_payoff'])
df['log_winner_odds'] = np.log(df['winner_odds'] + 1)   # +1: convert tote odds to net
df['log_second_odds'] = np.log(df['second_odds'] + 1)
df['log_third_odds']  = np.log(df['third_odds']  + 1)
df['log_pool']        = np.log(df['pool_size'])

formula = """
    log_payoff ~ log_winner_odds + log_second_odds + log_third_odds
               + log_pool + field_size + hhi
               + fav_in_combo + fav_won + fav_second + fav_third
               + log_winner_odds:fav_second    # price wins, fav runs 2nd
               + log_winner_odds:fav_third     # price wins, fav runs 3rd
               + C(surface)
"""

model = smf.ols(formula, data=df).fit()
print(model.summary())
```

The interaction terms `log_winner_odds:fav_second` and `log_winner_odds:fav_third` are the mathematical encoding of the AN1 finding: the overlay on "price over favorite" combos should show up as positive, significant coefficients on these interactions.

### Prediction Function

```python
def predict_payoff(
    finisher_odds: list[float],  # [winner, 2nd, 3rd, ...] for verticals
                                 # [leg1_winner, leg2_winner, ...] for horizontals
    all_field_odds: list[float], # full odds vector for all runners (verticals)
                                 # or list of per-leg winner odds (horizontals)
    pool_size:  float,           # total pool in dollars
    field_sizes: list[int],      # [field_size] for verticals, one per leg for horizontals
    hhi_values: list[float],     # [hhi] for verticals, one per leg for horizontals
    surface:    str,             # 'D', 'T', 'S' — verticals only, pass None for horizontals
    track:      str,             # for takeout lookup
    bet_type:   str,             # 'EXACTA', 'TRIFECTA', 'SUPERFECTA', 'HI_5',
                                 # 'QUINELLA', 'DAILY_DOUBLE', 'PICK_3' ... 'PICK_6'
    k:          float = 0.81,
) -> dict:
    """
    Unified payoff prediction for any exotic bet type.
    Returns three estimates: Harville/parlay fair, AN1-calibrated, and regression.
    Harville/parlay_fair is None for horizontals/verticals respectively.
    """
    takeout  = lookup_takeout(track, bet_type)
    is_horiz = bet_type in ('DAILY_DOUBLE','PICK_3','PICK_4','PICK_5','PICK_6')

    if not is_horiz:
        # Vertical: Stern-corrected Harville
        probs      = odds_to_probs(all_field_odds)
        harville   = stern_harville_fair(probs, finishers=range(len(finisher_odds)),
                                         k=k, takeout=takeout)
        choices    = [rank_from_odds(o, all_field_odds) for o in finisher_odds]
        ratio      = lookup_ratio_matrix(bet_type, *choices)
        calibrated = harville * ratio
        fair_key   = "harville_fair"
    else:
        # Horizontal: parlay fair price
        leg_probs  = [1.0 / (o + 1) for o in finisher_odds]  # rough win prob per leg
        parlay_p   = 1.0
        for p in leg_probs:
            parlay_p *= p
        harville   = (1 - takeout) / parlay_p
        # AN1 calibration: apply pool-to-parlay ratio for this bet type
        ratio      = lookup_parlay_premium(bet_type)
        calibrated = harville * ratio
        fair_key   = "parlay_fair"

    # Regression prediction (works for both families)
    X        = build_features(finisher_odds, pool_size, field_sizes,
                               hhi_values, surface, bet_type)
    pred     = MODELS[bet_type].get_prediction(X).summary_frame()
    log_yhat = pred['mean'].iloc[0]
    log_se   = pred['mean_se'].iloc[0]
    regression = np.exp(log_yhat)
    ci_90      = (np.exp(log_yhat - 1.645 * log_se),
                  np.exp(log_yhat + 1.645 * log_se))

    return {
        fair_key:     round(harville, 2),
        "calibrated": round(calibrated, 2),
        "regression": round(regression, 2),
        "ci_90":      (round(ci_90[0], 2), round(ci_90[1], 2)),
    }
```

### Expected Coefficient Signs

| Feature | Expected sign | Interpretation |
|---|---|---|
| `log_winner_odds` | + | Higher-priced winner → bigger payoff |
| `log_second_odds` | + | Higher-priced 2nd → bigger payoff |
| `log_third_odds` | + | Higher-priced 3rd → bigger payoff |
| `log_pool` | − | Larger pool → smaller per-ticket share (more tickets outstanding) |
| `field_size` | + | More runners → more possible combinations → less crowd coverage per combo |
| `hhi` | − | Chalky field → crowd concentrates → underpayment |
| `fav_in_combo` | − | Favorite on ticket → crowd over-bet it |
| `fav_second` | + | Favorite 2nd with price on top → crowd ignored this structure |
| `fav_third` | + | Favorite 3rd with price on top → same |
| `log_winner_odds:fav_second` | + | The bigger the price on top, the larger the overlay |

If the signs come out as expected, the regression is both well-specified and a direct quantitative confirmation of the ITP framework.

### Model Storage

Fitted models are serialized to `rkm/models/payoff_{bet_type}.pkl` using `joblib`. The `predict_payoff()` function loads them at import time. Coefficients are also written to a human-readable `rkm/models/payoff_coefficients.json` for inspection and documentation.

### Fit Diagnostics

After fitting, report:
- R² and adjusted R² (expect 0.75–0.85 for trifecta)
- Residual plot: log(actual) vs log(predicted) — should be uniform scatter around the diagonal
- Coefficient p-values: all primary features should be p < 0.001 at this sample size
- Out-of-sample RMSE: 20% holdout set stratified by year

---

## Deliverables

| Output | Description | Feeds Into |
|---|---|---|
| `exotic_harville_ratios` | Per-result payoff ratio vs Harville for all verticals | Phases 2, 3, 6 |
| Exacta payoff ratio matrix | Avg ratio by winner_choice × second_choice | Simulation spec |
| Trifecta payoff ratio matrix | Avg ratio by winner × 2nd × 3rd choice buckets | Simulation spec |
| Favorite position table | Avg ratio segmented by fav_finish_pos × bet_type | Core ITP hypothesis validation |
| `stern_calibration` | Optimal k per field_size × surface × race_type segment | Replaces fixed 0.81 in all fair-value models |
| Horizontal pool-to-parlay table | Premium/discount by bet_type × sequence fav involvement | Simulation horizontal spec |
| Jitter calibration table | log_normal_sigma per leg position | Simulation odds projection model |
| `payoff_{bet_type}.pkl` | Fitted log-linear regression models per bet type | `predict_payoff()` function, simulation |
| `payoff_coefficients.json` | Human-readable coefficients + p-values | Documentation, sanity check |

---

## Key Questions This Analysis Answers

1. Do favorites in exactas consistently underpay relative to Harville? By how much, and does it vary by field size?
2. What is the payoff ratio inflection point — at what level of favorite involvement does the exotic flip from underlay to overlay?
3. When the favorite finishes 2nd (still in the exacta), does the exacta still underpay because the crowd saved them underneath? Does the trifecta then become interesting because 3rd is genuinely uncovered?
4. When the favorite is completely off the board, how does the trifecta payoff ratio distribute across different winner choice ranks — is the overlay concentrated in short-priced winners emerging from chaos, or broadly distributed?
5. What is the empirically optimal Stern k for our dataset, and how much does it vary by race type and surface?
6. Do horizontal pools systematically over- or under-price sequences relative to their synthetic parlay value? Does the presence of a bad favorite in a leg reliably inflate the premium?
7. What is the log-normal standard deviation of odds uncertainty per leg position in a horizontal sequence — i.e., how wrong is the morning line, on average, expressed as a distributional parameter?
8. Given the odds of each finisher and a pool size, what is the predicted payoff for any exotic bet type — and what is the 90% prediction interval? Do the regression coefficients confirm the AN1 payoff ratio matrix findings across all verticals?
9. Do horizontal pools (Pick 3–6) show structurally different overlay patterns by leg count, and does the presence of a predicted bad favorite in a leg reliably inflate the pool-to-parlay premium?

---

## Status

`spec` — ready for construction once V002 and V003 migrations are confirmed live on the database.

## Dependencies

- V002 migration (canonical bet_type, exotic_race_legs) — must be applied before Phase 4
- V003 materialized views (race_probabilities, race_metrics) — must be refreshed before Phases 1-3
- `takeout_rates` table populated (E4) — required for fair-value computation

## Next Spec

`docs/specs/race-day-simulation.md` — the pre-race blinder, form presentation, odds jitter model, Harville/Stern probability engine, Kelly staking, and post-race reveal. Depends on the calibration outputs of this analysis.
