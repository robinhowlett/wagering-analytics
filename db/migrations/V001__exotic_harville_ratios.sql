SET search_path = handycapper;

-- exotic_harville_ratios — per-result Harville/Stern fair-value comparison
-- vs actual chart payoff. Populated by AN1 Phase 1 SQL (one row per
-- vertical exotic result with ehr.payoff_ratio = actual / fair).
--
-- Historically created via raw SQL during the AN1 analysis session and
-- never tracked in a migration. Captured here from production state on
-- 2026-05-29 so a clean wipe-and-rebuild can reconstruct it.

CREATE TABLE IF NOT EXISTS exotic_harville_ratios (
    id              bigserial PRIMARY KEY,
    race_id         bigint,
    bet_type        text,
    winning_numbers text,
    finish_choice_1 smallint,
    finish_choice_2 smallint,
    finish_choice_3 smallint,
    finish_choice_4 smallint,
    fav_on_board    boolean,
    fav_finish_pos  smallint,
    actual_payoff   numeric,
    harville_fair   numeric,
    stern_fair      numeric,
    payoff_ratio    numeric,
    pool_size       numeric,
    field_size      smallint,
    hhi             numeric,
    track           varchar(5),
    surface         varchar(12),
    race_date       date
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_ehr_id
    ON exotic_harville_ratios (id);
