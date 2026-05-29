SET search_path = handycapper;

-- race_wcmi — Win-pool Concentration / Market Informativeness per race.
-- Captures how concentrated the public's win-pool implied probabilities
-- are (max prob, runner count, derived WCMI score). Populated by
-- compute_wcmi.py.
--
-- Race-day-sim's run_simulation.py reads this to gate conviction
-- thresholds (a high-WCMI race signals strong public consensus, which
-- argues for higher minimum-edge thresholds). race-day-sim CLAUDE.md
-- lists race_wcmi as a dependency. The table was originally created
-- inline by compute_wcmi.py with CREATE TABLE IF NOT EXISTS; that
-- statement still exists for backward compatibility, but this migration
-- is the canonical schema source.

CREATE TABLE IF NOT EXISTS race_wcmi (
    race_id          bigint PRIMARY KEY,
    wcmi             numeric(5, 4),
    n_runners        smallint,
    max_implied_prob numeric(5, 4),
    computed_at      timestamp DEFAULT now()
);
