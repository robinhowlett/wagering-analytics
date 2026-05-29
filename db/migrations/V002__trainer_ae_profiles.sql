SET search_path = handycapper;

-- trainer_ae_profiles — aggregate trainer A/E profile across 6 dimensions
-- (overall, FTS, claim, drop, layoff, switch, jock_upgrade). Populated by
-- compute_trainer_profiles.py.
--
-- Audit WA-T1.4 (2026-05-27) flagged this table for static-aggregate
-- leakage (dates inside the 2005-2017 range used to fit the profiles
-- can subsequently see those same dates as "history"). race-day-sim
-- never queries this table during simulation — load_market_bias does
-- the trainer-A/E computation point-in-time inside the bias query
-- (slow until the racing-stats snapshots replaced it).
--
-- Existence is the risk. Recommend dropping after the racing-stats
-- snapshots fully supersede it. Until then, this migration captures
-- the schema so a clean wipe-and-rebuild can reconstruct what
-- compute_trainer_profiles.py expects.

CREATE TABLE IF NOT EXISTS trainer_ae_profiles (
    trainer_key         varchar(200) PRIMARY KEY,
    trainer_last        varchar(100),
    trainer_first       varchar(100),
    total_starts        integer,
    overall_win_pct     numeric(5, 4),
    overall_ae          numeric(5, 3),
    fts_starts          integer,
    fts_ae              numeric(5, 3),
    claim_starts        integer,
    claim_ae            numeric(5, 3),
    drop_starts         integer,
    drop_ae             numeric(5, 3),
    layoff_starts       integer,
    layoff_ae           numeric(5, 3),
    switch_starts       integer,
    switch_ae           numeric(5, 3),
    jock_upgrade_starts integer,
    jock_upgrade_ae     numeric(5, 3),
    computed_at         timestamp DEFAULT now()
);
