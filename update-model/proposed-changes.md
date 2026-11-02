# Proposed changes to v30 — minimal set

> All proposed as NEW work; nothing here edits the read-only `v*` source. v30 is the change target.
> Ordered by how directly they gate "beat Cydonia on Superhuman." The audit started from 327 cells
> and **324 of them produced no change** (the game is stock and winnable); only the items below survive.

| # | Change | Subsystem | Type | Why it gates the goal |
|---|--------|-----------|------|------------------------|
| 1 | Make episode difficulty = **Superhuman (4)**: either `new_game` sets `difficulty:4` (bridge `cmdNewGame` + `env.py`), or guarantee the save pool is difficulty-4-only | S12 / A | code **or** data | Without it the agent is not on Superhuman at all — the goal's difficulty is unmet |
| 2 | Curate a **staged Superhuman save pool** in `~/Documents/OpenXcom/rl_saves`, including **pre-Cydonia / `STR_MARS_CYDONIA_LANDING`** saves | S12 / B | data/curriculum | A 75k-step (~52-day) episode can't reach Cydonia from scratch; late saves make a win physically reachable |
| 3 | Run **long headless training** (headless envs, large timestep budget) instead of the all-visible smoke config | S12 / C | operational | 100k–357k visible steps can't learn the campaign; this is how the policy actually gets good enough |
| 4 | **Triage the crash-rank logs** (`python/logs/crash_rank*.txt`, engine `stderr.txt`) and fix any multi-env instability | S01 | diagnostic→fix | Crashing envs can't accumulate the long-horizon experience needed |

## Reductions already applied (assumptions tested, updates dropped)
- **324 ruleset/engine "cells" → 0 changes:** game is stock X-COM, winnable on Superhuman by
  construction (S02–S11 consolidated; S01 engine needs no edit).
- **Finding B: 2 candidate edits → 1.** Raising `MAX_STEPS` was dropped — it doesn't help unless the
  pool has late saves, so only the pool change (#2) survives.
- **Finding A: prefer the data fix over the code fix** where possible — if the pool is already all
  difficulty-4, #1's code change is unnecessary. (Needs the user to confirm pool contents.)

## Open items the repo can't answer (need the user / a live run)
- Actual difficulty + campaign stage of the saves in `~/Documents/OpenXcom/rl_saves`.
- Whether any training run has ever produced `ending==END_WIN` on difficulty 4.
