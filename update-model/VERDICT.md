# Final verdict — can v30 beat Cydonia on Superhuman as-is?

**Short answer: No, not as currently configured — but the blockers are in the RL setup, not the game.**

## The one-line answer
The *game* (stock OpenXCOM xcom1) is winnable on Superhuman by construction, so no ruleset or engine
changes are needed. The *agent* is not currently set up to achieve the goal: episodes don't run on
Superhuman, a single episode is too short to reach Cydonia from scratch, and the training budget is
smoke-test sized. Fix the four items in `proposed-changes.md` and the goal becomes reachable.

## How the 327 resolved
- **324 cells (S01–S11 across all 27 lenses): 0 changes.** Game config is stock and already
  completable on Superhuman; engine needs no edit.
- **The surviving work is 4 items**, all in S12 / RL-config + 1 diagnostic in S01:
  1. Set difficulty = Superhuman (new_game or pool) — *the Superhuman half of the goal is currently unwired.*
  2. Curate a staged Superhuman save pool incl. pre-Cydonia saves — *makes a win physically reachable in one episode.*
  3. Run long headless training instead of the all-visible smoke config.
  4. Triage the crash-rank logs.

## Biggest "thing you may have missed"
`new_game` sets **no difficulty**, and the bridge has zero difficulty handling — so unless the save
pool is all difficulty-4, **nothing in the pipeline is actually on Superhuman.** That is the highest-
leverage misconfiguration found.

## What this audit could NOT verify (needs you or a live run)
- Contents of `~/Documents/OpenXcom/rl_saves` (difficulty + campaign stage) — it lives outside the repo.
- Whether any run has ever hit `ending == END_WIN` on difficulty 4.
- Engine headless→ending round-trip and crash-log root cause (can't compile/run from here).

## The single decisive test (M3)
Load a **Superhuman** save staged at `STR_MARS_CYDONIA_LANDING`, run the current policy, and confirm
`ending == 1` with `CYDONIA_BONUS` firing. That one observation would validate the entire win path.

---
*Produced by the `update-model` skill. Source backups were read-only throughout; all outputs are new files.*
