# Meta-verification sweeps (parts 325–327)

Each sweeps across all 12 subsystems for anything missed/misconfigured over the whole project that
would keep Cydonia-on-Superhuman from being reached or won.

## M1 (325) — Planning-gap sweep: requirements never planned for
The original 12-subsystem decomposition is **game-config-centric** (rulesets, engine). But this is
an **RL project** — the dominant determinant of "can the agent beat Cydonia on Superhuman" is the
*learning setup*, which the subsystem list under-weights. Specifically unplanned-for in the
subsystem grid:
- **Difficulty selection** was assumed to live in a ruleset; it actually lives in the save /
  new_game path, and is currently unset → the *Superhuman* half of the goal was never wired up
  (Finding A). **This is the single biggest "thing you may have missed."**
- **Episode-horizon vs. campaign-length** mismatch (Finding B) — no subsystem owned "can one episode
  physically reach the final mission."
- **Curriculum / save-pool design** — the pool lives outside the repo (`~/Documents/OpenXcom/rl_saves`)
  and is effectively an un-versioned, un-audited dependency that the goal depends on.
Output: the goal spec needs an explicit "Superhuman + staged curriculum" requirement. (Non-empty →
real planning gap found.)

## M2 (326) — Implementation-gap sweep: planned but mis/!configured in v30
- `new_game` sends no difficulty (Finding A) — planned-for (Superhuman) but not implemented.
- `TOTAL_TIMESTEPS` default (100k) and the captured "all-visible" run config (Finding C) — training
  is configured for smoke-testing, not for reaching the goal.
- Crash-rank logs (S01) — multi-env runtime instability, un-triaged.
- Everything in S02–S11 is correctly configured (stock) — **no implementation gaps there.**
Output = the minimal proposed-change set (see `../proposed-changes.md`).

## M3 (327) — Testing-gap sweep: claims never verified in play
- No evidence in-repo of a completed episode that actually **reaches `ending == END_WIN` on
  difficulty 4**. The `CYDONIA_BONUS` path is coded but unproven end-to-end.
- Engine headless→ending round-trip asserted from "working" labels + CLAUDE.md, not re-verified here.
- Save pool contents (difficulty, campaign stage) unverifiable from the repo.
Output: the one test that would settle the whole audit = load a Superhuman save staged at
`STR_MARS_CYDONIA_LANDING`, run the current policy, confirm `ending==1` and `CYDONIA_BONUS` fires.
