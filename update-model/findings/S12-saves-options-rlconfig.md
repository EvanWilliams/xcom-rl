# Findings — S12 Saves, options & active mod set (+ the RL training config)

This is where the real goal-blockers are. The game is winnable (S02–S11); the open question is
whether the **agent is set up to reach Cydonia _on Superhuman_**. Three findings, each a genuine
"something may be misconfigured such that the goal isn't reachable."

## FINDING A — new_game never sets difficulty ⇒ fresh episodes are NOT Superhuman  [HIGH]
- `env.py` `_new_game()` sends the bridge `new_game` command (lines ~1998–2027); it passes an
  optional `save` field but **no difficulty**. Grep of `engine/bridge/` for `difficulty` →
  **zero matches**. So bridge-started games take the engine default difficulty (Beginner), not
  Superhuman (4).
- The only way an episode is on Superhuman is to **load a `.sav` created at difficulty 4**. The
  example save in `docs/engine-investigation.md` shows `difficulty: 2`.
- **Implication:** unless every save in the pool is difficulty 4, the agent is training/winning on
  the *wrong difficulty*, so "beat Cydonia **on Superhuman**" is not actually being optimized.
- **Candidate change (minimal):** have `new_game` accept and set `difficulty: 4` (bridge
  `cmdNewGame` + `env.py`), and/or guarantee the save pool is Superhuman-only. Assumption-test:
  if the pool is already all-Superhuman saves, the new_game change is unnecessary → prefer the
  pool guarantee (fewer code edits). **Cannot confirm the pool from here (it lives outside the
  repo at `~/Documents/OpenXcom/rl_saves`).**

## FINDING B — episode horizon can't reach Cydonia from scratch ⇒ curriculum is load-bearing  [HIGH]
- `MAX_STEPS = 75_000` one-minute steps ≈ **~52 in-game days** per episode (code comment itself
  notes "31 days ≈ 44 640").
- Reaching Cydonia in X-COM takes **multiple in-game months** (research Alien Origins → Martian
  Solution → Cydonia or Bust, build the Avenger, fly to Mars). ~52 days from a fresh start is not
  enough — the agent would be truncated long before the final mission exists.
- Therefore success depends entirely on the **save-curriculum** (`_pick_save` picks a random `.sav`
  each reset). To ever *win*, the pool must include **late-game Superhuman saves close to the final
  mission** (ideally some already at `STR_MARS_CYDONIA_LANDING`).
- **Candidate change:** (1) build/curate a Superhuman save pool staged across the campaign incl.
  pre-Cydonia saves; (2) consider raising `MAX_STEPS` only if starting late (raising it globally
  balloons episode length with no benefit from early saves). Assumption-test: raising MAX_STEPS
  alone does **not** help if the pool lacks late saves → the pool is the real lever; do that first,
  skip the MAX_STEPS bump. (Reduction: 2 candidate edits → 1.)

## FINDING C — training budget is smoke-test-sized  [MED/HIGH]
- Source default `TOTAL_TIMESTEPS = 100_000`; the captured `train_stdout.txt` shows a real run at
  `total_timesteps=357120`, `n_envs=8`, "all 8 envs visible (speed_ms=10)".
- 100k–357k steps is orders of magnitude too small to learn a full strategic campaign to a sparse
  `CYDONIA_BONUS=1000` win, even with the shaped Rank-1 goal rewards. And "all envs visible" at
  10 ms is a watch/demo configuration, not a throughput training run.
- **Candidate change:** run long **headless** training (n_envs = headless, no spectator), with a
  timestep budget 1–2 orders larger, from the Superhuman late-game curriculum. This is an
  operational change, not a code edit.

## Options / mod set (the lighter part of S12)
- Active bundled mods (PSX_Static_Cydonia_Map, XcomUtil_*, UFOextender_*) are stock OpenXCOM extras;
  none block the goal. Reward/obs/action config is documented thoroughly in `CLAUDE.md` and is
  internally consistent (checkpoint-compatible, actor-faction shot gating verified).

## Verdict
**v30 changes needed:** the goal-blockers are A (difficulty) and B (curriculum); C is operational.
Minimal code change = make difficulty configurable/Superhuman + curate a Superhuman staged save
pool. Everything else reduces away.
