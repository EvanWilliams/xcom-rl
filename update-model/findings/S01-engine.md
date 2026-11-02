# Findings — S01 Engine build & runtime

**What v30 is:** stock OpenXCOM (git `1.0.630130c5c`) + a custom **bridge** (`engine/bridge/`,
socket server) and headless/bridge hooks added into the engine so the Python RL stack can drive it.
Build: MSVC x86 / NMake (`cd engine/openxcom/build && nmake openxcom` → `build/bin/openxcom.exe`).

## Planning lenses (K1–K9)
- The goal requires the engine to (a) build, (b) run headless under bridge control, (c) faithfully
  execute the stock campaign incl. the two-stage Cydonia mission, and (d) emit the `ending` signal
  (win/lose) the agent trains on.
- Minimal requirement: a working `openxcom.exe` with the bridge `actor_faction`/headless click
  fixes compiled in (per `CLAUDE.md` these require rebuilds and are marked "done").

## Implementation lenses (K10–K18) — source vs v30
- `docs/engine-investigation.md` documents that stock engine had **no headless mode** at clone time
  (`SDL_SetVideoMode` unconditional). The project added bridge-driven headless + input injection
  (`BridgeGeoCommands`, `BridgeBattle`, `SocketServer.cpp`) and cross-platform socket shims.
- **Crash logs present:** `python/logs/crash_rank{0,1,2,3,8,9,10,14,16,17}.txt` and
  `engine/openxcom/build/bin/stderr.txt`. Their existence indicates engine/bridge instability during
  multi-env runs at some point. **Not yet root-caused in this audit** (would require reading them).
- Candidate change: none to *ruleset*; the engine change surface is the bridge. No evidence v30's
  binary is broken for single-env play (other versions are labelled "working-xcom").

## Testing lenses (K19–K27)
- Correctness proof would be: headless launch → bridge `new_game`/load → run to a known `ending`.
  I cannot compile or run the engine from here, so this is **verified only by inference** from the
  "working-xcom" version label and the CLAUDE.md "rebuild done" notes.
- Cheapest real check the user can run: `nmake openxcom` then a Python smoke test
  (`py -3 -m xcom_rl.train_ppo --help`) and a 1-episode headless run that reaches any `ending`.

## Verdict
**v30 change needed: NONE to ship the goal**, but **two unverified risks**:
1. The crash-rank logs suggest multi-env instability — if envs crash mid-episode, training can't
   accumulate the experience needed to reach Cydonia. **Action: triage the crash logs** (cheap,
   read-only) before any long run.
2. Headless end-to-end (reaching an `ending`) is asserted, not re-verified here.

## Reduction
No engine edits proposed. The only work item is *diagnostic* (read crash logs), not a change.
