# XCOM RL — Project Notes for Claude

## Quality Gates (required before any task is "done")

Run `/done` (`/.claude/commands/done.md`) after every task to enforce:
1. **No build failures** — rebuild engine for C++ changes; confirm Python imports clean
2. **Tests pass** — run smoke tests or write new ones; zero failures required
3. **Documentation updated** — update this file and inline comments for any changed rewards, obs fields, actions, or config constants

---

## Build

C++ engine (MSVC x86 / NMake):
```
# From a VS 2022 x86 dev shell:
cd engine\openxcom\build
nmake openxcom
```
Output: `engine\openxcom\build\bin\openxcom.exe`

Python smoke test (run as a module from `python/` so `import xcom_rl` resolves):
```
cd python
py -3 -m xcom_rl.train_ppo --help
```

---

## Architecture

### Action space
- **7,429 total actions**: 260 named buttons + 7,169 grid cells (every 3 px, 320×200 screen)
- Named actions: indices 0–259; grid actions: 260–7428
- Grid cells cover entire screen; max distance from any click ≤ 1.41 px

### Observation space (17 fields)
| Field | Type | Range | Notes |
|-------|------|-------|-------|
| funds | Box(1,) int32 | 0–100M | |
| score | Box(1,) int32 | -10k–10k | monthly score |
| total_score | Box(1,) int32 | -100k–100k | |
| month | Box(1,) int32 | 1–12 | |
| day | Box(1,) int32 | 1–31 | |
| hour | Box(1,) int32 | 0–23 | |
| craft_status | Discrete(3) | 0=ready,1=out,2=repair | first craft |
| craft_fuel_pct | Box(1,) int32 | 0–100 | |
| craft_damage_pct | Box(1,) int32 | 0–100 | |
| ufo_detected | Discrete(2) | 0/1 | any UFO detected |
| mission_site | Discrete(2) | 0/1 | terror site active |
| popup_depth | Box(1,) int32 | 0–10 | |
| screen_type | Discrete(N_SCREEN_TYPES=36) | | current screen |
| ending | Discrete(3) | 0=none,1=win,2=lose | |
| num_soldiers | Box(1,) int32 | 0–50 | |
| alien_bases_found | Box(1,) int32 | 0–20 | discovered alien bases |
| base_grid | Box(36,) int8 | 0–2 | 6×6 facility grid |

### XcomExtractor branches (train_ppo.py)
- economy: 3→16
- time: **2→8** (month, day)
- mission: **51→32** (includes alien_bases_found; was 50 before 2026-07-20)
- grid: 36→16
- fusion: 72→64

---

## Key Constants (env.py)

### Reward constants
| Name | Value | Notes |
|------|-------|-------|
| CLICK_REWARD | 0.01 | named nav click |
| COMMIT_CLICK_REWARD | 0.02 | strategic action |
| WAIT_PENALTY | -0.005 | |
| ABORT_PENALTY | -200.0 | applied **immediately** (same step, same episode) |
| OPTIONS_MENU_PENALTY | -100.0 | only for named/special actions (not grid clicks); capped at once per step |
| MENU_DWELL_PENALTY | -1.0 | per step spent at main_menu; scaled by reward_scale_penalty |
| MISSION_DISPATCH_REWARD | 75.0 | only when craft status == STR_OUT |
| RESEARCH_MILESTONE_REWARDS | see below | once per episode per milestone |
| UFO_SHOOT_DOWN_REWARD | 30.0 | |
| CYDONIA_BONUS | 1000.0 | win condition |

### Battlescape combat rewards (env.py `_process_battle_message`)
| Name | Value | Notes |
|------|-------|-------|
| SHOT_REWARD | 0.5 | per weapon fired **by the player**; gated on `actor_faction==0` (see below) |
| ALIEN_HIT_REWARD | 0.5 | per HP of damage dealt to aliens; **not** gated (covers reaction fire) |
| ALIEN_SPOT_REWARD | 3.0 | per alien spotted (LOS) for the first time this battle |
| EXPLORE_REWARD_PER_TILE | 0.1 | per newly uncovered fog-of-war tile |
| PICKUP_REWARD | 2.0 | per ground item picked up |

Engine emits `{"type":"battle","event":"shoot"}` on `BA_SNAPSHOT/AUTOSHOT/AIMEDSHOT/LAUNCH`
(`BattlescapeGame.cpp:1047`), broadcast for **every** unit action — aliens included — so
an ungated `SHOT_REWARD` let the agent farm reward by passing its turn while aliens fire.

**Fix (actor-faction attribution, 2026-08-17):** `broadcastBattle`/`serializeBattle`
(`BridgeBattle.cpp`, `BridgeState.h`) now take an `actorFaction` arg and emit
`"actor_faction"` (0=player, 1=hostile, 2=neutral; `null` for `battle_start`/`turn_start`).
The shoot call site passes `action.actor->getFaction()`. Python (`_process_battle_message`)
gates `SHOT_REWARD` on `actor_faction==0`, so it correctly **rewards player reaction fire
on the alien turn** and **excludes alien reaction fire on the player turn** — cases the old
`side==0` proxy got wrong. If `actor_faction` is absent (older binary) it falls back to
`side==0`. `ALIEN_HIT_REWARD` stays ungated (damaging aliens is always good).
Verified: player shot→+0.5, alien shot→0.0, player reaction hit (alien turn)→+0.5,
alien reaction fire (player turn)→0.0, plus the `side` fallback. **Requires an engine
rebuild** (done); the `actor_faction` string is present in the current `openxcom.exe`.

### Research milestones (RESEARCH_MILESTONE_REWARDS)
```python
"STR_ALIEN_ALLOYS":          50.0
"STR_ELERIUM_115":           50.0
"STR_ALIEN_ORIGINS":         75.0
"STR_THE_MARTIAN_SOLUTION": 100.0
"STR_CYDONIA_OR_BUST":      200.0
```

### _COMMIT_ACTIONS includes
- 232 = battle_end_turn (added 2026-07-20)
- Full list: ufo_intercept, mission_intercept, confirm_ok, research_info_ok, purchase_ok, craft_info_ok, craft_equip_ok, dismantle_ok, manufacture_ok, mfg_start, mfg_info_ok, craft_soldiers_ok, craft_weapons_row1–4, build_row0–5, place_pos0–35

---

## PPO Hyperparameters (train_ppo.py)

| Param | Value | Notes |
|-------|-------|-------|
| ENT_COEF_START | 0.20 | |
| ENT_COEF_END | 0.05 | was 0.01; raised for long Cydonia path |
| gae_lambda | 0.98 | longer horizon than default 0.95 |
| N_STEPS | 1024 | |
| BATCH_SIZE | 256 | |
| GAMMA | 0.99 | |

---

## Episode starts — non-deterministic save loading (added 2026-08-02)

Every `reset()` (and the initial engine launch) picks a **uniformly-random `*.sav`**
from a pool folder so episodes start from varied states. Handled entirely in
`env.py::_pick_save()`:

- Pool folder = `autoload_dir`, default `_DEFAULT_SAVE_DIR` = `~/Documents/OpenXcom/rl_saves`.
- Re-globbed **every reset** → drop new saves in mid-run, no restart needed.
- Uses the `random` module (not the seeded gym RNG) → stays non-deterministic even
  under a fixed seed.
- Priority: random pick from `autoload_dir` > single `autoload_save` > fresh game.
- **Empty/missing folder → falls back to a fresh game** (warns once), so it's a safe
  default before any saves exist.
- Engine side: the initial load uses `XCOM_BRIDGE_AUTOLOAD`; each reset loads in-process
  via the `new_game` `save` field (no engine restart). See `BridgeGeoCommands.cpp::cmdNewGame`.
- CLI: `--save-dir <folder>` (default as above); `--save-dir ""` disables the pool and
  uses `--save`/fresh games. Applies to all env users (train, eval, watch, demos).

## auto_confirm_base_name screen handlers

Screens handled automatically (agent never sees them):
- `base_name` — types "a" + Enter
- `place_lift` — clicks top-left cell
- `basescape` — dismiss once if new-base setup flow, else break
- `ufopaedia` — dismiss
- `statistics` — click OK at (160, 186)
- `load_game` — dismiss + LOAD_PENALTY
- `pause_menu` — dismiss + OPTIONS_MENU_PENALTY
- `facility_error` — dismiss error, wait, then dismiss PlaceFacilityState if still shown
- `research_complete` — click OK at (104, 154) [added 2026-07-20]
- `error_message` — dismiss [added 2026-07-20]
- `debriefing` — click OK at (36, 186)
- `briefing` — dismiss (Escape = OK on BriefingState) [added 2026-07-29]

---

## Bridge JSON additions (2026-07-20)

Research project objects now include:
```json
{"name": "STR_ALIEN_ALLOYS", "scientists": 10, "days_remaining": 5}
```
`days_remaining = getCost() - getSpent()`

---

## BC Pretraining (bc_pretrain.py)

- Uses **inverse-frequency class weights** via `F.cross_entropy(..., weight=class_weights)`
- `alien_bases_found` is an optional obs key (backward-compatible: zeros for old demos)
- Old demos missing `alien_bases_found` get zeros; re-collect for accurate BC

---

## Bridge input dispatch — battlescape clicks fix (2026-08-05)

Bridge `click`/`dismiss`/`key` commands are turned into real UI events (SDL events
routed to `State::handle`) so on-screen buttons respond. Two gaps made the
battlescape tactical UI — notably the **wide "shoot" action popup** that opens when
you click a weapon in a soldier's hand — ignore all clicks. Both are now fixed:

1. **Headless (`Game::run`, engine).** The headless main-loop branch previously
   `continue`d before the visible-mode `SDL_PollEvent` loop, so bridge-injected SDL
   events (pushed by `cmdClick`/`cmdDismiss`/`cmdKey`) were **never dispatched to any
   State** — every click in headless was silently dropped. It now drains the SDL
   queue and calls `_states.back()->handle()` each iteration (guarded by `_init` when
   the stack changes). Fixes clicks on the geoscape, popups, **and** the battlescape
   for the 4 headless training envs.

2. **Visible battlescape depth-1 (`BridgeCommands::dispatchCommands`, engine).** At
   the tactical screen (no popup), `dispatchCommands` drained the socket but only
   understood *semantic* battle commands (`move`/`shoot`/`end_turn`/…), so a plain
   `click` was dropped and the weapon action popup never opened. It now forwards
   click/mousemove/dismiss/key to `injectBridgeInput()` (new helper in
   `BridgeGeoCommands`), the same SDL-injection path geoscape/popups use. Depth-≥2
   popup clicks (the shoot buttons themselves) already worked via
   `dispatchPopupCommands`.

Flow that now works (validated end-to-end via a headless New Battle → 14 `shoot`
events): click hand weapon (e.g. grid cell at ~296,172) → `ActionMenuState` opens
(popup) → click a wide action row (e.g. ~160,140) → shot mode selected, menu closes,
targeting engaged → click a target tile → weapon fires. The env still suppresses
*named* battle clicks (see `env.py::_execute_action`); the agent reaches these buttons
via grid clicks, which the above fixes deliver.

---

## Human demo collector — concurrent-training safety (2026-08-17)

`human_demo_collector.py` can now run **while a training run is active**. Two
collisions used to break it:

1. **Wrong window (`_find_openxcom_hwnd`).** It matched by title prefix
   `"OpenXcom"` and grabbed whichever window enumerated first. With training
   windows present (titled `OpenXcom #N  reward=...` by `window_tiler.py`), it
   hijacked one of those — renamed it "XCOM Human Demo", raised it, warped the
   cursor there — while `env.step` observed the collector's *actual* engine, so
   recorded (obs, action) pairs were mismatched. Now it binds to the launched
   engine's **child PID** (`env._proc.pid`) via `GetWindowThreadProcessId`
   (`_set_target_pid` / `_hwnd_for_pid`), with the title match only as a fallback.

2. **Rank collision → PermissionError crash.** `socket_name` parsed to
   `rank=0` (`int("human")` → ValueError → 0 in `env.py`), so it shared
   `live_env_0.json.tmp` / `engine_0.pid` / `env_0.jsonl` with training env 0.
   `env.py`'s startup `os.remove(...tmp)` only catches `FileNotFoundError`, so
   the file lock raised `PermissionError`. Fixed by using
   `socket_name="openxcom_bridge_900.sock"` → rank 900 (distinct from training
   0..N and eval 100+).

`bc_pretrain.py` had the same latent rank-0 crash: `_make_policy` builds an
`XcomInterceptionEnv` (never `reset()`, so no engine — but `__init__` still
touches `live_env_0.json.tmp`) with `socket_name="openxcom_bridge_bc.sock"` →
rank 0. Now uses `openxcom_bridge_901.sock` → rank 901. Reserved ranks:
**900 = human demo collector, 901 = BC pretrain.**

### Larger, moveable demo window (2026-08-19)

The collector now launches a **3x-scaled 960×600 window** that is freely
draggable and user-resizable:

- New env ctor param `display_size: Optional[Tuple[int,int]]` (visible mode only).
  When set, `_launch_engine` appends `-displayWidth W -displayHeight H
  -allowResize true` — pre-existing OpenXcom options (generic `-KEY VALUE`
  parsing, like `-playIntro false`), so **no engine rebuild needed**. The engine
  scales its 320×200 image to fill the window and marks it `SDL_RESIZABLE`;
  Game.cpp handles the resize event and rescales.
- Collector passes `display_size=(_GAME_W*_WINDOW_SCALE, _GAME_H*_WINDOW_SCALE)`
  where `_WINDOW_SCALE = 3`.
- `_raise_game_window` no longer pins the window `HWND_TOPMOST` (which felt
  stuck) — it raises via TOPMOST then drops to `HWND_NOTOPMOST`, popping it above
  the terminal once without locking it on top.
- `_game_bounds()` reads the live client rect every poll, so click→game-pixel
  mapping stays correct after any move/resize.

### Battlescape demos record grid-only (2026-08-19)

The collector now records **grid cells only while `env._in_battlescape`** is
True, matching the env's real replay behaviour. `env._execute_action` suppresses
every *named* (non-`grid_`) click during battle (`if self._in_battlescape and
not name.startswith("grid_"): return self._last_state`), so the agent reaches
hand weapons, the fire-mode/Throw action menu, Prime Grenade, and inventory
grab/drop **only via grid cells**.

- Before: `_match_action` used the full pool, so a human clicking dead-centre on
  a named battle button (`battle_left_hand` @24,172, `battle_action_menu_1`
  @160,140, `battle_inventory`, `battle_end_turn`, …) recorded that *named*
  action — which the env turns into a no-op on replay, teaching BC dead actions.
- After: `_poll_clicks` takes an `in_battle` callable (`lambda: env._in_battlescape`)
  and passes `grid_only=True` to `_match_action`, which then matches against the
  new `_GRID_CLICKABLE` pool (the 7,169 grid cells, all left-click). Every battle
  left-click is saved as a replayable grid cell.
- **Right-clicks in the battlescape are dropped** (grid cells are all left-click;
  the action space has no right-click grid cell). All requested capabilities —
  firing any weapon in Snap/Aimed/Auto, throwing/priming grenades, picking items
  up into inventory — are left-click flows, so they record faithfully. OpenXcom
  inventory is click-grab-click-drop (not drag), so both halves are captured.
- Console feedback: battle clicks print `[BATTLE → grid]`; the `[ICON BAR]` hint
  is suppressed in battle (low clicks on hand slots are expected there).
