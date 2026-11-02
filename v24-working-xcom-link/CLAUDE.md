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

Python smoke test:
```
py -3 python\xcom_rl\train_ppo.py --help
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
- time: **3→8** (month, day, hour; hour added 2026-07-23 — old checkpoints resume via partial load, time branch reinits)
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
| BATTLE_ALIEN_OUT_REWARD | 10.0 | battle_only: per alien dead/stunned at episode end |
| BATTLE_SOLDIER_LOST_PENALTY | -10.0 | battle_only: per own soldier dead at episode end |

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

---

## Bridge JSON additions (2026-07-20)

Research project objects now include:
```json
{"name": "STR_ALIEN_ALLOYS", "scientists": 10, "days_remaining": 5}
```
`days_remaining = getCost() - getSpent()`

---

## Distributed Training (2026-07-23)

Two-machine setup: machine A runs the single PPO learner; machine B is an
**env farm** serving battle-only quick-battle episodes over TCP. One policy
learns both strategic and tactical play.

**Machine B** (same repo commit as A; engine built; firewall port open):
```
py -3 -m xcom_rl.env_server --port 7100 --n-envs 3 --battle-only
```
- battle_only envs are visible (GUI click-through of New Battle menu) — one window per env
- machine B runs no learner and must NEVER write checkpoints/registry.json (not multi-writer safe)

**Machine A**:
```
py -3 -m xcom_rl.train_ppo --n-envs 5 --remote 192.168.x.x:7100:3
```
- `--remote HOST:PORT[:COUNT]` is repeatable; rollout stepping is lockstep so keep remote speed_ms low
- hello handshake rejects mismatched N_ACTIONS (different commits)

**battle_only mode** (env.py): forces visible engine; reset clicks main menu →
New Battle (210,100) → Randomize (262,184) → OK (58,184) → Briefing OK (160,173);
terminates on debriefing (auto-OK) with outcome reward; MAX_STEPS_BATTLE=2000.
Engine support: autoStart injects GUI commands at the main menu, and quick-battle
debriefing nulls the save to return to a bridge-active menu (both 2026-07-23).

## BC Pretraining (bc_pretrain.py)

- Uses **inverse-frequency class weights** via `F.cross_entropy(..., weight=class_weights)`
- `alien_bases_found` is an optional obs key (backward-compatible: zeros for old demos)
- Old demos missing `alien_bases_found` get zeros; re-collect for accurate BC
