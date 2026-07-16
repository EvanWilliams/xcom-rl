# XCOM Reinforcement Learning

A PPO agent trained to play the OpenXcom strategic layer (Geoscape) using
only simulated mouse clicks — exactly as a human would play. The agent
intercepts UFOs, manages research, fights ground battles, and works toward
the final Cydonia mission.

A custom C++ bridge exposes game state over a Windows AF-UNIX socket and
accepts simulated SDL mouse/keyboard events, giving the agent full GUI
control without any internal API shortcuts.

---

## Quick Start

Double-click **`launch_training.bat`** in the project root.

It will automatically detect the most recent checkpoint, resume from it,
open 4 visible game windows, start TensorBoard at `http://localhost:6006`,
and begin training.

---

## Repository Layout

```
xcom-reinforcement-learning/
  launch_training.bat           Double-click to start/resume training

  engine/openxcom/              OpenXcom source with RL bridge patches
    src/Bridge/
      BridgeGeoCommands.cpp     Click/key/new_game command handlers
      BridgeState.cpp           Game state → JSON serializer
      SocketServer.cpp          AF-UNIX socket server
    build/bin/openxcom.exe      Engine binary (after build)

  python/
    xcom_bridge/                Low-level Windows socket client (ctypes)
    xcom_gym/
      env.py                    Gymnasium environment (XcomInterceptionEnv)
    xcom_rl/
      train_ppo.py              PPO training loop + resume logic
      checkpoint_registry.py   Checkpoint metadata store + Cydonia-progress ranking
      eval_checkpoint.py        CLI evaluation runner (also called automatically)
      demo_collector.py         Collects expert demonstrations for BC
      bc_pretrain.py            Behaviour-cloning pretraining
      test_gui_playthrough.py   GUI click regression tests
    checkpoints/                Saved model weights (.zip)
      registry.json             Auto-maintained quality index of all .zip checkpoints
    runs/                       TensorBoard logs
    logs/                       Per-env episode logs (NDJSON)
    data/                       Demo datasets (NDJSON)
```

---

## Prerequisites

| Requirement | Version |
|---|---|
| Python | 3.10+ |
| PyTorch | 2.x |
| stable-baselines3 | 2.x |
| gymnasium | 0.29+ |
| numpy | any |
| MSVC Build Tools | 2022 (for engine rebuild only) |
| OpenXcom data files | UFO: Enemy Unknown (original or Steam) |

```powershell
pip install stable-baselines3[extra] gymnasium torch numpy
```

Data files must be placed in `engine\openxcom\build\bin\UFO\`
(`GEODATA\`, `GEOGRAPH\`, `SOUND\`, etc.).

---

## Building the Engine

Only needed after modifying C++ bridge files.

```powershell
$vcvars = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars32.bat"
$build  = "engine\openxcom\build"
cmd /c "call `"$vcvars`" && cd /d `"$build`" && nmake /S"
```

Output: `engine\openxcom\build\bin\openxcom.exe`

---

## Training

All commands run from the `python\` directory.

### Fully visualized (recommended — use the batch file)

```
launch_training.bat
```

Starts 4 visible engine windows, resumes from the latest checkpoint,
opens TensorBoard. Everything automatic.

### From the command line

```powershell
cd python

# Fresh start, 4 visible windows
py -3 -m xcom_rl.train_ppo --n-envs 4 --all-visual --timesteps 1000000

# Resume from latest checkpoint
py -3 -m xcom_rl.train_ppo --n-envs 4 --all-visual --timesteps 1000000 --resume

# Headless (fastest, no windows)
py -3 -m xcom_rl.train_ppo --n-envs 8 --timesteps 1000000

# Single headless env (no SubprocVecEnv, easiest for debugging)
py -3 -m xcom_rl.train_ppo --n-envs 1
```

### CLI Flags

| Flag | Default | Description |
|---|---|---|
| `--n-envs N` | 9 | Parallel engine instances. Use 1 for single-env debugging. |
| `--timesteps N` | 100 000 | Total environment steps to train. |
| `--log-dir PATH` | `./runs/` | TensorBoard log directory. |
| `--all-visual` | off | Open a visible window for every env (no headless). |
| `--resume` | off | Load the newest checkpoint from `checkpoints/` and continue. |
| `--bc-weights PATH` | none | Initialize policy from a behaviour-cloning checkpoint. |
| `--run-name NAME` | auto | TensorBoard run name. |

---

## Action Space — Discrete(212)

The agent controls the game exclusively through simulated mouse clicks and
key presses. No internal game APIs are used. All 212 actions map to concrete
pixel coordinates in the 320×200 native resolution; the full table is in
`python/xcom_gym/env.py:ACTIONS`.

| Group | Indices | Count | Actions |
|---|---|---|---|
| Wait / no-op | 0, 199 | 2 | `wait`, `wait_2` |
| Geoscape sidebar | 1–3 | 3 | intercept, bases, options |
| Time speed | 4–9 | 6 | 5s, 1min, 5min, 30min, 1hr, 1day |
| UFO Detected dialog | 10–12 | 3 | intercept, centre, cancel |
| Mission Detected dialog | 13–14 | 2 | intercept, cancel |
| Intercept screen | 15–16, 186–188, 195 | 6 | craft rows 0–3, cancel, goto_base |
| Confirm Destination | 17–18 | 2 | OK, cancel |
| Dismiss (Escape) | 19 | 1 | close any info dialog |
| Random clicks | 20–21 | 2 | left-click / right-click anywhere |
| Keyboard | 22–23 | 2 | A key, Enter key |
| Menu escape | 24 | 1 | `menu_new_game` — bridge new_game command |
| Basescape navigation | 25–27, 50–57 | 11 | facilities, research, geoscape, info, soldiers, crafts, manufacture, transfer, purchase, sell |
| Research screens | 28–48 | 21 | new project, rows, info (OK, +/−scientists), exit |
| Battlescape | 49 | 1 | pickup |
| Purchase screen | 58–65, 200–205 | 12 | OK, cancel, +/−qty per row |
| Sell screen | 66–73, 206–211 | 12 | OK, cancel, +/−qty per row |
| Soldiers screen | 74–84 | 11 | OK, rows, prev/next/armor |
| Soldier armor | 85–91 | 7 | rows, cancel |
| Crafts screen | 92–96 | 5 | OK, craft rows |
| Craft info | 97–100 | 4 | OK, crew, equip, armor |
| Craft equipment | 101–112 | 12 | OK, inv, item rows, add buttons |
| Dismantle dialog | 113–114 | 2 | OK, cancel |
| Dogfight | 115–119 | 5 | standoff, cautious, standard, aggressive, disengage |
| Manufacture screens | 120–141, 196–198 | 25 | new item, OK, rows, list rows, start, info (eng/unit up/down, sell toggle) |
| Build Facilities list | 142–148 | 7 | facility type rows 0–5, close |
| Place Facility grid | 149–185 | 37 | cancel + all 36 base-grid cells |
| Craft Soldiers screen | 189–194 | 6 | OK, soldier toggle rows |

---

## Observation Space — Dict

| Key | Type | Range | Description |
|---|---|---|---|
| `funds` | Box int32 (1,) | 0 – 100 000 000 | Treasury in USD |
| `score` | Box int32 (1,) | −10 000 – 10 000 | Monthly research score |
| `total_score` | Box int32 (1,) | −100 000 – 100 000 | Strategic score (xcom − alien activity) |
| `month` | Box int32 (1,) | 1 – 12 | In-game month |
| `day` | Box int32 (1,) | 1 – 31 | In-game day |
| `hour` | Box int32 (1,) | 0 – 23 | In-game hour |
| `craft_status` | Discrete(3) | 0/1/2 | 0=ready, 1=in flight, 2=repairing |
| `craft_fuel_pct` | Box int32 (1,) | 0 – 100 | First craft fuel % |
| `craft_damage_pct` | Box int32 (1,) | 0 – 100 | First craft damage % |
| `ufo_detected` | Discrete(2) | 0/1 | 1 when a UFO is on radar |
| `mission_site` | Discrete(2) | 0/1 | 1 when a terror/mission site exists |
| `popup_depth` | Box int32 (1,) | 0 – 10 | Number of open dialogs (0 = geoscape) |
| `screen_type` | Discrete(34) | 0–33 | Which screen/dialog is currently showing |
| `ending` | Discrete(3) | 0/1/2 | 0=ongoing, 1=won, 2=lost |
| `num_soldiers` | Box int32 (1,) | 0 – 50 | Total soldiers across all bases |
| `base_grid` | Box int8 (36,) | 0–2 | Flat 6×6 facility occupancy (0=empty, 1=building, 2=built) |

### `screen_type` values

| Value | Screen | Value | Screen |
|---|---|---|---|
| 0 | geoscape | 17 | soldiers |
| 1 | ufo_detected | 18 | soldier_info |
| 2 | mission_detected | 19 | crafts |
| 3 | intercept | 20 | craft_info |
| 4 | confirm_dest | 21 | craft_soldiers |
| 5 | low_fuel | 22 | craft_equipment |
| 6 | craft_error | 23 | soldier_armor |
| 7 | monthly_report | 24 | dismantle_facility |
| 8 | items_arriving | 25 | dogfight |
| 9 | craft_patrol | 26 | ufopaedia |
| 10 | other | 27 | place_facility |
| 11 | main_menu | 28 | manufacture |
| 12 | basescape | 29 | build_facilities |
| 13 | research_new | 30 | research |
| 14 | research_info | 31 | briefing |
| 15 | purchase | 32 | debriefing |
| 16 | sell | 33 | statistics |

---

## Reward Structure

### Click rewards (three tiers)

| Type | Actions | Reward |
|---|---|---|
| Navigation click | sidebar, list rows, cancel/back buttons | +0.010 |
| Commit click | intercept, confirm, research_start, dispatch, purchase, dismantle, place facility | +0.020 |
| Random click | `random_click`, `random_right_click` | +0.003 |
| Wait / Dismiss / Key | `wait`, `dismiss`, `key_*`, `menu_new_game` | 0.000 |

### Strategic rewards

| Event | Reward |
|---|---|
| Wait (headless; each step = up to 60 game-minutes) | −0.005 |
| UFO first detected on radar | +10.0 |
| Craft dispatched (STR_OUT transition) | +50.0 |
| Mission site dispatched | +75.0 |
| Research project started | +25.0 |
| New facility built | +25.0 |
| Soldier / staff hired | +0.5 per person |
| New base established | +100.0 |
| Alien base destroyed (via score delta) | included in battle score |
| UFO shot down | +40.0 |
| Game won (Cydonia) | +1000.0 |

### Score-based rewards

| Condition | Multiplier |
|---|---|
| Routine score tick (\|delta\| < 10) | delta × 1.0 |
| Post-battle delta (\|delta\| ≥ 10) | delta × 2.0 |
| Research completion (positive score delta) | delta × 3.0 |

All rewards are further scaled by three configurable multipliers:
`reward_scale_nav` (navigation clicks), `reward_scale_milestone` (UFO, craft,
research, build), and `reward_scale_penalty` (abort, load-game, options).
These are set via `REWARD_SCHEDULE` in `train_ppo.py` and annealed over
training timesteps.

### Battle scoring detail (§4.4)

The post-battle multiplier covers all of the following when they appear as a
single-step `total_score` jump of ≥ 10:

| Category | Examples |
|---|---|
| Alien kills | Sectoid Soldier +10 … Ethereal Commander +40 |
| Alien stuns | 2× the kill value for each alien type |
| Civilian killed by X-COM | −50 |
| Civilian killed by aliens | −30 |
| X-COM Soldier killed / MIA | −20 (scales with rank) |
| HWP destroyed | −20 |
| Civilian saved | +30 |
| Alien base destroyed | +500 |

### Research scoring detail (§4.8)

Any positive delta in the `score` field means a research project just
completed. Selected values:

| Points | Projects |
|---|---|
| 5–10 | Ammo clips, laser weapons, stun bomb |
| 15–20 | Armor, Medi-Kit, Motion Scanner, Plasma Pistol |
| 25 | Plasma Rifle, Hyper-Wave Decoder, Psi-Lab, Hovertanks |
| 30 | Heavy Plasma, UFO systems, new craft (Firestorm, Lightning, Avenger) |
| 40 | Blaster Launcher, Elerium-115 |
| 50 | Alien Autopsies, Alien Interrogations (per new UFOpaedia entry) |
| 60 | Alien Origins, Cydonia or Bust, The Martian Solution |

---

## Policy Architecture (XcomExtractor)

```
Dict obs
  ├─ economy  [funds, score, total_score]              → Linear(3,16)  → ReLU → (16)
  ├─ time     [month, day]                             → Linear(2,8)   → ReLU → (8)
  ├─ mission  [craft_status(3), fuel(1), dmg(1),
               ufo_detected(2), mission_site(2),
               screen_type(34), popup_depth(1),
               ending(3), num_soldiers(1)]  (48 total) → Linear(48,32) → ReLU → (32)
  └─ grid     [base_grid(36)]                          → Linear(36,16) → ReLU → (16)
                                  fusion cat(16+8+32+16=72) → Linear(72,64) → ReLU → (64)

PPO heads:  pi  Linear(64,64) → Tanh → Linear(64,64) → Tanh → Linear(64,212)
            vf  Linear(64,64) → Tanh → Linear(64,64) → Tanh → Linear(64,1)
```

PPO hyperparameters: `n_steps=1024`, `batch_size=256`, `n_epochs=4`,
`gamma=0.99`, `lr=3e-4`, `ent_coef` anneals from `0.20` → `0.01` over the run.

---

## Checkpoint Registry & Best-of-Run Selection

Checkpoints are saved to `python\checkpoints\` every 10 000 timesteps. A
persistent registry (`checkpoints/registry.json`) tracks every checkpoint's
Cydonia-progress score — a composite metric weighted toward the behaviours
needed to win: surviving to late months, completing research, launching craft,
and ultimately winning at Cydonia.

### How It Works

**At the end of every training run** the trainer automatically:
1. Saves the final model (`xcom_ppo_{run_name}_final.zip`)
2. Registers it in the registry (metadata)
3. Launches 20 evaluation episodes across up to 4 headless envs
4. Records win rate, mean month reached, research/combat milestone rates, and
   composite score in `registry.json`

**At the start of any resumed run** with `--select-by composite` (or `reward`):
1. Scans `checkpoints/` for any .zip not yet in the registry
2. Runs a quick 4-episode eval on each unevaluated checkpoint
3. Selects the checkpoint with the highest composite score as the warm-start
   point — not just the most recently modified file

### Commands

```powershell
cd python

# --- Normal use: everything is automatic ---

# Start a new run (evaluates on finish, registers checkpoint)
py -3 -m xcom_rl.train_ppo --run-name run1 --timesteps 500000

# Resume from the BEST checkpoint (not just newest)
py -3 -m xcom_rl.train_ppo --resume --select-by composite

# Resume from checkpoint with highest raw reward instead
py -3 -m xcom_rl.train_ppo --resume --select-by reward

# Resume from newest checkpoint (original behaviour, default)
py -3 -m xcom_rl.train_ppo --resume


# --- Manual evaluation ---

# Evaluate a specific checkpoint and update the registry
py -3 -m xcom_rl.eval_checkpoint --checkpoint checkpoints/xcom_ppo_run1_50000_steps.zip --n-episodes 20

# Evaluate whichever checkpoint currently ranks best
py -3 -m xcom_rl.eval_checkpoint --checkpoint best --n-episodes 40

# Print the current rankings without running new episodes
py -3 -m xcom_rl.eval_checkpoint --list


# --- Demo extraction from the best checkpoint (for BC pretraining) ---

# Save high-quality episodes from the best checkpoint as training demos
py -3 -m xcom_rl.eval_checkpoint --checkpoint best --save-demos --demo-threshold 150

# Pretrain a fresh policy on those demos, then warm-start PPO from it
py -3 -m xcom_rl.bc_pretrain --epochs 20
py -3 -m xcom_rl.train_ppo --bc-weights ./checkpoints/bc_init
```

### Composite Score

Each checkpoint is scored 0–1 by a weighted sum oriented toward Cydonia:

| Component | Weight | What It Measures |
|---|---|---|
| Win rate (`ending==1`) | 0.30 | Actually reaching Cydonia |
| Survival rate (1 − loss rate) | 0.15 | Avoiding game-over |
| Mean month reached / 12 | 0.15 | Progress depth — later months = more research/equipment |
| Research milestone rate | 0.15 | Research is the gating requirement for Cydonia |
| Craft launch rate | 0.15 | Combat proficiency |
| Strategic score (total_score / 2000, capped) | 0.10 | Long-term strategic health |

A checkpoint that regularly reaches month 6 with research and craft launches
will outscore a later checkpoint that has drifted into a loop, even if the
later one has more training timesteps.

### CLI Flags (new)

| Flag | Default | Description |
|---|---|---|
| `--registry PATH` | `checkpoints/registry.json` | Path to the checkpoint quality index. Pass `""` to disable. |
| `--select-by` | `latest` | When `--resume`: `latest` (original behaviour), `composite`, or `reward`. |

If the saved action space doesn't match the current one (e.g. after adding
new actions), the trainer automatically falls back to partial weight loading —
all compatible layers are restored, only the action head is re-initialized.

---

## Behaviour Cloning

Collect expert demonstrations then pretrain the policy:

```powershell
cd python

# Record 30 episodes of expert play
py -3 -m xcom_rl.demo_collector --episodes 30 --steps 5000

# Pretrain via supervised learning
py -3 -m xcom_rl.bc_pretrain --epochs 20

# Run PPO initialized from BC weights
py -3 -m xcom_rl.train_ppo --n-envs 4 --all-visual --bc-weights ./checkpoints/bc_init
```

---

## Monitoring

### Live training watch window

```powershell
cd python
py -3 -m xcom_rl.watch
```

Displays a live terminal dashboard updating every second: per-env game state
(month, score, screen, reward), throughput (steps/sec), reward sparklines by
category, milestone hit-rates, base-grid panel, and policy health.

### TensorBoard

```powershell
tensorboard --logdir python\runs
```

Open `http://localhost:6006`. Key metrics:

| Metric | What to watch for |
|---|---|
| `rollout/ep_rew_mean` | Rising = agent learning. Should climb past +50 (craft launched) early. |
| `rollout/ep_len_mean` | Too short = dying/truncating before events; too long = not acting. |
| `train/entropy_loss` | Should stay negative. Collapse to 0 = policy stuck. |
| `train/explained_variance` | Should approach 1.0 as value function improves. |
| `train/ent_coef` | Anneals 0.20 → 0.01; watch for premature collapse. |
| `train/rscale_nav` / `rscale_milestone` / `rscale_penalty` | Active reward multipliers. |

### Socket naming

Each env process binds to `openxcom_bridge_{rank}.sock` in `%TEMP%`.
Training envs use ranks 0–(N−1). Eval envs (from `eval_checkpoint.py`)
use ranks 100+ so their log files never collide with training env logs.

---

## GUI Tests

Verify all dialogs are navigable by click before a training run:

```powershell
py -3 -m xcom_rl.test_gui_playthrough
```

Tests: time-speed buttons, UFO intercept sequence, info dialog dismiss,
mission detected cancel.

---

## Troubleshooting

**`RuntimeError: no initial state received from engine`**
The engine started but didn't reach the geoscape within the timeout. Check:
- No leftover `openxcom.exe` processes (`taskkill /IM openxcom.exe /F`)
- `engine\openxcom\build\bin\UFO\` contains the game data files
- `options.cfg` has `playIntro: false`

**Engine windows open but freeze / go black**
SDL events are not being processed. Make sure each env has a unique socket
name and no other instance is bound to the same socket.

**Policy collapses to always clicking one button**
Entropy has fallen to zero. Increase `ENT_COEF_START` in `train_ppo.py`
(currently 0.20 → 0.01 schedule) or reduce the learning rate.
Check `train/entropy_loss` and `train/ent_coef` in TensorBoard.

**`LNK1168: cannot open openxcom.exe`**
The linker can't overwrite the binary because it is still running. Kill all
processes before rebuilding: `taskkill /IM openxcom.exe /F`

**Partial weight load on resume**
Expected when the action space has grown since the checkpoint was saved.
The trainer prints which layers were skipped. Training continues normally
with the action head re-initialized randomly.
