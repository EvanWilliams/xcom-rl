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
      demo_collector.py         Collects expert demonstrations for BC
      bc_pretrain.py            Behaviour-cloning pretraining
      test_gui_playthrough.py   GUI click regression tests
    checkpoints/                Saved model weights (.zip)
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

## Action Space — Discrete(23)

The agent controls the game exclusively through simulated mouse clicks and
key presses. No internal game APIs are used.

| # | Name | Input | Screen |
|---|---|---|---|
| 0 | wait | — | Geoscape (headless: advances one 5s tick) |
| 1 | geo_intercept | click (288, 5) | Geoscape sidebar |
| 2 | geo_bases | click (288, 17) | Geoscape sidebar |
| 3 | geo_options | click (288, 53) | Geoscape sidebar |
| 4 | speed_5sec | click (272, 118) | Time speed |
| 5 | speed_1min | click (304, 118) | Time speed |
| 6 | speed_5min | click (272, 132) | Time speed |
| 7 | speed_30min | click (304, 132) | Time speed |
| 8 | speed_1hour | click (272, 146) | Time speed |
| 9 | speed_1day | click (304, 146) | Time speed |
| 10 | ufo_intercept | click (128, 124) | UFO Detected dialog |
| 11 | ufo_centre | click (128, 140) | UFO Detected dialog |
| 12 | ufo_cancel | click (128, 156) | UFO Detected dialog |
| 13 | mission_intercept | click (128, 138) | Mission Detected dialog |
| 14 | mission_cancel | click (128, 178) | Mission Detected dialog |
| 15 | intercept_craft1 | click (152, 82) | Intercept screen |
| 16 | intercept_cancel | click (244, 152) | Intercept screen |
| 17 | confirm_ok | click (93, 110) | Confirm Destination |
| 18 | confirm_cancel | click (163, 110) | Confirm Destination |
| 19 | dismiss | Escape key | Any info dialog |
| 20 | random_click | click (random x,y) | Anywhere on screen |
| 21 | key_a | A key | New game / name entry screens |
| 22 | key_enter | Enter key | New game / confirmation screens |

Coordinates are in the native 320×200 logical resolution.

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
| `screen_type` | Discrete(11) | 0–10 | Which screen/dialog is currently showing |
| `ending` | Discrete(3) | 0/1/2 | 0=ongoing, 1=won, 2=lost |
| `num_soldiers` | Box int32 (1,) | 0 – 50 | Total soldiers across all bases |

### `screen_type` values

| Value | Screen |
|---|---|
| 0 | geoscape |
| 1 | ufo_detected |
| 2 | mission_detected |
| 3 | intercept |
| 4 | confirm_dest |
| 5 | low_fuel |
| 6 | craft_error |
| 7 | monthly_report |
| 8 | items_arriving |
| 9 | craft_patrol |
| 10 | other |

---

## Reward Structure

| Event | Reward |
|---|---|
| Any mouse click (named or random) | +0.002 |
| Wait / Dismiss (no click) | +0.000 |
| UFO first detected on radar | +10.0 |
| Craft dispatched (STR_OUT) | +50.0 |
| Game won (Cydonia) | +1000.0 |
| Strategic score tick (small delta) | delta × 0.001 |
| **Post-battle score delta** (delta ≥ 10 pts) | **delta × 1.0** |
| **Research project completed** | **delta × 1.0** |

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
  ├─ economy  [funds, score, total_score]           → Linear(3,16)  → ReLU → (16)
  ├─ time     [month, day]                          → Linear(2,8)   → ReLU → (8)
  └─ mission  [craft_status(3), fuel(1), dmg(1),
               ufo_detected(2), mission_site(2),
               screen_type(11), popup_depth(1),
               ending(3), soldiers(1)]  (25 total)  → Linear(25,32) → ReLU → (32)
                                        fusion cat(56) → Linear(56,64) → ReLU → (64)

PPO heads:  pi  Linear(64,64) → Tanh → Linear(64,64) → Tanh → Linear(64,23)
            vf  Linear(64,64) → Tanh → Linear(64,64) → Tanh → Linear(64,1)
```

PPO hyperparameters: `n_steps=256`, `batch_size=64`, `n_epochs=4`,
`gamma=0.99`, `lr=3e-4`, `ent_coef=0.10`.

---

## Checkpoints and Resume

Checkpoints are saved to `python\checkpoints\` every 10 000 timesteps.

```powershell
# Resume automatically (picks newest checkpoint by file date)
py -3 -m xcom_rl.train_ppo --n-envs 4 --all-visual --resume

# Load specific weights (behaviour-cloning init)
py -3 -m xcom_rl.train_ppo --bc-weights ./checkpoints/bc_init
```

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
Entropy has fallen to zero. Increase `ent_coef` (default 0.10) or reduce
the learning rate. Check `train/entropy_loss` in TensorBoard.

**`LNK1168: cannot open openxcom.exe`**
The linker can't overwrite the binary because it is still running. Kill all
processes before rebuilding: `taskkill /IM openxcom.exe /F`

**Partial weight load on resume**
Expected when the action space has grown since the checkpoint was saved.
The trainer prints which layers were skipped. Training continues normally
with the action head re-initialized randomly.
