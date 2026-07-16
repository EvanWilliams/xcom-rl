# XCOM Reinforcement Learning

PPO agent trained on the OpenXcom strategic layer. The agent controls the
Geoscape — intercepting UFOs, managing funding, and working toward the final
Cydonia mission — using a custom bridge that exposes game state over a named
socket and accepts commands to advance time, launch craft, and dismiss menus.

---

## Repository layout

```
engine/openxcom/          OpenXcom source with RL bridge patches
  src/Bridge/             Socket server, state serializer, geo commands
  build/bin/openxcom.exe  Engine binary (after build)

python/
  xcom_bridge/            Low-level Windows socket client (ctypes)
  xcom_gym/
    env.py                Gymnasium environment (XcomInterceptionEnv)
  xcom_rl/
    train_ppo.py          PPO training entry point
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
| MSVC Build Tools | 2022 (for engine rebuild) |
| OpenXcom data files | UFO: Enemy Unknown (original or Steam) |

Install Python dependencies:

```powershell
pip install stable-baselines3[extra] gymnasium torch numpy
```

---

## Building the engine

From the repo root in a terminal that has `vcvars32.bat` on PATH (or run
inside a Developer Command Prompt):

```powershell
$vcvars = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars32.bat"
$build  = "$PSScriptRoot\engine\openxcom\build"
cmd /c "call `"$vcvars`" && cd /d `"$build`" && nmake"
```

The build is incremental — only changed `.cpp` files are recompiled.
Output binary: `engine\openxcom\build\bin\openxcom.exe`.

---

## Training

All commands are run from the `python\` directory:

```powershell
cd python
```

### Quickstart — single environment (no spectator window)

```powershell
py -3 -m xcom_rl.train_ppo --n-envs 1
```

Launches one headless engine process. Episodes fast-forward to the first UFO
before the agent takes control. Default: 100 000 timesteps.

### Standard run — 9 environments (1 visible spectator + 8 headless)

```powershell
py -3 -m xcom_rl.train_ppo --n-envs 9
```

- **rank 0**: visible window at 10 ms/tick so you can watch one agent play.
- **ranks 1–8**: headless, UFO pre-filled, running as fast as possible.
- Each engine binds its own socket (`openxcom_bridge_{rank}.sock` in `%TEMP%`).

### All-visual — every environment shows a window

```powershell
py -3 -m xcom_rl.train_ppo --n-envs 8 --all-visual
```

All 8 engines run at 10 ms/tick with visible windows. UFO prefill is disabled
(requires headless); episodes start on game-day 1 and wait for a natural UFO
spawn (~4 750 steps, ~4 in-game hours). Use `--timesteps 200000` or more.

### Long run with custom output directories

```powershell
py -3 -m xcom_rl.train_ppo --n-envs 9 --timesteps 500000 --log-dir ./runs/run01
```

---

## CLI flags

```
py -3 -m xcom_rl.train_ppo [OPTIONS]
```

| Flag | Type | Default | Description |
|---|---|---|---|
| `--n-envs N` | int | 9 | Number of parallel engine instances. Use 1 for a single headless env without SubprocVecEnv. |
| `--timesteps N` | int | 100 000 | Total environment steps to train for. |
| `--log-dir PATH` | str | `./runs/` | Directory for TensorBoard logs and VecMonitor CSV. |
| `--all-visual` | flag | off | Open a visible game window for every environment. Disables UFO prefill; requires larger `--timesteps`. |

---

## Environment constants (hardcoded in `train_ppo.py`)

These are not CLI flags but can be edited at the top of `train_ppo.py`:

| Constant | Default | Description |
|---|---|---|
| `SPECTATOR_SPEED_MS` | 10 | ms per 5-second game tick for visible instances (lower = faster). |
| `MAX_STEPS` | 3 000 | Per-episode step cap for headless environments (with UFO prefill). |
| `MAX_STEPS_VISUAL` | 100 000 | Per-episode step cap when `--all-visual` is set. |
| `CKPT_FREQ` | 10 000 | Save a checkpoint every N total timesteps. |
| `CKPT_DIR` | `./checkpoints/` | Checkpoint output directory. |
| `N_STEPS` | 256 | PPO rollout length per environment per update. |
| `BATCH_SIZE` | 64 | PPO minibatch size. |
| `N_EPOCHS` | 4 | PPO gradient passes per rollout. |
| `GAMMA` | 0.99 | Discount factor. |
| `LR` | 3e-4 | Adam learning rate. |
| `ENT_COEF` | 0.10 | Entropy coefficient (keeps exploration alive). |

---

## Output files

| Path | Description |
|---|---|
| `checkpoints/xcom_ppo_<N>_steps.zip` | Periodic checkpoint (every `CKPT_FREQ` steps). |
| `checkpoints/xcom_ppo_final.zip` | Final model after training completes. |
| `checkpoints/vecnormalize.pkl` | VecNormalize running statistics (required for eval). |
| `runs/monitor.*.csv` | Per-episode reward/length log (VecMonitor). |
| `runs/PPO_*/` | TensorBoard event files. |

---

## Monitoring with TensorBoard

```powershell
tensorboard --logdir python\runs
```

Then open `http://localhost:6006` in a browser. Key scalars to watch:

| Scalar | What it means |
|---|---|
| `rollout/ep_rew_mean` | Mean episode return. Baseline ≈ −30 (step penalty only); +10 means UFO spotted; +50 means craft launched. |
| `rollout/ep_len_mean` | Mean episode length. Too short = truncating before UFO; too long = not launching. |
| `train/entropy_loss` | Should stay negative and not collapse to zero. |
| `train/policy_gradient_loss` | Healthy range: −0.05 to +0.05. |

---

## Reward structure

| Event | Reward |
|---|---|
| Each step | −0.01 (step penalty) |
| UFO detected (phase 0 → 1) | +10.0 |
| Craft launched (phase 1 → 2) | +50.0 |
| Strategic score delta per step | `Δtotal_score × 0.001` |
| Game won (Cydonia mission complete) | +1000.0 (terminates episode) |
| Game lost | 0.0 (terminates episode) |

`total_score` is the sum of `(xcom_activity − alien_activity)` across all
funding countries — the same value the game uses for monthly council reviews.

---

## Observation space

| Key | Type | Range | Description |
|---|---|---|---|
| `funds` | Box int32 (1,) | 0 – 100 000 000 | Current treasury in USD. |
| `score` | Box int32 (1,) | −10 000 – 10 000 | Research completion score. |
| `total_score` | Box int32 (1,) | −100 000 – 100 000 | Strategic score (xcom vs alien activity). |
| `month` | Box int32 (1,) | 1 – 12 | Current in-game month. |
| `day` | Box int32 (1,) | 1 – 31 | Current in-game day. |
| `craft_status` | Discrete(3) | 0/1/2 | 0=ready, 1=in flight, 2=repairing/refuelling. |
| `craft_fuel_pct` | Box int32 (1,) | 0 – 100 | First craft fuel percentage. |
| `craft_damage_pct` | Box int32 (1,) | 0 – 100 | First craft damage percentage. |
| `ufo_detected` | Discrete(2) | 0/1 | 1 when a UFO is on radar. |
| `mission_phase` | Discrete(4) | 0/1/2/3 | 0=scanning, 1=ufo visible, 2=craft launched, 3=returned. |
| `popup_depth` | Box int32 (1,) | 0 – 10 | Number of UI dialogs currently open (0 = no popup). |
| `ending` | Discrete(3) | 0/1/2 | 0=ongoing, 1=won (Cydonia), 2=lost. |
| `num_soldiers` | Box int32 (1,) | 0 – 50 | Total soldiers across all bases. |

---

## Action space

| Action | Effect |
|---|---|
| `0` — wait | Headless: advance one 5-second game tick. Visible: poll for the next engine state. |
| `1` — launch | Send the ready craft to intercept the detected UFO. Falls back to wait if no craft/UFO is available. |
| `2` — dismiss | Inject an Escape key to close the current dialog or popup. In headless mode, auto-dismiss also fires after every step when `popup_depth > 0`. |

---

## Engine environment variables

The training script sets these automatically. Documented here for manual runs.

| Variable | Value | Effect |
|---|---|---|
| `XCOM_BRIDGE_AUTOSTART` | `1` | Skip the main menu; start a Beginner game at Thule Airfield immediately. |
| `XCOM_BRIDGE_SOCKET` | `openxcom_bridge.sock` | Socket filename (bare name; engine prepends `%TEMP%\`). Use unique names for multi-instance runs. |
| `XCOM_GAME_SPEED` | ms per tick (e.g. `10`) | Override the geoscape clock speed. 10 ms ≈ 8× faster than the default 80 ms tick. |

To launch the engine manually in auto-start mode:

```powershell
$env:XCOM_BRIDGE_AUTOSTART = "1"
.\engine\openxcom\build\bin\openxcom.exe

# Headless (no window, step-driven):
.\engine\openxcom\build\bin\openxcom.exe --headless
```

---

## Resuming from a checkpoint

```python
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from xcom_gym.env import XcomInterceptionEnv

vec_env = DummyVecEnv([lambda: XcomInterceptionEnv(headless=True)])
vec_env = VecNormalize.load("python/checkpoints/vecnormalize.pkl", vec_env)
vec_env.training = False
vec_env.norm_reward = False

model = PPO.load("python/checkpoints/xcom_ppo_final", env=vec_env)
obs = vec_env.reset()
while True:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    if done:
        obs = vec_env.reset()
```

---

## Troubleshooting

**Engine socket does not open within 10 s**
Cold-start asset loading takes 8–10 s on a typical machine. If it times out
consistently, check that `engine\openxcom\build\bin\` contains the UFO data
files (`GEODATA\`, `GEOGRAPH\`, etc.) and that `options.cfg` has
`playIntro: false`.

**All episodes truncate at step 3 000 with ep_rew ≈ −30**
Headless UFO prefill timed out (no UFO appeared within 60 in-game days).
This is rare; check that the engine is actually advancing time (run with
`--n-envs 1` and add some print statements to `env.py`).

**All episodes truncate at ~step 4 750 with ep_rew ≈ −27.5 in all-visual mode**
This is expected normal behaviour. Phase 2 (craft launched) terminates the
episode in headless mode to avoid a dogfight deadlock. The reward of −27.5 =
+10 (UFO) + 50 (launch) − 0.01 × 4 750 (steps) − penalties confirms the
agent is succeeding at both phase transitions. Increase `--timesteps` and let
the policy keep learning.

**Engine windows open but freeze immediately**
SDL dummy audio/video may conflict with another running instance. Make sure
each env has a unique `XCOM_BRIDGE_SOCKET` name and that no leftover
`openxcom.exe` processes are running (`taskkill /IM openxcom.exe /F`).

**`iteration 12` policy collapse (action always 0)**
Caused by `VecNormalize(norm_reward=True)` collapsing advantages when all
episodes return similar rewards. The training script uses `norm_reward=False`
by default; do not re-enable it.
