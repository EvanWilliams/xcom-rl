# Reward Structure

All values are defined as class constants on `XcomInterceptionEnv` in
`python/xcom_gym/env.py`.  Every step returns a scalar reward that is the
sum of whichever components fire that step.

---

## Click rewards

Every agent step that involves a mouse click earns a small positive reward.
The size depends on what kind of action it is.

| Action type | Value | Constant |
|---|---|---|
| Named UI button (navigation, cancel, info) | +0.01 | `CLICK_REWARD` |
| Named UI button that commits a strategic decision | +0.02 | `COMMIT_CLICK_REWARD` |
| Random / right-click exploration | +0.003 | `RANDOM_CLICK_REWARD` |
| Any click while UFOpedia is open | +0 | (suppressed) |

**Commit actions** are buttons that spend resources or dispatch units:
`ufo_intercept`, `mission_intercept`, `confirm_ok`, `research_info_ok`,
`purchase_ok`, `craft_info_ok`, `craft_equip_ok`, `dismantle_ok`,
`manufacture_ok`, `mfg_start`, `mfg_info_ok`, `craft_soldiers_ok`,
`build_row0-5`, `place_pos0-35`.

---

## Penalties

| Event | Value | Constant | Notes |
|---|---|---|---|
| `wait` action | −0.005 | `WAIT_PENALTY` | Per step; discourages idle polling |
| Entering UFOpedia | −10.0 | `UFOPAEDIA_PENALTY` | One-time on entry |
| Loading a saved game | −50.0 | `LOAD_PENALTY` | Applied when load is detected |
| Opening options/pause menu | −500.0 | `OPTIONS_MENU_PENALTY` | Strong deterrent |
| Returning to main menu mid-episode (abort) | −200.0 | `ABORT_PENALTY` | Deferred to first step of next episode |
| Dismantling a non-lift facility | −20.0 | `DISMANTLE_PENALTY` | Per facility |
| Clicking invalid spot in PlaceFacilityState | −5.0 | `FACILITY_ERROR_PENALTY` | Per invalid click |

---

## Strategic events (geoscape)

| Event | Value | Constant | Notes |
|---|---|---|---|
| First UFO detected this episode | +10.0 | `UFO_BONUS` | One-time; fires when any UFO becomes detected |
| Craft transitions to STR_OUT (launched) | +50.0 | `LAUNCH_BONUS` | One-time when first craft goes out |
| Craft dispatched to terror site / grounded UFO / alien base | +75.0 | `MISSION_DISPATCH_REWARD` | Per unique (craft, destination) pair |
| UFO shot down (FLYING → CRASHED or DESTROYED) | +30.0 | `UFO_SHOOT_DOWN_REWARD` | Per UFO |
| Alien base discovered | +50.0 | `ALIEN_BASE_FOUND_REWARD` | Per base |
| Main menu → geoscape transition | +5.0 | `MENU_ESCAPE_REWARD` | Rewards escaping the title screen |

---

## Score-based rewards

The game's internal `total_score` is sampled each step. The delta is scaled
differently depending on its magnitude.

| Situation | Scale | Constant |
|---|---|---|
| Score delta < 10 pts (routine: income, maintenance, patrols) | ×1.0 | `SCORE_SCALE` |
| Score delta ≥ 10 pts (post-battle or end-of-month totals) | ×2.0 | `BATTLE_SCORE_SCALE` |
| Research score delta (project completed) | ×3.0 | `RESEARCH_SCORE_SCALE` |

Score can be negative (alien attacks, civilian casualties, soldier deaths).

---

## Base development

| Event | Value | Constant |
|---|---|---|
| Facility construction started | +25.0 | `BUILD_REWARD` | Per facility |
| Facility dismantled | −20.0 | `DISMANTLE_PENALTY` | Per facility |
| Additional base built (only when first base ≥ 18 occupied squares) | +500.0 | `NEW_BASE_REWARD` |

---

## Personnel & economy

| Event | Value | Constant | Notes |
|---|---|---|---|
| Scientist or engineer hired | +0.5 | `HIRE_REWARD` | Per person |
| Soldier promoted in rank | +50.0 | `PROMOTION_REWARD` | Per promotion |
| Net-new item added to stores | +0.05 | `PURCHASE_REWARD` | Per item |
| Funds spent (purchases, facilities, staff) | ×0.00005 | `FUNDS_SPEND_SCALE` | e.g. $500 K spent → +25 |

---

## Research & manufacturing

| Event | Value | Constant |
|---|---|---|
| New research project started | +25.0 | `RESEARCH_START_REWARD` | Per project |
| New manufacturing project started | +20.0 | `MANUFACTURE_START_REWARD` | Per project |

---

## Time

| Event | Value | Constant |
|---|---|---|
| In-game time elapsed | +0.0002 | `TIME_REWARD_PER_MINUTE` | Per in-game minute; ~8.6 pts/month |

---

## Battlescape combat

These accumulate in `_battle_reward_pending` during battle message processing
and are flushed to the reward on the next geoscape step.

| Event | Value | Constant |
|---|---|---|
| Weapon fired | +2.0 | `SHOT_REWARD` | Per shoot event |
| Alien HP damage dealt | +0.5 | `ALIEN_HIT_REWARD` | Per HP |
| New fog-of-war tile uncovered | +0.5 | `EXPLORE_REWARD_PER_TILE` | Per tile |
| Alien spotted for the first time this battle | +3.0 | `ALIEN_SPOT_REWARD` | Per alien |
| Item picked up from the ground | +2.0 | `PICKUP_REWARD` | Per item |

---

## Episode termination

| Outcome | Bonus | Notes |
|---|---|---|
| Win (Cydonia mission completed) | +1000.0 | `CYDONIA_BONUS` — episode terminates |
| Loss (returned to main menu) | +0 | Episode terminates; abort penalty carried to next episode |
| Truncated (max steps reached) | +0 | No terminal bonus |
