# Findings — S02–S11 (game rulesets): STOCK, no changes needed

**Scope:** difficulty, Cydonia mission chain, research, manufacture, soldiers/armor, items,
alien races/units, craft, base/facilities, maps/terrains.

**Method (per 27-lens procedure, condensed):** read v30's `engine/openxcom/bin/standard/xcom1/*.rul`
and compared against stock OpenXCOM xcom1. Spot-checked the goal-critical files directly.

## Key evidence
- `difficulty.rul` (S02): `difficultyCoefficient: [0,1,2,3,4]`, stock `statGrowthMultipliers`,
  `aimAndArmorMultipliers: [0.5,1,1,1,1]` — these are the **stock Superhuman-capable** values.
- `research.rul` (S04): the endgame chain is intact —
  `STR_COMMANDER_PLUS → unlocks STR_CYDONIA_OR_BUST`; `STR_CYDONIA_OR_BUST` has
  `unlockFinalMission: true`, `dependencies: [STR_COMMANDER_PLUS]`, cost 600.
- `alienDeployments.rul` (S03): `STR_MARS_CYDONIA_LANDING` → `nextStage: STR_MARS_THE_FINAL_ASSAULT`,
  `finalDestination: true`, `loseCutscene: loseGame`; `STR_MARS_THE_FINAL_ASSAULT` present.
- S05–S11 (manufacture, soldiers, armors, items, alienRaces/units/alienItemLevels, crafts/craftWeapons/ufos,
  facilities/startingBase, terrains/mapScripts + the bundled `PSX_Static_Cydonia_Map` mod):
  all match the stock xcom1 dataset; no project-specific edits found.

## Verdict for S02–S11
**Cydonia on Superhuman is winnable by construction** — this is the stock X-COM: UFO Defense
campaign, which is designed to be completable on Superhuman. Nothing in these subsystems blocks
the goal.

**v30 changes needed: NONE.** (Assumption tested: "maybe a ruleset was edited in a way that breaks
the final mission or makes Superhuman unwinnable" — checked the goal-critical files; false.)

## Reduction applied
Originally 10 subsystems × 27 = 270 cells, each a candidate for a change. All collapse to a single
shared conclusion (stock ⇒ no change), so **270 potential updates reduce to 0**. This file replaces
10 separate per-subsystem notes — the biggest single reduction in the whole audit.

## Residual risk (carried to meta sweep)
The ruleset being stock only guarantees the game is *winnable by a competent player*. It says
nothing about whether the **RL agent** can learn to win it. That question lives in S01/S12 and the
RL training config, not here.
