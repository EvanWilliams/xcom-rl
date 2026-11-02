---
name: update-model
description: The standing review methodology for the OpenXCOM backups project ("update model"). Invoke whenever auditing whether the project is correctly built/configured to reach the ultimate goal — BEATING CYDONIA ON SUPERHUMAN — or when asked to find anything missed or misconfigured across the project, propose changes to the latest version (v30), or reduce the number of updates needed. Active until the user says otherwise.
---

# update model

Standing review system for this project. The ultimate goal is **beating Cydonia on Superhuman** in the OpenXCOM build. This skill checks whether anything was missed or misconfigured — across the whole history of the project — that would make that goal unreachable, and proposes the *minimum* set of changes to the most recent version to fix it.

## Hard constraints (never violate)

1. **Source is read-only.** Never edit, move, or delete anything inside the `v*` backup folders (v1…v30) or any source directory. Only ever CREATE new files. Proposed changes are written as new files/diff notes under `update-model/`, never applied to the source. (See memory `no-edit-source-files`.)
2. **v30 is the "most recent version."** All "should this change?" decisions target v30-working-xcom. Source data = earlier versions / the pristine X-COM data they derive from.
3. **Reduce, don't inflate.** Every proposed change must survive an assumption-test; drop any change a test shows unnecessary. Fewer updates is always better.

## The 327 structure

**12 subsystems × 27 lenses = 324 cells, + 3 meta-verification systems = 327 parts.**

### The 12 subsystems (S01–S12)
| ID | Subsystem | Why it can block Cydonia@Superhuman |
|----|-----------|--------------------------------------|
| S01 | Engine build & runtime (`engine/openxcom/src` → binary) | If it won't build/run, nothing else matters |
| S02 | Difficulty scaling — Superhuman (`difficulty.rul`) | Wrong coefficients = unwinnable or trivial |
| S03 | Cydonia mission chain (`alienDeployments`, `alienMissions`, `missionScripts`) | The two-stage Mars final must spawn and be completable |
| S04 | Research tree & gating (`research.rul`) | Must be able to unlock the path to Cydonia |
| S05 | Manufacturing & economy (`manufacture.rul`, funding) | Must be able to afford the war to get there |
| S06 | Soldiers, armor & capacity (`soldiers.rul`, `armors.rul`) | Must be able to survive Superhuman |
| S07 | Items, weapons & damage (`items.rul`) | Must out-damage late-game aliens |
| S08 | Alien races, units & item-levels (`alienRaces`, `units`, `alienItemLevels`) | Enemy strength tuning on Superhuman |
| S09 | Craft & interception (`crafts`, `craftWeapons`, `ufos`) | Must reach missions and survive the skies |
| S10 | Base & facilities (`facilities.rul`, `startingBase.rul`) | Infrastructure to support the campaign |
| S11 | Maps & terrains (`terrains`, `mapScripts`, `PSX_Static_Cydonia_Map`) | Cydonia maps must render and be playable |
| S12 | Saves, options & active mod set (config, enabled mods, savegames) | The actual run config must match intent |

### The 27 lenses (K1–K27)
Every combination of the three triads, in order **Phase → Treatment → Cognition**:
- **Phase:** Planning · Implementation+AssumptionRendering · Testing
- **Treatment:** Describe · Detail · Analyze
- **Cognition:** Understand · Elegant · define-Reasoning

Each lens is a distinct procedure (full text in `update-model/PLAN-327.md` legend). Key load-bearing lenses:
- **K13 (Impl/Detail/Understand):** look at the SOURCE DATA, diff it against v30 field-by-field.
- **K16 (Impl/Analyze/Understand):** decide whether v30 needs a change; state the candidate change.
- **K17 (Impl/Analyze/Elegant):** test the assumption behind the change; achieve it with fewer/no edits (REDUCE).
- **K25–K26 (Test/Analyze):** after testing, roll back any change proven unnecessary.
- **K27 (Test/Analyze/Reasoning):** record the reasoned verdict: does this subsystem block the goal?

### Per-cell procedure (identical for all 324 cells)
1. **Look at the source data** for this subsystem through the cell's lens.
2. **Determine whether v30 needs a change.**
3. **Test the assumption** behind any proposed change.
4. **Reduce** — merge or drop updates until only the necessary set remains.

### The 3 meta-verification systems (M1–M3 = parts 325–327)
Each sweeps **across all 12 subsystems** for anything missed/misconfigured over the entire project that blocks the goal:
- **M1 — Planning-gap sweep:** requirements that were never planned for.
- **M2 — Implementation-gap sweep:** things planned but never configured, or misconfigured, in v30.
- **M3 — Testing-gap sweep:** changes/claims never actually verified in play.

## How to run

1. Open `update-model/PLAN-327.md` (the enumerated checklist).
2. Work cells in subsystem order (or jump to a subsystem/meta on request).
3. For each cell, perform the per-cell procedure; write findings as NEW files under `update-model/findings/` (one proposed-change note per real finding, never an edit to source).
4. Maintain a running "proposed changes to v30" list and a "reductions applied" list so the update count only ever shrinks.
5. End with the M1–M3 sweeps and a final verdict: *is beating Cydonia on Superhuman reachable with v30 as-is, and if not, what is the minimum change set?*

## Output layout (new files only)
```
update-model/
  PLAN-327.md          # the 327-part enumerated plan
  README.md            # orientation
  findings/            # one note per real proposed change (created during runs)
  proposed-changes.md  # running minimal change set targeting v30 (created during runs)
```
