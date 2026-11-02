# update-model

Standing review system for this OpenXCOM backups project. Driven by the `/update-model` skill
(`.claude/skills/update-model/SKILL.md`).

**Ultimate goal being audited:** beating **Cydonia on Superhuman**.

**What this folder is:** the home for the 327-part review plan and every proposed-change output.
Everything here is *new* — nothing in this folder ever edits the `v*` source backups. v30 is the
"most recent version" that changes are *proposed against*; source backups (v1…v29 and the pristine
X-COM data) are read-only.

## Files
- `PLAN-327.md` — the full 327-part plan (12 subsystems × 27 lenses = 324 cells, + 3 meta sweeps).
- `findings/` — one note per real proposed change (created when the skill is run).
- `proposed-changes.md` — the running *minimal* change set targeting v30 (created when the skill is run).

## The 327 at a glance
- **12 subsystems** (S01–S12): the parts of the build that gate reaching/winning Cydonia@Superhuman.
- **27 lenses** (K1–K27): every combination of Phase × Treatment × Cognition.
- **3 meta-verification systems** (M1–M3): project-wide sweeps for anything missed/misconfigured.

To run: invoke the `update-model` skill, or just ask to "run update model" on a subsystem or meta sweep.
