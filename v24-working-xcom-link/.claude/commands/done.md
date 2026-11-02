Before marking any task complete, you must satisfy all three quality gates:

1. **No build failures** — Run the build and confirm it exits with no errors. For C++ changes, rebuild the engine (`nmake openxcom` in `engine/openxcom/build/`). For Python changes, confirm the module imports without error.

2. **Tests pass** — Run any relevant tests and confirm zero failures. If no automated tests exist for the changed code, write a minimal smoke test and run it.

3. **Documentation updated** — Update any docstrings, comments, or plan files that describe the changed behaviour. If the change adds or modifies a reward, action, observation field, or config constant, update the relevant section of CLAUDE.md or inline comments so the next developer (or future Claude session) has accurate information.

Do not report a task as done until all three gates are green.
