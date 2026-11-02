# Repository Guidelines

## Project Structure & Module Organization

- `robocasa/` contains the main benchmark code: environments, tasks, models, utilities, and data-collection scripts.
- `robocasa/environments/ithor_*` contains iTHOR room-specific tasks; shared iTHOR task mixins live in `robocasa/environments/ithor_atomic/`.
- `robocasa/environments/living_room/` and `robocasa/environments/procthor/` contain custom non-kitchen and ProcTHOR task stacks.
- `robocasa/models/assets/` stores MJCF, scene specs, object assets, and semantic calibration YAML.
- `robosuite_src/` is the vendored robosuite source used by this project.
- `tests/` contains pytest-based regression and environment checks.
- `docs/` contains installation, task, and asset reference documentation.

## Build, Test, and Development Commands

- Install locally:
  ```bash
  pip install -e .
  pip install -r requirements.txt
  ```
- Run a focused test:
  ```bash
  pytest tests/test_thor_semantics.py
  ```
- Run syntax checks after task edits:
  ```bash
  python -m py_compile path/to/file.py
  git diff --check
  ```
- Visualize an iTHOR task:
  ```bash
  python -m robocasa.scripts.visualize_scene --env IThorKitchenDrawerPnP --scene-id 28 --onscreen
  ```

## Coding Style & Naming Conventions

- Use Python with 4-space indentation and concise, task-specific classes.
- Prefer existing helpers in `ithor_base.py`, `ithor_atomic/`, and `robocasa/utils/` over duplicating logic.
- Environment classes use CamelCase, e.g. `IThorKitchenDrawerPnP`; task files use snake_case, e.g. `drawer_pnp.py`.
- Keep task code narrow: avoid extra switches, broad fallback cases, or unrelated refactors.
- For kitchen and living-room tasks, define language instructions (`lang`) directly in each task class, never in YAML or a centralized language table. Keep their YAML limited to scene and calibration parameters; handle scene-specific wording inside the task class. Do not extend this migration to other rooms without an explicit request.

## Testing Guidelines

- Use `pytest` for test files under `tests/`.
- Name tests `test_*.py` and keep fixtures deterministic when possible.
- Default to a single focused unit or static-check command for the changed behavior.
- For MuJoCo-heavy changes, run at most one representative headless `robosuite.make(...); env.reset()` per distinct initialization implementation. Do not test every supported task-and-scene combination by default.
- Treat a passing focused check as a stop condition. After a failure, fix the cause and rerun only the smallest affected case once; do not repeatedly expand or rerun validation.
- Do not run broad suites, high reset counts, full-room regressions, GUI trials, visualization, or data collection unless a concrete risk requires them or the user explicitly requests them.
- Documentation-only and Git-only changes require no test rerun.

## Commit & Pull Request Guidelines

- Recent commits use short imperative/update-style summaries, often starting with `Update:` or `fix`.
- Keep commits scoped to one task or subsystem.
- PRs should include: changed environments, validation commands, known limitations, and screenshots/video for viewer-facing changes.
- Mention any scene IDs used for validation, e.g. `IThorKitchenDrawerPnP --scene-id 28`.

## Agent-Specific Instructions

- Do not revert user changes unless explicitly asked.
- Prefer small, direct changes and remove stale or redundant logic when touching task code.
- Do not modify `docs/current_tasks.md` unless the task's underlying logic changes (for example, changing the destination from a table to a sink). Bug fixes, stage-message fixes, success-region alignment, calibration adjustments, and validation results alone do not warrant updates to this document.

## Benchmark Task Design

- Before adding, changing, extending, or reviewing a benchmark task, read `docs/task_design_checklist.md` in full and use it as a design and risk guide, not as a mandate to run every listed check.
- Keep validation within the minimal budget above and limited to the affected behavior. Reuse existing evidence instead of repeating checks.
- Report the checks actually performed and any unverified items.
