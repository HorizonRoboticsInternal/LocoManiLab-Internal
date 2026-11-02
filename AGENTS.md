# Project Agent Notes

## Python Environments

- Inference / policy / sim work on both PC and onboard Orin uses the root project.
- Teleop work on both PC and onboard Orin uses `venv/pico`.
- When configuring or deploying on G1, set `HF_HUB_OFFLINE=1` and
  `HF_HUB_DISABLE_TELEMETRY=1` in the shell environment before running any
  policy or teleop process. Persist both exports in the robot user's
  `~/.bashrc`, and also include them explicitly in prepared deployment commands
  so non-interactive shells never attempt a Hugging Face network request.
- When touching inference/runtime files, verify syntax with the root project when available:
  `uv run python -m py_compile <files>`
- When touching teleop files, verify syntax with the teleop project when available:
  `uv run --project venv/pico python -m py_compile <files>`

## Documentation Sync

- Keep English and Chinese documentation in sync.
- When changing a doc page, update the corresponding English and Chinese sources in the same change whenever both versions exist.
- For Docusaurus content, treat `docs/` and `docs/i18n/zh-Hans/docusaurus-plugin-content-docs/current/` as paired sources when both files exist.
- Apply the same sync rule to top-level docs such as `README.md` and `README_zh.md`.

## Runtime Artifacts and Public Source

- Checkpoint manifests are tracked with the code. Download only the selected
  controller's weights; never overwrite manifests with an artifact download.
- Keep downloaded weights, SDK binaries, credentials and host-local network
  profiles out of git. See `docs/artifacts.md` for the supported asset layout.
- Real-robot deployment runs offline (`HF_HUB_OFFLINE=1` and
  `HF_HUB_DISABLE_TELEMETRY=1`). Prepare required assets before deployment.
- Keep personal upload workflows and machine-specific synchronization commands
  outside this repository. Do not publish artifacts from private training runs
  unless the user explicitly requests it.
- Preserve upstream attribution and third-party licensing. The root project is
  MIT-licensed (see `LICENSE`); vendored code and assets keep their own terms
  (see `THIRD_PARTY_NOTICES.md`). Do not relicense third-party code.

## GitHub Operations

- Use the `gh` CLI for GitHub repository, API, issue, pull-request, and
  authentication operations. Do not use Codex plugins or app connectors for
  GitHub.
- Use normal `git` commands for local version control and pushes. Configure Git
  authentication through `gh auth setup-git`.
