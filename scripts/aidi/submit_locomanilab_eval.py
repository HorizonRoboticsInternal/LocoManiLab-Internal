# Project RoboOrchard
#
# Copyright (c) 2024-2026 Horizon Robotics. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.

"""Prepare a local, offline LocoManiLab AIDI payload; never submit a job."""

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path


def copy(source, target):
    source = Path(source)
    if source.is_dir():
        for link in source.rglob("*"):
            if link.is_symlink() and (
                not link.resolve().is_relative_to(source.resolve())
                or link.resolve().is_dir()
            ):
                raise ValueError(
                    f"External or directory symlink in payload source: {link}"
                )
        shutil.copytree(
            source,
            target,
            ignore=shutil.ignore_patterns(
                ".git",
                "__pycache__",
                "*.pyc",
                ".venv",
                "venv",
                "build",
                "*.egg-info",
            ),
        )
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--locomanilab-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--remote-paths",
        type=Path,
        required=True,
        help="JSON with job, models and assets_archive paths on the worker",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--image",
        required=True,
        help="Image matching the supplied dependency overlays (including Python ABI)",
    )
    parser.add_argument("--queue", required=True)
    parser.add_argument("--project-id", required=True)
    parser.add_argument("--input-bucket", required=True)
    parser.add_argument("--wall-time", type=int, default=120, help="Minutes")
    args = parser.parse_args()
    sys.path.insert(0, str(args.locomanilab_root.resolve() / "src"))
    import yaml
    from locomanilab.runtime.roboorchard.evaluate import (
        load_config,
        local_output,
    )

    cfg = load_config(args.config)
    remote = json.loads(args.remote_paths.read_text())
    if set(remote) != {"job", "models", "assets_archive"}:
        raise ValueError("Map exactly job, models, assets_archive")
    if any(
        not isinstance(p, str) or not p.startswith(("/", "${WORKING_PATH}/"))
        for p in remote.values()
    ):
        raise ValueError(
            "Remote inputs must be absolute or WORKING_PATH-relative"
        )
    if args.wall_time <= 0:
        raise ValueError("wall-time must be positive")
    output = local_output(args.output_dir, Path.cwd())
    # Avoid recursively packaging the output, or overwriting user inputs.
    sources = [args.locomanilab_root / "src"] + [
        Path(cfg[k])
        for k in (
            "benchmark_repo",
            "g1_backend_repo",
            "roboorchard_repo",
            "job",
        )
    ]
    sources += [
        Path(p)
        for role in ("model", "sim")
        for p in cfg[role + "_python_path"]
    ]
    sources += [
        Path(cfg["policy_config"]),
        Path(cfg["assets_archive"]),
        Path(cfg["resolved_policy"]["models_dir"]),
    ]
    if any(
        output.is_relative_to(p.resolve())
        or p.resolve().is_relative_to(output)
        for p in sources
    ):
        raise ValueError("Package output overlaps a source repository/input")
    output.mkdir(parents=True, exist_ok=False)
    payload = output / "locomanilab_payload"
    copy(args.locomanilab_root / "src", payload / "loco/src")
    for name in ("LICENSE", "THIRD_PARTY_NOTICES.md"):
        source = args.locomanilab_root / name
        if source.exists():
            copy(source, payload / "loco" / name)
    copy(cfg["g1_backend_repo"], payload / "g1")
    for name in ("robocasa", "robosuite_src", "LICENSE"):
        source = Path(cfg["benchmark_repo"]) / name
        if name != "LICENSE" or source.exists():
            copy(source, payload / "benchmark" / name)
    for name in (
        "robo_orchard_lab",
        "holomotion_tokenizer",
        "LICENSE",
        "projects/holobrain_internal/humanoid/scripts",
        "projects/holobrain_internal/humanoid/configs",
    ):
        source = Path(cfg["roboorchard_repo"]) / name
        if name != "LICENSE" or source.exists():
            copy(source, payload / "robo" / name)
    cfg.pop("resolved_policy", None)
    root = "${WORKING_PATH}/locomanilab_payload"
    for role in ("model", "sim"):
        paths = []
        for i, source in enumerate(cfg[role + "_python_path"]):
            relative = f"deps/{role}_{i}"
            copy(source, payload / relative)
            paths.append(root + "/" + relative)
        cfg[role + "_python_path"] = paths
        cfg[role + "_python"] = "python3"
    policy = yaml.safe_load(Path(cfg["policy_config"]).read_text())
    policy["models_dir"] = remote["models"]
    (payload / "policy.yaml").write_text(yaml.safe_dump(policy))
    cfg.update(
        job=remote["job"],
        assets_archive=remote["assets_archive"],
        roboorchard_repo=root + "/robo",
        benchmark_repo=root + "/benchmark",
        g1_backend_repo=root + "/g1",
        policy_config=root + "/policy.yaml",
        output="${WORKING_PATH}/locomanilab_results",
    )
    (payload / "eval.json").write_text(json.dumps(cfg, indent=2) + "\n")
    (payload / "run.py").write_text(
        "import sys\nfrom pathlib import Path\n"
        'sys.path.insert(0, str(Path(__file__).parent / "loco/src"))\n'
        "from locomanilab.runtime.roboorchard.evaluate import main\nmain()\n"
    )
    hashes = {}
    for path in sorted(payload.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Payload contains a symlink: {path}")
        if path.is_file():
            hashes[str(path.relative_to(payload))] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
    (payload / "manifest.json").write_text(
        json.dumps(
            {
                "image": args.image,
                "files_sha256": hashes,
                "dependency_contract": "Overlays require the exact image/Python ABI used in local validation",
            },
            indent=2,
        )
    )
    plan = dict(
        job_name="locomanilab-eval",
        workspace_folder=str(output / "workspace"),
        docker_image=args.image,
        input_bucket=args.input_bucket,
        output_bucket=None,
        num_workers=1,
        gpu_per_worker=1,
        cpu_per_worker=cfg["cpu_limit"],
        cpu_mem_ratio=16,
        wall_time=args.wall_time,
        queue_name=args.queue,
        project_id=args.project_id,
        to_upload=[str(payload)],
        execute=False,
        cmd=[
            "set -euo pipefail",
            "export PYTHONDONTWRITEBYTECODE=1",
            'exec python3 "${WORKING_PATH}/locomanilab_payload/run.py" '
            '--config "${WORKING_PATH}/locomanilab_payload/eval.json"',
        ],
    )
    (output / "submit.json").write_text(json.dumps(plan, indent=2) + "\n")
    print(output / "submit.json")


if __name__ == "__main__":
    main()
