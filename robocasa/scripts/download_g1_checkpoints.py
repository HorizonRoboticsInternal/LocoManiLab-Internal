"""Download the G1 policy weights from HuggingFace.

The G1 whole-body policies (SONIC / AMO / HEFT / ScaleBFM / ALMI) are ONNX and
TorchScript artifacts that live in the **g1_backend repo**, never in this one --
see ``g1_bridge/g1_runtime.py:resolve_ckpt_path``. Backends declare their
weights as paths relative to that repo's root (``checkpoints/amo/policy.yaml``
and friends), so the whole payload has to land at ``<g1_backend>/checkpoints/``
with its directory structure intact.

Only the **weights** are downloaded (see ``WEIGHT_SUFFIXES``). The manifests
that name them -- ``policy.yaml`` / ``policy.json`` -- are tracked in
g1_backend's git, so a checkout already has them and fetching them would
overwrite the checked-out version with whatever was uploaded last.

This script fetches that tree from the ``checkpoint/controller/`` folder of the
LocoManiLab release repository, https://huggingface.co/datasets/GammaRobotics/LocoManiLab.

    python robocasa/scripts/download_g1_checkpoints.py --family sonic_v1_1 --plan
    python robocasa/scripts/download_g1_checkpoints.py --family sonic_v1_1
    python robocasa/scripts/download_g1_checkpoints.py --all --dry-run

Deliberately importable without g1_backend installed: it reaches the repo root
through ``g1_bridge.g1_runtime.sim2real_root()``, which uses ``find_spec`` and
never imports the package (importing it would pull in torch/onnxruntime). Do
NOT import ``robocasa.g1_host.bridge`` here -- that module imports sim2real
at module level, which defeats the point.
"""

import argparse
import os
import pathlib
import shutil
import sys

from termcolor import colored

REPO_ID = "GammaRobotics/LocoManiLab"
REPO_TYPE = "dataset"
REVISION = "main"
#: Folder of the Hub repo that mirrors <g1_backend>/checkpoints/.
REPO_PREFIX = "checkpoint/controller/"

#: family -> its subtree under REPO_PREFIX (which mirrors <g1_backend>/checkpoints/).
#: Each family's policy.yaml is the entry point the backend loads:
#:   sonic      -> g1_bridge/backends/sonic.py
#:   sonic_v1_1 -> g1_bridge/backends/sonic.py (opt-in variant, see below)
#:   amo      -> g1_bridge/backends/amo.py
#:   heft     -> g1_bridge/backends/generic.py
#:   scalebfm -> g1_bridge/backends/generic.py
#:   almi     -> g1_bridge/backends/almi.py
#:
#: ``sonic_v1_1`` is a second SONIC export, not a replacement: its 64-D
#: motion_token is only meaningful to its own decoder, so it is opt-in
#: (``--policy-config .../sonic/v1_1/g1/policy.yaml``) and ``sonic`` stays the
#: default every existing dataset was tokenized with. It also ships NO
#: ``planner_sonic.onnx`` -- the kinematic planner is variant-independent and
#: lives under ``sonic``. Selecting v1_1 also includes that shared planner,
#: without downloading the low_latency encoder and decoder.
FAMILIES = {
    "sonic": "sonic/low_latency/g1",
    "sonic_v1_1": "sonic/v1_1/g1",
    "amo": "amo",
    "heft": "heft/pmg",
    "scalebfm": "scalebfm/humanoid_transformer_m",
    "almi": "almi/flat_dual",
}

SHARED_WEIGHTS = {
    "sonic_v1_1": ("sonic/low_latency/g1/planner_sonic.onnx",),
}

#: What the Hub is the source of truth for: the weight blobs, nothing else.
#: Everything else in a family folder (policy.yaml / policy.json) is tracked in
#: g1_backend's git -- see its .gitignore and docs/artifacts.md -- so it comes
#: with the checkout and must not be clobbered by a Hub copy that may lag it.
#: The Hub folder holds only weights, but filter here rather than trusting it.
WEIGHT_SUFFIXES = (".onnx", ".pt", ".pth")


def resolve_dest(dest_arg):
    """The checkpoints directory to write into: the dir that will hold sonic/, amo/, ...

    --dest wins, then $SIM2REAL_ROOT, then the installed package's repo root.
    """
    if dest_arg:
        return pathlib.Path(dest_arg).expanduser().resolve()

    root = os.environ.get("SIM2REAL_ROOT")
    if not root:
        # find_spec-based, no import. Returns None if g1_backend isn't installed.
        from g1_bridge.g1_runtime import sim2real_root

        root = sim2real_root()

    if not root:
        raise SystemExit(
            colored(
                "Cannot locate the g1_backend repo root. Do one of:\n"
                "  pip install -e /path/to/g1_backend --no-deps --ignore-requires-python\n"
                "  export SIM2REAL_ROOT=/path/to/g1_backend\n"
                "  python robocasa/scripts/download_g1_checkpoints.py "
                "--family sonic_v1_1 --dest /path/to/g1_backend/checkpoints",
                "red",
            )
        )
    return pathlib.Path(root).expanduser().resolve() / "checkpoints"


def warn_if_backends_will_miss_it(dest):
    """$SIM2REAL_ROOT is honoured by resolve_ckpt_path but NOT by the composite.

    ``robocasa/g1_host/bridge.py`` computes its root as
    ``Path(sim2real.__file__).parent.parent`` -- the *installed package's*
    parent, ignoring the env var entirely. So a --dest pointing anywhere else
    downloads files the backends will never look at.
    """
    try:
        import sim2real
    except ImportError:
        return
    expected = pathlib.Path(sim2real.__file__).resolve().parent.parent / "checkpoints"
    if expected != dest:
        print(
            colored(
                f"WARNING: downloading to {dest}, but robocasa.g1_host.bridge\n"
                f"         resolves checkpoints against {expected}\n"
                f"         (it reads sim2real.__file__, NOT $SIM2REAL_ROOT).\n"
                f"         The backends will not find these files.",
                "red",
            )
        )


def plan(repo_id, revision, families, dest, overwrite):
    """-> (to_download, skipped, nbytes) using the Hub's own file listing.

    Non-weight files are dropped outright, not merely "skipped": they are git's
    to provide, so they never count as missing however the local tree looks.
    """
    from huggingface_hub import HfApi

    prefixes = tuple(FAMILIES[f] + "/" for f in families)
    shared = {path for family in families for path in SHARED_WEIGHTS.get(family, ())}
    entries = HfApi().list_repo_tree(repo_id, repo_type=REPO_TYPE, revision=revision,
                                     path_in_repo=REPO_PREFIX.rstrip("/"), recursive=True)
    # Paths relative to REPO_PREFIX, i.e. relative to <g1_backend>/checkpoints/.
    sizes = {entry.path[len(REPO_PREFIX):]: entry.size for entry in entries
             if getattr(entry, "size", None) is not None}
    missing_shared = [path for path in sorted(shared) if path not in sizes and not (dest / path).is_file()]
    if missing_shared:
        raise ValueError("Missing shared controller weights: " + ", ".join(missing_shared))

    to_download, skipped, nbytes = [], [], 0
    for rel, size in sorted(sizes.items()):
        if not (rel.startswith(prefixes) or rel in shared):
            continue
        if not rel.endswith(WEIGHT_SUFFIXES):
            continue
        if not overwrite and (dest / rel).exists():
            skipped.append(rel)
            continue
        to_download.append(rel)
        nbytes += size or 0
    return to_download, skipped, nbytes


def download_checkpoints(
    families,
    dest=None,
    repo_id=REPO_ID,
    revision=REVISION,
    overwrite=False,
    dryrun=False,
    assume_yes=False,
    plan_only=False,
):
    if not families or any(family not in FAMILIES for family in families):
        raise ValueError("Select explicit checkpoint families, or use --all.")
    dest = resolve_dest(dest)
    if plan_only:
        print(f"Destination: {dest}")
        for family in families:
            print(f"  {FAMILIES[family]}/: weight files only")
        for path in sorted({path for family in families for path in SHARED_WEIGHTS.get(family, ())}):
            print(f"  {path}: shared controller dependency")
        print("Offline plan; no Hub requests, directories, or downloads. Use --dry-run for remote file sizes.")
        return dest
    warn_if_backends_will_miss_it(dest)

    to_download, skipped, nbytes = plan(repo_id, revision, families, dest, overwrite)

    for rel in skipped:
        print(colored(f"Already exists under {dest / rel}\nSkipping.\n", "yellow"))

    if not to_download:
        print(colored("Nothing to download.", "green"))
        return dest

    print(colored(f"Destination: {dest}", "cyan"))
    for rel in to_download:
        print(colored(f"  {rel}", "cyan"))
    print(
        colored(
            f"This will download {nbytes / 2 ** 20:.1f} MiB "
            f"({len(to_download)} files) for: {', '.join(families)}",
            "yellow",
        )
    )

    if dryrun:
        print(colored("Dry run; nothing downloaded.", "cyan"))
        return dest

    if not assume_yes and input("Proceed? (y/n) ").strip().lower() != "y":
        print("Aborting.")
        return dest

    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import HfHubHTTPError

    dest.mkdir(parents=True, exist_ok=True)
    # The Hub keeps the tree under REPO_PREFIX, so download into a staging dir
    # (checkpoints/.cache/ is gitignored by g1_backend) and move each file to its
    # exact relative path under dest, where onnxruntime and resolve_ckpt_path expect it.
    staging = dest / ".cache" / "hub-download"
    for attempt in range(3):
        try:
            # allow_patterns gets exact paths, so our own skip logic above is
            # what decides -- an already-complete family costs 0 bytes.
            snapshot_download(
                repo_id,
                repo_type=REPO_TYPE,
                revision=revision,
                local_dir=str(staging),
                allow_patterns=[REPO_PREFIX + rel for rel in to_download],
                force_download=overwrite,
                max_workers=4,
            )
            break
        except HfHubHTTPError as exc:
            print(colored(f"Error downloading after try #{attempt + 1}: {exc}", "red"))
            print(
                colored(
                    "If this persists: `unset all_proxy ALL_PROXY` (the Hub client's\n"
                    "httpx does not speak SOCKS) and make sure HF_ENDPOINT is not\n"
                    "pointing at a mirror (mirrors break LFS/xet validation).",
                    "red",
                )
            )
            if attempt == 2:
                raise
    for rel in to_download:
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staging / REPO_PREFIX / rel, target)
    shutil.rmtree(staging, ignore_errors=True)
    print(colored(f"Done: {dest}", "green"))
    return dest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download the G1 policy weights into the g1_backend repo "
        "(the policy.yaml/json manifests come from g1_backend's git)."
    )
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument(
        "--all",
        action="store_true",
        help="download every checkpoint family",
    )
    selection.add_argument(
        "--family",
        type=str,
        nargs="+",
        default=None,
        choices=sorted(FAMILIES),
        help="checkpoint families to download; SONIC v1.1 includes its shared planner",
    )
    parser.add_argument(
        "--dest",
        type=str,
        default=None,
        help="checkpoints directory to write into. Defaults to "
        "$SIM2REAL_ROOT/checkpoints, else the installed g1_backend's repo root.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="re-download files that already exist",
    )
    parser.add_argument(
        "--dry-run", "--dryrun", dest="dryrun",
        action="store_true",
        help="query Hub metadata for exact files/sizes without downloading weights",
    )
    parser.add_argument("--plan", action="store_true",
                        help="print selected paths offline; do not query the Hub or download")
    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="skip the interactive size confirmation",
    )
    parser.add_argument("--repo-id", type=str, default=REPO_ID)
    parser.add_argument("--revision", type=str, default=REVISION)

    args = parser.parse_args()
    families = sorted(FAMILIES) if args.all else args.family

    try:
        download_checkpoints(
            families=families,
            dest=args.dest,
            repo_id=args.repo_id,
            revision=args.revision,
            overwrite=args.overwrite,
            dryrun=args.dryrun,
            assume_yes=args.yes,
            plan_only=args.plan,
        )
    except KeyboardInterrupt:
        print(colored("\nInterrupted. Re-run to resume.", "yellow"))
        sys.exit(130)
    except ValueError as exc:
        parser.error(str(exc))
