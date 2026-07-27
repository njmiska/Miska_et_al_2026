#!/usr/bin/env python3
"""Run or preview a manuscript analysis from its declarative profile."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

try:
    from .profile_runtime import get_profile, resolve_run, runtime_environment
    from .reproducibility import REPO_ROOT, load_profiles
except ImportError:
    from profile_runtime import get_profile, resolve_run, runtime_environment
    from reproducibility import REPO_ROOT, load_profiles


def _run_ids(profile, requested):
    runs = profile.get("runs", [])
    valid = [run["run_id"] for run in runs] or ["default"]
    if requested:
        unknown = sorted(set(requested) - set(valid))
        if unknown:
            raise ValueError(f"Unknown runs {unknown}; valid runs are {valid}")
        return requested
    return valid


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile_id", nargs="?")
    parser.add_argument("--run", action="append", dest="runs")
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "outputs")
    parser.add_argument("--glmhmm-dir", type=Path)
    parser.add_argument("--one-cache-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args(argv)

    if args.list:
        for profile in load_profiles():
            print(f"{profile['profile_id']:<38} {profile['status']:<24} {profile['label']}")
        return 0
    if not args.profile_id:
        parser.error("profile_id is required unless --list is used")

    profile = get_profile(args.profile_id)
    entrypoint = profile.get("entrypoint")
    if not entrypoint:
        raise SystemExit(
            f"{profile['profile_id']} has no entrypoint ({profile['status']})")
    run_ids = _run_ids(profile, args.runs)

    plans = []
    for run_id in run_ids:
        resolved = resolve_run(profile, run_id)
        output_dir = (args.output_root / profile["profile_id"] / run_id).resolve()
        command = [sys.executable, str((REPO_ROOT / entrypoint).resolve())]
        plans.append({
            "profile_id": profile["profile_id"],
            "run_id": run_id,
            "status": profile["status"],
            "entrypoint": entrypoint,
            "output_dir": str(output_dir),
            "selection": resolved["selection"],
            "glmhmm": resolved["glmhmm"],
            "parameters": resolved["parameters"],
            "figure_prefix": resolved["output"].get("figure_prefix"),
            "command": " ".join(shlex.quote(part) for part in command),
        })

    print(json.dumps(plans, indent=2, sort_keys=True))
    if args.dry_run:
        return 0
    if profile["status"] != "ready":
        raise SystemExit(
            f"Raw execution is disabled while profile status is {profile['status']}; "
            "use --dry-run to inspect the frozen configuration.")

    for plan in plans:
        output_dir = Path(plan["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        env = runtime_environment(
            profile["profile_id"], plan["run_id"], output_dir,
            args.glmhmm_dir, args.one_cache_dir)
        subprocess.run(shlex.split(plan["command"]), cwd=REPO_ROOT,
                       env=env, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
