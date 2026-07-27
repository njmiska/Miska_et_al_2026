#!/usr/bin/env python3
"""Inspect declarative manuscript figure profiles and their resolved cohorts."""

from __future__ import annotations

import argparse
import json

try:  # ``python -m scripts.figure_registry``
    from .reproducibility import identifier_for_row, load_profiles, resolve_profile
except ImportError:  # ``python scripts/figure_registry.py``
    from reproducibility import identifier_for_row, load_profiles, resolve_profile


def _profile_summary(profile):
    rows = resolve_profile(profile)
    identifiers = {
        identifier_for_row(profile["pipeline"], row)[1] for row in rows
    }
    return {
        "profile_id": profile["profile_id"],
        "figure": profile["figure"],
        "pipeline": profile["pipeline"],
        "glmhmm": bool(profile["glmhmm"]["enabled"]),
        "status": profile["status"],
        "runs": max(1, len(profile["runs"])),
        "metadata_rows": len(rows),
        "unique_identifiers": len(identifiers),
        "label": profile["label"],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile_id", nargs="?", help="Show one profile in detail")
    parser.add_argument(
        "--identifiers", action="store_true",
        help="Include the resolved EIDs/PIDs for a selected profile",
    )
    args = parser.parse_args(argv)

    profiles = load_profiles()
    if args.profile_id is None:
        for profile in profiles:
            summary = _profile_summary(profile)
            mode = "GLM-HMM" if summary["glmhmm"] else "NOGLMHMM"
            print(
                f"{summary['profile_id']:<42} figure={summary['figure']!s:<20} "
                f"{mode:<9} n={summary['unique_identifiers']:<4} "
                f"runs={summary['runs']:<2} status={summary['status']:<22} "
                f"{summary['label']}"
            )
        return 0

    matches = [profile for profile in profiles
               if profile["profile_id"] == args.profile_id]
    if not matches:
        parser.error(f"Unknown profile_id={args.profile_id!r}")
    profile = matches[0]
    output = dict(profile)
    output.pop("_profile_path", None)
    output["resolved"] = _profile_summary(profile)
    if args.identifiers:
        output["identifiers"] = sorted({
            identifier_for_row(profile["pipeline"], row)[1]
            for row in resolve_profile(profile)
        })
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
