"""Apply a declarative manuscript profile to a legacy config namespace.

This module changes configuration values only.  It deliberately does not
reimplement any scientific calculation in the analysis pipelines.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

try:
    from .reproducibility import load_profiles
except ImportError:
    from reproducibility import load_profiles


ENV_PROFILE_ID = "MISKA_PROFILE_ID"
ENV_RUN_ID = "MISKA_RUN_ID"
ENV_OUTPUT_DIR = "MISKA_OUTPUT_DIR"
ENV_GLMHMM_DIR = "MISKA_GLMHMM_DIR"
ENV_ONE_CACHE_DIR = "MISKA_ONE_CACHE_DIR"


def get_profile(profile_id: str) -> dict[str, Any]:
    matches = [p for p in load_profiles() if p["profile_id"] == profile_id]
    if not matches:
        raise ValueError(f"Unknown manuscript profile: {profile_id}")
    return matches[0]


def resolve_run(profile: dict[str, Any], run_id: str | None) -> dict[str, Any]:
    """Return merged profile sections for one run-matrix member."""
    runs = profile.get("runs", [])
    if not runs:
        if run_id not in {None, "default"}:
            raise ValueError(f"{profile['profile_id']} has no run {run_id!r}")
        selected_run = {"run_id": "default"}
    else:
        if run_id is None:
            if len(runs) != 1:
                raise ValueError(
                    f"{profile['profile_id']} has {len(runs)} runs; choose one")
            selected_run = runs[0]
        else:
            matches = [run for run in runs if run["run_id"] == run_id]
            if not matches:
                valid = ", ".join(run["run_id"] for run in runs)
                raise ValueError(f"Unknown run {run_id!r}; choose from {valid}")
            selected_run = matches[0]

    merged = {
        "run_id": selected_run["run_id"],
        "selection": dict(profile.get("selection", {})),
        "glmhmm": dict(profile.get("glmhmm", {})),
        "parameters": dict(profile.get("parameters", {})),
        "output": dict(profile.get("output", {})),
    }
    for name in ("selection", "glmhmm", "parameters", "output"):
        merged[name].update(selected_run.get(f"{name}_overrides", {}))
    return merged


def _filter_value(value):
    if not isinstance(value, list):
        return value
    if len(value) == 1:
        return value[0]
    allowed = frozenset(value)
    return lambda actual, allowed=allowed: actual in allowed


def apply_runtime_profile(namespace: dict[str, Any], pipeline: str) -> dict[str, Any] | None:
    """Apply the profile named by environment variables to ``namespace``."""
    profile_id = os.environ.get(ENV_PROFILE_ID)
    if not profile_id:
        return None
    profile = get_profile(profile_id)
    if profile["pipeline"] != pipeline:
        raise ValueError(
            f"Profile {profile_id} uses {profile['pipeline']}, not {pipeline}")
    if profile["status"] != "ready":
        raise RuntimeError(
            f"Profile {profile_id} is {profile['status']}; raw execution is not enabled")

    resolved = resolve_run(profile, os.environ.get(ENV_RUN_ID))
    existing_filters = namespace.get("SESSION_FILTERS", {})
    filters = {key: None for key in existing_filters}
    filters.update({
        key: _filter_value(value)
        for key, value in resolved["selection"].items()
    })
    namespace["SESSION_FILTERS"] = filters

    glmhmm = resolved["glmhmm"]
    namespace["USE_GLMHMM"] = bool(glmhmm.get("enabled"))
    namespace["N_STATES"] = glmhmm.get("n_states")
    namespace["STATE_TYPE"] = glmhmm.get("state_type")
    namespace["STATE_DEF"] = glmhmm.get("state_definition")

    for key, value in resolved["parameters"].items():
        if key not in namespace:
            raise KeyError(f"Profile parameter {key} is not defined by {pipeline} config")
        namespace[key] = value

    namespace["FIGURE_PREFIX"] = resolved["output"].get(
        "figure_prefix", profile_id)
    output_dir = os.environ.get(ENV_OUTPUT_DIR)
    if output_dir:
        namespace["FIGURE_SAVE_PATH"] = Path(output_dir).expanduser().resolve()

    glmhmm_dir = os.environ.get(ENV_GLMHMM_DIR)
    if glmhmm_dir:
        base = Path(glmhmm_dir).expanduser().resolve()
        namespace["GLMHMM_BASE_DIR"] = base
        namespace["GLMHMM_STATES_FILE"] = base / "all_subject_states.csv"
        namespace["GLMHMM_ENGAGED_PREV_FILE"] = base / "engaged_prevtrial_indices.pkl"
        namespace["GLMHMM_DISENGAGED_PREV_FILE"] = base / "disengaged_prevtrial_indices.pkl"

    one_cache = os.environ.get(ENV_ONE_CACHE_DIR)
    if one_cache:
        namespace["ONE_CACHE_DIR"] = Path(one_cache).expanduser().resolve()

    namespace["ACTIVE_MANUSCRIPT_PROFILE"] = profile_id
    namespace["ACTIVE_MANUSCRIPT_RUN"] = resolved["run_id"]
    return resolved


def runtime_environment(profile_id: str, run_id: str, output_dir: Path,
                        glmhmm_dir: Path | None = None,
                        one_cache_dir: Path | None = None) -> dict[str, str]:
    env = dict(os.environ)
    env[ENV_PROFILE_ID] = profile_id
    env[ENV_RUN_ID] = run_id
    env[ENV_OUTPUT_DIR] = str(output_dir)
    if glmhmm_dir is not None:
        env[ENV_GLMHMM_DIR] = str(glmhmm_dir)
    if one_cache_dir is not None:
        env[ENV_ONE_CACHE_DIR] = str(one_cache_dir)
    return env
