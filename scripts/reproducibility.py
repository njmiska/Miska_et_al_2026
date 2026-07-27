"""Shared, standard-library-only helpers for manuscript reproducibility tools."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
PROFILE_DIR = REPO_ROOT / "configs" / "figures"

PIPELINE_METADATA_VARIABLE = {
    "opto_behavior": "sessions",
    "ephys_bs": "insertions",
    "zapit": "sessions",
    "bias_selectivity": "sessions",
    "state_occupancy": "sessions",
}

PIPELINE_IDENTIFIER = {
    "opto_behavior": ("EID", "eid"),
    "ephys_bs": ("PID", "pid"),
    "zapit": ("EID", "eid"),
    "bias_selectivity": ("EID", "eid"),
    "state_occupancy": ("EID", "eid"),
}


def load_profiles(profile_dir: Path = PROFILE_DIR) -> list[dict[str, Any]]:
    """Load and minimally validate all declarative figure profiles."""
    profiles = []
    seen = set()
    for path in sorted(Path(profile_dir).glob("*.json")):
        profile = json.loads(path.read_text(encoding="utf-8"))
        profile["_profile_path"] = str(path.relative_to(REPO_ROOT))
        required = {
            "schema_version", "profile_id", "figure", "label", "pipeline",
            "metadata_source", "selection", "glmhmm",
        }
        missing = required - set(profile)
        if missing:
            raise ValueError(f"{path}: missing profile keys {sorted(missing)}")
        if profile["profile_id"] in seen:
            raise ValueError(f"Duplicate profile_id={profile['profile_id']!r}")
        if profile["pipeline"] not in PIPELINE_METADATA_VARIABLE:
            raise ValueError(
                f"{path}: unknown pipeline={profile['pipeline']!r}")
        profile.setdefault("status", "ready")
        profile.setdefault("entrypoint", None)
        profile.setdefault("parameters", {})
        profile.setdefault("output", {})
        profile.setdefault("runs", [])
        if profile["status"] == "ready" and profile["metadata_source"] is None:
            raise ValueError(
                f"{path}: ready profiles require a metadata_source")
        seen.add(profile["profile_id"])
        profiles.append(profile)
    if not profiles:
        raise FileNotFoundError(f"No figure profiles found in {profile_dir}")
    return profiles


def _load_python_metadata(path: Path, variable: str) -> list[dict[str, Any]]:
    """Load a trusted repository metadata module under an isolated name."""
    path = Path(path)
    module_hash = hashlib.sha256(str(path).encode("utf-8")).hexdigest()[:12]
    spec = importlib.util.spec_from_file_location(
        f"_miska_metadata_{module_hash}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load metadata module {path}")
    module = importlib.util.module_from_spec(spec)
    root = str(REPO_ROOT)
    added_root = root not in sys.path
    if added_root:
        sys.path.insert(0, root)
    try:
        spec.loader.exec_module(module)
    finally:
        if added_root:
            sys.path.remove(root)
    rows = getattr(module, variable, None)
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise TypeError(f"{path}:{variable} must be a list of dictionaries")
    return rows


def load_metadata_rows(profile: dict[str, Any]) -> list[dict[str, Any]]:
    if profile["metadata_source"] is None:
        return []
    path = REPO_ROOT / profile["metadata_source"]
    variable = PIPELINE_METADATA_VARIABLE[profile["pipeline"]]
    return _load_python_metadata(path, variable)


def value_matches(actual: Any, expected: Any) -> bool:
    """Match JSON-friendly exact values or lists of allowed values."""
    if expected is None:
        return True
    if isinstance(expected, list):
        return actual in expected
    return actual == expected


def row_matches_selection(row: dict[str, Any], selection: dict[str, Any]) -> bool:
    return all(value_matches(row.get(key), expected)
               for key, expected in selection.items())


def resolve_profile(profile: dict[str, Any]) -> list[dict[str, Any]]:
    return [row for row in load_metadata_rows(profile)
            if row_matches_selection(row, profile["selection"])]


def compress_integer_ranges(values: Iterable[int]) -> str:
    """Encode integer trial IDs as compact, half-open ranges."""
    ordered = sorted({int(value) for value in values})
    if not ordered:
        return "NONE"
    parts = []
    start = previous = ordered[0]
    for value in ordered[1:]:
        if value == previous + 1:
            previous = value
            continue
        parts.append(f"{start}:{previous + 1}")
        start = previous = value
    parts.append(f"{start}:{previous + 1}")
    return ",".join(parts)


def compact_trial_selection(value: Any) -> str:
    if value is None:
        return "UNSPECIFIED"
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple, set, range)):
        return compress_integer_ranges(value)
    return str(value)


def canonical_row_fingerprint(row: dict[str, Any]) -> str:
    """Stable short fingerprint after compacting potentially huge trial lists."""
    compact = {}
    for key, value in sorted(row.items()):
        if key in {"Trials_Range", "opto inhibition trials",
                   "opto excitation trials"}:
            compact[key] = compact_trial_selection(value)
        else:
            compact[key] = value
    payload = json.dumps(compact, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def identifier_for_row(pipeline: str, row: dict[str, Any]) -> tuple[str, str]:
    source_key, identifier_type = PIPELINE_IDENTIFIER[pipeline]
    identifier = row.get(source_key)
    if identifier is None or str(identifier).strip() in {"", "None", "nan"}:
        raise ValueError(f"Missing {source_key} in {pipeline} metadata row")
    return identifier_type, str(identifier)
