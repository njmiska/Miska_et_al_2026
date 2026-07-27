#!/usr/bin/env python3
"""Build normalized public-data release manifests for IBL developers.

The script is intentionally standard-library-only and does not run any analysis.
It converts the final Python metadata files and figure profiles into compact CSV
and JSON files describing identifiers, cohort membership, requested ONE data
families, and non-ONE assets.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

try:  # ``python -m scripts.release_manifest``
    from .profile_runtime import resolve_run
    from .reproducibility import (
        REPO_ROOT,
        canonical_row_fingerprint,
        compact_trial_selection,
        identifier_for_row,
        load_metadata_rows,
        load_profiles,
        row_matches_selection,
    )
except ImportError:  # ``python scripts/release_manifest.py``
    from profile_runtime import resolve_run
    from reproducibility import (
        REPO_ROOT,
        canonical_row_fingerprint,
        compact_trial_selection,
        identifier_for_row,
        load_metadata_rows,
        load_profiles,
        row_matches_selection,
    )


DATA_REQUIREMENTS = {
    "opto_behavior": [
        ("alf_object", "trials", "ONE-resolved ALF collection", True,
         "Choices, contrasts, block probability, event times, feedback"),
        ("alf_object", "wheel", "ONE-resolved ALF collection", True,
         "Wheel timestamps and position"),
        ("dataset", "_ibl_laserStimulation.intervals.npy", "ONE-resolved", True,
         "Optogenetic trial timing"),
        ("dataset_pattern", "_iblrig_taskData.raw*", "raw_behavior_data", False,
         "Legacy fallback for optogenetic trial identification"),
    ],
    "zapit": [
        ("alf_object", "trials", "ONE-resolved ALF collection", True,
         "Choices, contrasts, block probability, event times, feedback"),
        ("alf_object", "wheel", "ONE-resolved ALF collection", True,
         "Wheel timestamps and position"),
        ("dataset", "_ibl_laserStimulation.intervals.npy", "ONE-resolved", True,
         "Optogenetic trial timing"),
        ("alyx_record", "session", "Alyx", True,
         "Session identity and subject metadata"),
    ],
    "ephys_bs": [
        ("alf_object", "trials", "ONE-resolved ALF collection", True,
         "Choices, contrasts, block probability and alignment events"),
        ("dataset", "_ibl_laserStimulation.intervals.npy", "ONE-resolved", True,
         "Optogenetic trial timing"),
        ("spike_sorting_object", "spikes", "SpikeSortingLoader-resolved", True,
         "Spike times, clusters, amplitudes and depths"),
        ("spike_sorting_object", "clusters", "SpikeSortingLoader-resolved", True,
         "Cluster metrics and anatomical assignments"),
        ("spike_sorting_object", "channels", "SpikeSortingLoader-resolved", True,
         "Channel geometry and anatomical assignments"),
        ("spike_sorting_object", "waveforms_or_templates", "SpikeSortingLoader-resolved", True,
         "Light-artifact and waveform-amplitude quality control"),
        ("alyx_record", "probe_insertion_and_session", "Alyx", True,
         "PID-to-EID mapping and insertion metadata"),
    ],
    "bias_selectivity": [
        ("alf_object", "trials", "ONE-resolved ALF collection", True,
         "Choices, contrasts and block probability for BWM bias selectivity"),
        ("spike_sorting_object", "spikes", "SpikeSortingLoader-resolved", True,
         "Spike times and cluster assignments"),
        ("spike_sorting_object", "clusters", "SpikeSortingLoader-resolved", True,
         "Cluster quality and anatomical assignments"),
    ],
    "state_occupancy": [],
}


EXTERNAL_ASSETS = [
    {
        "asset_group": "glmhmm",
        "path_or_name": "all_subject_states.csv (currently a pickled object despite suffix)",
        "required_by": "all GLM-HMM profiles",
        "release_action": "Export to a documented non-pickle CSV/NPZ bundle",
    },
    {
        "asset_group": "glmhmm",
        "path_or_name": "engaged_prevtrial_indices.pkl",
        "required_by": "previous-trial engaged-state profiles",
        "release_action": "Export indices to NPZ plus JSON metadata",
    },
    {
        "asset_group": "zapit",
        "path_or_name": "zapit/zapit_trials.yml",
        "required_by": "figure4_zapit_glmhmm",
        "release_action": "Already tracked; validate coverage and document schema",
    },
    {
        "asset_group": "zapit",
        "path_or_name": "zapit/zapit_log.yml",
        "required_by": "figure4_zapit_glmhmm",
        "release_action": "Already tracked; validate coordinates and document schema",
    },
    {
        "asset_group": "atlas",
        "path_or_name": "Allen CCF annotation volume and structure tree",
        "required_by": "figure4_zapit_glmhmm",
        "release_action": "Document authoritative download URL/version; do not vendor cache",
    },
    {
        "asset_group": "derived_data",
        "path_or_name": "figure-level CSV/JSON/NPZ bundles",
        "required_by": "all figures",
        "release_action": "Create after final output audit; archive large bundles outside GitHub",
    },
    {
        "asset_group": "source_blocker",
        "path_or_name": "Supplementary Figure 1 state-occupancy analysis",
        "required_by": "supp1_state_occupancy",
        "release_action": "Obtain colleague source, exact cohort, and accessed data families",
    },
    {
        "asset_group": "metadata_blocker",
        "path_or_name": "Final BWM bias-selectivity cohort",
        "required_by": "supp1_bwm_bias",
        "release_action": "Freeze exact EIDs/PIDs and selection criteria in metadata_final",
    },
]


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=fieldnames, extrasaction="ignore",
            lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _metadata_records(profiles):
    profiles_by_source = defaultdict(list)
    for profile in profiles:
        if profile["metadata_source"] is not None:
            profiles_by_source[profile["metadata_source"]].append(profile)

    records = []
    for metadata_source, source_profiles in sorted(profiles_by_source.items()):
        # All profiles using one source have the same analysis pipeline.
        pipeline = source_profiles[0]["pipeline"]
        template = source_profiles[0]
        rows = load_metadata_rows(template)
        for row_number, row in enumerate(rows):
            identifier_type, identifier = identifier_for_row(pipeline, row)
            matched_profiles = [
                profile["profile_id"] for profile in source_profiles
                if row_matches_selection(row, profile["selection"])
            ]
            trial_value = row.get(
                "Trials_Range", row.get("opto inhibition trials"))
            records.append({
                "metadata_source": metadata_source,
                "metadata_row": row_number,
                "row_fingerprint": canonical_row_fingerprint(row),
                "pipeline": pipeline,
                "identifier_type": identifier_type,
                "identifier": identifier,
                "eid": str(row.get("EID", "")),
                "pid": str(row.get("PID", "")),
                "probe_name": "",
                "mouse": str(row.get("Mouse_ID", row.get("mouse", ""))),
                "date": str(row.get("Date", "")),
                "brain_region": str(row.get("Brain_Region", row.get("brain region", ""))),
                "condition": str(row.get("condition", "")),
                "hemisphere": str(row.get("Hemisphere", "")),
                "trial_selection": compact_trial_selection(trial_value),
                "figure_profiles": ";".join(sorted(matched_profiles)),
            })
    return sorted(records, key=lambda row: (
        row["pipeline"], row["identifier"], row["metadata_row"]))


def _resolve_pid_links(metadata_records):
    """Populate parent EID/probe for PID records using lightweight Alyx calls."""
    try:
        from one.api import ONE
    except ImportError as exc:
        raise RuntimeError(
            "--resolve-pids-with-one requires the IBL ONE-api environment") from exc
    one = ONE(base_url="https://alyx.internationalbrainlab.org", silent=True)
    cache = {}
    for record in metadata_records:
        if record["identifier_type"] != "pid":
            continue
        pid = record["identifier"]
        if pid not in cache:
            eid, probe_name = one.pid2eid(pid)
            cache[pid] = (str(eid) if eid is not None else "", probe_name or "")
        record["eid"], record["probe_name"] = cache[pid]
    return cache


def _identifier_records(metadata_records):
    grouped = defaultdict(list)
    for record in metadata_records:
        grouped[(record["identifier_type"], record["identifier"])].append(record)
    rows = []
    for (identifier_type, identifier), group in sorted(grouped.items()):
        rows.append({
            "identifier_type": identifier_type,
            "identifier": identifier,
            "pipelines": ";".join(sorted({row["pipeline"] for row in group})),
            "figure_profiles": ";".join(sorted({
                profile
                for row in group
                for profile in row["figure_profiles"].split(";")
                if profile
            })),
            "metadata_rows": len(group),
            "mouse": ";".join(sorted({row["mouse"] for row in group if row["mouse"]})),
            "eid": ";".join(sorted({row["eid"] for row in group if row["eid"]})),
            "pid": ";".join(sorted({row["pid"] for row in group if row["pid"]})),
            "probe_name": ";".join(sorted({
                row["probe_name"] for row in group if row["probe_name"]
            })),
        })
    return rows


def _dataset_request_records(metadata_records):
    identifiers_by_pipeline = defaultdict(dict)
    for record in metadata_records:
        key = (record["identifier_type"], record["identifier"])
        identifiers_by_pipeline[record["pipeline"]][key] = (
            record["eid"], record["pid"], record["probe_name"])
    requests = []
    for pipeline, identifiers in sorted(identifiers_by_pipeline.items()):
        for (identifier_type, identifier), links in sorted(identifiers.items()):
            eid, pid, probe_name = links
            for request_kind, name, collection, required, purpose in DATA_REQUIREMENTS[pipeline]:
                requests.append({
                    "pipeline": pipeline,
                    "identifier_type": identifier_type,
                    "identifier": identifier,
                    "eid": eid,
                    "pid": pid,
                    "probe_name": probe_name,
                    "request_kind": request_kind,
                    "object_or_dataset": name,
                    "collection": collection,
                    "required": str(bool(required)).lower(),
                    "purpose": purpose,
                })
    return requests


def _profile_records(profiles, metadata_records):
    counts = defaultdict(lambda: {"rows": 0, "identifiers": set()})
    for record in metadata_records:
        for profile_id in filter(None, record["figure_profiles"].split(";")):
            counts[profile_id]["rows"] += 1
            counts[profile_id]["identifiers"].add(record["identifier"])
    rows = []
    for profile in profiles:
        count = counts[profile["profile_id"]]
        rows.append({
            "profile_id": profile["profile_id"],
            "figure": profile["figure"],
            "label": profile["label"],
            "pipeline": profile["pipeline"],
            "glmhmm_enabled": str(bool(profile["glmhmm"]["enabled"])).lower(),
            "status": profile["status"],
            "entrypoint": profile["entrypoint"] or "",
            "runs": max(1, len(profile["runs"])),
            "metadata_source": profile["metadata_source"] or "",
            "metadata_rows": count["rows"],
            "unique_identifiers": len(count["identifiers"]),
            "profile_path": profile["_profile_path"],
        })
    return rows


def _profile_run_records(profiles):
    records = []
    for profile in profiles:
        metadata = load_metadata_rows(profile)
        run_ids = [run["run_id"] for run in profile["runs"]] or ["default"]
        for run_id in run_ids:
            resolved = resolve_run(profile, run_id)
            matched = [
                row for row in metadata
                if row_matches_selection(row, resolved["selection"])
            ]
            identifiers = {
                identifier_for_row(profile["pipeline"], row)[1]
                for row in matched
            }
            records.append({
                "profile_id": profile["profile_id"],
                "run_id": run_id,
                "status": profile["status"],
                "pipeline": profile["pipeline"],
                "figure_prefix": resolved["output"].get("figure_prefix", ""),
                "glmhmm_enabled": str(bool(
                    resolved["glmhmm"].get("enabled"))).lower(),
                "n_states": resolved["glmhmm"].get("n_states", ""),
                "state_type": resolved["glmhmm"].get("state_type", ""),
                "state_definition": resolved["glmhmm"].get(
                    "state_definition", ""),
                "selection_json": json.dumps(
                    resolved["selection"], sort_keys=True, separators=(",", ":")),
                "parameters_json": json.dumps(
                    resolved["parameters"], sort_keys=True, separators=(",", ":")),
                "metadata_rows": len(matched),
                "unique_identifiers": len(identifiers),
            })
    return records


def generate_manifest(output_dir: Path, *, resolve_pids_with_one: bool = False):
    profiles = load_profiles()
    metadata_records = _metadata_records(profiles)
    resolved_pid_links = (
        _resolve_pid_links(metadata_records) if resolve_pids_with_one else {})
    identifier_records = _identifier_records(metadata_records)
    dataset_requests = _dataset_request_records(metadata_records)
    profile_records = _profile_records(profiles, metadata_records)
    profile_run_records = _profile_run_records(profiles)
    unresolved_pid_records = [
        {
            "identifier_type": "pid",
            "identifier": pid,
            "reason": "Alyx pid2eid returned no parent session",
        }
        for pid, (eid, _probe_name) in sorted(resolved_pid_links.items())
        if not eid
    ]

    output_dir = Path(output_dir)
    _write_csv(output_dir / "figure_profiles.csv", profile_records, [
        "profile_id", "figure", "label", "pipeline", "glmhmm_enabled",
        "status", "entrypoint", "runs", "metadata_source", "metadata_rows",
        "unique_identifiers", "profile_path",
    ])
    _write_csv(output_dir / "metadata_rows.csv", metadata_records, [
        "metadata_source", "metadata_row", "row_fingerprint", "pipeline",
        "identifier_type", "identifier", "eid", "pid", "mouse", "date",
        "probe_name", "brain_region", "condition", "hemisphere", "trial_selection",
        "figure_profiles",
    ])
    _write_csv(output_dir / "profile_runs.csv", profile_run_records, [
        "profile_id", "run_id", "status", "pipeline", "figure_prefix",
        "glmhmm_enabled", "n_states", "state_type", "state_definition",
        "selection_json", "parameters_json", "metadata_rows",
        "unique_identifiers",
    ])
    _write_csv(output_dir / "identifiers.csv", identifier_records, [
        "identifier_type", "identifier", "pipelines", "figure_profiles",
        "metadata_rows", "mouse", "eid", "pid", "probe_name",
    ])
    _write_csv(output_dir / "dataset_requests.csv", dataset_requests, [
        "pipeline", "identifier_type", "identifier", "eid", "pid", "probe_name", "request_kind",
        "object_or_dataset", "collection", "required", "purpose",
    ])
    _write_csv(output_dir / "external_assets.csv", EXTERNAL_ASSETS, [
        "asset_group", "path_or_name", "required_by", "release_action",
    ])
    _write_csv(output_dir / "unresolved_identifiers.csv", unresolved_pid_records, [
        "identifier_type", "identifier", "reason",
    ])

    summary = {
        "schema_version": 1,
        "profiles": len(profile_records),
        "profile_runs": len(profile_run_records),
        "metadata_rows": len(metadata_records),
        "unique_identifiers": len(identifier_records),
        "identifiers_by_type": {
            identifier_type: sum(
                row["identifier_type"] == identifier_type
                for row in identifier_records)
            for identifier_type in sorted({
                row["identifier_type"] for row in identifier_records})
        },
        "dataset_requests": len(dataset_requests),
        "resolved_pid_eids": sum(bool(eid) for eid, _ in resolved_pid_links.values()),
        "unresolved_pid_eids": len(unresolved_pid_records),
        "unresolved_pid_identifiers": [
            row["identifier"] for row in unresolved_pid_records
        ],
        "external_assets": len(EXTERNAL_ASSETS),
        "profile_counts": {
            row["profile_id"]: row["unique_identifiers"]
            for row in profile_records
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=REPO_ROOT / "release_manifest",
        help="Output directory (default: repository release_manifest/)",
    )
    parser.add_argument(
        "--resolve-pids-with-one", action="store_true",
        help="Query Alyx for each PID's parent EID and probe name (no data download)",
    )
    args = parser.parse_args(argv)
    summary = generate_manifest(
        args.output, resolve_pids_with_one=args.resolve_pids_with_one)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
