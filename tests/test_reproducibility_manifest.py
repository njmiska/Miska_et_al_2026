from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.profile_runtime import get_profile, resolve_run
from scripts.release_manifest import generate_manifest
from scripts.reproducibility import (
    REPO_ROOT,
    compress_integer_ranges,
    identifier_for_row,
    load_profiles,
    resolve_profile,
)


class ReproducibilityManifestTests(unittest.TestCase):
    def test_trial_ranges_are_compact_and_half_open(self):
        self.assertEqual(compress_integer_ranges([0, 1, 2, 5, 6, 10]),
                         "0:3,5:7,10:11")
        self.assertEqual(compress_integer_ranges([]), "NONE")

    def test_run_matrix_overrides_are_resolved(self):
        state = resolve_run(get_profile("supp1_glmhmm_states"), "state4")
        self.assertEqual(state["glmhmm"]["n_states"], 4)
        self.assertEqual(state["glmhmm"]["state_type"], "state4")
        self.assertEqual(state["output"]["figure_prefix"],
                         "supp1_glmhmm_state4")
        hemisphere = resolve_run(
            get_profile("supp1_hemisphere_specific"), "left")
        self.assertEqual(hemisphere["selection"]["Hemisphere"], ["left"])

    def test_no_glmhmm_thresholds_are_frozen(self):
        profile = get_profile("supp1_snr_no_glmhmm")
        self.assertFalse(profile["glmhmm"]["enabled"])
        self.assertEqual(profile["parameters"]["BASELINE_PERFORMANCE_THRESHOLD"], 0.8)
        self.assertEqual(profile["parameters"]["STIM_PERFORMANCE_THRESHOLD"], 0.5)
        self.assertEqual(profile["parameters"]["MIN_NUM_TRIALS"], 300)

    def test_final_glmhmm_contract(self):
        profiles = {p["profile_id"]: p for p in load_profiles()}
        disabled = {
            profile_id for profile_id, profile in profiles.items()
            if profile["glmhmm"].get("applicable", True)
            and not profile["glmhmm"]["enabled"]
        }
        self.assertEqual(disabled, {
            "figure2_snr_bs_no_glmhmm",
            "supp1_snr_no_glmhmm",
        })
        self.assertEqual(profiles["supp1_glmhmm_states"]["glmhmm"]["n_states"], 4)
        self.assertEqual(len(profiles["supp1_glmhmm_states"]["runs"]), 4)
        self.assertEqual(len(profiles["supp1_hemisphere_specific"]["runs"]), 3)

    def test_every_profile_resolves_identifiers(self):
        expected = {
            "figure1_snr_glmhmm": 37,
            "figure1_zi_glmhmm": 55,
            "figure2_snr_bs_no_glmhmm": 25,
            "figure3_stn_glmhmm": 55,
            "figure3_vls_d1_glmhmm": 129,
            "figure3_vls_d2_glmhmm": 139,
            "figure4_zapit_glmhmm": 267,
            "supp1_bwm_bias": 0,
            "supp1_glmhmm_states": 37,
            "supp1_hemisphere_specific": 137,
            "supp1_optogenetic_validation": 11,
            "supp1_snr_no_glmhmm": 37,
            "supp1_state_occupancy": 0,
        }
        for profile in load_profiles():
            with self.subTest(profile=profile["profile_id"]):
                identifiers = {
                    identifier_for_row(profile["pipeline"], row)[1]
                    for row in resolve_profile(profile)
                }
                self.assertEqual(
                    len(identifiers), expected[profile["profile_id"]])

    def test_known_incomplete_profiles_are_explicit(self):
        profiles = {p["profile_id"]: p for p in load_profiles()}
        self.assertEqual(
            profiles["supp1_bwm_bias"]["status"], "awaiting_cohort_metadata")
        self.assertEqual(
            profiles["supp1_state_occupancy"]["status"], "awaiting_colleague_code")
        for profile in profiles.values():
            if profile["entrypoint"] is not None:
                self.assertTrue((REPO_ROOT / profile["entrypoint"]).is_file())

    def test_ephys_snapshot_hashes(self):
        manifest = json.loads((REPO_ROOT / "ephys_analysis" /
                               "SOURCE_SNAPSHOT.json").read_text())
        for relative, expected in manifest["files"].items():
            digest = hashlib.sha256(
                (REPO_ROOT / "ephys_analysis" / relative).read_bytes()
            ).hexdigest()
            self.assertEqual(digest, expected, relative)

    def test_manifest_is_normalized_and_self_consistent(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            summary = generate_manifest(output)
            self.assertEqual(summary["profiles"], 13)
            self.assertEqual(summary["profile_runs"], 19)
            self.assertGreaterEqual(summary["identifiers_by_type"]["pid"], 64)
            self.assertGreater(summary["identifiers_by_type"]["eid"], 0)

            with (output / "identifiers.csv").open(newline="", encoding="utf-8") as stream:
                identifiers = list(csv.DictReader(stream))
            keys = [(row["identifier_type"], row["identifier"])
                    for row in identifiers]
            self.assertEqual(len(keys), len(set(keys)))

            with (output / "figure_profiles.csv").open(newline="", encoding="utf-8") as stream:
                profile_rows = list(csv.DictReader(stream))
            incomplete = {"supp1_bwm_bias", "supp1_state_occupancy"}
            self.assertEqual(
                {row["profile_id"] for row in profile_rows
                 if int(row["unique_identifiers"]) == 0}, incomplete)

            with (output / "profile_runs.csv").open(
                    newline="", encoding="utf-8") as stream:
                run_rows = list(csv.DictReader(stream))
            self.assertEqual(len(run_rows), 19)
            self.assertEqual(
                {row["run_id"] for row in run_rows
                 if row["profile_id"] == "supp1_glmhmm_states"},
                {"state1", "state2", "state3", "state4"})

            parsed_summary = json.loads(
                (output / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(parsed_summary, summary)

            with (output / "dataset_requests.csv").open(
                    newline="", encoding="utf-8") as stream:
                requests = list(csv.DictReader(stream))
            request_keys = [(
                row["pipeline"], row["identifier_type"], row["identifier"],
                row["request_kind"], row["object_or_dataset"],
            ) for row in requests]
            self.assertEqual(len(request_keys), len(set(request_keys)))


if __name__ == "__main__":
    unittest.main()
