# Miska et al. (2026): analysis and figure reproduction

This repository connects each manuscript panel to an explicit analysis cohort,
configuration, source entrypoint, and public-data request. It serves two users:

1. IBL developers can inspect `release_manifest/` to identify the EIDs, PIDs,
   trial ranges, ONE objects, spike-sorting objects, and external assets needed
   for public release.
2. Readers can use immutable profiles in `configs/figures/` instead of editing
   commented configuration blocks or machine-specific paths.

## Quick start

**Start with the [GLM-HMM guide](docs/glmhmm.md) to use saved trial states.**
Saved states are the default; installing the model source and recomputing states
are optional. The guide identifies the release links that are still pending.

List every profile, status, run count, and resolved cohort size:

```bash
python3 scripts/figure_registry.py
```

Preview the exact runs for a figure without loading IBL data:

```bash
python3 scripts/reproduce.py figure1_snr_glmhmm --dry-run
python3 scripts/reproduce.py supp1_glmhmm_states --dry-run
```

After obtaining the saved state archive and required figure data, configure its
location and run a profile whose status is `ready`:

```bash
python3 scripts/setup_data.py \
  --states-file /path/to/all_subject_states.npz \
  --one-cache-dir /path/to/ONE/cache \
  --output-root /path/to/outputs
python3 scripts/reproduce.py figure1_snr_glmhmm
```

Each profile supplies its own `figure_prefix`; runtime paths come from command
line arguments or the local path configuration. Multi-run profiles execute every member by default, or one can
be selected with `--run state2` or `--run left`.

Build the normalized IBL release inventory:

```bash
python3 scripts/release_manifest.py --resolve-pids-with-one
```

## Manuscript figure map

| Manuscript output | Profile(s) | Analysis entrypoint | GLM-HMM |
|---|---|---|---|
| Figure 1 | `figure1_snr_glmhmm`, `figure1_zi_glmhmm` | `opto_analysis/opto_analysis.py` | 2-state engaged, previous trial |
| Figure 2 | `figure2_snr_bs_no_glmhmm` | `ephys_analysis/SNr_inhibition_BS_downstream_effect.py` then `BS_postprocess.py` | No |
| Figure 3 | `figure3_vls_d1_glmhmm`, `figure3_vls_d2_glmhmm`, `figure3_stn_glmhmm` | `opto_analysis/opto_analysis.py` | 2-state engaged, previous trial |
| Figure 4 | `figure4_zapit_glmhmm` | `zapit/zapit_analysis.py` | 2-state engaged, previous trial |
| Supp. 1 BWM bias | `supp1_bwm_bias` | `bias_selectivity_analysis/quantify_biasselective_allregions_2.py` | Not applicable |
| Supp. 1 GLM-HMM states | `supp1_glmhmm_states` | `opto_analysis/opto_analysis.py`; four runs (`state1`-`state4`) | 4-state, previous trial |
| Supp. 1 state occupancy | `supp1_state_occupancy` | Awaiting colleague source | Uses GLM-HMM states |
| Supp. 1 hemisphere effects | `supp1_hemisphere_specific` | `opto_analysis/opto_analysis.py`; `left`, `right`, `both` runs | 2-state engaged, previous trial |
| Supp. 1 optogenetic validation | `supp1_optogenetic_validation` | Figure 2 ephys pipeline; SNr and ZI direct-stimulation runs | Not applicable |
| Supp. 1 NOGLMHMM | `supp1_snr_no_glmhmm` | `opto_analysis/opto_analysis.py` | No; performance/trial thresholds frozen in profile |

The NOGLMHMM sensitivity profile explicitly sets baseline performance to 0.8,
stim performance to 0.5, and minimum trial count to 300.

## Repository layout

```text
configs/figures/          immutable cohort and analysis profiles
metadata_final/           final row-level behavioral/ephys/ZAPIT metadata
opto_analysis/            Figures 1 and 3 behavioral optogenetics
ephys_analysis/           Figure 2 raw analysis and BS postprocessing snapshot
zapit/                    Figure 4 laser-scanning analysis and logs
bias_selectivity_analysis/ BWM bias-selectivity source
scripts/reproduce.py      profile-driven runtime adapter
scripts/release_manifest.py IBL public-data request generator
release_manifest/         generated release-facing CSV/JSON inventory
derived_data/             stable CSV/JSON/NPZ publication bundle contract
```

`CD_analysis_midbrain.py` is deliberately excluded from the documented Figure
2 pipeline because the coding-direction analysis is not currently part of the
frozen figure. It can be added later as a separate profile if used.

## Reproducibility stages

```text
public IBL data + documented external assets
                     |
                     v
            expensive raw analysis
                     |
                     v
         versioned CSV/JSON/NPZ bundle
                     |
                     v
             fast figure rendering
```

Pickle files may be archived for provenance, but they are not the required
public interchange format. See `derived_data/README.md`.

The behavioral and ZAPIT profiles are wired to the generic runner. The current
Figure 2 dependency set is present and inspectable, but raw execution remains
guarded until its numerical outputs are regression-tested. See `ephys_analysis/README.md` and `MIGRATION_PLAN.md`.

## Setup and local paths

The recorded environment is the author's current `iblenv` installation on
2026-09-28: Python 3.13.9. `requirements.txt` uses its direct dependency versions,
with the compatibility adjustments explained below.
The additional `future` pin satisfies the older ssm fork's declared dependency;
it was absent from the current environment. NumPy is deliberately pinned to 2.2.0
because ibllib 3.4.1 declares numpy<=2.2;
the installed runtime actually loads 2.3.5. The three-session inference regression
passed with both versions. `requirements-lock.txt` additionally pins the complete
resolved new setup; it is not a historical lock.
This is a setup based on an observed baseline, not a claim that historical runs used identical
versions or that every platform has been tested. The full package inventory and
GLM-HMM source fingerprints are under `provenance/`.

From this repository's root:

```bash
conda env create -f environment.yml
conda activate miska-2026
# Full resolved dependency pinning on the tested macOS/Python 3.13 platform:
python -m pip install -r requirements-lock.txt
# Optional, for the Plotly section of visualize_BS.py:
python -m pip install -r requirements-visualization.txt
```

State inference additionally requires the laboratory's GLM-HMM source and fitted
model assets. Its observed base commit is
`47ca6a1ae36dd42fe4db0464f7710e3fc66a31c1`; local Python modifications are also
required. Their source-file and patch fingerprints are recorded in
`provenance/local-source.json`. The patch is retained locally and is not included
in this public repository while code hosting is being discussed with Alberto.
The base commit alone is insufficient to recreate the local implementation.
This repository does not yet distribute the fitted model files or provide a
public download URL for them.

The original retrieval script uses the following separate `ssm` fork, whose
revision includes violation-trial mask handling:

```bash
git clone https://github.com/zashwood/ssm ../ssm
git -C ../ssm checkout e9b408b79e2ab22a05ca93c3a1f78a7dae461992
python -m pip install --no-build-isolation --no-deps -e ../ssm
```

The compiled extension requires a working C++ compiler. The author's installed
nested `GLM-HMM/ssm` is a different revision; both are recorded in
`provenance/local-source.json`. Use the separate revision above for the retrieval
script. Clean installation on a new machine remains to be verified.

Create a single local path configuration (the GLM-HMM data directory contains
its `GLM-HMM/data_by_animal` and `GLM-HMM/results` subdirectories):

```bash
python scripts/setup_data.py \
  --data-root /path/to/manuscript-data \
  --one-cache-dir /path/to/ONE/cache \
  --glmhmm-dir /path/to/GLM-HMM \
  --ssm-dir /path/to/ssm \
  --glmhmm-data-dir /path/to/GLM-HMM/outputs \
  --output-root /path/to/analysis-outputs
```

This creates `paths.local.json`, excluded from Git. Relative JSON paths resolve
against the JSON file's directory. Runtime precedence is command-line overrides,
then `MISKA_<KEY>` environment variables (hyphens become underscores), then JSON,
then repository-relative defaults. Use `--paths-config` on `reproduce.py` and the
retrieval script, or set `MISKA_PATHS_FILE`, to select a different configuration.
The existing `--glmhmm-dir`, `--one-cache-dir`, and `--output-root` runner options
continue to work. Dry-run plans show the resolved paths.

To cache a known, accessible session's trials, append `--eid SESSION_UUID` to the
setup command. Repeat `--eid` for more sessions; `--object` can select other ONE
objects instead of `trials`. Downloads use the configured ONE/Alyx access and
cache; unavailable or unreleased datasets raise an error. This is not a full
manuscript-data downloader. Consult `release_manifest/` for the release inventory.
Additional JSON keys cover the Allen annotation, structure tree, and BWM bias
selectivity table; these externally supplied assets are not downloaded implicitly.

## GLM-HMM retrieval and state archive

See the [dedicated GLM-HMM guide](docs/glmhmm.md) for the recommended saved-state
workflow, state-label meanings, and optional recomputation instructions.

`all_subject_states.csv` is historically a Python pickle, not CSV. Existing
archives remain readable with a warning. New retrieval output uses `.pkl`:

```bash
python scripts/retrieve_allsubject_states.py \
  --subject SWC_NM_110 --output /path/to/new_states.pkl
```

Repeat `--subject` to select more mice. The default cohort is
`metadata_final/metadata_opto_behavior.py`; use `--metadata` for the ZAPIT cohort
or another trusted metadata module containing `sessions`. Existing valid entries
are skipped unless `--overwrite` requests recomputation; the output must always
be a new file. `--states 4` denotes the fitted model. Analysis `n_states=2` means
pooling its states 2/3 as engaged and 1/4 as disengaged, not fitting a two-state model.

The trial-field loop includes all 18 fields and handles the `(N, 2)` intervals
array. If table construction fails because of unequal field lengths, inference stops
by default. Only after establishing that
specific fields really lack their first trial should you use
`--allow-leading-padding-eid SESSION_UUID`; this preserves the historical
one-zero-row padding and records repairs in a JSON sidecar. Gaps larger than one
are rejected. This option does not determine trial alignment for you.

For a public, non-pickle archive, convert a **trusted** local pickle:

```bash
python scripts/state_io.py /path/to/all_subject_states.csv /path/to/all_subject_states.npz
```

The exporter preserves dtypes, trial order, labels, probabilities, and empty
entries, and checks exact array equality after reading the result back. All state
readers accept NPZ; set `states-file` in the local JSON (or `MISKA_STATES_FILE`) to
use it. The NPZ has a versioned JSON manifest and requires no pickle loading.
Behavioral/ZAPIT entrypoints compute previous-trial filtering from these labels
and no longer load the unused previous-trial pickle files.

Validation and outstanding release items are documented in
[REPRODUCIBILITY_REVIEW.md](REPRODUCIBILITY_REVIEW.md).
