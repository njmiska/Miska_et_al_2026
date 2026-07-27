# Miska et al. (2026): analysis and figure reproduction

This repository connects each manuscript panel to an explicit analysis cohort,
configuration, source entrypoint, and public-data request. It serves two users:

1. IBL developers can inspect `release_manifest/` to identify the EIDs, PIDs,
   trial ranges, ONE objects, spike-sorting objects, and external assets needed
   for public release.
2. Readers can use immutable profiles in `configs/figures/` instead of editing
   commented configuration blocks or machine-specific paths.

## Quick start

List every profile, status, run count, and resolved cohort size:

```bash
python3 scripts/figure_registry.py
```

Preview the exact runs for a figure without loading IBL data:

```bash
python3 scripts/reproduce.py figure1_snr_glmhmm --dry-run
python3 scripts/reproduce.py supp1_glmhmm_states --dry-run
```

Run a profile whose status is `ready`:

```bash
python3 scripts/reproduce.py figure1_snr_glmhmm \
  --glmhmm-dir /path/to/GLM-HMM \
  --one-cache-dir /path/to/ONE/cache \
  --output-root /path/to/outputs
```

Each profile supplies its own `figure_prefix`; runtime paths come from command
line arguments. Multi-run profiles execute every member by default, or one can
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
guarded until its remaining absolute paths are adapted and numerical outputs
are regression-tested. See `ephys_analysis/README.md` and `MIGRATION_PLAN.md`.
