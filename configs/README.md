# Manuscript analysis profiles

Each JSON file in `configs/figures/` freezes one manuscript analysis cohort and
its major settings. Profiles are the source of truth for session filters,
GLM-HMM indexing, thresholds, output prefixes, and repeated run matrices.

Core fields:

- `profile_id`, `figure`, and `label`: stable manuscript identity.
- `pipeline` and `entrypoint`: source used for the analysis.
- `status`: whether raw execution is enabled or which dependency is missing.
- `metadata_source` and `selection`: cohort source and exact filters.
- `glmhmm`: applicability, enabled state, state count/type, and state timing.
- `parameters`: named legacy config values overridden for this run.
- `output.figure_prefix`: collision-free output filename prefix.
- `runs`: optional run matrix with selection, GLM-HMM, parameter, or output
  overrides.

The generic command consumes these profiles without changing analysis logic:

```bash
python3 scripts/reproduce.py figure3_stn_glmhmm --dry-run
python3 scripts/reproduce.py supp1_hemisphere_specific --run right --dry-run
```

Profiles with `status=ready` can run through the adapter. Other statuses are
intentional blockers rather than silent assumptions:

- `ready_source_snapshot`: source and metadata are present, but public raw
  execution awaits portability/regression work.
- `awaiting_cohort_metadata`: analysis source exists but its final cohort is not
  frozen.
- `awaiting_colleague_code`: the producing source and data inventory are not yet
  available.

For trial-filtering analyses, the final contract is two-state engaged GLM-HMM
indexing with `state_definition=previous`, except Figure 2 and the explicit
NOGLMHMM sensitivity profile. The four-state supplemental profile overrides
`N_STATES=4` and runs `STATE_TYPE=state1` through `state4`.
