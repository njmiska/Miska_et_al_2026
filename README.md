# Bilateral Optogenetics Analysis Pipeline

Analyzes behavioral data from the IBL 2-AFC task with bilateral optogenetic stimulation delivered via implanted fiber optic cannulae. Supports multiple brain regions, opsins, and two analysis modes.

## Overview

This pipeline consolidates two previously separate analysis scripts:

1. **Traditional mode** (`USE_GLMHMM = False`): Excludes trials and sessions based on reaction time, high-contrast performance, and baseline bias shift thresholds.

2. **GLM-HMM mode** (`USE_GLMHMM = True`): Uses pre-computed GLM-HMM state labels to isolate trials where the mouse was in a specific engagement state (e.g., "engaged" vs "disengaged"), then applies the same downstream analyses.

Both modes produce the same outputs: psychometric curves, bias shift quantification, wheel movement trajectories, and reaction time comparisons for stim vs. nonstim trials.

## File Structure

```
opto_analysis/
├── config.py              # All configurable settings (paths, thresholds, GLM-HMM toggle)
├── helpers.py             # Reusable helper functions
├── metadata_all.py        # Session metadata (EIDs, trials ranges, experimental parameters)
├── opto_analysis.py       # Main analysis pipeline
└── README.md              # This file
```

## Quick Start

1. **Update paths** in `config.py` to match your local setup (especially `BASE_DIR`, `FIGURE_SAVE_PATH`, and `GLMHMM_BASE_DIR`).

2. **Set session filters** in `config.py` under `SESSION_FILTERS` to select the experiments you want to analyze (by mouse, brain region, opsin, etc.).

3. **Choose analysis mode**: set `USE_GLMHMM = True` or `False`.

4. **Run**:
   ```bash
   python opto_analysis.py
   ```

## Configuration Reference

### Key Settings in `config.py`

| Setting | Description | Default |
|---------|-------------|---------|
| `USE_GLMHMM` | Toggle GLM-HMM state filtering | `False` |
| `N_STATES` | Number of GLM-HMM states (2 or 4) | `2` |
| `STATE_TYPE` | Which state to keep ('engaged', 'disengaged', 'bypass') | `'engaged'` |
| `STATE_DEF` | State label timing ('current' or 'previous') | `'current'` |
| `BASELINE_PERFORMANCE_THRESHOLD` | Min accuracy at ±100% and ±25% contrast | `0.7` |
| `MIN_BIAS_THRESHOLD` | Min summed nonstim bias shift to include session | `0.3` |
| `RT_THRESHOLD` | Max reaction time per trial (seconds) | `30` |
| `MIN_NUM_TRIALS` | Min trial count to include session | `300` |
| `ONLY_INCLUDE_BIAS_TRIALS` | Restrict to trials after first block switch (>89) | `True` |

### Session Filters

Filter sessions by any metadata field. Use `None` to skip, a value for exact match, or a lambda for custom logic:

```python
SESSION_FILTERS = {
    'Mouse_ID': lambda x: x in ['SWC_NM_053', 'SWC_NM_073'],
    'Brain_Region': 'VLS',
    'Opsin': 'GtACR2',
    'Hemisphere': 'both',
    'Stimulation_Params': 'QPRE',
    'Pulse_Params': 'cont',
}
```

### GLM-HMM Mode Notes

When `USE_GLMHMM = True`:
- State labels are loaded from pre-computed pickle files.
- `STATE_TYPE` controls which trials are kept:
  - `'engaged'`: States 2 and 3 (2-state model) or states 1 and 3 (4-state model).
  - `'disengaged'`: States 1 and 4 (2-state model) or states 2 and 4 (4-state model).
  - `'bypass'`: No state filtering (uses all trials — useful for comparison).
- `STATE_DEF = 'previous'` shifts state indices by +1, using the state on the trial *before* each stim trial.
- Performance thresholds can be relaxed (e.g., set to 0) since GLM-HMM filtering handles engagement.

## Pipeline Flow

```
For each session:
  1. Load trials + wheel from IBL database
  2. Load laser intervals (or legacy taskData)
  3. [GLM-HMM] Check state labels exist
  4. Determine valid trials range
  5. Compute reaction times
  6. Identify stim vs nonstim trials (by laser intervals or opto flag)
  7. [Optional] Shift to trials-after-stim
  8. [GLM-HMM] Filter by engagement state
  9. Session quality checks (performance, bias threshold)
  10. Compute bias shift (psychometric fit per block, stim and nonstim)
  11. Extract wheel trajectories
  12. Concatenate across sessions

Post-loop:
  - Paired t-tests on per-session bias shifts
  - RT comparison (Mann-Whitney U)
  - Psychometric curves, bias shift plots, wheel plots, RT histograms
```

## Outputs

- **Psychometric curves**: Side-by-side stim vs nonstim, split by block (pL=0.8 vs pL=0.2).
- **Bias shift comparison**: Paired per-session plot with statistics.
- **Wheel trajectories**: Mean ± SEM by block and stim condition.
- **RT distributions**: Histogram overlay of stim vs nonstim reaction times.
- **Console summary**: Per-session log, aggregate statistics, session acceptance/rejection reasons.

## Helper Functions

Key functions in `helpers.py`:

| Function | Description |
|----------|-------------|
| `load_session_data()` | Load trials and wheel from IBL database |
| `identify_stim_nonstim_trials()` | Classify trials as stim/nonstim (supports QPRE, SORE, legacy) |
| `filter_trials_by_state()` | Apply GLM-HMM state filtering |
| `subset_bunch()` | Slice all Bunch attributes by index (replaces manual 15-attribute subsetting) |
| `concat_bunches()` | Concatenate two Bunch objects |
| `compute_bias_shift()` | Psychometric fit + bias computation across contrasts |
| `extract_wheel_trajectory()` | Extract per-trial wheel movement aligned to event |
| `check_session_performance()` | Validate high-contrast accuracy |

## Troubleshooting

### "Failed to load eid"
- Check network connection to IBL servers.
- Verify EID exists in Alyx database.

### "No GLM-HMM states found"
- Verify the mouse/EID has entries in `all_subject_states.csv`.
- Check file paths in config (`GLMHMM_STATES_FILE`).

### "No sessions met criteria"
- Review `SESSION_FILTERS` in `config.py`.
- Temporarily lower quality thresholds to diagnose.
- Check that metadata entries in `metadata_all.py` match your filter values.

### Bias shift values differ from old pipeline
- The NaN-checking bug in the original script (lines 1057-1082) has been fixed.
  Previously, all NaN checks tested `biasshift_L100` regardless of which variable
  was just computed. This may cause small differences in sessions where non-100%
  contrast bins had missing data.

## Dependencies

- `numpy`, `pandas`, `matplotlib`, `seaborn`, `scipy`
- `psychofit` (`pip install Psychofit`)
- `ibllib` (IBL Python tools)
- `ONE-api` (IBL data access)

## Authors

**Nate Miska**

Developed with AI pair-programming assistance (Claude, Anthropic) for code refactoring and documentation.
