# Zapit Optogenetics Analysis Pipeline

Analysis pipeline for behavioral data from the IBL task with laser scanning optogenetic inhibition (Zapit system).

## Overview

Mice perform a two-alternative forced choice (2-AFC) task with prior blocks while receiving bilateral optogenetic stimulation at one of 52 cortical locations. This pipeline analyzes the behavioral effects of stimulation at each location.

### Key Analyses
- **Reaction time effects**: How stimulation at each location affects reaction times
- **Bias shift effects**: How stimulation modulates block-induced choice bias
- **Lapse rate changes**: Effects on high-contrast performance
- **Brain heatmaps**: Visualization of effects on Allen CCF brain atlas

## Installation

### Prerequisites
1. Python 3.8+
2. IBL environment (for data access): https://docs.internationalbrainlab.org/02_installation.html

### Required packages
```bash
pip install numpy pandas matplotlib seaborn scipy statsmodels
pip install ONE-api ibllib  # IBL data access
pip install psychofit  # Psychometric fitting
```

### Allen CCF Atlas Data
Download the annotation volume from the Allen Institute:
1. Go to: https://download.alleninstitute.org/informatics-archive/current-release/mouse_ccf/
2. Download `annotation_volume_10um.npy`
3. Download `structure_tree_safe_2017.csv`
4. Update paths in `config.py`

## File Structure

```
zapit_analysis/
├── zapit_analysis.py    # Main analysis script
├── zapit_helpers.py     # Helper functions module
├── config.py            # Configuration and paths
├── metadata_zapit.py    # Session metadata
└── README.md            # This file
```

## Configuration

Edit `config.py` to set:

### File Paths
```python
BASE_DIR = Path('/your/path/to/project')
ZAPIT_TRIALS_LOG = BASE_DIR / 'zapit_trials.yml'
ALLEN_CCF_ANNOTATION = Path('/path/to/annotation_volume_10um.npy')
FIGURE_SAVE_PATH = Path('/path/to/output/figures')
```

### Analysis Parameters
```python
BASELINE_PERFORMANCE_THRESHOLD = 0.79  # Min accuracy to include session
MIN_BIAS_THRESHOLD_ZAPIT = 1.0         # Min bias shift to include session
RT_THRESHOLD = 30                       # Max RT to include trial (seconds)
```

## Usage

### Basic Usage
```bash
python zapit_analysis.py
```

### Selecting Sessions
Edit the session filter in `zapit_analysis.py`:
```python
eids, trials_ranges, MouseIDs, stim_params = find_sessions_by_advanced_criteria(
    sessions,
    Stimulation_Params='zapit',
    Mouse_ID='SWC_NM_099',  # Specific mouse
    # Hemisphere='both',
)
```

### Adding Sessions
Add new sessions to `metadata_zapit.py`:
```python
sessions = [
    {
        'Mouse_ID': 'SWC_NM_XXX',
        'Date': '2025-01-28',
        'EID': 'your-eid-here',
        'Hemisphere': 'both',
        'P_Opto': 0.75,
        'Stimulation_Params': 'zapit',
        'Pulse_Params': 'motor_bilateral_mask',
        'Laser_V': 2,
        'Opsin': 'ChR2',
        'Brain_Region': 'motor_bilateral',
        'Genetic_Line': 'VGAT-ChR2',
        'Trials_Range': 'ALL'  # or list(range(start, end))
    },
    # ... more sessions
]
```

## Outputs

### Figures (saved to `FIGURE_SAVE_PATH`)
- `{prefix}_RT_heatmap.png` - Reaction time effect sizes on brain atlas
- `{prefix}_bias_pval_heatmap.png` - Block bias p-values on brain atlas
- `{prefix}_effect_size_heatmap.png` - Bias effect sizes on brain atlas
- `{prefix}_summary_stats.png` - Pooled comparison bar plots

### Data Files
- `{prefix}_bias_vals_LC_control.npy` - Control bias values for further analysis
- `{prefix}_results.csv` - Summary statistics per condition

## Statistical Methods

### Bias Analysis
Two complementary approaches are used:

1. **By contrast level** (paired structure):
   - Computes bias (L block - R block choice probability) at each contrast
   - Mann-Whitney U test and paired t-test for each condition vs control
   - Appropriate for detecting systematic shifts across the psychometric curve

2. **By trial cycles** (independent samples):
   - Groups low-contrast trials into cycles of N trials
   - Independent t-test comparing control vs stim cycles
   - Provides independent samples for robust statistical inference

### RT Analysis
- Independent t-test (Welch's) comparing each condition to control
- Cohen's d effect size
- Lapse rate comparison using proportions z-test

## Helper Functions

Key functions in `zapit_helpers.py`:

| Function | Description |
|----------|-------------|
| `load_session_data()` | Load trials and wheel data from IBL |
| `parse_zapit_log()` | Parse stim location log file |
| `build_stim_location_dict()` | Map trial numbers to stim locations |
| `compute_bias_values_by_contrast()` | Compute bias at each contrast level |
| `compute_bias_values_by_cycle()` | Compute bias in trial cycles |
| `run_rt_analysis()` | Statistical comparison of reaction times |
| `generate_mip_with_borders()` | Create brain atlas projection |

## Troubleshooting

### "Failed to load eid"
- Check network connection to IBL servers
- Verify EID exists in Alyx database
- Try: `one.search(subject='mousename')` to find valid EIDs

### "No sessions met criteria"
- Check session filters in main script
- Verify sessions in metadata_zapit.py have correct Stimulation_Params
- Lower quality thresholds temporarily to diagnose

### Coordinate system issues in brain plots
- The CCF coordinate transformation may need adjustment
- Check `stim_locations` coordinates match your setup
- Modify scaling factors in plotting section as needed

## Citation

If using this code, please cite:
- The International Brain Laboratory (IBL)
- Allen Institute for Brain Science (Allen CCF)

## License

[Your license here]

## Contact

[Your contact information]
