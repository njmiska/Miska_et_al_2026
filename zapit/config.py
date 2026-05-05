"""
Configuration settings for Zapit optogenetics analysis.

This file contains all configurable parameters including:
- File paths (adjust these to your local setup)
- GLM-HMM settings (toggle and parameters)
- Analysis thresholds
- Plotting options
- Session selection filters

Toggle USE_GLMHMM to switch between:
  - True:  Use GLM-HMM state labels to isolate engaged/disengaged trials
  - False: Use traditional accuracy, bias shift, and RT exclusion criteria
"""

import numpy as np
from pathlib import Path

# =============================================================================
# FILE PATHS - Adjust these to your local setup
# =============================================================================

# Base directory for the project (update this to your local path)
BASE_DIR = Path('/Users/natemiska/python/zapit')

# Zapit trial log file
ZAPIT_TRIALS_LOG = BASE_DIR / 'zapit_trials.yml'

# Zapit stimulation locations log (contains AP/ML coordinates)
ZAPIT_LOCATIONS_LOG = BASE_DIR / 'zapit_log.yml'

# Allen CCF atlas data (download from Allen Institute)
# See: https://download.alleninstitute.org/informatics-archive/current-release/mouse_ccf/
ALLEN_CCF_ANNOTATION = Path('/Users/natemiska/python/Allen/annotation_volume_10um.npy')
ALLEN_STRUCTURE_TREE = Path('/Users/natemiska/python/Allen/structure_tree_safe_2017.csv')

# Output directory for figures
FIGURE_SAVE_PATH = Path('/Users/natemiska/Desktop/zapit_check')

# =============================================================================
# GLM-HMM SETTINGS
# =============================================================================
# When USE_GLMHMM = True, trials are filtered using pre-computed GLM-HMM state
# labels to isolate engaged (or disengaged) trials. When False, only the
# traditional quality criteria below (accuracy, bias shift, RT, etc.) are used.
#
# When using GLM-HMM, you may want to relax the traditional thresholds below
# (BASELINE_PERFORMANCE_THRESHOLD, MIN_BIAS_THRESHOLD_ZAPIT, MIN_NUM_TRIALS)
# since engagement filtering substitutes for performance gating.

# Master toggle: True = filter trials by GLM-HMM state, False = traditional criteria
USE_GLMHMM = True

# Number of hidden states in the model (2 or 4)
N_STATES = 2

# Which state(s) to keep for analysis:
#   2-state: 'engaged', 'disengaged', 'bypass' (bypass = no filtering)
#   4-state: 'state1', 'state2', 'state3', 'state4', 'engaged', 'disengaged', 'bypass'
STATE_TYPE = 'engaged'

# How to assign state labels to stim trials:
#   'current'  = use the state label on the stim trial itself
#   'previous' = use the state label on the trial before stim (shifts indices +1)
STATE_DEF = 'previous'

# File paths for GLM-HMM state data
GLMHMM_BASE_DIR = Path('/Users/natemiska/int-brain-lab/GLM-HMM')
GLMHMM_STATES_FILE = GLMHMM_BASE_DIR / 'all_subject_states.csv'
GLMHMM_ENGAGED_PREV_FILE = GLMHMM_BASE_DIR / 'engaged_prevtrial_indices.pkl'
GLMHMM_DISENGAGED_PREV_FILE = GLMHMM_BASE_DIR / 'disengaged_prevtrial_indices.pkl'


# =============================================================================
# SESSION FILTERING THRESHOLDS
# =============================================================================

# Minimum performance at high contrasts (100%, 25%) to include session
# Note that this is stricter when using GLM-HMM
BASELINE_PERFORMANCE_THRESHOLD = 0.8 #0.79

# Minimum performance on stim trials at high contrasts
STIM_PERFORMANCE_THRESHOLD = 0.8 #0.5

# Minimum number of trials required to include session
MIN_NUM_TRIALS = 150 #300

# Minimum baseline bias shift to include Zapit session
MIN_BIAS_THRESHOLD_ZAPIT = 1.0

# Maximum reaction time to include trial (seconds)
# Set high (e.g. 100) when using GLM-HMM to avoid double-filtering
RT_THRESHOLD = 100 #30

# =============================================================================
# ANALYSIS OPTIONS
# =============================================================================

# Number of stimulation locations in Zapit grid
NUM_STIM_LOCATIONS = 52

# Number of trials to average for bias assessment
TRIALS_PER_DATAPOINT = 10

# Whether to use trials immediately after stim (for lagged analysis)
USE_TRIALS_AFTER_STIM = False

# Bias-reduction colormap range (used by the bias reduction heatmap)
# Reductions are typically small fractions of control, so a tight upper bound
# (e.g. 0.4) keeps the colormap dynamic range usefully spread across observed
# values. Lower the upper bound for datasets with subtler effects.
#   vmin = darkest blue end (negative = bias enhanced by stim, rare)
#   vmax = darkest red end  (positive = bias eliminated)
BIAS_REDUCTION_VMIN = -0.2
BIAS_REDUCTION_VMAX = 0.4

# Multiple-comparisons correction applied to the per-condition p-values used
# by the RT, QP, Bias-Reduction, and Lapse heatmaps.
#   'fdr_bh'      : Benjamini-Hochberg FDR (recommended default)
#   'bonferroni'  : strict family-wise correction (loses power)
#   'holm'        : Holm step-down family-wise correction
#   'fdr_by'      : Benjamini-Yekutieli (FDR under arbitrary dependence)
#    None         : no correction (raw p-values used everywhere)
#
# Both raw and corrected p-values are saved to {prefix}_results.csv. The
# heatmaps use the *corrected* values for circle size and the p<0.05 alpha
# threshold, so the visual story stays calibrated to the chosen FWER/FDR.
MULTIPLE_COMPARISONS_CORRECTION = 'fdr_bh'

# =============================================================================
# WHEEL ANALYSIS OPTIONS
# =============================================================================

# Duration of wheel movement to analyze (seconds)
WHEEL_ANALYSIS_DURATION = 10

# Time bin interval for wheel analysis (seconds)
WHEEL_TIME_INTERVAL = 0.1

# Alignment point for wheel analysis: 'QP', 'goCue', 'goCue_pre', 'feedback'
WHEEL_ALIGN_TO = 'QP'

# Whether to only include low contrast trials in wheel analysis
ONLY_LOW_CONTRASTS = False

# Threshold for defining "low" contrast (%)
LOW_CONTRAST_THRESHOLD = 13

# =============================================================================
# PLOTTING OPTIONS
# =============================================================================

# Whether to save figures to disk
SAVE_FIGURES = True

# Prefix for saved figure filenames
FIGURE_PREFIX = 'zapit_fdr'

# Flag to show failed session loads in console
FLAG_FAILED_LOADS = True

# =============================================================================
# CONDITION-SELECTED PSYCHOMETRIC PLOTS
# =============================================================================
# Generate psychometric curves comparing pooled trials at user-selected stim
# conditions versus pooled control trials. Useful for illustrating bias-shift
# reduction at specific cortical points (e.g. MOs cluster) in manuscript figures.
#
# Set to None or [] to disable. Otherwise, two formats are accepted:
#
#   - Flat list -> one figure pooling all listed conditions:
#         PSYCHOMETRIC_STIM_CONDITIONS = [16, 22]
#
#   - List of lists -> one figure per inner list:
#         PSYCHOMETRIC_STIM_CONDITIONS = [
#             [16, 22],          # MOs pair
#             [3, 4, 5, 11, 12], # Larger MOs region
#         ]
#
# For interactive iPython exploration after running the script, you can also
# call zapit_helpers.plot_zapit_psychometric(conds, condition_data) directly.

PSYCHOMETRIC_STIM_CONDITIONS = [[32], [27,28]]

# Optional matched titles for each figure (same length as outer list above).
# If None, an auto-generated title using condition numbers is used.
PSYCHOMETRIC_TITLES = ['MOs_32','MOp_27_28']

# Plot layout: True = overlay stim/control on a single axis (stim = dashed)
#              False = side-by-side panels for control and stim
PSYCHOMETRIC_OVERLAY = True

# Psychometric fit parameters (passed to psy.mle_fit_psycho)
PSYCHO_FIT_KWARGS = {
    'parmin': np.array([-50., 10., 0., 0.]),
    'parmax': np.array([50., 100., 1., 1.]),
    'parstart': np.array([0., 40., 0.1, 0.1]),
    'nfits': 50,
}

# =============================================================================
# CONTRAST LEVELS
# =============================================================================

# All contrast levels used in the task (%)
ALL_CONTRASTS = [-100.0, -25.0, -12.5, -6.25, 0.0, 6.25, 12.5, 25.0, 100.0]

# Low contrast levels for focused analysis (%)
LOW_CONTRASTS = [-12.5, -6.25, 0.0, 6.25, 12.5]

# Contrasts used for bias heatmap (typically low contrasts)
BIAS_HEATMAP_CONTRASTS = [-6.25, 0, 6.25]

# =============================================================================
# IBL API CONFIGURATION
# =============================================================================

# IBL Alyx database URL
ALYX_BASE_URL = 'https://alyx.internationalbrainlab.org'


# =============================================================================
# SESSION SELECTION CRITERIA
# =============================================================================
# These filters are passed to find_sessions_by_advanced_criteria()
# Use None to not filter on that criterion
# Use a specific value for exact match
# Use a lambda for custom filtering, e.g.: lambda x: x in ['val1', 'val2']

SESSION_FILTERS = {
    'Stimulation_Params': 'zapit',      # Required for Zapit analysis
    'Mouse_ID': None,                    # e.g., 'SWC_NM_099' or lambda x: x in [...]
    'Hemisphere': None,                  # e.g., 'both', 'left', 'right'
    'Pulse_Params': None,                # e.g., 'motor_bilateral_mask', '50hz'
    'Opsin': None,                       # e.g., 'ChR2', 'GtACR2'
    'Genetic_Line': None,                # e.g., 'VGAT-ChR2', 'D1-Cre'
    'Brain_Region': 'motor_bilateral',                # e.g., 'motor_bilateral'
    'Laser_V': None,                     # e.g., 2, or lambda x: x >= 1
    'Date': None,                        # e.g., '2024-10-24'
    'EID': None,                         # Specific session EID(s)
}


# =============================================================================
# PLOTTING CONFIGURATION
# =============================================================================

# Brain atlas display settings
#   BRAIN_BACKGROUND_COLOR controls the figure background ('white', 'black',
#     or any matplotlib colour). Border colour and axis text colour are
#     auto-chosen to contrast (black-on-white or white-on-black). Override
#     by setting BRAIN_BORDER_COLOR explicitly.
BRAIN_BACKGROUND_COLOR = 'white'
BRAIN_BORDER_COLOR = 'auto'   # 'auto' = pick contrasting colour from bg,
                              # or set explicitly e.g. 'black', 'white', 'grey'

# Bregma position in CCF coordinates (for 10um resolution)
# Format: [AP, DV, ML] - matching original script
BREGMA_CCF = np.array([540, 0, 570])

# Scale factor (negative to handle coordinate flip)
CCF_SCALE_FACTOR = -100

# Axis limits for plotting (mm from Bregma)
# Set to None to auto-scale, or specify (min, max).
# Negative ML = right hemisphere; positive ML = left hemisphere.
# Examples:
#   Right hemisphere only:    PLOT_ML_LIMITS = (0, 3.7)
#   Both hemispheres:         PLOT_ML_LIMITS = (-3.7, 3.7)
PLOT_AP_LIMITS = (-1.2, 3.8)      # AP extent (negative = posterior)
PLOT_ML_LIMITS = (-3.7, 3.7)      # ML extent (both hemispheres)

# Whether to show only right hemisphere data points.
# This toggle is only used as a fallback when PLOT_ML_LIMITS is None.
# If PLOT_ML_LIMITS is set explicitly (as above), the explicit limits win.
SHOW_RIGHT_HEMISPHERE_ONLY = False
