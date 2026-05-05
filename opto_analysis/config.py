"""
Configuration settings for bilateral optogenetics analysis pipeline.

This file contains all configurable parameters including:
- File paths (adjust these to your local setup)
- Analysis thresholds and options
- GLM-HMM settings (toggle and parameters)
- Wheel analysis options
- Plotting options
- Session selection filters

Toggle USE_GLMHMM to switch between:
  - True:  Use GLM-HMM state labels to isolate engaged/disengaged trials
  - False: Use traditional RT, performance, and bias exclusion criteria
"""

import numpy as np
from pathlib import Path


# =============================================================================
# FILE PATHS — Adjust these to your local setup
# =============================================================================

# Base directory for helper scripts
BASE_DIR = Path('/Users/natemiska/python/opto_analysis')

# Output directory for figures
FIGURE_SAVE_PATH = Path('/Users/natemiska/Desktop/opto_figures')

# IBL Alyx database URL
ALYX_BASE_URL = 'https://alyx.internationalbrainlab.org'


# =============================================================================
# GLM-HMM SETTINGS
# =============================================================================

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
STATE_DEF = 'current'

# File paths for GLM-HMM state data
GLMHMM_BASE_DIR = Path('/Users/natemiska/int-brain-lab/GLM-HMM')
GLMHMM_STATES_FILE = GLMHMM_BASE_DIR / 'all_subject_states.csv'
GLMHMM_ENGAGED_PREV_FILE = GLMHMM_BASE_DIR / 'engaged_prevtrial_indices.pkl'
GLMHMM_DISENGAGED_PREV_FILE = GLMHMM_BASE_DIR / 'disengaged_prevtrial_indices.pkl'


# =============================================================================
# SESSION FILTERING THRESHOLDS
# =============================================================================

# Minimum performance at high contrasts (100%, 25%) on NONSTIM trials to include session
# Set to 0 when using GLM-HMM (engagement filtering replaces performance gating)
BASELINE_PERFORMANCE_THRESHOLD = 0

# Minimum performance on STIM trials at high contrasts
STIM_PERFORMANCE_THRESHOLD = 0

# Minimum number of trials required to include session
MIN_NUM_TRIALS = 0

# Minimum baseline (nonstim) bias shift to include session
# (summed across all contrasts; ensures the mouse is block-engaged)
MIN_BIAS_THRESHOLD = 0.5

# Maximum reaction time to include trial (seconds)
# Set high (e.g. 100) when using GLM-HMM to avoid double-filtering
RT_THRESHOLD = 100

# Minimum number of stim trials required after all filtering to include session
MIN_STIM_TRIALS = 10

# Whether to only include trials after first block switch (trial > 89)
ONLY_INCLUDE_BIAS_TRIALS = True


# =============================================================================
# ANALYSIS OPTIONS
# =============================================================================

# Number of trials to average for rolling bias assessment
TRIALS_PER_DATAPOINT = 5

# Whether to analyze trials immediately AFTER stim (for lagged analysis)
USE_TRIALS_AFTER_STIM = False

# Whether to subsample stim trials to balance prev-choice × block conditions
SUBSAMPLE_TRIALS_AFTER_STIM = False

# Which contrasts to use when computing bias shift comparison
# 'all' = all 9 contrasts, 'high' = only ±100 and ±25, 'low' = only ±12.5, ±6.25, 0
BIAS_COMPARISON_CONTRASTS = 'all'


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
# CONTRAST LEVELS
# =============================================================================

# All contrast levels used in the task (%)
ALL_CONTRASTS = [-100.0, -25.0, -12.5, -6.25, 0.0, 6.25, 12.5, 25.0, 100.0]

# Low contrast levels for focused analysis (%)
LOW_CONTRASTS = [-12.5, -6.25, 0.0, 6.25, 12.5]

# High contrast levels
HIGH_CONTRASTS = [-100.0, -25.0, 25.0, 100.0]

# Psychometric fit parameters (for psychofit MLE fitting)
PSYCHO_FIT_KWARGS = {
    'parmin': np.array([-50., 10., 0., 0.]),
    'parmax': np.array([50., 100., 1., 1.]),
    'parstart': np.array([0., 40., 0.1, 0.1]),
    'nfits': 50,
}


# =============================================================================
# PLOTTING OPTIONS
# =============================================================================

# Whether to save figures to disk
SAVE_FIGURES = True

# Prefix for saved figure filenames
FIGURE_PREFIX = 'opto'

# Title text for plots (set automatically from session filters if None)
TITLE_TEXT = None

# Whether to use publication-quality formatting
PLOT_FOR_PAPER = True

# Flag to show failed session loads in console
FLAG_FAILED_LOADS = True


# =============================================================================
# SESSION SELECTION CRITERIA
# =============================================================================
# These filters are passed to find_sessions_by_advanced_criteria()
# Use None to not filter on that criterion
# Use a specific value for exact match
# Use a lambda for custom filtering, e.g.: lambda x: x in ['val1', 'val2']

SESSION_FILTERS = {

    'Stimulation_Params': 'QPRE',
    # Stimulation timing: 'QPRE', 'SORE', 'QP', 'ITI', or lambda

    'Mouse_ID': lambda x: x in ['SWC_NM_004', 'SWC_NM_008', 'SWC_NM_011', 'SWC_NM_012', 'SWC_NM_018', 'SWC_NM_016', 'SWC_NM_080', 'SWC_NM_102'],
    # e.g., 'SWC_NM_099' or lambda x: x in [...]

    'Hemisphere': None,
    # e.g., 'both', 'left', 'right'

    'Pulse_Params': 'cont',
    # e.g., 'cont', '50hz', '20hz', 'cont_c', 'motor_bilateral_mask'

    'Opsin': None,
    # e.g., 'ChR2', 'GtACR2', or lambda x: x in ['ChR2', 'GtACR2']

    'Genetic_Line': None,
    # e.g., 'VGAT-ChR2', 'D1-Cre'

    'Brain_Region': 'SNr',
    # e.g., 'VLS', 'SNr', 'STN', 'ZI', 'motor_bilateral'

    'Laser_V': None,
    # e.g., 2, or lambda x: x >= 1

    'Date': None,
    # e.g., '2024-10-24'

    'EID': None,
    # Specific session EID(s)
}
