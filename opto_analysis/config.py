"""
Configuration settings for optogenetics analysis pipeline (Miska et al. 2026).

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
STATE_DEF = 'previous'

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
BASELINE_PERFORMANCE_THRESHOLD = 0.8

# Minimum performance on STIM trials at high contrasts
STIM_PERFORMANCE_THRESHOLD = 0.8

# Minimum number of trials required to include session
MIN_NUM_TRIALS = 0

# Minimum baseline (nonstim) bias shift to include session
# (summed across all contrasts; ensures the mouse is block-engaged)
MIN_BIAS_THRESHOLD = 1

# Maximum reaction time to include trial (seconds)
# Set high (e.g. 100) when using GLM-HMM to avoid double-filtering
RT_THRESHOLD = 100

# Minimum number of stim trials required after all filtering to include session
MIN_STIM_TRIALS = 20

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
FIGURE_PREFIX = 'ZI_all'

# Title text for plots (set automatically from session filters if None)
TITLE_TEXT = None

# Whether to use publication-quality formatting
PLOT_FOR_PAPER = True

# Flag to show failed session loads in console
FLAG_FAILED_LOADS = True

# Psychometric plot layout: True = overlay stim (dashed) on nonstim in a single axis
#                           False = side-by-side axes for stim and nonstim
PSYCHOMETRIC_OVERLAY = True

# Psychometric data source: True = average each mouse's psychometric points (mean-of-means)
#                           False = pool all trials across mice (single fit)
PSYCHOMETRIC_MEAN_OF_MICE = True

# =============================================================================
# SESSION SELECTION CRITERIA
# =============================================================================
# These filters are passed to find_sessions_by_advanced_criteria()
# Use None to not filter on that criterion
# Use a specific value for exact match
# Use a lambda for custom filtering, e.g.: lambda x: x in ['val1', 'val2']

# SESSION_FILTERS = {

#     'Stimulation_Params': 'QPRE',
#     # Stimulation timing: 'QPRE', 'SORE', 'QP', 'ITI', or lambda

#     'Mouse_ID': 'SWC_NM_102',
#     # e.g., 'SWC_NM_099' or lambda x: x in [...]

#     'Hemisphere': 'both',
#     # e.g., 'both', 'left', 'right'

#     'Pulse_Params': 'cont',
#     # e.g., 'cont', '50hz', '20hz', 'cont_c', 'motor_bilateral_mask'

#     'Opsin': None,
#     # e.g., 'ChR2', 'GtACR2', or lambda x: x in ['ChR2', 'GtACR2']

#     'Genetic_Line': None,
#     # e.g., 'VGAT-ChR2', 'D1-Cre'

#     'Brain_Region': None,
#     # e.g., 'VLS', 'SNr', 'STN', 'ZI', 'motor_bilateral'

#     'Laser_V': None,
#     # e.g., 2, or lambda x: x >= 1

#     'Date': None,
#     # e.g., '2024-10-24'

#     'EID': None,
#     # Specific session EID(s)
# }


# ## SNr defaults:
# SESSION_FILTERS = {

#     'Stimulation_Params': lambda x: x in ['QPRE', 'QPRE*'],
#     # Stimulation timing: 'QPRE', 'SORE', 'QP', 'ITI', or lambda

#     'Mouse_ID': lambda x: x in ['SWC_NM_004', 'SWC_NM_008', 'SWC_NM_011', 'SWC_NM_012', 'SWC_NM_018', 'SWC_NM_016', 'SWC_NM_080', 'SWC_NM_096', 'SWC_NM_113'],
#     # e.g., 'SWC_NM_099' or lambda x: x in [...]

#     'Hemisphere': None,
#     # e.g., 'both', 'left', 'right'

#     'Pulse_Params': 'cont',
#     # e.g., 'cont', '50hz', '20hz', 'cont_c', 'motor_bilateral_mask'

#     'Opsin': None,
#     # e.g., 'ChR2', 'GtACR2', or lambda x: x in ['ChR2', 'GtACR2']

#     'Genetic_Line': None,
#     # e.g., 'VGAT-ChR2', 'D1-Cre'

#     'Brain_Region': 'SNr',
#     # e.g., 'VLS', 'SNr', 'STN', 'ZI', 'motor_bilateral'

#     'Laser_V': None,
#     # e.g., 2, or lambda x: x >= 1

#     'Date': None,
#     # e.g., '2024-10-24'

#     'EID': None,
#     # Specific session EID(s)
# }


## ZI defaults:
SESSION_FILTERS = {

    'Stimulation_Params': lambda x: x in ['QPRE', 'QPRE*'],
    # Stimulation timing: 'QPRE', 'SORE', 'QP', 'ITI', or lambda

    'Mouse_ID': lambda x: x in ['SWC_NM_003', 'SWC_NM_010', 'SWC_NM_022', 'SWC_NM_104', 'SWC_NM_111'],
    # e.g., 'SWC_NM_099' or lambda x: x in [...]

    'Hemisphere': None,
    # e.g., 'both', 'left', 'right'

    'Pulse_Params': 'cont',
    # e.g., 'cont', '50hz', '20hz', 'cont_c', 'motor_bilateral_mask'

    'Opsin': None,
    # e.g., 'ChR2', 'GtACR2', or lambda x: x in ['ChR2', 'GtACR2']

    'Genetic_Line': None,
    # e.g., 'VGAT-ChR2', 'D1-Cre'

    'Brain_Region': 'ZI',
    # e.g., 'VLS', 'SNr', 'STN', 'ZI', 'motor_bilateral'

    'Laser_V': None,
    # e.g., 2, or lambda x: x >= 1

    'Date': None,
    # e.g., '2024-10-24'

    'EID': None,
    # Specific session EID(s)
}

# ## D1 defaults:
# SESSION_FILTERS = {

#     'Stimulation_Params': lambda x: x in ['QPRE', 'QPRE*'],

#     'Mouse_ID': lambda x: x in ['SWC_NM_053', 'SWC_NM_038', 'SWC_NM_098', 'SWC_NM_099', 'SWC_NM_100', 'SWC_NM_101', 'SWC_NM_073', 'SWC_NM_105' ],

#     'Hemisphere': None,

#     'Pulse_Params': lambda x: x in ['50hz', 'cont_c'],

#     'Opsin': None,

#     'Genetic_Line': 'D1-Cre',

#     'Brain_Region': 'VLS',

#     'Laser_V': None,

#     'Date': None,

#     'EID': None,
# }


# ## D2 defaults:
# SESSION_FILTERS = {

#     'Stimulation_Params': lambda x: x in ['QPRE', 'QPRE*'],

#     'Mouse_ID': lambda x: x in ['SWC_NM_065', 'SWC_NM_109', 'SWC_NM_108', 'SWC_NM_089', 'SWC_NM_087', 'SWC_NM_083', 'SWC_NM_107'],#, 'SWC_NM_043'], ###43, 88

#     'Hemisphere': None,

#     'Pulse_Params': lambda x: x in ['50hz', 'cont_c'],

#     'Opsin': 'ChR2',

#     'Genetic_Line': None,
#     # e.g., 'VGAT-ChR2', 'D1-Cre'

#     'Brain_Region': 'VLS',

#     'Laser_V': None,

#     'Date': None,

#     'EID': None,
# }


# ## STN defaults:
# SESSION_FILTERS = {

#     'Stimulation_Params': lambda x: x in ['QPRE', 'QPRE*'],

#     'Mouse_ID': lambda x: x in ['SWC_NM_024', 'SWC_NM_025', 'SWC_NM_026'],

#     'Hemisphere': None,

#     'Pulse_Params': 'cont',

#     'Opsin': None,

#     'Genetic_Line': None,

#     'Brain_Region': 'STN',

#     'Laser_V': None,

#     'Date': None,

#     'EID': None,
# }


### Zapit aMOs
# SESSION_FILTERS = {

#     'Stimulation_Params': 'zapit',
#     # Stimulation timing: 'QPRE', 'SORE', 'QP', 'ITI', or lambda

#     'Mouse_ID': None,
#     # e.g., 'SWC_NM_099' or lambda x: x in [...]

#     'Hemisphere': None,
#     # e.g., 'both', 'left', 'right'

#     'Pulse_Params': lambda x: x in ['aMOs-4point', 'AnterolateralM2', 'FrontBackMedial', 'FrontBackLateral', 'AnterolateralM2_3pointweak', 'FrontBack', 'aMOs-targeted'],
#     # e.g., 'cont', '50hz', '20hz', 'cont_c', 'motor_bilateral_mask'

#     'Opsin': None,
#     # e.g., 'ChR2', 'GtACR2', or lambda x: x in ['ChR2', 'GtACR2']

#     'Genetic_Line': None,
#     # e.g., 'VGAT-ChR2', 'D1-Cre'

#     'Brain_Region': None,
#     # e.g., 'VLS', 'SNr', 'STN', 'ZI', 'motor_bilateral'

#     'Laser_V': None,
#     # e.g., 2, or lambda x: x >= 1

#     'Date': None,
#     # e.g., '2024-10-24'

#     'EID': None,
#     # Specific session EID(s)
# }
