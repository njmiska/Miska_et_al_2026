"""
Helper functions for Zapit optogenetics analysis.

This module contains reusable functions for:
- Data loading and parsing
- Trial data extraction
- Wheel analysis
- Bias calculations
- Statistical analysis
- Brain atlas visualization
"""

import re
import math
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest


# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================

def load_session_data(one, eid, flag_failed_loads=True):
    """
    Load trial and wheel data for a session from IBL database.
    
    Parameters
    ----------
    one : ONE
        ONE API instance
    eid : str
        Experiment ID
    flag_failed_loads : bool
        Whether to print warning on failed loads
        
    Returns
    -------
    trials : dict-like or None
        Trials object with all trial data
    wheel : dict-like or None
        Wheel object with position and timestamps
    """
    try:
        trials = one.load_object(eid, 'trials')
    except Exception:
        try:
            trials = one.load_object(eid, 'trials', 'alf')
        except Exception:
            if flag_failed_loads:
                print(f'Failed to load eid = {eid}')
            return None, None
    
    # Fix inconsistent trial object lengths (known IBL data issue)
    trials = fix_trials_length_inconsistency(trials)
    
    try:
        wheel = one.load_object(eid, 'wheel')
    except Exception:
        if flag_failed_loads:
            print(f'Failed to load wheel data for eid = {eid}')
        wheel = None
        
    return trials, wheel


def fix_trials_length_inconsistency(trials):
    """
    Fix inconsistent lengths in trials object by padding shorter arrays.
    
    This addresses a known issue where some trial attributes may be
    off by one element.
    
    Parameters
    ----------
    trials : dict-like
        Trials object
        
    Returns
    -------
    trials : dict-like
        Fixed trials object
    """
    lengths = [len(trials[k]) for k in trials.keys()]
    max_length = max(lengths)
    
    for k in trials.keys():
        if len(trials[k]) == max_length - 1:
            trials[k] = np.insert(trials[k], 0, 0)
            
    return trials


def load_laser_intervals(one, eid):
    """
    Load laser stimulation intervals for a session.
    
    Parameters
    ----------
    one : ONE
        ONE API instance
    eid : str
        Experiment ID
        
    Returns
    -------
    laser_intervals : ndarray
        Array of [onset, offset] times for each laser stimulation
    """
    try:
        # New format: collection is 'alf/task_00'
        laser_intervals = one.load_dataset(
            eid, '_ibl_laserStimulation.intervals', collection='alf/task_00'
        )
    except Exception:
        # Fall back to old format
        laser_intervals = one.load_dataset(eid, '_ibl_laserStimulation.intervals')
    
    return laser_intervals


def parse_zapit_log(file_path, session_start, eid=None):
    """
    Parse Zapit trial log file to extract stimulation events for a session.
    
    Parameters
    ----------
    file_path : str or Path
        Path to zapit_trials.yml file
    session_start : datetime
        Session start time
    eid : str, optional
        Experiment ID (for special case handling)
        
    Returns
    -------
    relevant_events : list
        List of event strings occurring during/after session start
    """
    # Handle known session-specific issues
    if eid == '21d33b44-f75f-4711-a2c7-0bdfe8eec386':
        session_start = datetime.strptime('2024-03-29T18:07:38.0'[:19], '%Y-%m-%dT%H:%M:%S')
    
    relevant_events = []
    
    with open(file_path, 'r') as file:
        next(file)  # Skip header line
        
        for line in file:
            # Extract timestamp (first 19 characters: YYYY-MM-DD HH:MM:SS)
            timestamp_str = line[:19]
            if len(timestamp_str) < 19:
                print('Warning: Error reading timestamp string for one line')
                continue
                
            try:
                event_timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
            except ValueError:
                continue
            
            # Keep events that occur during or after session start
            if event_timestamp >= session_start:
                relevant_events.append(line.strip())
    
    return relevant_events


def build_stim_location_dict(laser_intervals, trials, relevant_events, trials_range, eid=None):
    """
    Build dictionary mapping trial numbers to stimulation locations.
    
    Parameters
    ----------
    laser_intervals : ndarray
        Laser stimulation intervals [onset, offset]
    trials : dict-like
        Trials object
    relevant_events : list
        Zapit log events for this session
    trials_range : range or list
        Valid trial indices
    eid : str, optional
        Experiment ID (for special case handling)
        
    Returns
    -------
    stimtrial_location_dict_all : dict
        Maps trial number -> stim location (0 = control, 1-52 = stim locations)
    """
    stimtrial_location_dict = {}
    
    for k in range(1, len(laser_intervals[:, 0]) - 2):
        # Handle known session-specific issues
        if eid == '21d33b44-f75f-4711-a2c7-0bdfe8eec386' and k < 10:
            continue
            
        # Find trial number for this laser interval
        trial_matches = np.where(laser_intervals[k, 0] == trials.intervals[:, 0])[0]
        if len(trial_matches) == 0:
            continue
        trialnum = trial_matches[0]
        
        # Extract stim location from log (characters 20-22)
        stim_location_str = relevant_events[k][20:22]
        
        # Handle special case for session with off-by-one logged events
        if eid == '5a41494f-25b9-48d4-8159-527141bd4742':
            stim_location_str = relevant_events[k-1][20:22]
            
        # Clean and convert to integer
        stim_location = int(re.sub(r'\D', '', stim_location_str))
        stimtrial_location_dict[trialnum] = stim_location
    
    # Create full dict with 0 (control) for non-stim trials
    stimtrial_location_dict_all = {k: 0 for k in trials_range}
    stimtrial_location_dict_all.update(stimtrial_location_dict)
    
    return stimtrial_location_dict_all


def load_stim_locations_coordinates(file_path):
    """
    Load stimulation location coordinates from Zapit log file.
    
    Parameters
    ----------
    file_path : str or Path
        Path to zapit locations log file
        
    Returns
    -------
    stim_locations : dict
        Maps location number -> {'ML_left': float, 'ML_right': float, 'AP': float}
    """
    stim_locations = {}
    
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    current_location = None
    for line in lines:
        line = line.strip()
        
        # Check for stimLocations header
        location_match = re.search(r'stimLocations(\d+):', line)
        if location_match:
            current_location = int(location_match.group(1))
            stim_locations[current_location] = {'ML_left': None, 'ML_right': None, 'AP': None}
            
        elif line.startswith('ML: [') and current_location is not None:
            ml_coords = re.findall(r'\[([-.\d]+), ([-.\d]+)\]', line)
            if ml_coords:
                ml_left, ml_right = map(float, ml_coords[0])
                stim_locations[current_location]['ML_left'] = ml_left
                stim_locations[current_location]['ML_right'] = ml_right
                
        elif line.startswith('AP: [') and current_location is not None:
            ap_coords = re.findall(r'\[([-.\d]+), ([-.\d]+)\]', line)
            if ap_coords:
                stim_locations[current_location]['AP'] = float(ap_coords[0][0])
    
    return stim_locations


# =============================================================================
# GLM-HMM STATE FILTERING
# =============================================================================

def load_glmhmm_data(states_file, engaged_prev_file=None, disengaged_prev_file=None):
    """
    Load GLM-HMM state probability data from pickle files.

    Parameters
    ----------
    states_file : str or Path
        Path to all_subject_states.csv (pickle).
    engaged_prev_file : str or Path, optional
        Path to engaged_prevtrial_indices.pkl.
    disengaged_prev_file : str or Path, optional
        Path to disengaged_prevtrial_indices.pkl.

    Returns
    -------
    state_probability : dict
        Nested dict: state_probability[subject][eid][0] = state array.
    engaged_prev : dict or None
        Previous-trial engaged indices.
    disengaged_prev : dict or None
        Previous-trial disengaged indices.
    """
    with open(states_file, 'rb') as f:
        state_probability = pickle.load(f)

    engaged_prev = None
    disengaged_prev = None
    if engaged_prev_file is not None:
        with open(engaged_prev_file, 'rb') as f:
            engaged_prev = pickle.load(f)
    if disengaged_prev_file is not None:
        with open(disengaged_prev_file, 'rb') as f:
            disengaged_prev = pickle.load(f)

    return state_probability, engaged_prev, disengaged_prev


def is_eid_successful(state_dict, subject, session_eid):
    """Check whether valid GLM-HMM states exist for a given session."""
    return bool(state_dict.get(subject, {}).get(session_eid))


def get_glmhmm_indices(subject, eid, state_probability, n_states):
    """
    Extract trial indices for each GLM-HMM state.

    Parameters
    ----------
    subject : str
        Mouse ID.
    eid : str
        Experiment ID.
    state_probability : dict
        State probability data (from load_glmhmm_data).
    n_states : int
        Number of states (2 or 4).

    Returns
    -------
    For n_states=2:
        engaged_indices, disengaged_indices : tuple of ndarray
    For n_states=4:
        left_engaged, left_disengaged, right_engaged, right_disengaged : tuple
    """
    states = state_probability[subject][eid][0]

    if n_states == 2:
        engaged = np.where(np.logical_or(states == 2, states == 3))[0]
        disengaged = np.where(np.logical_or(states == 1, states == 4))[0]
        return engaged, disengaged

    elif n_states == 4:
        left_engaged = np.where(states == 3)
        left_disengaged = np.where(states == 4)
        right_engaged = np.where(states == 2)
        right_disengaged = np.where(states == 1)
        return left_engaged, left_disengaged, right_engaged, right_disengaged


def get_glmhmm_state_filter_indices(subject, eid, state_probability,
                                    n_states, state_type, state_def):
    """
    Get the set of trial indices to keep based on GLM-HMM state.

    This is a convenience wrapper that returns a single ndarray of trial
    indices in the desired state, with the +1 shift applied if state_def
    is 'previous'. Designed for filtering trial-number-keyed dicts (e.g.,
    Zapit's stimtrial_location_dict).

    Parameters
    ----------
    subject : str
        Mouse ID.
    eid : str
        Experiment ID.
    state_probability : dict
        State probability data (from load_glmhmm_data).
    n_states : int
        Number of states (2 or 4).
    state_type : str
        Which state to keep ('engaged', 'disengaged', 'bypass',
        'state1'-'state4').
    state_def : str
        'current' = use trial's own state label;
        'previous' = use previous trial's state label (shifts indices by +1).

    Returns
    -------
    state_indices : ndarray or None
        Integer array of trial indices in the requested state, with +1 shift
        applied if state_def='previous'. Returns None if state_type='bypass'
        (signals "no filtering needed").

    Raises
    ------
    ValueError
        If state_type or n_states is invalid.
    """
    if state_type == 'bypass':
        return None

    if n_states == 2:
        engaged_idx, disengaged_idx = get_glmhmm_indices(
            subject, eid, state_probability, n_states
        )

        if state_def == 'previous':
            engaged_idx = engaged_idx + 1
            disengaged_idx = disengaged_idx + 1

        if state_type == 'engaged':
            state_indices = engaged_idx
        elif state_type == 'disengaged':
            state_indices = disengaged_idx
        else:
            raise ValueError(f"Invalid state_type '{state_type}' for n_states=2")

    elif n_states == 4:
        s1, s2, s3, s4 = get_glmhmm_indices(
            subject, eid, state_probability, n_states
        )

        # Unpack tuple-wrapped arrays from np.where
        s1 = s1[0] if isinstance(s1, tuple) else s1
        s2 = s2[0] if isinstance(s2, tuple) else s2
        s3 = s3[0] if isinstance(s3, tuple) else s3
        s4 = s4[0] if isinstance(s4, tuple) else s4

        if state_def == 'previous':
            s1 = s1 + 1
            s2 = s2 + 1
            s3 = s3 + 1
            s4 = s4 + 1

        state_map = {
            'state1': s1,
            'state2': s2,
            'state3': s3,
            'state4': s4,
            'engaged': np.concatenate([s1, s3]),
            'disengaged': np.concatenate([s2, s4]),
        }
        if state_type not in state_map:
            raise ValueError(f"Invalid state_type '{state_type}' for n_states=4")
        state_indices = state_map[state_type]

    else:
        raise ValueError(f"n_states must be 2 or 4, got {n_states}")

    return state_indices.astype(int)


# =============================================================================
# TRIAL DATA PROCESSING
# =============================================================================

def signed_contrast(trials):
    """
    Compute signed contrast values from trial data.
    
    Parameters
    ----------
    trials : dict-like
        Trials object with contrastLeft and contrastRight
        
    Returns
    -------
    contrast : ndarray
        Signed contrast values in percent (negative = left, positive = right)
    """
    contrast = np.nan_to_num(np.c_[trials.contrastLeft, trials.contrastRight])
    return np.diff(contrast).flatten() * 100


def compute_reaction_time(trials, trial_number):
    """
    Compute reaction time for a single trial.
    
    Parameters
    ----------
    trials : dict-like
        Trials object
    trial_number : int
        Trial index
        
    Returns
    -------
    reaction_time : float
        Reaction time in seconds
    used_trigger : bool
        Whether goCueTrigger_times was used (vs goCue_times)
    """
    try:
        rt = trials.feedback_times[trial_number] - trials.goCueTrigger_times[trial_number]
        used_trigger = True
    except (IndexError, AttributeError):
        rt = trials.feedback_times[trial_number] - trials.goCue_times[trial_number]
        used_trigger = False
    
    # Fall back if NaN
    if np.isnan(rt):
        try:
            rt = trials.feedback_times[trial_number] - trials.goCueTrigger_times[trial_number]
        except (IndexError, AttributeError):
            rt = trials.feedback_times[trial_number] - trials.goCue_times[trial_number]
    
    return rt, used_trigger


def compute_quiescent_period(trials, trial_number, used_trigger=True):
    """
    Compute quiescent period duration for a single trial.
    
    Parameters
    ----------
    trials : dict-like
        Trials object
    trial_number : int
        Trial index
    used_trigger : bool
        Whether to use goCueTrigger_times
        
    Returns
    -------
    qp_time : float
        Quiescent period duration in seconds
    """
    if used_trigger:
        return trials.goCueTrigger_times[trial_number] - trials.intervals[trial_number][0]
    else:
        return trials.goCue_times[trial_number] - trials.intervals[trial_number][0]


def create_trial_data_dict(trials, trial_number, contrast_values):
    """
    Create a dictionary of trial data for a single trial.
    
    Parameters
    ----------
    trials : dict-like
        Trials object
    trial_number : int
        Trial index
    contrast_values : ndarray
        Pre-computed signed contrast values
        
    Returns
    -------
    trial_data : dict
        Dictionary with trial information
    """
    rt, used_trigger = compute_reaction_time(trials, trial_number)
    qp = compute_quiescent_period(trials, trial_number, used_trigger)
    
    return {
        'choice': trials.choice[trial_number],
        'reaction_times': rt,
        'qp_times': qp,
        'contrast': contrast_values[trial_number],
        'feedbackType': trials.feedbackType[trial_number],
        'probabilityLeft': trials.probabilityLeft[trial_number],
    }


def get_valid_trials_range(trials_ranges_entry, num_trials):
    """
    Convert trials range specification to actual range object.
    
    Parameters
    ----------
    trials_ranges_entry : str or list
        'ALL' or list of trial indices
    num_trials : int
        Total number of trials in session
        
    Returns
    -------
    trials_range : range or list
        Valid trial indices
    """
    if trials_ranges_entry == 'ALL':
        return range(0, num_trials)
    elif trials_ranges_entry[-1] == 9998:
        # 9998 indicates "until end of session"
        return [x for x in trials_ranges_entry if x < num_trials]
    else:
        return trials_ranges_entry


# =============================================================================
# SESSION QUALITY FILTERS
# =============================================================================

def check_session_accuracy(session_data, threshold, contrast_levels=(-100, -25, 25, 100)):
    """
    Check if session meets accuracy threshold at high contrasts.
    
    Parameters
    ----------
    session_data : list
        List of trial data dictionaries
    threshold : float
        Minimum accuracy (0-1)
    contrast_levels : tuple
        Contrast levels to check
        
    Returns
    -------
    passes : bool
        Whether session meets threshold
    accuracy : float
        Computed accuracy
    """
    feedback = [t['feedbackType'] for t in session_data 
                if t['contrast'] in contrast_levels]
    
    if len(feedback) == 0:
        return False, 0.0
    
    accuracy = np.sum(np.array(feedback) == 1) / len(feedback)
    return accuracy >= threshold, accuracy


def compute_session_bias_shift(session_data):
    """
    Compute total bias shift across all contrasts for session quality check.
    
    Parameters
    ----------
    session_data : list
        List of trial data dictionaries for control trials
        
    Returns
    -------
    total_bias_shift : float
        Sum of bias shifts across all contrasts
    """
    contrasts = [-100, -25, -12.5, -6.25, 0, 6.25, 12.5, 25, 100]
    total_shift = 0
    
    for c in contrasts:
        # Get choices for this contrast in each block
        choices_Lblock = [t['choice'] for t in session_data 
                         if t['contrast'] == c and t['probabilityLeft'] == 0.8]
        choices_Rblock = [t['choice'] for t in session_data 
                         if t['contrast'] == c and t['probabilityLeft'] == 0.2]
        
        if len(choices_Lblock) > 0 and len(choices_Rblock) > 0:
            p_right_Lblock = np.sum(np.array(choices_Lblock) == 1) / len(choices_Lblock)
            p_right_Rblock = np.sum(np.array(choices_Rblock) == 1) / len(choices_Rblock)
            shift = p_right_Lblock - p_right_Rblock
            if not np.isnan(shift):
                total_shift += shift
    
    return total_shift


# =============================================================================
# WHEEL ANALYSIS
# =============================================================================

def find_nearest_wheel_index(wheel_timestamps, target_time):
    """
    Find the wheel index closest to target time.
    
    Parameters
    ----------
    wheel_timestamps : ndarray
        Wheel timestamp array
    target_time : float
        Target time to find
        
    Returns
    -------
    index : int
        Index of nearest wheel timestamp
    """
    idx = np.searchsorted(wheel_timestamps, target_time)
    
    if idx == 0:
        return idx
    elif idx == len(wheel_timestamps):
        return idx - 1
    else:
        left_diff = target_time - wheel_timestamps[idx - 1]
        right_diff = wheel_timestamps[idx] - target_time
        return idx - 1 if left_diff <= right_diff else idx


def extract_wheel_trajectory(wheel, trials, trial_number, 
                             align_to='QP', duration=10, interval=0.1):
    """
    Extract wheel movement trajectory for a single trial.
    
    Parameters
    ----------
    wheel : dict-like
        Wheel object with position and timestamps
    trials : dict-like
        Trials object
    trial_number : int
        Trial index
    align_to : str
        Alignment point: 'QP', 'goCue', 'goCue_pre', 'feedback'
    duration : float
        Duration to analyze (seconds)
    interval : float
        Time bin size (seconds)
        
    Returns
    -------
    trajectory : ndarray
        Wheel position change at each time bin
    """
    whlpos, whlt = wheel.position, wheel.timestamps
    
    # Get alignment time
    try:
        if align_to == 'goCue':
            start_time = trials.goCueTrigger_times[trial_number]
        elif align_to == 'goCue_pre':
            start_time = trials.goCueTrigger_times[trial_number] - 0.5
        elif align_to == 'QP':
            start_time = trials.intervals[trial_number][0]
        elif align_to == 'feedback':
            start_time = trials.feedback_times[trial_number] - 0.6
    except (IndexError, AttributeError):
        if align_to == 'goCue':
            start_time = trials.goCue_times[trial_number]
        elif align_to == 'goCue_pre':
            start_time = trials.goCue_times[trial_number] - 0.5
        elif align_to == 'QP':
            start_time = trials.intervals[trial_number][0]
        elif align_to == 'feedback':
            start_time = trials.feedback_times[trial_number] - 0.6
    
    wheel_start_idx = find_nearest_wheel_index(whlt, start_time)
    
    # Get event boundaries for NaN masking
    try:
        go_cue_time = trials.goCueTrigger_times[trial_number]
    except (IndexError, AttributeError):
        go_cue_time = trials.goCue_times[trial_number]
    feedback_time = trials.feedback_times[trial_number]
    
    # Compute trajectory
    num_bins = int(duration / interval)
    trajectory = np.empty(num_bins)
    trajectory[:] = np.nan
    
    for i in range(num_bins):
        t = start_time + i * interval
        wheel_end_idx = find_nearest_wheel_index(whlt, t)
        
        # Mask based on alignment type
        if align_to == 'QP' and go_cue_time < whlt[wheel_end_idx]:
            continue  # Don't use wheel movement past go cue
        elif align_to in ['goCue', 'goCue_pre'] and (feedback_time + interval) < whlt[wheel_end_idx]:
            continue  # Don't use wheel movement past feedback
        
        trajectory[i] = whlpos[wheel_end_idx] - whlpos[wheel_start_idx]
    
    return trajectory


# =============================================================================
# MULTIPLE COMPARISONS CORRECTION
# =============================================================================

def correct_pvals_dict(pvals_dict, method='fdr_bh', alpha=0.05):
    """
    Apply multiple-comparisons correction to a dict of {condition: p-value}.

    Parameters
    ----------
    pvals_dict : dict
        Mapping of condition number -> raw p-value. Values that are NaN or
        non-finite are passed through unchanged (excluded from correction).
    method : str or None
        Correction method, passed to ``statsmodels.stats.multitest.multipletests``.
        Common choices:
          - 'fdr_bh'      : Benjamini-Hochberg FDR (recommended default)
          - 'bonferroni'  : Bonferroni family-wise
          - 'holm'        : Holm step-down
          - 'fdr_by'      : Benjamini-Yekutieli (FDR under arbitrary dependence)
        Pass ``None`` to disable correction (returns the input dict unchanged,
        useful for an off-switch).
    alpha : float
        Family-wise alpha for methods that need it.

    Returns
    -------
    dict
        New dict with same keys, values replaced by corrected p-values.
        Order of keys is preserved. NaN/non-finite values pass through.

    Notes
    -----
    Correction is applied only across the conditions that have valid p-values.
    Skipped (NaN/inf) conditions remain NaN in the output, so downstream code
    that uses ``-100 * log10(p)`` for marker size will treat them as
    non-significant rather than crashing.
    """
    if method is None or len(pvals_dict) == 0:
        return dict(pvals_dict)

    # Lazy import — keeps this module importable even if statsmodels is absent
    from statsmodels.stats.multitest import multipletests

    keys = list(pvals_dict.keys())
    raw = np.array([pvals_dict[k] for k in keys], dtype=float)

    valid_mask = np.isfinite(raw)
    if not np.any(valid_mask):
        return dict(pvals_dict)

    corrected = np.full_like(raw, np.nan)
    _, p_corr, _, _ = multipletests(raw[valid_mask], alpha=alpha, method=method)
    corrected[valid_mask] = p_corr

    return {k: float(c) for k, c in zip(keys, corrected)}


# =============================================================================
# BIAS ANALYSIS
# =============================================================================

def calculate_choice_probability(trials_list, block_type, contrast_level):
    """
    Calculate probability of leftward choice for a given block and contrast.
    
    Parameters
    ----------
    trials_list : list
        List of trial data dictionaries
    block_type : str
        'left' (probabilityLeft=0.8) or 'right' (probabilityLeft=0.2)
    contrast_level : float
        Contrast level to filter
        
    Returns
    -------
    p_left : float or None
        Probability of leftward choice, or None if no trials
    """
    prob_left_value = 0.8 if block_type == 'left' else 0.2
    
    relevant_trials = [t for t in trials_list 
                      if t['contrast'] == contrast_level 
                      and t['probabilityLeft'] == prob_left_value]
    
    if len(relevant_trials) == 0:
        return None
    
    leftward_choices = sum(1 for t in relevant_trials if t['choice'] == -1)
    return leftward_choices / len(relevant_trials)


def compute_bias_values_by_contrast(condition_data, contrasts, num_conditions=53):
    """
    Compute bias values (L block - R block choice probability) at each contrast.
    
    Parameters
    ----------
    condition_data : dict
        Maps condition number -> list of trial data dicts
    contrasts : list
        Contrast levels to compute
    num_conditions : int
        Number of conditions (including control)
        
    Returns
    -------
    bias_values : dict
        Maps condition -> list of bias values (one per contrast)
    left_block_probs : dict
        Maps condition -> list of left block probabilities
    right_block_probs : dict
        Maps condition -> list of right block probabilities
    """
    bias_values = {c: [] for c in range(num_conditions)}
    left_block_probs = {c: [] for c in range(num_conditions)}
    right_block_probs = {c: [] for c in range(num_conditions)}
    
    for contrast in contrasts:
        # Control condition (0)
        ctrl_left = calculate_choice_probability(condition_data[0], 'left', contrast)
        ctrl_right = calculate_choice_probability(condition_data[0], 'right', contrast)
        
        if ctrl_left is not None and ctrl_right is not None:
            ctrl_bias = ctrl_left - ctrl_right
            left_block_probs[0].append(ctrl_left)
            right_block_probs[0].append(ctrl_right)
            bias_values[0].append(ctrl_bias)
        
        # Stim conditions (1-52)
        for cond in range(1, num_conditions):
            stim_left = calculate_choice_probability(condition_data[cond], 'left', contrast)
            stim_right = calculate_choice_probability(condition_data[cond], 'right', contrast)
            
            if stim_left is not None and stim_right is not None:
                stim_bias = stim_left - stim_right
                left_block_probs[cond].append(stim_left)
                right_block_probs[cond].append(stim_right)
                bias_values[cond].append(stim_bias)
            elif ctrl_left is not None:
                # Fall back to control values if no data
                left_block_probs[cond].append(ctrl_left)
                right_block_probs[cond].append(ctrl_right)
                bias_values[cond].append(ctrl_bias)
    
    return bias_values, left_block_probs, right_block_probs


def compute_bias_values_by_cycle(condition_data, trials_per_cycle=5, 
                                  low_contrast_threshold=13, num_conditions=53):
    """
    Compute bias values in cycles of N trials for low contrast trials.
    
    This provides independent samples for statistical testing.
    
    Parameters
    ----------
    condition_data : dict
        Maps condition number -> list of trial data dicts
    trials_per_cycle : int
        Number of trials per cycle
    low_contrast_threshold : float
        Maximum contrast to include (%)
    num_conditions : int
        Number of conditions
        
    Returns
    -------
    bias_vals_LC : dict
        Maps condition -> array of bias values (one per cycle)
    """
    bias_vals_LC = {c: np.array([]) for c in range(num_conditions)}
    
    for condition in range(num_conditions):
        # Filter for low contrast trials in each block
        data_Lblock = [t for t in condition_data[condition] 
                      if abs(t['contrast']) < low_contrast_threshold and t['probabilityLeft'] == 0.8]
        data_Rblock = [t for t in condition_data[condition] 
                      if abs(t['contrast']) < low_contrast_threshold and t['probabilityLeft'] == 0.2]
        
        # Determine number of complete cycles
        num_cycles = min(len(data_Lblock), len(data_Rblock)) // trials_per_cycle
        
        if num_cycles == 0:
            continue
        
        bias_vals = np.empty(num_cycles)
        bias_vals[:] = np.nan
        
        for k in range(num_cycles):
            start_idx = k * trials_per_cycle
            end_idx = (k + 1) * trials_per_cycle
            
            choices_L = [t['choice'] for t in data_Lblock[start_idx:end_idx]]
            choices_R = [t['choice'] for t in data_Rblock[start_idx:end_idx]]
            
            mean_L = np.mean(choices_L)
            mean_R = np.mean(choices_R)
            bias_vals[k] = mean_L - mean_R
        
        bias_vals_LC[condition] = bias_vals
    
    return bias_vals_LC


def compute_effect_sizes(bias_values, only_low_contrasts=False):
    """
    Compute normalized effect sizes for each condition.
    
    Effect size = -(stim_bias_sum - ctrl_bias_sum) / ctrl_bias_sum
    
    Parameters
    ----------
    bias_values : dict
        Maps condition -> list of bias values
    only_low_contrasts : bool
        Whether bias_values contains only low contrasts
        
    Returns
    -------
    effect_sizes : dict
        Maps condition -> effect size
    """
    effect_sizes = {}
    
    if only_low_contrasts:
        indices = range(5)  # Low contrasts only
    else:
        indices = range(9)  # All contrasts
    
    ctrl_sum = sum(bias_values[0][i] for i in indices if i < len(bias_values[0]))
    
    if ctrl_sum == 0:
        return effect_sizes
    
    for condition in range(1, 53):
        if not bias_values[condition]:
            continue
        
        stim_sum = sum(bias_values[condition][i] for i in indices 
                      if i < len(bias_values[condition]))
        effect_sizes[condition] = -(stim_sum - ctrl_sum) / ctrl_sum
    
    return effect_sizes


# =============================================================================
# STATISTICAL TESTS
# =============================================================================

def run_condition_comparisons(bias_values, num_conditions=53):
    """
    Run statistical comparisons between control and each stim condition.
    
    Parameters
    ----------
    bias_values : dict
        Maps condition -> list of bias values
    num_conditions : int
        Number of conditions
        
    Returns
    -------
    results : dict
        Contains 'mannwhitney', 'ttest_paired', 'kruskal', 'anova' results
    """
    results = {
        'mannwhitney': {},
        'ttest_paired': {},
    }
    
    for cond in range(1, num_conditions):
        if bias_values[0] and bias_values[cond]:
            # Mann-Whitney U test
            _, p_mw = stats.mannwhitneyu(bias_values[0], bias_values[cond], 
                                          alternative='two-sided')
            results['mannwhitney'][cond] = p_mw
            
            # Paired t-test (across contrasts)
            if len(bias_values[0]) == len(bias_values[cond]):
                _, p_paired = stats.ttest_rel(bias_values[0], bias_values[cond])
                results['ttest_paired'][cond] = p_paired
    
    # Kruskal-Wallis across all conditions
    all_groups = [bias_values[c] for c in range(num_conditions) if bias_values[c]]
    if len(all_groups) > 1:
        results['kruskal_stat'], results['kruskal_p'] = stats.kruskal(*all_groups)
        results['anova_stat'], results['anova_p'] = stats.f_oneway(*all_groups)
    
    return results


def run_cycle_comparisons(bias_vals_LC, num_conditions=53):
    """
    Run independent t-tests on cycle-based bias values.
    
    Parameters
    ----------
    bias_vals_LC : dict
        Maps condition -> array of bias values (one per cycle)
    num_conditions : int
        Number of conditions
        
    Returns
    -------
    results : dict
        Contains 'ttest_ind', 'kruskal', 'anova' results
    """
    results = {'ttest_ind': {}}
    
    for cond in range(1, num_conditions):
        if len(bias_vals_LC[cond]) > 0 and len(bias_vals_LC[0]) > 0:
            _, p_ind = stats.ttest_ind(bias_vals_LC[0], bias_vals_LC[cond])
            results['ttest_ind'][cond] = p_ind
    
    # Kruskal-Wallis and ANOVA
    all_groups = [bias_vals_LC[c] for c in range(num_conditions) 
                  if len(bias_vals_LC[c]) > 0]
    if len(all_groups) > 1:
        results['kruskal_stat'], results['kruskal_p'] = stats.kruskal(*all_groups)
        results['anova_stat'], results['anova_p'] = stats.f_oneway(*all_groups)
    
    return results


def run_rt_analysis(condition_data, num_conditions=52):
    """
    Run reaction time analysis comparing each condition to control.
    
    Parameters
    ----------
    condition_data : dict
        Maps condition number -> list of trial data dicts
    num_conditions : int
        Number of stim conditions (excluding control)
        
    Returns
    -------
    rt_results : dict
        Maps condition -> {'p_val', 'effect_size', 'mean', 'std'}
    qp_results : dict
        Maps condition -> {'p_val', 'effect_size', 'mean', 'std'}
    lapse_results : dict
        Maps condition -> {'p_val', 'lapse_rate'}
    """
    # Control condition stats
    ctrl_rt = [t['reaction_times'] for t in condition_data[0] if not np.isnan(t['reaction_times'])]
    ctrl_qp = [t['qp_times'] for t in condition_data[0] if not np.isnan(t['qp_times'])]
    ctrl_fb = [t['feedbackType'] for t in condition_data[0] if t['contrast'] in (-100, 100)]
    ctrl_lapse = ((len(ctrl_fb) - np.sum(ctrl_fb)) / 2) / len(ctrl_fb) if ctrl_fb else 0
    
    rt_results = {0: {'mean': np.mean(ctrl_rt), 'std': np.std(ctrl_rt)}}
    qp_results = {0: {'mean': np.mean(ctrl_qp), 'std': np.std(ctrl_qp)}}
    lapse_results = {0: {'p_val': np.nan, 'lapse_rate': ctrl_lapse}}
    
    for cond in range(1, num_conditions + 1):
        stim_rt = [t['reaction_times'] for t in condition_data[cond] 
                   if not np.isnan(t['reaction_times'])]
        stim_qp = [t['qp_times'] for t in condition_data[cond] 
                   if not np.isnan(t['qp_times'])]
        stim_fb = [t['feedbackType'] for t in condition_data[cond] 
                   if t['contrast'] in (-100, -25, 100, 25)]
        
        if not stim_rt or not stim_qp:
            continue
        
        stim_lapse = ((len(stim_fb) - np.sum(stim_fb)) / 2) / len(stim_fb) if stim_fb else 0
        
        # T-tests
        _, p_rt = stats.ttest_ind(stim_rt, ctrl_rt, equal_var=False)
        _, p_qp = stats.ttest_ind(stim_qp, ctrl_qp, equal_var=False)
        
        # Cohen's d effect sizes
        pooled_std_rt = np.sqrt((np.std(stim_rt)**2 + np.std(ctrl_rt)**2) / 2)
        pooled_std_qp = np.sqrt((np.std(stim_qp)**2 + np.std(ctrl_qp)**2) / 2)
        d_rt = (np.mean(stim_rt) - np.mean(ctrl_rt)) / pooled_std_rt if pooled_std_rt > 0 else 0
        d_qp = (np.mean(stim_qp) - np.mean(ctrl_qp)) / pooled_std_qp if pooled_std_qp > 0 else 0
        
        # Proportions z-test for lapse rate
        _, p_lapse = proportions_ztest([ctrl_lapse, stim_lapse], 
                                        [len(ctrl_fb), len(stim_fb)])
        
        rt_results[cond] = {
            'p_val': p_rt, 'effect_size': d_rt, 
            'mean': np.mean(stim_rt), 'std': np.std(stim_rt)
        }
        qp_results[cond] = {
            'p_val': p_qp, 'effect_size': d_qp,
            'mean': np.mean(stim_qp), 'std': np.std(stim_qp)
        }
        lapse_results[cond] = {'p_val': p_lapse, 'lapse_rate': stim_lapse}
    
    return rt_results, qp_results, lapse_results


# =============================================================================
# PSYCHOMETRIC PLOTTING (CONDITION-SELECTED)
# =============================================================================

def organize_psychodata_from_dicts(trials_list):
    """
    Organize a list of trial dicts into psychometric-ready arrays per block.

    Each trial dict must contain at least 'choice', 'contrast', and
    'probabilityLeft'. This is the format produced by `create_trial_data_dict`
    and used throughout the Zapit pipeline (e.g. `condition_data[k]`).

    Parameters
    ----------
    trials_list : list of dict
        List of per-trial dicts.

    Returns
    -------
    data : dict
        Keys are probabilityLeft values (e.g. 0.2, 0.8), values are
        (3, n_contrasts) arrays of [contrasts, trial_counts, prop_rightward].
        Empty dict if no trials.
    """
    if len(trials_list) == 0:
        return {}

    choices = np.array([t['choice'] for t in trials_list])
    contrasts = np.array([t['contrast'] for t in trials_list])
    pLs = np.array([t['probabilityLeft'] for t in trials_list])

    # Mask out NaN trials (e.g. choice == 0 / no-go encoded as nan elsewhere)
    valid = ~(np.isnan(choices) | np.isnan(contrasts) | np.isnan(pLs))
    choices = choices[valid]
    contrasts = contrasts[valid]
    pLs = pLs[valid]

    rightward = (choices == -1).astype(float)

    data = {}
    for pL in np.unique(pLs):
        in_block = pLs == pL
        if not np.any(in_block):
            continue
        xx, nn = np.unique(contrasts[in_block], return_counts=True)
        pp = np.array([
            np.mean(rightward[(contrasts == x) & in_block]) for x in xx
        ])
        data[pL] = np.vstack((xx, nn, pp))
    return data


def plot_zapit_psychometric(stim_conditions, condition_data,
                            fit_kwargs=None, title='', save_path=None,
                            n_mice=None, num_sessions=None,
                            overlay=True, blocks_to_plot=(0.2, 0.8)):
    """
    Plot psychometric curves comparing pooled stim trials at user-selected
    Zapit stim locations versus pooled control (no-stim) trials.
 
    Designed for figure-making and interactive iPython exploration:
    pass any list of stim condition numbers (e.g. [16, 22] for a pair of
    points, or list(range(7, 14)) for a region) and the function pools
    the trials, fits psychometric curves per block, and plots in the
    same convention as opto_analysis.
 
    Parameters
    ----------
    stim_conditions : int or list of int
        Stim condition number(s) to pool (1..NUM_STIM_LOCATIONS).
        Condition 0 is always the control comparison.
    condition_data : dict
        Trial data keyed by condition number, as built in zapit_analysis.py.
        Each value is a list of trial dicts.
    fit_kwargs : dict or None
        Args for psy.mle_fit_psycho. If None, uses sensible defaults.
    title : str
        Figure title (e.g. 'MOs cluster: conds 16, 22').
    save_path : str, Path, or None
        If provided, save to this path; else show interactively.
    n_mice : int or None
        Annotation: number of unique mice contributing.
    num_sessions : int or None
        Annotation: number of sessions contributing.
    overlay : bool
        True = stim and control on a single axis (control solid,
        stim dashed). False = side-by-side panels.
    blocks_to_plot : iterable of float
        Which probabilityLeft block values to plot. Defaults to (0.2, 0.8),
        which excludes the 50/50 unbiased block. To include it, pass
        (0.2, 0.5, 0.8).
 
    Returns
    -------
    fig : matplotlib Figure
        The created figure (useful for further manipulation in iPython).
    """
    # Late imports so this module stays importable even if these aren't used
    import matplotlib.pyplot as plt
    import seaborn as sns
    try:
        import psychofit as psy
    except ImportError:
        raise ImportError(
            "psychofit is required for plot_zapit_psychometric. "
            "Install with: pip install Psychofit"
        )
 
    # Normalize stim_conditions to list
    if isinstance(stim_conditions, (int, np.integer)):
        stim_conditions = [int(stim_conditions)]
    else:
        stim_conditions = [int(c) for c in stim_conditions]
 
    # Normalize blocks_to_plot for set-membership checks
    blocks_to_plot = set(float(b) for b in blocks_to_plot)
 
    # Default fit kwargs (matching opto_analysis)
    if fit_kwargs is None:
        fit_kwargs = {
            'parmin': np.array([-50., 10., 0., 0.]),
            'parmax': np.array([50., 100., 1., 1.]),
            'parstart': np.array([0., 40., 0.1, 0.1]),
            'nfits': 50,
        }
 
    # Pool trials
    nonstim_trials = list(condition_data.get(0, []))
    stim_trials = []
    missing = []
    for c in stim_conditions:
        if c not in condition_data or len(condition_data[c]) == 0:
            missing.append(c)
        else:
            stim_trials.extend(condition_data[c])
 
    if missing:
        print(f"  [psychometric] No trials found for condition(s): {missing}")
    if len(stim_trials) == 0:
        print(f"  [psychometric] No stim trials available for {stim_conditions}, skipping plot.")
        return None
    if len(nonstim_trials) == 0:
        print(f"  [psychometric] No control trials available, skipping plot.")
        return None
 
    # Build psycho-ready dicts
    stim_data = organize_psychodata_from_dicts(stim_trials)
    nonstim_data = organize_psychodata_from_dicts(nonstim_trials)
 
    # Filter to only the blocks the caller wants. Use a small float tolerance
    # so e.g. 0.2 vs 0.20000001 still match.
    def _filter_blocks(data):
        return {pL: da for pL, da in data.items()
                if any(abs(float(pL) - b) < 1e-6 for b in blocks_to_plot)}
 
    stim_data = _filter_blocks(stim_data)
    nonstim_data = _filter_blocks(nonstim_data)
 
    # Plotting (matches opto_analysis convention)
    ctrl_colours = {0.2: 'xkcd:tangerine', 0.8: 'xkcd:violet', 0.5: 'k'}
    stim_colours = {0.2: 'xkcd:red',       0.8: 'xkcd:red',   0.5: 'k'}
    x_fit = np.arange(-100, 100)
 
    if overlay:
        fig, ax = plt.subplots(figsize=(4.5, 4.5))
 
        for data, linestyle, cond_label, colour_map in [
            (stim_data, '--', 'stim', stim_colours),
            (nonstim_data, '-', 'ctrl', ctrl_colours),
        ]:
            for pL, da in data.items():
                if da.shape[1] < 2:
                    continue
                color = colour_map.get(pL, 'grey')
                block_label = f'{int(pL * 100)}% left'
                try:
                    pars, _ = psy.mle_fit_psycho(da, 'erf_psycho_2gammas', **fit_kwargs)
                    ax.plot(da[0, :], da[2, :], 'o', color=color,
                            markersize=5, alpha=0.7)
                    ax.plot(x_fit, psy.erf_psycho_2gammas(pars, x_fit),
                            color=color, linestyle=linestyle, linewidth=1.8,
                            label=f'{block_label} {cond_label}')
                except Exception:
                    ax.plot(da[0, :], da[2, :], 'o', color=color,
                            markersize=5, alpha=0.7)
 
        ax.set_title(title)
        ax.set_xlabel('Contrast (%)')
        ax.set_ylabel('Proportion rightward')
        ax.axhline(0.5, color='k', linestyle=':', alpha=0.5)
        ax.axvline(0, color='k', linestyle=':', alpha=0.5)
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=8, loc='best')
        sns.despine(ax=ax, offset=10, trim=True)
 
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))
        for ax, data, label in [(ax1, nonstim_data, 'Control'),
                                (ax2, stim_data, 'Stim')]:
            for pL, da in data.items():
                if da.shape[1] < 2:
                    continue
                colour = ctrl_colours.get(pL, 'grey')
                try:
                    pars, _ = psy.mle_fit_psycho(da, 'erf_psycho_2gammas', **fit_kwargs)
                    ax.plot(da[0, :], da[2, :], 'o', color=colour)
                    ax.plot(x_fit, psy.erf_psycho_2gammas(pars, x_fit),
                            color=colour, label=f'{int(pL * 100)}% left')
                except Exception:
                    ax.plot(da[0, :], da[2, :], 'o', color=colour)
            ax.set_title(f'{label} — {title}')
            ax.set_xlabel('Contrast (%)')
            ax.set_ylabel('Proportion rightward')
            ax.axhline(0.5, color='k', linestyle=':', alpha=0.5)
            ax.axvline(0, color='k', linestyle=':', alpha=0.5)
            ax.set_ylim(-0.05, 1.05)
            ax.legend(fontsize=8)
            sns.despine(ax=ax, offset=10, trim=True)
 
    # Annotation: trial / session / mouse counts
    annotation_parts = [f'conds: {stim_conditions}']
    if n_mice is not None:
        annotation_parts.append(f'{n_mice} mice')
    if num_sessions is not None:
        annotation_parts.append(f'{num_sessions} sessions')
    annotation_parts.append(f'{len(nonstim_trials)} ctrl trials')
    annotation_parts.append(f'{len(stim_trials)} stim trials')
 
    fig.text(0.5, 0.01, ' | '.join(annotation_parts),
             ha='center', fontsize=8, style='italic', color='grey')
    fig.subplots_adjust(bottom=0.15)
    plt.tight_layout(rect=[0, 0.04, 1, 1])
 
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
 
    return fig

# =============================================================================
# BRAIN ATLAS VISUALIZATION
# =============================================================================

def transform_to_ccf(x, y, z, resolution=10):
    """
    Transform stereotaxic coordinates to Allen CCF coordinates.
    
    Parameters
    ----------
    x, y, z : float
        Coordinates in micrometers
    resolution : int
        CCF resolution in micrometers per pixel
        
    Returns
    -------
    X, Y, Z : float
        Transformed coordinates in CCF space
    """
    # Bregma position for 10um resolution
    x_bregma, y_bregma, z_bregma = 540, 44, 570
    x -= x_bregma
    y -= y_bregma
    z -= z_bregma
    
    # Rotate CCF (5 degrees)
    angle_rad = 5 * (np.pi / 180)
    X = x * np.cos(angle_rad) - y * np.sin(angle_rad)
    Y = x * np.sin(angle_rad) + y * np.cos(angle_rad)
    
    # Squeeze DV axis
    Y *= 0.9434
    
    # Scale to resolution
    X, Y, Z = X / resolution, Y / resolution, z / resolution
    
    return X, Y, Z


def generate_mip_with_borders(annotation_volume):
    """
    Generate maximum intensity projection with region borders.
    
    Parameters
    ----------
    annotation_volume : ndarray
        3D Allen CCF annotation volume (AP x DV x ML)
        
    Returns
    -------
    mip : ndarray
        2D maximum intensity projection
    edges : ndarray
        2D edge detection result for region borders
    """
    # Find dorsal surface (first non-zero label along DV axis)
    dorsal_surface_index = np.argmax(annotation_volume > 0, axis=1)
    
    ap_length, ml_length = dorsal_surface_index.shape
    dv_length = annotation_volume.shape[1]
    mip = np.zeros((ap_length, ml_length), dtype=annotation_volume.dtype)
    
    # Populate MIP
    for x in range(ml_length):
        for y in range(ap_length):
            dv_idx = dorsal_surface_index[y, x]
            if dv_idx < dv_length:
                mip[y, x] = annotation_volume[y, dv_idx, x]
    
    # Edge detection for borders
    grad_x, grad_y = np.gradient(mip)
    edges = np.sqrt(grad_x**2 + grad_y**2)
    
    return mip, edges
