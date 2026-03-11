"""
Helper functions for bilateral optogenetics analysis pipeline.

This module contains reusable functions for:
- Data loading and session management
- GLM-HMM state filtering
- Trial identification (stim vs. nonstim)
- Bunch subsetting and concatenation
- Bias shift computation (psychometric fitting)
- Wheel trajectory extraction
- Session quality checks
- Plotting utilities
"""

import copy
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import psychofit as psy
from scipy import stats
from ibllib.io.raw_data_loaders import load_data


# =============================================================================
# DATA LOADING
# =============================================================================

def load_session_data(one, eid, flag_failed_loads=True):
    """
    Load trial and wheel data for a session from the IBL database.

    Parameters
    ----------
    one : ONE
        ONE API instance.
    eid : str
        Experiment ID.
    flag_failed_loads : bool
        Whether to print warning on failed loads.

    Returns
    -------
    trials : Bunch or None
        Trials object with all trial data.
    wheel : Bunch or None
        Wheel object with position and timestamps.
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
            print(f'Failed to load wheel for eid = {eid}')
        return None, None

    return trials, wheel


def fix_trials_length_inconsistency(trials):
    """
    Fix known IBL issue where trial attribute arrays differ in length by 1.

    If any attribute is exactly (max_length - 1), prepend a zero to align it.

    Parameters
    ----------
    trials : Bunch
        Trials object.

    Returns
    -------
    trials : Bunch
        Fixed trials object.
    """
    lengths = [len(trials[k]) for k in trials.keys()]
    max_length = max(lengths)

    for k in trials.keys():
        if len(trials[k]) == max_length - 1:
            trials[k] = np.insert(trials[k], 0, 0)

    return trials


def load_laser_intervals(one, eid):
    """
    Load laser stimulation intervals for a session, trying multiple collection paths.

    Parameters
    ----------
    one : ONE
        ONE API instance.
    eid : str
        Experiment ID.

    Returns
    -------
    laser_intervals : ndarray or None
        Array of [onset, offset] times for each laser stimulation.
    """
    for collection in [None, 'alf', 'alf/task_00', 'alf/task00']:
        try:
            if collection is None:
                return one.load_dataset(eid, '_ibl_laserStimulation.intervals')
            else:
                return one.load_dataset(
                    eid, '_ibl_laserStimulation.intervals', collection=collection
                )
        except Exception:
            continue
    return None


def load_task_data(one, eid):
    """
    Load raw task data (deprecated format) as fallback for older sessions.

    Parameters
    ----------
    one : ONE
        ONE API instance.
    eid : str
        Experiment ID.

    Returns
    -------
    task_data : list or None
        Raw task data (list of dicts per trial).
    ses_path : Path or None
        Session path on disk.
    """
    try:
        dset = '_iblrig_taskData.raw*'
        one.load_dataset(eid, dataset=dset, collection='raw_behavior_data')
        ses_path = one.eid2path(eid)
        task_data = load_data(ses_path)
        return task_data, ses_path
    except Exception:
        return None, None


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


def filter_trials_by_state(trials_numbers, stim_trials_numbers, nonstim_trials_numbers,
                           state_probability, mouse_id, eid, n_states, state_type, state_def):
    """
    Filter trial indices using GLM-HMM state labels.

    This intersects the existing stim/nonstim trial numbers with the indices
    belonging to the requested state, effectively keeping only trials where
    the mouse was in the desired engagement state.

    Parameters
    ----------
    trials_numbers : ndarray
        All valid trial indices.
    stim_trials_numbers : ndarray
        Stim trial indices.
    nonstim_trials_numbers : ndarray
        Nonstim trial indices.
    state_probability : dict
        State probability data.
    mouse_id : str
        Mouse ID.
    eid : str
        Experiment ID.
    n_states : int
        Number of states (2 or 4).
    state_type : str
        Which state to keep ('engaged', 'disengaged', 'bypass', 'state1'-'state4').
    state_def : str
        'current' or 'previous'.

    Returns
    -------
    trials_numbers : ndarray
        Filtered trial indices.
    stim_trials_numbers : ndarray
        Filtered stim trial indices.
    nonstim_trials_numbers : ndarray
        Filtered nonstim trial indices.

    Raises
    ------
    ValueError
        If GLM-HMM states cannot be loaded for this session.
    """
    if state_type == 'bypass':
        return trials_numbers, stim_trials_numbers, nonstim_trials_numbers

    # Get state indices
    if n_states == 2:
        engaged_idx, disengaged_idx = get_glmhmm_indices(
            mouse_id, eid, state_probability, n_states
        )

        # Shift by +1 for 'previous' state definition
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
            mouse_id, eid, state_probability, n_states
        )

        if state_def == 'previous':
            s1 = s1[0] + 1
            s2 = s2[0] + 1
            s3 = s3[0] + 1
            s4 = s4[0] + 1

        state_map = {
            'state1': s1[0] if isinstance(s1, tuple) else s1,
            'state2': s2[0] if isinstance(s2, tuple) else s2,
            'state3': s3[0] if isinstance(s3, tuple) else s3,
            'state4': s4[0] if isinstance(s4, tuple) else s4,
            'engaged': np.concatenate([
                s1[0] if isinstance(s1, tuple) else s1,
                s3[0] if isinstance(s3, tuple) else s3,
            ]),
            'disengaged': np.concatenate([
                s2[0] if isinstance(s2, tuple) else s2,
                s4[0] if isinstance(s4, tuple) else s4,
            ]),
        }
        if state_type not in state_map:
            raise ValueError(f"Invalid state_type '{state_type}' for n_states=4")
        state_indices = state_map[state_type]
    else:
        raise ValueError(f"n_states must be 2 or 4, got {n_states}")

    # Intersect with existing trial sets
    trials_numbers = np.intersect1d(state_indices, trials_numbers)
    stim_trials_numbers = np.intersect1d(trials_numbers, stim_trials_numbers)
    nonstim_trials_numbers = np.intersect1d(trials_numbers, nonstim_trials_numbers)

    # Clean: ensure no nans and integer type
    stim_trials_numbers = stim_trials_numbers[~np.isnan(stim_trials_numbers)].astype(int)
    nonstim_trials_numbers = nonstim_trials_numbers[~np.isnan(nonstim_trials_numbers)].astype(int)

    # Enforce mutual exclusivity
    nonstim_trials_numbers = np.setdiff1d(nonstim_trials_numbers, stim_trials_numbers)

    return trials_numbers, stim_trials_numbers, nonstim_trials_numbers


# =============================================================================
# TRIAL IDENTIFICATION
# =============================================================================

def signed_contrast(trials):
    """
    Compute signed contrast in percent from trial data.

    Returns array where negative = left stimulus, positive = right stimulus.
    """
    contrast = np.nan_to_num(np.c_[trials.contrastLeft, trials.contrastRight])
    return np.diff(contrast).flatten() * 100


def get_valid_trials_range(trials_ranges_j, trials):
    """
    Convert metadata trials range to a usable range of trial indices.

    Parameters
    ----------
    trials_ranges_j : str or list
        'ALL', or a list of trial indices (with 9998 as sentinel for last trial).
    trials : Bunch
        Trials object (used to determine total trial count).

    Returns
    -------
    trials_range : list or range
        Valid trial indices for this session.
    """
    if trials_ranges_j == 'ALL':
        return range(0, len(trials['contrastLeft']))
    elif trials_ranges_j[-1] == 9998:
        return [x for x in trials_ranges_j if x < np.size(trials.probabilityLeft)]
    else:
        return trials_ranges_j


def compute_reaction_times(trials, task_data=None):
    """
    Compute reaction times and quiescent period times for all trials.

    Parameters
    ----------
    trials : Bunch
        Trials object.
    task_data : list or None
        Raw task data (deprecated format), used as primary source if available.

    Returns
    -------
    reaction_times : ndarray
        Reaction time for each trial (seconds). NaN if invalid.
    quiescent_period_times : ndarray
        Quiescent period duration for each trial (seconds). NaN if invalid.
    """
    n_trials = len(trials['contrastLeft'])
    reaction_times = np.full(n_trials, np.nan)
    quiescent_period_times = np.full(n_trials, np.nan)

    for tr in range(n_trials):
        # Compute RT from goCue (prefer goCueTrigger_times, fall back to goCue_times)
        try:
            rt = trials['feedback_times'][tr] - trials['goCueTrigger_times'][tr]
        except (KeyError, IndexError):
            try:
                rt = trials['feedback_times'][tr] - trials['goCue_times'][tr]
            except (KeyError, IndexError):
                continue

        if rt > 59.9:
            continue

        # Compute QP
        try:
            if task_data is not None:
                trial_start = task_data[tr]['behavior_data']['States timestamps']['trial_start'][0][0]
                stim_on = task_data[tr]['behavior_data']['States timestamps']['stim_on'][0][0]
            else:
                trial_start = trials.intervals[tr, 0]
                # Prefer goCueTrigger_times, fall back to goCue_times
                try:
                    stim_on = trials.goCueTrigger_times[tr]
                except (AttributeError, KeyError):
                    stim_on = trials.goCue_times[tr]
        except Exception:
            trial_start = trials.intervals[tr, 0]
            try:
                stim_on = trials.goCueTrigger_times[tr]
            except (AttributeError, KeyError):
                stim_on = trials.goCue_times[tr]

        reaction_times[tr] = rt
        quiescent_period_times[tr] = stim_on - trial_start

    return reaction_times, quiescent_period_times


def identify_stim_nonstim_trials(trials, trials_range, stim_param, laser_intervals=None,
                                 task_data=None, rt_threshold=30):
    """
    Identify which trials in trials_range are stim vs nonstim.

    Supports multiple stimulation protocols:
    - 'QPRE': trial start matches laser interval onset
    - 'SORE': laser onset falls within trial interval
    - Legacy: taskData[k]['opto'] == 1

    Parameters
    ----------
    trials : Bunch
        Trials object.
    trials_range : range or list
        Valid trial indices.
    stim_param : str
        Stimulation protocol identifier ('QPRE', 'SORE', etc.).
    laser_intervals : ndarray or None
        Laser stimulation intervals [onset, offset].
    task_data : list or None
        Raw task data (legacy format).
    rt_threshold : float
        Maximum reaction time to include a trial.

    Returns
    -------
    stim_trials : ndarray of int
        Indices of stim trials.
    nonstim_trials : ndarray of int
        Indices of nonstim trials.
    """
    n_trials = len(trials['contrastLeft'])
    stim_arr = np.full(n_trials, np.nan)
    nonstim_arr = np.full(n_trials, np.nan)

    def _get_rt(k):
        """Get reaction time for trial k."""
        try:
            return trials['feedback_times'][k] - trials['goCueTrigger_times'][k]
        except (KeyError, IndexError):
            try:
                return trials['feedback_times'][k] - trials['goCue_times'][k]
            except (KeyError, IndexError):
                return np.nan

    # Try laser intervals first, fall back to task data
    if laser_intervals is not None:
        for k in trials_range:
            rt = _get_rt(k)
            if np.isnan(rt) or rt >= rt_threshold:
                continue

            is_stim = False
            if stim_param in ('QPRE', 'QPRE*', 'QP', 'QP*'):
                is_stim = trials.intervals[k, 0] in laser_intervals[:, 0]
            elif stim_param == 'SORE':
                start_trial = trials.intervals[k, 0]
                end_trial = trials.intervals[k, 1]
                is_stim = np.any(
                    (laser_intervals[:, 0] >= start_trial) &
                    (laser_intervals[:, 0] <= end_trial)
                )
            else:
                # Default: match on trial start time
                is_stim = trials.intervals[k, 0] in laser_intervals[:, 0]

            if is_stim:
                stim_arr[k] = k
            else:
                nonstim_arr[k] = k

    elif task_data is not None:
        for k in trials_range:
            rt = _get_rt(k)
            if np.isnan(rt) or rt >= rt_threshold:
                continue
            try:
                if task_data[k]['opto'] == 1:
                    stim_arr[k] = k
                else:
                    nonstim_arr[k] = k
            except (KeyError, IndexError):
                nonstim_arr[k] = k
    else:
        raise ValueError("Either laser_intervals or task_data must be provided")

    # Clean up
    stim_trials = stim_arr[~np.isnan(stim_arr)].astype(int)
    nonstim_trials = nonstim_arr[~np.isnan(nonstim_arr)].astype(int)

    return stim_trials, nonstim_trials


def apply_trials_after_stim(stim_trials, nonstim_trials):
    """
    Shift stim trial indices +1 for lagged (trials-after-stim) analysis.

    Removes stim trials where the next trial is also a stim trial
    (i.e., no clean non-stim trial to shift to), and removes shifted
    indices that extend past the last nonstim trial.

    Parameters
    ----------
    stim_trials : ndarray
        Original stim trial indices.
    nonstim_trials : ndarray
        Nonstim trial indices.

    Returns
    -------
    shifted_stim : ndarray
        Shifted stim trial indices (each +1 from original).
    """
    shifted = stim_trials.copy().astype(float)
    for i in range(len(shifted)):
        if i == len(shifted) - 1:
            if shifted[i] > nonstim_trials[-1]:
                shifted[i] = np.nan
            else:
                shifted[i] += 1
        else:
            if stim_trials[i + 1] - stim_trials[i] == 1:
                shifted[i] = np.nan  # consecutive stim trials — skip
            else:
                shifted[i] += 1

    return shifted[~np.isnan(shifted)].astype(int)


def subsample_stim_trials_balanced(stim_trials, trials):
    """
    Subsample stim trials to balance previous-choice × block conditions.

    This ensures that the stim trial set is not confounded by an imbalance
    in which block the mouse was in or what choice it made on the previous trial.

    Parameters
    ----------
    stim_trials : ndarray
        Stim trial indices.
    trials : Bunch
        Full trials object.

    Returns
    -------
    stim_trials : ndarray
        Subsampled stim trial indices.
    """
    import random

    Rblock = stim_trials[trials.probabilityLeft[stim_trials] == 0.2]
    Lblock = stim_trials[trials.probabilityLeft[stim_trials] == 0.8]
    PrevL = stim_trials[trials.choice[stim_trials - 1] == -1]
    PrevR = stim_trials[trials.choice[stim_trials - 1] == 1]

    pairs = [
        (np.intersect1d(Lblock, PrevL), np.intersect1d(Lblock, PrevR)),
        (np.intersect1d(Rblock, PrevL), np.intersect1d(Rblock, PrevR)),
    ]

    to_remove = []
    for group_a, group_b in pairs:
        diff = len(group_a) - len(group_b)
        if diff > 0:
            to_remove.extend(random.sample(list(group_a), diff))
        elif diff < 0:
            to_remove.extend(random.sample(list(group_b), abs(diff)))

    if to_remove:
        stim_trials = np.setdiff1d(stim_trials, to_remove)

    return stim_trials


# =============================================================================
# BUNCH SUBSETTING AND CONCATENATION
# =============================================================================

def subset_bunch(bunch_obj, indices):
    """
    Create a new Bunch by slicing all trial-length array attributes.

    This replaces the repeated 15-attribute manual subsetting that appeared
    throughout the original scripts.

    Parameters
    ----------
    bunch_obj : Bunch
        IBL trials Bunch object.
    indices : ndarray or slice
        Indices to select.

    Returns
    -------
    new_bunch : Bunch
        Sliced copy of the Bunch.
    """
    new_bunch = copy.copy(bunch_obj)

    for k in list(new_bunch.__dict__.keys()):
        arr = new_bunch.__dict__[k]
        if isinstance(arr, np.ndarray) and arr.ndim > 0 and arr.shape[0] > 10:
            try:
                new_bunch.__dict__[k] = arr[indices]
            except (IndexError, TypeError):
                pass

    return new_bunch


def concat_bunches(bunch1, bunch2):
    """
    Concatenate two Bunch objects by appending arrays in bunch2 to bunch1.

    Parameters
    ----------
    bunch1 : Bunch or None
    bunch2 : Bunch or None

    Returns
    -------
    merged : Bunch
    """
    if bunch1 is None:
        return bunch2
    if bunch2 is None:
        return bunch1

    merged = copy.copy(bunch1)
    attributes = [attr for attr in dir(bunch1) if not attr.startswith('__')]

    for attr in attributes:
        val1 = getattr(bunch1, attr)
        if hasattr(bunch2, attr):
            val2 = getattr(bunch2, attr)
            if isinstance(val1, np.ndarray) and isinstance(val2, np.ndarray):
                try:
                    setattr(merged, attr, np.concatenate((val1, val2)))
                except ValueError:
                    pass

    return merged


# =============================================================================
# BIAS SHIFT COMPUTATION
# =============================================================================

def organize_psychodata(trials_bunch, contrast_array):
    """
    Organize trials by block and contrast for psychometric fitting.

    Parameters
    ----------
    trials_bunch : Bunch
        Subset of trials (stim or nonstim).
    contrast_array : ndarray
        Signed contrast values for each trial in the bunch.

    Returns
    -------
    data : dict
        Keys are probabilityLeft values (0.2, 0.8), values are (3, n_contrasts)
        arrays of [contrasts, trial_counts, proportion_rightward].
    """
    data = {}
    for pL in np.unique(trials_bunch.probabilityLeft):
        if np.isnan(pL):
            continue
        in_block = trials_bunch.probabilityLeft == pL
        xx, nn = np.unique(contrast_array[in_block], return_counts=True)
        rightward = trials_bunch.choice == -1
        pp = np.array([np.mean(rightward[(x == contrast_array) & in_block]) for x in xx])
        data[pL] = np.vstack((xx, nn, pp))
    return data


def compute_bias_shift(psycho_data, fit_kwargs, contrast_set='all'):
    """
    Compute bias shift from psychometric data by fitting each block separately.

    Fits erf_psycho_2gammas to each block's data, then computes the difference
    in fitted values (block 0.2 - block 0.8) at each contrast level. The
    bias shift is the sum of these differences.

    Parameters
    ----------
    psycho_data : dict
        Output of organize_psychodata().
    fit_kwargs : dict
        Keyword arguments for psy.mle_fit_psycho.
    contrast_set : str
        Which contrasts to sum over: 'all', 'high', or 'low'.

    Returns
    -------
    bias_shift_total : float
        Total summed bias shift.
    bias_shift_lc : float
        Low-contrast component of bias shift.
    bias_shift_hc : float
        High-contrast component of bias shift.
    bias_shift_0 : float
        Zero-contrast component of bias shift.
    per_contrast : dict
        Bias shift at each individual contrast level.
    """
    contrast_points = {
        'high': [-100., -25., 25., 100.],
        'low': [-12.5, -6.25, 0., 6.25, 12.5],
        'all': [-100., -25., -12.5, -6.25, 0., 6.25, 12.5, 25., 100.],
    }

    fits = {}
    for pL, da in psycho_data.items():
        if da.shape[1] < 2:
            continue
        try:
            pars, L = psy.mle_fit_psycho(da, 'erf_psycho_2gammas', **fit_kwargs)
            fits[pL] = pars
        except Exception:
            continue

    if 0.8 not in fits or 0.2 not in fits:
        return np.nan, np.nan, np.nan, np.nan, {}

    # Compute per-contrast bias shift
    per_contrast = {}
    for c in contrast_points['all']:
        val_20 = psy.erf_psycho_2gammas(fits[0.2], c)
        val_80 = psy.erf_psycho_2gammas(fits[0.8], c)
        per_contrast[c] = val_20 - val_80

    # Aggregate
    bias_shift_0 = per_contrast.get(0., 0.)
    bias_shift_hc = sum(per_contrast.get(c, 0.) for c in contrast_points['high'])
    bias_shift_lc = sum(per_contrast.get(c, 0.) for c in contrast_points['low'])
    bias_shift_total = sum(per_contrast.values())

    return bias_shift_total, bias_shift_lc, bias_shift_hc, bias_shift_0, per_contrast


# =============================================================================
# SESSION QUALITY CHECKS
# =============================================================================

def check_session_performance(trials_bunch, contrast_array, threshold, check_type='nonstim'):
    """
    Check whether high-contrast accuracy exceeds a threshold.

    Parameters
    ----------
    trials_bunch : Bunch
        Subset of trials.
    contrast_array : ndarray
        Signed contrasts for the bunch.
    threshold : float
        Minimum accuracy required (0-1).

    Returns
    -------
    passes : bool
        Whether the session passes.
    accuracy_left : float
        Accuracy on left high-contrast trials.
    accuracy_right : float
        Accuracy on right high-contrast trials.
    """
    if threshold <= 0:
        return True, np.nan, np.nan

    hc_right_mask = contrast_array == 100
    hc_left_mask = contrast_array == -100

    n_right = np.sum(hc_right_mask)
    n_left = np.sum(hc_left_mask)

    if n_right == 0 or n_left == 0:
        return False, 0., 0.

    correct_right = np.sum(trials_bunch.rewardVolume[hc_right_mask]) / 1.5
    correct_left = np.sum(trials_bunch.rewardVolume[hc_left_mask]) / 1.5

    acc_right = correct_right / n_right
    acc_left = correct_left / n_left

    passes = (acc_right >= threshold) and (acc_left >= threshold)
    return passes, acc_left, acc_right


def compute_choice_bias_deviation(trials, stim_trials, nonstim_trials):
    """
    Compute the choice bias deviation between stim and nonstim trials.

    Bias = (rightward - leftward) / total for each condition.
    Deviation = stim_bias - nonstim_bias.

    Parameters
    ----------
    trials : Bunch
        Full trials object.
    stim_trials : ndarray
        Stim trial indices.
    nonstim_trials : ndarray
        Nonstim trial indices.

    Returns
    -------
    deviation : float
        Stim bias minus nonstim bias.
    """
    nonstim_choices = trials.choice[nonstim_trials]
    nonstim_bias = (np.sum(nonstim_choices == -1) - np.sum(nonstim_choices == 1)) / len(nonstim_choices)

    stim_choices = trials.choice[stim_trials]
    stim_bias = (np.sum(stim_choices == -1) - np.sum(stim_choices == 1)) / len(stim_choices)

    return stim_bias - nonstim_bias


# =============================================================================
# WHEEL ANALYSIS
# =============================================================================

def extract_wheel_trajectory(trials, trial_num, whlpos, whlt, align_to='QP',
                             duration=10, interval=0.1):
    """
    Extract wheel movement trajectory for a single trial.

    Parameters
    ----------
    trials : Bunch
        Trials object.
    trial_num : int
        Trial index.
    whlpos : ndarray
        Wheel position array.
    whlt : ndarray
        Wheel timestamps array.
    align_to : str
        Alignment point ('QP', 'goCue', 'goCue_pre', 'feedback').
    duration : float
        Duration to analyze (seconds).
    interval : float
        Time bin size (seconds).

    Returns
    -------
    trajectory : ndarray
        Cumulative wheel movement at each time bin. NaN where data is
        truncated (e.g., past goCue when aligned to QP).
    """
    # Determine alignment time
    if align_to == 'goCue':
        try:
            start_time = trials.goCueTrigger_times[trial_num]
        except (AttributeError, KeyError):
            start_time = trials.goCue_times[trial_num]
    elif align_to == 'goCue_pre':
        try:
            start_time = trials.goCueTrigger_times[trial_num] - 0.5
        except (AttributeError, KeyError):
            start_time = trials.goCue_times[trial_num] - 0.5
    elif align_to == 'QP':
        start_time = trials.intervals[trial_num][0]
    elif align_to == 'feedback':
        start_time = trials.feedback_times[trial_num] - 0.6
    else:
        raise ValueError(f"Unknown align_to: {align_to}")

    if np.isnan(start_time):
        return np.full(int(duration / interval), np.nan)

    # Find wheel start index (closest timestamp to start_time)
    wheel_start_index = _find_nearest_index(whlt, start_time)

    # Build trajectory
    n_bins = int(duration / interval)
    trajectory = np.full(n_bins, np.nan)

    for i in range(n_bins):
        t = start_time + i * interval
        wheel_end_index = _find_nearest_index(whlt, t)

        # Truncate past relevant event boundaries
        truncate = False
        if align_to == 'QP':
            try:
                if trials.goCueTrigger_times[trial_num] < whlt[wheel_end_index]:
                    truncate = True
            except (AttributeError, KeyError):
                if trials.goCue_times[trial_num] < whlt[wheel_end_index]:
                    truncate = True
        elif align_to in ('goCue', 'goCue_pre'):
            if (trials.feedback_times[trial_num] + interval) < whlt[wheel_end_index]:
                truncate = True

        if not truncate:
            trajectory[i] = whlpos[wheel_end_index] - whlpos[wheel_start_index]

    return trajectory


def _find_nearest_index(timestamps, target):
    """Find index of the closest timestamp to target."""
    idx = np.searchsorted(timestamps, target)
    if idx == 0:
        return 0
    elif idx == len(timestamps):
        return len(timestamps) - 1
    else:
        left_diff = target - timestamps[idx - 1]
        right_diff = timestamps[idx] - target
        return idx - 1 if left_diff <= right_diff else idx


# =============================================================================
# PLOTTING UTILITIES
# =============================================================================

def makepretty():
    """Format psychometric plots with standard axes."""
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.plot((0, 0), (0, 1), 'k:')
    plt.plot((-100, 100), (.5, .5), 'k:')
    plt.gca().set(ylim=[-.05, 1.05], xlabel='contrast (%)', ylabel='proportion rightward')
    sns.despine(offset=10, trim=True)


def calculate_accuracy_bycontrast(trials_bunch, contrast_mask):
    """
    Calculate accuracy (proportion correct) for a masked subset of trials.

    Parameters
    ----------
    trials_bunch : Bunch
        Trials data.
    contrast_mask : ndarray of bool
        Boolean mask selecting which trials to include.

    Returns
    -------
    accuracy : float
        Proportion correct (feedbackType == 1) among selected trials.
    """
    feedback = trials_bunch.feedbackType[contrast_mask]
    n_trials = len(feedback)
    if n_trials == 0:
        return np.nan
    return np.sum(feedback == 1) / n_trials


def plot_psychometric_curves(stim_data, nonstim_data, fit_kwargs, title='',
                             save_path=None, plot_for_paper=False,
                             n_stim_trials=None, n_nonstim_trials=None,
                             n_mice=None):
    """
    Plot psychometric curves for stim and nonstim conditions, split by block.

    Parameters
    ----------
    stim_data : dict
        Output of organize_psychodata for stim trials.
    nonstim_data : dict
        Output of organize_psychodata for nonstim trials.
    fit_kwargs : dict
        Psychometric fit parameters.
    title : str
        Plot title.
    save_path : str or None
        Path to save figure. None = show interactively.
    plot_for_paper : bool
        Use publication-quality formatting.
    n_stim_trials : int or None
        Total stim trials for annotation.
    n_nonstim_trials : int or None
        Total nonstim trials for annotation.
    n_mice : int or None
        Number of unique mice for annotation.
    """
    colours = {0.2: 'xkcd:tangerine', 0.8: 'xkcd:violet', 0.5: 'k'}
    x_fit = np.arange(-100, 100)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for ax, data, label in [(ax1, nonstim_data, 'Nonstim'), (ax2, stim_data, 'Stim')]:
        for pL, da in data.items():
            if da.shape[1] < 2:
                continue
            try:
                pars, L = psy.mle_fit_psycho(da, 'erf_psycho_2gammas', **fit_kwargs)
                ax.plot(da[0, :], da[2, :], 'o', color=colours.get(pL, 'grey'))
                ax.plot(x_fit, psy.erf_psycho_2gammas(pars, x_fit),
                        color=colours.get(pL, 'grey'),
                        label=f'{int(pL * 100)}% left')
            except Exception:
                ax.plot(da[0, :], da[2, :], 'o', color=colours.get(pL, 'grey'))

        ax.set_title(f'{label} — {title}')
        ax.set_xlabel('Contrast (%)')
        ax.set_ylabel('Proportion rightward')
        ax.axhline(0.5, color='k', linestyle=':', alpha=0.5)
        ax.axvline(0, color='k', linestyle=':', alpha=0.5)
        ax.set_ylim(-0.05, 1.05)
        ax.legend()
        sns.despine(ax=ax, offset=10, trim=True)

    # Add trial/mouse count annotation
    annotation_parts = []
    if n_mice is not None:
        annotation_parts.append(f'n = {n_mice} mice')
    if n_nonstim_trials is not None:
        annotation_parts.append(f'{n_nonstim_trials} control trials')
    if n_stim_trials is not None:
        annotation_parts.append(f'{n_stim_trials} stim trials')

    if annotation_parts:
        fig.text(0.5, 0.01, ' | '.join(annotation_parts),
                 ha='center', fontsize=10, style='italic', color='grey')
        fig.subplots_adjust(bottom=0.12)

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_bias_shift_comparison(bias_stim, bias_nonstim, title='',
                               save_path=None, plot_for_paper=False):
    """
    Plot per-session bias shift comparison between stim and nonstim.

    Parameters
    ----------
    bias_stim : ndarray
        Per-session stim bias shift values.
    bias_nonstim : ndarray
        Per-session nonstim bias shift values.
    title : str
    save_path : str or None
    plot_for_paper : bool
    """
    # Remove NaN sessions
    valid = ~np.isnan(bias_stim) & ~np.isnan(bias_nonstim)
    bs = bias_stim[valid]
    bn = bias_nonstim[valid]

    if len(bs) == 0:
        print("No valid sessions for bias shift comparison.")
        return

    fig, ax = plt.subplots(figsize=(6, 6))

    # Paired lines
    for i in range(len(bs)):
        ax.plot([0, 1], [bn[i], bs[i]], 'k-', alpha=0.3, linewidth=0.8)

    ax.scatter(np.zeros(len(bn)), bn, color='grey', s=40, zorder=5, label='Nonstim')
    ax.scatter(np.ones(len(bs)), bs, color='dodgerblue', s=40, zorder=5, label='Stim')

    # Means
    ax.scatter([0], [np.mean(bn)], color='black', s=120, marker='D', zorder=10)
    ax.scatter([1], [np.mean(bs)], color='navy', s=120, marker='D', zorder=10)

    # Stats
    if len(bs) > 1:
        t_stat, p_val = stats.ttest_rel(bn, bs)
        ax.set_title(f'{title}\np = {p_val:.4f} (paired t-test, n={len(bs)})')
    else:
        ax.set_title(title)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Nonstim', 'Stim'])
    ax.set_ylabel('Bias Shift (sum across contrasts)')
    ax.legend()
    sns.despine(offset=10)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_wheel_comparison(Rblock_stim, Lblock_stim, Rblock_nonstim, Lblock_nonstim,
                          align_to='QP', interval=0.1, duration=10,
                          title='', save_path=None):
    """
    Plot mean wheel trajectories by block and stim condition.

    Parameters
    ----------
    Rblock_stim, Lblock_stim : ndarray
        Wheel trajectories (n_trials × n_bins) for R/L blocks, stim.
    Rblock_nonstim, Lblock_nonstim : ndarray
        Same for nonstim.
    align_to : str
    interval : float
    duration : float
    title : str
    save_path : str or None
    """
    x = np.arange(0, duration, interval)

    fig, ax = plt.subplots(figsize=(10, 6))

    for data, color, style, label in [
        (Lblock_nonstim, 'xkcd:violet', 'solid', 'CTR L-block'),
        (Rblock_nonstim, 'xkcd:tangerine', 'solid', 'CTR R-block'),
        (Lblock_stim, 'xkcd:violet', 'dashed', 'Stim L-block'),
        (Rblock_stim, 'xkcd:tangerine', 'dashed', 'Stim R-block'),
    ]:
        if isinstance(data, np.ndarray) and data.ndim == 2 and data.shape[0] > 0:
            mean = np.nanmean(data, axis=0)
            n = np.sum(~np.isnan(data), axis=0)
            sem = np.nanstd(data, axis=0, ddof=1) / np.sqrt(np.maximum(n, 1))
            ax.plot(x[:len(mean)], mean, color=color, linestyle=style, label=label)
            ax.fill_between(x[:len(mean)], mean - sem, mean + sem,
                            color=color, alpha=0.15)

    ax.set_xlabel(f'Time from {align_to} onset (s)')
    ax.set_ylabel('Cumulative wheel movement')
    ax.set_title(title or 'Wheel movement: stim vs control')
    ax.legend()
    sns.despine(offset=10)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_bars_by_mouse(mouse_df, y_limits, mode='Bias_LC',
                       save_path=None, prefix='', stim_label='Stim'):
    """
    Plot paired bar + line chart comparing control vs stim per mouse.

    Each mouse is a connected dot-pair overlaid on group mean bars with SEM
    error bars. A paired t-test p-value is annotated.

    Parameters
    ----------
    mouse_df : DataFrame
        Per-mouse summary with columns for control and stim values.
        Expected column names depend on `mode` (see below).
    y_limits : array-like
        [ymin, ymax] for the plot.
    mode : str
        Which metric to plot. One of:
        'Bias_LC', 'Bias_All', 'Accuracy_HC', 'Accuracy_0', 'RT', 'QP'
    save_path : str or None
        Directory to save figure. None = show interactively.
    prefix : str
        Filename prefix for saved figure.
    stim_label : str
        Label for the stim condition (e.g., 'SNr Inhibition', 'Stim').

    Returns
    -------
    p_val : float
        P-value from paired t-test.
    """
    mode_config = {
        'Bias_LC': {
            'col_ctrl': 'Bias_Values_Nonstim_LC',
            'col_stim': 'Bias_Values_Stim_LC',
            'title': 'Bias Shift (Low Contrast)',
            'ylabel': 'Bias shift',
        },
        'Bias_All': {
            'col_ctrl': 'Bias_Values_Nonstim',
            'col_stim': 'Bias_Values_Stim',
            'title': 'Bias Shift (All Contrasts)',
            'ylabel': 'Bias shift',
        },
        'Accuracy_HC': {
            'col_ctrl': 'Accuracy_HC_Control',
            'col_stim': 'Accuracy_HC_Stim',
            'title': 'High Contrast Accuracy',
            'ylabel': 'Accuracy',
        },
        'Accuracy_0': {
            'col_ctrl': 'Accuracy_0_Control',
            'col_stim': 'Accuracy_0_Stim',
            'title': 'Zero Contrast Accuracy',
            'ylabel': 'Accuracy',
        },
        'RT': {
            'col_ctrl': 'RT_Control',
            'col_stim': 'RT_Stim',
            'title': 'Reaction Time',
            'ylabel': 'Reaction time (s)',
        },
        'QP': {
            'col_ctrl': 'QP_Control',
            'col_stim': 'QP_Stim',
            'title': 'Quiescence Time',
            'ylabel': 'Quiescence time (s)',
        },
    }

    if mode not in mode_config:
        print(f"Unknown mode '{mode}'. Available: {list(mode_config.keys())}")
        return np.nan

    cfg = mode_config[mode]
    col_ctrl = cfg['col_ctrl']
    col_stim = cfg['col_stim']

    # Drop mice with NaN in either column
    df_plot = mouse_df[[col_ctrl, col_stim]].dropna()
    ctrl_vals = df_plot[col_ctrl].values
    stim_vals = df_plot[col_stim].values
    n_mice = len(df_plot)

    if n_mice < 2:
        print(f"Not enough mice for paired test in mode '{mode}' (n={n_mice})")
        return np.nan

    # Paired t-test
    t_stat, p_val = stats.ttest_rel(ctrl_vals, stim_vals)

    # Plot
    plt.figure(figsize=(4, 5))

    # Individual mice (spaghetti lines)
    for i in range(n_mice):
        plt.plot([0, 1], [ctrl_vals[i], stim_vals[i]],
                 color='gray', alpha=0.5, linewidth=1, marker='o', markersize=5)

    # Group mean bars
    mean_ctrl = np.mean(ctrl_vals)
    mean_stim = np.mean(stim_vals)
    sem_ctrl = stats.sem(ctrl_vals)
    sem_stim = stats.sem(stim_vals)

    plt.bar(0, mean_ctrl, color='black', alpha=0.3, width=0.6, label='Control')
    plt.bar(1, mean_stim, color='blue', alpha=0.3, width=0.6, label=stim_label)

    # Error bars on means
    plt.errorbar(0, mean_ctrl, yerr=sem_ctrl, color='black', linewidth=3, capsize=5)
    plt.errorbar(1, mean_stim, yerr=sem_stim, color='blue', linewidth=3, capsize=5)

    # Aesthetics
    plt.xticks([0, 1], ['Control', stim_label])
    plt.ylabel(cfg['ylabel'])
    plt.title(f"{cfg['title']}\n(n={n_mice} mice)")
    plt.xlim(-0.6, 1.6)
    plt.ylim(y_limits[0], y_limits[1])

    # P-value annotation
    y_max = max(np.max(ctrl_vals), np.max(stim_vals))
    plt.text(0.5, y_max * 1.05, f'p = {p_val:.4f}', ha='center', fontsize=12)

    sns.despine()
    plt.tight_layout()

    if save_path:
        fname = f"{save_path}/{prefix}_PerMouse_{mode}.png"
        plt.savefig(fname, dpi=150, bbox_inches='tight')
        print(f"Saved: {fname}")
        plt.close()
    else:
        plt.show()

    return p_val

def auto_ylimits(df, col_ctrl, col_stim, padding_frac=0.1):
    """Compute y-limits from data with padding."""
    all_vals = pd.concat([df[col_ctrl], df[col_stim]]).dropna()
    ymin = all_vals.min()
    ymax = all_vals.max()
    margin = (ymax - ymin) * padding_frac
    return [ymin - margin, ymax + margin]