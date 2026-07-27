from datetime import datetime, timezone
# import json
import numpy as np
import sys
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import hashlib
from ibllib.io.raw_data_loaders import load_data
from one.api import ONE
# import brainbox.io.one as bbone
# from brainbox.io.one import load_spike_sorting
from brainbox.io.one import SpikeSortingLoader, load_lfp
from brainbox import singlecell
# import brainbox.plot as bbp
from iblatlas.atlas import AllenAtlas, BrainRegions
# import brainbox.behavior.pyschofit as psy
# from ibl_pipeline import behavior, acquisition, subject
# from ibl_pipeline.analyses.behavior import PsychResultsBlock, PsychResults
from scipy import stats
import statistics
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

from miska_analysis.functions_optostim import (
    generate_pseudo_sessions,
    isbiasblockselective_perm_vector,
    peri_event_time_histogram,
    signed_contrast,
)
from optostim_preprocessing import (
    load_session, unit_qc_table, recorded_region_flags,
    compute_drift_unit_ids, compute_qp_drift_unit_ids,
    compute_qp_nonstationarity_metrics, save_qp_nonstationarity_qc,
    make_qc_dir, select_insertions, apply_beginning_block_trial_filter,
    cluster_peak_waveform, UnitQCParams,
    apply_glmhmm_opto_trial_policy, coerce_glmhmm_engaged_indices,
)
from metadata_optostim_new import insertions as optostim_insertions

# GLM-HMM engagement labels (shared with the CD pipeline).
sys.path.append('/Users/natemiska/int-brain-lab/GLM-HMM')
from psychometric_utils import get_glmhmm_indices
import pickle
with open("/Users/natemiska/int-brain-lab/GLM-HMM/all_subject_states.csv", 'rb') as _pf:
    state_probability = pickle.load(_pf)

# All user-tunable options, the condition registry, and the per-PID trial
# exclusions now live in BS_config.py.
import BS_config as _bs_cfg
from BS_config import (
    condition, hemisphere_filter, onset_alignment, onset_alignments_to_run,
    alignment_time_windows, figure_prefix, start_pid_idx,
    t_before, t_after, bin_size, smoothing, post_smooth_window_ms,
    IBL_quality_label_threshold, firing_rate_threshold,
    plot_each_cluster, plot_only_BS_units, only_plot_FR,
    CONDITIONS, TRIALS_TO_REMOVE, resolve_condition,
    analysis_mode, insertion_brain_regions, insertion_conditions, insertion_pids,
    use_GLMHMM_engaged_indices, opto_trials_GLMHMM, n_states,
    beginning_block_trials_remove, remove_drift_units, drift_threshold, drift_window_s, drift_epoch,
    remove_nonstationary_units, nonstationarity_n_segments,
    nonstationarity_min_trials, nonstationarity_min_trials_per_segment,
    nonstationarity_min_trials_per_block_segment,
    nonstationarity_low_fr_fraction_of_median,
    nonstationarity_min_median_fr_hz,
    max_qp_fr_segment_range_frac, max_qp_resid_drift_range_frac,
    max_qp_resid_drift_cv, max_qp_resid_abs_rho_time,
    max_qp_low_activity_fraction, max_qp_max_low_activity_run,
    min_qp_block_effect_sign_consistency,
    max_qp_block_effect_segment_cv, max_qp_block_effect_dominance,
    match_nonstim_to_inhibition_range,
    normalize_mode, scalar_baseline_window, scalar_min_fr, zero_nan_threshold,
    save_diagnostic_traces, save_raw_block_peths, diagnostic_random_seed,
    diagnostic_trialmatch_repeats, diagnostic_min_events_per_peth,
    diagnostic_crossfit_guard_trials,
    save_futureproof_sufficient_stats,
    use_batched_peths, peth_cluster_batch_size,
    save_legacy_base_pickle, save_combined_alignment_pickle,
    save_qc_outputs, figures_path,
    compute_recorded_region, compute_min_IBL_label, compute_min_firing_rate, bs_output_path,
    bs_definition_trial_mode,
)
# (metadata_optostim pid-lists are no longer imported here; the post-analysis
#  masks now filter on the 'hemisphere' column added to clusters_info_DF.)

# one = ONE(base_url='https://alyx.internationalbrainlab.org')
# one=ONE(mode='remote')

one = ONE(base_url='https://alyx.internationalbrainlab.org', cache_dir=Path.home() / '/Users/natemiska/Downloads/ONE/alyx.internationalbrainlab.org')
# temporarily uses local cache

ba = AllenAtlas()
br = BrainRegions()

suppress_print_output = 1

#####################################################################################
# Resolve the per-condition trial-list bundle from BS_config, filtered by hemisphere.
#   - condition and hemisphere_filter are set in BS_config.py
#   - resolve_condition() returns concatenated lists across the requested
#     hemisphere(s), plus a pid -> hemisphere lookup used below to tag each
#     cluster row in clusters_info_DF.
#####################################################################################
# Metadata-driven selection (mirrors CD). Falls back to the legacy
# resolve_condition() registry when analysis_mode == 'legacy_condition'.
if analysis_mode == 'all_insertions':
    selected_insertions = select_insertions(
        optostim_insertions,
        brain_regions=insertion_brain_regions,
        conditions=insertion_conditions,
        pids_filter=insertion_pids,
    )
    pids = [ins['PID'] for ins in selected_insertions]
    inhibition_trials_range_list = [ins['opto inhibition trials'] for ins in selected_insertions]
    excitation_trials_range_list = [[] for _ in selected_insertions]   # excitation retired
    light_artifact_units_list    = [[] for _ in selected_insertions]   # auto-detected now
    pid_to_hemisphere   = {ins['PID']: ins['condition'] for ins in selected_insertions}
    pid_to_region       = {ins['PID']: ins['brain region'] for ins in selected_insertions}
    pid_to_mouse        = {ins['PID']: ins.get('mouse', 'nan') for ins in selected_insertions}
    pid_to_hemi_stim    = {ins['PID']: ins.get('hemisphere stim', 'nan') for ins in selected_insertions}
    pid_to_hemi_recorded= {ins['PID']: ins.get('hemisphere recorded', 'nan') for ins in selected_insertions}
else:
    (pids,
     excitation_trials_range_list,
     inhibition_trials_range_list,
     light_artifact_units_list,
     pid_to_hemisphere) = resolve_condition(condition, hemisphere_filter)
    pid_to_region = {p_: 'nan' for p_ in pids}
    pid_to_mouse = {p_: 'nan' for p_ in pids}
    pid_to_hemi_stim = {p_: 'nan' for p_ in pids}
    pid_to_hemi_recorded = {p_: 'nan' for p_ in pids}
print(f'Analysis mode: {analysis_mode}; selected {len(pids)} insertions')

# Hemisphere suffix appended to saved-pickle filenames so different hemisphere
# selections of the same condition do not overwrite each other on disk.
_hemi_suffix = '' if hemisphere_filter == 'both' else '_' + hemisphere_filter


def drop_trials_from_arrays(trial_idx_arrays, trials_to_remove):
    """Remove specified trial numbers from each int array, preserving order."""
    to_remove = np.asarray(trials_to_remove)
    return [arr[~np.isin(arr, to_remove)] for arr in trial_idx_arrays]

def rolling_window_mean_1d(arr: np.ndarray, window_bins: int) -> np.ndarray:
    """
    Computes a CAUSAL sliding-window mean for a single 1D time series.
    """
    if window_bins <= 1:
        return arr

    # Pad ONLY on the left (the past) with the first value (edge padding).
    # pad_width is now a single tuple: (pad_before, pad_after)
    arr_padded = np.pad(arr, (window_bins, 0), mode='edge')

    # Cumulative sum trick for fast calculation
    csum = np.cumsum(arr_padded)

    # Subtract the lagged csum: (csum[t] - csum[t - window]) / window
    win_sum = csum[window_bins:] - csum[:-window_bins]

    return win_sum / window_bins

def logical_and_multiple(bool_arrays):
    """
    Returns a boolean array that is True only where all input arrays are True.
    """
    if not bool_arrays:
        return np.array([], dtype=bool)

    # 1. Stack the boolean arrays into a 2D matrix
    # Shape will be (N_arrays, Array_length)
    stacked_arrays = np.stack(bool_arrays, axis=0)

    # 2. Check if ALL values along the first axis (axis=0, the array dimension) are True
    # The result has shape (Array_length,)
    result = np.all(stacked_arrays, axis=0)

    return result


def _stable_rng(pid, salt=''):
    """Deterministic per-PID RNG so diagnostic splits are reproducible."""
    msg = f'{diagnostic_random_seed}:{pid}:{salt}'.encode('utf-8')
    seed = int.from_bytes(hashlib.sha256(msg).digest()[:8], 'little') % (2**32)
    return np.random.default_rng(seed)


def _as_sorted_int_array(values):
    if values is None:
        return np.array([], dtype=int)
    return np.asarray(sorted(int(v) for v in values), dtype=int)


def _choice_side_from_trials(trials, trial_indices):
    """Return choice side using the local side code: -1 left, +1 right."""
    trial_indices = np.asarray(trial_indices, dtype=int)
    choice = np.asarray(trials.choice[trial_indices], dtype=float)
    side = np.zeros(len(choice), dtype=int)
    # IBL convention used elsewhere in this codebase: choice == -1 is rightward,
    # choice == +1 is leftward. Convert to the side code used for block side.
    side[np.isclose(choice, 1.0)] = -1
    side[np.isclose(choice, -1.0)] = 1
    return side


def _prior_choice_congruent_trial_mask(trials, trial_indices):
    """Trials where choice is congruent with the high-probability block side."""
    trial_indices = np.asarray(trial_indices, dtype=int)
    if trial_indices.size == 0:
        return np.zeros(0, dtype=bool)
    probability_left = np.asarray(trials.probabilityLeft[trial_indices], dtype=float)
    block_side = np.zeros(trial_indices.size, dtype=int)
    block_side[probability_left > 0.5] = -1
    block_side[probability_left < 0.5] = 1
    choice_side = _choice_side_from_trials(trials, trial_indices)
    return (block_side != 0) & (choice_side != 0) & (choice_side == block_side)


def _apply_bs_definition_trial_mode(trials, trial_arrays, mode):
    """Apply the configured BS trial-definition filter to trial-number arrays."""
    mode = str(mode or 'standard')
    labels = [
        'excitation', 'inhibition', 'nonstim', 'nonstim_ex', 'nonstim_in',
    ]
    if mode == 'standard':
        filtered = [np.asarray(arr, dtype=int) for arr in trial_arrays]
        counts = {
            'bs_definition_trial_mode': mode,
            'bs_definition_reason': 'standard',
        }
        for label, arr in zip(labels, filtered):
            counts[f'n_{label}_trials_before_bs_definition'] = int(arr.size)
            counts[f'n_{label}_trials_after_bs_definition'] = int(arr.size)
            counts[f'n_{label}_trials_removed_by_bs_definition'] = 0
        return filtered, counts
    if mode != 'prior_choice_congruent':
        raise ValueError(
            "bs_definition_trial_mode must be 'standard' or "
            "'prior_choice_congruent'"
        )

    filtered = []
    counts = {
        'bs_definition_trial_mode': mode,
        'bs_definition_reason': 'prior_choice_congruent',
    }
    for label, arr in zip(labels, trial_arrays):
        arr = np.asarray(arr, dtype=int)
        keep = _prior_choice_congruent_trial_mask(trials, arr)
        filtered_arr = arr[keep]
        filtered.append(filtered_arr)
        counts[f'n_{label}_trials_before_bs_definition'] = int(arr.size)
        counts[f'n_{label}_trials_after_bs_definition'] = int(filtered_arr.size)
        counts[f'n_{label}_trials_removed_by_bs_definition'] = int(arr.size - filtered_arr.size)
    return filtered, counts


def _alignment_prefix(alignment):
    mapping = {
        'Laser onset': 'LaserOnset',
        'Go cue onset': 'GoCueOnset',
        'Feedback': 'Feedback',
    }
    if alignment in mapping:
        return mapping[alignment]
    return str(alignment).replace(' ', '').replace('/', '_')


def _alignments_for_run():
    alignments = list(onset_alignments_to_run or ())
    if not alignments:
        alignments = [onset_alignment]
    if onset_alignment not in alignments:
        alignments.insert(0, onset_alignment)
    out = []
    for align in alignments:
        if align not in out:
            out.append(align)
    return out


def _alignment_window(alignment):
    if alignment in alignment_time_windows:
        return tuple(alignment_time_windows[alignment])
    return (t_before, t_after)


def _peth_time_for_alignment(alignment):
    tb, ta = _alignment_window(alignment)
    return np.arange(-tb, ta, bin_size)


def _feedback_times_for_trials(trials_obj, trial_numbers=None):
    if trial_numbers is None:
        if hasattr(trials_obj, 'feedback_times'):
            return np.asarray(trials_obj.feedback_times, dtype=float)
        return np.asarray(trials_obj.intervals[:, 1], dtype=float)
    trial_numbers = _as_sorted_int_array(trial_numbers)
    if hasattr(trials_obj, 'feedback_times'):
        return np.asarray(trials_obj.feedback_times[trial_numbers], dtype=float)
    return np.asarray(trials_obj.intervals[trial_numbers, 1], dtype=float)


def _event_times_for_alignment(trials_obj, alignment, trial_numbers=None):
    if trial_numbers is not None:
        trial_numbers = _as_sorted_int_array(trial_numbers)
    if alignment == 'Laser onset':
        if trial_numbers is None:
            return np.asarray(trials_obj.intervals[:, 0], dtype=float)
        return np.asarray(trials_obj.intervals[trial_numbers, 0], dtype=float)
    if alignment == 'Feedback':
        return _feedback_times_for_trials(trials_obj, trial_numbers)
    if trial_numbers is None:
        return np.asarray(trials_obj.goCue_times, dtype=float)
    return np.asarray(trials_obj.goCue_times[trial_numbers], dtype=float)


def _alignment_output_path(alignment):
    base = Path(bs_output_path).expanduser()
    known = tuple(_alignment_prefix(a) for a in ('Laser onset', 'Go cue onset', 'Feedback'))
    stem = base.stem
    for suffix in known:
        token = f'_{suffix}'
        if stem.endswith(token):
            stem = stem[:-len(token)]
            break
    return base.with_name(f'{stem}_{_alignment_prefix(alignment)}{base.suffix}')


def _combined_alignment_output_path():
    base = Path(bs_output_path).expanduser()
    known = tuple(_alignment_prefix(a) for a in ('Laser onset', 'Go cue onset', 'Feedback'))
    stem = base.stem
    for suffix in known:
        token = f'_{suffix}'
        if stem.endswith(token):
            stem = stem[:-len(token)]
            break
    return base.with_name(f'{stem}_by_alignment{base.suffix}')


# The legacy single-alignment code path below uses module-level t_before/t_after.
# Keep it in sync with the primary alignment's configured window.
t_before, t_after = _alignment_window(onset_alignment)


def _split_trial_numbers(values, rng):
    values = _as_sorted_int_array(values)
    if len(values) == 0:
        return values.copy(), values.copy()
    shuffled = rng.permutation(values)
    n_a = int(np.ceil(len(shuffled) / 2))
    return _as_sorted_int_array(shuffled[:n_a]), _as_sorted_int_array(shuffled[n_a:])


def _sample_trial_numbers(values, n, rng):
    values = _as_sorted_int_array(values)
    n = int(min(max(n, 0), len(values)))
    if n == 0:
        return np.array([], dtype=int)
    return _as_sorted_int_array(rng.choice(values, size=n, replace=False))


def _trial_onsets(trials, trial_numbers, alignment=None):
    trial_numbers = _as_sorted_int_array(trial_numbers)
    if len(trial_numbers) == 0:
        return np.array([], dtype=float)
    return _event_times_for_alignment(trials, alignment or onset_alignment, trial_numbers)


def _block_run_and_half_ids(trials, guard_trials=0):
    """Return block-run ids and contiguous early/late half labels per trial.

    Half 0/1 is assigned from the midpoint of each uninterrupted probabilityLeft
    run. A symmetric guard about that midpoint is labelled -1 and is therefore
    excluded from both halves. Splitting whole block runs this way keeps the
    sign-training trials temporally separate from the held-out control/opto
    trials while ensuring that both folds sample every available bias block.
    """
    probability_left = np.asarray(trials.probabilityLeft, dtype=float)
    n_trials = probability_left.size
    if n_trials == 0:
        return np.array([], dtype=int), np.array([], dtype=int)

    same_as_previous = np.isclose(
        probability_left[1:], probability_left[:-1], equal_nan=True)
    starts_new_run = np.r_[True, ~same_as_previous]
    run_ids = np.cumsum(starts_new_run).astype(int) - 1
    half_ids = np.full(n_trials, -1, dtype=int)
    guard = max(int(guard_trials), 0)

    for run_id in np.unique(run_ids):
        run_trials = np.flatnonzero(run_ids == run_id)
        if run_trials.size == 0:
            continue
        split = int((int(run_trials[0]) + int(run_trials[-1]) + 1) // 2)
        early_stop = split - guard
        late_start = split + guard
        half_ids[run_trials[run_trials < early_stop]] = 0
        half_ids[run_trials[run_trials >= late_start]] = 1
    return run_ids, half_ids


def _trials_in_half(values, half_ids, half):
    values = _as_sorted_int_array(values)
    if values.size == 0:
        return values
    valid = (values >= 0) & (values < len(half_ids))
    values = values[valid]
    return values[half_ids[values] == int(half)]


def _nearest_control_stim_pairs(control_trials, stim_trials, run_ids):
    """Greedily pair control/opto trials by trial distance within each block run.

    Both returned arrays have the same length. Restricting matches to the same
    uninterrupted block run makes the held-out control distribution track the
    opto distribution in both block identity and session time.
    """
    control_trials = _as_sorted_int_array(control_trials)
    stim_trials = _as_sorted_int_array(stim_trials)
    if control_trials.size == 0 or stim_trials.size == 0:
        empty = np.array([], dtype=int)
        return empty, empty

    matched_control = []
    matched_stim = []
    candidate_runs = np.intersect1d(
        np.unique(run_ids[control_trials]), np.unique(run_ids[stim_trials]))
    for run_id in candidate_runs:
        available_control = control_trials[run_ids[control_trials] == run_id].copy()
        available_stim = stim_trials[run_ids[stim_trials] == run_id].copy()
        while available_control.size and available_stim.size:
            distances = np.abs(
                available_control[:, None] - available_stim[None, :])
            control_i, stim_i = np.unravel_index(
                int(np.argmin(distances)), distances.shape)
            matched_control.append(int(available_control[control_i]))
            matched_stim.append(int(available_stim[stim_i]))
            available_control = np.delete(available_control, control_i)
            available_stim = np.delete(available_stim, stim_i)
    return _as_sorted_int_array(matched_control), _as_sorted_int_array(matched_stim)


def _event_set_from_trial_numbers(trials, block80, block20, alignment):
    block80 = _as_sorted_int_array(block80)
    block20 = _as_sorted_int_array(block20)
    all_trials = _as_sorted_int_array(np.concatenate([block80, block20]))
    return {
        '80': _trial_onsets(trials, block80, alignment),
        '20': _trial_onsets(trials, block20, alignment),
        'all': _trial_onsets(trials, all_trials, alignment),
    }


def _make_block_crossfit_sets(trials, nonstim_80, nonstim_20,
                              stim_80, stim_20, alignment):
    """Build two temporally separated, trial-matched cross-fit folds.

    For evaluation half A, control trials from half B establish the unit's block
    preference; matched control and opto trials from A are evaluated. The second
    fold swaps A/B. Control/opto evaluation counts are matched within the same
    uninterrupted block run. Neither evaluation condition contributes to its
    fold's sign decision.
    """
    run_ids, half_ids = _block_run_and_half_ids(
        trials, diagnostic_crossfit_guard_trials)
    nonstim_by_half = {
        half: {
            '80': _trials_in_half(nonstim_80, half_ids, half),
            '20': _trials_in_half(nonstim_20, half_ids, half),
        }
        for half in (0, 1)
    }
    stim_by_half = {
        half: {
            '80': _trials_in_half(stim_80, half_ids, half),
            '20': _trials_in_half(stim_20, half_ids, half),
        }
        for half in (0, 1)
    }

    folds = []
    counts = {'crossfit_guard_trials': int(diagnostic_crossfit_guard_trials)}
    for eval_half, fold_name in ((0, 'a'), (1, 'b')):
        reference_half = 1 - eval_half
        matched_control = {}
        matched_stim = {}
        for block in ('80', '20'):
            matched_control[block], matched_stim[block] = _nearest_control_stim_pairs(
                nonstim_by_half[eval_half][block],
                stim_by_half[eval_half][block],
                run_ids,
            )
            counts[f'crossfit_{fold_name}_control_{block}'] = int(
                matched_control[block].size)
            counts[f'crossfit_{fold_name}_stim_{block}'] = int(
                matched_stim[block].size)
            counts[f'crossfit_{fold_name}_reference_{block}'] = int(
                nonstim_by_half[reference_half][block].size)

        folds.append({
            'name': fold_name,
            'trial_numbers': {
                'reference': {
                    '80': nonstim_by_half[reference_half]['80'].astype(np.int32),
                    '20': nonstim_by_half[reference_half]['20'].astype(np.int32),
                },
                'control_eval': {
                    '80': matched_control['80'].astype(np.int32),
                    '20': matched_control['20'].astype(np.int32),
                },
                'stim_eval': {
                    '80': matched_stim['80'].astype(np.int32),
                    '20': matched_stim['20'].astype(np.int32),
                },
            },
            'reference': _event_set_from_trial_numbers(
                trials,
                nonstim_by_half[reference_half]['80'],
                nonstim_by_half[reference_half]['20'],
                alignment,
            ),
            'control_eval': _event_set_from_trial_numbers(
                trials, matched_control['80'], matched_control['20'], alignment),
            'stim_eval': _event_set_from_trial_numbers(
                trials, matched_stim['80'], matched_stim['20'], alignment),
        })
    return folds, counts


def _make_diagnostic_trial_sets(pid, trials, nonstim_delta, nonstim_80, nonstim_20,
                                stim_80, stim_20, alignment=None):
    """Create deterministic split-half and trial-count-matched control sets."""
    alignment = alignment or onset_alignment
    rng_split = _stable_rng(pid, 'split')
    nonstim_delta = _as_sorted_int_array(nonstim_delta)
    nonstim_80 = _as_sorted_int_array(nonstim_80)
    nonstim_20 = _as_sorted_int_array(nonstim_20)
    stim_80 = _as_sorted_int_array(stim_80)
    stim_20 = _as_sorted_int_array(stim_20)

    nonstim_other = np.setdiff1d(nonstim_delta, np.union1d(nonstim_80, nonstim_20))
    a80, b80 = _split_trial_numbers(nonstim_80, rng_split)
    a20, b20 = _split_trial_numbers(nonstim_20, rng_split)
    aother, bother = _split_trial_numbers(nonstim_other, rng_split)

    split_a_all = _as_sorted_int_array(np.concatenate([a80, a20, aother]))
    split_b_all = _as_sorted_int_array(np.concatenate([b80, b20, bother]))

    n_match_80 = min(len(nonstim_80), len(stim_80))
    n_match_20 = min(len(nonstim_20), len(stim_20))
    n_repeats = max(1, int(diagnostic_trialmatch_repeats))
    match_repeats = []
    for rep in range(n_repeats):
        rng_match = _stable_rng(pid, f'trialmatch:{rep}')
        m80 = _sample_trial_numbers(nonstim_80, n_match_80, rng_match)
        m20 = _sample_trial_numbers(nonstim_20, n_match_20, rng_match)
        match_repeats.append({
            '80': _trial_onsets(trials, m80, alignment),
            '20': _trial_onsets(trials, m20, alignment),
            'all': _trial_onsets(trials, np.concatenate([m80, m20]), alignment),
        })

    block_crossfit, crossfit_counts = _make_block_crossfit_sets(
        trials, nonstim_80, nonstim_20, stim_80, stim_20, alignment)

    return {
        'split_a': {
            '80': _trial_onsets(trials, a80, alignment),
            '20': _trial_onsets(trials, a20, alignment),
            'all': _trial_onsets(trials, split_a_all, alignment),
        },
        'split_b': {
            '80': _trial_onsets(trials, b80, alignment),
            '20': _trial_onsets(trials, b20, alignment),
            'all': _trial_onsets(trials, split_b_all, alignment),
        },
        'trialmatched_repeats': match_repeats,
        'block_crossfit': block_crossfit,
        'counts': {
            'split_a_80': len(a80), 'split_a_20': len(a20),
            'split_b_80': len(b80), 'split_b_20': len(b20),
            'trialmatch_80': n_match_80, 'trialmatch_20': n_match_20,
            'trialmatch_repeats': n_repeats,
            **crossfit_counts,
        },
    }


def _pack_pseudo_block_labels(pseudo_20, pseudo_80, n_trials):
    """Pack filtered pseudo-session block identities into a compact uint8 matrix."""
    n_pseudo = min(len(pseudo_20), len(pseudo_80))
    packed = np.zeros((n_pseudo, int(n_trials)), dtype=np.uint8)
    for i in range(n_pseudo):
        idx20 = np.asarray(pseudo_20[i], dtype=int)
        idx80 = np.asarray(pseudo_80[i], dtype=int)
        idx20 = idx20[(idx20 >= 0) & (idx20 < n_trials)]
        idx80 = idx80[(idx80 >= 0) & (idx80 < n_trials)]
        packed[i, idx20] = 1
        packed[i, idx80] = 2
    return packed


def _crossfit_trial_number_payload(diagnostic_trial_sets):
    """Extract exact trial ids from event-time diagnostic structures."""
    if diagnostic_trial_sets is None:
        return {}
    out = {}
    for fold in diagnostic_trial_sets.get('block_crossfit', []):
        fold_name = str(fold['name'])
        out[fold_name] = {}
        for role, blocks in fold.get('trial_numbers', {}).items():
            out[fold_name][role] = {
                block: np.asarray(values, dtype=np.int32)
                for block, values in blocks.items()
            }
    return out


def _quiescent_fr_per_trial(unit_spike_times, trials_obj,
                            before_gocue_end_time=0.01):
    """Per-trial FR matching isbiasblockselective_perm_vector's QP window."""
    spike_times = np.asarray(unit_spike_times, dtype=float)
    win_ends = np.asarray(trials_obj.goCue_times, dtype=float) - float(
        before_gocue_end_time)
    quiescence = np.asarray(trials_obj.quiescencePeriod, dtype=float)
    win_starts = win_ends - quiescence
    left = np.searchsorted(spike_times, win_starts, side='left')
    right = np.searchsorted(spike_times, win_ends, side='left')
    duration = np.maximum(quiescence - float(before_gocue_end_time), 1e-9)
    with np.errstate(invalid='ignore', divide='ignore'):
        fr = (right - left).astype(float) / duration
    fr[~np.isfinite(win_starts) | ~np.isfinite(win_ends)] = np.nan
    return np.asarray(fr, dtype=np.float32)


def _pipeline_provenance():
    """Hash the source files that materially define the saved analysis."""
    source_paths = {
        'analysis_script': Path(__file__),
        'config': Path(_bs_cfg.__file__),
        'bs_statistic': Path(
            isbiasblockselective_perm_vector.__code__.co_filename),
        'preprocessing': Path(unit_qc_table.__code__.co_filename),
    }
    files = {}
    for name, path in source_paths.items():
        resolved = path.expanduser().resolve()
        try:
            digest = hashlib.sha256(resolved.read_bytes()).hexdigest()
        except OSError:
            digest = None
        files[name] = {'path': str(resolved), 'sha256': digest}
    return {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'python_version': sys.version,
        'numpy_version': np.__version__,
        'pandas_version': pd.__version__,
        'source_files': files,
    }


def _empty_trace(t_before_s=None, t_after_s=None):
    tb = t_before if t_before_s is None else t_before_s
    ta = t_after if t_after_s is None else t_after_s
    return np.full(len(np.arange(-tb, ta, bin_size)), np.nan, dtype=float)


def _peth_mean_for_events(unit_spike_times, events, t_before_s=None, t_after_s=None):
    tb = t_before if t_before_s is None else t_before_s
    ta = t_after if t_after_s is None else t_after_s
    events = np.asarray(events, dtype=float)
    events = events[np.isfinite(events)]
    if len(events) < diagnostic_min_events_per_peth:
        return _empty_trace(tb, ta)
    unit_spike_times = np.asarray(unit_spike_times, dtype=float)
    unit_spike_clusters = np.zeros(len(unit_spike_times), dtype=int)
    try:
        peths, _ = singlecell.calculate_peths(
            unit_spike_times, unit_spike_clusters, [0], events,
            tb, ta, bin_size, smoothing, True)
        return np.asarray(peths.means[0], dtype=float)
    except Exception:
        return _empty_trace(tb, ta)


def _normalize_delta_modes(block80, block20, all_mean, t_before_s=None):
    tb = t_before if t_before_s is None else t_before_s
    block80 = np.asarray(block80, dtype=float)
    block20 = np.asarray(block20, dtype=float)
    all_mean = np.asarray(all_mean, dtype=float)
    delta = block80 - block20

    denom = all_mean.copy()
    denom[denom == 0] = 0.1
    with np.errstate(invalid='ignore', divide='ignore'):
        per_bin = delta / denom

    if scalar_baseline_window is None:
        scalar = np.nanmean(all_mean)
    else:
        b0 = max(0, int(round((tb + scalar_baseline_window[0]) / bin_size)))
        b1 = int(round((tb + scalar_baseline_window[1]) / bin_size))
        b1 = min(len(all_mean), max(b0 + 1, b1))
        scalar = np.nanmean(all_mean[b0:b1])
    scalar = max(scalar, scalar_min_fr) if np.isfinite(scalar) else scalar_min_fr
    with np.errstate(invalid='ignore', divide='ignore'):
        baseline_scalar = delta / scalar

    z = zero_nan_threshold
    z80 = block80.copy(); z80[z80 <= z] = np.nan
    z20 = block20.copy(); z20[z20 <= z] = np.nan
    zall = all_mean.copy(); zall[zall <= z] = np.nan
    with np.errstate(invalid='ignore', divide='ignore'):
        zero_2_nan = (z80 - z20) / zall

    return {
        'per_bin': per_bin,
        'baseline_scalar': baseline_scalar,
        'zero_2_nan': zero_2_nan,
    }


def _diagnostic_raw_for_events(unit_spike_times, event_set,
                               t_before_s=None, t_after_s=None):
    m80 = _peth_mean_for_events(unit_spike_times, event_set['80'], t_before_s, t_after_s)
    m20 = _peth_mean_for_events(unit_spike_times, event_set['20'], t_before_s, t_after_s)
    n80 = int(np.sum(np.isfinite(np.asarray(event_set['80'], dtype=float))))
    n20 = int(np.sum(np.isfinite(np.asarray(event_set['20'], dtype=float))))
    nall = int(np.sum(np.isfinite(np.asarray(event_set['all'], dtype=float))))
    # In the matched/cross-fit sets, `all` is exactly 80 + 20. PETH averaging is
    # linear, so reconstructing its mean avoids a third expensive spike pass.
    if nall == n80 + n20 and n80 > 0 and n20 > 0:
        mall = (n80 * m80 + n20 * m20) / float(nall)
    else:
        mall = _peth_mean_for_events(
            unit_spike_times, event_set['all'], t_before_s, t_after_s)
    return {
        'block80_raw': np.asarray(m80, dtype=np.float32),
        'block20_raw': np.asarray(m20, dtype=np.float32),
        'all_mean': np.asarray(mall, dtype=np.float32),
    }


def _diagnostic_modes_for_events(unit_spike_times, event_set,
                                 t_before_s=None, t_after_s=None,
                                 return_raw=False):
    raw = _diagnostic_raw_for_events(
        unit_spike_times, event_set, t_before_s, t_after_s)
    modes = _normalize_delta_modes(
        raw['block80_raw'], raw['block20_raw'],
        raw['all_mean'], t_before_s)
    if return_raw:
        return modes, raw
    return modes


def _compute_unit_diagnostic_traces(unit_spike_times, diagnostic_trial_sets,
                                    t_before_s=None, t_after_s=None):
    split_a = _diagnostic_modes_for_events(unit_spike_times, diagnostic_trial_sets['split_a'], t_before_s, t_after_s)
    split_b = _diagnostic_modes_for_events(unit_spike_times, diagnostic_trial_sets['split_b'], t_before_s, t_after_s)

    block_crossfit = {}
    block_crossfit_raw = {}
    for fold in diagnostic_trial_sets.get('block_crossfit', []):
        fold_name = str(fold['name'])
        block_crossfit[fold_name] = {}
        block_crossfit_raw[fold_name] = {}
        for role in ('reference', 'control_eval', 'stim_eval'):
            modes, raw = _diagnostic_modes_for_events(
                unit_spike_times, fold[role], t_before_s, t_after_s,
                return_raw=True)
            block_crossfit[fold_name][role] = modes
            block_crossfit_raw[fold_name][role] = raw

    by_mode = {mode: [] for mode in ('per_bin', 'baseline_scalar', 'zero_2_nan')}
    for event_set in diagnostic_trial_sets['trialmatched_repeats']:
        modes = _diagnostic_modes_for_events(unit_spike_times, event_set, t_before_s, t_after_s)
        for mode, trace in modes.items():
            by_mode[mode].append(trace)

    trialmatched = {}
    trialmatched_sem = {}
    for mode, traces in by_mode.items():
        arr = np.asarray(traces, dtype=float)
        finite = np.isfinite(arr)
        n = np.sum(finite, axis=0)
        summed = np.nansum(arr, axis=0)
        mean = np.full(arr.shape[1], np.nan, dtype=float)
        np.divide(summed, n, out=mean, where=n > 0)
        trialmatched[mode] = mean
        sem = np.full(arr.shape[1], np.nan, dtype=float)
        if arr.shape[0] > 1:
            std = np.nanstd(arr, axis=0, ddof=1)
            np.divide(std, np.sqrt(n), out=sem, where=n > 1)
        trialmatched_sem[mode] = sem

    return {
        'split_a': split_a,
        'split_b': split_b,
        'block_crossfit': block_crossfit,
        'block_crossfit_raw': block_crossfit_raw,
        'trialmatched': trialmatched,
        'trialmatched_sem': trialmatched_sem,
    }


def _new_alignment_store():
    raw = {
        'trace_nonstim_80_raw': [],
        'trace_nonstim_20_raw': [],
        'trace_stim_80_raw': [],
        'trace_stim_20_raw': [],
        'trace_nonstim_80_sem_raw': [],
        'trace_nonstim_20_sem_raw': [],
        'trace_stim_80_sem_raw': [],
        'trace_stim_20_sem_raw': [],
        'trace_nonstim_all_sem_raw': [],
        'trace_stim_all_sem_raw': [],
    }
    diag = {}
    for _diag_prefix in ('trace_nonstim_split_a', 'trace_nonstim_split_b',
                         'trace_nonstim_trialmatched',
                         'trace_nonstim_trialmatched_sem'):
        for _diag_mode in ('per_bin', 'baseline_scalar', 'zero_2_nan'):
            diag[f'{_diag_prefix}_{_diag_mode}'] = []
    for _fold in ('a', 'b'):
        for _role in ('reference', 'control_eval', 'stim_eval'):
            for _diag_mode in ('per_bin', 'baseline_scalar', 'zero_2_nan'):
                diag[f'trace_block_crossfit_{_role}_{_fold}_{_diag_mode}'] = []
            if save_futureproof_sufficient_stats == 1:
                for _raw_name in ('block80_raw', 'block20_raw', 'all_mean'):
                    diag[
                        f'trace_block_crossfit_{_role}_{_fold}_{_raw_name}'
                    ] = []
    return {
        'trace_nonstim': [],
        'trace_stim': [],
        'trace_zscore': [],
        'trace_nonstim_per_bin': [],
        'trace_stim_per_bin': [],
        'trace_nonstim_baseline_scalar': [],
        'trace_stim_baseline_scalar': [],
        'trace_nonstim_zero_2_nan': [],
        'trace_stim_zero_2_nan': [],
        'trace_stim_all': [],
        'trace_nonstim_all': [],
        'raw': raw,
        'diagnostic': diag,
    }


def _alignment_effect_indices(alignment, t_before_s):
    if alignment == 'Laser onset':
        first = int((t_before_s / bin_size) + 0.2 / bin_size)
        last = int((t_before_s / bin_size) + 0.8 / bin_size)
    else:
        first = int((t_before_s / bin_size) - 0.4 / bin_size)
        last = int(t_before_s / bin_size)
    return max(first, 0), max(last, first + 1)


def _calculate_unit_peth(unit_spike_times, events, t_before_s, t_after_s,
                         min_events=1):
    """Numerical one-unit PETH fallback with no Matplotlib side effects."""
    events = np.asarray(events, dtype=float)
    events = events[np.isfinite(events)]
    empty = _empty_trace(t_before_s, t_after_s)
    if len(events) < int(min_events):
        return empty, empty
    unit_spike_times = np.asarray(unit_spike_times, dtype=float)
    unit_spike_clusters = np.zeros(unit_spike_times.size, dtype=np.int8)
    try:
        peths, _ = singlecell.calculate_peths(
            unit_spike_times, unit_spike_clusters, [0], events,
            t_before_s, t_after_s, bin_size, smoothing, True)
        return np.asarray(peths.means[0], dtype=float), np.asarray(peths.stds[0], dtype=float)
    except Exception:
        return empty, empty


def _peth_view(mean_trace, std_trace):
    """Minimal attribute-compatible view used by the legacy downstream code."""
    return SimpleNamespace(
        means=np.asarray([mean_trace], dtype=float),
        stds=np.asarray([std_trace], dtype=float),
    )


def _prepare_scoring_spikes(spike_times, spike_clusters, cluster_ids):
    """Filter once to scored clusters and build time-sorted per-cluster views."""
    cluster_ids = np.asarray(sorted(set(int(x) for x in cluster_ids)), dtype=int)
    spike_times = np.asarray(spike_times, dtype=float)
    spike_clusters = np.asarray(spike_clusters)
    if cluster_ids.size == 0:
        return spike_times[:0], spike_clusters[:0], {}

    keep = np.isin(spike_clusters, cluster_ids)
    kept_times = spike_times[keep]
    kept_clusters = spike_clusters[keep].astype(int, copy=False)
    # Session spike times are already time ordered. Stable cluster sorting keeps
    # that order within each cluster, which searchsorted-based QP code requires.
    order = np.argsort(kept_clusters, kind='stable')
    grouped_clusters = kept_clusters[order]
    grouped_times = kept_times[order]
    unique, starts, counts = np.unique(
        grouped_clusters, return_index=True, return_counts=True)
    by_cluster = {
        int(cid): np.asarray(grouped_times[start:start + count], dtype=float)
        for cid, start, count in zip(unique, starts, counts)
    }
    for cid in cluster_ids:
        by_cluster.setdefault(int(cid), np.array([], dtype=float))
    return kept_times, kept_clusters, by_cluster


def _regular_alignment_events(trial_bunches, alignment):
    return {
        'nonstim_80': _event_times_for_alignment(
            trial_bunches['nonstim_80'], alignment),
        'stim_80': _event_times_for_alignment(
            trial_bunches['stim_80'], alignment),
        'nonstim_20': _event_times_for_alignment(
            trial_bunches['nonstim_20'], alignment),
        'stim_20': _event_times_for_alignment(
            trial_bunches['stim_20'], alignment),
        'nonstim_all': _event_times_for_alignment(
            trial_bunches['nonstim_all'], alignment),
        'stim_all': _event_times_for_alignment(
            trial_bunches['stim_all'], alignment),
    }


def _diagnostic_event_specs(diagnostic_sets):
    """Flatten diagnostic event sets into cacheable, shared keys.

    The legacy diagnostic path reconstructs ``all`` from the 80/20 means only
    when those trial counts add up exactly and both blocks are represented.
    Split-half sets can also contain other-prior trials, while sparse cross-fit
    cells can lack one block. Cache an explicit ``all`` PETH only in those
    cases so the optimized path preserves that rule exactly.
    """
    if diagnostic_sets is None:
        return {}
    specs = {}

    def _add_event_set(prefix, event_set):
        for block in ('80', '20'):
            specs[prefix + (block,)] = event_set[block]
        n80 = int(np.sum(np.isfinite(
            np.asarray(event_set['80'], dtype=float))))
        n20 = int(np.sum(np.isfinite(
            np.asarray(event_set['20'], dtype=float))))
        nall = int(np.sum(np.isfinite(
            np.asarray(event_set['all'], dtype=float))))
        if not (nall == n80 + n20 and n80 > 0 and n20 > 0):
            specs[prefix + ('all',)] = event_set['all']

    for half in ('a', 'b'):
        event_set = diagnostic_sets[f'split_{half}']
        _add_event_set(('split', half), event_set)
    for rep, event_set in enumerate(
            diagnostic_sets.get('trialmatched_repeats', [])):
        _add_event_set(('trialmatched', int(rep)), event_set)
    for fold in diagnostic_sets.get('block_crossfit', []):
        fold_name = str(fold['name'])
        for role in ('reference', 'control_eval', 'stim_eval'):
            _add_event_set(
                ('block_crossfit', fold_name, role), fold[role])
    return specs


def _calculate_batched_event_cache(
        spike_times, spike_clusters, cluster_ids, event_specs,
        t_before_s, t_after_s, *, keep_std, min_events,
        unit_spike_times_by_cluster, strict_finite=False):
    """Calculate shared event-set PETHs in exact, memory-bounded cluster chunks.

    brainbox treats clusters independently. Calling calculate_peths with several
    cluster ids therefore yields the same per-cluster means/stds as separate
    calls, while event bin construction and spike-table filtering are shared.
    """
    cluster_ids = np.asarray(sorted(set(int(x) for x in cluster_ids)), dtype=int)
    row_by_cluster = {int(cid): row for row, cid in enumerate(cluster_ids)}
    n_bins = len(_empty_trace(t_before_s, t_after_s))
    traces = {}
    batch_size = max(1, int(peth_cluster_batch_size))

    for key, raw_events in event_specs.items():
        raw_events = np.asarray(raw_events, dtype=float)
        finite_mask = np.isfinite(raw_events)
        invalid_strict = bool(strict_finite and not np.all(finite_mask))
        events = raw_events[finite_mask]
        means = np.full((cluster_ids.size, n_bins), np.nan, dtype=float)
        stds = (
            np.full((cluster_ids.size, n_bins), np.nan, dtype=float)
            if keep_std else None)

        if not invalid_strict and events.size >= int(min_events):
            for start in range(0, cluster_ids.size, batch_size):
                chunk = cluster_ids[start:start + batch_size]
                try:
                    peths, binned_spikes = singlecell.calculate_peths(
                        spike_times, spike_clusters, chunk, events,
                        t_before_s, t_after_s, bin_size, smoothing, True)
                    returned_ids = np.asarray(peths.cscale, dtype=int)
                    for local_row, cid in enumerate(returned_ids):
                        out_row = row_by_cluster[int(cid)]
                        means[out_row] = np.asarray(
                            peths.means[local_row], dtype=float)
                        if keep_std:
                            stds[out_row] = np.asarray(
                                peths.stds[local_row], dtype=float)
                    # The trial x cluster x bin tensor is only an intermediate;
                    # release it before the next chunk to bound peak memory.
                    del binned_spikes, peths
                except Exception as exc:
                    print(
                        f'Batched PETH fallback for key={key}, '
                        f'clusters {int(chunk[0])}-{int(chunk[-1])}: {exc}')
                    for cid in chunk:
                        mean, std = _calculate_unit_peth(
                            unit_spike_times_by_cluster.get(
                                int(cid), np.array([], dtype=float)),
                            events, t_before_s, t_after_s,
                            min_events=min_events)
                        out_row = row_by_cluster[int(cid)]
                        means[out_row] = mean
                        if keep_std:
                            stds[out_row] = std

        traces[key] = {
            'mean': means,
            'std': stds,
            'n_events': int(events.size),
            'strict_finite_valid': not invalid_strict,
        }
    return {'row_by_cluster': row_by_cluster, 'traces': traces}


def _cached_peth(cache, key, cluster_id):
    row = cache['row_by_cluster'][int(cluster_id)]
    entry = cache['traces'][key]
    mean = np.asarray(entry['mean'][row], dtype=float)
    std = entry.get('std')
    if std is None:
        std = np.full(mean.shape, np.nan, dtype=float)
    else:
        std = np.asarray(std[row], dtype=float)
    return mean, std


def _cached_diagnostic_raw(cache, cluster_id, key80, key20):
    m80, _ = _cached_peth(cache, key80, cluster_id)
    m20, _ = _cached_peth(cache, key20, cluster_id)
    n80 = int(cache['traces'][key80]['n_events'])
    n20 = int(cache['traces'][key20]['n_events'])
    key_all = key80[:-1] + ('all',)
    if key_all in cache['traces']:
        mall, _ = _cached_peth(cache, key_all, cluster_id)
    elif n80 > 0 and n20 > 0:
        mall = (n80 * m80 + n20 * m20) / float(n80 + n20)
    else:
        mall = np.full(m80.shape, np.nan, dtype=float)
    return {
        'block80_raw': np.asarray(m80, dtype=np.float32),
        'block20_raw': np.asarray(m20, dtype=np.float32),
        'all_mean': np.asarray(mall, dtype=np.float32),
    }


def _cached_diagnostic_modes(cache, cluster_id, key80, key20, t_before_s):
    raw = _cached_diagnostic_raw(cache, cluster_id, key80, key20)
    modes = _normalize_delta_modes(
        raw['block80_raw'], raw['block20_raw'], raw['all_mean'], t_before_s)
    return modes, raw


def _compute_unit_diagnostic_traces_from_cache(
        cluster_id, diagnostic_sets, cache, t_before_s=None):
    tb = t_before if t_before_s is None else t_before_s
    split_a, _ = _cached_diagnostic_modes(
        cache, cluster_id, ('split', 'a', '80'),
        ('split', 'a', '20'), tb)
    split_b, _ = _cached_diagnostic_modes(
        cache, cluster_id, ('split', 'b', '80'),
        ('split', 'b', '20'), tb)

    block_crossfit = {}
    block_crossfit_raw = {}
    for fold in diagnostic_sets.get('block_crossfit', []):
        fold_name = str(fold['name'])
        block_crossfit[fold_name] = {}
        block_crossfit_raw[fold_name] = {}
        for role in ('reference', 'control_eval', 'stim_eval'):
            modes, raw = _cached_diagnostic_modes(
                cache, cluster_id,
                ('block_crossfit', fold_name, role, '80'),
                ('block_crossfit', fold_name, role, '20'), tb)
            block_crossfit[fold_name][role] = modes
            block_crossfit_raw[fold_name][role] = raw

    by_mode = {mode: [] for mode in ('per_bin', 'baseline_scalar', 'zero_2_nan')}
    for rep, _ in enumerate(diagnostic_sets.get('trialmatched_repeats', [])):
        modes, _ = _cached_diagnostic_modes(
            cache, cluster_id, ('trialmatched', int(rep), '80'),
            ('trialmatched', int(rep), '20'), tb)
        for mode, trace in modes.items():
            by_mode[mode].append(trace)

    trialmatched = {}
    trialmatched_sem = {}
    for mode, traces in by_mode.items():
        arr = np.asarray(traces, dtype=float)
        finite = np.isfinite(arr)
        n = np.sum(finite, axis=0)
        summed = np.nansum(arr, axis=0)
        mean = np.full(arr.shape[1], np.nan, dtype=float)
        np.divide(summed, n, out=mean, where=n > 0)
        trialmatched[mode] = mean
        sem = np.full(arr.shape[1], np.nan, dtype=float)
        if arr.shape[0] > 1:
            std = np.nanstd(arr, axis=0, ddof=1)
            np.divide(std, np.sqrt(n), out=sem, where=n > 1)
        trialmatched_sem[mode] = sem

    return {
        'split_a': split_a,
        'split_b': split_b,
        'block_crossfit': block_crossfit,
        'block_crossfit_raw': block_crossfit_raw,
        'trialmatched': trialmatched,
        'trialmatched_sem': trialmatched_sem,
    }


def _build_alignment_peth_cache(
        alignment, trial_bunches, diagnostic_sets, cluster_ids,
        spike_times, spike_clusters, unit_spike_times_by_cluster):
    tb, ta = _alignment_window(alignment)
    regular_events = _regular_alignment_events(trial_bunches, alignment)
    started = perf_counter()
    regular = _calculate_batched_event_cache(
        spike_times, spike_clusters, cluster_ids, regular_events, tb, ta,
        keep_std=True, min_events=2,
        unit_spike_times_by_cluster=unit_spike_times_by_cluster,
        # The old primary plotting wrapper rejected any non-finite event. Keep
        # that behavior for the primary alignment; secondary numerical paths
        # historically filtered non-finite events.
        strict_finite=(alignment == onset_alignment),
    )
    diagnostic_specs = _diagnostic_event_specs(diagnostic_sets)
    diagnostic = None
    if diagnostic_specs:
        diagnostic = _calculate_batched_event_cache(
            spike_times, spike_clusters, cluster_ids, diagnostic_specs, tb, ta,
            keep_std=False, min_events=diagnostic_min_events_per_peth,
            unit_spike_times_by_cluster=unit_spike_times_by_cluster,
            strict_finite=False,
        )
    elapsed = perf_counter() - started
    print(
        f'  Batched {alignment} PETH cache: {len(cluster_ids)} clusters, '
        f'{len(regular_events) + len(diagnostic_specs)} event sets, '
        f'{elapsed:.1f}s')
    return {'regular': regular, 'diagnostic': diagnostic}


def _safe_sem_from_std(std_trace, n_events):
    std_trace = np.asarray(std_trace, dtype=float)
    return std_trace / np.sqrt(max(int(n_events), 1))


def _event_count(events):
    return int(np.sum(np.isfinite(np.asarray(events, dtype=float))))


def _compute_alignment_payload(cluster_id, alignment, trial_bunches,
                               unit_spike_times=None, diagnostic_sets=None,
                               peth_cache=None):
    tb, ta = _alignment_window(alignment)
    events = _regular_alignment_events(trial_bunches, alignment)

    if peth_cache is not None:
        regular_cache = peth_cache['regular']
        nonstim_80, nonstim_80_std = _cached_peth(
            regular_cache, 'nonstim_80', cluster_id)
        stim_80, stim_80_std = _cached_peth(
            regular_cache, 'stim_80', cluster_id)
        nonstim_20, nonstim_20_std = _cached_peth(
            regular_cache, 'nonstim_20', cluster_id)
        stim_20, stim_20_std = _cached_peth(
            regular_cache, 'stim_20', cluster_id)
        nonstim_all, nonstim_all_std = _cached_peth(
            regular_cache, 'nonstim_all', cluster_id)
        stim_all, stim_all_std = _cached_peth(
            regular_cache, 'stim_all', cluster_id)
    else:
        nonstim_80, nonstim_80_std = _calculate_unit_peth(
            unit_spike_times, events['nonstim_80'], tb, ta, min_events=1)
        stim_80, stim_80_std = _calculate_unit_peth(
            unit_spike_times, events['stim_80'], tb, ta, min_events=1)
        nonstim_20, nonstim_20_std = _calculate_unit_peth(
            unit_spike_times, events['nonstim_20'], tb, ta, min_events=1)
        stim_20, stim_20_std = _calculate_unit_peth(
            unit_spike_times, events['stim_20'], tb, ta, min_events=1)
        nonstim_all, nonstim_all_std = _calculate_unit_peth(
            unit_spike_times, events['nonstim_all'], tb, ta, min_events=1)
        stim_all, stim_all_std = _calculate_unit_peth(
            unit_spike_times, events['stim_all'], tb, ta, min_events=1)

    delta_nonstim_raw = nonstim_80 - nonstim_20
    delta_stim_raw = stim_80 - stim_20
    nonstim_err = np.sqrt(
        (nonstim_80_std**2 / max(_event_count(events['nonstim_80']), 1)) +
        (nonstim_20_std**2 / max(_event_count(events['nonstim_20']), 1))
    )
    stim_err = np.sqrt(
        (stim_80_std**2 / max(_event_count(events['stim_80']), 1)) +
        (stim_20_std**2 / max(_event_count(events['stim_20']), 1))
    )
    with np.errstate(invalid='ignore', divide='ignore'):
        z_score = (delta_stim_raw - delta_nonstim_raw) / np.sqrt(nonstim_err**2 + stim_err**2)

    nonstim_modes = _normalize_delta_modes(nonstim_80, nonstim_20, nonstim_all, tb)
    stim_modes = _normalize_delta_modes(stim_80, stim_20, stim_all, tb)
    default_nonstim, default_stim = {
        'per_bin': (nonstim_modes['per_bin'], stim_modes['per_bin']),
        'baseline_scalar': (nonstim_modes['baseline_scalar'], stim_modes['baseline_scalar']),
        'zero_2_nan': (nonstim_modes['zero_2_nan'], stim_modes['zero_2_nan']),
    }.get(normalize_mode, (nonstim_modes['per_bin'], stim_modes['per_bin']))

    first, last = _alignment_effect_indices(alignment, tb)
    mean_nonstim = np.nanmean(nonstim_modes['baseline_scalar'][first:last])
    mean_stim = np.nanmean(stim_modes['baseline_scalar'][first:last])

    diag = None
    if ((save_diagnostic_traces == 1 or save_futureproof_sufficient_stats == 1)
            and diagnostic_sets is not None and unit_spike_times is not None):
        if peth_cache is not None and peth_cache.get('diagnostic') is not None:
            diag = _compute_unit_diagnostic_traces_from_cache(
                cluster_id, diagnostic_sets, peth_cache['diagnostic'], tb)
        else:
            diag = _compute_unit_diagnostic_traces(
                unit_spike_times, diagnostic_sets, tb, ta)

    return {
        'trace_nonstim': default_nonstim,
        'trace_stim': default_stim,
        'trace_zscore': z_score,
        'trace_nonstim_per_bin': nonstim_modes['per_bin'],
        'trace_stim_per_bin': stim_modes['per_bin'],
        'trace_nonstim_baseline_scalar': nonstim_modes['baseline_scalar'],
        'trace_stim_baseline_scalar': stim_modes['baseline_scalar'],
        'trace_nonstim_zero_2_nan': nonstim_modes['zero_2_nan'],
        'trace_stim_zero_2_nan': stim_modes['zero_2_nan'],
        'trace_stim_all': stim_all,
        'trace_nonstim_all': nonstim_all,
        'raw': {
            'trace_nonstim_80_raw': nonstim_80,
            'trace_nonstim_20_raw': nonstim_20,
            'trace_stim_80_raw': stim_80,
            'trace_stim_20_raw': stim_20,
            'trace_nonstim_80_sem_raw': _safe_sem_from_std(nonstim_80_std, _event_count(events['nonstim_80'])),
            'trace_nonstim_20_sem_raw': _safe_sem_from_std(nonstim_20_std, _event_count(events['nonstim_20'])),
            'trace_stim_80_sem_raw': _safe_sem_from_std(stim_80_std, _event_count(events['stim_80'])),
            'trace_stim_20_sem_raw': _safe_sem_from_std(stim_20_std, _event_count(events['stim_20'])),
            'trace_nonstim_all_sem_raw': _safe_sem_from_std(nonstim_all_std, _event_count(events['nonstim_all'])),
            'trace_stim_all_sem_raw': _safe_sem_from_std(stim_all_std, _event_count(events['stim_all'])),
        },
        'diagnostic': diag,
        'mean_delta_nonstim': mean_nonstim,
        'mean_delta_stim': mean_stim,
    }


def _append_alignment_payload(store, payload):
    for key in ('trace_nonstim', 'trace_stim', 'trace_zscore',
                'trace_nonstim_per_bin', 'trace_stim_per_bin',
                'trace_nonstim_baseline_scalar', 'trace_stim_baseline_scalar',
                'trace_nonstim_zero_2_nan', 'trace_stim_zero_2_nan',
                'trace_stim_all', 'trace_nonstim_all'):
        store[key].append(np.asarray(payload[key], dtype=float))
    if save_raw_block_peths == 1:
        for key in store['raw']:
            store['raw'][key].append(np.asarray(payload['raw'][key], dtype=float))
    if save_diagnostic_traces == 1 or save_futureproof_sufficient_stats == 1:
        diag = payload.get('diagnostic')
        if diag is None:
            trace_len = len(payload['trace_nonstim'])
            for key in store['diagnostic']:
                store['diagnostic'][key].append(np.full(trace_len, np.nan, dtype=float))
        else:
            for mode in ('per_bin', 'baseline_scalar', 'zero_2_nan'):
                store['diagnostic'][f'trace_nonstim_split_a_{mode}'].append(diag['split_a'][mode])
                store['diagnostic'][f'trace_nonstim_split_b_{mode}'].append(diag['split_b'][mode])
                store['diagnostic'][f'trace_nonstim_trialmatched_{mode}'].append(diag['trialmatched'][mode])
                store['diagnostic'][f'trace_nonstim_trialmatched_sem_{mode}'].append(diag['trialmatched_sem'][mode])
                for fold in ('a', 'b'):
                    fold_data = diag.get('block_crossfit', {}).get(fold, {})
                    for role in ('reference', 'control_eval', 'stim_eval'):
                        trace = fold_data.get(role, {}).get(mode)
                        if trace is None:
                            trace = np.full(len(payload['trace_nonstim']), np.nan, dtype=float)
                        store['diagnostic'][
                            f'trace_block_crossfit_{role}_{fold}_{mode}'
                        ].append(np.asarray(trace, dtype=np.float32))
        if save_futureproof_sufficient_stats == 1 and diag is not None:
            for fold in ('a', 'b'):
                fold_raw = diag.get('block_crossfit_raw', {}).get(fold, {})
                for role in ('reference', 'control_eval', 'stim_eval'):
                    role_raw = fold_raw.get(role, {})
                    for raw_name in ('block80_raw', 'block20_raw', 'all_mean'):
                        trace = role_raw.get(raw_name)
                        if trace is None:
                            trace = np.full(
                                len(payload['trace_nonstim']), np.nan, dtype=np.float32)
                        store['diagnostic'][
                            f'trace_block_crossfit_{role}_{fold}_{raw_name}'
                        ].append(np.asarray(trace, dtype=np.float32))


def _build_results_payload(units_df, store, alignment):
    tb, ta = _alignment_window(alignment)
    results = {
        'units': units_df.reset_index(drop=True),
        'trace_nonstim': store['trace_nonstim'],
        'trace_stim': store['trace_stim'],
        'trace_zscore': store['trace_zscore'],
        'trace_nonstim_per_bin': store['trace_nonstim_per_bin'],
        'trace_stim_per_bin': store['trace_stim_per_bin'],
        'trace_nonstim_baseline_scalar': store['trace_nonstim_baseline_scalar'],
        'trace_stim_baseline_scalar': store['trace_stim_baseline_scalar'],
        'trace_nonstim_zero_2_nan': store['trace_nonstim_zero_2_nan'],
        'trace_stim_zero_2_nan': store['trace_stim_zero_2_nan'],
        'trace_stim_all': store['trace_stim_all'],
        'trace_nonstim_all': store['trace_nonstim_all'],
        'peth_time': np.arange(-tb, ta, bin_size),
        'bin_size': bin_size,
        't_before': tb,
        't_after': ta,
        'onset_alignment': alignment,
        'smoothing_window_ms': post_smooth_window_ms,
        'peth_smoothing_s': smoothing,
        'peth_smoothing_ms': float(smoothing) * 1000.0,
        'diagnostic_trace_modes': ('per_bin', 'baseline_scalar', 'zero_2_nan'),
        'run_config': {
            'compute_recorded_region': compute_recorded_region,
            'compute_min_IBL_label': compute_min_IBL_label,
            'compute_min_firing_rate': compute_min_firing_rate,
            'drift_epoch': drift_epoch,
            'remove_drift_units': remove_drift_units,
            'remove_nonstationary_units': remove_nonstationary_units,
            'nonstationarity_n_segments': nonstationarity_n_segments,
            'nonstationarity_min_trials': nonstationarity_min_trials,
            'nonstationarity_low_fr_fraction_of_median': nonstationarity_low_fr_fraction_of_median,
            'nonstationarity_min_median_fr_hz': nonstationarity_min_median_fr_hz,
            'max_qp_fr_segment_range_frac': max_qp_fr_segment_range_frac,
            'max_qp_resid_drift_range_frac': max_qp_resid_drift_range_frac,
            'max_qp_resid_drift_cv': max_qp_resid_drift_cv,
            'max_qp_resid_abs_rho_time': max_qp_resid_abs_rho_time,
            'max_qp_low_activity_fraction': max_qp_low_activity_fraction,
            'max_qp_max_low_activity_run': max_qp_max_low_activity_run,
            'min_qp_block_effect_sign_consistency': min_qp_block_effect_sign_consistency,
            'max_qp_block_effect_segment_cv': max_qp_block_effect_segment_cv,
            'max_qp_block_effect_dominance': max_qp_block_effect_dominance,
            'beginning_block_trials_remove': beginning_block_trials_remove,
            'use_GLMHMM_engaged_indices': use_GLMHMM_engaged_indices,
            'opto_trials_GLMHMM': opto_trials_GLMHMM,
            'n_states': n_states,
            'bs_definition_trial_mode': bs_definition_trial_mode,
            'insertion_brain_regions': insertion_brain_regions,
            'insertion_conditions': insertion_conditions,
            'match_nonstim_to_inhibition_range': match_nonstim_to_inhibition_range,
            'normalize_mode': normalize_mode,
            'scalar_baseline_window': scalar_baseline_window,
            'scalar_min_fr': scalar_min_fr,
            'zero_nan_threshold': zero_nan_threshold,
            'peth_smoothing_s': smoothing,
            'post_smooth_window_ms': post_smooth_window_ms,
            'save_diagnostic_traces': save_diagnostic_traces,
            'save_raw_block_peths': save_raw_block_peths,
            'diagnostic_random_seed': diagnostic_random_seed,
            'diagnostic_trialmatch_repeats': diagnostic_trialmatch_repeats,
            'diagnostic_min_events_per_peth': diagnostic_min_events_per_peth,
            'diagnostic_crossfit_guard_trials': diagnostic_crossfit_guard_trials,
            'diagnostic_crossfit_method': 'contiguous_block_halves_nearest_control',
            'save_futureproof_sufficient_stats': save_futureproof_sufficient_stats,
            'use_batched_peths': use_batched_peths,
            'peth_cluster_batch_size': peth_cluster_batch_size,
            'peth_compute_method': 'brainbox_exact_cluster_batch',
            'save_legacy_base_pickle': save_legacy_base_pickle,
            'save_combined_alignment_pickle': save_combined_alignment_pickle,
            'onset_alignments_to_run': tuple(_alignments_for_run()),
            'alignment_time_windows': dict(alignment_time_windows),
        },
    }
    if save_raw_block_peths == 1:
        results.update(store['raw'])
    if save_diagnostic_traces == 1 or save_futureproof_sufficient_stats == 1:
        results.update(store['diagnostic'])
    return results


clusters_of_interest = list()
acronyms_for_clusters_of_interest = list()
pids_per_cluster = list()
BS_score_per_cluster = list()
pval_ex_per_cluster = list()
pval_in_per_cluster = list()
delta_fr_nonstim_all = list()
delta_fr_inhibition_all = list()
zscore_all = list()
stim_all_trace = list()   # raw mean PETH (Hz) for stim trials, per unit -- laser-alignment QC
nonstim_all_trace = list()   # raw mean PETH (Hz) for control (nonstim) trials, per unit
# Per-unit delta-FR traces for all three normalization modes (saved in parallel).
delta_fr_nonstim_per_bin = list();    delta_fr_stim_per_bin = list()
delta_fr_nonstim_scalar = list();     delta_fr_stim_scalar = list()
delta_fr_nonstim_zero_2_nan = list(); delta_fr_stim_zero_2_nan = list()

# Optional diagnostic payloads, row-aligned to clusters_info_DF when enabled.
raw_block_peth_traces = {
    'trace_nonstim_80_raw': [],
    'trace_nonstim_20_raw': [],
    'trace_stim_80_raw': [],
    'trace_stim_20_raw': [],
    'trace_nonstim_80_sem_raw': [],
    'trace_nonstim_20_sem_raw': [],
    'trace_stim_80_sem_raw': [],
    'trace_stim_20_sem_raw': [],
    'trace_nonstim_all_sem_raw': [],
    'trace_stim_all_sem_raw': [],
}
diagnostic_trace_lists = {}
for _diag_prefix in ('trace_nonstim_split_a', 'trace_nonstim_split_b',
                     'trace_nonstim_trialmatched',
                     'trace_nonstim_trialmatched_sem'):
    for _diag_mode in ('per_bin', 'baseline_scalar', 'zero_2_nan'):
        diagnostic_trace_lists[f'{_diag_prefix}_{_diag_mode}'] = []
for _fold in ('a', 'b'):
    for _role in ('reference', 'control_eval', 'stim_eval'):
        for _diag_mode in ('per_bin', 'baseline_scalar', 'zero_2_nan'):
            diagnostic_trace_lists[
                f'trace_block_crossfit_{_role}_{_fold}_{_diag_mode}'
            ] = []
        if save_futureproof_sufficient_stats == 1:
            for _raw_name in ('block80_raw', 'block20_raw', 'all_mean'):
                diagnostic_trace_lists[
                    f'trace_block_crossfit_{_role}_{_fold}_{_raw_name}'
                ] = []

# Compact row-aligned/session-shared sufficient statistics. These make it
# possible to cross-fit the BS call itself or rebuild normalization after the
# expensive spike-level pass has finished.
qp_fr_per_trial_all = []
trial_metadata_by_pid = {}

additional_alignment_stores = {
    _alignment: _new_alignment_store()
    for _alignment in _alignments_for_run()
    if _alignment != onset_alignment
}

excitation_traces_percluster = list()
inhibition_traces_percluster = list()
nonstim_traces_percluster = list()
excitation_stds_percluster = list()
inhibition_stds_percluster = list()
nonstim_stds_percluster = list()

clusters_info_DF = pd.DataFrame()

normalize_to_baseline = 0

##### start main loop
for main_loop_num in range(start_pid_idx, len(pids)):

    pid = pids[main_loop_num]
    print('starting analysis of pid = ' + pid)
    # Canonical session load (shared with CD): includes waveform templates
    # via ssl.load_spike_sorting_object('waveforms'), replacing the heavy
    # _phy_spikes_subset per-spike snippets.
    sb = load_session(pid, one, ba, load_waveforms=True)
    eid = sb.eid
    current_mouse_ID = sb.mouse_id
    trials = sb.trials
    ses_path = sb.ses_path
    probe_label = sb.probe_label
    spikes = sb.spikes
    clusters = sb.clusters
    channels = sb.channels
    clusters_labels = sb.clusters_labels
    waveforms = sb.waveforms
    allspikes = spikes
    laser_intervals = None   # reset each iteration; set in the laser try-block below

    light_artifact_units = light_artifact_units_list[main_loop_num]
    excitation_trials_range = excitation_trials_range_list[main_loop_num]
    inhibition_trials_range = inhibition_trials_range_list[main_loop_num]

    # spikes, clusters = load_spike_sorting(eid, one=one, probe=probe_label, spike_sorter='pykilosort')
    # clusters_labels = clusters[probe_label]['metrics']['label']
    # allspikes = spikes[probe_label]

    excitation_trials = trials.copy()
    inhibition_trials = trials.copy()
    nonstim_trials = trials.copy()
    nonstim_trials_ex = trials.copy()
    nonstim_trials_in = trials.copy()

    try:
        laser_intervals = one.load_dataset(eid, '_ibl_laserStimulation.intervals')
        excitation_trials_numbers = np.empty(len(trials.contrastLeft))
        excitation_trials_numbers[:] = np.nan
        inhibition_trials_numbers = np.empty(len(trials.contrastLeft))
        inhibition_trials_numbers[:] = np.nan
        nonstim_trials_numbers = np.empty(len(trials.contrastLeft))
        nonstim_trials_numbers[:] = np.nan
        nonstim_trials_numbers_ex = np.empty(len(trials.contrastLeft))
        nonstim_trials_numbers_ex[:] = np.nan
        nonstim_trials_numbers_in = np.empty(len(trials.contrastLeft))
        nonstim_trials_numbers_in[:] = np.nan

        ### conditional statement for assigning trials to 'excitation' trial, 'inhibition' trial, or nonstim trial
        if inhibition_trials_range == 'ALL':
            inhibition_trials_range = range(0,len(trials['contrastLeft']))
            #### use last trial as end of range when end of range set to 9999

        for k in range(0,len(trials.contrastLeft)-1):
            if trials.intervals[k,0] in laser_intervals[:,0] and k in excitation_trials_range:
                excitation_trials_numbers[k] = k
            elif trials.intervals[k,0] in laser_intervals[:,0] and k in inhibition_trials_range:
                inhibition_trials_numbers[k] = k
            else:
                nonstim_trials_numbers[k] = k
        # for k in excitation_trials_range:
        #     if trials.intervals[k,0] in laser_intervals[:,0]:
        #         # react = trials['feedback_times'][k] - trials['goCue_times'][k]
        #         # if react < stim_rt_threshold:
        #         excitation_trials_numbers[k] = k
        #     else:
        #         # react = trials['feedback_times'][k] - trials['goCue_times'][k]
        #         # if react < stim_rt_threshold:
        #         nonstim_trials_numbers[k] = k
        #         nonstim_trials_numbers_ex[k] = k
        # for k in inhibition_trials_range:
        #     if trials.intervals[k,0] in laser_intervals[:,0]:
        #         # react = trials['feedback_times'][k] - trials['goCue_times'][k]
        #         # if react < stim_rt_threshold:
        #         inhibition_trials_numbers[k] = k
        #     else:
        #         # react = trials['feedback_times'][k] - trials['goCue_times'][k]
        #         # if react < stim_rt_threshold:
        #         nonstim_trials_numbers[k] = k
        #         nonstim_trials_numbers_in[k] = k

    except:
        if suppress_print_output == 0:
            print('Laser intervals data not found; loading depricated taskData')
        if inhibition_trials_range == 'ALL':
            inhibition_trials_range = range(0,len(trials['contrastLeft']))
            #### use last trial as end of range when end of range set to 9999
        taskData = load_data(ses_path)
        excitation_trials_numbers = np.empty(len(taskData))
        excitation_trials_numbers[:] = np.nan
        inhibition_trials_numbers = np.empty(len(taskData))
        inhibition_trials_numbers[:] = np.nan
        nonstim_trials_numbers = np.empty(len(taskData))
        nonstim_trials_numbers[:] = np.nan
        nonstim_trials_numbers_ex = np.empty(len(taskData))
        nonstim_trials_numbers_ex[:] = np.nan
        nonstim_trials_numbers_in = np.empty(len(taskData))
        nonstim_trials_numbers_in[:] = np.nan
        for k in range(0,len(trials.contrastLeft)-1):
            if taskData[k]['opto'] == 1 and k in excitation_trials_range:
                excitation_trials_numbers[k] = k
            elif taskData[k]['opto'] == 1 and k in inhibition_trials_range:
                inhibition_trials_numbers[k] = k
            else:
                nonstim_trials_numbers[k] = k
        # for k in excitation_trials_range:
        #     if taskData[k]['opto'] == 1:
        #         excitation_trials_numbers[k] = k
        #     else:
        #         nonstim_trials_numbers[k] = k
        #         nonstim_trials_numbers_ex[k] = k
        # for k in inhibition_trials_range:
        #     if taskData[k]['opto'] == 1:
        #         inhibition_trials_numbers[k] = k
        #     else:
        #         nonstim_trials_numbers[k] = k
        #         nonstim_trials_numbers_in[k] = k

    excitation_trials_numbers = excitation_trials_numbers[~np.isnan(excitation_trials_numbers)]
    inhibition_trials_numbers = inhibition_trials_numbers[~np.isnan(inhibition_trials_numbers)]
    nonstim_trials_numbers = nonstim_trials_numbers[~np.isnan(nonstim_trials_numbers)]
    nonstim_trials_numbers_ex = nonstim_trials_numbers_ex[~np.isnan(nonstim_trials_numbers_ex)]
    nonstim_trials_numbers_in = nonstim_trials_numbers_in[~np.isnan(nonstim_trials_numbers_in)]
    excitation_trials_numbers = excitation_trials_numbers.astype(int)
    inhibition_trials_numbers = inhibition_trials_numbers.astype(int)
    nonstim_trials_numbers = nonstim_trials_numbers.astype(int)
    nonstim_trials_numbers_ex = nonstim_trials_numbers_ex.astype(int)
    nonstim_trials_numbers_in = nonstim_trials_numbers_in.astype(int)

    # Apply per-PID trial exclusions defined in BS_config.TRIALS_TO_REMOVE.
    if pid in TRIALS_TO_REMOVE:
        (excitation_trials_numbers,
         inhibition_trials_numbers,
         nonstim_trials_numbers,
         nonstim_trials_numbers_ex,
         nonstim_trials_numbers_in) = drop_trials_from_arrays(
            [excitation_trials_numbers,
             inhibition_trials_numbers,
             nonstim_trials_numbers,
             nonstim_trials_numbers_ex,
             nonstim_trials_numbers_in],
            TRIALS_TO_REMOVE[pid],
        )

    # GLM-HMM engagement restriction (shared with CD). Nonstim/control trials
    # are always filtered by their own state; opto trials follow
    # opto_trials_GLMHMM ('standard', 'bypass', or 'prior state').
    engaged_idx = np.asarray([], dtype=int)
    if use_GLMHMM_engaged_indices == 1:
        try:
            glmhmm_result = get_glmhmm_indices(current_mouse_ID, str(eid), state_probability, n_states)
            engaged_idx = coerce_glmhmm_engaged_indices(glmhmm_result, n_states)
            opto_trials_numbers = np.union1d(excitation_trials_numbers, inhibition_trials_numbers)
            all_trial_numbers = np.union1d(opto_trials_numbers, nonstim_trials_numbers)
            _, opto_trials_keep, nonstim_trials_keep, _ = apply_glmhmm_opto_trial_policy(
                all_trial_numbers,
                opto_trials_numbers,
                nonstim_trials_numbers,
                nonstim_trials_numbers,
                engaged_idx,
                opto_trials_GLMHMM,
            )
            excitation_trials_numbers = np.intersect1d(opto_trials_keep, excitation_trials_numbers)
            inhibition_trials_numbers = np.intersect1d(opto_trials_keep, inhibition_trials_numbers)
            nonstim_trials_numbers = nonstim_trials_keep
            nonstim_trials_numbers_ex = np.intersect1d(nonstim_trials_keep, nonstim_trials_numbers_ex)
            nonstim_trials_numbers_in = np.intersect1d(nonstim_trials_keep, nonstim_trials_numbers_in)
        except Exception as e:
            print(f'GLM-HMM filtering failed for PID = {pid}: {e}; skipping session...')
            continue

    (excitation_trials_numbers,
     inhibition_trials_numbers,
     nonstim_trials_numbers,
     nonstim_trials_numbers_ex,
     nonstim_trials_numbers_in), bs_definition_summary = _apply_bs_definition_trial_mode(
        trials,
        [
            excitation_trials_numbers,
            inhibition_trials_numbers,
            nonstim_trials_numbers,
            nonstim_trials_numbers_ex,
            nonstim_trials_numbers_in,
        ],
        bs_definition_trial_mode,
    )
    if bs_definition_trial_mode != 'standard':
        print(
            f"  BS trial mode={bs_definition_trial_mode}: "
            f"nonstim {bs_definition_summary.get('n_nonstim_trials_before_bs_definition', 0)}"
            f"->{bs_definition_summary.get('n_nonstim_trials_after_bs_definition', 0)}, "
            f"inhibition {bs_definition_summary.get('n_inhibition_trials_before_bs_definition', 0)}"
            f"->{bs_definition_summary.get('n_inhibition_trials_after_bs_definition', 0)}"
        )

    # --- Matched control set for the delta-FR PETHs -----------------------
    # nonstim_trials_numbers is broad (all non-laser trials session-wide), but the
    # opto (inhibition) trials live only inside inhibition_trials_range. If that
    # range is a biased slice of the session, the control 80/20 delta sits
    # above/below the opto delta everywhere -- including pre-laser -- producing a
    # constant baseline offset (the residual pre-laser dip). nonstim_trials_numbers_delta
    # restricts the control trials used for the DELTA-FR PETHs (the 80/20 split
    # AND the all-trial normalizer) to the same trial-index range as the opto
    # trials, so control and opto share a baseline. The BS *score* (perm test,
    # below) deliberately keeps the broad nonstim_trials_numbers for power.
    if match_nonstim_to_inhibition_range == 1:
        _rng_set = set(int(x) for x in inhibition_trials_range)
        nonstim_trials_numbers_delta = np.array(
            [int(k) for k in nonstim_trials_numbers if int(k) in _rng_set], dtype=int)
    else:
        nonstim_trials_numbers_delta = np.asarray(nonstim_trials_numbers, dtype=int)

    # if use_trials_after_stim == 1:
    #     stim_trials_numbers = stim_trials_numbers +1
    #     if stim_trials_numbers[np.size(stim_trials_numbers)-1] == len(trials['contrastLeft']):
    #         stim_trials_numbers = stim_trials_numbers[range(len(stim_trials_numbers)-1)]

    excitation_trials.contrastRight = trials.contrastRight[excitation_trials_numbers]
    excitation_trials.contrastLeft = trials.contrastLeft[excitation_trials_numbers]
    excitation_trials.goCueTrigger_times = trials.goCueTrigger_times[excitation_trials_numbers]
    excitation_trials.feedback_times = trials.feedback_times[excitation_trials_numbers]
    excitation_trials.response_times = trials.response_times[excitation_trials_numbers]
    excitation_trials.feedbackType = trials.feedbackType[excitation_trials_numbers]
    excitation_trials.goCue_times = trials.goCue_times[excitation_trials_numbers]
    excitation_trials.firstMovement_times = trials.firstMovement_times[excitation_trials_numbers]
    # excitation_trials.excitationOnTrigger_times = trials.stimOnTrigger_times[excitation_trials_numbers]
    excitation_trials.probabilityLeft = trials.probabilityLeft[excitation_trials_numbers]
    excitation_trials.stimOn_times = trials.stimOn_times[excitation_trials_numbers]
    excitation_trials.choice = trials.choice[excitation_trials_numbers]
    excitation_trials.rewardVolume = trials.rewardVolume[excitation_trials_numbers]
    # excitation_trials.included = trials.included[excitation_trials_numbers]
    excitation_trials.intervals = trials.intervals[excitation_trials_numbers]

    inhibition_trials.contrastRight = trials.contrastRight[inhibition_trials_numbers]
    inhibition_trials.contrastLeft = trials.contrastLeft[inhibition_trials_numbers]
    inhibition_trials.goCueTrigger_times = trials.goCueTrigger_times[inhibition_trials_numbers]
    inhibition_trials.feedback_times = trials.feedback_times[inhibition_trials_numbers]
    inhibition_trials.response_times = trials.response_times[inhibition_trials_numbers]
    inhibition_trials.feedbackType = trials.feedbackType[inhibition_trials_numbers]
    inhibition_trials.goCue_times = trials.goCue_times[inhibition_trials_numbers]
    inhibition_trials.firstMovement_times = trials.firstMovement_times[inhibition_trials_numbers]
    # inhibition_trials.inhibitionOnTrigger_times = trials.stimOnTrigger_times[inhibition_trials_numbers]
    inhibition_trials.probabilityLeft = trials.probabilityLeft[inhibition_trials_numbers]
    inhibition_trials.stimOn_times = trials.stimOn_times[inhibition_trials_numbers]
    inhibition_trials.choice = trials.choice[inhibition_trials_numbers]
    inhibition_trials.rewardVolume = trials.rewardVolume[inhibition_trials_numbers]
    # inhibition_trials.included = trials.included[inhibition_trials_numbers]
    inhibition_trials.intervals = trials.intervals[inhibition_trials_numbers]
    nonstim_trials.contrastRight = trials.contrastRight[nonstim_trials_numbers_delta]
    nonstim_trials.contrastLeft = trials.contrastLeft[nonstim_trials_numbers_delta]
    nonstim_trials.goCueTrigger_times = trials.goCueTrigger_times[nonstim_trials_numbers_delta]
    nonstim_trials.feedback_times = trials.feedback_times[nonstim_trials_numbers_delta]
    nonstim_trials.response_times = trials.response_times[nonstim_trials_numbers_delta]
    nonstim_trials.feedbackType = trials.feedbackType[nonstim_trials_numbers_delta]
    nonstim_trials.goCue_times = trials.goCue_times[nonstim_trials_numbers_delta]
    nonstim_trials.firstMovement_times = trials.firstMovement_times[nonstim_trials_numbers_delta]
    # nonstim_trials.stimOnTrigger_times = trials.stimOnTrigger_times[nonstim_trials_numbers_delta]
    nonstim_trials.probabilityLeft = trials.probabilityLeft[nonstim_trials_numbers_delta]
    nonstim_trials.stimOn_times = trials.stimOn_times[nonstim_trials_numbers_delta]
    nonstim_trials.choice = trials.choice[nonstim_trials_numbers_delta]
    nonstim_trials.rewardVolume = trials.rewardVolume[nonstim_trials_numbers_delta]
    # nonstim_trials.included = trials.included[nonstim_trials_numbers_delta]
    nonstim_trials.intervals = trials.intervals[nonstim_trials_numbers_delta]
    nonstim_trials_ex.contrastRight = trials.contrastRight[nonstim_trials_numbers_ex]
    nonstim_trials_ex.contrastLeft = trials.contrastLeft[nonstim_trials_numbers_ex]
    nonstim_trials_ex.goCueTrigger_times = trials.goCueTrigger_times[nonstim_trials_numbers_ex]
    nonstim_trials_ex.feedback_times = trials.feedback_times[nonstim_trials_numbers_ex]
    nonstim_trials_ex.response_times = trials.response_times[nonstim_trials_numbers_ex]
    nonstim_trials_ex.feedbackType = trials.feedbackType[nonstim_trials_numbers_ex]
    nonstim_trials_ex.goCue_times = trials.goCue_times[nonstim_trials_numbers_ex]
    nonstim_trials_ex.firstMovement_times = trials.firstMovement_times[nonstim_trials_numbers_ex]
    # nonstim_trials_ex.stimOnTrigger_times = trials.stimOnTrigger_times[nonstim_trials_numbers_ex]
    nonstim_trials_ex.probabilityLeft = trials.probabilityLeft[nonstim_trials_numbers_ex]
    nonstim_trials_ex.stimOn_times = trials.stimOn_times[nonstim_trials_numbers_ex]
    nonstim_trials_ex.choice = trials.choice[nonstim_trials_numbers_ex]
    nonstim_trials_ex.rewardVolume = trials.rewardVolume[nonstim_trials_numbers_ex]
    # nonstim_trials_ex.included = trials.included[nonstim_trials_numbers_ex]
    nonstim_trials_ex.intervals = trials.intervals[nonstim_trials_numbers_ex]
    nonstim_trials_in.contrastRight = trials.contrastRight[nonstim_trials_numbers_in]
    nonstim_trials_in.contrastLeft = trials.contrastLeft[nonstim_trials_numbers_in]
    nonstim_trials_in.goCueTrigger_times = trials.goCueTrigger_times[nonstim_trials_numbers_in]
    nonstim_trials_in.feedback_times = trials.feedback_times[nonstim_trials_numbers_in]
    nonstim_trials_in.response_times = trials.response_times[nonstim_trials_numbers_in]
    nonstim_trials_in.feedbackType = trials.feedbackType[nonstim_trials_numbers_in]
    nonstim_trials_in.goCue_times = trials.goCue_times[nonstim_trials_numbers_in]
    nonstim_trials_in.firstMovement_times = trials.firstMovement_times[nonstim_trials_numbers_in]
    # nonstim_trials_in.stimOnTrigger_times = trials.stimOnTrigger_times[nonstim_trials_numbers_in]
    nonstim_trials_in.probabilityLeft = trials.probabilityLeft[nonstim_trials_numbers_in]
    nonstim_trials_in.stimOn_times = trials.stimOn_times[nonstim_trials_numbers_in]
    nonstim_trials_in.choice = trials.choice[nonstim_trials_numbers_in]
    nonstim_trials_in.rewardVolume = trials.rewardVolume[nonstim_trials_numbers_in]
    # nonstim_trials_in.included = trials.included[nonstim_trials_numbers_in]
    nonstim_trials_in.intervals = trials.intervals[nonstim_trials_numbers_in]

    excitation_trials_contrast = signed_contrast(excitation_trials)
    inhibition_trials_contrast = signed_contrast(inhibition_trials)
    nonstim_trials_contrast = signed_contrast(nonstim_trials)
    nonstim_trials_ex_contrast = signed_contrast(nonstim_trials_ex)
    nonstim_trials_in_contrast = signed_contrast(nonstim_trials_in)

    try:
        brain_acronyms_percluster = clusters['acronym']
    except:
        brain_acronyms_percluster = np.empty(len(clusters['ks2_label']))
        brain_acronyms_percluster[:] = np.nan


    ######copypasta ; turn this shit into a function or something
    trials_leftprob = trials.probabilityLeft
    filterval = 10 ###number of trials to remove at beginning of block
    early_block_trials_threshold = 15
    late_block_trials_threshold = 16
    earlytrials_50 = np.zeros((1, np.size(trials_leftprob)), dtype=int)
    latetrials_50 = np.zeros((1, np.size(trials_leftprob)), dtype=int)
    earlytrials_20 = np.zeros((1, np.size(trials_leftprob)), dtype=int)
    latetrials_20 = np.zeros((1, np.size(trials_leftprob)), dtype=int)
    earlytrials_80 = np.zeros((1, np.size(trials_leftprob)), dtype=int)
    latetrials_80 = np.zeros((1, np.size(trials_leftprob)), dtype=int)
    alltrialcounts_50 = np.zeros((1, np.size(trials_leftprob)), dtype=int)
    alltrialcounts_20 = np.zeros((1, np.size(trials_leftprob)), dtype=int)
    alltrialcounts_80 = np.zeros((1, np.size(trials_leftprob)), dtype=int)
    alltrialcounts_20_index = np.zeros((1, np.size(trials_leftprob)), dtype=int)
    alltrialcounts_80_index = np.zeros((1, np.size(trials_leftprob)), dtype=int)
    alltrialcounts_20_index_filtered = np.zeros((1, np.size(trials_leftprob)), dtype=int)
    alltrialcounts_80_index_filtered = np.zeros((1, np.size(trials_leftprob)), dtype=int)
    alltrials_block_length = np.zeros((1, np.size(trials_leftprob)))
    previous_trial_block_ID = 0.5
    current_trial_length = 0
    for l in range(0, np.size(trials_leftprob)):
        current_trial_block_ID = trials_leftprob[l]
        if current_trial_block_ID == previous_trial_block_ID:
            current_trial_length = current_trial_length + 1
            alltrials_block_length[:,l] = current_trial_length
            if current_trial_block_ID == 0.5:
                alltrialcounts_50[:,l] = current_trial_length
            if current_trial_block_ID == 0.2:
                alltrialcounts_20[:,l] = current_trial_length
                alltrialcounts_20_index[:,l] = l
                if current_trial_length > filterval:
                    alltrialcounts_20_index_filtered[:,l] = l
            if current_trial_block_ID == 0.8:
                alltrialcounts_80[:,l] = current_trial_length
                alltrialcounts_80_index[:,l] = l
                if current_trial_length > filterval:
                    alltrialcounts_80_index_filtered[:,l] = l
            if current_trial_block_ID == 0.5 and current_trial_length <= early_block_trials_threshold:
                earlytrials_50[:,l] = l
            if current_trial_block_ID == 0.5 and current_trial_length >= early_block_trials_threshold:
                latetrials_50[:,l] = l
            if current_trial_block_ID == 0.2 and current_trial_length <= early_block_trials_threshold:
                earlytrials_20[:,l] = l
            if current_trial_block_ID == 0.2 and current_trial_length >= early_block_trials_threshold:
                latetrials_20[:,l] = l
            if current_trial_block_ID == 0.8 and current_trial_length <= early_block_trials_threshold:
                earlytrials_80[:,l] = l
            if current_trial_block_ID == 0.8 and current_trial_length >= early_block_trials_threshold:
                latetrials_80[:,l] = l
        else:
            current_trial_length = 1
            alltrials_block_length[:,l] = 1
            if current_trial_block_ID == 0.5:
                alltrialcounts_50[:,l] = 1
            if current_trial_block_ID == 0.2:
                alltrialcounts_20[:,l] = 1
            if current_trial_block_ID == 0.8:
                alltrialcounts_80[:,l] = 1
            if current_trial_block_ID == 0.5:
                earlytrials_50[:,l] = l
            if current_trial_block_ID == 0.2:
                earlytrials_20[:,l] = l
            if current_trial_block_ID == 0.8:
                earlytrials_80[:,l] = l

        previous_trial_block_ID = current_trial_block_ID

    earlytrials_50 = earlytrials_50[(0.1 < earlytrials_50)]
    latetrials_50 = latetrials_50[(0.1 < latetrials_50)]
    earlytrials_20 = earlytrials_20[(0.1 < earlytrials_20)]
    latetrials_20 = latetrials_20[(0.1 < latetrials_20)]
    earlytrials_80 = earlytrials_80[(0.1 < earlytrials_80)]
    latetrials_80 = latetrials_80[(0.1 < latetrials_80)]
    alltrialcounts_80_index_filtered = alltrialcounts_80_index_filtered[(0.1 < alltrialcounts_80_index_filtered)]
    alltrialcounts_20_index_filtered = alltrialcounts_20_index_filtered[(0.1 < alltrialcounts_20_index_filtered)]

    inhibition_trials_numbers_on_80_block = list(set(inhibition_trials_numbers).intersection(alltrialcounts_80_index_filtered))
    inhibition_trials_numbers_on_20_block = list(set(inhibition_trials_numbers).intersection(alltrialcounts_20_index_filtered))

    ###############
    ### could use nonstim_trials_numbers_in, though that includes less trials and also is not what is used for BS calculation
    # nonstim_trials_numbers_on_80_block = list(set(nonstim_trials_numbers_in).intersection(alltrialcounts_80_index_filtered))
    # nonstim_trials_numbers_on_20_block = list(set(nonstim_trials_numbers_in).intersection(alltrialcounts_20_index_filtered))
    nonstim_trials_numbers_on_80_block = list(set(nonstim_trials_numbers_delta).intersection(alltrialcounts_80_index_filtered))
    nonstim_trials_numbers_on_20_block = list(set(nonstim_trials_numbers_delta).intersection(alltrialcounts_20_index_filtered))

    inhibition_trials_80 = trials.copy()
    inhibition_trials_20 = trials.copy()
    nonstim_trials_80 = trials.copy()
    nonstim_trials_20 = trials.copy()
    inhibition_trials_80.contrastRight = trials.contrastRight[inhibition_trials_numbers_on_80_block]
    inhibition_trials_80.contrastLeft = trials.contrastLeft[inhibition_trials_numbers_on_80_block]
    inhibition_trials_80.goCueTrigger_times = trials.goCueTrigger_times[inhibition_trials_numbers_on_80_block]
    inhibition_trials_80.feedback_times = trials.feedback_times[inhibition_trials_numbers_on_80_block]
    inhibition_trials_80.response_times = trials.response_times[inhibition_trials_numbers_on_80_block]
    inhibition_trials_80.feedbackType = trials.feedbackType[inhibition_trials_numbers_on_80_block]
    inhibition_trials_80.goCue_times = trials.goCue_times[inhibition_trials_numbers_on_80_block]
    inhibition_trials_80.firstMovement_times = trials.firstMovement_times[inhibition_trials_numbers_on_80_block]
    inhibition_trials_80.probabilityLeft = trials.probabilityLeft[inhibition_trials_numbers_on_80_block]
    inhibition_trials_80.stimOn_times = trials.stimOn_times[inhibition_trials_numbers_on_80_block]
    inhibition_trials_80.choice = trials.choice[inhibition_trials_numbers_on_80_block]
    inhibition_trials_80.rewardVolume = trials.rewardVolume[inhibition_trials_numbers_on_80_block]
    inhibition_trials_80.intervals = trials.intervals[inhibition_trials_numbers_on_80_block]
    inhibition_trials_20.contrastRight = trials.contrastRight[inhibition_trials_numbers_on_20_block]
    inhibition_trials_20.contrastLeft = trials.contrastLeft[inhibition_trials_numbers_on_20_block]
    inhibition_trials_20.goCueTrigger_times = trials.goCueTrigger_times[inhibition_trials_numbers_on_20_block]
    inhibition_trials_20.feedback_times = trials.feedback_times[inhibition_trials_numbers_on_20_block]
    inhibition_trials_20.response_times = trials.response_times[inhibition_trials_numbers_on_20_block]
    inhibition_trials_20.feedbackType = trials.feedbackType[inhibition_trials_numbers_on_20_block]
    inhibition_trials_20.goCue_times = trials.goCue_times[inhibition_trials_numbers_on_20_block]
    inhibition_trials_20.firstMovement_times = trials.firstMovement_times[inhibition_trials_numbers_on_20_block]
    inhibition_trials_20.probabilityLeft = trials.probabilityLeft[inhibition_trials_numbers_on_20_block]
    inhibition_trials_20.stimOn_times = trials.stimOn_times[inhibition_trials_numbers_on_20_block]
    inhibition_trials_20.choice = trials.choice[inhibition_trials_numbers_on_20_block]
    inhibition_trials_20.rewardVolume = trials.rewardVolume[inhibition_trials_numbers_on_20_block]
    inhibition_trials_20.intervals = trials.intervals[inhibition_trials_numbers_on_20_block]

    nonstim_trials_80.contrastRight = trials.contrastRight[nonstim_trials_numbers_on_80_block]
    nonstim_trials_80.contrastLeft = trials.contrastLeft[nonstim_trials_numbers_on_80_block]
    nonstim_trials_80.goCueTrigger_times = trials.goCueTrigger_times[nonstim_trials_numbers_on_80_block]
    nonstim_trials_80.feedback_times = trials.feedback_times[nonstim_trials_numbers_on_80_block]
    nonstim_trials_80.response_times = trials.response_times[nonstim_trials_numbers_on_80_block]
    nonstim_trials_80.feedbackType = trials.feedbackType[nonstim_trials_numbers_on_80_block]
    nonstim_trials_80.goCue_times = trials.goCue_times[nonstim_trials_numbers_on_80_block]
    nonstim_trials_80.firstMovement_times = trials.firstMovement_times[nonstim_trials_numbers_on_80_block]
    nonstim_trials_80.probabilityLeft = trials.probabilityLeft[nonstim_trials_numbers_on_80_block]
    nonstim_trials_80.stimOn_times = trials.stimOn_times[nonstim_trials_numbers_on_80_block]
    nonstim_trials_80.choice = trials.choice[nonstim_trials_numbers_on_80_block]
    nonstim_trials_80.rewardVolume = trials.rewardVolume[nonstim_trials_numbers_on_80_block]
    nonstim_trials_80.intervals = trials.intervals[nonstim_trials_numbers_on_80_block]
    nonstim_trials_20.contrastRight = trials.contrastRight[nonstim_trials_numbers_on_20_block]
    nonstim_trials_20.contrastLeft = trials.contrastLeft[nonstim_trials_numbers_on_20_block]
    nonstim_trials_20.goCueTrigger_times = trials.goCueTrigger_times[nonstim_trials_numbers_on_20_block]
    nonstim_trials_20.feedback_times = trials.feedback_times[nonstim_trials_numbers_on_20_block]
    nonstim_trials_20.response_times = trials.response_times[nonstim_trials_numbers_on_20_block]
    nonstim_trials_20.feedbackType = trials.feedbackType[nonstim_trials_numbers_on_20_block]
    nonstim_trials_20.goCue_times = trials.goCue_times[nonstim_trials_numbers_on_20_block]
    nonstim_trials_20.firstMovement_times = trials.firstMovement_times[nonstim_trials_numbers_on_20_block]
    nonstim_trials_20.probabilityLeft = trials.probabilityLeft[nonstim_trials_numbers_on_20_block]
    nonstim_trials_20.stimOn_times = trials.stimOn_times[nonstim_trials_numbers_on_20_block]
    nonstim_trials_20.choice = trials.choice[nonstim_trials_numbers_on_20_block]
    nonstim_trials_20.rewardVolume = trials.rewardVolume[nonstim_trials_numbers_on_20_block]
    nonstim_trials_20.intervals = trials.intervals[nonstim_trials_numbers_on_20_block]

    # ---- Per-PID guard: skip the session if any 80/20 trial subset is empty.
    # The downstream PETHs (lines ~588, 608, 626, 644) call np.min on each
    # subset's event-times array; an empty subset would raise
    # "zero-size array to reduction operation minimum which has no identity"
    # for every cluster in this PID. The 80/20 classification is per-PID, so
    # if a subset is empty here it is empty for all clusters; better to skip
    # the whole PID with a clear log than to fail mid-loop.
    _subset_sizes = {
        'nonstim_trials_80':    len(nonstim_trials_numbers_on_80_block),
        'inhibition_trials_80': len(inhibition_trials_numbers_on_80_block),
        'nonstim_trials_20':    len(nonstim_trials_numbers_on_20_block),
        'inhibition_trials_20': len(inhibition_trials_numbers_on_20_block),
    }
    _empty = [name for name, n in _subset_sizes.items() if n == 0]
    if _empty:
        print(
            f"  SKIP pid={pid} ({pid_to_hemisphere.get(pid, '?')}): empty trial subset(s) "
            f"{_empty}. Subset sizes: {_subset_sizes}"
        )
        continue

    diagnostic_trial_sets_by_alignment = {}
    diagnostic_trial_sets = None
    if save_diagnostic_traces == 1 or save_futureproof_sufficient_stats == 1:
        for _alignment in _alignments_for_run():
            diagnostic_trial_sets_by_alignment[_alignment] = _make_diagnostic_trial_sets(
                pid, trials,
                nonstim_trials_numbers_delta,
                nonstim_trials_numbers_on_80_block,
                nonstim_trials_numbers_on_20_block,
                inhibition_trials_numbers_on_80_block,
                inhibition_trials_numbers_on_20_block,
                alignment=_alignment,
            )
        diagnostic_trial_sets = diagnostic_trial_sets_by_alignment.get(onset_alignment)

    ###generate 1000 pseudo sessions
    pseudo_20_index_filtered,pseudo_80_index_filtered = generate_pseudo_sessions(trials)

    if save_futureproof_sufficient_stats == 1:
        _run_ids, _half_ids = _block_run_and_half_ids(
            trials, diagnostic_crossfit_guard_trials)
        trial_metadata_by_pid[str(pid)] = {
            'eid': str(eid),
            'probe_label': str(probe_label),
            'n_trials': int(len(trials.probabilityLeft)),
            'probability_left': np.asarray(trials.probabilityLeft, dtype=np.float32),
            # Compact behavioral/event columns make later stratified matching,
            # regression adjustment, and alignment audits possible without
            # reloading the raw session or spikes.
            'choice': np.asarray(trials.choice, dtype=np.float32),
            'feedback_type': np.asarray(trials.feedbackType, dtype=np.float32),
            'contrast_left': np.asarray(trials.contrastLeft, dtype=np.float32),
            'contrast_right': np.asarray(trials.contrastRight, dtype=np.float32),
            'quiescence_period': np.asarray(
                trials.quiescencePeriod, dtype=np.float32),
            'go_cue_times': np.asarray(trials.goCue_times, dtype=np.float64),
            'go_cue_trigger_times': np.asarray(
                trials.goCueTrigger_times, dtype=np.float64),
            'stim_on_times': np.asarray(trials.stimOn_times, dtype=np.float64),
            'feedback_times': np.asarray(
                trials.feedback_times, dtype=np.float64),
            'response_times': np.asarray(
                trials.response_times, dtype=np.float64),
            'first_movement_times': np.asarray(
                trials.firstMovement_times, dtype=np.float64),
            'interval_start_times': np.asarray(
                trials.intervals[:, 0], dtype=np.float64),
            'interval_end_times': np.asarray(
                trials.intervals[:, 1], dtype=np.float64),
            'block_run_id': np.asarray(_run_ids, dtype=np.int32),
            'crossfit_half_id': np.asarray(_half_ids, dtype=np.int8),
            'nonstim_trials_bs': np.asarray(nonstim_trials_numbers, dtype=np.int32),
            'nonstim_trials_delta': np.asarray(
                nonstim_trials_numbers_delta, dtype=np.int32),
            'stim_trials': np.asarray(inhibition_trials_numbers, dtype=np.int32),
            'nonstim_80_delta': np.asarray(
                nonstim_trials_numbers_on_80_block, dtype=np.int32),
            'nonstim_20_delta': np.asarray(
                nonstim_trials_numbers_on_20_block, dtype=np.int32),
            'stim_80_delta': np.asarray(
                inhibition_trials_numbers_on_80_block, dtype=np.int32),
            'stim_20_delta': np.asarray(
                inhibition_trials_numbers_on_20_block, dtype=np.int32),
            'crossfit_trial_numbers': _crossfit_trial_number_payload(
                diagnostic_trial_sets),
            'crossfit_counts': dict(
                (diagnostic_trial_sets or {}).get('counts', {})),
            'glmhmm_engaged_trials': np.asarray(engaged_idx, dtype=np.int32),
            'pseudo_block_labels': _pack_pseudo_block_labels(
                pseudo_20_index_filtered, pseudo_80_index_filtered,
                len(trials.probabilityLeft)),
            'qp_before_gocue_end_time': 0.01,
            'bs_blocklength_filterval': 10,
        }

    # ---- 'Compute everything, exclude later' ----
    # Score a permissive set of units and record every QC criterion as a field.
    # Brain region / hemisphere / drift / quality / axonal / region etc. are NOT
    # excluded here; they are columns the post-processing script filters on.
    qc_dir = make_qc_dir(figures_path, pid, save_qc_outputs)
    unit_qc_params = UnitQCParams.from_config(_bs_cfg)
    laser_onsets = laser_intervals[:, 0] if laser_intervals is not None else np.array([])

    # Computational gate (which units get the expensive BS test): IBL-label
    # floor, and optionally restrict scoring to the recorded region of interest.
    _label_ok = np.where(clusters_labels >= compute_min_IBL_label)[0]
    _allen, _beryl, _is_mid, _is_cort, _ = recorded_region_flags(sb, br, unit_qc_params)
    if compute_recorded_region == 'midbrain':
        compute_ids = np.intersect1d(_label_ok, np.where(_is_mid)[0])
    else:  # 'all'
        compute_ids = _label_ok
    compute_ids = np.asarray(compute_ids, dtype=int)

    # Per-unit QC metrics (computed, not applied).
    qc_table = unit_qc_table(
        sb, unit_qc_params, br, compute_ids,
        laser_onsets=laser_onsets, qc_dir=qc_dir,
        manual_light_artifact_unit_ids=light_artifact_units,
    )

    # Drift flag (QP-based by default) -- ALWAYS computed and stored as a per-unit
    # field so it can be filtered in BS_postprocess; independent of
    # remove_drift_units (which now only controls optional in-pipeline removal).
    drift_set = set()
    nonstationarity_set = set()
    nonstationarity_metrics_df = pd.DataFrame()
    qp_population_activity_df = pd.DataFrame()
    if len(compute_ids) > 0:
        _inhib_range = inhibition_trials_range
        if isinstance(_inhib_range, str) and _inhib_range == 'ALL':
            _inhib_range = np.arange(len(trials['contrastLeft']))
        _inhib_range = np.asarray(list(_inhib_range), dtype=int)
        _inhib_range, _ = apply_beginning_block_trial_filter(
            _inhib_range, trials['probabilityLeft'], beginning_block_trials_remove,
            qc_dir, pid, save_qc_outputs,
        )
        if len(_inhib_range) > 0:
            _block = (trials.probabilityLeft[_inhib_range] > 0.5).astype(int)
            if drift_epoch == 'quiescence':
                _d = compute_qp_drift_unit_ids(
                    spikes, compute_ids,
                    trials.goCue_times[_inhib_range], trials.quiescencePeriod[_inhib_range],
                    _block, drift_threshold=drift_threshold)
            else:
                _align = (trials.intervals[_inhib_range, 0] if onset_alignment == 'Laser onset'
                          else trials.goCue_times[_inhib_range])
                _d = compute_drift_unit_ids(
                    spikes, compute_ids, _align, _block,
                    t_before=drift_window_s[0], t_after=drift_window_s[1],
                    bin_size=bin_size, drift_threshold=drift_threshold)
            drift_set = set(int(x) for x in _d)
            try:
                nonstationarity_metrics_df, qp_population_activity_df = compute_qp_nonstationarity_metrics(
                    spikes, compute_ids,
                    trials.goCue_times[_inhib_range], trials.quiescencePeriod[_inhib_range],
                    _block,
                    n_segments=nonstationarity_n_segments,
                    min_trials=nonstationarity_min_trials,
                    min_trials_per_segment=nonstationarity_min_trials_per_segment,
                    min_trials_per_block_segment=nonstationarity_min_trials_per_block_segment,
                    low_fr_fraction_of_median=nonstationarity_low_fr_fraction_of_median,
                    min_median_fr_hz=nonstationarity_min_median_fr_hz,
                    max_qp_fr_segment_range_frac=max_qp_fr_segment_range_frac,
                    max_qp_resid_drift_range_frac=max_qp_resid_drift_range_frac,
                    max_qp_resid_drift_cv=max_qp_resid_drift_cv,
                    max_qp_resid_abs_rho_time=max_qp_resid_abs_rho_time,
                    max_qp_low_activity_fraction=max_qp_low_activity_fraction,
                    max_qp_max_low_activity_run=max_qp_max_low_activity_run,
                    min_qp_block_effect_sign_consistency=min_qp_block_effect_sign_consistency,
                    max_qp_block_effect_segment_cv=max_qp_block_effect_segment_cv,
                    max_qp_block_effect_dominance=max_qp_block_effect_dominance,
                    return_trial_metrics=True,
                )
                nonstationarity_set = set(
                    int(x) for x in nonstationarity_metrics_df.loc[
                        nonstationarity_metrics_df['flagged_nonstationary'], 'cluster_id'
                    ].to_numpy(dtype=int)
                )
                if save_qc_outputs == 1:
                    save_qp_nonstationarity_qc(
                        nonstationarity_metrics_df, qp_population_activity_df, qc_dir, pid,
                    )
            except Exception as e:
                print(f'QP nonstationarity metrics failed for PID={pid}: {e}')
                nonstationarity_set = set()
                nonstationarity_metrics_df = pd.DataFrame()
                qp_population_activity_df = pd.DataFrame()

    compute_set = set(int(x) for x in compute_ids)
    # Optional in-pipeline hard removal of drift units (default 0 -> keep & flag,
    # then exclude in BS_postprocess via exclude_drift_units).
    if remove_drift_units == 1 and drift_set:
        compute_set -= drift_set
    if remove_nonstationary_units == 1 and nonstationarity_set:
        compute_set -= nonstationarity_set
    _n_ax = int(qc_table['ax_unit'].sum()) if len(qc_table) else 0
    _n_la = int(qc_table['light_artifact_auto'].sum()) if len(qc_table) else 0
    _n_amp = int(qc_table['waveform_amplitude_outlier'].sum()) if len(qc_table) else 0
    if suppress_print_output == 0:
        print(f'PID {pid}: scoring {len(compute_set)} units (region gate={compute_recorded_region}); '
            f'flags -> {len(drift_set)} drift, {_n_ax} axonal, '
            f'{_n_la} light-artifact, {_n_amp} amplitude-outlier, '
            f'{len(nonstationarity_set)} nonstationary')
    nonstationarity_by_cluster = (
        nonstationarity_metrics_df.set_index('cluster_id')
        if len(nonstationarity_metrics_df) and 'cluster_id' in nonstationarity_metrics_df
        else pd.DataFrame()
    )

    # PETH trial sets are shared by every scored unit in this insertion. Build
    # their exact brainbox means/stds once in cluster chunks, rather than
    # rebuilding the same event bins and rescanning the spike table per unit.
    _scoring_cluster_ids = np.asarray(sorted(compute_set), dtype=int)
    (_scoring_spike_times,
     _scoring_spike_clusters,
     _unit_spike_times_by_cluster) = _prepare_scoring_spikes(
        allspikes.times, allspikes.clusters, _scoring_cluster_ids)
    _trial_bunches = {
        'nonstim_80': nonstim_trials_80,
        'stim_80': inhibition_trials_80,
        'nonstim_20': nonstim_trials_20,
        'stim_20': inhibition_trials_20,
        'nonstim_all': nonstim_trials,
        'stim_all': inhibition_trials,
    }
    _alignment_peth_caches = {}
    if (use_batched_peths == 1 and only_plot_FR == 0
            and _scoring_cluster_ids.size):
        for _alignment in _alignments_for_run():
            _alignment_peth_caches[_alignment] = _build_alignment_peth_cache(
                _alignment, _trial_bunches,
                diagnostic_trial_sets_by_alignment.get(_alignment),
                _scoring_cluster_ids, _scoring_spike_times,
                _scoring_spike_clusters, _unit_spike_times_by_cluster,
            )

    # for j in clusters[probe_label].metrics['cluster_id']:
    for j in clusters['cluster_id']:

        # Only units in the permissive compute set get scored. All QC metrics
        # for this unit are in qc_table and recorded into the row below; the
        # post-processing script decides what to exclude.
        if int(j) not in compute_set:
            continue
        qrow = qc_table.loc[int(j)]
        nsrow = (
            nonstationarity_by_cluster.loc[int(j)]
            if len(nonstationarity_by_cluster) and int(j) in nonstationarity_by_cluster.index
            else None
        )
        def _ns_metric(col, default=np.nan):
            if nsrow is None or col not in nsrow:
                return default
            val = nsrow[col]
            if isinstance(val, (bool, np.bool_)):
                return int(val)
            try:
                val = float(val)
            except Exception:
                return default
            return val if np.isfinite(val) else default

        if suppress_print_output == 0:
            print('cluster # = ' + str(j) + ', label = ' + str(clusters_labels[j]) + ', depth = ' + str(clusters.depths[j]) + ', region = ' + str(brain_acronyms_percluster[j]))

        current_cluster_spike_times = _unit_spike_times_by_cluster.get(
            int(j), np.array([], dtype=float))

        # firing_rate is recorded (filter post-hoc); only a permissive compute
        # floor is applied here to skip essentially-silent units.
        firing_rate = np.size(current_cluster_spike_times) / spikes.times[-1]
        if firing_rate <= compute_min_firing_rate:
            if suppress_print_output == 0:
                print('Firing rate below compute floor, skipping...')
            plt.close('all')
            continue

        # try:
        if plot_each_cluster == 1:
            if only_plot_FR == 1:
                plt.rcParams["figure.figsize"] = (7,5)
                fig, (ax1) = plt.subplots(1, 1)
            else:
                plt.rcParams["figure.figsize"] = (15,6)
                fig, (ax1,ax2,ax3) = plt.subplots(1, 3)
        else:
            ax1 = None
            ax2 = None
            ax3 = None

        _primary_events = _regular_alignment_events(
            _trial_bunches, onset_alignment)
        _primary_cache = _alignment_peth_caches.get(onset_alignment)
        _primary_keys = (
            ('nonstim_80', 'xkcd:violet', 'xkcd:violet', 0.4, None, 2),
            ('stim_80', 'blue', 'xkcd:violet', 0.4, 'dashed', 2),
            ('nonstim_20', 'xkcd:tangerine', 'xkcd:tangerine', 0.4, None, 2),
            ('stim_20', 'blue', 'xkcd:tangerine', 0.4, 'dashed', 2),
        )
        if only_plot_FR == 0:
            _primary_keys += (
                ('nonstim_all', 'black', 'black', 0.5, None, 2),
                ('stim_all', 'green', 'black', 0.2, None, 0.5),
            )

        try:
            _primary_peths = {}
            if _primary_cache is not None and plot_each_cluster == 0:
                for _key, *_ in _primary_keys:
                    _mean, _std = _cached_peth(
                        _primary_cache['regular'], _key, j)
                    _primary_peths[_key] = _peth_view(_mean, _std)
            elif plot_each_cluster == 0:
                # Exact low-memory fallback when batching is disabled. Match the
                # old plotting wrapper's finite-event and >=2-event checks.
                for _key, *_ in _primary_keys:
                    _events = np.asarray(_primary_events[_key], dtype=float)
                    if _events.size == 1 or not np.all(np.isfinite(_events)):
                        raise ValueError(f'invalid primary PETH events: {_key}')
                    _mean, _std = _calculate_unit_peth(
                        current_cluster_spike_times, _events,
                        t_before, t_after, min_events=2)
                    _primary_peths[_key] = _peth_view(_mean, _std)
            else:
                # Plotting remains an explicit opt-in path. The numerical
                # calculation is still the original peri_event_time_histogram.
                for (_key, _line_color, _err_color, _err_alpha,
                     _linestyle, _linewidth) in _primary_keys:
                    _line_kwargs = {'color': _line_color, 'lw': _linewidth}
                    if _linestyle is not None:
                        _line_kwargs['linestyle'] = _linestyle
                    _event_alpha = 0.4 if _key == 'stim_20' else (
                        0.6 if _key in ('nonstim_all', 'stim_all') else 0)
                    ax1, _, _peth = peri_event_time_histogram(
                        allspikes.times, allspikes.clusters,
                        _primary_events[_key], [j],
                        t_before=t_before, t_after=t_after,
                        error_bars='sem', smoothing=smoothing,
                        bin_size=bin_size, include_raster=False,
                        n_rasters=55, ax=ax1, yticks=False,
                        pethline_kwargs=_line_kwargs,
                        errbar_kwargs={
                            'color': _err_color, 'alpha': _err_alpha},
                        eventline_kwargs={
                            'color': 'black', 'alpha': _event_alpha},
                        raster_kwargs={'color': 'black', 'lw': 0.5},
                        normalize_to_baseline=normalize_to_baseline,
                    )
                    _primary_peths[_key] = _peth

            nonstim_80_peth = _primary_peths['nonstim_80']
            stim_80_peth = _primary_peths['stim_80']
            nonstim_20_peth = _primary_peths['nonstim_20']
            stim_20_peth = _primary_peths['stim_20']
            if only_plot_FR == 0:
                nonstim_all_peth = _primary_peths['nonstim_all']
                stim_all_peth = _primary_peths['stim_all']
        except Exception as exc:
            print(f'Error during PETH ({exc}). Skipping unit...')
            continue

        if plot_each_cluster == 0:
            plt.close('all')

        nonstim_80_means = nonstim_80_peth.means[0]
        nonstim_20_means = nonstim_20_peth.means[0]
        stim_80_means = stim_80_peth.means[0]
        stim_20_means = stim_20_peth.means[0]
        nonstim_80_means_err = nonstim_80_peth.stds[0]
        nonstim_20_means_err = nonstim_20_peth.stds[0]
        stim_80_means_err = stim_80_peth.stds[0]
        stim_20_means_err = stim_20_peth.stds[0]
        if only_plot_FR == 0:
            nonstim_all_means = nonstim_all_peth.means[0]
            stim_all_means = stim_all_peth.means[0]
            # Raw mean stim-trial PETH (Hz), captured before the 0->0.1 normalizer
            # floor below. Stored per unit so the post-hoc alignment check sees the
            # true laser-locked feature at t=0.
            stim_all_trace_thisunit = np.array(stim_all_peth.means[0], dtype=float)
            nonstim_all_trace_thisunit = np.array(nonstim_all_peth.means[0], dtype=float)

        # for k in range(np.size(delta_FR_8020_nonstim)):
        #     if stim_80_means[k] - stim_20_means[k] > 0 and delta_FR_8020_nonstim[k] > 0:
        #         delta_FR_8020_stim[k] = stim_80_means[k] - stim_20_means[k]
        #     elif stim_80_means[k] - stim_20_means[k] > 0 and delta_FR_8020_nonstim[k] < 0:
        #         delta_FR_8020_stim[k] = (stim_80_means[k] - stim_20_means[k]) * -1
        #     elif stim_80_means[k] - stim_20_means[k] < 0 and delta_FR_8020_nonstim[k] < 0:
        #         delta_FR_8020_stim[k] = (stim_80_means[k] - stim_20_means[k]) * -1
        #     elif stim_80_means[k] - stim_20_means[k] < 0 and delta_FR_8020_nonstim[k] > 0:
        #         delta_FR_8020_stim[k] = (stim_80_means[k] - stim_20_means[k])
        #     else:
        #         print('theres some problem with your delta FR measurement, skipping...')
        #         plt.close('all')
        #         continue

        ###calculate indices for quiescent period
        if onset_alignment == 'Go cue onset':
            first_index_for_mean = int((t_before/bin_size) - 0.4/bin_size)
            last_index_for_mean = int((t_before/bin_size))
        elif onset_alignment == 'Feedback':
            first_index_for_mean = int((t_before/bin_size) - 0.4/bin_size)
            last_index_for_mean = int((t_before/bin_size))
        elif onset_alignment == 'Laser onset':
            first_index_for_mean = int((t_before/bin_size) + 0.2/bin_size)
            last_index_for_mean = int((t_before/bin_size) + 0.8/bin_size)

        QP_firing_rate = (np.nanmean(nonstim_80_means[first_index_for_mean:last_index_for_mean]) + np.nanmean(nonstim_20_means[first_index_for_mean:last_index_for_mean])) / 2

        ### calculate difference in firing rate curves
        # delta_FR_8020_nonstim = nonstim_80_means - nonstim_20_means
        # ### way to ensure delta_FR_8020_stim reflects sign of change relative to nonstim
        # delta_FR_8020_stim = (stim_80_means - stim_20_means) * np.sign(delta_FR_8020_nonstim)
        # ### delta_FR_8020_nonstim should always be positive
        # delta_FR_8020_nonstim = abs(delta_FR_8020_nonstim)

        ############CONTROL: add curves instead of subtracting
        delta_FR_8020_nonstim = nonstim_80_means - nonstim_20_means
        delta_FR_8020_stim = stim_80_means - stim_20_means

        ### calculate z-score
        delta_FR_8020_nonstim_err_est = np.sqrt((nonstim_80_means_err**2 / len(nonstim_trials_80.intervals[:,0])) + (nonstim_20_means_err**2 / len(nonstim_trials_20.intervals[:,0])))
        delta_FR_8020_stim_err_est = np.sqrt((stim_80_means_err**2 / len(inhibition_trials_80.intervals[:,0])) + (stim_20_means_err**2 / len(inhibition_trials_20.intervals[:,0])))

        z_score = (delta_FR_8020_stim - delta_FR_8020_nonstim) / np.sqrt(delta_FR_8020_nonstim_err_est**2 + delta_FR_8020_stim_err_est**2)

        if np.logical_or(np.isnan(z_score), np.isinf(z_score)).any():
            if suppress_print_output == 0:
                print('Z-score contains INF or NaN values, skipping...')
            plt.close('all')
            continue
        elif np.logical_or(z_score > 5, z_score < -5).any():
            if suppress_print_output == 0:
                print('Z-score out of reasonable bounds, skipping...')
            plt.close('all')
            continue

        ### normalization for plotting -- compute ALL THREE modes in parallel so
        ### they can be compared post-hoc on the SAME units. The z-score above is
        ### computed from the raw (un-normalized) deltas, so it is identical across
        ### modes; only the saved delta-FR traces differ.
        if only_plot_FR == 0:
            # (1) per_bin (original): 0 -> 0.1 floor, divide by per-bin all-trial PETH.
            _ns_pb = nonstim_all_means.astype(float).copy(); _ns_pb[_ns_pb == 0] = 0.1
            _st_pb = stim_all_means.astype(float).copy();     _st_pb[_st_pb == 0] = 0.1
            delta_nonstim_per_bin = delta_FR_8020_nonstim / _ns_pb
            delta_stim_per_bin    = delta_FR_8020_stim / _st_pb

            # (2) baseline_scalar: divide by a single floored baseline scalar (whole
            # window if scalar_baseline_window is None, else that window). Robust to
            # the pre-laser dead zone.
            if scalar_baseline_window is None:
                _ns_scalar = np.nanmean(nonstim_all_means)
                _st_scalar = np.nanmean(stim_all_means)
            else:
                _onset = t_before
                _b0 = max(0, int(round((_onset + scalar_baseline_window[0]) / bin_size)))
                _b1 = max(_b0 + 1, int(round((_onset + scalar_baseline_window[1]) / bin_size)))
                _ns_scalar = np.nanmean(nonstim_all_means[_b0:_b1])
                _st_scalar = np.nanmean(stim_all_means[_b0:_b1])
            _ns_scalar = max(_ns_scalar, scalar_min_fr) if np.isfinite(_ns_scalar) else scalar_min_fr
            _st_scalar = max(_st_scalar, scalar_min_fr) if np.isfinite(_st_scalar) else scalar_min_fr
            delta_nonstim_scalar = delta_FR_8020_nonstim / _ns_scalar
            delta_stim_scalar    = delta_FR_8020_stim / _st_scalar

            # (3) zero_2_nan: treat near-zero-FR bins as MISSING in both the block
            # means and the normalizer. NOTE: the PETH means are Gaussian-smoothed
            # (smoothing>0), so "empty" bins are tiny positive tails (~1e-40), NOT
            # exactly 0 -- an `== 0` test would never fire. We therefore threshold at
            # zero_nan_threshold (Hz): any bin at/below it is treated as empty.
            _z = zero_nan_threshold
            _ns80 = nonstim_80_means.astype(float).copy(); _ns80[_ns80 <= _z] = np.nan
            _ns20 = nonstim_20_means.astype(float).copy(); _ns20[_ns20 <= _z] = np.nan
            _st80 = stim_80_means.astype(float).copy();    _st80[_st80 <= _z] = np.nan
            _st20 = stim_20_means.astype(float).copy();    _st20[_st20 <= _z] = np.nan
            _ns_all_nan = nonstim_all_means.astype(float).copy(); _ns_all_nan[_ns_all_nan <= _z] = np.nan
            _st_all_nan = stim_all_means.astype(float).copy();    _st_all_nan[_st_all_nan <= _z] = np.nan
            with np.errstate(invalid='ignore', divide='ignore'):
                delta_nonstim_zero_2_nan = (_ns80 - _ns20) / _ns_all_nan
                delta_stim_zero_2_nan    = (_st80 - _st20) / _st_all_nan

            # Default traces (back-compat + existing plots) follow normalize_mode.
            _sel = {'per_bin': (delta_nonstim_per_bin, delta_stim_per_bin),
                    'baseline_scalar': (delta_nonstim_scalar, delta_stim_scalar),
                    'zero_2_nan': (delta_nonstim_zero_2_nan, delta_stim_zero_2_nan)}
            delta_FR_8020_nonstim_normalized, delta_FR_8020_stim_normalized = _sel.get(
                normalize_mode, _sel['per_bin'])

            # Inclusion gate: mode-INDEPENDENT and robust. Use the scalar mode (which
            # is floored and cannot blow up) so we drop only genuinely broken units,
            # NOT per-bin blow-ups -- those low-baseline/ITI-quiescent units are
            # exactly what the scalar/zero_2_nan modes are meant to rescue, and they
            # must be present in all three saved trace sets for a fair comparison.
            mean_delta_FR_8020_nonstim = np.nanmean(delta_nonstim_scalar[first_index_for_mean:last_index_for_mean])
            mean_delta_FR_8020_stim = np.nanmean(delta_stim_scalar[first_index_for_mean:last_index_for_mean])
            if np.isnan(mean_delta_FR_8020_nonstim) or np.isnan(mean_delta_FR_8020_stim):
                if suppress_print_output == 0:
                    print('MEAN delta FR is NaN (no valid bins in window), skipping...')
                plt.close('all')
                continue

        current_unit_spike_times = current_cluster_spike_times
        qp_fr_per_trial_thisunit = None
        if save_futureproof_sufficient_stats == 1:
            qp_fr_per_trial_thisunit = _quiescent_fr_per_trial(
                current_unit_spike_times, trials)

        #### perform BS analysis using previously created pseudo sessions

        # BS_score, pval_real, pct95_pseudo, fr_80_trials_nonstim, fr_20_trials_nonstim, fr_80_trials_inhibition, fr_20_trials_inhibition = isbiasblockselective_03(current_unit_spike_times, trials.probabilityLeft, trials.goCue_times, excitation_trials_numbers,inhibition_trials_numbers,nonstim_trials_numbers,
        #                 pseudo_20_index_filtered, pseudo_80_index_filtered)

        BS_score, p_empirical, pval_real, stat_real, stat_pseudo, fr_80_nonstim, fr_20_nonstim, fr_80_inhib, fr_20_inhib = isbiasblockselective_perm_vector(
                        current_unit_spike_times, trials.probabilityLeft, trials.goCue_times, inhibition_trials_numbers, nonstim_trials_numbers,
                        pseudo_20_index_filtered, pseudo_80_index_filtered, trials.quiescencePeriod)

        # if exclude_drifty_units == 1:
        #     if pct50_pseudo < 0.05:
        #         print('Drifty unit, skipping...')
        #         plt.close('all')
        #         continue

        # delta_fr_nonstim = np.nanmean(fr_80_trials_nonstim) - np.nanmean(fr_20_trials_nonstim)
        # delta_fr_inhibition = np.nanmean(fr_80_trials_inhibition) - np.nanmean(fr_20_trials_inhibition)

        # Axonal flag from the per-unit QC table (recorded, never excluded here).
        axonal_unit_score = int(qrow['ax_unit'])

        if suppress_print_output == 0:
            print('Axonal unit score = ' + str(axonal_unit_score))

            print('BS score = ' + str(BS_score) + ', P value = ' + str(p_empirical))
        if only_plot_FR == 0:
            if suppress_print_output == 0:
                print('Mean FR change nonstim = ' + str(mean_delta_FR_8020_nonstim) + ', QP FR = ' + str(QP_firing_rate))
                print('Mean FR change stim = ' + str(mean_delta_FR_8020_stim))

        if plot_only_BS_units == 1:
            if BS_score == 0:
                plt.close('all')
                continue

        unit_diagnostic_traces = None
        if (only_plot_FR == 0
                and (save_diagnostic_traces == 1
                     or save_futureproof_sufficient_stats == 1)
                and diagnostic_trial_sets is not None):
            if (_primary_cache is not None
                    and _primary_cache.get('diagnostic') is not None):
                unit_diagnostic_traces = (
                    _compute_unit_diagnostic_traces_from_cache(
                        j, diagnostic_trial_sets,
                        _primary_cache['diagnostic'], t_before))
            else:
                unit_diagnostic_traces = _compute_unit_diagnostic_traces(
                    current_cluster_spike_times, diagnostic_trial_sets)

        # if analyze_latency == 1:

        #     latencies = np.empty(len(trials.intervals[excitation_trials_numbers][:,0]))
        #     latencies[:] = np.NaN
        #     for k in np.arange(0,np.size(trials.intervals[excitation_trials_numbers][:,0])):
        #         if current_cluster_spike_times[np.where(current_cluster_spike_times > trials.intervals[excitation_trials_numbers][:,0][k])].size == 0:
        #             continue
        #         latencies[k] = current_cluster_spike_times[np.where(current_cluster_spike_times > trials.intervals[excitation_trials_numbers][:,0][k])][0] - trials.intervals[excitation_trials_numbers][:,0][k]

        #     print('median EX latency = ' + str(np.nanmedian(latencies)))
        #     print('mean EX latency = ' + str(np.nanmean(latencies)))
        #     percent_below_2ms = np.size(np.where(latencies < 0.002))/np.size(latencies[~np.isnan(latencies)]) * 100
        #     print('percent EX latency below 2ms = ' + str(percent_below_2ms))
        #     # print('latency confidence 90 = ' + str(np.nanpercentile(latencies,10)))
        #     # print('latency confidence 85 = ' + str(np.nanpercentile(latencies,15)))
        #     # print('latency confidence 80 = ' + str(np.nanpercentile(latencies,20)))
        #     latencies_EX = latencies

        #     latencies = np.empty(len(trials.intervals[inhibition_trials_numbers][:,0]))
        #     latencies[:] = np.NaN
        #     for k in np.arange(0,np.size(trials.intervals[inhibition_trials_numbers][:,0])):
        #         if current_cluster_spike_times[np.where(current_cluster_spike_times > trials.intervals[inhibition_trials_numbers][:,0][k])].size == 0:
        #             continue
        #         latencies[k] = current_cluster_spike_times[np.where(current_cluster_spike_times > trials.intervals[inhibition_trials_numbers][:,0][k])][0] - trials.intervals[inhibition_trials_numbers][:,0][k]

        #     print('median IN latency = ' + str(np.nanmedian(latencies)))
        #     print('mean IN latency = ' + str(np.nanmean(latencies)))
        #     percent_below_2ms = np.size(np.where(latencies < 0.002))/np.size(latencies[~np.isnan(latencies)]) * 100
        #     print('percent IN latency below 2ms = ' + str(percent_below_2ms))

        #     latencies_IN = latencies

        #     pre_inhibition_FR = np.empty(len(trials.intervals[inhibition_trials_numbers][:,0]))
        #     pre_inhibition_FR[:] = np.NaN
        #     post_inhibition_FR = np.empty(len(trials.intervals[inhibition_trials_numbers][:,0]))
        #     post_inhibition_FR[:] = np.NaN
        #     inhibition_latency_period = latency_threshold
        #     for k in np.arange(0,np.size(trials.intervals[inhibition_trials_numbers][:,0])):
        #         spike_times_in_post_stim = current_cluster_spike_times[np.where(np.logical_and(current_cluster_spike_times > trials.intervals[inhibition_trials_numbers][:,0][k], current_cluster_spike_times < trials.intervals[inhibition_trials_numbers][:,0][k] + inhibition_latency_period))[0]]
        #         spike_times_in_pre_stim = current_cluster_spike_times[np.where(np.logical_and(current_cluster_spike_times < trials.intervals[inhibition_trials_numbers][:,0][k], current_cluster_spike_times > trials.intervals[inhibition_trials_numbers][:,0][k] - inhibition_latency_period))[0]]
        #         post_inhibition_FR[k] = np.size(spike_times_in_post_stim)/inhibition_latency_period
        #         pre_inhibition_FR[k] = np.size(spike_times_in_pre_stim)/inhibition_latency_period

        #     if np.nanmean(pre_inhibition_FR) == 0:
        #         print('no spikes detected in any pre trials, skipping...')
        #         plt.close('all')
        #         continue

        #     x, pval_inverselatency = stats.wilcoxon(post_inhibition_FR,pre_inhibition_FR)
        #     print('pval inverse latency = ' + str(pval_inverselatency))

        #     if use_latency_threshold == 1:
        #         if np.nanmedian(latencies_EX) > latency_threshold and np.nanmedian(latencies_IN) > latency_threshold and pval_inverselatency > 0.05:
        #             print('Opto latency below threshold, skipping...')
        #             plt.close('all')
        #             continue

        # if only_analyze_responsive_units == 1:
        #     inhibition_FR_pre_responsive = np.empty(len(trials.intervals[inhibition_trials_numbers][:,0]))
        #     inhibition_FR_pre_responsive[:] = np.NaN
        #     inhibition_FR_post_responsive = np.empty(len(trials.intervals[inhibition_trials_numbers][:,0]))
        #     inhibition_FR_post_responsive[:] = np.NaN
        #     inhibition_latency_period = responsive_window
        #     for k in np.arange(0,np.size(trials.intervals[inhibition_trials_numbers][:,0])):
        #         spike_times_in_post_stim = current_cluster_spike_times[np.where(np.logical_and(current_cluster_spike_times > trials.intervals[inhibition_trials_numbers][:,0][k], current_cluster_spike_times < trials.intervals[inhibition_trials_numbers][:,0][k] + inhibition_latency_period))[0]]
        #         spike_times_in_pre_stim = current_cluster_spike_times[np.where(np.logical_and(current_cluster_spike_times < trials.intervals[inhibition_trials_numbers][:,0][k], current_cluster_spike_times > trials.intervals[inhibition_trials_numbers][:,0][k] - inhibition_latency_period))[0]]
        #         inhibition_FR_post_responsive[k] = np.size(spike_times_in_post_stim)/inhibition_latency_period
        #         inhibition_FR_pre_responsive[k] = np.size(spike_times_in_pre_stim)/inhibition_latency_period

        #     if np.nanmean(inhibition_FR_post_responsive) == 0 and np.nanmean(inhibition_FR_pre_responsive) == 0:
        #         print('No firing in pre or post responsive window, skipping...')
        #         plt.close('all')
        #         continue
        #     x, pval_responsive_inhibition = stats.wilcoxon(inhibition_FR_post_responsive,inhibition_FR_pre_responsive)
        #     print('pval inhibition response = ' + str(pval_responsive_inhibition))

        #     excitation_FR_pre_responsive = np.empty(len(trials.intervals[excitation_trials_numbers][:,0]))
        #     excitation_FR_pre_responsive[:] = np.NaN
        #     excitation_FR_post_responsive = np.empty(len(trials.intervals[excitation_trials_numbers][:,0]))
        #     excitation_FR_post_responsive[:] = np.NaN
        #     excitation_latency_period = responsive_window
        #     for k in np.arange(0,np.size(trials.intervals[excitation_trials_numbers][:,0])):
        #         spike_times_in_post_stim = current_cluster_spike_times[np.where(np.logical_and(current_cluster_spike_times > trials.intervals[excitation_trials_numbers][:,0][k], current_cluster_spike_times < trials.intervals[excitation_trials_numbers][:,0][k] + excitation_latency_period))[0]]
        #         spike_times_in_pre_stim = current_cluster_spike_times[np.where(np.logical_and(current_cluster_spike_times < trials.intervals[excitation_trials_numbers][:,0][k], current_cluster_spike_times > trials.intervals[excitation_trials_numbers][:,0][k] - excitation_latency_period))[0]]
        #         excitation_FR_post_responsive[k] = np.size(spike_times_in_post_stim)/excitation_latency_period
        #         excitation_FR_pre_responsive[k] = np.size(spike_times_in_pre_stim)/excitation_latency_period

        #     if np.nanmean(excitation_FR_post_responsive) == 0 and np.nanmean(excitation_FR_post_responsive) == 0:
        #         print('No firing in pre or post responsive window, skipping...')
        #         plt.close('all')
        #         continue
        #     x, pval_responsive_excitation = stats.wilcoxon(excitation_FR_post_responsive,excitation_FR_pre_responsive)
        #     print('pval excitation response = ' + str(pval_responsive_excitation))

        #     if pval_responsive_excitation > 0.01 and pval_responsive_inhibition > 0.01:
        #         print('Non-responsive unit! (not skipping yet)')

        current_unit_allen_label = brain_acronyms_percluster[j]
        if isinstance(current_unit_allen_label, str) == 0:
            current_unit_beryl_label = np.nan
        else:
            current_unit_beryl_label = br.acronym2acronym(current_unit_allen_label, mapping='Beryl')


        ### All data gets appended here
        if only_plot_FR == 0:
            delta_fr_nonstim_all.append(delta_FR_8020_nonstim_normalized)
            delta_fr_inhibition_all.append(delta_FR_8020_stim_normalized)
            zscore_all.append(z_score)
            delta_fr_nonstim_per_bin.append(delta_nonstim_per_bin)
            delta_fr_stim_per_bin.append(delta_stim_per_bin)
            delta_fr_nonstim_scalar.append(delta_nonstim_scalar)
            delta_fr_stim_scalar.append(delta_stim_scalar)
            delta_fr_nonstim_zero_2_nan.append(delta_nonstim_zero_2_nan)
            delta_fr_stim_zero_2_nan.append(delta_stim_zero_2_nan)
            stim_all_trace.append(stim_all_trace_thisunit)
            nonstim_all_trace.append(nonstim_all_trace_thisunit)
            if save_futureproof_sufficient_stats == 1:
                qp_fr_per_trial_all.append(
                    np.asarray(qp_fr_per_trial_thisunit, dtype=np.float32))
            if save_raw_block_peths == 1:
                raw_block_peth_traces['trace_nonstim_80_raw'].append(np.asarray(nonstim_80_means, dtype=float))
                raw_block_peth_traces['trace_nonstim_20_raw'].append(np.asarray(nonstim_20_means, dtype=float))
                raw_block_peth_traces['trace_stim_80_raw'].append(np.asarray(stim_80_means, dtype=float))
                raw_block_peth_traces['trace_stim_20_raw'].append(np.asarray(stim_20_means, dtype=float))
                raw_block_peth_traces['trace_nonstim_80_sem_raw'].append(
                    np.asarray(nonstim_80_means_err, dtype=float) / np.sqrt(max(len(nonstim_trials_80.intervals[:, 0]), 1)))
                raw_block_peth_traces['trace_nonstim_20_sem_raw'].append(
                    np.asarray(nonstim_20_means_err, dtype=float) / np.sqrt(max(len(nonstim_trials_20.intervals[:, 0]), 1)))
                raw_block_peth_traces['trace_stim_80_sem_raw'].append(
                    np.asarray(stim_80_means_err, dtype=float) / np.sqrt(max(len(inhibition_trials_80.intervals[:, 0]), 1)))
                raw_block_peth_traces['trace_stim_20_sem_raw'].append(
                    np.asarray(stim_20_means_err, dtype=float) / np.sqrt(max(len(inhibition_trials_20.intervals[:, 0]), 1)))
                raw_block_peth_traces['trace_nonstim_all_sem_raw'].append(
                    np.asarray(nonstim_all_peth.stds[0], dtype=float) / np.sqrt(max(len(nonstim_trials.intervals[:, 0]), 1)))
                raw_block_peth_traces['trace_stim_all_sem_raw'].append(
                    np.asarray(stim_all_peth.stds[0], dtype=float) / np.sqrt(max(len(inhibition_trials.intervals[:, 0]), 1)))
            if save_diagnostic_traces == 1 or save_futureproof_sufficient_stats == 1:
                if unit_diagnostic_traces is None:
                    for _key in diagnostic_trace_lists:
                        diagnostic_trace_lists[_key].append(_empty_trace())
                else:
                    for _mode in ('per_bin', 'baseline_scalar', 'zero_2_nan'):
                        diagnostic_trace_lists[f'trace_nonstim_split_a_{_mode}'].append(
                            unit_diagnostic_traces['split_a'][_mode])
                        diagnostic_trace_lists[f'trace_nonstim_split_b_{_mode}'].append(
                            unit_diagnostic_traces['split_b'][_mode])
                        diagnostic_trace_lists[f'trace_nonstim_trialmatched_{_mode}'].append(
                            unit_diagnostic_traces['trialmatched'][_mode])
                        diagnostic_trace_lists[f'trace_nonstim_trialmatched_sem_{_mode}'].append(
                            unit_diagnostic_traces['trialmatched_sem'][_mode])
                        for _fold in ('a', 'b'):
                            _fold_data = unit_diagnostic_traces.get(
                                'block_crossfit', {}).get(_fold, {})
                            for _role in ('reference', 'control_eval', 'stim_eval'):
                                _trace = _fold_data.get(_role, {}).get(_mode)
                                if _trace is None:
                                    _trace = _empty_trace()
                                diagnostic_trace_lists[
                                    f'trace_block_crossfit_{_role}_{_fold}_{_mode}'
                                ].append(np.asarray(_trace, dtype=np.float32))
                    if save_futureproof_sufficient_stats == 1:
                        for _fold in ('a', 'b'):
                            _fold_raw = unit_diagnostic_traces.get(
                                'block_crossfit_raw', {}).get(_fold, {})
                            for _role in ('reference', 'control_eval', 'stim_eval'):
                                _role_raw = _fold_raw.get(_role, {})
                                for _raw_name in ('block80_raw', 'block20_raw', 'all_mean'):
                                    _trace = _role_raw.get(_raw_name)
                                    if _trace is None:
                                        _trace = _empty_trace()
                                    diagnostic_trace_lists[
                                        f'trace_block_crossfit_{_role}_{_fold}_{_raw_name}'
                                    ].append(np.asarray(_trace, dtype=np.float32))
            if additional_alignment_stores:
                for _alignment, _store in additional_alignment_stores.items():
                    _payload = _compute_alignment_payload(
                        j, _alignment, _trial_bunches,
                        unit_spike_times=current_cluster_spike_times,
                        diagnostic_sets=diagnostic_trial_sets_by_alignment.get(_alignment),
                        peth_cache=_alignment_peth_caches.get(_alignment),
                    )
                    _append_alignment_payload(_store, _payload)
            _diag_counts = (diagnostic_trial_sets or {}).get('counts', {})

            clusters_info_DF = pd.concat([clusters_info_DF, pd.DataFrame(
                index=[clusters_info_DF.shape[0] + 1], data={
                    # --- session-level metadata (for post-hoc region/hemisphere filters) ---
                    'pid': pid,
                    'eid': str(eid),
                    'probe_label': str(probe_label),
                    'mouse': pid_to_mouse.get(pid, 'nan'),
                    'brain_region_inhibited': pid_to_region.get(pid, 'nan'),
                    'condition': pid_to_hemisphere[pid],
                    'hemisphere': pid_to_hemisphere[pid],          # backward-compat alias
                    'hemisphere_stim': pid_to_hemi_stim.get(pid, 'nan'),
                    'hemisphere_recorded': pid_to_hemi_recorded.get(pid, 'nan'),
                    'clustnum': j,
                    # --- recorded-region labels (Allen + broad midbrain via depth overrides) ---
                    'Allenregion': str(current_unit_allen_label),
                    'Berylregion': str(current_unit_beryl_label),
                    'is_midbrain': bool(qrow['is_midbrain']),
                    'is_cortical': bool(qrow['is_cortical']),
                    'used_depth_override': bool(qrow['used_depth_override']),
                    'depth': float(qrow['depth']),
                    # --- per-unit QC (recorded, not applied) ---
                    'IBL_label': clusters_labels[j],
                    'presence_ratio': float(qrow['presence_ratio']),
                    'firing_rate': float(firing_rate),
                    'ax_unit': axonal_unit_score,
                    'pt_ratio': float(qrow['pt_ratio']),
                    'light_artifact_auto': int(qrow['light_artifact_auto']),
                    'waveform_amplitude_outlier': int(qrow['waveform_amplitude_outlier']),
                    'drift_unit': int(int(j) in drift_set),
                    'nonstationary_unit': int(int(j) in nonstationarity_set),
                    'qp_fr_median': _ns_metric('qp_fr_median'),
                    'qp_fr_segment_range_frac': _ns_metric('qp_fr_segment_range_frac'),
                    'qp_resid_drift_range_frac': _ns_metric('qp_resid_drift_range_frac'),
                    'qp_resid_drift_cv': _ns_metric('qp_resid_drift_cv'),
                    'qp_resid_abs_rho_time': _ns_metric('qp_resid_abs_rho_time'),
                    'qp_low_activity_fraction': _ns_metric('qp_low_activity_fraction'),
                    'qp_max_low_activity_run': _ns_metric('qp_max_low_activity_run'),
                    'qp_n_block_effect_segments': _ns_metric('qp_n_block_effect_segments'),
                    'qp_block_effect_global': _ns_metric('qp_block_effect_global'),
                    'qp_block_effect_segment_cv': _ns_metric('qp_block_effect_segment_cv'),
                    'qp_block_effect_sign_consistency': _ns_metric('qp_block_effect_sign_consistency'),
                    'qp_block_effect_dominance': _ns_metric('qp_block_effect_dominance'),
                    'nonstationarity_reasons': (
                        str(nsrow['nonstationarity_reasons'])
                        if nsrow is not None and 'nonstationarity_reasons' in nsrow else ''
                    ),
                    # --- BS trial definition provenance ---
                    'bs_definition_trial_mode': bs_definition_summary.get('bs_definition_trial_mode', bs_definition_trial_mode),
                    'n_nonstim_trials_before_bs_definition': bs_definition_summary.get('n_nonstim_trials_before_bs_definition', np.nan),
                    'n_nonstim_trials_after_bs_definition': bs_definition_summary.get('n_nonstim_trials_after_bs_definition', np.nan),
                    'n_nonstim_trials_removed_by_bs_definition': bs_definition_summary.get('n_nonstim_trials_removed_by_bs_definition', np.nan),
                    'n_inhibition_trials_before_bs_definition': bs_definition_summary.get('n_inhibition_trials_before_bs_definition', np.nan),
                    'n_inhibition_trials_after_bs_definition': bs_definition_summary.get('n_inhibition_trials_after_bs_definition', np.nan),
                    'n_inhibition_trials_removed_by_bs_definition': bs_definition_summary.get('n_inhibition_trials_removed_by_bs_definition', np.nan),
                    # --- BS test results + power diagnostics ---
                    'BS_score': BS_score,
                    'pval_real': pval_real,
                    'pval_empirical': p_empirical,
                    'stat_real': stat_real,
                    'n_80_nonstim': int(len(fr_80_nonstim)),
                    'n_20_nonstim': int(len(fr_20_nonstim)),
                    'n_80_inhib': int(len(fr_80_inhib)),
                    'n_20_inhib': int(len(fr_20_inhib)),
                    'n_80_nonstim_delta_peth': int(len(nonstim_trials_numbers_on_80_block)),
                    'n_20_nonstim_delta_peth': int(len(nonstim_trials_numbers_on_20_block)),
                    'n_80_inhib_delta_peth': int(len(inhibition_trials_numbers_on_80_block)),
                    'n_20_inhib_delta_peth': int(len(inhibition_trials_numbers_on_20_block)),
                    'diag_n_split_a_80_nonstim': _diag_counts.get('split_a_80', np.nan),
                    'diag_n_split_a_20_nonstim': _diag_counts.get('split_a_20', np.nan),
                    'diag_n_split_b_80_nonstim': _diag_counts.get('split_b_80', np.nan),
                    'diag_n_split_b_20_nonstim': _diag_counts.get('split_b_20', np.nan),
                    'diag_n_trialmatched_80_nonstim': _diag_counts.get('trialmatch_80', np.nan),
                    'diag_n_trialmatched_20_nonstim': _diag_counts.get('trialmatch_20', np.nan),
                    'diag_n_trialmatched_repeats': _diag_counts.get('trialmatch_repeats', np.nan),
                    'diag_crossfit_guard_trials': _diag_counts.get('crossfit_guard_trials', np.nan),
                    'diag_n_crossfit_a_reference_80': _diag_counts.get('crossfit_a_reference_80', np.nan),
                    'diag_n_crossfit_a_reference_20': _diag_counts.get('crossfit_a_reference_20', np.nan),
                    'diag_n_crossfit_a_control_80': _diag_counts.get('crossfit_a_control_80', np.nan),
                    'diag_n_crossfit_a_control_20': _diag_counts.get('crossfit_a_control_20', np.nan),
                    'diag_n_crossfit_a_stim_80': _diag_counts.get('crossfit_a_stim_80', np.nan),
                    'diag_n_crossfit_a_stim_20': _diag_counts.get('crossfit_a_stim_20', np.nan),
                    'diag_n_crossfit_b_reference_80': _diag_counts.get('crossfit_b_reference_80', np.nan),
                    'diag_n_crossfit_b_reference_20': _diag_counts.get('crossfit_b_reference_20', np.nan),
                    'diag_n_crossfit_b_control_80': _diag_counts.get('crossfit_b_control_80', np.nan),
                    'diag_n_crossfit_b_control_20': _diag_counts.get('crossfit_b_control_20', np.nan),
                    'diag_n_crossfit_b_stim_80': _diag_counts.get('crossfit_b_stim_80', np.nan),
                    'diag_n_crossfit_b_stim_20': _diag_counts.get('crossfit_b_stim_20', np.nan),
                    'Delta_nonstim': mean_delta_FR_8020_nonstim,
                    'Delta_stim': mean_delta_FR_8020_stim})])


        # if excitation_traces_percluster == []:
        #     if zscore_normalize == 0:
        #         nonstim80_traces_percluster = nonstim_80_means
        #         nonstim20_traces_percluster = nonstim_20_means
        #         stim80_traces_percluster = stim_80_means
        #         stim20_traces_percluster = stim_20_means
        #     # else:
        #         # excitation_traces_percluster = excitation_means_z
        #         # inhibition_traces_percluster = inhibition_means_z
        # else:
        #     if zscore_normalize == 0:
        #         nonstim80_traces_percluster = np.vstack([nonstim80_traces_percluster, nonstim_80_means])
        #         nonstim20_traces_percluster = np.vstack([nonstim20_traces_percluster, nonstim_20_means])
        #         stim80_traces_percluster = np.vstack([stim80_traces_percluster, stim_80_means])
        #         stim20_traces_percluster = np.vstack([stim20_traces_percluster, stim_20_means])
        #     # else:
        #     #     excitation_traces_percluster = np.vstack([excitation_traces_percluster, excitation_means_z])
        #     #     inhibition_traces_percluster = np.vstack([inhibition_traces_percluster, inhibition_means_z])

        if only_plot_FR == 0 and plot_each_cluster == 1:
            # Canonical templates give an averaged peak-channel waveform; the
            # old per-spike snippet overlay is not available from templates.
            cluster_wf = cluster_peak_waveform(waveforms, j)
            if cluster_wf is not None:
                ax3.plot(cluster_wf, 'k-', linewidth=3)
            ax2.axhline(y=0, color='gray', linestyle='--', linewidth=1)
            ax2.plot(z_score, 'b-', linewidth=3)

        if plot_each_cluster == 1:
            if plot_edge1 > plot_edge:
                plot_limit = plot_edge1
            else:
                plot_limit = plot_edge

            if np.isnan(plot_limit) == 1:
                plot_limit = 1

            if np.isinf(plot_limit) == 1:
                plot_limit = 100

            if plot_limit > 100:
                ax1.set_yticks(np.arange(0, 200, step=20))
            if plot_limit < 100 and plot_limit > 20:
                ax1.set_yticks(np.arange(0, 100, step=5))
            if plot_limit < 20:
                ax1.set_yticks(np.arange(0, 20, step=1))
            ax1.set_ylim([0, plot_limit])
            ax1.set_xlabel('Time from laser onset (s)', fontsize = 13)
            ax1.set_ylabel('Firing rate (spikes/s)', fontsize = 13)

            plt.show()
            plt.waitforbuttonpress
        plt.close('all')
        # except:
        #     print('Error with cluster (numspikes = ' + str(np.size(current_cluster_spike_indices)) + '). Skipping cluster...')
        #     continue

################# SAVE  -> single results file for BS_postprocess.py
if only_plot_FR == 0:
    import pickle
    # Traces are parallel lists, row-aligned to clusters_info_DF (appended
    # together per unit). reset_index so positional index 0..N-1 matches.
    bs_results = {
        'units': clusters_info_DF.reset_index(drop=True),
        'trace_nonstim': delta_fr_nonstim_all,     # delta FR 80-20, normalised, per unit
        'trace_stim': delta_fr_inhibition_all,
        'trace_zscore': zscore_all,
        # All three normalization modes, parallel, same units (select with
        # BS_postprocess.use_norm(data, mode)). 'trace_nonstim'/'trace_stim' above
        # mirror whichever normalize_mode was set, for back-compat.
        'trace_nonstim_per_bin': delta_fr_nonstim_per_bin,
        'trace_stim_per_bin': delta_fr_stim_per_bin,
        'trace_nonstim_baseline_scalar': delta_fr_nonstim_scalar,
        'trace_stim_baseline_scalar': delta_fr_stim_scalar,
        'trace_nonstim_zero_2_nan': delta_fr_nonstim_zero_2_nan,
        'trace_stim_zero_2_nan': delta_fr_stim_zero_2_nan,
        'trace_stim_all': stim_all_trace,          # raw mean stim-trial PETH (Hz) -- laser-alignment QC
        'trace_nonstim_all': nonstim_all_trace,    # raw mean control-trial PETH (Hz)
        'peth_time': np.arange(-t_before, t_after, bin_size),
        'bin_size': bin_size,
        't_before': t_before,
        't_after': t_after,
        'onset_alignment': onset_alignment,
        'smoothing_window_ms': post_smooth_window_ms,
        'peth_smoothing_s': smoothing,
        'peth_smoothing_ms': float(smoothing) * 1000.0,
        'diagnostic_trace_modes': ('per_bin', 'baseline_scalar', 'zero_2_nan'),
        'run_config': {
            'compute_recorded_region': compute_recorded_region,
            'compute_min_IBL_label': compute_min_IBL_label,
            'compute_min_firing_rate': compute_min_firing_rate,
            'drift_epoch': drift_epoch,
            'remove_drift_units': remove_drift_units,
            'remove_nonstationary_units': remove_nonstationary_units,
            'nonstationarity_n_segments': nonstationarity_n_segments,
            'nonstationarity_min_trials': nonstationarity_min_trials,
            'nonstationarity_low_fr_fraction_of_median': nonstationarity_low_fr_fraction_of_median,
            'nonstationarity_min_median_fr_hz': nonstationarity_min_median_fr_hz,
            'max_qp_fr_segment_range_frac': max_qp_fr_segment_range_frac,
            'max_qp_resid_drift_range_frac': max_qp_resid_drift_range_frac,
            'max_qp_resid_drift_cv': max_qp_resid_drift_cv,
            'max_qp_resid_abs_rho_time': max_qp_resid_abs_rho_time,
            'max_qp_low_activity_fraction': max_qp_low_activity_fraction,
            'max_qp_max_low_activity_run': max_qp_max_low_activity_run,
            'min_qp_block_effect_sign_consistency': min_qp_block_effect_sign_consistency,
            'max_qp_block_effect_segment_cv': max_qp_block_effect_segment_cv,
            'max_qp_block_effect_dominance': max_qp_block_effect_dominance,
            'beginning_block_trials_remove': beginning_block_trials_remove,
            'use_GLMHMM_engaged_indices': use_GLMHMM_engaged_indices,
            'opto_trials_GLMHMM': opto_trials_GLMHMM,
            'n_states': n_states,
            'bs_definition_trial_mode': bs_definition_trial_mode,
            'insertion_brain_regions': insertion_brain_regions,
            'insertion_conditions': insertion_conditions,
            'match_nonstim_to_inhibition_range': match_nonstim_to_inhibition_range,
            'normalize_mode': normalize_mode,
            'scalar_baseline_window': scalar_baseline_window,
            'scalar_min_fr': scalar_min_fr,
            'zero_nan_threshold': zero_nan_threshold,
            'peth_smoothing_s': smoothing,
            'post_smooth_window_ms': post_smooth_window_ms,
            'save_diagnostic_traces': save_diagnostic_traces,
            'save_raw_block_peths': save_raw_block_peths,
            'diagnostic_random_seed': diagnostic_random_seed,
            'diagnostic_trialmatch_repeats': diagnostic_trialmatch_repeats,
            'diagnostic_min_events_per_peth': diagnostic_min_events_per_peth,
            'diagnostic_crossfit_guard_trials': diagnostic_crossfit_guard_trials,
            'diagnostic_crossfit_method': 'contiguous_block_halves_nearest_control',
            'save_futureproof_sufficient_stats': save_futureproof_sufficient_stats,
            'use_batched_peths': use_batched_peths,
            'peth_cluster_batch_size': peth_cluster_batch_size,
            'peth_compute_method': 'brainbox_exact_cluster_batch',
            'save_legacy_base_pickle': save_legacy_base_pickle,
            'save_combined_alignment_pickle': save_combined_alignment_pickle,
            'onset_alignments_to_run': tuple(_alignments_for_run()),
            'alignment_time_windows': dict(alignment_time_windows),
        },
    }
    if save_raw_block_peths == 1:
        bs_results.update(raw_block_peth_traces)
    if save_diagnostic_traces == 1 or save_futureproof_sufficient_stats == 1:
        bs_results.update(diagnostic_trace_lists)
    alignment_results = {onset_alignment: bs_results}
    for _alignment, _store in additional_alignment_stores.items():
        alignment_results[_alignment] = _build_results_payload(clusters_info_DF, _store, _alignment)

    _provenance = _pipeline_provenance()
    for _results in alignment_results.values():
        _results['pipeline_provenance'] = _provenance

    if save_futureproof_sufficient_stats == 1:
        if len(qp_fr_per_trial_all) != len(clusters_info_DF):
            raise RuntimeError(
                'qp_fr_per_trial row alignment failed: '
                f'{len(qp_fr_per_trial_all)} traces for {len(clusters_info_DF)} units')
        for _results in alignment_results.values():
            _results['qp_fr_per_trial'] = qp_fr_per_trial_all
            _results['trial_metadata_by_pid'] = trial_metadata_by_pid
            _results['sufficient_stats_schema'] = {
                'version': 1,
                'qp_fr_per_trial': (
                    'float32 per-unit arrays, row-aligned to units; same QP '
                    'window used by isbiasblockselective_perm_vector'),
                'trial_metadata_by_pid': (
                    'behavior/event columns, trial/block/fold ids, GLM-HMM '
                    'engagement, and packed pseudo labels shared by PID'),
                'crossfit_raw_trace_suffixes': (
                    'block80_raw, block20_raw, all_mean; float32 PETHs'),
                'purpose': (
                    'nested BS selection, alternate sign windows, and post-hoc '
                    'condition-specific or common normalization'),
            }

    out_path = str(Path(bs_output_path).expanduser())
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    base_saved = False
    if save_legacy_base_pickle == 1:
        with open(out_path, 'wb') as f:
            pickle.dump(bs_results, f, protocol=pickle.HIGHEST_PROTOCOL)
        base_saved = True
        print(f'Saved legacy unsuffixed BS results -> {out_path}')
    for _alignment, _results in alignment_results.items():
        _alignment_path = _alignment_output_path(_alignment)
        if str(_alignment_path) == str(Path(out_path)) and base_saved:
            continue
        _alignment_path.parent.mkdir(parents=True, exist_ok=True)
        with open(_alignment_path, 'wb') as f:
            pickle.dump(_results, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f'Saved {_alignment} BS results -> {_alignment_path}')
    if len(alignment_results) > 1 and save_combined_alignment_pickle == 1:
        _combined_path = _combined_alignment_output_path()
        with open(_combined_path, 'wb') as f:
            pickle.dump(
                alignment_results, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f'Saved BS results by alignment -> {_combined_path}')
    print(f'Saved BS results: {len(clusters_info_DF)} units across '
          f'{clusters_info_DF["pid"].nunique()} insertions')
    print('Filter / plot it with BS_postprocess.py (no re-run needed).')

# ---------------------------------------------------------------------------
# Post-analysis / plotting has moved to BS_postprocess.py, which loads the
# results file above and exposes region / hemisphere / drift / quality / etc.
# as post-hoc filters. This script's job ends at saving the full results.
# ---------------------------------------------------------------------------
