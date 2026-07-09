#!/usr/bin/env python3
"""
Bilateral Optogenetics Analysis Pipeline

Analyzes behavioral data from the IBL 2-AFC task with bilateral optogenetic
stimulation delivered via implanted fiber optic cannulae. Supports multiple
brain regions (VLS, SNr, STN, ZI, motor cortex) and opsins (ChR2, stGtACR2).

This pipeline merges two previously separate analysis modes:
  - Traditional: exclude trials by RT, performance, and bias thresholds
  - GLM-HMM:    use pre-computed state labels to isolate 'engaged' trials

Toggle between modes via USE_GLMHMM in config.py.

Key analyses:
  - Psychometric curve fitting (per block, stim vs nonstim)
  - Bias shift quantification (block-induced choice bias)
  - Reaction time comparisons (stim vs nonstim)
  - Wheel movement trajectories
  - Per-session and across-session statistics

Usage:
    1. Configure paths, options, and session filters in config.py
    2. Ensure session metadata is up to date in metadata_all.py
    3. Run: python opto_analysis.py

Author: Nate Miska
        Developed with AI pair-programming assistance (Claude, Anthropic)
        for code refactoring and documentation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from one.api import ONE
from pathlib import Path

# Local imports
from config import (
    ALYX_BASE_URL,
    FIGURE_SAVE_PATH, FIGURE_PREFIX, SAVE_FIGURES, TITLE_TEXT, PLOT_FOR_PAPER,
    FLAG_FAILED_LOADS,
    USE_GLMHMM, N_STATES, STATE_TYPE, STATE_DEF,
    GLMHMM_STATES_FILE, GLMHMM_ENGAGED_PREV_FILE, GLMHMM_DISENGAGED_PREV_FILE,
    BASELINE_PERFORMANCE_THRESHOLD, STIM_PERFORMANCE_THRESHOLD,
    MIN_BIAS_THRESHOLD, RT_THRESHOLD, MIN_NUM_TRIALS, MIN_STIM_TRIALS,
    ONLY_INCLUDE_BIAS_TRIALS,
    TRIALS_PER_DATAPOINT,
    USE_TRIALS_AFTER_STIM, SUBSAMPLE_TRIALS_AFTER_STIM,
    BIAS_COMPARISON_CONTRASTS,
    WHEEL_ANALYSIS_DURATION, WHEEL_TIME_INTERVAL, WHEEL_ALIGN_TO,
    WHEEL_QP_X_LIMITS, WHEEL_QP_Y_LIMITS,
    WHEEL_GOCUE_LOW_CONTRAST_DURATION,
    WHEEL_QP_SCALAR_TIME, WHEEL_GOCUE_SCALAR_TIME,
    WHEEL_SCALAR_BAR_FIGSIZE, WHEEL_DIFFERENCE_BAR_FIGSIZE,
    ONLY_LOW_CONTRASTS, LOW_CONTRAST_THRESHOLD,
    ALL_CONTRASTS, LOW_CONTRASTS, HIGH_CONTRASTS,
    PSYCHO_FIT_KWARGS,
    SESSION_FILTERS,
    PSYCHOMETRIC_OVERLAY,
    PSYCHOMETRIC_MEAN_OF_MICE,
)
from metadata_all import sessions, find_sessions_by_advanced_criteria
from helpers import (
    load_session_data, load_laser_intervals, load_task_data,
    load_glmhmm_data, is_eid_successful, filter_trials_by_state,
    signed_contrast, get_valid_trials_range,
    compute_reaction_times, identify_stim_nonstim_trials,
    apply_trials_after_stim, subsample_stim_trials_balanced,
    subset_bunch, concat_bunches,
    organize_psychodata, compute_bias_shift,
    check_session_performance, compute_choice_bias_deviation,
    calculate_accuracy_bycontrast,
    extract_wheel_trajectory,
    plot_psychometric_curves, plot_bias_shift_comparison, plot_wheel_comparison,
    plot_bars_by_mouse, plot_per_mouse_psychometrics,
    compute_mean_psychometric_across_mice,
)


# =============================================================================
# INITIALIZATION
# =============================================================================

one = ONE(base_url='https://alyx.internationalbrainlab.org', cache_dir=Path.home() / 'Downloads' / 'ONE' / 'alyx.internationalbrainlab.org')

# Session selection
active_filters = {k: v for k, v in SESSION_FILTERS.items() if v is not None}
eids, trials_ranges, MouseIDs, stim_params = find_sessions_by_advanced_criteria(
    sessions, **active_filters
)

print(f"Found {len(eids)} sessions matching criteria")
if active_filters:
    print(f"Active filters: {active_filters}")

# Build title from filters if not manually set
if TITLE_TEXT is None:
    title_parts = []
    for k, v in active_filters.items():
        if not callable(v):
            title_parts.append(f"{v}")
    title_text = ' | '.join(title_parts) if title_parts else 'All Sessions'
else:
    title_text = TITLE_TEXT

# Load GLM-HMM data if enabled
state_probability = None
if USE_GLMHMM:
    print(f"GLM-HMM enabled: n_states={N_STATES}, state_type={STATE_TYPE}, state_def={STATE_DEF}")
    state_probability, _, _ = load_glmhmm_data(
        GLMHMM_STATES_FILE,
        GLMHMM_ENGAGED_PREV_FILE,
        GLMHMM_DISENGAGED_PREV_FILE,
    )


# =============================================================================
# PER-SESSION DATA STRUCTURES
# =============================================================================

n_sessions = len(eids)

# Per-session bias shift tracking
bias_shift_sum_all_nonstim = np.full(n_sessions, np.nan)
bias_shift_sum_all_nonstim_LC = np.full(n_sessions, np.nan)
bias_shift_sum_all_nonstim_LEFT = np.full(n_sessions, np.nan)
bias_shift_sum_all_nonstim_RIGHT = np.full(n_sessions, np.nan)
bias_shift0_all_nonstim = np.full(n_sessions, np.nan)

bias_shift_sum_all_stim = np.full(n_sessions, np.nan)
bias_shift_sum_all_stim_LC = np.full(n_sessions, np.nan)
bias_shift_sum_all_stim_LEFT = np.full(n_sessions, np.nan)
bias_shift_sum_all_stim_RIGHT = np.full(n_sessions, np.nan)
bias_shift0_all_stim = np.full(n_sessions, np.nan)

# Per-session accuracy tracking (for GLM-HMM mode diagnostics)
accuracy_lc_stim = np.full(n_sessions, np.nan)
accuracy_lc_nonstim = np.full(n_sessions, np.nan)

# Wheel data
Rblock_wheel_movements_stim = []
Lblock_wheel_movements_stim = []
Rblock_wheel_movements_nonstim = []
Lblock_wheel_movements_nonstim = []

# Additional go-cue-aligned, low-contrast wheel data.
Rblock_wheel_movements_stim_gocue_low = []
Lblock_wheel_movements_stim_gocue_low = []
Rblock_wheel_movements_nonstim_gocue_low = []
Lblock_wheel_movements_nonstim_gocue_low = []
wheel_scalar_rows = []


def _stack_wheel_trajectory(existing, trajectory):
    trajectory = np.asarray(trajectory, dtype=float)
    if trajectory.ndim != 1:
        return existing
    if not isinstance(existing, np.ndarray) or existing.size == 0:
        return trajectory[np.newaxis, :]
    return np.vstack([existing, trajectory])


def _store_wheel_trajectory(trajectory, is_stim, is_nonstim, is_rblock, is_lblock,
                            r_stim, l_stim, r_nonstim, l_nonstim):
    if is_nonstim:
        if is_rblock:
            r_nonstim = _stack_wheel_trajectory(r_nonstim, trajectory)
        elif is_lblock:
            l_nonstim = _stack_wheel_trajectory(l_nonstim, trajectory)
    elif is_stim:
        if is_rblock:
            r_stim = _stack_wheel_trajectory(r_stim, trajectory)
        elif is_lblock:
            l_stim = _stack_wheel_trajectory(l_stim, trajectory)
    return r_stim, l_stim, r_nonstim, l_nonstim


def _has_wheel_data(*arrays):
    return any(isinstance(arr, np.ndarray) and arr.ndim == 2 and arr.shape[0] > 0
               for arr in arrays)


def _wheel_scalar_at_or_before(trajectory, sample_time_s, interval):
    trajectory = np.asarray(trajectory, dtype=float)
    if trajectory.ndim != 1 or trajectory.size == 0:
        return np.nan
    max_idx = min(int(np.floor(float(sample_time_s) / float(interval))), trajectory.size - 1)
    if max_idx < 0:
        return np.nan
    vals = trajectory[:max_idx + 1]
    finite = np.where(np.isfinite(vals))[0]
    if finite.size == 0:
        return np.nan
    return float(vals[finite[-1]])


def _append_wheel_scalar_row(rows, *, mouse_id, eid, trial_num, alignment,
                             sample_time_s, trajectory, is_stim, is_nonstim,
                             is_rblock, is_lblock, signed_contrast_value):
    if not (is_stim or is_nonstim):
        return
    if not (is_rblock or is_lblock):
        return
    rows.append({
        'MouseID': mouse_id,
        'EID': eid,
        'trial_num': int(trial_num),
        'alignment': alignment,
        'sample_time_s': float(sample_time_s),
        'condition': 'stim' if is_stim else 'control',
        'block': 'R' if is_rblock else 'L',
        'signed_contrast': float(signed_contrast_value),
        'wheel_value': _wheel_scalar_at_or_before(
            trajectory, sample_time_s, WHEEL_TIME_INTERVAL),
    })


def _wheel_mouse_summary(wheel_scalar_rows):
    df_trial = pd.DataFrame(wheel_scalar_rows)
    if len(df_trial) == 0:
        return df_trial, pd.DataFrame()
    df_mouse = (
        df_trial
        .groupby(['alignment', 'sample_time_s', 'MouseID', 'block', 'condition'], as_index=False)
        .agg(
            wheel_value=('wheel_value', 'mean'),
            n_trials=('wheel_value', lambda x: int(np.sum(np.isfinite(x)))),
        )
    )
    return df_trial, df_mouse


def _plot_wheel_scalar_bars(df_mouse, alignment, *, title='', save_path=None,
                            figsize=(4, 4)):
    sub = df_mouse[df_mouse['alignment'] == alignment].copy()
    if len(sub) == 0:
        return []

    stats_rows = []
    fig, ax = plt.subplots(figsize=figsize)
    xmap = {('L', 'control'): 0, ('L', 'stim'): 1,
            ('R', 'control'): 3, ('R', 'stim'): 4}
    colors = {'control': 'black', 'stim': 'red'}
    labels = {'control': 'Control', 'stim': 'Stim'}

    for block in ['L', 'R']:
        pivot = (
            sub[sub['block'] == block]
            .pivot(index='MouseID', columns='condition', values='wheel_value')
        )
        if {'control', 'stim'}.issubset(pivot.columns):
            paired = pivot[['control', 'stim']].dropna()
        else:
            paired = pd.DataFrame(columns=['control', 'stim'])
        stats_rows.append(_paired_t_summary(
            'mouse', f'wheel_{alignment}_{block}_block',
            paired['control'].to_numpy(float) if len(paired) else [],
            paired['stim'].to_numpy(float) if len(paired) else [],
            control_label='Control', stim_label='Stim',
        ))

        for _, row in paired.iterrows():
            ax.plot([xmap[(block, 'control')], xmap[(block, 'stim')]],
                    [row['control'], row['stim']],
                    color='0.65', alpha=0.7, linewidth=1, zorder=1)

    for block in ['L', 'R']:
        for condition in ['control', 'stim']:
            vals = sub[(sub['block'] == block) & (sub['condition'] == condition)]['wheel_value'].to_numpy(float)
            vals = vals[np.isfinite(vals)]
            x = xmap[(block, condition)]
            mean = np.nanmean(vals) if vals.size else np.nan
            sem = stats.sem(vals) if vals.size > 1 else np.nan
            ax.bar(x, mean, color=colors[condition], alpha=0.7, width=0.75,
                   label=labels[condition] if block == 'L' else None, zorder=2)
            if np.isfinite(sem):
                ax.errorbar(x, mean, yerr=sem, color='black', linewidth=1.5,
                            capsize=3, zorder=3)
            if vals.size:
                jitter = np.linspace(-0.08, 0.08, vals.size) if vals.size > 1 else np.array([0.0])
                ax.scatter(np.full(vals.size, x) + jitter, vals, color='0.2',
                           s=16, alpha=0.75, zorder=4)

    ax.axhline(0, color='0.5', linewidth=0.8)
    ax.set_xticks([0.5, 3.5])
    ax.set_xticklabels(['L block', 'R block'])
    ax.set_ylabel('Wheel movement')
    ax.set_title(title or alignment)
    # ax.legend(frameon=False)
    sns.despine()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()
    return stats_rows


def _wheel_difference_summary(df_mouse):
    if len(df_mouse) == 0:
        return pd.DataFrame()
    pivot = (
        df_mouse
        .pivot_table(
            index=['alignment', 'sample_time_s', 'MouseID', 'condition'],
            columns='block',
            values='wheel_value',
            aggfunc='mean',
        )
        .reset_index()
    )
    if 'L' not in pivot.columns:
        pivot['L'] = np.nan
    if 'R' not in pivot.columns:
        pivot['R'] = np.nan
    pivot['R_minus_L'] = pivot['R'] - pivot['L']
    return pivot[['alignment', 'sample_time_s', 'MouseID', 'condition', 'L', 'R', 'R_minus_L']]


def _plot_wheel_difference_bars(df_diff, alignment, *, title='', save_path=None,
                                figsize=(2, 4)):
    sub = df_diff[df_diff['alignment'] == alignment].copy()
    if len(sub) == 0:
        return []
    pivot = sub.pivot(index='MouseID', columns='condition', values='R_minus_L')
    if {'control', 'stim'}.issubset(pivot.columns):
        paired = pivot[['control', 'stim']].dropna()
    else:
        paired = pd.DataFrame(columns=['control', 'stim'])

    stats_row = _paired_t_summary(
        'mouse', f'wheel_{alignment}_R_minus_L',
        paired['control'].to_numpy(float) if len(paired) else [],
        paired['stim'].to_numpy(float) if len(paired) else [],
        control_label='Control', stim_label='Stim',
    )

    fig, ax = plt.subplots(figsize=figsize)
    colors = {'control': 'black', 'stim': 'red'}
    for _, row in paired.iterrows():
        ax.plot([0, 1], [row['control'], row['stim']],
                color='0.65', alpha=0.75, linewidth=1, zorder=1)

    for x, condition in enumerate(['control', 'stim']):
        vals = paired[condition].to_numpy(float) if len(paired) else np.asarray([])
        vals = vals[np.isfinite(vals)]
        mean = np.nanmean(vals) if vals.size else np.nan
        sem = stats.sem(vals) if vals.size > 1 else np.nan
        ax.bar(x, mean, color=colors[condition], alpha=0.7, width=0.7, zorder=2)
        if np.isfinite(sem):
            ax.errorbar(x, mean, yerr=sem, color='black', linewidth=1.5,
                        capsize=3, zorder=3)
        if vals.size:
            jitter = np.linspace(-0.06, 0.06, vals.size) if vals.size > 1 else np.array([0.0])
            ax.scatter(np.full(vals.size, x) + jitter, vals, color='0.2',
                       s=18, alpha=0.75, zorder=4)

    ax.axhline(0, color='0.5', linewidth=0.8)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Control', 'Stim'])
    ax.set_ylabel('R block - L block wheel')
    ax.set_title(title or alignment)
    sns.despine()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()
    return [stats_row]


def _paired_t_summary(level, metric, control_values, stim_values,
                      control_label='Control', stim_label='Stim'):
    control_values = np.asarray(control_values, dtype=float)
    stim_values = np.asarray(stim_values, dtype=float)
    valid = np.isfinite(control_values) & np.isfinite(stim_values)
    ctrl = control_values[valid]
    stim = stim_values[valid]
    row = {
        'analysis_level': level,
        'metric': metric,
        'test': 'paired_t_test',
        'control_label': control_label,
        'stim_label': stim_label,
        'n': int(len(ctrl)),
        'control_mean': np.nanmean(ctrl) if len(ctrl) else np.nan,
        'stim_mean': np.nanmean(stim) if len(stim) else np.nan,
        'control_sd': np.nanstd(ctrl, ddof=1) if len(ctrl) > 1 else np.nan,
        'stim_sd': np.nanstd(stim, ddof=1) if len(stim) > 1 else np.nan,
        'stim_minus_control_mean': np.nanmean(stim - ctrl) if len(ctrl) else np.nan,
        'control_minus_stim_mean': np.nanmean(ctrl - stim) if len(ctrl) else np.nan,
        'statistic': np.nan,
        'p_value': np.nan,
    }
    if len(ctrl) > 1:
        t_stat, p_val = stats.ttest_rel(ctrl, stim)
        row['statistic'] = float(t_stat)
        row['p_value'] = float(p_val)
    return row


def _mannwhitney_summary(level, metric, control_values, stim_values,
                         control_label='Control', stim_label='Stim'):
    control_values = np.asarray(control_values, dtype=float)
    stim_values = np.asarray(stim_values, dtype=float)
    ctrl = control_values[np.isfinite(control_values)]
    stim = stim_values[np.isfinite(stim_values)]
    row = {
        'analysis_level': level,
        'metric': metric,
        'test': 'mannwhitneyu_two_sided',
        'control_label': control_label,
        'stim_label': stim_label,
        'n_control': int(len(ctrl)),
        'n_stim': int(len(stim)),
        'control_mean': np.nanmean(ctrl) if len(ctrl) else np.nan,
        'stim_mean': np.nanmean(stim) if len(stim) else np.nan,
        'control_median': np.nanmedian(ctrl) if len(ctrl) else np.nan,
        'stim_median': np.nanmedian(stim) if len(stim) else np.nan,
        'stim_minus_control_mean': np.nanmean(stim) - np.nanmean(ctrl) if len(ctrl) and len(stim) else np.nan,
        'statistic': np.nan,
        'p_value': np.nan,
    }
    if len(ctrl) > 0 and len(stim) > 0:
        u_stat, p_val = stats.mannwhitneyu(stim, ctrl, alternative='two-sided')
        row['statistic'] = float(u_stat)
        row['p_value'] = float(p_val)
    return row


# RT data
rt_stimtrials_all = np.array([])
qp_stimtrials_all = np.array([])
rt_nonstimtrials_all = np.array([])
qp_nonstimtrials_all = np.array([])
rt_stimtrials_persubject = np.array([])
rt_nonstimtrials_persubject = np.array([])

# Concatenated trial objects
stim_trials_master = None
nonstim_trials_master = None

# Session tracking
num_analyzed_sessions = 0
num_unique_mice = 0
previous_mouse_id = None
deviation_list = []
MouseIDs_analyzed = []
EIDs_analyzed = []
nonstim_trials_per_session = []
stim_trials_per_session = []
session_summary_rows = []
statistics_summary_rows = []

# Per-mouse trial aggregation (for per-mouse bar plots)
mouse_trials_container = {}


# =============================================================================
# MAIN SESSION LOOP
# =============================================================================

for j in range(n_sessions):
    eid = eids[j]
    current_mouse_id = MouseIDs[j]
    print(f"\n--- Session {j+1}/{n_sessions}: {eid} ({current_mouse_id}) ---")

    # -----------------------------------------------------------------
    # 1. LOAD DATA
    # -----------------------------------------------------------------
    trials, wheel = load_session_data(one, eid, flag_failed_loads=FLAG_FAILED_LOADS)
    if trials is None:
        continue

    # Load laser intervals (try first) or fall back to task data
    laser_intervals = load_laser_intervals(one, eid)
    task_data = None
    if laser_intervals is None:
        task_data, ses_path = load_task_data(one, eid)
        if task_data is None:
            print(f'  No laser intervals or task data found for {eid}, skipping...')
            continue

    # GLM-HMM: check that states exist for this session
    if USE_GLMHMM:
        if not is_eid_successful(state_probability, current_mouse_id, eid):
            print(f'  No GLM-HMM states found, skipping...')
            continue

    # -----------------------------------------------------------------
    # 2. DETERMINE VALID TRIALS RANGE
    # -----------------------------------------------------------------
    trials_range = get_valid_trials_range(trials_ranges[j], trials)

    if len(trials_range) < MIN_NUM_TRIALS:
        print(f'  Not enough trials ({len(trials_range)} < {MIN_NUM_TRIALS}), skipping...')
        continue

    # -----------------------------------------------------------------
    # 3. COMPUTE REACTION TIMES
    # -----------------------------------------------------------------
    reaction_times, quiescent_period_times = compute_reaction_times(trials, task_data)

    # -----------------------------------------------------------------
    # 4. IDENTIFY STIM vs NONSTIM TRIALS
    # -----------------------------------------------------------------
    stim_trials_numbers, nonstim_trials_numbers = identify_stim_nonstim_trials(
        trials, trials_range, stim_params[j],
        laser_intervals=laser_intervals, task_data=task_data,
        rt_threshold=RT_THRESHOLD,
    )

    # Combine into all valid trials
    trials_numbers = np.union1d(stim_trials_numbers, nonstim_trials_numbers)

    # Optionally restrict to bias-block trials (after first block switch, trial > 89)
    if ONLY_INCLUDE_BIAS_TRIALS:
        stim_trials_numbers = stim_trials_numbers[stim_trials_numbers > 89]
        nonstim_trials_numbers = nonstim_trials_numbers[nonstim_trials_numbers > 89]
        trials_numbers = trials_numbers[trials_numbers > 89]

    # Handle trials-after-stim mode
    if USE_TRIALS_AFTER_STIM:
        stim_trials_numbers = apply_trials_after_stim(stim_trials_numbers, nonstim_trials_numbers)

    # Remove stim trials from nonstim set (mutual exclusivity)
    nonstim_trials_numbers = np.setdiff1d(nonstim_trials_numbers, stim_trials_numbers)

    # Optional subsampling for balanced prev-choice × block
    if SUBSAMPLE_TRIALS_AFTER_STIM:
        stim_trials_numbers = subsample_stim_trials_balanced(stim_trials_numbers, trials)

    # -----------------------------------------------------------------
    # 5. GLM-HMM STATE FILTERING (if enabled)
    # -----------------------------------------------------------------
    if USE_GLMHMM:
        try:
            trials_numbers, stim_trials_numbers, nonstim_trials_numbers = filter_trials_by_state(
                trials_numbers, stim_trials_numbers, nonstim_trials_numbers,
                state_probability, current_mouse_id, eid,
                N_STATES, STATE_TYPE, STATE_DEF,
            )
        except Exception as e:
            print(f'  GLM-HMM filtering failed: {e}, skipping...')
            continue

        if len(stim_trials_numbers) == 0:
            print('  No trials remain after GLM-HMM filtering, skipping...')
            continue

    # Check minimum stim trials
    if len(stim_trials_numbers) < MIN_STIM_TRIALS:
        print(f'  Fewer than {MIN_STIM_TRIALS} stim trials, skipping...')
        continue

    # -----------------------------------------------------------------
    # 6. COMPUTE CHOICE BIAS DEVIATION (stim vs nonstim directional bias)
    # -----------------------------------------------------------------
    if len(nonstim_trials_numbers) > 0 and len(stim_trials_numbers) > 0:
        dev = compute_choice_bias_deviation(trials, stim_trials_numbers, nonstim_trials_numbers)
        deviation_list.append(dev)

    # -----------------------------------------------------------------
    # 7. CREATE TRIAL SUBSETS AND COMPUTE BIAS SHIFTS
    # -----------------------------------------------------------------
    stim_temp = subset_bunch(trials, stim_trials_numbers)
    nonstim_temp = subset_bunch(trials, nonstim_trials_numbers)
    stim_contrast = signed_contrast(stim_temp)
    nonstim_contrast = signed_contrast(nonstim_temp)

    # --- Nonstim bias shift (baseline) ---
    nonstim_psycho = organize_psychodata(nonstim_temp, nonstim_contrast)
    ns_total, ns_lc, ns_hc, ns_0, ns_per = compute_bias_shift(
        nonstim_psycho, PSYCHO_FIT_KWARGS
    )

    # Session quality gate: minimum baseline bias
    if np.isnan(ns_total) or ns_total < MIN_BIAS_THRESHOLD:
        print(f'  Baseline bias shift ({ns_total:.2f}) below threshold ({MIN_BIAS_THRESHOLD}), skipping...')
        continue

    bias_shift_sum_all_nonstim[j] = ns_total
    bias_shift_sum_all_nonstim_LC[j] = ns_lc
    bias_shift0_all_nonstim[j] = ns_0
    bias_shift_sum_all_nonstim_LEFT[j] = sum(ns_per.get(c, 0.) for c in [-100., -25., -12.5, -6.25, 0.])
    bias_shift_sum_all_nonstim_RIGHT[j] = sum(ns_per.get(c, 0.) for c in [0., 100., 25., 12.5, 6.25])

    # --- Nonstim performance check ---
    passes_bl, acc_l, acc_r = check_session_performance(
        nonstim_temp, nonstim_contrast, BASELINE_PERFORMANCE_THRESHOLD
    )
    if not passes_bl:
        print(f'  Nonstim accuracy below threshold ({acc_l:.2f}L, {acc_r:.2f}R), skipping...')
        continue

    # --- Stim performance check ---
    passes_stim, stim_acc_l, stim_acc_r = check_session_performance(
        stim_temp, stim_contrast, STIM_PERFORMANCE_THRESHOLD
    )
    if not passes_stim:
        print(f'  Stim accuracy below threshold ({stim_acc_l:.2f}L, {stim_acc_r:.2f}R), skipping...')
        continue

    # --- Stim bias shift ---
    stim_psycho = organize_psychodata(stim_temp, stim_contrast)
    st_total, st_lc, st_hc, st_0, st_per = compute_bias_shift(
        stim_psycho, PSYCHO_FIT_KWARGS
    )

    bias_shift_sum_all_stim[j] = st_total
    bias_shift_sum_all_stim_LC[j] = st_lc
    bias_shift0_all_stim[j] = st_0
    bias_shift_sum_all_stim_LEFT[j] = sum(st_per.get(c, 0.) for c in [-100., -25., -12.5, -6.25, 0.])
    bias_shift_sum_all_stim_RIGHT[j] = sum(st_per.get(c, 0.) for c in [0., 100., 25., 12.5, 6.25])

    print(f'  Nonstim bias: {ns_total:.2f} | Stim bias: {st_total:.2f}')

    # -----------------------------------------------------------------
    # 8. WHEEL ANALYSIS
    # -----------------------------------------------------------------
    if not USE_TRIALS_AFTER_STIM:
        whlpos, whlt = wheel.position, wheel.timestamps
        all_contrast = signed_contrast(trials)

        for trial_num in trials_numbers:
            trial_num = int(trial_num)
            is_low_contrast = abs(all_contrast[trial_num]) <= LOW_CONTRAST_THRESHOLD

            # Optional low contrast filter
            if ONLY_LOW_CONTRASTS and not is_low_contrast:
                continue

            is_stim = np.isin(trial_num, stim_trials_numbers)
            is_nonstim = np.isin(trial_num, nonstim_trials_numbers)
            is_rblock = trials.probabilityLeft[trial_num] == 0.2
            is_lblock = trials.probabilityLeft[trial_num] == 0.8

            trajectory = extract_wheel_trajectory(
                trials, trial_num, whlpos, whlt,
                align_to=WHEEL_ALIGN_TO,
                duration=WHEEL_ANALYSIS_DURATION,
                interval=WHEEL_TIME_INTERVAL,
            )
            (
                Rblock_wheel_movements_stim,
                Lblock_wheel_movements_stim,
                Rblock_wheel_movements_nonstim,
                Lblock_wheel_movements_nonstim,
            ) = _store_wheel_trajectory(
                trajectory, is_stim, is_nonstim, is_rblock, is_lblock,
                Rblock_wheel_movements_stim,
                Lblock_wheel_movements_stim,
                Rblock_wheel_movements_nonstim,
                Lblock_wheel_movements_nonstim,
            )
            _append_wheel_scalar_row(
                wheel_scalar_rows,
                mouse_id=current_mouse_id, eid=eid, trial_num=trial_num,
                alignment='QP',
                sample_time_s=WHEEL_QP_SCALAR_TIME,
                trajectory=trajectory,
                is_stim=is_stim, is_nonstim=is_nonstim,
                is_rblock=is_rblock, is_lblock=is_lblock,
                signed_contrast_value=all_contrast[trial_num],
            )

            if is_low_contrast:
                trajectory_gocue_low = extract_wheel_trajectory(
                    trials, trial_num, whlpos, whlt,
                    align_to='goCue',
                    duration=WHEEL_GOCUE_LOW_CONTRAST_DURATION,
                    interval=WHEEL_TIME_INTERVAL,
                )
                (
                    Rblock_wheel_movements_stim_gocue_low,
                    Lblock_wheel_movements_stim_gocue_low,
                    Rblock_wheel_movements_nonstim_gocue_low,
                    Lblock_wheel_movements_nonstim_gocue_low,
                ) = _store_wheel_trajectory(
                    trajectory_gocue_low, is_stim, is_nonstim, is_rblock, is_lblock,
                    Rblock_wheel_movements_stim_gocue_low,
                    Lblock_wheel_movements_stim_gocue_low,
                    Rblock_wheel_movements_nonstim_gocue_low,
                    Lblock_wheel_movements_nonstim_gocue_low,
                )
                _append_wheel_scalar_row(
                    wheel_scalar_rows,
                    mouse_id=current_mouse_id, eid=eid, trial_num=trial_num,
                    alignment='GoCue_low_contrast',
                    sample_time_s=WHEEL_GOCUE_SCALAR_TIME,
                    trajectory=trajectory_gocue_low,
                    is_stim=is_stim, is_nonstim=is_nonstim,
                    is_rblock=is_rblock, is_lblock=is_lblock,
                    signed_contrast_value=all_contrast[trial_num],
                )

    # -----------------------------------------------------------------
    # 9. CONCATENATE ACROSS SESSIONS
    # -----------------------------------------------------------------
    # Add prev_choice attribute before concatenating
    stim_session = subset_bunch(trials, stim_trials_numbers)
    stim_session.prev_choice = trials.choice[stim_trials_numbers - 1]
    stim_session.reaction_times = (
        stim_session.feedback_times - stim_session.goCueTrigger_times
        if hasattr(stim_session, 'goCueTrigger_times')
        else stim_session.feedback_times - stim_session.goCue_times
    )

    nonstim_session = subset_bunch(trials, nonstim_trials_numbers)
    nonstim_session.prev_choice = trials.choice[nonstim_trials_numbers - 1]
    nonstim_session.reaction_times = (
        nonstim_session.feedback_times - nonstim_session.goCueTrigger_times
        if hasattr(nonstim_session, 'goCueTrigger_times')
        else nonstim_session.feedback_times - nonstim_session.goCue_times
    )

    stim_trials_master = concat_bunches(stim_trials_master, stim_session)
    nonstim_trials_master = concat_bunches(nonstim_trials_master, nonstim_session)

    # Per-mouse accumulation (for bar plots)
    if current_mouse_id not in mouse_trials_container:
        mouse_trials_container[current_mouse_id] = {
            'stim': stim_session,
            'nonstim': nonstim_session,
        }
    else:
        mouse_trials_container[current_mouse_id]['stim'] = concat_bunches(
            mouse_trials_container[current_mouse_id]['stim'], stim_session
        )
        mouse_trials_container[current_mouse_id]['nonstim'] = concat_bunches(
            mouse_trials_container[current_mouse_id]['nonstim'], nonstim_session
        )

    # RT data
    rt_stim = reaction_times[stim_trials_numbers]
    rt_nonstim = reaction_times[nonstim_trials_numbers]
    qp_stim = quiescent_period_times[stim_trials_numbers]
    qp_nonstim = quiescent_period_times[nonstim_trials_numbers]

    rt_stimtrials_all = np.append(rt_stimtrials_all, rt_stim)
    rt_nonstimtrials_all = np.append(rt_nonstimtrials_all, rt_nonstim)
    qp_stimtrials_all = np.append(qp_stimtrials_all, qp_stim)
    qp_nonstimtrials_all = np.append(qp_nonstimtrials_all, qp_nonstim)
    rt_stimtrials_persubject = np.append(rt_stimtrials_persubject, np.nanmean(rt_stim))
    rt_nonstimtrials_persubject = np.append(rt_nonstimtrials_persubject, np.nanmean(rt_nonstim))

    # Session bookkeeping
    MouseIDs_analyzed.append(current_mouse_id)
    EIDs_analyzed.append(eid)
    nonstim_trials_per_session.append(len(nonstim_trials_numbers))
    stim_trials_per_session.append(len(stim_trials_numbers))
    session_summary_rows.append({
        'MouseID': current_mouse_id,
        'EID': eid,
        'n_stim_trials': int(len(stim_trials_numbers)),
        'n_nonstim_trials': int(len(nonstim_trials_numbers)),
        'Bias_Values_Nonstim': ns_total,
        'Bias_Values_Stim': st_total,
        'Bias_Values_Nonstim_LC': ns_lc,
        'Bias_Values_Stim_LC': st_lc,
        'Bias_Values_Nonstim_0': ns_0,
        'Bias_Values_Stim_0': st_0,
        'Accuracy_LowContrast_Control_LeftBlock': acc_l,
        'Accuracy_LowContrast_Control_RightBlock': acc_r,
        'Accuracy_LowContrast_Stim_LeftBlock': stim_acc_l,
        'Accuracy_LowContrast_Stim_RightBlock': stim_acc_r,
        'RT_Control_Mean': np.nanmean(rt_nonstim),
        'RT_Stim_Mean': np.nanmean(rt_stim),
        'QP_Control_Mean': np.nanmean(qp_nonstim),
        'QP_Stim_Mean': np.nanmean(qp_stim),
    })
    num_analyzed_sessions += 1
    if current_mouse_id != previous_mouse_id:
        num_unique_mice += 1
        previous_mouse_id = current_mouse_id

    print(f'  Session accepted: {len(stim_trials_numbers)} stim, {len(nonstim_trials_numbers)} nonstim trials')


# =============================================================================
# POST-LOOP SUMMARY
# =============================================================================

print(f'\n{"="*60}')
print(f'ANALYSIS COMPLETE')
print(f'  Sessions analyzed: {num_analyzed_sessions} / {n_sessions}')
print(f'  Unique mice: {num_unique_mice}')
if stim_trials_master is not None:
    print(f'  Total stim trials: {sum(stim_trials_per_session)}')
    print(f'  Total nonstim trials: {sum(nonstim_trials_per_session)}')
if USE_GLMHMM:
    print(f'  GLM-HMM mode: {STATE_TYPE} (n_states={N_STATES}, def={STATE_DEF})')
print(f'{"="*60}')

if num_analyzed_sessions == 0:
    print('\nNo sessions met criteria. Check filters and thresholds in config.py.')
    raise SystemExit


# =============================================================================
# ACROSS-SESSION STATISTICS
# =============================================================================

# Paired bias shift comparison
valid_mask = ~np.isnan(bias_shift_sum_all_nonstim) & ~np.isnan(bias_shift_sum_all_stim)

if np.sum(valid_mask) > 1:
    statistics_summary_rows.append(_paired_t_summary(
        'session', 'bias_shift_all_contrasts',
        bias_shift_sum_all_nonstim[valid_mask],
        bias_shift_sum_all_stim[valid_mask],
        control_label='Nonstim', stim_label='Stim',
    ))
    t_stat, p_val = stats.ttest_rel(
        bias_shift_sum_all_nonstim[valid_mask],
        bias_shift_sum_all_stim[valid_mask],
    )
    print(f'\nBias shift (all contrasts):')
    print(f'  Nonstim: {np.nanmean(bias_shift_sum_all_nonstim[valid_mask]):.3f} +/- {np.nanstd(bias_shift_sum_all_nonstim[valid_mask]):.3f}')
    print(f'  Stim:    {np.nanmean(bias_shift_sum_all_stim[valid_mask]):.3f} +/- {np.nanstd(bias_shift_sum_all_stim[valid_mask]):.3f}')
    print(f'  Paired t-test: t={t_stat:.3f}, p={p_val:.4f}, n={np.sum(valid_mask)}')

# Low-contrast comparison
valid_lc = ~np.isnan(bias_shift_sum_all_nonstim_LC) & ~np.isnan(bias_shift_sum_all_stim_LC)
if np.sum(valid_lc) > 1:
    statistics_summary_rows.append(_paired_t_summary(
        'session', 'bias_shift_low_contrasts',
        bias_shift_sum_all_nonstim_LC[valid_lc],
        bias_shift_sum_all_stim_LC[valid_lc],
        control_label='Nonstim', stim_label='Stim',
    ))
    t_lc, p_lc = stats.ttest_rel(
        bias_shift_sum_all_nonstim_LC[valid_lc],
        bias_shift_sum_all_stim_LC[valid_lc],
    )
    print(f'\nBias shift (low contrasts):')
    print(f'  Nonstim: {np.nanmean(bias_shift_sum_all_nonstim_LC[valid_lc]):.3f}')
    print(f'  Stim:    {np.nanmean(bias_shift_sum_all_stim_LC[valid_lc]):.3f}')
    print(f'  Paired t-test: t={t_lc:.3f}, p={p_lc:.4f}')

# RT comparison
rt_stim_clean = rt_stimtrials_all[~np.isnan(rt_stimtrials_all)]
rt_nonstim_clean = rt_nonstimtrials_all[~np.isnan(rt_nonstimtrials_all)]
if len(rt_stim_clean) > 0 and len(rt_nonstim_clean) > 0:
    statistics_summary_rows.append(_mannwhitney_summary(
        'trial', 'reaction_time',
        rt_nonstim_clean, rt_stim_clean,
        control_label='Nonstim', stim_label='Stim',
    ))
    u_stat, p_rt = stats.mannwhitneyu(rt_stim_clean, rt_nonstim_clean, alternative='two-sided')
    print(f'\nReaction times:')
    print(f'  Stim:    {np.nanmean(rt_stim_clean):.3f}s (median {np.nanmedian(rt_stim_clean):.3f}s)')
    print(f'  Nonstim: {np.nanmean(rt_nonstim_clean):.3f}s (median {np.nanmedian(rt_nonstim_clean):.3f}s)')
    print(f'  Mann-Whitney U: p={p_rt:.4f}')


# =============================================================================
# SESSION LOG (for reproducibility)
# =============================================================================

print(f'\nAnalyzed sessions:')
for i, (mid, eid_used) in enumerate(zip(MouseIDs_analyzed, EIDs_analyzed)):
    print(f'  {i+1}. {mid} | {eid_used} | '
          f'{stim_trials_per_session[i]} stim, {nonstim_trials_per_session[i]} nonstim')


# =============================================================================
# PLOTTING
# =============================================================================

# =============================================================================
# PER-MOUSE ANALYSIS (for bar + line plots)
# =============================================================================

print('\n--- Per-Mouse Aggregation ---')

mouse_bias_nonstim_lc = []
mouse_bias_stim_lc = []
mouse_bias_nonstim_all = []
mouse_bias_stim_all = []
mouse_acc_hc_nonstim = []
mouse_acc_hc_stim = []
mouse_acc_0_nonstim = []
mouse_acc_0_stim = []
mouse_rt_nonstim = []
mouse_rt_stim = []
mouse_qp_nonstim = []
mouse_qp_stim = []
mouse_id_list = []

HC_values = np.array(HIGH_CONTRASTS)
ZERO_values = np.array([0.])

for mouse_id, data in mouse_trials_container.items():
    stim_mouse = data['stim']
    nonstim_mouse = data['nonstim']

    # Check minimum trials
    if (len(stim_mouse.contrastLeft) < 20 or len(nonstim_mouse.contrastLeft) < 20):
        print(f'  Skipping {mouse_id}: not enough trials')
        continue

    print(f'  {mouse_id}: {len(stim_mouse.contrastLeft)} stim, {len(nonstim_mouse.contrastLeft)} nonstim')

    stim_c = signed_contrast(stim_mouse)
    nonstim_c = signed_contrast(nonstim_mouse)

    # Bias shift per mouse (fit psychometric per block)
    stim_psycho = organize_psychodata(stim_mouse, stim_c)
    nonstim_psycho = organize_psychodata(nonstim_mouse, nonstim_c)

    ns_total, ns_lc, ns_hc, ns_0, _ = compute_bias_shift(nonstim_psycho, PSYCHO_FIT_KWARGS)
    st_total, st_lc, st_hc, st_0, _ = compute_bias_shift(stim_psycho, PSYCHO_FIT_KWARGS)

    mouse_bias_nonstim_lc.append(ns_lc)
    mouse_bias_stim_lc.append(st_lc)
    mouse_bias_nonstim_all.append(ns_total)
    mouse_bias_stim_all.append(st_total)

    # Accuracy
    mouse_acc_hc_nonstim.append(calculate_accuracy_bycontrast(nonstim_mouse, np.isin(nonstim_c, HC_values)))
    mouse_acc_hc_stim.append(calculate_accuracy_bycontrast(stim_mouse, np.isin(stim_c, HC_values)))
    mouse_acc_0_nonstim.append(calculate_accuracy_bycontrast(nonstim_mouse, np.isin(nonstim_c, ZERO_values)))
    mouse_acc_0_stim.append(calculate_accuracy_bycontrast(stim_mouse, np.isin(stim_c, ZERO_values)))

    # RT and QP (mean per mouse across all their concatenated trials)
    mouse_rt_nonstim.append(np.nanmean(nonstim_mouse.reaction_times))
    mouse_rt_stim.append(np.nanmean(stim_mouse.reaction_times))

    try:
        qp_ctrl = nonstim_mouse.goCue_times - nonstim_mouse.intervals[:, 0]
        qp_stim_vals = stim_mouse.goCue_times - stim_mouse.intervals[:, 0]
    except (AttributeError, IndexError):
        try:
            qp_ctrl = nonstim_mouse.goCueTrigger_times - nonstim_mouse.intervals[:, 0]
            qp_stim_vals = stim_mouse.goCueTrigger_times - stim_mouse.intervals[:, 0]
        except Exception:
            qp_ctrl = np.array([np.nan])
            qp_stim_vals = np.array([np.nan])

    mouse_qp_nonstim.append(np.nanmean(qp_ctrl))
    mouse_qp_stim.append(np.nanmean(qp_stim_vals))

    mouse_id_list.append(mouse_id)

# Build per-mouse DataFrame
df_mouse = pd.DataFrame({
    'MouseID': mouse_id_list,
    'Bias_Values_Nonstim_LC': mouse_bias_nonstim_lc,
    'Bias_Values_Stim_LC': mouse_bias_stim_lc,
    'Bias_Values_Nonstim': mouse_bias_nonstim_all,
    'Bias_Values_Stim': mouse_bias_stim_all,
    'Accuracy_HC_Control': mouse_acc_hc_nonstim,
    'Accuracy_HC_Stim': mouse_acc_hc_stim,
    'Accuracy_0_Control': mouse_acc_0_nonstim,
    'Accuracy_0_Stim': mouse_acc_0_stim,
    'RT_Control': mouse_rt_nonstim,
    'RT_Stim': mouse_rt_stim,
    'QP_Control': mouse_qp_nonstim,
    'QP_Stim': mouse_qp_stim,
})

print(f'\nPer-mouse DataFrame ({len(df_mouse)} mice):')
print(df_mouse.to_string(index=False))
df_mouse_raw = df_mouse.copy()

# Floor negative bias values to zero (negative bias is not meaningful here)
for col in ['Bias_Values_Nonstim_LC', 'Bias_Values_Stim_LC',
            'Bias_Values_Nonstim', 'Bias_Values_Stim']:
    df_mouse[col] = df_mouse[col].clip(lower=0)


# =============================================================================
# PLOTTING
# =============================================================================

save_prefix = f'{FIGURE_SAVE_PATH}/{FIGURE_PREFIX}' if SAVE_FIGURES else None

# Determine stim label from session filters for plot labels
stim_label = 'Stim'
if SESSION_FILTERS.get('Brain_Region') and not callable(SESSION_FILTERS['Brain_Region']):
    stim_label = f"{SESSION_FILTERS['Brain_Region']} Stim"

stats_output_dir = Path(FIGURE_SAVE_PATH)
stats_output_dir.mkdir(parents=True, exist_ok=True)
stats_prefix = FIGURE_PREFIX if FIGURE_PREFIX else 'opto_analysis'

df_wheel_scalar_trials, df_wheel_scalar_mouse = _wheel_mouse_summary(wheel_scalar_rows)
df_wheel_difference_mouse = _wheel_difference_summary(df_wheel_scalar_mouse)
wheel_scalar_trials_path = stats_output_dir / f'{stats_prefix}_wheel_scalar_trials.csv'
wheel_scalar_mouse_path = stats_output_dir / f'{stats_prefix}_wheel_scalar_per_mouse.csv'
wheel_difference_mouse_path = stats_output_dir / f'{stats_prefix}_wheel_R_minus_L_per_mouse.csv'
df_wheel_scalar_trials.to_csv(wheel_scalar_trials_path, index=False)
df_wheel_scalar_mouse.to_csv(wheel_scalar_mouse_path, index=False)
df_wheel_difference_mouse.to_csv(wheel_difference_mouse_path, index=False)

if len(df_wheel_scalar_mouse):
    statistics_summary_rows.extend(_plot_wheel_scalar_bars(
        df_wheel_scalar_mouse,
        'QP',
        title=f'{title_text}\nQP wheel @ {WHEEL_QP_SCALAR_TIME:g}s',
        save_path=stats_output_dir / f'{stats_prefix}_wheel_QP_scalar_bars.png',
        figsize=WHEEL_SCALAR_BAR_FIGSIZE,
    ))
    statistics_summary_rows.extend(_plot_wheel_scalar_bars(
        df_wheel_scalar_mouse,
        'GoCue_low_contrast',
        title=f'{title_text}\nGo cue low contrast wheel @ {WHEEL_GOCUE_SCALAR_TIME:g}s',
        save_path=stats_output_dir / f'{stats_prefix}_wheel_gocue_low_contrast_scalar_bars.png',
        figsize=WHEEL_SCALAR_BAR_FIGSIZE,
    ))
if len(df_wheel_difference_mouse):
    statistics_summary_rows.extend(_plot_wheel_difference_bars(
        df_wheel_difference_mouse,
        'QP',
        title=f'{title_text}\nQP R-L wheel @ {WHEEL_QP_SCALAR_TIME:g}s',
        save_path=stats_output_dir / f'{stats_prefix}_wheel_QP_R_minus_L_bars.png',
        figsize=WHEEL_DIFFERENCE_BAR_FIGSIZE,
    ))
    statistics_summary_rows.extend(_plot_wheel_difference_bars(
        df_wheel_difference_mouse,
        'GoCue_low_contrast',
        title=f'{title_text}\nGo cue low contrast R-L wheel @ {WHEEL_GOCUE_SCALAR_TIME:g}s',
        save_path=stats_output_dir / f'{stats_prefix}_wheel_gocue_low_contrast_R_minus_L_bars.png',
        figsize=WHEEL_DIFFERENCE_BAR_FIGSIZE,
    ))

for metric, ctrl_col, stim_col in [
    ('bias_shift_low_contrasts', 'Bias_Values_Nonstim_LC', 'Bias_Values_Stim_LC'),
    ('bias_shift_all_contrasts', 'Bias_Values_Nonstim', 'Bias_Values_Stim'),
    ('high_contrast_accuracy', 'Accuracy_HC_Control', 'Accuracy_HC_Stim'),
    ('zero_contrast_accuracy', 'Accuracy_0_Control', 'Accuracy_0_Stim'),
    ('reaction_time', 'RT_Control', 'RT_Stim'),
    ('quiescence_period', 'QP_Control', 'QP_Stim'),
]:
    statistics_summary_rows.append(_paired_t_summary(
        'mouse', metric,
        df_mouse[ctrl_col].to_numpy(float),
        df_mouse[stim_col].to_numpy(float),
        control_label='Control', stim_label=stim_label,
    ))

df_session = pd.DataFrame(session_summary_rows)
df_stats = pd.DataFrame(statistics_summary_rows)

session_summary_path = stats_output_dir / f'{stats_prefix}_session_summary.csv'
per_mouse_raw_summary_path = stats_output_dir / f'{stats_prefix}_per_mouse_summary_raw.csv'
per_mouse_summary_path = stats_output_dir / f'{stats_prefix}_per_mouse_summary.csv'
statistics_summary_path = stats_output_dir / f'{stats_prefix}_statistics_summary.csv'
p_values_path = stats_output_dir / f'{stats_prefix}_p_values.csv'

df_session.to_csv(session_summary_path, index=False)
df_mouse_raw.to_csv(per_mouse_raw_summary_path, index=False)
df_mouse.to_csv(per_mouse_summary_path, index=False)
df_stats.to_csv(statistics_summary_path, index=False)
if len(df_stats):
    p_cols = [
        col for col in [
            'analysis_level', 'metric', 'test', 'n', 'n_control', 'n_stim',
            'control_mean', 'stim_mean', 'stim_minus_control_mean',
            'statistic', 'p_value',
        ]
        if col in df_stats.columns
    ]
    df_stats[p_cols].to_csv(p_values_path, index=False)

print('\nSaved summary/statistics tables:')
print(f'  Session values: {session_summary_path}')
print(f'  Per-mouse values (raw): {per_mouse_raw_summary_path}')
print(f'  Per-mouse values (plot/stat): {per_mouse_summary_path}')
print(f'  Wheel scalar trial values: {wheel_scalar_trials_path}')
print(f'  Wheel scalar per-mouse values: {wheel_scalar_mouse_path}')
print(f'  Wheel R-L per-mouse values: {wheel_difference_mouse_path}')
print(f'  Statistics: {statistics_summary_path}')
if len(df_stats):
    print(f'  P-values quick view: {p_values_path}')

# 1. Concatenated psychometric curves (with trial/mouse count annotation)
if stim_trials_master is not None:
    if PSYCHOMETRIC_MEAN_OF_MICE:
        # Mean-of-mice: average each mouse's psychometric points, then plot
        stim_psycho_master, nonstim_psycho_master = compute_mean_psychometric_across_mice(
            mouse_trials_container, PSYCHO_FIT_KWARGS
        )
    else:
        # Pooled: fit single psychometric to all concatenated trials
        stim_master_contrast = signed_contrast(stim_trials_master)
        nonstim_master_contrast = signed_contrast(nonstim_trials_master)
        stim_psycho_master = organize_psychodata(stim_trials_master, stim_master_contrast)
        nonstim_psycho_master = organize_psychodata(nonstim_trials_master, nonstim_master_contrast)

    plot_psychometric_curves(
        stim_psycho_master, nonstim_psycho_master, PSYCHO_FIT_KWARGS,
        title=title_text,
        save_path=f'{save_prefix}_psychometric.png' if save_prefix else None,
        plot_for_paper=PLOT_FOR_PAPER,
        n_stim_trials=sum(stim_trials_per_session),
        n_nonstim_trials=sum(nonstim_trials_per_session),
        n_mice=num_unique_mice,
        overlay=PSYCHOMETRIC_OVERLAY,
    )

# 2. Bias shift comparison (per-session paired)
if np.sum(valid_mask) > 1:
    plot_bias_shift_comparison(
        bias_shift_sum_all_stim, bias_shift_sum_all_nonstim,
        title=title_text,
        save_path=f'{save_prefix}_bias_shift.png' if save_prefix else None,
        plot_for_paper=PLOT_FOR_PAPER,
    )

# 3. Wheel trajectories
if _has_wheel_data(
    Rblock_wheel_movements_stim, Lblock_wheel_movements_stim,
    Rblock_wheel_movements_nonstim, Lblock_wheel_movements_nonstim,
):
    plot_wheel_comparison(
        Rblock_wheel_movements_stim, Lblock_wheel_movements_stim,
        Rblock_wheel_movements_nonstim, Lblock_wheel_movements_nonstim,
        align_to=WHEEL_ALIGN_TO, interval=WHEEL_TIME_INTERVAL,
        duration=WHEEL_ANALYSIS_DURATION,
        title=title_text,
        save_path=f'{save_prefix}_wheel.png' if save_prefix else None,
        x_limits=WHEEL_QP_X_LIMITS,
        y_limits=WHEEL_QP_Y_LIMITS,
        figsize=(3, 2.5),
    )

if _has_wheel_data(
    Rblock_wheel_movements_stim_gocue_low, Lblock_wheel_movements_stim_gocue_low,
    Rblock_wheel_movements_nonstim_gocue_low, Lblock_wheel_movements_nonstim_gocue_low,
):
    plot_wheel_comparison(
        Rblock_wheel_movements_stim_gocue_low,
        Lblock_wheel_movements_stim_gocue_low,
        Rblock_wheel_movements_nonstim_gocue_low,
        Lblock_wheel_movements_nonstim_gocue_low,
        align_to='goCue', interval=WHEEL_TIME_INTERVAL,
        duration=WHEEL_GOCUE_LOW_CONTRAST_DURATION,
        title=f'{title_text}\nLow contrasts only',
        save_path=f'{save_prefix}_wheel_gocue_low_contrast.png' if save_prefix else None,
        x_limits=(0, WHEEL_GOCUE_LOW_CONTRAST_DURATION),
        figsize=(3, 2.5),
    )

# 4. RT distribution
fig, ax = plt.subplots(figsize=(8, 5))
bins = np.linspace(0, min(RT_THRESHOLD, 10), 50)
ax.hist(rt_nonstim_clean, bins=bins, alpha=0.5, color='grey', label='Nonstim', density=True)
ax.hist(rt_stim_clean, bins=bins, alpha=0.5, color='dodgerblue', label='Stim', density=True)
ax.set_xlabel('Reaction time (s)')
ax.set_ylabel('Density')
ax.set_title(f'RT distributions — {title_text}')
ax.legend()
sns.despine(offset=10)
plt.tight_layout()
if save_prefix:
    plt.savefig(f'{save_prefix}_RT_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
else:
    plt.show()

# 5. Per-mouse bar + line plots
if len(df_mouse) >= 2:
    bar_save = str(FIGURE_SAVE_PATH) if SAVE_FIGURES else None

    plot_bars_by_mouse(df_mouse, mode='Bias_LC',
                       save_path=bar_save, prefix=FIGURE_PREFIX, stim_label=stim_label)

    plot_bars_by_mouse(df_mouse, mode='Bias_All',
                       save_path=bar_save, prefix=FIGURE_PREFIX, stim_label=stim_label)

    plot_bars_by_mouse(df_mouse, mode='Accuracy_HC',
                       save_path=bar_save, prefix=FIGURE_PREFIX, stim_label=stim_label)

    plot_bars_by_mouse(df_mouse, mode='Accuracy_0',
                       save_path=bar_save, prefix=FIGURE_PREFIX, stim_label=stim_label)

    plot_bars_by_mouse(df_mouse, mode='RT',
                       save_path=bar_save, prefix=FIGURE_PREFIX, stim_label=stim_label)

    plot_bars_by_mouse(df_mouse, mode='QP',
                       save_path=bar_save, prefix=FIGURE_PREFIX, stim_label=stim_label)
else:
    print('\nNot enough mice (need >= 2) for per-mouse bar plots.')

# 6. Per-mouse psychometric curves (grid)
if len(mouse_trials_container) > 0:
    plot_per_mouse_psychometrics(
        mouse_trials_container, PSYCHO_FIT_KWARGS,
        title=title_text,
        save_path=f'{save_prefix}_psychometric_per_mouse.png' if save_prefix else None,
    )

print('\nDone.')
