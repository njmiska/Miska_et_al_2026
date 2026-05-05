#!/usr/bin/env python3
"""
Zapit Optogenetics Analysis Pipeline

Analyzes behavioral data from the IBL task with laser scanning optogenetic
inhibition (Zapit system). Mice perform a 2-AFC task with prior blocks while
receiving bilateral optogenetic stimulation at one of 52 cortical locations.

Key analyses:
- Reaction time effects by stimulation location
- Bias shift effects (block-induced choice bias)

Usage:
    1. Configure paths, options, and session filters in config.py
    2. Update session metadata in metadata_zapit.py
    3. Run this script: python zapit_analysis.py

Author: Nate Miska
        Developed with AI pair-programming assistance (Claude, Anthropic)
        for code refactoring and documentation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from datetime import datetime
from scipy import stats
from one.api import ONE
from pathlib import Path
from statsmodels.stats.proportion import proportions_ztest

# Local imports
from config import (
    ZAPIT_TRIALS_LOG, ZAPIT_LOCATIONS_LOG,
    ALLEN_CCF_ANNOTATION, ALLEN_STRUCTURE_TREE,
    FIGURE_SAVE_PATH, FIGURE_PREFIX, SAVE_FIGURES,
    BASELINE_PERFORMANCE_THRESHOLD, STIM_PERFORMANCE_THRESHOLD,
    MIN_BIAS_THRESHOLD_ZAPIT, RT_THRESHOLD, MIN_NUM_TRIALS,
    NUM_STIM_LOCATIONS, TRIALS_PER_DATAPOINT,
    USE_TRIALS_AFTER_STIM,
    BIAS_REDUCTION_VMIN, BIAS_REDUCTION_VMAX,
    WHEEL_ANALYSIS_DURATION, WHEEL_TIME_INTERVAL, WHEEL_ALIGN_TO,
    ONLY_LOW_CONTRASTS, LOW_CONTRAST_THRESHOLD,
    ALL_CONTRASTS, LOW_CONTRASTS, BIAS_HEATMAP_CONTRASTS,
    ALYX_BASE_URL, FLAG_FAILED_LOADS,
    SESSION_FILTERS,
    BREGMA_CCF, CCF_SCALE_FACTOR,
    PLOT_AP_LIMITS, PLOT_ML_LIMITS,
    SHOW_RIGHT_HEMISPHERE_ONLY,
    BRAIN_BACKGROUND_COLOR, BRAIN_BORDER_COLOR,
    USE_GLMHMM, N_STATES, STATE_TYPE, STATE_DEF,
    GLMHMM_STATES_FILE, GLMHMM_ENGAGED_PREV_FILE, GLMHMM_DISENGAGED_PREV_FILE,
    PSYCHOMETRIC_STIM_CONDITIONS, PSYCHOMETRIC_TITLES,
    PSYCHOMETRIC_OVERLAY, PSYCHO_FIT_KWARGS,
    MULTIPLE_COMPARISONS_CORRECTION,
)
from metadata_zapit import sessions, find_sessions_by_advanced_criteria
from zapit_helpers import (
    load_session_data, load_laser_intervals, parse_zapit_log,
    build_stim_location_dict, load_stim_locations_coordinates,
    signed_contrast, create_trial_data_dict, get_valid_trials_range,
    check_session_accuracy, compute_session_bias_shift,
    extract_wheel_trajectory,
    compute_bias_values_by_contrast, compute_bias_values_by_cycle,
    compute_effect_sizes,
    run_condition_comparisons, run_cycle_comparisons, run_rt_analysis,
    generate_mip_with_borders,
    load_glmhmm_data, is_eid_successful, get_glmhmm_state_filter_indices,
    plot_zapit_psychometric,
    correct_pvals_dict,
)


# =============================================================================
# SESSION SELECTION (configured in config.py)
# =============================================================================

# Filter out None values from session filters
active_filters = {k: v for k, v in SESSION_FILTERS.items() if v is not None}

eids, trials_ranges, MouseIDs, stim_params = find_sessions_by_advanced_criteria(
    sessions, **active_filters
)

print(f"Found {len(eids)} sessions matching criteria")
if active_filters:
    print(f"Active filters: {active_filters}")


# =============================================================================
# INITIALIZE DATA STRUCTURES
# =============================================================================

# Connect to IBL database
one = ONE(base_url=ALYX_BASE_URL, cache_dir=Path.home() / 'Downloads' / 'ONE' / 'alyx.internationalbrainlab.org')

# Load GLM-HMM data if enabled
state_probability = None
if USE_GLMHMM:
    print(f"\nGLM-HMM enabled: n_states={N_STATES}, state_type={STATE_TYPE}, state_def={STATE_DEF}")
    state_probability, _, _ = load_glmhmm_data(
        GLMHMM_STATES_FILE,
        GLMHMM_ENGAGED_PREV_FILE,
        GLMHMM_DISENGAGED_PREV_FILE,
    )

# Trial data organized by condition (0 = control, 1-52 = stim locations)
condition_data = {i: [] for i in range(NUM_STIM_LOCATIONS + 1)}

# Wheel data by condition and block
Rblock_wheel_by_condition = [[] for _ in range(NUM_STIM_LOCATIONS + 1)]
Lblock_wheel_by_condition = [[] for _ in range(NUM_STIM_LOCATIONS + 1)]

# Session tracking
num_analyzed_sessions = 0
num_unique_mice = 0
previous_mouse_id = None


# =============================================================================
# MAIN SESSION LOOP
# =============================================================================

print("\n" + "="*60)
print("Processing sessions...")
print("="*60 + "\n")

for j, eid in enumerate(eids):
    print(f"Processing session {j+1}/{len(eids)}: {eid}")
    
    # -------------------------------------------------------------------------
    # Load session data
    # -------------------------------------------------------------------------
    trials, wheel = load_session_data(one, eid, flag_failed_loads=FLAG_FAILED_LOADS)
    if trials is None:
        continue
    
    laser_intervals = load_laser_intervals(one, eid)
    current_mouse_id = MouseIDs[j]
    
    # GLM-HMM: check that states exist for this session
    if USE_GLMHMM:
        if not is_eid_successful(state_probability, current_mouse_id, eid):
            print(f"  Skipping: No GLM-HMM states found for {current_mouse_id} / {eid}")
            continue
    
    # -------------------------------------------------------------------------
    # Determine valid trial range
    # -------------------------------------------------------------------------
    num_trials = len(trials['contrastLeft'])
    trials_range = get_valid_trials_range(trials_ranges[j], num_trials)
    
    if len(trials_range) < MIN_NUM_TRIALS:
        print(f"  Skipping: Only {len(trials_range)} trials (minimum: {MIN_NUM_TRIALS})")
        continue
    
    # -------------------------------------------------------------------------
    # Parse Zapit log and build stim location mapping
    # -------------------------------------------------------------------------
    details = one.get_details(eid)
    session_start = datetime.strptime(details['start_time'][:19], '%Y-%m-%dT%H:%M:%S')
    
    relevant_events = parse_zapit_log(ZAPIT_TRIALS_LOG, session_start, eid)
    
    stimtrial_location_dict = build_stim_location_dict(
        laser_intervals, trials, relevant_events, trials_range, eid
    )
    
    # Handle "trials after stim" analysis mode
    if USE_TRIALS_AFTER_STIM:
        stimtrial_location_dict_original = stimtrial_location_dict.copy()
        stimtrial_location_dict = {
            k + 1: v for k, v in stimtrial_location_dict.items()
            if min(trials_range) <= k < max(trials_range)
        }
    
    # -------------------------------------------------------------------------
    # GLM-HMM state filtering (if enabled)
    # -------------------------------------------------------------------------
    if USE_GLMHMM:
        try:
            state_filter = get_glmhmm_state_filter_indices(
                current_mouse_id, eid, state_probability,
                N_STATES, STATE_TYPE, STATE_DEF,
            )
        except Exception as e:
            print(f"  Skipping: GLM-HMM filtering failed ({e})")
            continue
        
        if state_filter is not None:
            # Convert to set for O(1) membership checks
            state_filter_set = set(state_filter.tolist())
            n_before = len(stimtrial_location_dict)
            stimtrial_location_dict = {
                k: v for k, v in stimtrial_location_dict.items()
                if k in state_filter_set
            }
            n_after = len(stimtrial_location_dict)
            print(f"  GLM-HMM filter ({STATE_TYPE}, {STATE_DEF}): "
                  f"kept {n_after}/{n_before} trials")
            
            if n_after == 0:
                print(f"  Skipping: No trials remain after GLM-HMM filtering")
                continue
    
    # -------------------------------------------------------------------------
    # Session quality check: accuracy at high contrasts
    # -------------------------------------------------------------------------
    # Get control trials for quality check
    control_trial_nums = [t for t, c in stimtrial_location_dict.items() if c == 0]
    
    contrast_values = signed_contrast(trials)
    session_control_data = []
    
    for trial_num in control_trial_nums:
        trial_data = create_trial_data_dict(trials, trial_num, contrast_values)
        if trial_data['reaction_times'] <= RT_THRESHOLD and not np.isnan(trial_data['reaction_times']):
            session_control_data.append(trial_data)
    
    passes_accuracy, accuracy = check_session_accuracy(
        session_control_data, BASELINE_PERFORMANCE_THRESHOLD
    )
    if not passes_accuracy:
        print(f"  Skipping: Accuracy {accuracy:.2f} below threshold {BASELINE_PERFORMANCE_THRESHOLD}")
        continue
    
    # -------------------------------------------------------------------------
    # Session quality check: stim-trial accuracy at high contrasts
    # -------------------------------------------------------------------------
    # Pool *all* stim trials (across the 52 stim conditions) and check
    # whether the mouse is performing acceptably under stim. Pooling avoids
    # the circularity of per-condition gating, which would exclude the very
    # sessions where stim has the strongest behavioural effects.
    #
    # Threshold is absolute (matches the BASELINE_PERFORMANCE_THRESHOLD
    # convention). Set STIM_PERFORMANCE_THRESHOLD = 0 in config.py to disable.
    stim_trial_nums = [t for t, c in stimtrial_location_dict.items() if c != 0]
    session_stim_data = []
    for trial_num in stim_trial_nums:
        trial_data = create_trial_data_dict(trials, trial_num, contrast_values)
        if trial_data['reaction_times'] <= RT_THRESHOLD and not np.isnan(trial_data['reaction_times']):
            session_stim_data.append(trial_data)
    
    passes_stim_accuracy, stim_accuracy = check_session_accuracy(
        session_stim_data, STIM_PERFORMANCE_THRESHOLD
    )
    
    # Diagnostic printout — useful for picking a sensible STIM_PERFORMANCE_THRESHOLD.
    # Reports nonstim and stim high-contrast accuracy side-by-side per session.
    print(f"  High-contrast accuracy:  nonstim={accuracy:.3f}  "
          f"stim={stim_accuracy:.3f}  "
          f"(n_nonstim={len(session_control_data)}, n_stim={len(session_stim_data)})")
    
    if not passes_stim_accuracy:
        print(f"  Skipping: Stim accuracy {stim_accuracy:.2f} below threshold {STIM_PERFORMANCE_THRESHOLD}")
        continue
    
    # -------------------------------------------------------------------------
    # Session quality check: bias shift
    # -------------------------------------------------------------------------
    total_bias_shift = compute_session_bias_shift(session_control_data)
    print(f"  Accuracy: {accuracy:.2f}, Bias shift: {total_bias_shift:.2f}")
    
    if total_bias_shift < MIN_BIAS_THRESHOLD_ZAPIT:
        print(f"  Skipping: Bias shift below threshold {MIN_BIAS_THRESHOLD_ZAPIT}")
        continue
    
    # -------------------------------------------------------------------------
    # Collect trial data for all conditions
    # -------------------------------------------------------------------------
    whlpos, whlt = wheel.position, wheel.timestamps
    
    for trial_num, condition_num in stimtrial_location_dict.items():
        # Skip trials that don't meet RT criteria
        trial_data = create_trial_data_dict(trials, trial_num, contrast_values)
        
        if np.isnan(trial_data['reaction_times']):
            continue
        if trial_data['reaction_times'] > RT_THRESHOLD:
            continue
        
        # Optional: filter for low contrasts only
        if ONLY_LOW_CONTRASTS:
            if abs(trial_data['contrast']) > LOW_CONTRAST_THRESHOLD:
                continue
        
        # Add to condition data
        condition_data[condition_num].append(trial_data)
        
        # Extract wheel trajectory
        if wheel is not None:
            trajectory = extract_wheel_trajectory(
                wheel, trials, trial_num,
                align_to=WHEEL_ALIGN_TO,
                duration=WHEEL_ANALYSIS_DURATION,
                interval=WHEEL_TIME_INTERVAL
            )
            
            # Store by block type
            if trial_data['probabilityLeft'] == 0.2:  # R block
                if len(Rblock_wheel_by_condition[condition_num]) == 0:
                    Rblock_wheel_by_condition[condition_num] = trajectory.reshape(1, -1)
                else:
                    Rblock_wheel_by_condition[condition_num] = np.vstack([
                        Rblock_wheel_by_condition[condition_num], trajectory
                    ])
            else:  # L block
                if len(Lblock_wheel_by_condition[condition_num]) == 0:
                    Lblock_wheel_by_condition[condition_num] = trajectory.reshape(1, -1)
                else:
                    Lblock_wheel_by_condition[condition_num] = np.vstack([
                        Lblock_wheel_by_condition[condition_num], trajectory
                    ])
    
    # Update session counters
    num_analyzed_sessions += 1
    if current_mouse_id != previous_mouse_id:
        num_unique_mice += 1
        previous_mouse_id = current_mouse_id
    
    print(f"  ✓ Session added (Control: {len(condition_data[0])} trials)")


# =============================================================================
# CHECK THAT WE HAVE DATA
# =============================================================================

if num_analyzed_sessions == 0:
    raise RuntimeError("No sessions met criteria for analysis!")

print("\n" + "="*60)
print(f"Analysis complete: {num_analyzed_sessions} sessions, {num_unique_mice} mice")
print(f"Total control trials: {len(condition_data[0])}")
print(f"Total stim trials: {sum(len(condition_data[c]) for c in range(1, 53))}")
print("="*60 + "\n")


# =============================================================================
# COMPUTE BIAS METRICS
# =============================================================================

print("Computing bias metrics...")

# Bias by contrast level (for paired comparisons)
contrasts_to_use = LOW_CONTRASTS if ONLY_LOW_CONTRASTS else ALL_CONTRASTS
bias_values, left_block_probs, right_block_probs = compute_bias_values_by_contrast(
    condition_data, contrasts_to_use, NUM_STIM_LOCATIONS + 1
)

# Bias by trial cycles (for independent comparisons)
bias_vals_LC = compute_bias_values_by_cycle(
    condition_data, 
    trials_per_cycle=TRIALS_PER_DATAPOINT,
    low_contrast_threshold=LOW_CONTRAST_THRESHOLD,
    num_conditions=NUM_STIM_LOCATIONS + 1
)

# Effect sizes
effect_sizes = compute_effect_sizes(bias_values, only_low_contrasts=ONLY_LOW_CONTRASTS)
effect_sizes_LC = {}
ctrl_mean = np.mean(bias_vals_LC[0]) if len(bias_vals_LC[0]) > 0 else 0
for cond in range(1, 53):
    if len(bias_vals_LC[cond]) > 0 and ctrl_mean != 0:
        stim_mean = np.mean(bias_vals_LC[cond])
        effect_sizes_LC[cond] = -(stim_mean - ctrl_mean) / ctrl_mean


# =============================================================================
# STATISTICAL COMPARISONS
# =============================================================================

print("Running statistical tests...")

# Contrast-based comparisons (Mann-Whitney, paired t-test)
contrast_stats = run_condition_comparisons(bias_values, NUM_STIM_LOCATIONS + 1)

# Cycle-based comparisons (independent t-test)
cycle_stats = run_cycle_comparisons(bias_vals_LC, NUM_STIM_LOCATIONS + 1)

# RT analysis
rt_results, qp_results, lapse_results = run_rt_analysis(
    condition_data, NUM_STIM_LOCATIONS
)

print(f"  Kruskal-Wallis (contrasts): p = {contrast_stats.get('kruskal_p', 'N/A'):.4g}")
print(f"  Kruskal-Wallis (cycles): p = {cycle_stats.get('kruskal_p', 'N/A'):.4g}")


# =============================================================================
# LOAD STIM LOCATION COORDINATES
# =============================================================================

stim_locations = load_stim_locations_coordinates(ZAPIT_LOCATIONS_LOG)


# =============================================================================
# PLOTTING
# =============================================================================

print("\nGenerating figures...")

# Set up figure directory
FIGURE_SAVE_PATH.mkdir(parents=True, exist_ok=True)

# Load Allen CCF
print("  Loading Allen CCF data...")
allenCCF_data = np.load(ALLEN_CCF_ANNOTATION)
structure_tree = pd.read_csv(ALLEN_STRUCTURE_TREE)

# Generate brain surface image with borders
print("  Generating max intensity projection (this may take a moment)...")
mip, edges = generate_mip_with_borders(allenCCF_data)

# Create binary border image (white borders on black)
dorsal_mip_with_borders = np.where(edges > 0, 1, 0)

# Calculate the extents using original method (negative scale factor handles flip)
bregma = BREGMA_CCF  # [AP, DV, ML]
scale_factor = CCF_SCALE_FACTOR  # -100

left_extent = -bregma[2] / scale_factor
right_extent = (dorsal_mip_with_borders.shape[1] - bregma[2]) / scale_factor
lower_extent = (dorsal_mip_with_borders.shape[0] - bregma[0]) / scale_factor
upper_extent = -bregma[0] / scale_factor

extent = [left_extent, right_extent, lower_extent, upper_extent]

# -----------------------------------------------------------------------------
# Brain background / border styling (driven by BRAIN_BACKGROUND_COLOR
# and BRAIN_BORDER_COLOR in config.py)
# -----------------------------------------------------------------------------
# Choose the colormap so that the binary border image renders the requested
# colours: dorsal_mip_with_borders is 0 (off-border) / 1 (border).
#   - 'gray'   -> 0=black, 1=white   (white borders on black bg)
#   - 'gray_r' -> 0=white, 1=black   (black borders on white bg)
_brain_bg_lower = str(BRAIN_BACKGROUND_COLOR).lower()
_is_dark_bg = _brain_bg_lower in ('black', 'k') or _brain_bg_lower.startswith('#0')

if _is_dark_bg:
    BRAIN_CMAP = 'gray'           # 1 -> white borders on black bg
    AXIS_TEXT_COLOR = 'white'
else:
    BRAIN_CMAP = 'gray_r'          # 1 -> black borders on light bg
    AXIS_TEXT_COLOR = 'black'

# If user explicitly set BRAIN_BORDER_COLOR to something specific, use that
# for the text contrast colour as well (lets users force e.g. dark-grey text
# on a white plot for a softer look).
if BRAIN_BORDER_COLOR not in (None, '', 'auto'):
    # only override text colour if it makes sense (i.e. not "white" on white bg)
    _border_lower = str(BRAIN_BORDER_COLOR).lower()
    if (_is_dark_bg and _border_lower not in ('black', 'k')) or \
       (not _is_dark_bg and _border_lower not in ('white', 'w')):
        AXIS_TEXT_COLOR = BRAIN_BORDER_COLOR


def _style_brain_axes(fig, ax, cbar=None, legend=None):
    """
    Apply BRAIN_BACKGROUND_COLOR + AXIS_TEXT_COLOR styling to a heatmap figure.

    Sets figure background, axes facecolor, spine/tick/label/title colours,
    and (optionally) restyles a colorbar and a legend so all text remains
    readable against the chosen background.
    """
    fig.patch.set_facecolor(BRAIN_BACKGROUND_COLOR)
    ax.set_facecolor(BRAIN_BACKGROUND_COLOR)

    # Title, labels, ticks, spines
    ax.title.set_color(AXIS_TEXT_COLOR)
    ax.xaxis.label.set_color(AXIS_TEXT_COLOR)
    ax.yaxis.label.set_color(AXIS_TEXT_COLOR)
    ax.tick_params(axis='both', colors=AXIS_TEXT_COLOR)
    for spine in ax.spines.values():
        spine.set_color(AXIS_TEXT_COLOR)

    # Colorbar (if provided)
    if cbar is not None:
        cbar.ax.yaxis.label.set_color(AXIS_TEXT_COLOR)
        cbar.ax.tick_params(colors=AXIS_TEXT_COLOR)
        for spine in cbar.ax.spines.values():
            spine.set_color(AXIS_TEXT_COLOR)

    # Legend (if provided)
    if legend is not None:
        for text in legend.get_texts():
            text.set_color(AXIS_TEXT_COLOR)
        # Frame edge / face — keep slightly transparent to feel light-touch
        frame = legend.get_frame()
        frame.set_edgecolor(AXIS_TEXT_COLOR)


# -----------------------------------------------------------------------------
# Figure 1: RT Heatmap on Brain Atlas
# -----------------------------------------------------------------------------
print("  Generating RT heatmap...")

fig, ax = plt.subplots(figsize=(10, 8))

# Show brain borders (cmap chosen above based on BRAIN_BACKGROUND_COLOR)
ax.imshow(dorsal_mip_with_borders, cmap=BRAIN_CMAP, extent=extent)

# Compute mean RT and p-values for each condition
rt_means = {}
rt_pvals = {}
ctrl_rts = [t['reaction_times'] for t in condition_data[0] if not np.isnan(t['reaction_times'])]
ctrl_mean_rt = np.mean(ctrl_rts) if ctrl_rts else 1.0

for cond in range(1, NUM_STIM_LOCATIONS + 1):
    stim_rts = [t['reaction_times'] for t in condition_data[cond] if not np.isnan(t['reaction_times'])]
    if len(stim_rts) > 0:
        rt_means[cond] = np.mean(stim_rts)
        if len(ctrl_rts) > 0:
            _, p = stats.ttest_ind(stim_rts, ctrl_rts, equal_var=False)
            rt_pvals[cond] = p
        else:
            rt_pvals[cond] = 1.0

# Multiple-comparisons correction across the 52 stim conditions.
# Keep raw p-values for the CSV; use corrected values for plotting / size encoding.
rt_pvals_raw = dict(rt_pvals)
rt_pvals = correct_pvals_dict(rt_pvals_raw, method=MULTIPLE_COMPARISONS_CORRECTION)
if MULTIPLE_COMPARISONS_CORRECTION:
    print(f"  RT p-values corrected via '{MULTIPLE_COMPARISONS_CORRECTION}'")

# Set up colormap - using TwoSlopeNorm centered on control RT (matching original)
from matplotlib import colors
divnorm = colors.TwoSlopeNorm(
    vmin=ctrl_mean_rt - 0.4 * ctrl_mean_rt,
    vcenter=ctrl_mean_rt,
    vmax=ctrl_mean_rt + ctrl_mean_rt
)
sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=divnorm)

# Marker edge colour: contrast with background
_marker_edge = AXIS_TEXT_COLOR

# Plot each stim location (both hemispheres, control display with axis limits)
for condition, coords in stim_locations.items():
    if condition not in rt_means:
        continue
    
    mean_rt = rt_means[condition]
    p_val = rt_pvals.get(condition, 1.0)
    
    # Size based on -100 * log10(p_val) - matching original
    size = -100 * np.log10(p_val) if p_val > 0 else 300
    color = sm.to_rgba(mean_rt)
    alpha = 0.5 if p_val >= 0.05 else 1.0
    
    ml_left = coords.get('ML_left')
    ml_right = coords.get('ML_right')
    ap = coords.get('AP')
    
    if ml_left is not None and ml_right is not None and ap is not None:
        # Plot both hemispheres (axis limits will control what's visible)
        ax.scatter(ml_left, ap, color=color, alpha=alpha, edgecolors=_marker_edge, s=size)
        ax.scatter(ml_right, ap, color=color, alpha=alpha, edgecolors=_marker_edge, s=size)

# Set axis limits
if PLOT_AP_LIMITS is not None:
    ax.set_ylim(bottom=PLOT_AP_LIMITS[0], top=PLOT_AP_LIMITS[1])
if PLOT_ML_LIMITS is not None:
    ax.set_xlim(left=PLOT_ML_LIMITS[0], right=PLOT_ML_LIMITS[1])
elif SHOW_RIGHT_HEMISPHERE_ONLY:
    ax.set_xlim(left=0)  # Show only right hemisphere (ML > 0)

# Labels and title
ax.set_xlabel('Mediolateral Position (mm from Bregma)', fontsize=14)
ax.set_ylabel('Anteroposterior Position (mm from Bregma)', fontsize=14)
ax.set_title(f'Reaction Time by Stimulation Location\n({num_analyzed_sessions} sessions, {num_unique_mice} mice)', fontsize=14)
ax.tick_params(axis='both', which='major', labelsize=12)

# P-value legend (matching original style)
# Marker face colour adapts: we want sized dots that read well over the bg.
_legend_marker_face = AXIS_TEXT_COLOR
p_values_legend = [0.001, 0.05, 0.2]
sizes_legend = [-100 * np.log10(p) for p in p_values_legend]
for p, size in zip(p_values_legend, sizes_legend):
    ax.scatter([], [], s=size, label=f'p = {p}',
               edgecolors=_marker_edge, color=_legend_marker_face)
legend = ax.legend(loc='upper right', labelspacing=1.5, fontsize=10)

# Colorbar
cbar = plt.colorbar(sm, ax=ax, orientation='vertical', fraction=0.046, pad=0.04, aspect=12)
cbar.set_label('Reaction time (s)', fontsize=14, labelpad=15)
cbar.ax.tick_params(labelsize=12)

# Apply BRAIN_BACKGROUND_COLOR styling to the entire figure
_style_brain_axes(fig, ax, cbar=cbar, legend=legend)

plt.tight_layout()

if SAVE_FIGURES:
    plt.savefig(FIGURE_SAVE_PATH / f'{FIGURE_PREFIX}_RT_heatmap.png',
                dpi=150, bbox_inches='tight',
                facecolor=BRAIN_BACKGROUND_COLOR)
    plt.close()
else:
    plt.show()


# -----------------------------------------------------------------------------
# Table 1: RT Statistics by Stimulation Location
# -----------------------------------------------------------------------------
print("  Generating RT statistics table...")

# Compute SEM for each condition
rt_sems = {}
rt_n = {}
for cond in range(NUM_STIM_LOCATIONS + 1):
    rts = [t['reaction_times'] for t in condition_data[cond] if not np.isnan(t['reaction_times'])]
    if len(rts) > 0:
        rt_sems[cond] = stats.sem(rts)
        rt_n[cond] = len(rts)
    else:
        rt_sems[cond] = np.nan
        rt_n[cond] = 0

# Build table data
table_data = []

# Add control row first
table_data.append({
    'Condition': 0,
    'AP (mm)': 'N/A',
    'ML (mm)': 'N/A',
    'RT Mean (s)': f"{ctrl_mean_rt:.3f}",
    'RT SEM (s)': f"{rt_sems.get(0, np.nan):.3f}",
    'P-value': 'N/A (control)',
    'n trials': rt_n.get(0, 0)
})

# Add stim location rows
for cond in range(1, NUM_STIM_LOCATIONS + 1):
    coords = stim_locations.get(cond, {})
    ap = coords.get('AP', np.nan)
    # Use ML_left as the positive (right hemisphere) coordinate
    ml = coords.get('ML_left', np.nan)
    
    rt_mean = rt_means.get(cond, np.nan)
    rt_sem = rt_sems.get(cond, np.nan)
    p_val = rt_pvals.get(cond, np.nan)
    n = rt_n.get(cond, 0)
    
    # Format p-value with scientific notation for very small values
    if not np.isnan(p_val):
        if p_val < 0.001:
            p_str = f"{p_val:.2e}"
        else:
            p_str = f"{p_val:.4f}"
    else:
        p_str = "N/A"
    
    table_data.append({
        'Condition': cond,
        'AP (mm)': f"{ap:.2f}" if not np.isnan(ap) else "N/A",
        'ML (mm)': f"{ml:.2f}" if not np.isnan(ml) else "N/A",
        'RT Mean (s)': f"{rt_mean:.3f}" if not np.isnan(rt_mean) else "N/A",
        'RT SEM (s)': f"{rt_sem:.3f}" if not np.isnan(rt_sem) else "N/A",
        'P-value': p_str,
        'n trials': n
    })

# Create DataFrame and save
rt_table_df = pd.DataFrame(table_data)

if SAVE_FIGURES:
    # Save as CSV
    rt_table_df.to_csv(FIGURE_SAVE_PATH / f'{FIGURE_PREFIX}_RT_statistics.csv', index=False)
    print(f"  RT statistics table saved to {FIGURE_SAVE_PATH / f'{FIGURE_PREFIX}_RT_statistics.csv'}")


# -----------------------------------------------------------------------------
# Figure: Condition-selected psychometric curves
# -----------------------------------------------------------------------------
# Generates one or more psychometric figures comparing pooled stim trials at
# user-specified condition numbers vs pooled control trials. Configured via
# PSYCHOMETRIC_STIM_CONDITIONS in config.py.
#
# For interactive exploration in iPython after running this script:
#     from zapit_helpers import plot_zapit_psychometric
#     plot_zapit_psychometric([16, 22], condition_data,
#                             fit_kwargs=PSYCHO_FIT_KWARGS, title='MOs cluster')

if PSYCHOMETRIC_STIM_CONDITIONS:
    print("\n  Generating condition-selected psychometric curves...")

    # Normalize to list-of-lists: a flat list of ints becomes [[...]]
    psy_specs = PSYCHOMETRIC_STIM_CONDITIONS
    if all(isinstance(c, (int, np.integer)) for c in psy_specs):
        psy_specs = [list(psy_specs)]
    else:
        psy_specs = [list(s) for s in psy_specs]

    # Match titles if provided
    titles = PSYCHOMETRIC_TITLES if PSYCHOMETRIC_TITLES else [None] * len(psy_specs)
    if len(titles) < len(psy_specs):
        titles = list(titles) + [None] * (len(psy_specs) - len(titles))

    for spec, custom_title in zip(psy_specs, titles):
        if custom_title is None:
            title = f"Stim conditions: {spec}"
        else:
            title = custom_title

        # Filename-safe condition tag for saving
        cond_tag = '_'.join(str(c) for c in spec)
        save_path = (FIGURE_SAVE_PATH / f'{FIGURE_PREFIX}_psychometric_{cond_tag}.png'
                     if SAVE_FIGURES else None)

        plot_zapit_psychometric(
            stim_conditions=spec,
            condition_data=condition_data,
            fit_kwargs=PSYCHO_FIT_KWARGS,
            title=title,
            save_path=save_path,
            n_mice=num_unique_mice,
            num_sessions=num_analyzed_sessions,
            overlay=PSYCHOMETRIC_OVERLAY,
        )


# -----------------------------------------------------------------------------
# Figure 2: QP (Quiescent Period) Heatmap on Brain Atlas
# -----------------------------------------------------------------------------
# Mean quiescent-period duration on stim trials at each location, vs control.
# Colour: TwoSlopeNorm-like centring around control QP ± 30%
#         (red = longer QP; blue = shorter QP).
# Size:   -100 * log10(p-value) of Welch's t-test stim vs control.
print("  Generating QP heatmap...")

fig, ax = plt.subplots(figsize=(10, 8))
ax.imshow(dorsal_mip_with_borders, cmap=BRAIN_CMAP, extent=extent)

# Compute mean QP and p-values for each condition
qp_means = {}
qp_pvals = {}
ctrl_qps = [t['qp_times'] for t in condition_data[0] if not np.isnan(t['qp_times'])]
ctrl_mean_qp = np.mean(ctrl_qps) if ctrl_qps else 0.5

for cond in range(1, NUM_STIM_LOCATIONS + 1):
    stim_qps = [t['qp_times'] for t in condition_data[cond] if not np.isnan(t['qp_times'])]
    if len(stim_qps) > 0:
        qp_means[cond] = np.mean(stim_qps)
        if len(ctrl_qps) > 0:
            _, p = stats.ttest_ind(stim_qps, ctrl_qps, equal_var=False)
            qp_pvals[cond] = p
        else:
            qp_pvals[cond] = 1.0

# Multiple-comparisons correction across the 52 stim conditions.
qp_pvals_raw = dict(qp_pvals)
qp_pvals = correct_pvals_dict(qp_pvals_raw, method=MULTIPLE_COMPARISONS_CORRECTION)
if MULTIPLE_COMPARISONS_CORRECTION:
    print(f"  QP p-values corrected via '{MULTIPLE_COMPARISONS_CORRECTION}'")

# Colormap centred on control QP ± 30% (matches original convention)
norm_qp = mcolors.Normalize(
    vmin=ctrl_mean_qp - 0.3 * ctrl_mean_qp,
    vmax=ctrl_mean_qp + 0.3 * ctrl_mean_qp,
)
sm_qp = plt.cm.ScalarMappable(cmap='coolwarm', norm=norm_qp)

for condition, coords in stim_locations.items():
    if condition not in qp_means:
        continue
    mean_qp = qp_means[condition]
    p_val = qp_pvals.get(condition, 1.0)

    size = -100 * np.log10(p_val) if p_val > 0 else 300
    color = sm_qp.to_rgba(mean_qp)
    alpha = 0.5 if p_val >= 0.05 else 1.0

    ml_left = coords.get('ML_left')
    ml_right = coords.get('ML_right')
    ap = coords.get('AP')

    if ml_left is not None and ml_right is not None and ap is not None:
        ax.scatter(ml_left,  ap, color=color, alpha=alpha, s=size, edgecolors=AXIS_TEXT_COLOR)
        ax.scatter(ml_right, ap, color=color, alpha=alpha, s=size, edgecolors=AXIS_TEXT_COLOR)

# Axis limits
if PLOT_AP_LIMITS is not None:
    ax.set_ylim(bottom=PLOT_AP_LIMITS[0], top=PLOT_AP_LIMITS[1])
if PLOT_ML_LIMITS is not None:
    ax.set_xlim(left=PLOT_ML_LIMITS[0], right=PLOT_ML_LIMITS[1])
elif SHOW_RIGHT_HEMISPHERE_ONLY:
    ax.set_xlim(left=0)

ax.set_xlabel('Mediolateral Position (mm from Bregma)', fontsize=14)
ax.set_ylabel('Anteroposterior Position (mm from Bregma)', fontsize=14)
ax.set_title(
    f'Quiescent Period by Stimulation Location\n'
    f'({num_analyzed_sessions} sessions, {num_unique_mice} mice)',
    fontsize=14,
)
ax.tick_params(axis='both', which='major', labelsize=12)

# P-value legend (matching RT/Bias style)
p_values_legend = [0.001, 0.05, 0.2]
for p in p_values_legend:
    s = -100 * np.log10(p)
    ax.scatter([], [], s=s, label=f'p = {p}',
               edgecolors=AXIS_TEXT_COLOR, color=AXIS_TEXT_COLOR)
legend_qp = ax.legend(loc='upper right', labelspacing=1.5, fontsize=10)

# Colorbar
cbar = plt.colorbar(sm_qp, ax=ax, orientation='vertical',
                    fraction=0.046, pad=0.04, aspect=12)
cbar.set_label('Quiescent Period (s)', fontsize=14, labelpad=15)
cbar.ax.tick_params(labelsize=12)

# Apply BRAIN_BACKGROUND_COLOR styling
_style_brain_axes(fig, ax, cbar=cbar, legend=legend_qp)

plt.tight_layout()

if SAVE_FIGURES:
    plt.savefig(FIGURE_SAVE_PATH / f'{FIGURE_PREFIX}_QP_heatmap.png',
                dpi=150, bbox_inches='tight',
                facecolor=BRAIN_BACKGROUND_COLOR)
    plt.close()
else:
    plt.show()


# -----------------------------------------------------------------------------
# Figure 3: Bias Reduction Heatmap
# -----------------------------------------------------------------------------
# Quantifies block-bias reduction at each stim location relative to control.
# For each condition, bias values (mean L-block choice − mean R-block choice
# at low contrasts) are computed in non-overlapping cycles of TRIALS_PER_DATAPOINT
# trials. A Welch's t-test compares stim-condition cycles against control cycles,
# and the colour encodes how much of the control bias was eliminated by stim:
#   blue (~0)  -> bias unchanged from control
#   red  (~1)  -> bias fully eliminated
#   negative   -> bias enhanced by stim (rare, but possible)
print("  Generating bias reduction heatmap...")

# Compute bias values per cycle for each condition
bias_vals_by_condition = {cond: np.array([]) for cond in range(NUM_STIM_LOCATIONS + 1)}

for condition in range(NUM_STIM_LOCATIONS + 1):
    # Filter for low-contrast trials in each block
    data_Lblock = [t for t in condition_data[condition]
                   if abs(t['contrast']) < LOW_CONTRAST_THRESHOLD
                   and t['probabilityLeft'] == 0.8]
    data_Rblock = [t for t in condition_data[condition]
                   if abs(t['contrast']) < LOW_CONTRAST_THRESHOLD
                   and t['probabilityLeft'] == 0.2]

    # Determine number of complete cycles
    num_cycles = min(len(data_Lblock), len(data_Rblock)) // TRIALS_PER_DATAPOINT
    if num_cycles == 0:
        continue

    bias_vals = np.full(num_cycles, np.nan)
    for k in range(num_cycles):
        s, e = k * TRIALS_PER_DATAPOINT, (k + 1) * TRIALS_PER_DATAPOINT
        choices_L = [t['choice'] for t in data_Lblock[s:e]]
        choices_R = [t['choice'] for t in data_Rblock[s:e]]
        bias_vals[k] = np.mean(choices_L) - np.mean(choices_R)

    bias_vals_by_condition[condition] = bias_vals

# Control bias statistics (for normalization)
ctrl_bias_vals = bias_vals_by_condition[0]
ctrl_mean_bias = np.mean(ctrl_bias_vals) if len(ctrl_bias_vals) > 0 else 0
ctrl_abs_mean_bias = np.abs(ctrl_mean_bias)

# Compute reduction and p-value for each stim condition vs control
bias_reduction = {}
bias_pvals = {}
for cond in range(1, NUM_STIM_LOCATIONS + 1):
    stim_bias_vals = bias_vals_by_condition[cond]
    if len(stim_bias_vals) == 0 or len(ctrl_bias_vals) == 0:
        continue

    # Welch's t-test: stim cycles vs control cycles
    _, p_val = stats.ttest_ind(ctrl_bias_vals, stim_bias_vals, equal_var=False)
    bias_pvals[cond] = p_val

    # Reduction = (|ctrl| − |stim|) / |ctrl|
    stim_mean_bias = np.mean(stim_bias_vals)
    if ctrl_abs_mean_bias > 0:
        bias_reduction[cond] = (ctrl_abs_mean_bias - np.abs(stim_mean_bias)) / ctrl_abs_mean_bias
    else:
        bias_reduction[cond] = 0.0

# Multiple-comparisons correction across the 52 stim conditions.
bias_pvals_raw = dict(bias_pvals)
bias_pvals = correct_pvals_dict(bias_pvals_raw, method=MULTIPLE_COMPARISONS_CORRECTION)
if MULTIPLE_COMPARISONS_CORRECTION:
    print(f"  Bias p-values corrected via '{MULTIPLE_COMPARISONS_CORRECTION}'")

# Plot
fig, ax = plt.subplots(figsize=(10, 8))
ax.imshow(dorsal_mip_with_borders, cmap=BRAIN_CMAP, extent=extent)

# Colormap: blue = no reduction, red = full reduction
cmap_bias = plt.get_cmap('coolwarm')

# Use config-driven range so the colormap can be tuned per-dataset.
# Values outside [vmin, vmax] saturate at the endpoints.
norm_bias = mcolors.Normalize(vmin=BIAS_REDUCTION_VMIN, vmax=BIAS_REDUCTION_VMAX)

for condition, coords in stim_locations.items():
    if condition not in bias_pvals:
        continue

    reduction = bias_reduction.get(condition, 0)
    p_val = bias_pvals[condition]
    color = cmap_bias(norm_bias(reduction))

    # Size scales with significance, matching RT/QP convention
    size = -100 * np.log10(p_val) if p_val > 0 else 300
    alpha = 0.5 if p_val >= 0.05 else 1.0

    ml_left = coords.get('ML_left')
    ml_right = coords.get('ML_right')
    ap = coords.get('AP')

    if ml_left is not None and ml_right is not None and ap is not None:
        ax.scatter(ml_left,  ap, color=color, alpha=alpha, s=size, edgecolors=AXIS_TEXT_COLOR)
        ax.scatter(ml_right, ap, color=color, alpha=alpha, s=size, edgecolors=AXIS_TEXT_COLOR)

# Axis limits (same convention as RT plot)
if PLOT_AP_LIMITS is not None:
    ax.set_ylim(bottom=PLOT_AP_LIMITS[0], top=PLOT_AP_LIMITS[1])
if PLOT_ML_LIMITS is not None:
    ax.set_xlim(left=PLOT_ML_LIMITS[0], right=PLOT_ML_LIMITS[1])
elif SHOW_RIGHT_HEMISPHERE_ONLY:
    ax.set_xlim(left=0)

ax.set_xlabel('Mediolateral Position (mm from Bregma)', fontsize=14)
ax.set_ylabel('Anteroposterior Position (mm from Bregma)', fontsize=14)
ax.set_title(
    f'Block Bias Reduction by Stimulation Location\n'
    f'({num_analyzed_sessions} sessions, {num_unique_mice} mice)',
    fontsize=14,
)
ax.tick_params(axis='both', which='major', labelsize=12)

# P-value legend
p_values_legend = [0.001, 0.05, 0.2]
sizes_legend = [-100 * np.log10(p) for p in p_values_legend]
for p, s in zip(p_values_legend, sizes_legend):
    ax.scatter([], [], s=s, label=f'p = {p}',
               edgecolors=AXIS_TEXT_COLOR, color=AXIS_TEXT_COLOR)
legend_bias = ax.legend(loc='upper right', labelspacing=1.5, fontsize=10)

# Colorbar
sm_bias = plt.cm.ScalarMappable(cmap=cmap_bias, norm=norm_bias)
sm_bias.set_array([])
cbar = plt.colorbar(sm_bias, ax=ax, orientation='vertical',
                    fraction=0.046, pad=0.04, aspect=12)
cbar.set_label('Bias reduction (fraction of control)', fontsize=14, labelpad=15)
cbar.ax.tick_params(labelsize=12)

# Apply BRAIN_BACKGROUND_COLOR styling
_style_brain_axes(fig, ax, cbar=cbar, legend=legend_bias)

plt.tight_layout()

if SAVE_FIGURES:
    plt.savefig(FIGURE_SAVE_PATH / f'{FIGURE_PREFIX}_bias_reduction_heatmap.png',
                dpi=150, bbox_inches='tight',
                facecolor=BRAIN_BACKGROUND_COLOR)
    plt.close()
else:
    plt.show()


# -----------------------------------------------------------------------------
# Figure 4: Bias Heatmap (P-value based) — TEMPORARILY DISABLED
# -----------------------------------------------------------------------------
# This is an alternative quantification of block-bias modulation, where colour
# encodes -log10(p-value) for the difference between L-block and R-block choices
# *within* each condition (rather than vs control). It is currently disabled
# because the analysis isn't behaving as expected; revisit later.
#
# To re-enable, un-indent / un-comment the block below.
#
# print("  Generating bias p-value heatmap...")
#
# # Compute p-values for block bias difference at low contrasts
# p_values = {}
# choices_by_condition = {cond: {'R_block': [], 'L_block': []} for cond in condition_data}
#
# for cond, trials_list in condition_data.items():
#     for trial in trials_list:
#         if trial['contrast'] in BIAS_HEATMAP_CONTRASTS:
#             block_type = 'R_block' if trial['probabilityLeft'] == 0.2 else 'L_block'
#             choices_by_condition[cond][block_type].append(trial['choice'])
#
# for cond, blocks in choices_by_condition.items():
#     if blocks['R_block'] and blocks['L_block']:
#         _, p_val = stats.ttest_ind(blocks['R_block'], blocks['L_block'])
#         p_values[cond] = p_val
#     else:
#         p_values[cond] = np.nan
#
# fig, ax = plt.subplots(figsize=(10, 8))
# ax.imshow(dorsal_mip_with_borders, cmap=BRAIN_CMAP, extent=extent)
#
# cmap = plt.get_cmap('coolwarm_r')
# log_p_values = {c: -np.log10(p) if p > 0 and not np.isnan(p) else 0
#                 for c, p in p_values.items()}
# min_log_p, max_log_p = 0.7, 9.5
# norm = mcolors.Normalize(vmin=min_log_p, vmax=max_log_p)
#
# for condition, coords in stim_locations.items():
#     if condition not in p_values or np.isnan(p_values.get(condition, np.nan)):
#         continue
#     log_p = log_p_values.get(condition, 0)
#     color = cmap(norm(log_p))
#     p_val = p_values[condition]
#     alpha = 0.5 if p_val >= 0.05 else 1.0
#     ml_left = coords.get('ML_left')
#     ml_right = coords.get('ML_right')
#     ap = coords.get('AP')
#     if ml_left is not None and ml_right is not None and ap is not None:
#         ax.scatter(ml_left,  ap, color=color, alpha=alpha, s=100, edgecolors=AXIS_TEXT_COLOR)
#         ax.scatter(ml_right, ap, color=color, alpha=alpha, s=100, edgecolors=AXIS_TEXT_COLOR)
#
# sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
# sm.set_array([])
# cbar = plt.colorbar(sm, ax=ax, orientation='vertical', fraction=0.046, pad=0.04, aspect=12)
# cbar.set_label('-log10(P-value)', fontsize=14, labelpad=15)
# cbar.ax.tick_params(labelsize=12)
#
# ax.set_title(f'Block Bias Difference by Location\n'
#              f'({num_analyzed_sessions} sessions, {num_unique_mice} mice)', fontsize=14)
# ax.set_xlabel('Mediolateral Position (mm from Bregma)', fontsize=14)
# ax.set_ylabel('Anteroposterior Position (mm from Bregma)', fontsize=14)
# ax.tick_params(axis='both', which='major', labelsize=12)
#
# if PLOT_AP_LIMITS is not None:
#     ax.set_ylim(bottom=PLOT_AP_LIMITS[0], top=PLOT_AP_LIMITS[1])
# if PLOT_ML_LIMITS is not None:
#     ax.set_xlim(left=PLOT_ML_LIMITS[0], right=PLOT_ML_LIMITS[1])
# elif SHOW_RIGHT_HEMISPHERE_ONLY:
#     ax.set_xlim(left=0)
#
# _style_brain_axes(fig, ax, cbar=cbar)
# plt.tight_layout()
#
# if SAVE_FIGURES:
#     plt.savefig(FIGURE_SAVE_PATH / f'{FIGURE_PREFIX}_bias_pval_heatmap.png',
#                 dpi=150, bbox_inches='tight',
#                 facecolor=BRAIN_BACKGROUND_COLOR)
#     plt.close()
# else:
#     plt.show()

# -----------------------------------------------------------------------------
# Figure: Lapse Rate Heatmap on Brain Atlas
# -----------------------------------------------------------------------------
# Lapse rate = error fraction at high (easy) contrasts. Tests whether stim at
# each location increases the error rate vs control on trials where the correct
# answer is unambiguous. Uses a two-proportion z-test (binomial-appropriate),
# but size encoding and overall layout match RT/QP for visual consistency.
#
# Note on GLM-HMM: when USE_GLMHMM is True both stim and control trials are
# filtered identically (engaged-only), so the *difference* in lapse rate stays
# interpretable. The caveat is that stim-induced disengagement is partly
# censored — a site whose effect is "knocks the mouse out of engaged state"
# will look attenuated here. For a sanity check on any site of interest,
# re-run with USE_GLMHMM = False.
print("  Generating lapse rate heatmap...")
 
# What counts as "high contrast" (i.e. an easy trial where errors = lapses)
LAPSE_CONTRASTS = [-100.0, -25.0, 25.0, 100.0]
 
# Tally errors and total high-contrast trials per condition.
# A "correct" trial is one where choice sign matches the contrast sign:
#   contrast > 0  -> stimulus on right -> correct choice = -1 (rightward)
#   contrast < 0  -> stimulus on left  -> correct choice = +1 (leftward)
def _is_high_contrast_error(trial):
    c = trial['contrast']
    ch = trial['choice']
    if np.isnan(c) or np.isnan(ch):
        return None
    if abs(c) not in [abs(x) for x in LAPSE_CONTRASTS]:
        return None
    if ch == 0:  # no-go, if encoded that way
        return None
    correct_choice = -1 if c > 0 else 1
    return ch != correct_choice  # True = lapse/error
 
lapse_counts = {}  # cond -> (n_errors, n_total)
for cond in range(NUM_STIM_LOCATIONS + 1):
    n_err, n_tot = 0, 0
    for t in condition_data[cond]:
        result = _is_high_contrast_error(t)
        if result is None:
            continue
        n_tot += 1
        if result:
            n_err += 1
    lapse_counts[cond] = (n_err, n_tot)
 
ctrl_err, ctrl_tot = lapse_counts[0]
ctrl_lapse_rate = (ctrl_err / ctrl_tot) if ctrl_tot > 0 else 0.0
print(f"  Control lapse rate: {ctrl_err}/{ctrl_tot} = {ctrl_lapse_rate:.3f}")
 
# Compute per-condition lapse rate and p-value (two-proportion z-test vs control)
lapse_means = {}
lapse_pvals = {}
for cond in range(1, NUM_STIM_LOCATIONS + 1):
    stim_err, stim_tot = lapse_counts[cond]
    if stim_tot < 5 or ctrl_tot < 5:
        continue  # not enough trials for a meaningful test
    lapse_means[cond] = stim_err / stim_tot
    counts = np.array([stim_err, ctrl_err])
    nobs = np.array([stim_tot, ctrl_tot])
    try:
        _, p_val = proportions_ztest(counts, nobs)
        lapse_pvals[cond] = p_val if not np.isnan(p_val) else 1.0
    except Exception:
        lapse_pvals[cond] = 1.0

# Multiple-comparisons correction across the 52 stim conditions.
lapse_pvals_raw = dict(lapse_pvals)
lapse_pvals = correct_pvals_dict(lapse_pvals_raw, method=MULTIPLE_COMPARISONS_CORRECTION)
if MULTIPLE_COMPARISONS_CORRECTION:
    print(f"  Lapse p-values corrected via '{MULTIPLE_COMPARISONS_CORRECTION}'")

# --- Plot ---
fig, ax = plt.subplots(figsize=(10, 8))
ax.imshow(dorsal_mip_with_borders, cmap=BRAIN_CMAP, extent=extent)
 
# Colormap centred on control lapse rate. Range = ctrl ± max(0.05, ctrl*1.0),
# so even a low-lapse mouse (e.g. ctrl=2%) still gets a visible spread.
half_range = max(0.05, ctrl_lapse_rate * 1.0)
norm_lapse = mcolors.Normalize(
    vmin=max(0.0, ctrl_lapse_rate - half_range),
    vmax=ctrl_lapse_rate + half_range,
)
sm_lapse = plt.cm.ScalarMappable(cmap='coolwarm', norm=norm_lapse)
 
for condition, coords in stim_locations.items():
    if condition not in lapse_means:
        continue
    rate = lapse_means[condition]
    p_val = lapse_pvals.get(condition, 1.0)
 
    size = -100 * np.log10(p_val) if p_val > 0 else 300
    color = sm_lapse.to_rgba(rate)
    alpha = 0.5 if p_val >= 0.05 else 1.0
 
    ml_left = coords.get('ML_left')
    ml_right = coords.get('ML_right')
    ap = coords.get('AP')
    if ml_left is not None and ml_right is not None and ap is not None:
        ax.scatter(ml_left,  ap, color=color, alpha=alpha, s=size, edgecolors=AXIS_TEXT_COLOR)
        ax.scatter(ml_right, ap, color=color, alpha=alpha, s=size, edgecolors=AXIS_TEXT_COLOR)
 
# Axis limits (same convention as other heatmaps)
if PLOT_AP_LIMITS is not None:
    ax.set_ylim(bottom=PLOT_AP_LIMITS[0], top=PLOT_AP_LIMITS[1])
if PLOT_ML_LIMITS is not None:
    ax.set_xlim(left=PLOT_ML_LIMITS[0], right=PLOT_ML_LIMITS[1])
elif SHOW_RIGHT_HEMISPHERE_ONLY:
    ax.set_xlim(left=0)
 
ax.set_xlabel('Mediolateral Position (mm from Bregma)', fontsize=14)
ax.set_ylabel('Anteroposterior Position (mm from Bregma)', fontsize=14)
ax.set_title(
    f'Lapse Rate by Stimulation Location\n'
    f'({num_analyzed_sessions} sessions, {num_unique_mice} mice)',
    fontsize=14,
)
ax.tick_params(axis='both', which='major', labelsize=12)
 
# P-value legend
p_values_legend = [0.001, 0.05, 0.2]
for p in p_values_legend:
    s = -100 * np.log10(p)
    ax.scatter([], [], s=s, label=f'p = {p}',
               edgecolors=AXIS_TEXT_COLOR, color=AXIS_TEXT_COLOR)
legend_lapse = ax.legend(loc='upper right', labelspacing=1.5, fontsize=10)
 
# Colorbar
cbar = plt.colorbar(sm_lapse, ax=ax, orientation='vertical',
                    fraction=0.046, pad=0.04, aspect=12)
cbar.set_label(f'Lapse rate at high contrast\n(control = {ctrl_lapse_rate:.3f})',
               fontsize=12, labelpad=15)
cbar.ax.tick_params(labelsize=12)
 
# Apply BRAIN_BACKGROUND_COLOR styling
_style_brain_axes(fig, ax, cbar=cbar, legend=legend_lapse)
 
plt.tight_layout()
 
if SAVE_FIGURES:
    plt.savefig(FIGURE_SAVE_PATH / f'{FIGURE_PREFIX}_lapse_heatmap.png',
                dpi=150, bbox_inches='tight',
                facecolor=BRAIN_BACKGROUND_COLOR)
    plt.close()
else:
    plt.show()


# =============================================================================
# SAVE RESULTS
# =============================================================================

if SAVE_FIGURES:
    print("\nSaving analysis results...")
    
    # Save bias arrays for further analysis
    np.save(FIGURE_SAVE_PATH / f'{FIGURE_PREFIX}_bias_vals_LC_control.npy', bias_vals_LC[0])
    
    # Save effect sizes and statistics.
    # For each metric we save BOTH the raw (uncorrected) p-value and the
    # corrected p-value. The columns named '{metric}_pval' contain the
    # corrected values that drive the heatmap circle sizes; '{metric}_pval_raw'
    # contains the uncorrected values. The correction method is in config.py
    # (MULTIPLE_COMPARISONS_CORRECTION). When correction is None, the two
    # columns are identical.
    effect_size_df = pd.DataFrame([
        {'condition': c,
         'effect_size': effect_sizes.get(c, np.nan),
         # RT
         'rt_mean': rt_means.get(c, np.nan),
         'rt_pval': rt_pvals.get(c, np.nan),
         'rt_pval_raw': rt_pvals_raw.get(c, np.nan),
         # QP
         'qp_mean': qp_means.get(c, np.nan),
         'qp_pval': qp_pvals.get(c, np.nan),
         'qp_pval_raw': qp_pvals_raw.get(c, np.nan),
         # Bias reduction
         'bias_reduction': bias_reduction.get(c, np.nan),
         'bias_pval': bias_pvals.get(c, np.nan),
         'bias_pval_raw': bias_pvals_raw.get(c, np.nan),
         'bias_pval_ind': cycle_stats['ttest_ind'].get(c, np.nan),
         # Lapse rate (high-contrast error rate)
         'lapse_rate': lapse_means.get(c, np.nan),
         'lapse_pval': lapse_pvals.get(c, np.nan),
         'lapse_pval_raw': lapse_pvals_raw.get(c, np.nan)}
        for c in range(1, NUM_STIM_LOCATIONS + 1)
    ])
    effect_size_df.to_csv(FIGURE_SAVE_PATH / f'{FIGURE_PREFIX}_results.csv', index=False)
    
    print(f"Results saved to {FIGURE_SAVE_PATH}")


print("\n" + "="*60)
print("Analysis complete!")
print("="*60)
