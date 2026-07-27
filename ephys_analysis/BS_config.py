"""
BS_config.py
============
Configuration for SNr_inhibition_BS_downstream_effect.py

All user-tunable options live here so the main analysis script does not
need to be edited between runs. Contents:

    1. Run-level options       (which condition, hemisphere, alignment,
                                output prefix, start index)
    2. PETH parameters         (windows, bin/smoothing, post-smoothing)
    3. Cluster quality filters (IBL label, min firing rate)
    4. Plotting options        (per-cluster plotting toggles)
    5. CONDITIONS              (registry: condition -> hemisphere -> trial-list data)
    6. TRIALS_TO_REMOVE        (per-PID trial exclusions)
    7. resolve_condition()     (helper: pulls flat lists + pid->hemisphere
                                lookup out of CONDITIONS for the main loop)
"""

from metadata_optostim import (
    pids_list_SNr_trained,
    pids_list_SNr_contra_trained,
    excitation_trials_range_list_SNr_trained,
    inhibition_trials_range_list_SNr_trained,
    excitation_trials_range_list_SNr_contra_trained,
    inhibition_trials_range_list_SNr_contra_trained,
    light_artifact_units_SNr_trained,
    light_artifact_units_SNr_contra_trained,
    pids_list_ZI_trained,
    pids_list_ZI_trained_contra,
    excitation_trials_range_list_ZI_trained,
    inhibition_trials_range_list_ZI_trained,
    excitation_trials_range_list_ZI_trained_contra,
    inhibition_trials_range_list_ZI_trained_contra,
    light_artifact_units_ZI_trained,
    light_artifact_units_ZI_trained_contra,
    pids_list_SNr_reverse,
    excitation_trials_range_list_SNr_reverse,
    inhibition_trials_range_list_SNr_reverse,
    light_artifact_units_SNr_reverse,
    pids_list_STN_ipsi,
    pids_list_STN_contra,
    excitation_trials_range_list_STN_ipsi,
    inhibition_trials_range_list_STN_ipsi,
    excitation_trials_range_list_STN_contra,
    inhibition_trials_range_list_STN_contra,
    light_artifact_units_STN_ipsi,
    light_artifact_units_STN_contra,
)

# =====================================================================
# 1. Run-level options
# =====================================================================
condition = 'Laser onset'   # one of CONDITIONS.keys() below
hemisphere_filter = 'both'        # 'both' | 'ipsi' | 'contra' | 'reverse'
                                  # 'both' pools every hemisphere defined
                                  # for the chosen condition; otherwise
                                  # only the named hemisphere is loaded.
onset_alignment = 'Laser onset'   # legacy/primary alignment used for the back-compat output path.
onset_alignments_to_run = ('Laser onset', 'Feedback')
figure_prefix = 'GLMHMM_crossfit'
start_pid_idx = 0                 # index into pids list to start from (0 = full run)

# =====================================================================
# 2. PETH parameters
# =====================================================================
t_before = 5
t_after = 10
bin_size = 0.05
smoothing = 0.05
post_smooth_window_ms = 300       # rolling-window smoothing for group mean Δ-FR trace

# Per-alignment PETH windows, seconds before/after the alignment event. These
# control saved trace length for each output pickle. If an alignment is missing
# here, the script falls back to t_before/t_after above.
alignment_time_windows = {
    'Laser onset': (t_before, t_after),
    'Go cue onset': (5, 2),
    'Feedback': (5, 2),
}

# =====================================================================
# 3. Cluster quality filters
# =====================================================================
# IBL quality labels are discrete: {0, 1/3, 2/3, 1} depending on how many
# of 3 quality metrics were passed. A threshold of 0.6 keeps {2/3, 1}.
IBL_quality_label_threshold = 0
firing_rate_threshold = 1         # Hz

# =====================================================================
# 4. Plotting options
# =====================================================================
plot_each_cluster = 0
plot_only_BS_units = 0
only_plot_FR = 0

# =====================================================================
# 5. Condition registry
# =====================================================================
# Each condition maps to a dict of hemisphere -> trial-list-bundle.
# Hemisphere labels are 'ipsi', 'contra', or 'reverse' (SNr_reverse only).
# The four parallel lists inside each bundle are aligned by session.
CONDITIONS = {
    'SNr_forBSanalysis': {
        'ipsi': dict(
            pids                          = pids_list_SNr_trained,
            excitation_trials_range_list  = excitation_trials_range_list_SNr_trained,
            inhibition_trials_range_list  = inhibition_trials_range_list_SNr_trained,
            light_artifact_units_list     = light_artifact_units_SNr_trained,
        ),
        'contra': dict(
            pids                          = pids_list_SNr_contra_trained,
            excitation_trials_range_list  = excitation_trials_range_list_SNr_contra_trained,
            inhibition_trials_range_list  = inhibition_trials_range_list_SNr_contra_trained,
            light_artifact_units_list     = light_artifact_units_SNr_contra_trained,
        ),
    },
    'ZI_forBSanalysis': {
        'ipsi': dict(
            pids                          = pids_list_ZI_trained,
            excitation_trials_range_list  = excitation_trials_range_list_ZI_trained,
            inhibition_trials_range_list  = inhibition_trials_range_list_ZI_trained,
            light_artifact_units_list     = light_artifact_units_ZI_trained,
        ),
        'contra': dict(
            pids                          = pids_list_ZI_trained_contra,
            excitation_trials_range_list  = excitation_trials_range_list_ZI_trained_contra,
            inhibition_trials_range_list  = inhibition_trials_range_list_ZI_trained_contra,
            light_artifact_units_list     = light_artifact_units_ZI_trained_contra,
        ),
    },
    'SNr_reverse': {
        'reverse': dict(
            pids                          = pids_list_SNr_reverse,
            excitation_trials_range_list  = excitation_trials_range_list_SNr_reverse,
            inhibition_trials_range_list  = inhibition_trials_range_list_SNr_reverse,
            light_artifact_units_list     = light_artifact_units_SNr_reverse,
        ),
    },
    'STN_forBSanalysis': {
        'ipsi': dict(
            pids                          = pids_list_STN_ipsi,
            excitation_trials_range_list  = excitation_trials_range_list_STN_ipsi,
            inhibition_trials_range_list  = inhibition_trials_range_list_STN_ipsi,
            light_artifact_units_list     = light_artifact_units_STN_ipsi,
        ),
        'contra': dict(
            pids                          = pids_list_STN_contra,
            excitation_trials_range_list  = excitation_trials_range_list_STN_contra,
            inhibition_trials_range_list  = inhibition_trials_range_list_STN_contra,
            light_artifact_units_list     = light_artifact_units_STN_contra,
        ),
    },
}

# =====================================================================
# 6. Per-PID trial exclusions
# =====================================================================
# Trial numbers to drop from all trial-index arrays (excitation,
# inhibition, nonstim, nonstim_ex, nonstim_in) for specific PIDs.
TRIALS_TO_REMOVE = {
    'e44cb3ae-d436-4149-9110-415a276fb58e': [
        6, 9, 10, 11, 15, 17, 23, 26, 28, 30, 31, 32, 39, 41, 44, 46, 50,
        51, 53, 54, 63, 64, 70, 71, 72, 74, 75, 76, 77, 78, 81, 82, 83,
        84, 85, 86, 89, 91, 92, 95, 96, 97, 98, 99, 100, 101, 102, 104,
        105, 106, 107, 108, 111, 113, 114, 117, 118, 119, 123, 125, 126,
        130, 131, 133, 134, 135, 136, 137, 139, 140, 143, 144, 145, 146,
        149, 150, 156, 157, 166, 169, 170, 171, 172, 173, 175, 176, 178,
        179, 183, 190, 191, 192, 193, 194, 195, 197, 198, 204,
    ],
    'bfa8f605-2eda-4b31-80fb-4a889fa0e22a': [
        6, 7, 10, 17, 18, 19, 20, 21, 22, 23, 31, 34, 35, 36, 45, 47, 49,
        50, 51, 52, 53, 57, 58, 59, 60, 63, 72, 73, 74, 75, 79, 80, 84,
        87, 88, 89, 93, 94, 95, 97, 99, 100, 103, 104, 108, 109, 110,
        112, 114, 115, 116, 117, 120, 123, 124, 126, 127, 128, 129, 130,
        135, 142, 143, 145, 146, 147, 149, 150, 154, 155, 159, 160, 165,
        166, 167, 168, 169, 170, 177, 179, 182, 183, 184, 185, 188, 189,
        191, 192, 193, 196, 202, 204, 206, 207, 209, 211, 212, 215, 219,
        225, 227, 232, 234, 235, 237, 238, 239, 240, 244, 249, 250, 251,
        252, 253, 256, 260, 262, 266, 268, 270, 271, 272, 275, 277, 280,
        281, 282, 283, 284, 285, 288, 290, 294, 303, 308, 310, 313, 317,
        318, 321, 326, 334, 336, 342, 343, 344, 345, 352, 353, 357, 363,
        365,
    ],
    '9fedd1c6-33eb-48b6-b508-8deebe3ee44c': [
        1, 2, 3, 20, 32, 33, 35, 36, 37, 42, 46, 47, 48, 57, 61, 64, 82,
        83, 84, 85, 86, 87, 90, 95, 104, 106, 109, 110, 116, 117, 124,
        129, 130, 131, 132, 136, 138, 140, 141, 142, 149, 153, 156, 158,
    ],
}


# =====================================================================
# 6b. Metadata-driven insertion selection (new pipeline)
# =====================================================================
# Mirrors CD: drive the run off metadata_optostim_new.insertions instead of
# the legacy CONDITIONS registry. Set analysis_mode = 'legacy_condition' to
# fall back to resolve_condition().
analysis_mode = 'all_insertions'      # 'all_insertions' | 'legacy_condition'
# Run across ALL insertions/regions/conditions and record region+condition as
# per-unit fields, so brain-region / hemisphere selection becomes a post-hoc
# filter (see BS_postprocess.py) rather than a reason to re-run. Set these to
# restrict the run if you really want to.
insertion_brain_regions = None        # None = all (SNr+ZI+STN); or e.g. ['SNr']
insertion_conditions = None           # None = ipsi+contra; or e.g. ['ipsi']
insertion_pids = None

# ---- 'Compute everything' gates -------------------------------------------
# BS scoring (the permutation test) is the expensive step, so we still gate
# WHICH units get scored. Everything that survives these gates is scored and
# fully characterised; all other exclusion criteria are recorded as fields
# and applied post-hoc. Loosen these (and re-run) only if you need to.
compute_recorded_region = 'midbrain'  # 'midbrain' (recorded region of interest;
                                      #  specific Allen regions still sub-filter
                                      #  post-hoc) or 'all' (score every region).
compute_min_IBL_label = 0/3           # score units with IBL label >= this floor;
                                      #  IBL_quality_label_threshold is applied
                                      #  post-hoc and can be raised freely.
compute_min_firing_rate = 0         # Hz; skip ~silent units. firing_rate is
                                      #  stored so it can be filtered post-hoc.
bs_output_path = '~/python/saved_figures/BS_all_insertions_' + figure_prefix + '.pkl'

# =====================================================================
# 6c. Shared unit-QC cascade (mirrors CD_config; every stage toggleable)
# =====================================================================
# IBL_quality_label_threshold and firing_rate_threshold are defined above.
# firing_rate_threshold is a BS-specific extra gate (CD does not apply it);
# set it to 0 to match CD's unit set exactly.
presence_threshold = 0

remove_light_artifact_units        = 1
remove_waveform_amplitude_outliers = 1
remove_axonal_units                = 0
remove_drift_units                 = 0

# Automatic laser-locked light-artifact detector
light_artifact_window_s                   = (0.000, 0.005)
light_artifact_baseline_window_s          = (-0.050, -0.005)
light_artifact_z_threshold                = 8.0
light_artifact_min_event_fraction         = 0.20
light_artifact_min_excess_spikes_per_event = 0.05

# Waveform-amplitude outlier QC
waveform_amplitude_low_percentile  = 0.5
waveform_amplitude_high_percentile = 99.5

# Axonal classification
axonal_pt_ratio_threshold = 1.0

# Region selection. DEPTH_THRESHOLD_OVERRIDES is imported from CD_config so the
# two pipelines apply identical manual depth thresholds for PIDs lacking
# histology (single source of truth).
analyze_region = 'midbrain'           # 'midbrain' | 'isocortex'
from CD_config import DEPTH_THRESHOLD_OVERRIDES

# Drift removal. Computed over the beginning-block-filtered admissible trial
# range, aligned to trial start, mean FR per trial vs trial-index/block.
# *** drift_window_s is the one drift parameter worth confirming for BS. ***
drift_threshold = 0.35
# Epoch used to estimate drift:
#   'quiescence'  : mean FR in each trial's enforced quiescence period
#                   [goCue - quiescencePeriod, goCue] (guaranteed motor-free). DEFAULT.
#   'fixed_window': legacy fixed window about trial start (uses drift_window_s).
drift_epoch = 'quiescence'
drift_window_s  = (0.0, 2.0)          # (t_before, t_after) s; only used if drift_epoch=='fixed_window'

# QP firing-rate nonstationarity filter. Shared with CD_config/CD pipeline.
# This is designed for non-monotonic drifts/dropouts: QP firing is measured per
# trial, block-specific medians are subtracted, and contiguous session segments
# are compared. Defaults are report-only: metrics are saved to the output pickle
# and QC CSV/PNG files, but no units are removed unless thresholds are set and
# remove_nonstationary_units=1.
remove_nonstationary_units = 0
nonstationarity_n_segments = 6
nonstationarity_min_trials = 30
nonstationarity_min_trials_per_segment = 8
nonstationarity_min_trials_per_block_segment = 3
nonstationarity_low_fr_fraction_of_median = 0.2
nonstationarity_min_median_fr_hz = 0.1
max_qp_fr_segment_range_frac = None          # raw segment FR range / median FR
max_qp_resid_drift_range_frac = None         # block-residual segment FR range / median FR
max_qp_resid_drift_cv = None                 # block-residual segment mean SD / median FR
max_qp_resid_abs_rho_time = None             # abs(Spearman(block-residual FR, trial order))
max_qp_low_activity_fraction = None          # fraction of QP trials below low_fr_fraction*median FR
max_qp_max_low_activity_run = None           # longest consecutive low-activity run in QP trials
min_qp_block_effect_sign_consistency = None  # fraction of valid segments with same block-effect sign
max_qp_block_effect_segment_cv = None        # segment block-effect SD / global block effect
max_qp_block_effect_dominance = None         # largest segment |effect| / sum segment |effects|

# =====================================================================
# 6d. Trial-QC additions (shared criteria with CD)
# =====================================================================
beginning_block_trials_remove = 5    # also used for the drift trial set
use_GLMHMM_engaged_indices    = 1
# When use_GLMHMM_engaged_indices == 1, controls/nonstim trials are always
# restricted to their own engaged-state trials. This option controls opto trials:
#   'standard'    : keep only opto trials whose own GLM-HMM state is engaged
#   'bypass'      : keep all opto trials regardless of GLM-HMM state
#   'prior state' : use the most recent previous non-opto trial state for each
#                   opto trial, so opto-evoked state changes cannot remove it
opto_trials_GLMHMM = 'prior state'

# BS trial definition. This controls which trials are allowed into the BS
# permutation test and the saved 80/20 delta-PETH traces.
#   'standard'
#       Keep the usual post-QC stim/nonstim trial sets.
#   'prior_choice_congruent'
#       Keep only trials where the animal's choice is in the direction favored
#       by the current block prior: left choices in p(left)>0.5 blocks and right
#       choices in p(left)<0.5 blocks. Neutral p(left)==0.5 trials and invalid
#       choices are excluded.
bs_definition_trial_mode = 'standard'

# Matched control baseline for the delta-FR PETHs. When 1, the nonstim (control)
# 80/20 delta and its all-trial normalizer are computed only from nonstim trials
# WITHIN inhibition_trials_range, so control and opto deltas share a session
# window and baseline (removes the constant pre-laser offset). The BS *score*
# (perm test) always keeps the broad session-wide nonstim set for power, so this
# toggle does not change which units are flagged bias-selective -- only the
# plotted/quantified delta-FR traces. Set 0 to recover the old broad-control
# behavior (useful for an A/B comparison).
match_nonstim_to_inhibition_range = 1

# Delta-FR normalization. ALL THREE modes are now computed and saved in parallel
# on the same units every run; normalize_mode only sets which one the back-compat
# 'trace_nonstim'/'trace_stim' keys mirror. In post-processing switch freely with
# BS_postprocess.use_norm(data, 'per_bin' | 'baseline_scalar' | 'zero_2_nan').
#   'per_bin'         : delta / per-bin all-trial PETH, 0->0.1 floor (original).
#   'baseline_scalar' : delta / single floored baseline scalar (robust to the
#                       pre-laser dead zone).
#   'zero_2_nan'      : 0-FR bins treated as missing in the block means and the
#                       normalizer (diagnostic for the zero-bin/floor artifact).
# Only the delta-FR traces differ; the BS score and z-score are unaffected.
normalize_mode = 'per_bin'
scalar_baseline_window = None    # None -> whole-window mean FR (robust, avoids the pre-laser dead zone);
                                 # or e.g. (-5.0, -2.0) for a specific pre-laser window away from onset.
scalar_min_fr = 0.5              # Hz floor on the normalizer scalar so a near-zero baseline can't explode the delta.
zero_nan_threshold = 0.1         # Hz: in 'zero_2_nan', bins with mean FR <= this are treated as empty (NaN).
                                 # Must be > 0 because smoothed PETH "zeros" are tiny positive tails, not exact 0.

# Extra saved traces for post-hoc diagnostics and preference-aligned inference.
# These do not change the BS score or the main trace keys. In addition to the
# older random split/trial-count checks, the saved block-crossfit folds learn
# preference from one contiguous half of each bias block and evaluate matched
# control/opto trials in the other half, then swap.
save_diagnostic_traces = 1
save_raw_block_peths = 1          # Saves raw 80/20 block PETH means for later normalization audits; increases pickle size.
diagnostic_random_seed = 20260607 # Stable deterministic split/match seed.
diagnostic_trialmatch_repeats = 1 # 1 is fast and deterministic; increase for subsampling variability estimates.
diagnostic_min_events_per_peth = 2
# For the control-preference cross-fit, each bias-block run is divided into
# contiguous early/late halves. Trials this close to the midpoint are omitted
# from both halves so the sign-training and held-out PETHs do not share spikes
# merely because adjacent trial windows overlap at the split boundary.
diagnostic_crossfit_guard_trials = 2
# Save compact sufficient statistics for future re-analysis without returning to
# spikes: per-trial quiescent FR, exact fold/match trial ids, packed pseudo-block
# labels, and raw cross-fit 80/20/normalizer traces. This is far smaller than a
# unit x trial x time tensor and enables nested BS selection/common normalization.
save_futureproof_sufficient_stats = 1
# Exact PETH acceleration: calculate each event-set PETH for a bounded chunk of
# independent clusters at once. brainbox.calculate_peths produces identical
# per-cluster means/stds in multi-cluster and one-cluster calls; batching avoids
# rescanning the full insertion spike table and rebuilding event bins per unit.
use_batched_peths = 1
peth_cluster_batch_size = 64       # bounds temporary event x cluster x bin arrays
# Multi-alignment runs already write one self-contained pickle per alignment.
# The historical unsuffixed copy and the combined dict duplicate several GB and
# are not needed by BS_postprocess, so keep them off unless explicitly required.
save_legacy_base_pickle = 0
save_combined_alignment_pickle = 0
n_states                      = 2

# =====================================================================
# 6e. QC output
# =====================================================================
save_qc_outputs = 1
save_figures    = 1
figures_path    = '/Users/natemiska/Desktop/bs_figures'


# =====================================================================
# 7. resolve_condition helper
# =====================================================================
def resolve_condition(condition_name, hemisphere_filter='both'):
    """
    Flatten the (condition, hemisphere) registry into the parallel lists
    that the main loop iterates over.

    Parameters
    ----------
    condition_name : str
        Key into CONDITIONS.
    hemisphere_filter : {'both', 'ipsi', 'contra', 'reverse'}
        If 'both', include every hemisphere present in the condition.
        Otherwise include only the named hemisphere (raises ValueError if
        that hemisphere is not defined for the chosen condition).

    Returns
    -------
    pids : list
    excitation_trials_range_list : list
    inhibition_trials_range_list : list
    light_artifact_units_list : list
    pid_to_hemisphere : dict
        Maps each pid in `pids` to its hemisphere label string.
    """
    if condition_name not in CONDITIONS:
        raise KeyError(
            f"condition {condition_name!r} not in CONDITIONS "
            f"(available: {list(CONDITIONS.keys())})"
        )
    cfg = CONDITIONS[condition_name]

    if hemisphere_filter == 'both':
        hemis_to_use = list(cfg.keys())
    elif hemisphere_filter in cfg:
        hemis_to_use = [hemisphere_filter]
    else:
        raise ValueError(
            f"hemisphere_filter={hemisphere_filter!r} not defined for "
            f"condition {condition_name!r} (available hemispheres: "
            f"{list(cfg.keys())})"
        )

    pids = []
    exc, inh, art = [], [], []
    pid_to_hemisphere = {}
    for h in hemis_to_use:
        sub = cfg[h]
        pids.extend(sub['pids'])
        exc.extend(sub['excitation_trials_range_list'])
        inh.extend(sub['inhibition_trials_range_list'])
        art.extend(sub['light_artifact_units_list'])
        for pid in sub['pids']:
            pid_to_hemisphere[pid] = h

    return pids, exc, inh, art, pid_to_hemisphere
