"""
Configuration file for CD_analysis_midbrain.py
================================================
All user-tunable parameters live here. The analysis script imports from
this file via `from CD_config import *`.

To run a different analysis (e.g., restrict to SNr, switch alignments, toggle
drift removal), edit the values below and rerun the analysis script.
"""

# =========================================================================
# Dataset selection
# =========================================================================
# Analyze rows from metadata_optostim_new.insertions. Leave filters as None to
# include all metadata rows, or set one/more filters to restrict the pipeline.
insertion_brain_regions = ['SNr']    # e.g. ['SNr', 'ZI', 'STN']
insertion_conditions = ['ipsi']       # e.g. ['ipsi'] or ['contra']
insertion_pids = None #['f54b959b-fee4-4130-951e-e366d34a5cbc']             # e.g. ['518b61c2-45bc-40c2-bee1-d87b0d1986ac']


# =========================================================================
# Trial alignment
# =========================================================================
# Choose one or more projection alignments. The CD itself is computed once from
# neural activity during the enforced quiescence period (motor-free), then that
# same CD is projected onto each requested alignment:
#   - 'Laser onset': t=0 is laser onset (= trial start = start of QP).
#   - 'Go cue onset': t=0 is go cue.
#   - 'Feedback': t=0 is trials.feedback_times (reward/error/timeout).
# Comparing the two alignments lets you dissociate laser-time-locked transients
# (which should appear in laser-aligned plots but disappear in go-cue-aligned
# plots if they recover before the choice period) from persistent effects
# that survive into the choice epoch.
onset_alignments_to_run = ('Laser onset', 'Feedback')  # use ('Go cue onset',) for Go-only, etc.
alignment_time_windows = {
    'Laser onset': (2.0, 5.0),   # (t_before, t_after)
    'Go cue onset': (5.0, 2.0),
    'Feedback': (5.0, 2.0),
}


# =========================================================================
# Time windows
# =========================================================================
bin_size = 0.05  # seconds per bin

# Which neural epoch to use for computing CD on control trials.
#
#   'strict_qp'       : variable per-trial enforced QP only, anchored to end
#                       at go cue. Mask = [-enforced_qp_len[i], 0) for go cue
#                       alignment, or [0, +enforced_qp_len[i]) for laser
#                       alignment. The cleanest motor-free pre-choice epoch.
#                       RECOMMENDED DEFAULT.
#
#   'full_cd_window'  : the entire QP, from trial onset to go cue
#
#   'full_trial'      : the entire broad trial window. Uses a large X tensor for
#                       CD computation, so the
#                       CD reflects activity across the whole trial including
#                       movement and choice periods. Useful as a sanity check
#                       but NOT recommended as the primary CD source, since
#                       motor-related variance will contaminate the CD.
#.  'ITI'             : uses ITI (anchored to trial onset) for CD calculation

cd_window_mode = 'strict_qp'


# Exclude early trials after each probabilityLeft block transition. Implemented
# in optostim_preprocessing.prepare_trials before CD computation and before
# projection-tensor construction, so removed trials cannot contribute to CD or
# any projection trace. Positions are computed from the full session's
# probabilityLeft sequence; e.g. 5 removes absolute within-block positions 1-5
# even if the metadata trial range starts mid-session.
beginning_block_trials_remove = 1

# Keep only actual biased probability-left blocks for CD/projection analyses.
# The pipeline labels blocks as probabilityLeft > 0.5, so leaving neutral
# probabilityLeft == 0.5 trials in the analysis would silently group them with
# the right-bias/low-prob-left block. Set to None for legacy behavior.
allowed_probability_left_values = (0.2, 0.8)
probability_left_tolerance = 1e-6

# Sanity filter for current-laser-aligned plots: remove opto/stim trials whose
# immediately preceding absolute trial was also opto/stim within the analyzed
# range. Nonstim trials are retained. This is applied in prepare_trials before
# CD/projection binning, and final opto/control counts are recomputed after the
# removal. Keeping it on makes the pre-current-laser window less vulnerable to
# residual effects of the previous trial's laser.
remove_stim_trials_preceded_by_stim = 0

# Optional behavioral timing filters. These are applied to ALL trials in the
# analysis range, before both CD calculation and projection traces. Leave all
# values as None to disable. Reaction time is event_time - goCue_time, where
# event_time is trials[reaction_time_source]; use 'firstMovement_times',
# 'response_times', or 'auto' (firstMovement_times if present, otherwise
# response_times). Non-finite values fail only when the corresponding filter is
# enabled.
min_reaction_time_s = None          # e.g. 0.08 to remove unrealistically fast movement trials
max_reaction_time_s = None#1.5          # e.g. 1.5 to remove very slow/late response trials
reaction_time_source = 'feedback_times'
min_quiescence_period_s = None      # e.g. 0.2 to require enough enforced QP
max_quiescence_period_s = None      # e.g. 1.5 to remove unusually long waiting trials

# Laser alignment QC: for Laser onset alignment, compare each opto trial's
# actual TTL onset from _ibl_laserStimulation.intervals against the alignment
# timestamp used to build X. Saves per-insertion CSV/PNG summaries.
save_laser_alignment_qc = 0
laser_alignment_match_tolerance_s = 0.02

# Which control trials are eligible for defining the CD vector. This mask is
# applied after the global trial cascade above. When
# use_heldout_control_projection=1, this same eligible-control pool is split
# into CD-training controls and held-out controls for projection/statistics.
#
#   'standard'
#       CD is computed from correct nonstim/control trials that passed the
#       global trial cascade.
#
#   'ALL'
#       CD is computed from all nonstim/control trials that passed the global
#       trial cascade, with no additional correctness, choice-congruence, or
#       contrast filtering for CD definition. This is useful as a diagnostic
#       for whether CD-definition behavioral filters are making control and
#       opto projection pools intrinsically different before laser onset.
#
#   'prior_choice_congruent'
#       Hypothesis test for "strong prior-expression" trials: CD is computed from
#       nonstim/control trials where the mouse's CHOICE was congruent with the
#       current block side, regardless of feedback/correctness. This includes
#       incorrect anti-block-stimulus trials when the animal followed the prior,
#       and excludes correct anti-block-choice trials when the visual stimulus
#       drove behavior away from the prior. Contrast is controlled by
#       cd_definition_contrasts_percent below.
#
#   'prior_congruent_low_contrast_correct'
#       Legacy alias for prior_choice_congruent, retained so older config
#       snippets still run. Despite the old name, the current implementation is
#       choice-congruent and does not require feedbackType > 0.
cd_definition_trial_mode = 'prior_choice_congruent'  # 'prior_choice_congruent' #'standard'#

use_GLMHMM_engaged_indices = 0      # restrict trial set to GLM-HMM engaged trials in prepare_trials
# When use_GLMHMM_engaged_indices == 1, controls/nonstim trials are always
# restricted to their own engaged-state trials. This option controls opto trials:
#   'standard'    : keep only opto trials whose own GLM-HMM state is engaged
#   'bypass'      : keep all opto trials regardless of GLM-HMM state
#   'prior state' : use the most recent previous non-opto trial state for each
#                   opto trial, so opto-evoked state changes cannot remove it
opto_trials_GLMHMM = 'prior state'
n_states = 2

# Contrast selector for the prior-congruent CD-definition mode. Use None to
# accept all contrasts; use a tuple such as (0.0, 6.25, 12.5) to restrict the CD
# definition to a subset of absolute visual contrasts.
cd_definition_contrasts_percent = None
# Legacy alias retained so old notebooks/config snippets do not immediately
# break. Prefer editing cd_definition_contrasts_percent above.
cd_definition_low_contrasts_percent = cd_definition_contrasts_percent
cd_definition_contrast_tolerance_percent = 1e-6
# cd_definition_include_ambiguous_zero_contrast = 1  # Deprecated: old stimulus-side CD-definition mode only.
cd_definition_min_trials_per_block = 4

# Main train/held-out split for CD projections. When enabled, the CD is
# computed from a stratified subset of CD-eligible nonstim/control trials.
# Opto trials are never used to compute the CD, and all remaining opto trials
# are still projected. The control projection/evaluation pool is controlled by
# heldout_control_projection_source below, and by default excludes only the
# CD-training controls.
use_heldout_control_projection = 1
cd_train_control_fraction = 0.5       # fraction of eligible control trials per block used to compute CD
cd_control_split_seed = 0             # deterministic split seed; PID is mixed in so each insertion has its own split
cd_control_split_min_trials_per_block = 4  # require at least this many train AND held-out controls per block

# Which nonstim/control trials are used for projection/statistics after the CD
# training split. The recommended/default value compares all opto trials against
# all non-training controls, avoiding train/test reuse without selecting only
# prior-choice-congruent/correct controls for the black trace.
#
#   'all_nontraining_controls'
#       Project every nonstim/control trial not used to train the CD. This best
#       matches the opto trace, which uses all opto/stim trials that pass the
#       global trial filters.
#
#   'heldout_cd_definition_controls'
#       Legacy diagnostic behavior: project only held-out controls from the same
#       CD-definition-eligible pool used for CD training.
heldout_control_projection_source = 'all_nontraining_controls'

# Optional per-unit scalar normalization before CD computation/projection.
#   'none'
#       Use raw binned spike counts/rates exactly as before.
#   'baseline_scalar'
#       For each unit, estimate its baseline firing rate from the CD-training
#       control trials in the CD-definition epoch, then divide that unit's
#       activity by the corresponding mean baseline counts/bin. The same unit
#       scaling is applied to the CD-definition tensor and every projection
#       tensor, so CD estimation and projection remain in the same coordinate
#       system. Low-rate units are protected by the rate floor and max-scale cap.
cd_unit_normalization_mode = 'none'   # set to 'baseline_scalar' to test high-FR dominance
cd_unit_baseline_min_rate_hz = 1.0    # denominator floor; prevents huge gains for very quiet units
cd_unit_baseline_max_scale = 10.0     # max multiplier applied to any unit; set None to rely only on rate floor


# =========================================================================
# Unit quality and filtering
# =========================================================================
# Basic cluster-quality filters, all applied before CD computation.
IBL_quality_label_threshold = 2/3   # keep clusters with IBL label >= this value
firing_rate_threshold = 0           # currently imported for compatibility; region/unit cascade does not enforce it
presence_threshold = 0            # keep clusters with clusters.presence_ratio > this value when available

# Firing-rate drift filter. In the default 'quiescence' mode, each unit's mean
# firing rate is measured trial-by-trial in [goCue - quiescencePeriod, goCue].
# A unit is flagged when abs(Spearman(FR, trial_index)) > drift_threshold AND
# that time correlation is larger than abs(Spearman(FR, block_id)). Per-unit
# metrics are saved to qc_reports/<PID>/<PID>_qp_drift_metrics.csv and plotted
# in qc_reports/<PID>/<PID>_qp_drift_metric_histogram.png, so threshold choice
# can be checked by inspecting each insertion's real value distribution. Lower
# thresholds are stricter; higher thresholds remove fewer units.
remove_drift_units = 1
drift_threshold = 0.35
#   'quiescence'  : motor-free enforced-QP FR, recommended/default.
#   'fixed_window': legacy full CD-window tensor, more vulnerable to motor timing.
drift_epoch = 'quiescence'

# Firing-rate nonstationarity filter. This is shared with the BS pipeline and
# catches non-monotonic drifts/dropouts that the monotonic Spearman drift metric
# can miss. For each unit, QP firing rate is measured trial-by-trial, block-
# specific medians are subtracted, and contiguous session segments are compared.
# The CSV/PNG outputs are always useful for threshold exploration. By default
# remove_nonstationary_units=0 and all thresholds are None, so the metrics are
# saved/reported but no extra units are removed until thresholds are chosen.
remove_nonstationary_units = 0
nonstationarity_n_segments = 6
nonstationarity_min_trials = 30
nonstationarity_min_trials_per_segment = 8
nonstationarity_min_trials_per_block_segment = 3
nonstationarity_low_fr_fraction_of_median = 0.2
nonstationarity_min_median_fr_hz = 0.1

max_qp_fr_segment_range_frac = None          # raw segment FR range / median FR
max_qp_resid_drift_range_frac = None#1.5#2#1.5 #0.8         # block-residual segment FR range / median FR #0.8-1?
max_qp_resid_drift_cv = None#0.4#1.2#0.8  #0.4               # block-residual segment mean SD / median FR #0.4?
max_qp_resid_abs_rho_time = None#0.15# 0.15  #0.13           # abs(Spearman(block-residual FR, trial order)) #0.13?
max_qp_low_activity_fraction = None          # fraction of QP trials below low_fr_fraction*median FR
max_qp_max_low_activity_run = None           # longest consecutive low-activity run in QP trials
min_qp_block_effect_sign_consistency = None#0.75#0.5 #0.75  # fraction of valid segments with same block-effect sign
max_qp_block_effect_segment_cv = None        # segment block-effect SD / global block effect
max_qp_block_effect_dominance = None#1.8#0.9         # largest segment |effect| / sum segment |effects|

# Cross-validated block-vs-time model comparison. For each unit, QP firing rate
# is predicted with four models: null, block-only, time-only, and block+time.
# Unique block R2 = CV_R2(block+time) - CV_R2(time-only). Unique time R2 =
# CV_R2(block+time) - CV_R2(block-only). This catches units whose apparent block
# effect is better explained by smooth session-time drift. Metrics/plots are
# saved when compute_qp_block_time_model=1. Units are removed only when
# remove_qp_block_time_model_units=1 and the active thresholds fail according to
# block_time_model_flag_logic.
compute_qp_block_time_model = 0
remove_qp_block_time_model_units = 0
block_time_model_min_trials = 40
block_time_model_min_trials_per_block = 8
block_time_model_n_folds = 5
block_time_model_fold_mode = 'interleaved'  # 'interleaved', 'blocked', or 'random'
block_time_model_time_degree = 3
block_time_model_ridge_alpha = 1e-6
block_time_model_flag_logic = 'any'  # aggressive enrichment: remove a unit if any active threshold marks weak block/time-heavy coding
max_qp_unique_time_r2 = 0.05          # remove units whose QP FR has strong unique session-time/drift prediction
max_qp_time_over_block_ratio = 2.0   # remove units when unique time prediction is >2x unique block prediction
min_qp_unique_block_r2 = 0.005       # remove units with almost no cross-validated block-predictive signal
min_qp_block_time_preference = -0.5   # remove units strongly dominated by time; +1 block-dominated, -1 time-dominated

# Spike-amplitude drift filter. This is distinct from the static waveform
# amplitude percentile filter below. It uses per-spike amplitudes (spikes['amps'])
# across the analyzed trial time span and flags units by late-vs-early median
# amplitude change and/or Spearman(amplitude, spike_time). Defaults are report
# only/off; set a threshold and remove_amplitude_drift_units=1 to enforce.
remove_amplitude_drift_units = 0
amplitude_drift_max_fractional_change = 0.5  # e.g. 0.5 flags |late-early| > 50% of median amp
amplitude_drift_max_abs_spearman = 0.5       # e.g. 0.5 flags strong monotonic amp-time drift
amplitude_drift_min_spikes = 100

remove_axonal_units = 0             # remove waveform_classify classical axonal units when templates are available
only_include_BS_units = 0           # optional legacy CD pre-filter: compute per-unit BS and keep BS_score == 1

# Session-level hard inclusion thresholds applied after trial/unit exclusions.
minimum_unit_number = 10 #20            # skip insertion if final unit count is below this
minimum_inhibition_trials_number = 30  # skip insertion if final opto-trial count is below this
minimum_control_trials_per_block = None # e.g. 25; checked after trial filters
minimum_opto_trials_per_block = 10    # e.g. 10; checked after trial filters
minimum_block_transitions = 2        # e.g. 3; count of block switches represented after trial filters


# =========================================================================
# Additional QC diagnostics and optional filters
# =========================================================================
# These are designed to be conservative and transparent. Most of them save
# per-insertion CSV/PNG outputs under figures_path/qc_reports/<PID>/.
save_qc_outputs = 1

# Ultra-fast laser-locked spike detector. For each candidate unit, spikes are
# counted in light_artifact_window_s after each laser onset and compared to the
# pre-laser baseline window scaled to the artifact-window duration. A unit is
# flagged only if all three criteria are met: z threshold, fraction of laser
# events with a spike, and excess spikes/event. If remove_light_artifact_units=1,
# flagged units are removed before CD computation; metrics are saved either way.
remove_light_artifact_units = 1
light_artifact_window_s = (0.000, 0.005)
light_artifact_baseline_window_s = (-0.050, -0.005)
light_artifact_z_threshold = 8.0
light_artifact_min_event_fraction = 0.20
light_artifact_min_excess_spikes_per_event = 0.05

# Static waveform-amplitude outlier filter. Uses the average waveform template
# for each cluster, takes peak-to-peak amplitude on the peak channel, then flags
# units outside the configured within-insertion percentiles. This is NOT a
# drift-over-time metric; spike-amplitude drift is controlled above.
remove_waveform_amplitude_outliers = 1
waveform_amplitude_low_percentile = 0.5
waveform_amplitude_high_percentile = 99.5

# Cross-validated CD reliability. Stratifies correct CD-eligible control trials
# by block, computes a CD from half of each block, and measures held-out control
# block separation in the CD epoch across random splits. Saved in session['qc'].
# The main analysis does not skip insertions based on these values here; the
# plotting script can filter by minimum_cv_abs_control_separation and
# minimum_cv_sign_consistency.
compute_cd_reliability = 1
cd_reliability_n_splits = 50
cd_reliability_seed = 0

# Recommended plotting/exclusion criteria are handled in the plotting script,
# but these values are saved in each session's qc dict:
#   cv_mean_control_separation, cv_abs_mean_control_separation,
#   cv_sem_control_separation, cv_fraction_positive, cv_sign_consistency
# Practical first-pass thresholds to try in plotting:
#   minimum_cv_abs_control_separation = 0.05 or 0.10
# Sign consistency can also be used as a descriptive diagnostic, but because
# CD sign is arbitrary it should usually be secondary to absolute held-out separation.

# Extra held-out control diagnostic for in-sample CD inflation. This predates
# the main train/held-out split above and is now optional/redundant for most
# runs: it computes an additional diagnostic CD from a stratified subset of
# nonstim control trials and saves a QC-only comparison of train controls,
# held-out controls, and opto trials. It does not replace the main CD.
compute_heldout_control_diagnostic = 0
heldout_control_fraction = 0.5
heldout_control_seed = 0
heldout_control_min_trials_per_block = 4

# Pseudo-opto null diagnostic. Repeatedly chooses fake opto trials from the
# real nonstim/control trial pool, stratified by block and count-matched to
# the true opto trials when possible. For each repeat, the pseudo-opto trials
# are treated exactly like opto trials: excluded from CD computation and then
# compared against the remaining pseudo-control trials.
compute_pseudo_opto_null = 0
pseudo_opto_n_repeats = 100
pseudo_opto_seed = 0
pseudo_opto_min_trials_per_block = 4
pseudo_opto_match_real_opto_counts = 1

# Unit leverage diagnostic/filter. Computes each unit's abs(CD weight) divided
# by the sum of abs weights. max/top5/top10 values are saved in session['qc'].
# Set any maximum_* threshold below to skip insertions dominated by one or a few
# units during the main analysis; leave as None to report only and optionally
# filter later in CD_analysis_midbrain_plotting.py.
compute_unit_leverage = 1
unit_leverage_top_n = 10
maximum_unit_leverage = None       # e.g. 0.25 skips if one unit carries >25% abs CD weight
maximum_top5_unit_leverage = None  # e.g. 0.60 skips if top 5 units carry >60%
maximum_top10_unit_leverage = None # e.g. 0.80 skips if top 10 units carry >80%

# Transition consistency diagnostic/filter. After CD computation, each trial's
# CD-epoch projection is classified as block0/block1 using the midpoint between
# the control block means. Accuracy is evaluated on control trials close to
# block transitions, defined by absolute within-block position <=
# transition_consistency_max_block_position. If beginning_block_trials_remove=5
# and max position=10, this tests retained positions 6-10 after each switch.
# Set minimum_transition_decode_accuracy to skip sessions below a threshold;
# leave None to save/report only.
compute_transition_consistency = 1
transition_consistency_max_block_position = 10
transition_consistency_use_correct_trials = 1
minimum_transition_decode_accuracy = None  # e.g. 0.75


# =========================================================================
# Region selection
# =========================================================================
# 'midbrain' keeps only midbrain units, 'isocortex' keeps only cortical units.
# Cortex/midbrain classification uses Allen atlas ancestry, unless the PID
# is in DEPTH_THRESHOLD_OVERRIDES, in which case a manual depth threshold
# is used instead (for PIDs lacking histology).
# recorded_region_beryl optionally narrows the atlas-labeled units further by
# Beryl acronym, e.g. ['MRN', 'SCm', 'SCs']. Leave None to disable. This
# Beryl filter is skipped for PIDs in DEPTH_THRESHOLD_OVERRIDES so mouse 102
# insertions are still analyzed using their depth-based midbrain/isocortex call.
analyze_region = 'midbrain'         # 'midbrain' or 'isocortex'
recorded_region_beryl = None#['MRN', 'SCm', 'SCs']  # Set None to disable; e.g. ['MRN', 'SCm', 'SCs']

# Manual depth thresholds (µm) for PIDs lacking histology. Units at depths
# <= threshold are classified as midbrain, > threshold as isocortex.
# Values determined from depth_opto_localizer.py analysis.
DEPTH_THRESHOLD_OVERRIDES = {
    'c9a6b866-2d9b-481c-86ec-0d4937fbd696': 2500,  # SWC_NM_102 10/2 L
    '68288763-9572-4678-9eb4-3866e3e9fb3d': 2700,  # SWC_NM_102 11/2 L
    'fc4f446b-177c-4b94-89d2-14c0500374a4': 3200,  # SWC_NM_102 12/2 L
    '32425853-de5f-4e5d-8a73-fe1285893c7f': 2900,  # SWC_NM_102 13/2 L
    '9583d73c-ee29-45d1-9aa1-2b5917bcf726': 3300,  # SWC_NM_102 14/2 L
    'a327ddee-8b7c-4463-9c24-6f82d2bfe590': 2500,  # SWC_NM_102 10/2 R
    '6bf18fe0-fca9-4cd3-aa69-546d34d24c12': 2500,  # SWC_NM_102 11/2 R
    '77946f89-7b49-43b0-b34d-c17fc70504c4': 3900,  # SWC_NM_102 12/2 R
    '4743a9f7-24d3-4cac-b956-d0323d4269db': 2100,  # SWC_NM_102 13/2 R
}


# =========================================================================
# Output paths
# =========================================================================
save_figures = 1
figures_path = '/Users/natemiska/Desktop/cd_figures'
individual_pid_prefix = '8020_NOGLMHMM_optoprior_IBL23_drift35_5050'

# Optional run label for population pickle filenames. Leave '' or None for the
# default names, e.g. CD_midbrain_all_insertions_LaserOnset.pkl. Set this when
# launching parallel runs with different options, e.g. 'choice_congruent_drift01'
# -> choice_congruent_drift01_CD_midbrain_all_insertions_LaserOnset.pkl.
figure_prefix = '8020_NOGLMHMM_optoprior_IBL23_drift35_5050' #''

# Output organization. When enabled, all analysis outputs for this run are saved
# under figures_path/<figure_prefix>/, including population pickle files,
# insertion-level outputs, QC reports, and run configuration manifests. Set 0 to
# recover the older flat figures_path behavior.
organize_outputs_by_figure_prefix = 1

# Per-insertion outputs. When both organization toggles are enabled, these are
# saved under figures_path/<figure_prefix>/insertions/<PID>/ with shared QC files
# in qc/ and alignment-specific figures in LaserOnset/, GoCueOnset/, etc. A
# global insertion_qc_manifest.csv is also written in the run folder and includes
# kept/excluded status plus exclusion reasons.
organize_outputs_by_insertion = 1
insertion_outputs_folder = 'insertions'
save_insertion_qc_manifest = 1

# Per-insertion outputs saved in figures_path. The overlay figure uses the same
# control/opto block color scheme as population_projection_overlay_* panels:
# control block1 black, control block0 gray, opto block1 magenta, opto block0
# xkcd:tangerine. The QC summary is a one-row CSV per PID with final unit/trial
# counts, removed-trial counts, drift/artifact/leverage/transition metrics, and
# scalar CD separation values.
save_individual_projection_overlay = 1
save_insertion_qc_summary = 1
