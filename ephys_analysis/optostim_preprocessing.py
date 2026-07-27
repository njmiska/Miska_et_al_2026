"""
optostim_preprocessing.py
=========================
Shared preprocessing / QC layer for the optostim analyses.

This module is the SINGLE SOURCE OF TRUTH for the decisions that must match
between the two pipelines:

    CD_analysis_midbrain.py              (coding-direction collapse)
    SNr_inhibition_BS_downstream_effect.py (per-cluster bias selectivity)

What lives here:
    1. Canonical session loading, including the lightweight IBL waveform
       templates (`ssl.load_spike_sorting_object('waveforms')`).
    2. The unit-selection cascade (quality label, manual + automatic light
       artifact, static waveform-amplitude outliers, axonal classification,
       region selection, presence ratio), plus optional post-trial-span
       spike-amplitude drift QC.
    3. The trial-selection cascade (range normalization, beginning-of-block
       removal, laser/stim classification, GLM-HMM engagement, RT/quiescence
       duration filtering, previous-stim filtering, per-PID exclusions).
    4. Shared binning (`build_binned_X`) and a post-binning drift helper, so
       drift is computed identically in both pipelines.
    5. A thin metadata interface over `metadata_optostim_new.insertions`.

Design notes
------------
* No ONE / atlas object is constructed at import time. Callers pass `one`,
  `ba` (AllenAtlas) and `br` (BrainRegions) in. This keeps the module
  importable and testable without network access, and means each pipeline
  keeps ownership of its ONE configuration.
* Parameters are grouped into two small dataclasses (`UnitQCParams`,
  `TrialQCParams`). Each has a `from_config(cfg_module)` constructor that
  reads attributes by name, so the existing flat config files barely change:
  you just add the missing knobs to BS_config and call `from_config`.
* Unit selection and trial preparation are split into independent functions.
  This reorders CD's interleaved cascade into "trials-then-units", which is
  result-identical because the two operate on disjoint index spaces (cluster
  IDs vs trial indices). The only cross-dependency — the auto light-artifact
  detector needing `laser_onsets` — is honoured by preparing trials first.
* Every filter remains individually toggleable via the dataclass fields, so
  either pipeline can opt in/out of each stage from its own config.

The BS-only pre-filter (`only_include_BS_units`) is intentionally NOT here:
it requires pseudo-session machinery and is analysis-specific. CD keeps it as
a CD-side step; BS produces BS scores as its primary output anyway.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Callable, Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Shared analysis helpers (already imported by both pipelines today).
from miska_analysis.functions_optostim import get_drift_indices
from waveform_classify import classify_and_plot_axonal_units


# =====================================================================
# Parameter containers
# =====================================================================
@dataclass
class UnitQCParams:
    """Unit-selection knobs. Field names match the existing config files so
    `from_config` is a direct attribute copy.

    Each `remove_*` flag is an independent toggle. Setting one to 0 reports
    the relevant metric (and still writes QC outputs) without excluding units,
    which keeps the "report-before-automate" workflow intact.
    """
    # Quality
    IBL_quality_label_threshold: float = 0.6
    presence_threshold: float = 0.75

    # Toggles
    remove_light_artifact_units: int = 1
    remove_waveform_amplitude_outliers: int = 0
    remove_axonal_units: int = 1
    remove_drift_units: int = 1   # applied post-binning by the caller

    # Automatic light-artifact detector
    light_artifact_window_s: Tuple[float, float] = (0.000, 0.005)
    light_artifact_baseline_window_s: Tuple[float, float] = (-0.050, -0.005)
    light_artifact_z_threshold: float = 8.0
    light_artifact_min_event_fraction: float = 0.20
    light_artifact_min_excess_spikes_per_event: float = 0.05

    # Waveform-amplitude outlier QC
    waveform_amplitude_low_percentile: float = 0.5
    waveform_amplitude_high_percentile: float = 99.5

    # Spike-amplitude drift QC. This uses per-spike amplitudes when available
    # (spikes['amps']) over the analyzed trial time span, not the static average
    # waveform template above.
    remove_amplitude_drift_units: int = 0
    amplitude_drift_max_fractional_change: Optional[float] = None
    amplitude_drift_max_abs_spearman: Optional[float] = None
    amplitude_drift_min_spikes: int = 100

    # Axonal classification
    axonal_pt_ratio_threshold: float = 1.0

    # Region selection
    analyze_region: str = 'midbrain'   # 'midbrain' | 'isocortex'
    recorded_region_beryl: Optional[List[str]] = None
    DEPTH_THRESHOLD_OVERRIDES: Dict[str, float] = field(default_factory=dict)

    # Drift
    drift_threshold: float = 0.35

    # Quiescence-period firing-rate nonstationarity. These metrics are designed
    # to catch non-monotonic drift/dropouts after accounting for block identity.
    # Thresholds default to None so the metrics are saved without excluding
    # units until the user chooses empirical cutoffs.
    remove_nonstationary_units: int = 0
    nonstationarity_n_segments: int = 6
    nonstationarity_min_trials: int = 30
    nonstationarity_min_trials_per_segment: int = 8
    nonstationarity_min_trials_per_block_segment: int = 3
    nonstationarity_low_fr_fraction_of_median: float = 0.2
    nonstationarity_min_median_fr_hz: float = 0.1
    max_qp_fr_segment_range_frac: Optional[float] = None
    max_qp_resid_drift_range_frac: Optional[float] = None
    max_qp_resid_drift_cv: Optional[float] = None
    max_qp_resid_abs_rho_time: Optional[float] = None
    max_qp_low_activity_fraction: Optional[float] = None
    max_qp_max_low_activity_run: Optional[int] = None
    min_qp_block_effect_sign_consistency: Optional[float] = None
    max_qp_block_effect_segment_cv: Optional[float] = None
    max_qp_block_effect_dominance: Optional[float] = None

    # Cross-validated block-vs-time model comparison. This asks whether trial
    # time explains a unit's QP firing rate after block is accounted for, and
    # whether block still explains firing after trial time is accounted for.
    compute_qp_block_time_model: int = 1
    remove_qp_block_time_model_units: int = 0
    block_time_model_min_trials: int = 40
    block_time_model_min_trials_per_block: int = 8
    block_time_model_n_folds: int = 5
    block_time_model_fold_mode: str = 'interleaved'
    block_time_model_time_degree: int = 3
    block_time_model_ridge_alpha: float = 1e-6
    block_time_model_flag_logic: str = 'all'
    max_qp_unique_time_r2: Optional[float] = None
    max_qp_time_over_block_ratio: Optional[float] = None
    min_qp_unique_block_r2: Optional[float] = None
    min_qp_block_time_preference: Optional[float] = None

    # Output / bookkeeping
    save_qc_outputs: int = 1
    save_figures: int = 1
    figures_path: str = '.'

    @classmethod
    def from_config(cls, cfg) -> "UnitQCParams":
        """Build from a config module, reading whatever attributes exist and
        falling back to the dataclass defaults otherwise. Unknown/missing
        attributes are silently defaulted so a partially-populated BS_config
        still works."""
        kwargs = {}
        for f in cls.__dataclass_fields__:
            if hasattr(cfg, f):
                kwargs[f] = getattr(cfg, f)
        return cls(**kwargs)


@dataclass
class TrialQCParams:
    """Trial-selection knobs. Field names match the config files."""
    beginning_block_trials_remove: int = 10
    allowed_probability_left_values: Optional[Tuple[float, ...]] = None
    probability_left_tolerance: float = 1e-6
    remove_stim_trials_preceded_by_stim: int = 0

    # Behavioral timing filters. Applied to the full trial set used for CD
    # computation and projection, before final opto/control masks are returned.
    min_reaction_time_s: Optional[float] = None
    max_reaction_time_s: Optional[float] = None
    reaction_time_source: str = 'firstMovement_times'  # 'firstMovement_times', 'response_times', or 'auto'
    min_quiescence_period_s: Optional[float] = None
    max_quiescence_period_s: Optional[float] = None

    use_GLMHMM_engaged_indices: int = 1
    opto_trials_GLMHMM: str = 'standard'
    n_states: int = 2

    # Per-PID hard exclusions (BS uses this today; CD can adopt it).
    TRIALS_TO_REMOVE: Dict[str, List[int]] = field(default_factory=dict)

    # Output / bookkeeping
    save_qc_outputs: int = 1
    figures_path: str = '.'

    @classmethod
    def from_config(cls, cfg) -> "TrialQCParams":
        kwargs = {}
        for f in cls.__dataclass_fields__:
            if hasattr(cfg, f):
                kwargs[f] = getattr(cfg, f)
        return cls(**kwargs)


@dataclass
class SessionBundle:
    """Everything a per-PID analysis needs from raw loading, in one place."""
    pid: str
    eid: Any
    mouse_id: str
    probe_label: str
    ses_path: Any
    spikes: Any
    clusters: Any
    channels: Any
    trials: Any
    clusters_labels: np.ndarray
    brain_acronyms_percluster: np.ndarray
    waveforms: Optional[dict]   # output of ssl.load_spike_sorting_object('waveforms') or None


# =====================================================================
# Metadata interface (thin wrapper over metadata_optostim_new)
# =====================================================================
def _as_filter_set(x):
    if x is None:
        return None
    if isinstance(x, str):
        return {x}
    return set(x)


def select_insertions(insertions, brain_regions=None, conditions=None, pids_filter=None):
    """Filter the one-dict-per-insertion metadata. Identical semantics to the
    `_select_insertions_from_metadata` helper currently in CD, exposed here so
    BS uses exactly the same selection logic.
    """
    brain_regions = _as_filter_set(brain_regions)
    conditions = _as_filter_set(conditions)
    pids_filter = _as_filter_set(pids_filter)
    selected = []
    for ins in insertions:
        if brain_regions is not None and ins.get('brain region') not in brain_regions:
            continue
        if conditions is not None and ins.get('condition') not in conditions:
            continue
        if pids_filter is not None and ins.get('PID') not in pids_filter:
            continue
        selected.append(ins)
    return selected


# =====================================================================
# Session loading
# =====================================================================
def load_session(pid, one, ba, *, load_waveforms=True, enforce_version=False) -> SessionBundle:
    """Load spikes/clusters/channels/trials and (optionally) the canonical IBL
    waveform templates for a single insertion.

    Waveforms are loaded via `ssl.load_spike_sorting_object('waveforms')`,
    which returns averaged, cluster-indexed templates (small download), rather
    than the per-spike `_phy_spikes_subset` snippets used by the old BS path.
    Older pykilosort collections store the same small averaged templates as
    ``templates.waveforms.npy``; that exact dataset is used as a safe fallback.
    Neither path requests ``waveforms.traces`` or ``_phy_spikes_subset``.
    """
    from brainbox.io.one import SpikeSortingLoader  # local import: avoids hard dep at module import

    ssl = SpikeSortingLoader(pid=pid, one=one, atlas=ba)
    eid = ssl.eid
    ses_info = one.get_details(eid)
    mouse_id = ses_info['subject']
    probe_label = ssl.pname
    ses_path = one.eid2path(eid)

    trials = one.load_object(eid, 'trials')
    spikes, clusters, channels = ssl.load_spike_sorting(enforce_version=enforce_version)

    waveforms = None
    if load_waveforms:
        try:
            waveforms = ssl.load_spike_sorting_object('waveforms')
        except Exception as e:
            try:
                # Legacy pykilosort layout: one averaged template per cluster.
                # Loading this named 3-D array explicitly prevents ONE from
                # resolving to the large per-spike waveform subset.
                templates = one.load_dataset(
                    eid, 'templates.waveforms.npy', collection=ssl.collection,
                )
                waveforms = {'templates': templates}
                print(
                    'Canonical waveform object unavailable; using safe legacy '
                    'templates.waveforms.npy averaged templates.'
                )
            except Exception as fallback_e:
                print(
                    f'waveform loading failed ({e}); safe averaged-template '
                    f'fallback also failed ({fallback_e}); waveform-based QC '
                    'will be skipped...'
                )
                waveforms = None

    clusters = ssl.merge_clusters(spikes, clusters, channels)
    clusters_labels = clusters['label']

    try:
        brain_acronyms_percluster = clusters['acronym']
    except Exception:
        # Probe alignment missing -> acronyms are nan.
        brain_acronyms_percluster = np.empty(len(clusters['ks2_label']))
        brain_acronyms_percluster[:] = np.nan

    return SessionBundle(
        pid=pid, eid=eid, mouse_id=mouse_id, probe_label=probe_label,
        ses_path=ses_path, spikes=spikes, clusters=clusters, channels=channels,
        trials=trials, clusters_labels=clusters_labels,
        brain_acronyms_percluster=brain_acronyms_percluster, waveforms=waveforms,
    )


# =====================================================================
# QC output plumbing
# =====================================================================
def make_qc_dir(figures_path, pid, save_qc_outputs=1) -> Path:
    qc_dir = Path(figures_path) / 'qc_reports' / str(pid)
    if save_qc_outputs == 1:
        qc_dir.mkdir(parents=True, exist_ok=True)
    return qc_dir


def save_filter_cascade(qc_dir, pid, rows, save_qc_outputs=1):
    if save_qc_outputs != 1 or not rows:
        return
    df = pd.DataFrame(rows)
    df.to_csv(qc_dir / f'{pid}_unit_filter_counts.csv', index=False)
    try:
        plt.figure(figsize=(8, 4))
        plt.plot(np.arange(len(df)), df['n_units'].values, marker='o')
        plt.xticks(np.arange(len(df)), df['stage'].values, rotation=45, ha='right')
        plt.ylabel('Units remaining')
        plt.title(f'Unit filter cascade\n{pid[:12]}')
        plt.tight_layout()
        plt.savefig(qc_dir / f'{pid}_unit_filter_cascade.png', dpi=200)
        plt.close()
    except Exception as e:
        print(f'QC filter cascade plotting failed for {pid}: {e}')


# =====================================================================
# Waveform-amplitude outlier QC  (lifted faithfully from CD)
# =====================================================================
def _extract_waveform_amplitudes(cluster_ids, waveforms):
    """Peak-to-peak template amplitude (uV) per cluster, from the canonical
    'waveforms' object (dict with 'templates' of shape
    (n_clusters, n_channels, n_samples)). Uses the same template accessor as
    the axonal/peak-waveform code so the format matches."""
    from waveform_classify import _extract_cluster_template_uv
    templates = waveforms['templates'] if (isinstance(waveforms, dict) and 'templates' in waveforms) else waveforms
    amps = []
    for cid in np.asarray(cluster_ids, dtype=int):
        try:
            tmpl = _extract_cluster_template_uv(templates, int(cid))  # (n_samples, n_channels)
            if tmpl is None:
                amps.append(np.nan)
                continue
            peak_chan = int(np.argmax(np.max(np.abs(tmpl), axis=0)))
            wf = tmpl[:, peak_chan]
            amps.append(float(np.nanmax(wf) - np.nanmin(wf)))
        except Exception:
            amps.append(np.nan)
    return np.asarray(amps, dtype=float)


def waveform_amplitude_outlier_qc(cluster_ids, waveforms, qc_dir, pid, p: UnitQCParams):
    cluster_ids = np.asarray(cluster_ids, dtype=int)
    amps = _extract_waveform_amplitudes(cluster_ids, waveforms)
    df = pd.DataFrame({'cluster_id': cluster_ids, 'template_ptp_amplitude': amps})
    finite = np.isfinite(amps)
    flagged = np.zeros(len(cluster_ids), dtype=bool)
    lo = hi = np.nan
    if finite.sum() >= 10:
        lo, hi = np.nanpercentile(
            amps[finite],
            [p.waveform_amplitude_low_percentile, p.waveform_amplitude_high_percentile],
        )
        flagged = finite & ((amps < lo) | (amps > hi))
    df['flagged_waveform_amplitude_outlier'] = flagged
    df['low_threshold'] = lo
    df['high_threshold'] = hi
    if p.save_qc_outputs == 1:
        df.to_csv(qc_dir / f'{pid}_waveform_amplitude_metrics.csv', index=False)
        try:
            plt.figure(figsize=(6, 4))
            plt.hist(amps[finite], bins=50)
            if np.isfinite(lo):
                plt.axvline(lo, linestyle='--')
            if np.isfinite(hi):
                plt.axvline(hi, linestyle='--')
            plt.xlabel('Template peak-to-peak amplitude')
            plt.ylabel('Unit count')
            plt.title(f'Waveform amplitude QC\n{pid[:12]}')
            plt.tight_layout()
            plt.savefig(qc_dir / f'{pid}_waveform_amplitude_qc.png', dpi=200)
            plt.close()
        except Exception as e:
            print(f'Waveform amplitude QC plot failed for {pid}: {e}')
    return cluster_ids[flagged], df


def _spikes_field(spikes, key):
    try:
        return spikes[key]
    except Exception:
        return getattr(spikes, key, None)


def compute_spike_amplitude_drift_unit_ids(spikes, cluster_ids, analysis_start_time,
                                           analysis_end_time, qc_dir, pid,
                                           p: UnitQCParams):
    """Flag units whose spike amplitudes drift over the analyzed time span.

    This is distinct from `waveform_amplitude_outlier_qc`: the waveform filter
    uses one static average template per cluster, while this metric uses
    per-spike amplitudes (`spikes['amps']`) across session time. A unit is
    flagged if either configured criterion is exceeded:
      abs((late_median - early_median) / all_median) >
          amplitude_drift_max_fractional_change
      abs(Spearman(spike_amp, spike_time)) >
          amplitude_drift_max_abs_spearman

    Returns (flagged_cluster_ids, metrics_df). If spike amplitudes are
    unavailable, returns no flagged units and writes a small skipped table.
    """
    cluster_ids = np.asarray(cluster_ids, dtype=int)
    times = np.asarray(_spikes_field(spikes, 'times'), dtype=float)
    clus = np.asarray(_spikes_field(spikes, 'clusters'))
    amps = _spikes_field(spikes, 'amps')
    if amps is None:
        df = pd.DataFrame({
            'cluster_id': cluster_ids,
            'amplitude_drift_skipped_reason': 'spikes amps unavailable',
            'flagged_amplitude_drift': False,
        })
        if p.save_qc_outputs == 1:
            df.to_csv(qc_dir / f'{pid}_spike_amplitude_drift_metrics.csv', index=False)
        return np.asarray([], dtype=int), df

    from scipy.stats import spearmanr

    amps = np.asarray(amps, dtype=float)
    start = float(analysis_start_time)
    end = float(analysis_end_time)
    mid = start + (end - start) / 2.0
    in_span = np.isfinite(times) & np.isfinite(amps) & (times >= start) & (times <= end)

    rows = []
    flagged_ids = []
    for cid in cluster_ids:
        cid = int(cid)
        m = in_span & (clus == cid)
        tt = times[m]
        aa = amps[m]
        n_spikes = int(aa.size)
        early = aa[tt < mid]
        late = aa[tt >= mid]
        early_med = float(np.nanmedian(early)) if early.size else np.nan
        late_med = float(np.nanmedian(late)) if late.size else np.nan
        all_med = float(np.nanmedian(aa)) if aa.size else np.nan
        if np.isfinite(all_med) and all_med != 0:
            frac_change = (late_med - early_med) / abs(all_med)
        else:
            frac_change = np.nan
        if n_spikes >= int(p.amplitude_drift_min_spikes) and np.unique(tt).size >= 3:
            try:
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    rho, rho_p = spearmanr(tt, aa)
            except Exception:
                rho, rho_p = np.nan, np.nan
        else:
            rho, rho_p = np.nan, np.nan

        flag_frac = (
            p.amplitude_drift_max_fractional_change is not None
            and np.isfinite(frac_change)
            and abs(frac_change) > float(p.amplitude_drift_max_fractional_change)
        )
        flag_rho = (
            p.amplitude_drift_max_abs_spearman is not None
            and np.isfinite(rho)
            and abs(rho) > float(p.amplitude_drift_max_abs_spearman)
        )
        flagged = bool(n_spikes >= int(p.amplitude_drift_min_spikes) and (flag_frac or flag_rho))
        if flagged:
            flagged_ids.append(cid)
        rows.append({
            'cluster_id': cid,
            'n_spikes_in_analysis_span': n_spikes,
            'early_median_amp': early_med,
            'late_median_amp': late_med,
            'all_median_amp': all_med,
            'late_minus_early_fraction_of_median': float(frac_change) if np.isfinite(frac_change) else np.nan,
            'amp_time_spearman_rho': float(rho) if np.isfinite(rho) else np.nan,
            'amp_time_spearman_p': float(rho_p) if np.isfinite(rho_p) else np.nan,
            'amplitude_drift_min_spikes': int(p.amplitude_drift_min_spikes),
            'amplitude_drift_max_fractional_change': p.amplitude_drift_max_fractional_change,
            'amplitude_drift_max_abs_spearman': p.amplitude_drift_max_abs_spearman,
            'flagged_amplitude_drift': flagged,
        })

    df = pd.DataFrame(rows)
    if p.save_qc_outputs == 1:
        df.to_csv(qc_dir / f'{pid}_spike_amplitude_drift_metrics.csv', index=False)
        try:
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            axes[0].hist(df['late_minus_early_fraction_of_median'].dropna(), bins=40)
            axes[0].axvline(0, color='0.5', linestyle='--')
            if p.amplitude_drift_max_fractional_change is not None:
                thr = float(p.amplitude_drift_max_fractional_change)
                axes[0].axvline(thr, color='r', linestyle=':')
                axes[0].axvline(-thr, color='r', linestyle=':')
            axes[0].set_xlabel('Late - early spike amp / median amp')
            axes[0].set_ylabel('Units')
            axes[1].hist(df['amp_time_spearman_rho'].dropna(), bins=40)
            axes[1].axvline(0, color='0.5', linestyle='--')
            if p.amplitude_drift_max_abs_spearman is not None:
                thr = float(p.amplitude_drift_max_abs_spearman)
                axes[1].axvline(thr, color='r', linestyle=':')
                axes[1].axvline(-thr, color='r', linestyle=':')
            axes[1].set_xlabel('Spearman(spike amp, time)')
            fig.suptitle(f'Spike amplitude drift QC\n{pid[:12]} flagged={len(flagged_ids)}')
            fig.tight_layout()
            fig.savefig(qc_dir / f'{pid}_spike_amplitude_drift_qc.png', dpi=200)
            plt.close(fig)
        except Exception as e:
            print(f'Spike amplitude drift QC plot failed for {pid}: {e}')
    return np.asarray(flagged_ids, dtype=int), df


# =====================================================================
# Automatic light-artifact detection  (lifted faithfully from CD)
# =====================================================================
def _count_spikes_in_windows(spike_times_unit, events, win):
    counts = np.zeros(len(events), dtype=float)
    for i, ev in enumerate(events):
        counts[i] = np.sum((spike_times_unit >= ev + win[0]) & (spike_times_unit < ev + win[1]))
    return counts


def detect_light_artifact_units(spike_times, spike_clusters, cluster_ids, laser_onsets,
                                qc_dir, pid, p: UnitQCParams):
    """Detect ultra-fast laser-locked spikes consistent with light artifact."""
    cluster_ids = np.asarray(cluster_ids, dtype=int)
    laser_onsets = np.asarray(laser_onsets, dtype=float)
    rows = []
    flagged = []
    if laser_onsets.size < 5 or cluster_ids.size == 0:
        df = pd.DataFrame({'cluster_id': cluster_ids})
        if p.save_qc_outputs == 1:
            df.to_csv(qc_dir / f'{pid}_light_artifact_metrics.csv', index=False)
        return np.asarray([], dtype=int), df

    artifact_dur = p.light_artifact_window_s[1] - p.light_artifact_window_s[0]
    baseline_dur = p.light_artifact_baseline_window_s[1] - p.light_artifact_baseline_window_s[0]
    for cid in cluster_ids:
        st = spike_times[spike_clusters == cid]
        art_counts = _count_spikes_in_windows(st, laser_onsets, p.light_artifact_window_s)
        base_counts = _count_spikes_in_windows(st, laser_onsets, p.light_artifact_baseline_window_s)
        art_rate = art_counts.mean() / artifact_dur if artifact_dur > 0 else np.nan
        base_rate = base_counts.mean() / baseline_dur if baseline_dur > 0 else np.nan
        expected = base_counts.mean() * (artifact_dur / baseline_dur) if baseline_dur > 0 else 0.0
        z = (art_counts.mean() - expected) / np.sqrt(expected + 1e-9)
        event_fraction = float(np.mean(art_counts > 0))
        excess_spikes_per_event = float(art_counts.mean() - expected)
        is_flagged = (
            z >= p.light_artifact_z_threshold and
            event_fraction >= p.light_artifact_min_event_fraction and
            excess_spikes_per_event >= p.light_artifact_min_excess_spikes_per_event
        )
        if is_flagged:
            flagged.append(cid)
        rows.append({
            'cluster_id': int(cid),
            'laser_events': int(laser_onsets.size),
            'artifact_spikes_per_event': float(art_counts.mean()),
            'baseline_spikes_per_event_scaled': float(expected),
            'artifact_rate_hz': float(art_rate),
            'baseline_rate_hz': float(base_rate),
            'artifact_z': float(z),
            'event_fraction_with_artifact_spike': event_fraction,
            'excess_spikes_per_event': excess_spikes_per_event,
            'flagged_light_artifact': bool(is_flagged),
        })
    df = pd.DataFrame(rows)
    if p.save_qc_outputs == 1:
        df.to_csv(qc_dir / f'{pid}_light_artifact_metrics.csv', index=False)
        try:
            plt.figure(figsize=(6, 5))
            plt.scatter(df['event_fraction_with_artifact_spike'], df['artifact_z'], s=20)
            flagged_df = df[df['flagged_light_artifact']]
            if len(flagged_df):
                plt.scatter(flagged_df['event_fraction_with_artifact_spike'],
                            flagged_df['artifact_z'], s=30)
            plt.axhline(p.light_artifact_z_threshold, linestyle='--')
            plt.axvline(p.light_artifact_min_event_fraction, linestyle='--')
            plt.xlabel('Fraction laser events with spike in artifact window')
            plt.ylabel('Laser-locked artifact z')
            plt.title(f'Light artifact QC\n{pid[:12]} flagged={len(flagged)}')
            plt.tight_layout()
            plt.savefig(qc_dir / f'{pid}_light_artifact_qc.png', dpi=200)
            plt.close()
        except Exception as e:
            print(f'Light artifact QC plot failed for {pid}: {e}')
    return np.asarray(flagged, dtype=int), df


# =====================================================================
# Region selection  (lifted faithfully from CD)
# =====================================================================
def _region_selector_values(selector):
    if selector is None:
        return None
    if isinstance(selector, (str, bytes)):
        return {str(selector)}
    return {str(val) for val in selector}


def _scalar_acronym(value):
    arr = np.asarray(value)
    if arr.size == 0:
        return ''
    return str(arr.ravel()[0])


def _beryl_acronyms_from_allen(br, allen_acronyms):
    beryl = []
    for acronym in np.asarray(allen_acronyms):
        try:
            beryl.append(_scalar_acronym(br.acronym2acronym(str(acronym), mapping='Beryl')))
        except Exception:
            beryl.append('')
    return np.asarray(beryl, dtype=object)


def select_region_units(cluster_ids, sb: SessionBundle, br, p: UnitQCParams):
    """Keep units in `p.analyze_region`, optionally narrowed by Beryl acronym.

    Uses a manual depth threshold for PIDs in DEPTH_THRESHOLD_OVERRIDES
    (no histology), otherwise Allen-atlas ancestry. SNr units are always removed
    in the atlas path.
    """
    cluster_ids = np.asarray(cluster_ids, dtype=int)
    n_before = len(cluster_ids)
    pid = sb.pid
    clusters = sb.clusters
    brain_acronyms_percluster = sb.brain_acronyms_percluster

    if pid in p.DEPTH_THRESHOLD_OVERRIDES:
        depth_threshold = p.DEPTH_THRESHOLD_OVERRIDES[pid]
        print(f'Using depth-based region filter (threshold = {depth_threshold} um)')
        cluster_depths = clusters.depths
        midbrain_IDs = np.where(cluster_depths <= depth_threshold)[0]
        isocortex_IDs = np.where(cluster_depths > depth_threshold)[0]
        if p.analyze_region == 'midbrain':
            cluster_ids = np.intersect1d(cluster_ids, midbrain_IDs)
        elif p.analyze_region == 'isocortex':
            cluster_ids = np.intersect1d(cluster_ids, isocortex_IDs)
    else:
        # Always remove SNr units regardless of analyze_region.
        SNr_unit_IDs = np.where(brain_acronyms_percluster == 'SNr')[0]
        if len(SNr_unit_IDs) == 0:
            print('no SNr units detected')
        else:
            print('SNr units detected - removing')
            cluster_ids = np.setdiff1d(cluster_ids, SNr_unit_IDs)

        isocortex_id = br.acronym2id('Isocortex')[0]
        region_ids = br.acronym2id(brain_acronyms_percluster)
        cortical_IDs = []
        for cluster_ID, region_ID in zip(range(0, len(brain_acronyms_percluster)), region_ids):
            ancestors = br.ancestors(region_ID)
            if isocortex_id in ancestors.id:
                cortical_IDs.append(cluster_ID)
        if p.analyze_region == 'midbrain':
            cluster_ids = np.setdiff1d(cluster_ids, cortical_IDs)
            print(f'{len(cortical_IDs)} cortical units removed (keeping midbrain)')
        elif p.analyze_region == 'isocortex':
            cluster_ids = np.intersect1d(cluster_ids, cortical_IDs)

    beryl_selector = _region_selector_values(getattr(p, 'recorded_region_beryl', None))
    if beryl_selector is not None:
        if pid in p.DEPTH_THRESHOLD_OVERRIDES:
            print(
                'recorded_region_beryl requested, but this PID uses a manual '
                'depth override rather than atlas labels; skipping Beryl filter.'
            )
        else:
            beryl_acronyms = _beryl_acronyms_from_allen(br, brain_acronyms_percluster)
            beryl_ids = np.flatnonzero(np.isin(beryl_acronyms, list(beryl_selector)))
            n_pre_beryl = len(cluster_ids)
            cluster_ids = np.intersect1d(cluster_ids, beryl_ids)
            print(
                f'Beryl region filter {sorted(beryl_selector)}: '
                f'{n_pre_beryl} -> {len(cluster_ids)} units'
            )

    print(f'Region filter: {n_before} -> {len(cluster_ids)} units ({p.analyze_region})')
    return cluster_ids


# =====================================================================
# Unit-selection orchestrator (Phase A: everything except drift)
# =====================================================================
# =====================================================================
# Per-unit QC table (computes but does NOT apply any exclusion)
# =====================================================================
def recorded_region_flags(sb: SessionBundle, br, p: UnitQCParams):
    """Per-cluster recorded-region labels using the SAME depth-override / atlas
    logic as `select_region_units`, but returning a label per cluster instead
    of filtering.

    Returns (allen_acronyms, beryl_acronyms, is_midbrain, is_cortical,
    used_depth_override), each array indexed by cluster index 0..n_clusters-1.
    For depth-override PIDs (no histology) `is_midbrain` is depth <= threshold
    (the blanket 'midbrain' label) and `is_cortical` is its complement.
    """
    acr = sb.brain_acronyms_percluster
    n = len(acr)
    depths = np.asarray(sb.clusters.depths, dtype=float)
    is_mid = np.zeros(n, dtype=bool)
    is_cort = np.zeros(n, dtype=bool)
    beryl = _beryl_acronyms_from_allen(br, acr)

    if sb.pid in p.DEPTH_THRESHOLD_OVERRIDES:
        thr = p.DEPTH_THRESHOLD_OVERRIDES[sb.pid]
        is_mid = depths <= thr
        is_cort = depths > thr
        return acr, beryl, is_mid, is_cort, True

    # Atlas-ancestry path (histology available).
    try:
        isocortex_id = br.acronym2id('Isocortex')[0]
        region_ids = br.acronym2id(acr)
        for ci, rid in enumerate(region_ids):
            try:
                if isocortex_id in br.ancestors(rid).id:
                    is_cort[ci] = True
            except Exception:
                pass
    except Exception as e:
        print(f'region ancestry lookup failed for {sb.pid}: {e}')
    snr = np.asarray([str(a) == 'SNr' for a in acr], dtype=bool)
    is_mid = (~is_cort) & (~snr)
    return acr, beryl, is_mid, is_cort, False


def unit_qc_table(sb: SessionBundle, p: UnitQCParams, br, cluster_ids, *,
                  laser_onsets=None, qc_dir=None,
                  manual_light_artifact_unit_ids=None):
    """Per-unit QC metrics for `cluster_ids`, computed but NOT applied.

    Returns a DataFrame indexed by cluster id with columns:
      IBL_label, presence_ratio, depth, allen_region, beryl_region, is_midbrain,
      is_cortical, used_depth_override, pt_ratio, peak_before_trough,
      ax_unit, light_artifact_auto, light_artifact_manual,
      waveform_amplitude_outlier.

    Designed for a 'compute everything, exclude later' pipeline: every column
    is a recorded criterion the caller can filter on downstream.
    """
    cluster_ids = np.asarray(cluster_ids, dtype=int)
    manual_ids = set(int(x) for x in (manual_light_artifact_unit_ids or []))

    acr, beryl, is_mid, is_cort, used_override = recorded_region_flags(sb, br, p)

    # Waveform shape metrics (pt ratio, peak-before-trough, axonal flag).
    wf_idx = None
    if sb.waveforms is not None and len(cluster_ids):
        try:
            from waveform_classify import (compute_metrics_for_population,
                                           classify_axonal_classical)
            mdf, _, _ = compute_metrics_for_population(cluster_ids, sb.waveforms)
            if len(mdf):
                mdf = classify_axonal_classical(mdf, pt_ratio_threshold=p.axonal_pt_ratio_threshold)
                wf_idx = mdf.set_index('cluster_id')
        except Exception as e:
            print(f'waveform metrics failed for {sb.pid}: {e}')

    # Auto light-artifact + amplitude-outlier flag sets.
    light_auto = set()
    if laser_onsets is not None and len(np.asarray(laser_onsets)) > 0:
        try:
            ids, _ = detect_light_artifact_units(
                sb.spikes['times'], sb.spikes['clusters'], cluster_ids,
                np.asarray(laser_onsets), qc_dir, sb.pid, p)
            light_auto = set(int(x) for x in ids)
        except Exception as e:
            print(f'light-artifact detection failed for {sb.pid}: {e}')
    amp_out = set()
    if sb.waveforms is not None and len(cluster_ids):
        try:
            ids, _ = waveform_amplitude_outlier_qc(cluster_ids, sb.waveforms, qc_dir, sb.pid, p)
            amp_out = set(int(x) for x in ids)
        except Exception as e:
            print(f'amplitude QC failed for {sb.pid}: {e}')

    presence = sb.clusters['presence_ratio'] if 'presence_ratio' in sb.clusters else None

    rows = []
    for j in cluster_ids:
        jj = int(j)
        row = {
            'clustnum': jj,
            'IBL_label': float(sb.clusters_labels[jj]),
            'presence_ratio': float(presence[jj]) if presence is not None else np.nan,
            'depth': float(sb.clusters.depths[jj]),
            'allen_region': str(acr[jj]),
            'beryl_region': str(beryl[jj]),
            'is_midbrain': bool(is_mid[jj]),
            'is_cortical': bool(is_cort[jj]),
            'used_depth_override': bool(used_override),
            'light_artifact_auto': int(jj in light_auto),
            'light_artifact_manual': int(jj in manual_ids),
            'waveform_amplitude_outlier': int(jj in amp_out),
        }
        if wf_idx is not None and jj in wf_idx.index:
            row['pt_ratio'] = float(wf_idx.loc[jj, 'pt_ratio'])
            row['peak_before_trough'] = bool(wf_idx.loc[jj, 'peak_before_trough'])
            row['ax_unit'] = int(bool(wf_idx.loc[jj, 'is_axonal']))
        else:
            row['pt_ratio'] = np.nan
            row['peak_before_trough'] = False
            row['ax_unit'] = 0
        rows.append(row)
    return pd.DataFrame(rows).set_index('clustnum')


def select_units(sb: SessionBundle, p: UnitQCParams, qc_dir, br, *,
                 manual_light_artifact_unit_ids=None,
                 laser_onsets=None,
                 figure_prefix=None) -> Tuple[np.ndarray, List[dict], Dict[str, np.ndarray]]:
    """Run the full unit cascade EXCEPT drift removal (which needs the binned
    tensor and is applied by the caller via `compute_drift_unit_ids`).

    `br` is a BrainRegions object, used for atlas-ancestry region selection.
    `figure_prefix` controls the axonal-classification figure filename; if None
    it defaults to the first 12 chars of the PID.

    Returns
    -------
    cluster_ids : np.ndarray
        Surviving cluster IDs.
    qc_filter_rows : list of dict
        Cascade bookkeeping ({'stage', 'n_units'}), ready for save_filter_cascade.
    flagged : dict
        {'auto_light_artifact', 'waveform_amplitude_outlier', 'axonal'} arrays
        of cluster IDs flagged at each stage (whether or not removed).
    """
    manual_light_artifact_unit_ids = np.asarray(
        manual_light_artifact_unit_ids if manual_light_artifact_unit_ids is not None else [],
        dtype=int,
    )
    qc_filter_rows = []
    flagged = {
        'auto_light_artifact': np.asarray([], dtype=int),
        'waveform_amplitude_outlier': np.asarray([], dtype=int),
        'axonal': np.asarray([], dtype=int),
    }

    # 1. Quality label + manual light-artifact removal.
    cluster_ids = np.where(sb.clusters_labels >= p.IBL_quality_label_threshold)[0]
    cluster_ids = np.setdiff1d(cluster_ids, manual_light_artifact_unit_ids)
    qc_filter_rows.append({'stage': 'IBL quality + manual artifacts', 'n_units': int(len(cluster_ids))})

    # 2. Automatic light-artifact detection (needs laser onsets).
    if laser_onsets is not None and len(np.asarray(laser_onsets)) > 0:
        auto_ids, _ = detect_light_artifact_units(
            sb.spikes['times'], sb.spikes['clusters'], cluster_ids,
            np.asarray(laser_onsets), qc_dir, sb.pid, p,
        )
        flagged['auto_light_artifact'] = auto_ids
        print(f'Automatic light-artifact detector flagged {len(auto_ids)} units')
        if p.remove_light_artifact_units == 1 and len(auto_ids) > 0:
            cluster_ids = np.setdiff1d(cluster_ids, auto_ids)
        qc_filter_rows.append({'stage': 'auto light artifact filter', 'n_units': int(len(cluster_ids))})

    # 3. Waveform-amplitude outlier QC.
    if sb.waveforms is not None:
        amp_ids, _ = waveform_amplitude_outlier_qc(cluster_ids, sb.waveforms, qc_dir, sb.pid, p)
        flagged['waveform_amplitude_outlier'] = amp_ids
        print(f'Waveform amplitude QC flagged {len(amp_ids)} units')
        if p.remove_waveform_amplitude_outliers == 1 and len(amp_ids) > 0:
            cluster_ids = np.setdiff1d(cluster_ids, amp_ids)
        qc_filter_rows.append({'stage': 'waveform amplitude filter', 'n_units': int(len(cluster_ids))})

    # 4. Region selection.
    cluster_ids = select_region_units(cluster_ids, sb, br=br, p=p)
    beryl_selector = _region_selector_values(getattr(p, 'recorded_region_beryl', None))
    region_stage = f'region filter: {p.analyze_region}'
    if beryl_selector is not None:
        region_stage += f', Beryl in {sorted(beryl_selector)}'
    qc_filter_rows.append({'stage': region_stage, 'n_units': int(len(cluster_ids))})

    # 5. Axonal classification (canonical waveform templates).
    if p.remove_axonal_units == 1 and sb.waveforms is not None:
        ax_prefix = figure_prefix if figure_prefix is not None else sb.pid[:12]
        try:
            axonal_ids, _ = classify_and_plot_axonal_units(
                cluster_ids, sb.waveforms,
                save_path=p.figures_path if p.save_figures else None,
                prefix=ax_prefix,
                title=f'Axonal classification - PID: {sb.pid[:12]}...',
                pt_ratio_threshold=p.axonal_pt_ratio_threshold,
            )
            flagged['axonal'] = np.asarray(axonal_ids, dtype=int)
            cluster_ids = np.setdiff1d(cluster_ids, axonal_ids)
            print(f'{len(axonal_ids)} axonal units removed (classical criterion)')
        except Exception as e:
            print(f'Axonal classification failed: {e}')
    qc_filter_rows.append({'stage': 'axonal filter', 'n_units': int(len(cluster_ids))})

    # 6. Presence ratio.
    if 'presence_ratio' in sb.clusters:
        n_before = len(cluster_ids)
        stable_units = np.where(sb.clusters['presence_ratio'] > p.presence_threshold)[0]
        cluster_ids = np.intersect1d(cluster_ids, stable_units)
        print(f'Removed {n_before - len(cluster_ids)} units due to low presence ratio.')
        qc_filter_rows.append({'stage': 'presence ratio filter', 'n_units': int(len(cluster_ids))})

    return cluster_ids, qc_filter_rows, flagged


# =====================================================================
# Binning (shared so both pipelines bin identically)
# =====================================================================
def build_binned_X(spike_times, spike_clusters, selected_cluster_ids, align_times,
                   t_before, t_after, bin_size, as_rate=False):
    """Build X with shape (n_trials, n_time, n_neurons) from flat spike arrays.
    Lifted verbatim from CD so the binning grid (with an exact 0.0 sample) is
    identical across pipelines.
    """
    spike_times = np.asarray(spike_times)
    spike_clusters = np.asarray(spike_clusters)
    neuron_ids = np.asarray(selected_cluster_ids).astype(int)
    align_times = np.asarray(align_times)

    n_neurons = neuron_ids.size
    id_to_col = pd.Series(np.arange(n_neurons), index=neuron_ids)

    n_before = int(round(t_before / bin_size))
    n_after = int(round(t_after / bin_size))
    idx = np.arange(-n_before, n_after, dtype=int)
    time = idx.astype(np.float64) * bin_size
    bin_edges = np.concatenate([time, [time[-1] + bin_size]])
    n_time = time.size
    n_trials = align_times.size

    X = np.zeros((n_trials, n_time, n_neurons), dtype=np.float32)
    for j, t_align in enumerate(align_times):
        t0 = t_align - t_before
        t1 = t_align + t_after
        m = (spike_times >= t0) & (spike_times < t1)
        if not np.any(m):
            continue
        rel_t = spike_times[m] - t_align
        cl = spike_clusters[m].astype(int)
        col = id_to_col.reindex(cl).values
        keep = ~np.isnan(col)
        if not np.any(keep):
            continue
        rel_t = rel_t[keep]
        col = col[keep].astype(int)
        H, _, _ = np.histogram2d(rel_t, col, bins=[bin_edges, np.arange(n_neurons + 1)])
        if as_rate:
            H = H / bin_size
        X[j] = H.astype(np.float32)
    return X, time, neuron_ids


def cluster_peak_waveform(waveforms, cluster_id):
    """Baseline-subtracted peak-channel mean waveform (µV) for one cluster,
    from the canonical templates. Returns None if unavailable.

    Provided so a per-cluster pipeline (e.g. BS) can plot an averaged waveform
    without the old per-spike `_phy_spikes_subset` snippets.
    """
    if waveforms is None:
        return None
    try:
        from waveform_classify import _extract_cluster_template_uv, N_BASELINE_SAMPLES
        tmpl = _extract_cluster_template_uv(waveforms['templates'], int(cluster_id))
        if tmpl is None:
            return None
        chan_max = np.max(np.abs(tmpl), axis=0)
        peak_chan = int(np.argmax(chan_max))
        wf = tmpl[:, peak_chan]
        return wf - np.mean(wf[:N_BASELINE_SAMPLES])
    except Exception:
        return None


def _qp_mean_fr_matrix(spikes, cluster_ids, gocue_times, quiescence_periods):
    """Per-trial, per-neuron mean firing rate (Hz) in each trial's enforced
    quiescence period [goCue - quiescencePeriod, goCue].

    Returns an (n_trials, n_neurons) array. Trials with a non-finite go-cue or
    quiescence value (or non-positive duration) are filled with NaN so the
    caller can drop them before correlation.
    """
    times = np.asarray(spikes['times'], dtype=float)
    clus = np.asarray(spikes['clusters'])
    gocue = np.asarray(gocue_times, dtype=float)
    qp = np.asarray(quiescence_periods, dtype=float)
    cluster_ids = np.asarray(cluster_ids)

    starts = gocue - qp
    ends = gocue
    n_trials = len(gocue)
    n_neurons = len(cluster_ids)
    fr = np.full((n_trials, n_neurons), np.nan)

    col = {int(c): k for k, c in enumerate(cluster_ids)}
    sel = np.isin(clus, cluster_ids)
    times_sel = times[sel]
    clus_sel = clus[sel]
    order = np.argsort(times_sel)
    times_sorted = times_sel[order]
    clus_sorted = clus_sel[order]

    for t in range(n_trials):
        s, e = starts[t], ends[t]
        if not (np.isfinite(s) and np.isfinite(e)) or e <= s:
            continue
        dur = e - s
        lo = np.searchsorted(times_sorted, s, side='left')
        hi = np.searchsorted(times_sorted, e, side='right')
        fr[t, :] = 0.0
        if hi > lo:
            uniq, counts = np.unique(clus_sorted[lo:hi], return_counts=True)
            for u, c in zip(uniq, counts):
                k = col.get(int(u))
                if k is not None:
                    fr[t, k] = c / dur
    return fr


def compute_qp_drift_metrics(spikes, cluster_ids, gocue_times, quiescence_periods,
                             block_ids, drift_threshold=0.35):
    """Compute quiescence-period firing-rate drift metrics per unit.

    This isolates a guaranteed motor-free epoch, so the drift test reflects
    slow firing-rate changes across the session rather than movement that a
    fixed post-onset window would capture across variable trial structure.

    The same decision rule as functions_optostim.get_drift_indices is used:
    a unit is flagged when abs(Spearman(FR, trial index)) > drift_threshold
    and that absolute time correlation is stronger than the absolute block
    correlation.
    """
    from scipy.stats import spearmanr

    cluster_ids = np.asarray(cluster_ids)
    if len(cluster_ids) == 0:
        return pd.DataFrame(columns=[
            'cluster_id', 'rho_time', 'rho_block', 'abs_rho_time',
            'abs_rho_block', 'flagged_drift',
        ])
    fr = _qp_mean_fr_matrix(spikes, cluster_ids, gocue_times, quiescence_periods)
    block = np.asarray(block_ids)
    valid = ~np.any(np.isnan(fr), axis=1)
    trial_indices = np.arange(valid.sum())
    rows = []
    for local_i, cid in enumerate(cluster_ids):
        if valid.sum() < 3:
            rho_time, rho_block = np.nan, np.nan
        else:
            y = fr[valid, local_i]
            try:
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    rho_time, _ = spearmanr(y, trial_indices)
            except Exception:
                rho_time = np.nan
            try:
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    rho_block, _ = spearmanr(y, block[valid])
            except Exception:
                rho_block = np.nan
        abs_time = abs(rho_time) if np.isfinite(rho_time) else np.nan
        abs_block = abs(rho_block) if np.isfinite(rho_block) else np.nan
        flagged = bool(np.isfinite(abs_time) and np.isfinite(abs_block)
                       and abs_time > float(drift_threshold)
                       and abs_time > abs_block)
        rows.append({
            'cluster_id': int(cid),
            'n_valid_trials_for_drift': int(valid.sum()),
            'rho_time': float(rho_time) if np.isfinite(rho_time) else np.nan,
            'rho_block': float(rho_block) if np.isfinite(rho_block) else np.nan,
            'abs_rho_time': float(abs_time) if np.isfinite(abs_time) else np.nan,
            'abs_rho_block': float(abs_block) if np.isfinite(abs_block) else np.nan,
            'drift_threshold': float(drift_threshold),
            'flagged_drift': flagged,
        })
    return pd.DataFrame(rows)


def compute_qp_drift_unit_ids(spikes, cluster_ids, gocue_times, quiescence_periods,
                              block_ids, drift_threshold=0.35):
    """Flag drift units using mean firing rate in each trial's enforced
    quiescence period [goCue - quiescencePeriod, goCue].

    Returns the array of flagged CLUSTER IDs (not local indices).
    """
    df = compute_qp_drift_metrics(
        spikes, cluster_ids, gocue_times, quiescence_periods,
        block_ids, drift_threshold=drift_threshold,
    )
    if len(df) == 0 or 'flagged_drift' not in df:
        return np.asarray([], dtype=int)
    return df.loc[df['flagged_drift'], 'cluster_id'].to_numpy(dtype=int)


def _safe_float(x):
    try:
        x = float(x)
    except Exception:
        return np.nan
    return x if np.isfinite(x) else np.nan


def _max_true_run(mask):
    mask = np.asarray(mask, dtype=bool)
    best = cur = 0
    for val in mask:
        if val:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return int(best)


def _segment_slices(n_obs, n_segments, min_trials_per_segment):
    n_obs = int(n_obs)
    n_segments = max(1, int(n_segments or 1))
    min_trials_per_segment = max(1, int(min_trials_per_segment or 1))
    if n_obs < min_trials_per_segment:
        return []
    return [seg for seg in np.array_split(np.arange(n_obs), n_segments)
            if len(seg) >= min_trials_per_segment]


def _fractional_range(values, denom):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    denom = abs(_safe_float(denom))
    if values.size < 2 or not np.isfinite(denom) or denom <= 0:
        return np.nan
    return float((np.nanmax(values) - np.nanmin(values)) / denom)


def _fractional_cv(values, denom):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    denom = abs(_safe_float(denom))
    if values.size < 2 or not np.isfinite(denom) or denom <= 0:
        return np.nan
    return float(np.nanstd(values, ddof=1) / denom)


def _threshold_failed_gt(value, threshold):
    if threshold is None:
        return False
    try:
        value = float(value)
        threshold = float(threshold)
    except Exception:
        return False
    if not np.isfinite(threshold):
        return False
    if np.isposinf(value):
        return True
    return bool(np.isfinite(value) and value > threshold)


def _threshold_failed_lt(value, threshold):
    if threshold is None:
        return False
    try:
        value = float(value)
        threshold = float(threshold)
    except Exception:
        return False
    if not np.isfinite(threshold):
        return False
    if np.isneginf(value):
        return True
    return bool(np.isfinite(value) and value < threshold)


def _block_time_fold_ids(n_obs, n_folds, mode='interleaved'):
    n_obs = int(n_obs)
    n_folds = max(2, min(int(n_folds or 2), n_obs))
    mode = str(mode or 'interleaved').lower()
    fold_ids = np.zeros(n_obs, dtype=int)
    if mode == 'blocked':
        for fold, idx in enumerate(np.array_split(np.arange(n_obs), n_folds)):
            fold_ids[idx] = fold
    elif mode == 'random':
        rng = np.random.default_rng(0)
        order = rng.permutation(n_obs)
        for fold, idx in enumerate(np.array_split(order, n_folds)):
            fold_ids[idx] = fold
    else:
        fold_ids = np.arange(n_obs, dtype=int) % n_folds
    return fold_ids


def _standardize_train_test(x_train, x_test):
    x_train = np.asarray(x_train, dtype=float)
    x_test = np.asarray(x_test, dtype=float)
    if x_train.ndim != 2:
        x_train = np.reshape(x_train, (x_train.shape[0], -1))
    if x_test.ndim != 2:
        x_test = np.reshape(x_test, (x_test.shape[0], -1))
    if x_train.shape[1] == 0:
        return x_train, x_test, np.asarray([]), np.asarray([])
    mu = np.nanmean(x_train, axis=0)
    sd = np.nanstd(x_train, axis=0)
    sd[~np.isfinite(sd) | (sd <= 0)] = 1.0
    return (x_train - mu) / sd, (x_test - mu) / sd, mu, sd


def _ridge_beta(x_train, y_train, alpha):
    x_train = np.asarray(x_train, dtype=float)
    y_train = np.asarray(y_train, dtype=float)
    if x_train.ndim != 2:
        x_train = np.reshape(x_train, (x_train.shape[0], -1))
    x_aug = np.column_stack([np.ones(x_train.shape[0]), x_train])
    alpha = max(0.0, float(alpha or 0.0))
    penalty = np.eye(x_aug.shape[1]) * alpha
    penalty[0, 0] = 0.0
    try:
        return np.linalg.solve(x_aug.T @ x_aug + penalty, x_aug.T @ y_train)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(x_aug.T @ x_aug + penalty) @ x_aug.T @ y_train


def _cv_r2_for_design(y, design, fold_ids, ridge_alpha):
    y = np.asarray(y, dtype=float)
    design = np.asarray(design, dtype=float)
    if design.ndim != 2:
        design = np.reshape(design, (design.shape[0], -1))
    fold_ids = np.asarray(fold_ids, dtype=int)
    pred = np.full(y.shape, np.nan, dtype=float)
    for fold in np.unique(fold_ids):
        test = fold_ids == fold
        train = ~test
        if np.sum(train) < 2 or np.sum(test) == 0:
            continue
        x_train, x_test, _, _ = _standardize_train_test(design[train], design[test])
        beta = _ridge_beta(x_train, y[train], ridge_alpha)
        x_test_aug = np.column_stack([np.ones(np.sum(test)), x_test])
        pred[test] = x_test_aug @ beta
    valid_pred = np.isfinite(pred) & np.isfinite(y)
    if np.sum(valid_pred) < 3:
        return np.nan, pred
    denom = float(np.nansum((y[valid_pred] - np.nanmean(y[valid_pred])) ** 2))
    if denom <= 0:
        return np.nan, pred
    sse = float(np.nansum((y[valid_pred] - pred[valid_pred]) ** 2))
    return float(1.0 - sse / denom), pred


def _full_model_coefficients(y, design, ridge_alpha):
    y = np.asarray(y, dtype=float)
    design = np.asarray(design, dtype=float)
    if design.ndim != 2:
        design = np.reshape(design, (design.shape[0], -1))
    x_std, _, _, _ = _standardize_train_test(design, design)
    beta = _ridge_beta(x_std, y, ridge_alpha)
    return beta[1:] if beta.size > 1 else np.asarray([], dtype=float)


def compute_qp_block_time_model_metrics(
        spikes, cluster_ids, gocue_times, quiescence_periods, block_ids, *,
        min_trials=40, min_trials_per_block=8, n_folds=5,
        fold_mode='interleaved', time_degree=3, ridge_alpha=1e-6,
        flag_logic='all',
        max_qp_unique_time_r2=None, max_qp_time_over_block_ratio=None,
        min_qp_unique_block_r2=None, min_qp_block_time_preference=None):
    """Cross-validated block-vs-time model comparison for QP firing rates.

    Each unit is scored with four models: null, block-only, time-only, and
    block+time. Unique block R2 is full minus time-only; unique time R2 is full
    minus block-only. Thresholds default to None, which reports metrics without
    flagging units.
    """
    cluster_ids = np.asarray(cluster_ids, dtype=int)
    empty_cols = [
        'cluster_id', 'n_valid_trials_block_time_model',
        'n_block_time_model_block0', 'n_block_time_model_block1',
        'block_time_model_n_folds', 'block_time_model_fold_mode',
        'block_time_model_time_degree', 'block_time_model_ridge_alpha',
        'block_time_model_flag_logic', 'block_time_model_n_active_thresholds',
        'block_time_model_n_failed_thresholds',
        'qp_cv_r2_null', 'qp_cv_r2_block', 'qp_cv_r2_time', 'qp_cv_r2_full',
        'qp_block_only_r2', 'qp_time_only_r2',
        'qp_unique_block_r2', 'qp_unique_time_r2',
        'qp_time_over_block_ratio', 'qp_block_time_preference',
        'qp_full_model_block_coef', 'qp_full_model_time_coef_norm',
        'flagged_block_time_model', 'block_time_model_reasons',
    ]
    if len(cluster_ids) == 0:
        return pd.DataFrame(columns=empty_cols)

    flag_logic = str(flag_logic or 'all').lower()
    if flag_logic not in ('all', 'any'):
        flag_logic = 'all'
    active_thresholds = [
        max_qp_unique_time_r2,
        max_qp_time_over_block_ratio,
        min_qp_unique_block_r2,
        min_qp_block_time_preference,
    ]
    n_active_thresholds = int(sum(thr is not None for thr in active_thresholds))

    fr = _qp_mean_fr_matrix(spikes, cluster_ids, gocue_times, quiescence_periods)
    block = np.asarray(block_ids, dtype=float)
    if block.size != fr.shape[0]:
        raise ValueError('block_ids must have one entry per QP trial.')

    rows = []
    eps = 1e-9
    time_degree = max(1, int(time_degree or 1))
    for local_i, cid in enumerate(cluster_ids):
        y_full = fr[:, local_i]
        valid = np.isfinite(y_full) & np.isfinite(block)
        y = y_full[valid].astype(float)
        b = block[valid].astype(int)
        trial_pos = np.flatnonzero(valid).astype(float)

        row = {col: np.nan for col in empty_cols}
        row['cluster_id'] = int(cid)
        row['n_valid_trials_block_time_model'] = int(y.size)
        row['n_block_time_model_block0'] = int(np.sum(b == 0))
        row['n_block_time_model_block1'] = int(np.sum(b == 1))
        row['block_time_model_n_folds'] = int(n_folds)
        row['block_time_model_fold_mode'] = str(fold_mode)
        row['block_time_model_time_degree'] = int(time_degree)
        row['block_time_model_ridge_alpha'] = float(ridge_alpha or 0.0)
        row['block_time_model_flag_logic'] = flag_logic
        row['block_time_model_n_active_thresholds'] = n_active_thresholds
        row['block_time_model_n_failed_thresholds'] = 0
        reasons = []

        enough_trials = y.size >= int(min_trials)
        enough_blocks = (
            np.sum(b == 0) >= int(min_trials_per_block)
            and np.sum(b == 1) >= int(min_trials_per_block)
        )
        if enough_trials and enough_blocks and np.nanstd(y) > 0:
            if trial_pos.size > 1:
                t = 2.0 * (trial_pos - np.nanmin(trial_pos)) / max(np.nanmax(trial_pos) - np.nanmin(trial_pos), eps) - 1.0
            else:
                t = np.zeros_like(trial_pos, dtype=float)
            block_design = b.reshape(-1, 1).astype(float)
            time_design = np.column_stack([t ** deg for deg in range(1, time_degree + 1)])
            null_design = np.zeros((y.size, 0), dtype=float)
            full_design = np.column_stack([block_design, time_design])
            fold_ids = _block_time_fold_ids(y.size, n_folds, fold_mode)

            r2_null, _ = _cv_r2_for_design(y, null_design, fold_ids, ridge_alpha)
            r2_block, _ = _cv_r2_for_design(y, block_design, fold_ids, ridge_alpha)
            r2_time, _ = _cv_r2_for_design(y, time_design, fold_ids, ridge_alpha)
            r2_full, _ = _cv_r2_for_design(y, full_design, fold_ids, ridge_alpha)
            row['qp_cv_r2_null'] = r2_null
            row['qp_cv_r2_block'] = r2_block
            row['qp_cv_r2_time'] = r2_time
            row['qp_cv_r2_full'] = r2_full
            row['qp_block_only_r2'] = r2_block - r2_null if np.isfinite(r2_block) and np.isfinite(r2_null) else np.nan
            row['qp_time_only_r2'] = r2_time - r2_null if np.isfinite(r2_time) and np.isfinite(r2_null) else np.nan
            row['qp_unique_block_r2'] = r2_full - r2_time if np.isfinite(r2_full) and np.isfinite(r2_time) else np.nan
            row['qp_unique_time_r2'] = r2_full - r2_block if np.isfinite(r2_full) and np.isfinite(r2_block) else np.nan

            ub = _safe_float(row['qp_unique_block_r2'])
            ut = _safe_float(row['qp_unique_time_r2'])
            if np.isfinite(ut) and np.isfinite(ub):
                if ut <= 0:
                    row['qp_time_over_block_ratio'] = 0.0
                elif ub <= eps:
                    row['qp_time_over_block_ratio'] = np.inf
                else:
                    row['qp_time_over_block_ratio'] = float(ut / ub)
                denom = abs(ub) + abs(ut)
                if denom > eps:
                    row['qp_block_time_preference'] = float((ub - ut) / denom)

            beta = _full_model_coefficients(y, full_design, ridge_alpha)
            if beta.size:
                row['qp_full_model_block_coef'] = float(beta[0])
                row['qp_full_model_time_coef_norm'] = float(np.linalg.norm(beta[1:])) if beta.size > 1 else 0.0

            checks = [
                ('qp_unique_time_r2', max_qp_unique_time_r2, '>'),
                ('qp_time_over_block_ratio', max_qp_time_over_block_ratio, '>'),
                ('qp_unique_block_r2', min_qp_unique_block_r2, '<'),
                ('qp_block_time_preference', min_qp_block_time_preference, '<'),
            ]
            for metric, threshold, direction in checks:
                if direction == '>':
                    failed = _threshold_failed_gt(row.get(metric), threshold)
                else:
                    failed = _threshold_failed_lt(row.get(metric), threshold)
                if failed:
                    reasons.append(metric)
            row['block_time_model_n_failed_thresholds'] = int(len(reasons))
        elif not enough_trials:
            row['block_time_model_reasons'] = 'insufficient_trials'
        elif not enough_blocks:
            row['block_time_model_reasons'] = 'insufficient_trials_per_block'
        else:
            row['block_time_model_reasons'] = 'zero_variance_fr'

        if n_active_thresholds == 0:
            row['flagged_block_time_model'] = False
        elif flag_logic == 'any':
            row['flagged_block_time_model'] = bool(reasons)
        else:
            row['flagged_block_time_model'] = bool(len(reasons) == n_active_thresholds)
        if reasons:
            row['block_time_model_reasons'] = ';'.join(reasons)
        elif not isinstance(row.get('block_time_model_reasons'), str):
            row['block_time_model_reasons'] = ''
        rows.append(row)

    return pd.DataFrame(rows)


def save_qp_block_time_model_qc(metrics_df, qc_dir, pid):
    """Save CSV and compact visual QC for the block-vs-time model metric."""
    if metrics_df is None or len(metrics_df) == 0:
        return
    qc_dir = Path(qc_dir)
    qc_dir.mkdir(parents=True, exist_ok=True)
    try:
        metrics_df.to_csv(qc_dir / f'{pid}_qp_block_time_model_metrics.csv', index=False)
    except Exception as e:
        print(f'QP block-vs-time model CSV save failed for {pid}: {e}')

    try:
        flagged = (
            metrics_df['flagged_block_time_model'].to_numpy(bool)
            if 'flagged_block_time_model' in metrics_df.columns
            else np.zeros(len(metrics_df), dtype=bool)
        )
        fig, axes = plt.subplots(2, 2, figsize=(9, 7))
        color = np.where(flagged, 'tab:red', '0.45')
        alpha = np.where(flagged, 0.85, 0.55)

        ax = axes[0, 0]
        x = metrics_df['qp_unique_block_r2'].to_numpy(float)
        y = metrics_df['qp_unique_time_r2'].to_numpy(float)
        finite = np.isfinite(x) & np.isfinite(y)
        if np.any(finite):
            ax.scatter(x[finite], y[finite], c=color[finite], alpha=alpha[finite], s=24, edgecolors='none')
            lim = np.nanmax(np.abs(np.r_[x[finite], y[finite], 0.01]))
            ax.plot([-lim, lim], [-lim, lim], color='0.65', linestyle='--', lw=1)
            ax.axhline(0, color='0.75', lw=0.8)
            ax.axvline(0, color='0.75', lw=0.8)
            ax.set_xlim(-0.05 * lim, lim * 1.05)
            ax.set_ylim(-0.05 * lim, lim * 1.05)
        ax.set_xlabel('Unique block CV R2')
        ax.set_ylabel('Unique time CV R2')
        ax.set_title('Full model increments')

        ax = axes[0, 1]
        xb = metrics_df['qp_cv_r2_block'].to_numpy(float)
        xt = metrics_df['qp_cv_r2_time'].to_numpy(float)
        finite = np.isfinite(xb) & np.isfinite(xt)
        if np.any(finite):
            ax.scatter(xb[finite], xt[finite], c=color[finite], alpha=alpha[finite], s=24, edgecolors='none')
            lo = np.nanmin(np.r_[xb[finite], xt[finite], 0])
            hi = np.nanmax(np.r_[xb[finite], xt[finite], 0.01])
            ax.plot([lo, hi], [lo, hi], color='0.65', linestyle='--', lw=1)
        ax.set_xlabel('Block-only CV R2')
        ax.set_ylabel('Time-only CV R2')
        ax.set_title('Single-predictor comparison')

        ax = axes[1, 0]
        pref = metrics_df['qp_block_time_preference'].to_numpy(float)
        finite = np.isfinite(pref)
        if np.any(finite):
            ax.hist(pref[finite & ~flagged], bins=30, color='0.65', edgecolor='0.25')
            if np.any(finite & flagged):
                ax.hist(pref[finite & flagged], bins=30, color='tab:red', alpha=0.65)
        ax.axvline(0, color='0.25', linestyle='--', lw=1)
        ax.set_xlabel('Block-time preference\n(+ block, - time)')
        ax.set_ylabel('Unit count')
        ax.set_title('Preference distribution')

        ax = axes[1, 1]
        ratio = metrics_df['qp_time_over_block_ratio'].replace([np.inf, -np.inf], np.nan).to_numpy(float)
        finite = np.isfinite(ratio)
        if np.any(finite):
            order = np.argsort(ratio[finite])
            xx = np.arange(np.sum(finite))
            rr = ratio[finite][order]
            ff = flagged[finite][order]
            ax.scatter(xx[~ff], rr[~ff], color='0.45', alpha=0.55, s=18, edgecolors='none')
            if np.any(ff):
                ax.scatter(xx[ff], rr[ff], color='tab:red', alpha=0.85, s=24, edgecolors='none')
        ax.set_xlabel('Units sorted by ratio')
        ax.set_ylabel('Unique time / unique block')
        ax.set_title('Time dominance')

        fig.suptitle(f'QP block-vs-time model QC\n{pid[:12]} flagged={int(np.sum(flagged))}/{len(metrics_df)}')
        fig.tight_layout()
        fig.savefig(qc_dir / f'{pid}_qp_block_time_model_qc.png', dpi=200)
        plt.close(fig)
    except Exception as e:
        print(f'QP block-vs-time model QC plot failed for {pid}: {e}')


def compute_qp_nonstationarity_metrics(
        spikes, cluster_ids, gocue_times, quiescence_periods, block_ids, *,
        n_segments=6, min_trials=30, min_trials_per_segment=8,
        min_trials_per_block_segment=3, low_fr_fraction_of_median=0.2,
        min_median_fr_hz=0.1, max_qp_fr_segment_range_frac=None,
        max_qp_resid_drift_range_frac=None, max_qp_resid_drift_cv=None,
        max_qp_resid_abs_rho_time=None, max_qp_low_activity_fraction=None,
        max_qp_max_low_activity_run=None,
        min_qp_block_effect_sign_consistency=None,
        max_qp_block_effect_segment_cv=None,
        max_qp_block_effect_dominance=None,
        return_trial_metrics=False):
    """Compute QP firing-rate nonstationarity metrics after accounting for block.

    These metrics are intentionally different from `compute_qp_drift_metrics`.
    The Spearman drift metric catches monotonic FR-vs-trial trends. This helper
    targets non-monotonic drift, dropout epochs, and "one segment dominates the
    apparent block code" artifacts while preserving genuine block selectivity:

      * block-specific medians are subtracted before residual drift metrics;
      * block effects are recomputed in contiguous session segments;
      * low-activity runs are defined relative to each unit's own QP median FR.

    Threshold arguments default to None. In that case the metrics are reported
    but `flagged_nonstationary` remains False.
    """
    from scipy.stats import spearmanr
    import warnings

    cluster_ids = np.asarray(cluster_ids, dtype=int)
    empty_cols = [
        'cluster_id', 'n_valid_trials_nonstationarity', 'qp_fr_median',
        'qp_fr_mean', 'qp_fr_segment_range_frac', 'qp_fr_segment_cv',
        'qp_resid_drift_range_frac', 'qp_resid_drift_cv',
        'qp_resid_abs_rho_time', 'qp_low_activity_fraction',
        'qp_max_low_activity_run', 'qp_n_block_effect_segments',
        'qp_block_effect_global', 'qp_block_effect_segment_mean',
        'qp_block_effect_segment_cv', 'qp_block_effect_sign_consistency',
        'qp_block_effect_dominance', 'flagged_nonstationary',
        'nonstationarity_reasons',
    ]
    if len(cluster_ids) == 0:
        out = pd.DataFrame(columns=empty_cols)
        if return_trial_metrics:
            return out, pd.DataFrame()
        return out

    fr = _qp_mean_fr_matrix(spikes, cluster_ids, gocue_times, quiescence_periods)
    block = np.asarray(block_ids, dtype=float)
    if block.size != fr.shape[0]:
        raise ValueError('block_ids must have one entry per QP trial.')

    trial_valid = np.isfinite(block) & ~np.all(np.isnan(fr), axis=1)
    active_floor = float(min_median_fr_hz if min_median_fr_hz is not None else 0.1)
    active_floor = max(active_floor, 0.0)
    trial_df = pd.DataFrame({
        'trial_position': np.arange(fr.shape[0], dtype=int),
        'block_id': block,
        'qp_population_mean_fr': np.nanmean(fr, axis=1),
        'qp_population_median_fr': np.nanmedian(fr, axis=1),
        'qp_active_unit_fraction': np.nanmean(fr > active_floor, axis=1),
        'qp_n_finite_units': np.sum(np.isfinite(fr), axis=1),
        'qp_trial_valid': trial_valid,
    })

    rows = []
    for local_i, cid in enumerate(cluster_ids):
        y_full = fr[:, local_i]
        valid = np.isfinite(y_full) & np.isfinite(block)
        y = y_full[valid]
        b = block[valid].astype(int)
        n_valid = int(y.size)

        row = {col: np.nan for col in empty_cols}
        row['cluster_id'] = int(cid)
        row['n_valid_trials_nonstationarity'] = n_valid
        row['nonstationarity_n_segments'] = int(n_segments)
        row['nonstationarity_min_trials'] = int(min_trials)
        row['nonstationarity_low_fr_fraction_of_median'] = float(low_fr_fraction_of_median)
        row['nonstationarity_min_median_fr_hz'] = float(min_median_fr_hz)

        reasons = []
        if n_valid >= int(min_trials):
            med_fr = float(np.nanmedian(y))
            mean_fr = float(np.nanmean(y))
            denom = max(abs(med_fr), float(min_median_fr_hz or 0.0))
            if denom <= 0:
                denom = np.nan
            row['qp_fr_median'] = med_fr
            row['qp_fr_mean'] = mean_fr

            resid = y.astype(float).copy()
            for blk in np.unique(b):
                blk_mask = b == blk
                if np.any(blk_mask):
                    resid[blk_mask] -= np.nanmedian(y[blk_mask])

            segments = _segment_slices(
                n_valid, n_segments, min_trials_per_segment,
            )
            fr_seg_means = [np.nanmean(y[seg]) for seg in segments]
            resid_seg_means = [np.nanmean(resid[seg]) for seg in segments]
            row['qp_fr_segment_range_frac'] = _fractional_range(fr_seg_means, denom)
            row['qp_fr_segment_cv'] = _fractional_cv(fr_seg_means, denom)
            row['qp_resid_drift_range_frac'] = _fractional_range(resid_seg_means, denom)
            row['qp_resid_drift_cv'] = _fractional_cv(resid_seg_means, denom)

            try:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    rho_resid, _ = spearmanr(resid, np.flatnonzero(valid))
                row['qp_resid_abs_rho_time'] = abs(float(rho_resid))
            except Exception:
                row['qp_resid_abs_rho_time'] = np.nan

            if np.isfinite(med_fr) and med_fr >= float(min_median_fr_hz):
                low_thr = float(low_fr_fraction_of_median) * med_fr
                low_activity = y <= low_thr
                row['qp_low_activity_fraction'] = float(np.mean(low_activity))
                row['qp_max_low_activity_run'] = int(_max_true_run(low_activity))

            if np.sum(b == 1) > 0 and np.sum(b == 0) > 0:
                global_effect = float(np.nanmean(y[b == 1]) - np.nanmean(y[b == 0]))
                row['qp_block_effect_global'] = global_effect
                effect_segments = []
                for seg in segments:
                    bs = b[seg]
                    ys = y[seg]
                    if (np.sum(bs == 1) >= int(min_trials_per_block_segment)
                            and np.sum(bs == 0) >= int(min_trials_per_block_segment)):
                        effect_segments.append(
                            float(np.nanmean(ys[bs == 1]) - np.nanmean(ys[bs == 0]))
                        )
                effect_segments = np.asarray(effect_segments, dtype=float)
                effect_segments = effect_segments[np.isfinite(effect_segments)]
                row['qp_n_block_effect_segments'] = int(effect_segments.size)
                if effect_segments.size:
                    row['qp_block_effect_segment_mean'] = float(np.nanmean(effect_segments))
                    row['qp_block_effect_segment_cv'] = _fractional_cv(
                        effect_segments, global_effect,
                    )
                    abs_sum = float(np.nansum(np.abs(effect_segments)))
                    if abs_sum > 0:
                        row['qp_block_effect_dominance'] = float(
                            np.nanmax(np.abs(effect_segments)) / abs_sum
                        )
                    if global_effect != 0:
                        same = np.sign(effect_segments) == np.sign(global_effect)
                        row['qp_block_effect_sign_consistency'] = float(np.mean(same))

            checks = [
                ('qp_fr_segment_range_frac', max_qp_fr_segment_range_frac, '>'),
                ('qp_resid_drift_range_frac', max_qp_resid_drift_range_frac, '>'),
                ('qp_resid_drift_cv', max_qp_resid_drift_cv, '>'),
                ('qp_resid_abs_rho_time', max_qp_resid_abs_rho_time, '>'),
                ('qp_low_activity_fraction', max_qp_low_activity_fraction, '>'),
                ('qp_max_low_activity_run', max_qp_max_low_activity_run, '>'),
                ('qp_block_effect_sign_consistency',
                 min_qp_block_effect_sign_consistency, '<'),
                ('qp_block_effect_segment_cv', max_qp_block_effect_segment_cv, '>'),
                ('qp_block_effect_dominance', max_qp_block_effect_dominance, '>'),
            ]
            for metric, threshold, direction in checks:
                if direction == '>':
                    failed = _threshold_failed_gt(row.get(metric), threshold)
                else:
                    failed = _threshold_failed_lt(row.get(metric), threshold)
                if failed:
                    reasons.append(metric)

        row['flagged_nonstationary'] = bool(reasons)
        row['nonstationarity_reasons'] = ';'.join(reasons)
        rows.append(row)

    out = pd.DataFrame(rows)
    if return_trial_metrics:
        return out, trial_df
    return out


def save_qp_nonstationarity_qc(metrics_df, trial_df, qc_dir, pid):
    """Save compact per-insertion nonstationarity diagnostics."""
    if metrics_df is None or len(metrics_df) == 0:
        return
    qc_dir = Path(qc_dir)
    qc_dir.mkdir(parents=True, exist_ok=True)
    try:
        metrics_df.to_csv(qc_dir / f'{pid}_qp_nonstationarity_metrics.csv', index=False)
    except Exception as e:
        print(f'QP nonstationarity metrics CSV save failed for {pid}: {e}')
    try:
        if trial_df is not None and len(trial_df):
            trial_df.to_csv(qc_dir / f'{pid}_qp_population_activity_by_trial.csv', index=False)
    except Exception as e:
        print(f'QP population activity CSV save failed for {pid}: {e}')

    try:
        flagged = (
            metrics_df['flagged_nonstationary'].to_numpy(bool)
            if 'flagged_nonstationary' in metrics_df.columns
            else np.zeros(len(metrics_df), dtype=bool)
        )
        fig, axes = plt.subplots(2, 2, figsize=(9, 6))
        metric_specs = [
            ('qp_resid_drift_range_frac', 'Residual FR range / median FR'),
            ('qp_low_activity_fraction', 'Low-activity trial fraction'),
            ('qp_max_low_activity_run', 'Max low-activity run (trials)'),
            ('qp_block_effect_sign_consistency', 'Block-effect sign consistency'),
        ]
        for ax, (col, label) in zip(axes.ravel(), metric_specs):
            if col not in metrics_df.columns:
                ax.axis('off')
                continue
            vals = metrics_df[col].to_numpy(float)
            finite = np.isfinite(vals)
            if not np.any(finite):
                ax.text(0.5, 0.5, f'No finite {col}', ha='center', va='center')
                ax.axis('off')
                continue
            ax.hist(vals[finite], bins=30, color='0.7', edgecolor='0.25')
            fvals = vals[finite & flagged]
            if fvals.size:
                ax.hist(fvals, bins=30, color='red', alpha=0.6)
            ax.set_xlabel(label)
            ax.set_ylabel('Unit count')
        fig.suptitle(f'QP nonstationarity QC\n{pid[:12]} flagged={int(np.sum(flagged))}/{len(metrics_df)}')
        fig.tight_layout()
        fig.savefig(qc_dir / f'{pid}_qp_nonstationarity_metric_histogram.png', dpi=200)
        plt.close(fig)
    except Exception as e:
        print(f'QP nonstationarity histogram failed for {pid}: {e}')

    try:
        if trial_df is None or len(trial_df) == 0:
            return
        fig, ax1 = plt.subplots(figsize=(9, 3.5))
        x = trial_df['trial_position'].to_numpy(float)
        ax1.plot(x, trial_df['qp_population_median_fr'].to_numpy(float),
                 color='black', lw=1.3, label='median unit FR')
        ax1.plot(x, trial_df['qp_population_mean_fr'].to_numpy(float),
                 color='0.45', lw=1.0, alpha=0.8, label='mean unit FR')
        ax1.set_xlabel('Analysis trial position')
        ax1.set_ylabel('QP firing rate (Hz)')
        ax2 = ax1.twinx()
        ax2.plot(x, trial_df['qp_active_unit_fraction'].to_numpy(float),
                 color='tab:blue', lw=1.0, alpha=0.75, label='active unit fraction')
        ax2.set_ylabel('Active unit fraction')
        ax2.set_ylim(-0.02, 1.02)
        lines = ax1.get_lines() + ax2.get_lines()
        ax1.legend(lines, [ln.get_label() for ln in lines], loc='best', frameon=False)
        ax1.set_title(f'QP population activity by trial\n{pid[:12]}')
        fig.tight_layout()
        fig.savefig(qc_dir / f'{pid}_qp_population_activity_by_trial.png', dpi=200)
        plt.close(fig)
    except Exception as e:
        print(f'QP population activity plot failed for {pid}: {e}')


def compute_drift_unit_ids(spikes, cluster_ids, align_times, block_ids,
                           t_before, t_after, bin_size, drift_threshold=0.35):
    """Bin a CD-style tensor and return the CLUSTER IDs flagged as drift units.

    This wraps build_binned_X + get_drift_indices so both pipelines apply an
    identical drift criterion. CD already bins this tensor for its CD window;
    BS — which otherwise works per-cluster — can call this to bin an equivalent
    tensor over its analysed trials purely for the drift decision.

    Returns an array of cluster IDs (mapped back from local neuron indices).
    """
    cluster_ids = np.asarray(cluster_ids, dtype=int)
    X, _, _ = build_binned_X(spikes['times'], spikes['clusters'], cluster_ids,
                             align_times, t_before, t_after, bin_size, as_rate=False)
    local_idx = get_drift_indices(X, block_ids, drift_threshold=drift_threshold)
    if len(local_idx) == 0:
        return np.asarray([], dtype=int)
    return cluster_ids[local_idx]


# =====================================================================
# Trial helpers  (lifted faithfully from CD)
# =====================================================================
def normalize_trial_range(trial_spec, n_trials):
    """Convert metadata trial specs into valid 0-based indices, handling 'ALL'
    and legacy open-ended ranges (e.g. range(493, 9999)) by clipping."""
    if isinstance(trial_spec, str):
        if trial_spec.upper() == 'ALL':
            return np.arange(n_trials, dtype=int)
        raise ValueError(f"Unknown trial spec string: {trial_spec!r}")
    arr = np.asarray(list(trial_spec), dtype=int)
    arr = arr[(arr >= 0) & (arr < n_trials)]
    return np.unique(arr)


def _block_positions_from_probability_left(probability_left):
    p = np.asarray(probability_left, dtype=float)
    pos = np.zeros(p.size, dtype=int)
    current = 0
    prev = np.nan
    for i, val in enumerate(p):
        same_as_prev = False
        if i > 0:
            if np.isfinite(val) and np.isfinite(prev):
                same_as_prev = (val == prev)
            elif (not np.isfinite(val)) and (not np.isfinite(prev)):
                same_as_prev = True
        if i == 0 or not same_as_prev:
            current = 1
        else:
            current += 1
        pos[i] = current
        prev = val
    return pos


def apply_beginning_block_trial_filter(trial_indices, probability_left, n_remove,
                                       qc_dir, pid, save_qc_outputs=1):
    """Remove the first n_remove trials of each contiguous probabilityLeft block."""
    trial_indices = np.asarray(trial_indices, dtype=int)
    n_remove = int(n_remove or 0)
    if n_remove <= 0 or trial_indices.size == 0:
        return trial_indices, np.asarray([], dtype=int)

    probability_left = np.asarray(probability_left, dtype=float)
    positions = _block_positions_from_probability_left(probability_left)
    valid = (trial_indices >= 0) & (trial_indices < positions.size)
    trial_indices = trial_indices[valid]
    remove_mask = positions[trial_indices] <= n_remove
    removed = trial_indices[remove_mask]
    kept = trial_indices[~remove_mask]

    if save_qc_outputs == 1:
        try:
            df = pd.DataFrame({
                'trial_index': trial_indices,
                'probabilityLeft': probability_left[trial_indices],
                'position_in_block': positions[trial_indices],
                'removed_by_beginning_block_filter': remove_mask,
            })
            df.to_csv(qc_dir / f'{pid}_beginning_block_trial_filter.csv', index=False)
        except Exception as e:
            print(f'Beginning-block trial filter QC failed for {pid}: {e}')
    return kept, removed


def apply_probability_left_trial_filter(trial_indices, probability_left,
                                        allowed_values, tolerance,
                                        qc_dir, pid, save_qc_outputs=1):
    """Keep only configured probabilityLeft values before block-ID binarizing."""
    trial_indices = np.asarray(trial_indices, dtype=int)
    if allowed_values is None or trial_indices.size == 0:
        return trial_indices, np.asarray([], dtype=int)

    allowed = np.asarray(list(allowed_values), dtype=float)
    if allowed.size == 0:
        return trial_indices, np.asarray([], dtype=int)

    probability_left = np.asarray(probability_left, dtype=float)
    probs = np.full(trial_indices.size, np.nan)
    valid = (trial_indices >= 0) & (trial_indices < probability_left.size)
    probs[valid] = probability_left[trial_indices[valid]]

    tol = float(tolerance if tolerance is not None else 0.0)
    keep = np.zeros(trial_indices.size, dtype=bool)
    finite = np.isfinite(probs)
    keep[finite] = np.any(
        np.isclose(probs[finite, None], allowed[None, :], atol=tol, rtol=0.0),
        axis=1,
    )

    removed = trial_indices[~keep]
    kept = trial_indices[keep]

    if save_qc_outputs == 1:
        try:
            df = pd.DataFrame({
                'trial_index': trial_indices,
                'probabilityLeft': probs,
                'allowed_probability_left_values': ','.join(str(v) for v in allowed),
                'probability_left_tolerance': tol,
                'removed_by_probability_left_filter': ~keep,
            })
            df.to_csv(qc_dir / f'{pid}_probability_left_trial_filter.csv', index=False)
        except Exception as e:
            print(f'Probability-left trial filter QC failed for {pid}: {e}')
    return kept, removed


def remove_stim_trials_preceded_by_stim_filter(trial_indices, perturbation,
                                               qc_dir, pid, save_qc_outputs=1):
    """Remove opto trials immediately preceded by an opto trial (nonstim kept)."""
    trial_indices = np.asarray(trial_indices, dtype=int)
    perturbation = np.asarray(perturbation, dtype=bool)
    if trial_indices.size == 0:
        return trial_indices, perturbation, np.asarray([], dtype=int)

    opto_by_abs_trial = {int(t): bool(o) for t, o in zip(trial_indices, perturbation)}
    remove_mask = np.zeros(trial_indices.size, dtype=bool)
    for i, trial_idx in enumerate(trial_indices):
        if not perturbation[i]:
            continue
        if opto_by_abs_trial.get(int(trial_idx) - 1, False):
            remove_mask[i] = True

    removed = trial_indices[remove_mask]
    kept_idx = trial_indices[~remove_mask]
    kept_perturbation = perturbation[~remove_mask]
    return kept_idx, kept_perturbation, removed


def _trial_field_or_none(trials, name):
    try:
        return np.asarray(trials[name], dtype=float)
    except Exception:
        val = getattr(trials, name, None)
        return None if val is None else np.asarray(val, dtype=float)


def _trial_scalar_values(trials, name, trial_indices, default=np.nan):
    arr = _trial_field_or_none(trials, name)
    trial_indices = np.asarray(trial_indices, dtype=int)
    if arr is None:
        return np.full(trial_indices.size, default, dtype=float)
    arr = np.asarray(arr, dtype=float)
    out = np.full(trial_indices.size, default, dtype=float)
    valid = (trial_indices >= 0) & (trial_indices < arr.shape[0])
    if arr.ndim == 1:
        out[valid] = arr[trial_indices[valid]]
    return out


def _contrast_arrays_percent(trials, trial_indices):
    left = _trial_scalar_values(trials, 'contrastLeft', trial_indices)
    right = _trial_scalar_values(trials, 'contrastRight', trial_indices)
    finite_vals = np.r_[left[np.isfinite(left)], right[np.isfinite(right)]]
    scale = 100.0 if finite_vals.size and np.nanmax(np.abs(finite_vals)) <= 1.5 else 1.0
    left0 = np.nan_to_num(left, nan=0.0)
    right0 = np.nan_to_num(right, nan=0.0)
    signed = (right0 - left0) * scale
    return left, right, signed


def _stim_side_from_contrast_fields(left, right, signed_contrast_percent, tol_percent=1e-6):
    """Return -1 for left, +1 for right, 0 for ambiguous/no-side trials."""
    left_present = np.isfinite(left)
    right_present = np.isfinite(right)
    side = np.zeros(len(signed_contrast_percent), dtype=int)
    side[signed_contrast_percent < -tol_percent] = -1
    side[signed_contrast_percent > tol_percent] = 1
    zeroish = np.abs(signed_contrast_percent) <= tol_percent
    side[zeroish & left_present & ~right_present] = -1
    side[zeroish & right_present & ~left_present] = 1
    return side


def _choice_side_from_trials(trials, trial_indices):
    """Return choice side using the local side code: -1 left, +1 right."""
    choice = _trial_scalar_values(trials, 'choice', trial_indices)
    side = np.zeros(len(choice), dtype=int)
    # IBL convention used elsewhere in this codebase: choice == -1 is rightward,
    # choice == +1 is leftward. Convert to the side code used for block/stim side.
    side[np.isclose(choice, 1.0)] = -1
    side[np.isclose(choice, -1.0)] = 1
    return side


def _trial_interval_column(trials, trial_indices, col):
    trial_indices = np.asarray(trial_indices, dtype=int)
    try:
        intervals = np.asarray(trials['intervals'], dtype=float)
    except Exception:
        intervals = np.asarray(getattr(trials, 'intervals', []), dtype=float)
    out = np.full(trial_indices.size, np.nan, dtype=float)
    if intervals.ndim == 2 and intervals.shape[1] > col:
        valid = (trial_indices >= 0) & (trial_indices < intervals.shape[0])
        out[valid] = intervals[trial_indices[valid], col]
    return out


def save_inhibition_range_behavior_trials(
        trials, trial_indices, perturbation, final_trial_indices, removed,
        qc_dir, pid, p: TrialQCParams):
    """Save behavior for every metadata inhibition-range trial.

    This diagnostic is intentionally broader than the final analysis trial set:
    it preserves trials that later filters remove, then records those filter
    flags as columns. It is used downstream to inspect whether zero-contrast
    choices track block identity across the behavioral session.
    """
    if p.save_qc_outputs != 1:
        return
    trial_indices = np.asarray(trial_indices, dtype=int)
    perturbation = np.asarray(perturbation, dtype=bool)
    if trial_indices.size == 0 or perturbation.size != trial_indices.size:
        return

    try:
        probability_left = _trial_scalar_values(trials, 'probabilityLeft', trial_indices)
        left, right, signed_contrast = _contrast_arrays_percent(trials, trial_indices)
        abs_contrast = np.abs(signed_contrast)
        stim_side = _stim_side_from_contrast_fields(left, right, signed_contrast)
        choice_raw = _trial_scalar_values(trials, 'choice', trial_indices)
        choice_side = _choice_side_from_trials(trials, trial_indices)
        feedback_type = _trial_scalar_values(trials, 'feedbackType', trial_indices)
        correct = feedback_type > 0
        block_id = probability_left > 0.5
        block_side = np.where(block_id, -1, np.where(probability_left < 0.5, 1, 0))
        position_in_block = np.full(trial_indices.size, np.nan)
        probability_left_all = _trial_field_or_none(trials, 'probabilityLeft')
        if probability_left_all is not None:
            positions = _block_positions_from_probability_left(probability_left_all)
            valid_pos = (trial_indices >= 0) & (trial_indices < positions.size)
            position_in_block[valid_pos] = positions[trial_indices[valid_pos]]

        removed = removed or {}
        final_set = set(np.asarray(final_trial_indices, dtype=int).tolist())

        def removed_mask(name):
            vals = np.asarray(removed.get(name, np.asarray([], dtype=int)), dtype=int)
            return np.isin(trial_indices, vals)

        df = pd.DataFrame({
            'local_inhibition_range_index': np.arange(trial_indices.size),
            'absolute_trial_index': trial_indices,
            'is_control_trial': ~perturbation,
            'is_opto_trial': perturbation,
            'probabilityLeft': probability_left,
            'block_id': block_id.astype(int),
            'block_side_code': block_side.astype(int),
            'position_in_block': position_in_block,
            'contrastLeft': left,
            'contrastRight': right,
            'signed_contrast_percent': signed_contrast,
            'abs_contrast_percent': abs_contrast,
            'is_zero_contrast': np.isclose(abs_contrast, 0.0, atol=1e-6, rtol=0.0),
            'stim_side_code': stim_side,
            'choice_raw': choice_raw,
            'choice_side_code': choice_side,
            'valid_choice_side': choice_side != 0,
            'feedbackType': feedback_type,
            'correct': correct,
            'choice_block_congruent': choice_side == block_side,
            'stim_block_congruent': stim_side == block_side,
            'goCue_times': _trial_scalar_values(trials, 'goCue_times', trial_indices),
            'firstMovement_times': _trial_scalar_values(trials, 'firstMovement_times', trial_indices),
            'response_times': _trial_scalar_values(trials, 'response_times', trial_indices),
            'feedback_times': _trial_scalar_values(trials, 'feedback_times', trial_indices),
            'interval_start': _trial_interval_column(trials, trial_indices, 0),
            'interval_end': _trial_interval_column(trials, trial_indices, 1),
            'removed_by_hard_exclusion': removed_mask('hard_exclusion'),
            'removed_by_probability_left_filter': removed_mask('probability_left'),
            'removed_by_beginning_block_filter': removed_mask('beginning_block'),
            'removed_by_glmhmm_filter': removed_mask('glmhmm'),
            'removed_by_behavior_timing_filter': removed_mask('behavior_timing'),
            'removed_by_prev_stim_filter': removed_mask('prev_stim'),
            'included_after_trial_preprocessing': [int(t) in final_set for t in trial_indices],
        })
        removed_cols = [
            'removed_by_hard_exclusion',
            'removed_by_probability_left_filter',
            'removed_by_beginning_block_filter',
            'removed_by_glmhmm_filter',
            'removed_by_behavior_timing_filter',
            'removed_by_prev_stim_filter',
        ]
        df['removed_by_any_trial_filter'] = df[removed_cols].any(axis=1)
        df.to_csv(qc_dir / f'{pid}_inhibition_range_behavior_trials.csv', index=False)
    except Exception as e:
        print(f'Inhibition-range behavior trial QC save failed for {pid}: {e}')


def normalize_opto_trials_glmhmm_mode(mode):
    """Return canonical opto-trial GLM-HMM policy name."""
    if mode is None:
        return 'standard'
    key = str(mode).strip().lower().replace('_', ' ').replace('-', ' ')
    aliases = {
        'standard': 'standard',
        'current': 'standard',
        'bypass': 'bypass',
        'all': 'bypass',
        'prior state': 'prior state',
        'previous state': 'prior state',
        'prior nonopto': 'prior state',
        'previous nonopto': 'prior state',
    }
    if key not in aliases:
        raise ValueError(
            "opto_trials_GLMHMM must be 'standard', 'bypass', or 'prior state'"
        )
    return aliases[key]


def coerce_glmhmm_engaged_indices(glmhmm_result, n_states=2):
    """Extract engaged trial indices from `get_glmhmm_indices` output.

    The current analyses normally use `n_states == 2`, where the helper returns
    `(engaged, disengaged)`. For completeness, the 4-state return is collapsed
    across left/right engaged states.
    """
    if glmhmm_result is None:
        return np.asarray([], dtype=int)
    if int(n_states or 2) == 4 and len(glmhmm_result) >= 3:
        parts = [np.asarray(glmhmm_result[0]).ravel(), np.asarray(glmhmm_result[2]).ravel()]
        return np.unique(np.concatenate(parts)).astype(int)
    return np.asarray(glmhmm_result[0]).ravel().astype(int)


def apply_glmhmm_opto_trial_policy(trial_range, opto_numbers, nonstim_numbers,
                                  full_nonstim_numbers, engaged_idx,
                                  opto_trials_GLMHMM='standard'):
    """Apply GLM-HMM engagement filtering with opto-specific behavior.

    Non-opto/control trials are always filtered by their own GLM-HMM state.
    Opto trials use one of three policies:
      standard    : keep only opto trials whose own state is engaged
      bypass      : keep all opto trials, regardless of GLM-HMM state
      prior state : keep opto trials when the most recent previous non-opto
                    trial was engaged
    """
    mode = normalize_opto_trials_glmhmm_mode(opto_trials_GLMHMM)
    trial_range = np.asarray(trial_range, dtype=int)
    opto_numbers = np.asarray(opto_numbers, dtype=int)
    nonstim_numbers = np.asarray(nonstim_numbers, dtype=int)
    full_nonstim_numbers = np.asarray(full_nonstim_numbers, dtype=int)
    engaged_idx = np.asarray(engaged_idx, dtype=int)

    nonstim_keep = np.intersect1d(engaged_idx, nonstim_numbers)
    if mode == 'standard':
        opto_keep = np.intersect1d(engaged_idx, opto_numbers)
    elif mode == 'bypass':
        opto_keep = opto_numbers
    else:
        full_nonstim_sorted = np.sort(full_nonstim_numbers)
        opto_sorted = np.sort(opto_numbers)
        prior_pos = np.searchsorted(full_nonstim_sorted, opto_sorted, side='left') - 1
        has_prior = prior_pos >= 0
        prior_nonstim = np.full(opto_sorted.size, -1, dtype=int)
        prior_nonstim[has_prior] = full_nonstim_sorted[prior_pos[has_prior]]
        opto_keep = opto_sorted[has_prior & np.isin(prior_nonstim, engaged_idx)]

    keep = np.union1d(nonstim_keep, opto_keep)
    kept_range = np.intersect1d(trial_range, keep)
    removed = np.setdiff1d(trial_range, kept_range)
    return kept_range.astype(int), opto_keep.astype(int), nonstim_keep.astype(int), removed.astype(int)


def apply_behavior_timing_trial_filter(trials, trial_indices, p: TrialQCParams,
                                       qc_dir, pid):
    """Apply optional RT and quiescence-duration filters to trial indices.

    Reaction time is `reaction_time_source - goCue_times`, where source is
    usually firstMovement_times. Quiescence duration is trials.quiescencePeriod.
    Trials outside any configured min/max bound are removed from the entire
    analysis, so they cannot contribute to CD computation or projection traces.
    """
    trial_indices = np.asarray(trial_indices, dtype=int)
    if trial_indices.size == 0:
        return trial_indices, np.asarray([], dtype=int), pd.DataFrame()

    any_rt = p.min_reaction_time_s is not None or p.max_reaction_time_s is not None
    any_qp = p.min_quiescence_period_s is not None or p.max_quiescence_period_s is not None
    if not any_rt and not any_qp:
        return trial_indices, np.asarray([], dtype=int), pd.DataFrame()

    go = _trial_field_or_none(trials, 'goCue_times')
    keep = np.ones(trial_indices.size, dtype=bool)
    rows = {
        'trial_index': trial_indices,
        'removed_by_behavior_timing_filter': np.zeros(trial_indices.size, dtype=bool),
    }

    if any_rt:
        source = str(p.reaction_time_source or 'firstMovement_times')
        candidate_sources = ['firstMovement_times', 'response_times'] if source == 'auto' else [source]
        event = None
        event_source = ''
        for src in candidate_sources:
            event = _trial_field_or_none(trials, src)
            if event is not None:
                event_source = src
                break
        if event is None or go is None:
            print(f'Behavior timing filter requested RT for {pid}, but no usable RT source was found; RT filter skipped.')
            rt = np.full(trial_indices.size, np.nan)
            rt_keep = np.ones(trial_indices.size, dtype=bool)
        else:
            rt = event[trial_indices] - go[trial_indices]
            rt_keep = np.isfinite(rt)
            if p.min_reaction_time_s is not None:
                rt_keep &= rt >= float(p.min_reaction_time_s)
            if p.max_reaction_time_s is not None:
                rt_keep &= rt <= float(p.max_reaction_time_s)
            keep &= rt_keep
        rows['reaction_time_source'] = event_source
        rows['reaction_time_s'] = rt
        rows['rt_keep'] = rt_keep

    if any_qp:
        qp_all = _trial_field_or_none(trials, 'quiescencePeriod')
        if qp_all is None:
            print(f'Behavior timing filter requested quiescence duration for {pid}, but trials.quiescencePeriod is unavailable; QP filter skipped.')
            qp = np.full(trial_indices.size, np.nan)
            qp_keep = np.ones(trial_indices.size, dtype=bool)
        else:
            qp = qp_all[trial_indices]
            qp_keep = np.isfinite(qp)
            if p.min_quiescence_period_s is not None:
                qp_keep &= qp >= float(p.min_quiescence_period_s)
            if p.max_quiescence_period_s is not None:
                qp_keep &= qp <= float(p.max_quiescence_period_s)
            keep &= qp_keep
        rows['quiescence_period_s'] = qp
        rows['quiescence_keep'] = qp_keep

    removed = trial_indices[~keep]
    kept = trial_indices[keep]
    rows['removed_by_behavior_timing_filter'] = ~keep
    df = pd.DataFrame(rows)
    if p.save_qc_outputs == 1:
        try:
            df.to_csv(qc_dir / f'{pid}_behavior_timing_trial_filter.csv', index=False)
        except Exception as e:
            print(f'Behavior timing trial-filter QC save failed for {pid}: {e}')
    return kept, removed, df


# =====================================================================
# Trial preparation orchestrator
# =====================================================================
@dataclass
class TrialSelection:
    """Result of prepare_trials. Each pipeline consumes these as needed."""
    inhibition_trials_range: np.ndarray   # final admissible absolute trial indices
    inhibition_trials_numbers: np.ndarray  # opto trials within the range
    nonstim_trials_numbers: np.ndarray     # control trials within the range
    laser_intervals: np.ndarray
    laser_onsets: np.ndarray
    perturbation: np.ndarray               # bool over inhibition_trials_range
    correct: np.ndarray                    # bool over inhibition_trials_range
    block_ids: np.ndarray                  # int over inhibition_trials_range (probabilityLeft > 0.5)
    removed: Dict[str, np.ndarray]


def prepare_trials(sb: SessionBundle, inhibition_trials_spec, p: TrialQCParams, qc_dir, *,
                   one, glmhmm_indices_fn: Optional[Callable] = None,
                   glmhmm_state_probability=None) -> Optional[TrialSelection]:
    """Run the full trial cascade and return a TrialSelection, or None if the
    session should be skipped (no admissible trials, GLM-HMM load failure).

    Block IDs use the updated CD definition: probabilityLeft > 0.5. If
    `allowed_probability_left_values` is set, non-bias blocks such as 0.5 are
    removed before this binarization.
    """
    trials = sb.trials
    removed = {}

    # 1. Normalize the metadata trial spec to valid indices.
    full_inhibition_trials_range = normalize_trial_range(inhibition_trials_spec, len(trials['contrastLeft']))
    inhibition_trials_range = full_inhibition_trials_range.copy()

    # 2. Per-PID hard exclusions (BS's TRIALS_TO_REMOVE).
    hard_exclusion_removed = np.asarray([], dtype=int)
    if sb.pid in p.TRIALS_TO_REMOVE:
        to_remove = np.asarray(p.TRIALS_TO_REMOVE[sb.pid])
        hard_exclusion_removed = inhibition_trials_range[np.isin(inhibition_trials_range, to_remove)]
        inhibition_trials_range = inhibition_trials_range[~np.isin(inhibition_trials_range, to_remove)]
    removed['hard_exclusion'] = hard_exclusion_removed

    # 3. Optional probabilityLeft-value filter. This should happen before
    # beginning-of-block removal and opto/control counting so neutral blocks are
    # never folded into the low-prob-left block by the later boolean label.
    probability_left_removed = np.asarray([], dtype=int)
    if p.allowed_probability_left_values is not None:
        inhibition_trials_range, probability_left_removed = apply_probability_left_trial_filter(
            inhibition_trials_range, trials['probabilityLeft'],
            p.allowed_probability_left_values, p.probability_left_tolerance,
            qc_dir, sb.pid, p.save_qc_outputs,
        )
    removed['probability_left'] = probability_left_removed
    if len(inhibition_trials_range) == 0:
        print('No valid trials after probability-left filtering; skipping...')
        return None

    # 4. Beginning-of-block removal.
    beginning_block_removed = np.asarray([], dtype=int)
    if int(p.beginning_block_trials_remove or 0) > 0:
        inhibition_trials_range, beginning_block_removed = apply_beginning_block_trial_filter(
            inhibition_trials_range, trials['probabilityLeft'],
            p.beginning_block_trials_remove, qc_dir, sb.pid, p.save_qc_outputs,
        )
    removed['beginning_block'] = beginning_block_removed
    if len(inhibition_trials_range) == 0:
        print('No valid trials after range/beginning-block filtering; skipping...')
        return None

    # 5. Laser load + stim/nonstim classification.
    try:
        laser_intervals = one.load_dataset(sb.eid, '_ibl_laserStimulation.intervals')
        laser_intervals = np.asarray(laser_intervals)
        laser_onsets = laser_intervals[:, 0]
        session_nonstim_numbers = []
        for k in range(len(trials['contrastLeft'])):
            if not np.any(np.isclose(trials.intervals[k, 0], laser_onsets, atol=1e-6)):
                session_nonstim_numbers.append(k)
        session_nonstim_numbers = np.asarray(session_nonstim_numbers, dtype=int)
        full_inhibition_numbers = np.full(len(trials['contrastLeft']), np.nan)
        full_nonstim_numbers = np.full(len(trials['contrastLeft']), np.nan)
        for k in full_inhibition_trials_range:
            if np.any(np.isclose(trials.intervals[k, 0], laser_onsets, atol=1e-6)):
                full_inhibition_numbers[k] = k
            else:
                full_nonstim_numbers[k] = k
    except Exception:
        print('Laser intervals not found; falling back to deprecated taskData')
        from ibllib.io.raw_data_loaders import load_data
        taskData = load_data(sb.ses_path)
        laser_intervals = np.empty((0, 2))
        laser_onsets = np.asarray([])
        session_nonstim_numbers = np.asarray(
            [k for k in range(len(taskData)) if taskData[k]['opto'] != 1],
            dtype=int,
        )
        full_inhibition_numbers = np.full(len(taskData), np.nan)
        full_nonstim_numbers = np.full(len(taskData), np.nan)
        for k in full_inhibition_trials_range:
            if taskData[k]['opto'] == 1:
                full_inhibition_numbers[k] = k
            else:
                full_nonstim_numbers[k] = k

    full_inhibition_numbers = full_inhibition_numbers[~np.isnan(full_inhibition_numbers)].astype(int)
    full_nonstim_numbers = full_nonstim_numbers[~np.isnan(full_nonstim_numbers)].astype(int)
    full_perturbation = np.isin(full_inhibition_trials_range, full_inhibition_numbers)
    inhibition_numbers = np.intersect1d(full_inhibition_numbers, inhibition_trials_range)
    nonstim_numbers = np.intersect1d(full_nonstim_numbers, inhibition_trials_range)

    # 6. GLM-HMM engagement restriction.
    glmhmm_removed = np.asarray([], dtype=int)
    if int(p.use_GLMHMM_engaged_indices or 0) == 1:
        if glmhmm_indices_fn is None:
            print('GLM-HMM requested but no glmhmm_indices_fn provided; skipping GLM-HMM restriction.')
        else:
            try:
                before_glmhmm_range = np.asarray(inhibition_trials_range, dtype=int)
                glmhmm_result = glmhmm_indices_fn(
                    sb.mouse_id, str(sb.eid), glmhmm_state_probability, p.n_states
                )
                engaged_idx = coerce_glmhmm_engaged_indices(glmhmm_result, p.n_states)
                (inhibition_trials_range,
                 inhibition_numbers,
                 nonstim_numbers,
                 glmhmm_removed) = apply_glmhmm_opto_trial_policy(
                    before_glmhmm_range,
                    inhibition_numbers,
                    nonstim_numbers,
                    session_nonstim_numbers,
                    engaged_idx,
                    p.opto_trials_GLMHMM,
                )
            except Exception as e:
                print(f'GLM-HMM filtering failed for PID = {sb.pid}: {e}; skipping session...')
                return None
    removed['glmhmm'] = glmhmm_removed

    # Perturbation mask over the (possibly GLM-HMM-restricted) range.
    perturbation = np.isin(inhibition_trials_range, inhibition_numbers)

    # 7. Behavioral timing filter (RT / quiescence duration). This removes
    # trials from the full analysis range, then subsets the opto/control lists.
    timing_removed = np.asarray([], dtype=int)
    inhibition_trials_range, timing_removed, _ = apply_behavior_timing_trial_filter(
        trials, inhibition_trials_range, p, qc_dir, sb.pid,
    )
    if len(inhibition_trials_range) == 0:
        print('No valid trials after behavior timing filter; skipping...')
        return None
    if timing_removed.size:
        inhibition_numbers = np.intersect1d(inhibition_numbers, inhibition_trials_range)
        nonstim_numbers = np.intersect1d(nonstim_numbers, inhibition_trials_range)
        perturbation = np.isin(inhibition_trials_range, inhibition_numbers)
    removed['behavior_timing'] = timing_removed

    # 8. Previous-stim filter.
    prev_stim_removed = np.asarray([], dtype=int)
    if int(p.remove_stim_trials_preceded_by_stim or 0) == 1:
        inhibition_trials_range, perturbation, prev_stim_removed = \
            remove_stim_trials_preceded_by_stim_filter(
                inhibition_trials_range, perturbation, qc_dir, sb.pid, p.save_qc_outputs,
            )
        if len(inhibition_trials_range) == 0:
            print('No valid trials after previous-stim filter; skipping...')
            return None
    removed['prev_stim'] = prev_stim_removed

    # Recompute final absolute opto/control trial lists after all filters. The
    # previous-stim and behavior-timing filters operate on the range/mask, so the
    # original lists would otherwise over-count removed trials.
    inhibition_numbers = inhibition_trials_range[np.asarray(perturbation, dtype=bool)]
    nonstim_numbers = inhibition_trials_range[~np.asarray(perturbation, dtype=bool)]

    # Final derived masks.
    correct = (trials.feedbackType[inhibition_trials_range] > 0)
    block_ids = (trials.probabilityLeft[inhibition_trials_range] > 0.5).astype(int)

    save_inhibition_range_behavior_trials(
        trials, full_inhibition_trials_range, full_perturbation,
        inhibition_trials_range, removed, qc_dir, sb.pid, p,
    )

    return TrialSelection(
        inhibition_trials_range=np.asarray(inhibition_trials_range, dtype=int),
        inhibition_trials_numbers=inhibition_numbers,
        nonstim_trials_numbers=nonstim_numbers,
        laser_intervals=laser_intervals,
        laser_onsets=np.asarray(laser_onsets, dtype=float),
        perturbation=np.asarray(perturbation, dtype=bool),
        correct=np.asarray(correct, dtype=bool),
        block_ids=block_ids,
        removed=removed,
    )
