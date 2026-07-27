"""
BS_postprocess.py
=================
Post-processing + plotting for the SNr-downstream bias-selectivity (BS)
pipeline. The main script (SNr_inhibition_BS_downstream_effect.py) now scores a
permissive set of units across ALL insertions and records every QC / exclusion
criterion as a per-unit field. This script loads that single results file and
lets you choose what to INCLUDE post-hoc, with NO re-running.

Workflow
--------
    1. Run SNr_inhibition_BS_downstream_effect.py once -> one self-contained
       suffixed pickle per alignment in onset_alignments_to_run
    2. Edit the OPTIONS block at the bottom of this file (or import and call
       filter_units / the plot_* functions yourself).
    3. python BS_postprocess.py

Available post-hoc filters (see filter_units):
    brain_region_inhibited : 'SNr' | 'ZI' | 'STN' | list | None(all)
    condition              : 'ipsi' | 'contra' | None(both)
    recorded_region        : 'midbrain' (broad, via CD depth-override logic)
                             | list of Allen acronyms | exclude_regions([...])
                             | ['~MRN', '~SCm'] style exclusions | None(any)
    recorded_region_beryl  : list of Beryl acronyms | exclude_regions([...])
                             | ['~MRN', '~SCm'] style exclusions | None
    bs_only                : keep only BS_score == 1
    max_pval_empirical     : keep units with permutation/pseudo-informed
                             pval_empirical <= threshold
    exclude_drift_units    : drop drift_unit == 1
    exclude_nonstationary_units : drop nonstationary_unit == 1
    max_qp_* / min_qp_*    : post-hoc thresholds for QP nonstationarity
                             columns saved by the current BS pipeline
    exclude_axonal_units   : drop ax_unit == 1
    exclude_light_artifact : drop light_artifact_auto == 1
    exclude_amplitude_outliers : drop waveform_amplitude_outlier == 1
    IBL_quality_label_threshold : keep IBL_label >= threshold
    presence_threshold     : keep presence_ratio > threshold
    min_firing_rate        : keep firing_rate >= value (Hz)
    min_n_per_block        : keep n_80_nonstim >= and n_20_nonstim >= value
    min_n_per_delta_block  : keep plotted delta-PETH control/opto 80/20 counts
                             >= value; requires newly saved *_delta_peth columns
    max_prelaser_delta_gap : drop units whose raw stim-control delta-FR baseline
                             gap is too large pre-laser (percent units)
    max_cv_prelaser_delta_gap : same, after split-half control sign orientation
                                when diagnostic traces are available
    max_prelaser_zdev      : drop units whose opto-vs-control block-delta already
                             differs pre-laser (|mean z| over [-prelaser_window_s,0])
    min_prelaser_separation_frac : require a stable pre-laser block code
    max_prelaser_trace_std : drop noisy pre-laser opto traces
    min_prelaser_baseline_fr : require enough raw pre-laser stim FR
    baseline_fr_window_s   : pre-laser raw-FR window for min_prelaser_baseline_fr
    prelaser_window_s      : pre-laser window for the two checks above (default 0.5)
    pids / exclude_pids    : restrict to / drop specific insertions

Call available_filter_options() for the same list from Python. The same keys
can be passed to browse_bs_unit_block_peths(..., restrict={...}).
"""

import ast
import hashlib
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from scipy import stats

# 'none' | 'block_crossfit' (requires a freshly rerun diagnostic-trace pickle)
# | 'split_half' (older diagnostic) | 'legacy' (circular historical comparison)
sign_mode_option = 'block_crossfit'

# One consistent condition color across population, insertion, unit, raster,
# and paired-summary views.  Event markers keep their separate red/gray colors.
OPTO_COLOR = 'deepskyblue'

# Publication/example-unit palette. These block colors (plus the optional
# single-opto overlay color) are deliberately kept in
# one obvious place so alternative shades can be tested without touching any
# plotting logic. Every unit-PETH function also accepts a ``colors={...}``
# override with any subset of these keys.
UNIT_BLOCK_PETH_COLORS = {
    'control_20': '0.55',       # gray
    'control_80': 'black',
    'opto_20': 'lightskyblue',
    'opto_80': 'deepskyblue',
    'opto_all': 'deepskyblue',
}

# A redundant line-style cue makes the 20/80 distinction survive grayscale
# printing and color-vision differences. These can likewise be overridden.
UNIT_BLOCK_PETH_LINESTYLES = {
    '20': '--',
    '80': '-',
    'opto_all': '-',
}

DEFAULT_BS_UNIT_LASER_RESULTS = (
    '~/python/saved_figures/'
    'BS_all_insertions_NOGLMHMM_standard_crossfit_LaserOnset.pkl'
)
DEFAULT_BS_UNIT_FEEDBACK_RESULTS = (
    '~/python/saved_figures/'
    'BS_all_insertions_NOGLMHMM_standard_crossfit_Feedback.pkl'
)

# Post-hoc common normalizer for ``norm_mode='qp_control_scalar'``. Units below
# this control quiescent-period firing rate are treated as missing rather than
# divided by a floor. Pass a different value to use_norm(..., qp_min_fr=...).
QP_CONTROL_SCALAR_MIN_FR = 0.5

# Same depth-override table the CD pipeline uses, so the broad 'midbrain'
# qualifier here ascribes a blanket midbrain label to units below threshold in
# no-histology sessions exactly as CD does. (is_midbrain is already computed
# this way at run time; we import the overrides to honour the same source of
# truth and to backfill the flag for older results files.)
try:
    from CD_config import DEPTH_THRESHOLD_OVERRIDES
except Exception as _e:  # pragma: no cover
    print(f'(could not import DEPTH_THRESHOLD_OVERRIDES from CD_config: {_e})')
    DEPTH_THRESHOLD_OVERRIDES = {}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
class _QpControlNormalizedDeltaSequence:
    """Lazy row-aligned ``(block80 - block20) / QP baseline`` traces."""

    def __init__(self, block80, block20, denominator_hz):
        self.block80 = block80
        self.block20 = block20
        self.denominator_hz = np.asarray(denominator_hz, dtype=float)
        if len(block80) != len(block20) or len(block80) != len(self.denominator_hz):
            raise ValueError('Raw block traces and QP denominators are not row-aligned.')

    def __len__(self):
        return len(self.denominator_hz)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        if not np.isscalar(index):
            return [self[int(i)] for i in np.asarray(index).ravel()]
        i = int(index)
        block80 = np.asarray(self.block80[i], dtype=float)
        block20 = np.asarray(self.block20[i], dtype=float)
        denominator = float(self.denominator_hz[i])
        if (block80.shape != block20.shape or not np.isfinite(denominator)
                or denominator <= 0):
            return np.full(block80.shape, np.nan, dtype=np.float32)
        with np.errstate(invalid='ignore', divide='ignore'):
            return np.asarray((block80 - block20) / denominator, dtype=np.float32)


def _duration_weighted_qp_fr(qp_fr, quiescence, trial_ids, cutoff_s):
    """Pool saved QP spike counts/durations for one block of control trials."""
    qp_fr = np.asarray(qp_fr, dtype=float)
    quiescence = np.asarray(quiescence, dtype=float)
    trial_ids = np.asarray(trial_ids, dtype=int)
    valid_ids = ((trial_ids >= 0) & (trial_ids < len(qp_fr))
                 & (trial_ids < len(quiescence)))
    trial_ids = trial_ids[valid_ids]
    if trial_ids.size == 0:
        return np.nan
    duration = quiescence[trial_ids] - float(cutoff_s)
    firing = qp_fr[trial_ids]
    valid = np.isfinite(duration) & (duration > 0) & np.isfinite(firing)
    if not np.any(valid):
        return np.nan
    duration = duration[valid]
    firing = firing[valid]
    # qp_fr was saved as spike_count / duration. Weighting by duration exactly
    # reconstructs pooled spikes / pooled time (up to float32 storage precision).
    return float(np.sum(firing * duration) / np.sum(duration))


def _prepare_qp_control_scalar(data, min_fr=QP_CONTROL_SCALAR_MIN_FR):
    """Create lazy common-QP-normalized full and block-crossfit trace views.

    One denominator is calculated per unit from control trials in the delta-PETH
    time range. The 80 and 20 block QP rates are duration-weighted separately and
    then averaged equally, so block-count imbalance cannot set the denominator.
    The same denominator is applied to control/opto and to every alignment.
    """
    min_fr = float(min_fr)
    if not np.isfinite(min_fr) or min_fr < 0:
        raise ValueError('qp_control_scalar min_fr must be finite and >= 0 Hz.')
    if (data.get('_qp_control_scalar_prepared')
            and data.get('_qp_control_scalar_min_fr') == min_fr):
        return data

    required = (
        'units', 'qp_fr_per_trial', 'trial_metadata_by_pid',
        'trace_nonstim_80_raw', 'trace_nonstim_20_raw',
        'trace_stim_80_raw', 'trace_stim_20_raw',
    )
    missing = [key for key in required if key not in data]
    if missing:
        raise KeyError(
            "norm_mode='qp_control_scalar' needs the fresh sufficient-statistics "
            f"pickle; missing keys: {missing}")
    units = data['units']
    if 'pid' not in units:
        raise KeyError("data['units'] needs a 'pid' column for QP normalization.")
    n_units = len(units)
    qp_rows = data['qp_fr_per_trial']
    metadata = data['trial_metadata_by_pid']
    if len(qp_rows) != n_units:
        raise ValueError(
            f'qp_fr_per_trial has {len(qp_rows)} rows for {n_units} units.')

    baseline80 = np.full(n_units, np.nan, dtype=float)
    baseline20 = np.full(n_units, np.nan, dtype=float)
    for row_i, pid in enumerate(units['pid'].astype(str)):
        meta = metadata.get(pid)
        if not isinstance(meta, dict):
            continue
        qp_fr = np.asarray(qp_rows[row_i], dtype=float)
        quiescence = np.asarray(meta.get('quiescence_period', []), dtype=float)
        cutoff = float(meta.get('qp_before_gocue_end_time', 0.01))
        baseline80[row_i] = _duration_weighted_qp_fr(
            qp_fr, quiescence, meta.get('nonstim_80_delta', []), cutoff)
        baseline20[row_i] = _duration_weighted_qp_fr(
            qp_fr, quiescence, meta.get('nonstim_20_delta', []), cutoff)

    baseline = 0.5 * (baseline80 + baseline20)
    valid = (np.isfinite(baseline80) & np.isfinite(baseline20)
             & np.isfinite(baseline) & (baseline >= min_fr))
    denominator = np.where(valid, baseline, np.nan)

    data['qp_control_scalar_baseline_80_hz'] = baseline80
    data['qp_control_scalar_baseline_20_hz'] = baseline20
    data['qp_control_scalar_baseline_hz'] = baseline
    data['qp_control_scalar_valid'] = valid
    data['_qp_control_scalar_min_fr'] = min_fr

    def _install(output_key, block80_key, block20_key):
        if block80_key in data and block20_key in data:
            data[output_key] = _QpControlNormalizedDeltaSequence(
                data[block80_key], data[block20_key], denominator)

    _install(
        'trace_nonstim_qp_control_scalar',
        'trace_nonstim_80_raw', 'trace_nonstim_20_raw')
    _install(
        'trace_stim_qp_control_scalar',
        'trace_stim_80_raw', 'trace_stim_20_raw')
    for fold in ('a', 'b'):
        for role in ('reference', 'control_eval', 'stim_eval'):
            prefix = f'trace_block_crossfit_{role}_{fold}'
            _install(
                f'{prefix}_qp_control_scalar',
                f'{prefix}_block80_raw', f'{prefix}_block20_raw')

    data['_qp_control_scalar_prepared'] = True
    print(
        'Prepared qp_control_scalar normalization: '
        f'{int(valid.sum())}/{n_units} units have control QP baseline '
        f'>= {min_fr:g} Hz')
    return data


def use_norm(data, mode='baseline_scalar', *,
             qp_min_fr=QP_CONTROL_SCALAR_MIN_FR):
    """Select which parallel normalization mode the trace-based plots/filters use.

    The pipeline saves three delta-FR trace sets on the SAME units, and the
    fresh sufficient-statistics pickle supports a fourth post-hoc mode:
      'per_bin'         - original: delta / per-bin all-trial PETH (0->0.1 floor)
      'baseline_scalar' - delta / single floored baseline scalar (robust)
      'zero_2_nan'      - 0-FR bins treated as missing (diagnostic for the
                          zero-bin/floor artifact behind the pre-laser collapse)
      'qp_control_scalar' - raw delta / one control-only, block-balanced QP
                            firing-rate scalar; identical for control/opto and
                            for laser/feedback alignments
    This points data['trace_nonstim']/data['trace_stim'] at the chosen mode, so
    plot_delta_fr, plot_paired_bar, the pre-laser filters, etc. all use it. The
    z-score trace is mode-independent and unaffected. Returns data.
    """
    valid = ('per_bin', 'baseline_scalar', 'zero_2_nan', 'qp_control_scalar')
    if mode not in valid:
        raise ValueError(f"mode must be one of {valid}")
    if mode == 'qp_control_scalar':
        _prepare_qp_control_scalar(data, min_fr=qp_min_fr)
    kn, ks = f'trace_nonstim_{mode}', f'trace_stim_{mode}'
    if kn not in data or ks not in data:
        print(f"'{kn}'/'{ks}' not in this pickle (older run?); leaving traces unchanged.")
        return data
    data['trace_nonstim'] = data[kn]
    data['trace_stim'] = data[ks]
    data['_active_norm'] = mode
    print(f"Active delta-FR normalization mode: {mode}")
    return data


def mask_low_fr_bins(data, threshold_hz=0.1, which='both'):
    """POST-HOC zero_2_nan-style diagnostic -- needs NO pipeline re-run.

    Sets the currently active delta traces (data['trace_nonstim']/['trace_stim'],
    as selected by use_norm) to NaN wherever the raw all-trial firing rate is at
    or below threshold_hz -- i.e. the near-empty bins (the pre-laser dead zone).
    Uses the saved raw PETHs 'trace_stim_all'/'trace_nonstim_all'. NaNs render as
    gaps in plot_delta_fr (it uses nanmean/nanstd). Modifies data in place; call
    load_results again to undo.

    which : 'stim'  -> mask by stim FR,  'nonstim' -> by control FR,
            'both'  -> mask a bin if EITHER is near-zero (default).
    """
    sa, na = data.get('trace_stim_all'), data.get('trace_nonstim_all')
    if sa is None or na is None:
        print("Need 'trace_stim_all' and 'trace_nonstim_all' in the pickle.")
        return data
    tn_out, ts_out = [], []
    masked = 0
    for i in range(len(data['trace_stim'])):
        sfr = np.asarray(sa[i], dtype=float)
        nfr = np.asarray(na[i], dtype=float)
        if which == 'stim':
            bad = sfr <= threshold_hz
        elif which == 'nonstim':
            bad = nfr <= threshold_hz
        else:
            bad = (sfr <= threshold_hz) | (nfr <= threshold_hz)
        tn = np.asarray(data['trace_nonstim'][i], dtype=float).copy()
        ts = np.asarray(data['trace_stim'][i], dtype=float).copy()
        tn[bad] = np.nan; ts[bad] = np.nan
        tn_out.append(tn); ts_out.append(ts)
        masked += int(bad.sum())
    data['trace_nonstim'] = tn_out
    data['trace_stim'] = ts_out
    n = max(len(ts_out), 1)
    print(f"Masked {masked} bins total (~{masked / n:.1f}/unit) with FR <= {threshold_hz} Hz "
          f"({which}). Reload to undo.")
    return data


def load_results(path):
    """Load a BS results file written by SNr_inhibition_BS_downstream_effect.py.

    Returns a dict with keys: units (DataFrame), trace_nonstim, trace_stim,
    trace_zscore (lists of 1D arrays, row-aligned to units), peth_time, etc.
    """
    with open(str(Path(path).expanduser()), 'rb') as f:
        data = pickle.load(f)
    df = data['units'].reset_index(drop=True)
    data['units'] = df
    # Backfill the broad midbrain flag for older files / override sessions.
    data['units'] = _ensure_midbrain_flag(df)
    n = len(df)
    for k in ('trace_nonstim', 'trace_stim', 'trace_zscore', 'trace_stim_all', 'trace_nonstim_all'):
        if k in data and len(data[k]) != n:
            print(f'WARNING: {k} length {len(data[k])} != n units {n}; '
                  f'trace-based plots may be misaligned.')
    return data


def audit_futureproof_payload(data, verbose=True):
    """Verify compact rerun sufficient statistics and cross-fit trial coverage."""
    n_units = len(data.get('units', []))
    qp = data.get('qp_fr_per_trial')
    metadata = data.get('trial_metadata_by_pid')
    raw_keys = [
        f'trace_block_crossfit_{role}_{fold}_{raw_name}'
        for fold in ('a', 'b')
        for role in ('reference', 'control_eval', 'stim_eval')
        for raw_name in ('block80_raw', 'block20_raw', 'all_mean')
    ]
    missing_raw_keys = [key for key in raw_keys if key not in data]
    qp_rows_ok = qp is not None and len(qp) == n_units
    expected_pids = (
        set(data['units']['pid'].astype(str))
        if n_units and 'pid' in data['units'] else set())
    metadata_pids = set(metadata) if isinstance(metadata, dict) else set()

    qp_length_mismatches = 0
    if qp_rows_ok and isinstance(metadata, dict) and expected_pids:
        for row_i, pid in enumerate(data['units']['pid'].astype(str)):
            n_trials = metadata.get(pid, {}).get('n_trials')
            if n_trials is None or len(qp[row_i]) != int(n_trials):
                qp_length_mismatches += 1

    per_trial_meta_fields = (
        'probability_left', 'choice', 'feedback_type', 'contrast_left',
        'contrast_right', 'quiescence_period', 'go_cue_times',
        'go_cue_trigger_times', 'stim_on_times', 'feedback_times',
        'response_times', 'first_movement_times', 'interval_start_times',
        'interval_end_times', 'block_run_id', 'crossfit_half_id',
    )
    metadata_length_mismatches = []
    pseudo_shape_mismatches = []
    if isinstance(metadata, dict):
        for pid, meta in metadata.items():
            n_trials = int(meta.get('n_trials', -1))
            for field in per_trial_meta_fields:
                value = meta.get(field)
                if value is None or len(value) != n_trials:
                    metadata_length_mismatches.append((str(pid), field))
            pseudo = meta.get('pseudo_block_labels')
            if (pseudo is None or np.ndim(pseudo) != 2
                    or np.shape(pseudo)[1] != n_trials):
                pseudo_shape_mismatches.append(str(pid))

    low_count_fold_blocks = []
    if isinstance(metadata, dict):
        for pid, meta in metadata.items():
            folds = meta.get('crossfit_trial_numbers', {})
            for fold in ('a', 'b'):
                for role in ('reference', 'control_eval', 'stim_eval'):
                    for block in ('80', '20'):
                        n = len(folds.get(fold, {}).get(role, {}).get(block, []))
                        if n < 2:
                            low_count_fold_blocks.append(
                                (str(pid), fold, role, block, int(n)))

    summary = {
        'n_units': int(n_units),
        'sufficient_stats_schema_version': data.get(
            'sufficient_stats_schema', {}).get('version', np.nan),
        'qp_rows_ok': bool(qp_rows_ok),
        'qp_length_mismatches': int(qp_length_mismatches),
        'n_metadata_pids': int(len(metadata_pids)),
        'n_missing_metadata_pids': int(len(expected_pids - metadata_pids)),
        'n_metadata_length_mismatches': int(len(metadata_length_mismatches)),
        'n_pseudo_shape_mismatches': int(len(pseudo_shape_mismatches)),
        'n_missing_crossfit_raw_keys': int(len(missing_raw_keys)),
        'n_low_count_fold_blocks': int(len(low_count_fold_blocks)),
        'missing_crossfit_raw_keys': missing_raw_keys,
        'metadata_length_mismatches': metadata_length_mismatches,
        'pseudo_shape_mismatches': pseudo_shape_mismatches,
        'low_count_fold_blocks': low_count_fold_blocks,
    }
    if verbose:
        printable = dict(summary)
        printable['missing_crossfit_raw_keys'] = (
            missing_raw_keys[:6] + (['...'] if len(missing_raw_keys) > 6 else []))
        printable['metadata_length_mismatches'] = (
            metadata_length_mismatches[:6]
            + ([('...', '')] if len(metadata_length_mismatches) > 6 else []))
        printable['pseudo_shape_mismatches'] = (
            pseudo_shape_mismatches[:6]
            + (['...'] if len(pseudo_shape_mismatches) > 6 else []))
        printable['low_count_fold_blocks'] = (
            low_count_fold_blocks[:10] +
            ([('...', '', '', '', '')] if len(low_count_fold_blocks) > 10 else []))
        print(pd.Series(printable).to_string())
    return summary


def _ensure_midbrain_flag(df):
    """Guarantee an is_midbrain column, re-deriving via the CD depth-override
    logic for any override PIDs (depth <= threshold => midbrain)."""
    df = df.copy()
    if 'is_midbrain' not in df.columns:
        df['is_midbrain'] = False
    if 'depth' in df.columns and 'pid' in df.columns:
        for pid, thr in DEPTH_THRESHOLD_OVERRIDES.items():
            m = df['pid'] == pid
            if m.any():
                df.loc[m, 'is_midbrain'] = df.loc[m, 'depth'] <= thr
    return df


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------
def _as_set(x):
    if x is None:
        return None
    if isinstance(x, (str, bytes)):
        return {x}
    return set(x)


def include_regions(acronyms):
    """Explicit include selector for recorded_region(_beryl) filters."""
    return {'include': acronyms}


def exclude_regions(acronyms):
    """Explicit exclude selector for recorded_region(_beryl) filters.

    Example:
        recorded_region_beryl=exclude_regions(['MRN', 'SCm', 'SCs'])
    """
    return {'exclude': acronyms}


def _clean_region_acronym(value, include_unknown=False):
    """Normalize Allen/Beryl acronym values for readable inventory/filtering.

    Some saved Beryl acronyms are stringified one-item lists, e.g. "['MRN']".
    This helper turns those into "MRN" while preserving ordinary acronyms.
    """
    if isinstance(value, (list, tuple, set, np.ndarray)):
        values = [
            _clean_region_acronym(v, include_unknown=include_unknown)
            for v in list(value)
        ]
        values = [v for v in values if v is not None]
        if not values:
            return 'UNKNOWN' if include_unknown else None
        return '|'.join(values)
    if pd.isna(value):
        return 'UNKNOWN' if include_unknown else None
    text = str(value).strip()
    if not text or text.lower() in {'nan', 'none'}:
        return 'UNKNOWN' if include_unknown else None
    if text.startswith('[') and text.endswith(']'):
        try:
            parsed = ast.literal_eval(text)
            return _clean_region_acronym(parsed, include_unknown=include_unknown)
        except Exception:
            pass
    return text


def _strip_region_negation(value):
    text = str(value).strip()
    if text.startswith(('~', '!')):
        return True, text[1:].strip()
    return False, value


def _region_selector_mode_values(selector):
    """Return ('include'|'exclude', cleaned_values) for region filters."""
    if selector is None:
        return 'include', None

    if isinstance(selector, dict):
        include_keys = ('include', 'only', 'in')
        exclude_keys = ('exclude', 'not', 'not_in', 'without')
        include_vals = [selector[k] for k in include_keys if k in selector]
        exclude_vals = [selector[k] for k in exclude_keys if k in selector]
        if include_vals and exclude_vals:
            raise ValueError('Region selector cannot mix include and exclude keys.')
        if exclude_vals:
            raw_values = _as_set(exclude_vals[0])
            mode = 'exclude'
        elif include_vals:
            raw_values = _as_set(include_vals[0])
            mode = 'include'
        else:
            raise ValueError(
                "Region selector dict must use 'include' or 'exclude', "
                "e.g. exclude_regions(['MRN', 'SCm'])."
            )
    elif (
            isinstance(selector, (list, tuple))
            and len(selector) == 2
            and isinstance(selector[0], str)
            and selector[0].lower() in {'exclude', 'not', 'not_in', 'without', '~'}):
        raw_values = _as_set(selector[1])
        mode = 'exclude'
    else:
        raw_values = _as_set(selector)
        negated = []
        stripped = []
        for value in raw_values:
            is_negated, clean_value = _strip_region_negation(value)
            negated.append(is_negated)
            stripped.append(clean_value)
        if any(negated) and not all(negated):
            raise ValueError(
                'Region selector cannot mix included and negated acronyms; '
                "use either ['MRN', 'SCm'] or ['~MRN', '~SCm']."
            )
        mode = 'exclude' if any(negated) else 'include'
        raw_values = stripped

    values = {
        _clean_region_acronym(value)
        for value in raw_values
    }
    values = {value for value in values if value is not None}
    return mode, values


def _apply_region_selector_mask(df, column, selector):
    mode, values = _region_selector_mode_values(selector)
    if values is None:
        return pd.Series(True, index=df.index), mode, values
    observed = df[column].map(_clean_region_acronym)
    match = df[column].isin(values) | observed.isin(values)
    if mode == 'exclude':
        match = ~match
    return match, mode, values


def _prelaser_metrics(data, prelaser_window_s=0.5):
    """Per-unit pre-laser stability metrics over [-prelaser_window_s, 0].

    Returns (zdev, sepfrac, noise), each length = n units, positionally aligned
    to the units DataFrame:
      zdev    : |mean z-scored (opto-control) block-delta| in the pre-laser window
                -- large => opto and control already differ before the laser
                (a consistent baseline offset).
      sepfrac : |mean control block-delta pre-laser| / mean |control block-delta
                over the whole window| -- small => no stable baseline block code.
      noise   : std of the OPTO block-delta (% baseline) over the pre-laser
                window -- large => a jittery/insufficient-data baseline. This
                catches the noisy traces the z-metric misses: a low-trial unit
                has a large SEM and therefore a SMALL z, so it slips past zdev.
    """
    t = np.asarray(data['peth_time'], dtype=float)
    twin = abs(prelaser_window_s)
    pre = (t >= -twin) & (t < 0)
    ns, st, z = data['trace_nonstim'], data['trace_stim'], data['trace_zscore']
    n = len(ns)
    zdev = np.full(n, np.nan)
    sepfrac = np.full(n, np.nan)
    noise = np.full(n, np.nan)
    if not pre.any():
        print('WARNING: pre-laser window empty for peth_time; check prelaser_window_s.')
        return zdev, sepfrac, noise
    for i in range(n):
        tn = np.asarray(ns[i], dtype=float)
        ts = np.asarray(st[i], dtype=float)
        tz = np.asarray(z[i], dtype=float)
        zdev[i] = abs(np.nanmean(tz[pre]))
        ctrl_pre = abs(np.nanmean(tn[pre]))
        ctrl_all = np.nanmean(np.abs(tn)) + 1e-9
        sepfrac[i] = ctrl_pre / ctrl_all
        noise[i] = np.nanstd(ts[pre]) * 100.0   # % baseline units, matches the plot
    return zdev, sepfrac, noise


def _nanmean(arr, axis=None):
    arr = np.asarray(arr, dtype=float)
    finite = np.isfinite(arr)
    if axis is None:
        n = int(finite.sum())
        return np.nan if n == 0 else float(np.nansum(arr) / n)
    n = np.sum(finite, axis=axis)
    summed = np.nansum(arr, axis=axis)
    out = np.full_like(summed, np.nan, dtype=float)
    np.divide(summed, n, out=out, where=n > 0)
    return out


def _window(data, window):
    t = np.asarray(data['peth_time'], dtype=float)
    return (t >= window[0]) & (t < window[1])


def _prelaser_window(data, prelaser_window_s):
    twin = abs(prelaser_window_s)
    return _window(data, (-twin, 0.0))


def _stack_key(data, key, idx=None, scale=1.0):
    if key not in data:
        raise KeyError(f"'{key}' not found. Re-run SNr_inhibition_BS_downstream_effect.py "
                       "with save_diagnostic_traces=1 / save_raw_block_peths=1 as needed.")
    if idx is None:
        idx = np.arange(len(data['units']))
    return np.vstack([np.asarray(data[key][i], dtype=float) for i in idx]) * scale


def _active_mode(data, mode=None):
    selected = mode or data.get('_active_norm', 'per_bin')
    if selected == 'qp_control_scalar':
        _prepare_qp_control_scalar(
            data, min_fr=data.get(
                '_qp_control_scalar_min_fr', QP_CONTROL_SCALAR_MIN_FR))
    return selected


def diagnostic_traces_available(data, mode='zero_2_nan'):
    keys = [
        f'trace_nonstim_split_a_{mode}',
        f'trace_nonstim_split_b_{mode}',
        f'trace_nonstim_trialmatched_{mode}',
        f'trace_stim_{mode}',
    ]
    return all(k in data for k in keys)


def split_half_traces_available(data, mode='zero_2_nan'):
    """Whether the saved control A/B traces needed for cross-fitting exist."""
    keys = [
        f'trace_nonstim_split_a_{mode}',
        f'trace_nonstim_split_b_{mode}',
        f'trace_stim_{mode}',
    ]
    return all(k in data for k in keys)


def block_crossfit_traces_available(data, mode='zero_2_nan'):
    """Whether matched held-out control/opto block-crossfit traces exist."""
    keys = [
        f'trace_block_crossfit_{role}_{fold}_{mode}'
        for fold in ('a', 'b')
        for role in ('reference', 'control_eval', 'stim_eval')
    ]
    return all(k in data for k in keys)


# Orthogonal primary-analysis options. These deliberately separate how firing
# rate is scaled, how the 80/20 relationship is oriented, and which trials are
# evaluated. The older ``sign_mode`` API remains available below for exact
# historical reproduction.
PRIMARY_ORIENTATION_MODES = ('qp_preference', 'independent_absolute', 'signed_80_minus_20')
PRIMARY_TRIAL_ESTIMATORS = ('all_trials', 'matched_crossfit')
PRIMARY_NORM_MODES = (
    'raw_hz', 'whole_control_scalar', 'qp_control_scalar',
    'per_bin', 'baseline_scalar', 'zero_2_nan',
)


def _canonical_primary_orientation(mode):
    key = str(mode).strip().lower().replace('-', '_').replace(' ', '_')
    aliases = {
        'qp': 'qp_preference',
        'qp_preference': 'qp_preference',
        'preferred_block': 'qp_preference',
        'absolute': 'independent_absolute',
        'abs': 'independent_absolute',
        'independent_absolute': 'independent_absolute',
        'signed': 'signed_80_minus_20',
        'none': 'signed_80_minus_20',
        'signed_80_minus_20': 'signed_80_minus_20',
    }
    if key not in aliases:
        raise ValueError(
            f'orientation_mode must be one of {PRIMARY_ORIENTATION_MODES}; got {mode!r}.')
    return aliases[key]


def _canonical_primary_estimator(mode):
    key = str(mode).strip().lower().replace('-', '_').replace(' ', '_')
    aliases = {
        'all': 'all_trials',
        'full': 'all_trials',
        'all_trials': 'all_trials',
        'unmatched': 'all_trials',
        'matched': 'matched_crossfit',
        'crossfit': 'matched_crossfit',
        'matched_crossfit': 'matched_crossfit',
        'matched_heldout': 'matched_crossfit',
    }
    if key not in aliases:
        raise ValueError(
            f'trial_estimator must be one of {PRIMARY_TRIAL_ESTIMATORS}; got {mode!r}.')
    return aliases[key]


def _canonical_primary_norm(data, mode):
    selected = data.get('_active_norm', 'per_bin') if mode is None else mode
    key = str(selected).strip().lower().replace('-', '_').replace(' ', '_')
    aliases = {
        'raw': 'raw_hz',
        'hz': 'raw_hz',
        'raw_hz': 'raw_hz',
        'whole_control': 'whole_control_scalar',
        'whole_trial_control': 'whole_control_scalar',
        'whole_control_scalar': 'whole_control_scalar',
        'qp': 'qp_control_scalar',
        'qp_control_scalar': 'qp_control_scalar',
        'per_bin': 'per_bin',
        'baseline_scalar': 'baseline_scalar',
        'zero_2_nan': 'zero_2_nan',
    }
    if key not in aliases:
        raise ValueError(
            f'norm_mode must be one of {PRIMARY_NORM_MODES}; got {mode!r}.')
    return aliases[key]


def _requests_primary_api(norm_mode, orientation_mode, trial_estimator):
    """Whether a call opted into the orthogonal API rather than legacy signs."""
    if orientation_mode is not None or trial_estimator is not None:
        return True
    if norm_mode is None:
        return False
    key = str(norm_mode).strip().lower().replace('-', '_').replace(' ', '_')
    return key in {
        'raw', 'hz', 'raw_hz', 'whole_control', 'whole_trial_control',
        'whole_control_scalar',
    }


def _whole_control_scalar(data, idx, min_fr=QP_CONTROL_SCALAR_MIN_FR):
    """Block-balanced mean control PETH firing over the entire saved trace."""
    min_fr = float(min_fr)
    if not np.isfinite(min_fr) or min_fr < 0:
        raise ValueError('whole_control_min_fr must be finite and >= 0 Hz.')
    c80 = _stack_key(data, 'trace_nonstim_80_raw', idx)
    c20 = _stack_key(data, 'trace_nonstim_20_raw', idx)
    denominator = 0.5 * (_nanmean(c80, axis=1) + _nanmean(c20, axis=1))
    valid = np.isfinite(denominator) & (denominator >= min_fr)
    return np.where(valid, denominator, np.nan)


def _primary_delta_matrix(data, idx, *, condition, norm_mode,
                          trial_estimator, fold=None,
                          whole_control_min_fr=QP_CONTROL_SCALAR_MIN_FR):
    """Return one condition's 80-20 matrix in publication display units."""
    condition = str(condition)
    if condition not in ('control', 'opto'):
        raise ValueError("condition must be 'control' or 'opto'.")
    if trial_estimator == 'all_trials':
        if fold is not None:
            raise ValueError('fold is only valid for matched_crossfit.')
        raw_prefix = 'trace_nonstim' if condition == 'control' else 'trace_stim'
        normalized_key = f'{raw_prefix}_{norm_mode}'
        raw80_key = f'{raw_prefix}_80_raw'
        raw20_key = f'{raw_prefix}_20_raw'
    else:
        if fold not in ('a', 'b'):
            raise ValueError("matched_crossfit requires fold='a' or 'b'.")
        role = 'control_eval' if condition == 'control' else 'stim_eval'
        raw_prefix = f'trace_block_crossfit_{role}_{fold}'
        normalized_key = f'{raw_prefix}_{norm_mode}'
        raw80_key = f'{raw_prefix}_block80_raw'
        raw20_key = f'{raw_prefix}_block20_raw'

    if norm_mode == 'raw_hz':
        return (_stack_key(data, raw80_key, idx)
                - _stack_key(data, raw20_key, idx))
    if norm_mode == 'whole_control_scalar':
        denominator = _whole_control_scalar(
            data, idx, min_fr=whole_control_min_fr)
        raw_delta = (_stack_key(data, raw80_key, idx)
                     - _stack_key(data, raw20_key, idx))
        with np.errstate(invalid='ignore', divide='ignore'):
            return raw_delta / denominator[:, None] * 100.0

    # Existing normalized traces are stored as fractions. Convert to percent
    # here so all new primary-analysis outputs share exactly the same units.
    active_mode = _active_mode(data, norm_mode)
    if normalized_key not in data:
        raise KeyError(
            f"'{normalized_key}' is unavailable for norm_mode={active_mode!r}. ")
    return _stack_key(data, normalized_key, idx, scale=100.0)


def _trials_after_bs_block_filter(meta, block_probability):
    """Reconstruct the exact control block trials eligible for the BS call."""
    probability = np.asarray(meta.get('probability_left', []), dtype=float)
    selected = np.asarray(meta.get('nonstim_trials_bs', []), dtype=int)
    selected = selected[(selected >= 0) & (selected < probability.size)]
    if probability.size == 0 or selected.size == 0:
        return np.array([], dtype=int)
    remove_n = max(0, int(meta.get('bs_blocklength_filterval', 10)))
    keep = np.ones(probability.size, dtype=bool)
    switches = np.r_[True, probability[1:] != probability[:-1]]
    if remove_n:
        for switch_i in np.flatnonzero(switches):
            keep[switch_i:min(probability.size, switch_i + remove_n)] = False
    return selected[
        keep[selected] & np.isclose(probability[selected], float(block_probability))]


def _qp_preference_sign(data, idx, *, reference_fold=None):
    """Return sign(mean QP FR80 - mean QP FR20) for each selected unit.

    With ``reference_fold=None``, trials match the broad control-QP BS
    definition (including the block-start removal). With fold ``a`` or ``b``,
    only that cross-fit fold's reference trials determine preference.
    """
    required = ('qp_fr_per_trial', 'trial_metadata_by_pid')
    missing = [key for key in required if key not in data]
    if missing:
        raise KeyError(
            'QP-defined preference needs the future-proof sufficient statistics; '
            f'missing {missing}.')
    idx = np.asarray(idx, dtype=int)
    units = data['units'].reset_index(drop=True)
    metadata = data['trial_metadata_by_pid']
    qp_rows = data['qp_fr_per_trial']
    signs = np.full(idx.size, np.nan, dtype=float)
    deltas = np.full(idx.size, np.nan, dtype=float)
    for out_i, source_i in enumerate(idx):
        pid = str(units.iloc[int(source_i)]['pid'])
        meta = metadata.get(pid, {})
        qp = np.asarray(qp_rows[int(source_i)], dtype=float)
        if reference_fold is None:
            ids80 = _trials_after_bs_block_filter(meta, 0.8)
            ids20 = _trials_after_bs_block_filter(meta, 0.2)
        else:
            folds = meta.get('crossfit_trial_numbers', {})
            fold = folds.get(str(reference_fold), {}) if isinstance(folds, dict) else {}
            reference = fold.get('reference', {}) if isinstance(fold, dict) else {}
            ids80 = np.asarray(reference.get('80', []), dtype=int)
            ids20 = np.asarray(reference.get('20', []), dtype=int)
        ids80 = ids80[(ids80 >= 0) & (ids80 < qp.size)]
        ids20 = ids20[(ids20 >= 0) & (ids20 < qp.size)]
        if ids80.size == 0 or ids20.size == 0:
            continue
        delta = float(_nanmean(qp[ids80]) - _nanmean(qp[ids20]))
        deltas[out_i] = delta
        if np.isfinite(delta) and delta != 0:
            signs[out_i] = np.sign(delta)
    return signs[:, None], deltas


def _primary_delta_traces(data, idx=None, *, norm_mode='raw_hz',
                          orientation_mode='qp_preference',
                          trial_estimator='all_trials',
                          whole_control_min_fr=QP_CONTROL_SCALAR_MIN_FR):
    """Build paired traces from orthogonal normalization/orientation/trial choices.

    ``all_trials`` evaluates every eligible inhibition-range control/opto trial.
    ``matched_crossfit`` evaluates the saved within-run matched held-out folds.
    For QP preference, the latter learns preference from each fold's independent
    control-QP reference trials; absolute differences never use a sign reference.
    """
    if idx is None:
        idx = np.arange(len(data['units']))
    idx = np.asarray(idx, dtype=int)
    norm_mode = _canonical_primary_norm(data, norm_mode)
    orientation_mode = _canonical_primary_orientation(orientation_mode)
    trial_estimator = _canonical_primary_estimator(trial_estimator)

    def _orient(control, opto, reference_fold=None):
        if orientation_mode == 'qp_preference':
            sign, _ = _qp_preference_sign(
                data, idx, reference_fold=reference_fold)
            return sign * control, sign * opto
        if orientation_mode == 'independent_absolute':
            return np.abs(control), np.abs(opto)
        return control, opto

    if trial_estimator == 'all_trials':
        control = _primary_delta_matrix(
            data, idx, condition='control', norm_mode=norm_mode,
            trial_estimator=trial_estimator,
            whole_control_min_fr=whole_control_min_fr)
        opto = _primary_delta_matrix(
            data, idx, condition='opto', norm_mode=norm_mode,
            trial_estimator=trial_estimator,
            whole_control_min_fr=whole_control_min_fr)
        control, opto = _orient(control, opto)
    else:
        control_folds, opto_folds = [], []
        for fold in ('a', 'b'):
            control = _primary_delta_matrix(
                data, idx, condition='control', norm_mode=norm_mode,
                trial_estimator=trial_estimator, fold=fold,
                whole_control_min_fr=whole_control_min_fr)
            opto = _primary_delta_matrix(
                data, idx, condition='opto', norm_mode=norm_mode,
                trial_estimator=trial_estimator, fold=fold,
                whole_control_min_fr=whole_control_min_fr)
            control, opto = _orient(control, opto, reference_fold=fold)
            control_folds.append(control)
            opto_folds.append(opto)
        control = _nanmean(np.stack(control_folds, axis=0), axis=0)
        opto = _nanmean(np.stack(opto_folds, axis=0), axis=0)
    return (
        np.asarray(control, dtype=float), np.asarray(opto, dtype=float),
        norm_mode, orientation_mode, trial_estimator,
    )


def _delta_value_label(norm_mode):
    labels = {
        'raw_hz': 'Hz',
        'whole_control_scalar': '% whole-control mean FR',
        'qp_control_scalar': '% control QP baseline',
    }
    return labels.get(str(norm_mode), '% baseline')


def _sign_from_reference(ref, mask=None):
    if mask is None:
        ref_mean = _nanmean(ref, axis=1)
    else:
        ref_mean = _nanmean(ref[:, mask], axis=1)
    sign = np.full(ref_mean.shape, np.nan, dtype=float)
    sign[np.isfinite(ref_mean) & (ref_mean > 0)] = 1.0
    sign[np.isfinite(ref_mean) & (ref_mean < 0)] = -1.0
    return sign[:, None]


DEFAULT_SIGN_WINDOW = (-2.0, 0.0)


def _canonical_sign_mode(sign_mode=sign_mode_option, sign_flip=None):
    """Resolve public/legacy sign options to a supported implementation.

    ``sign_flip`` is retained only for compatibility with older calls. True now
    requests the safe split-half implementation; False requests no orientation.
    The original same-data sign flip remains available only through the explicit
    ``legacy``/``legacy_unit_nonstim`` modes.
    """
    if sign_flip is not None:
        return 'split_half' if bool(sign_flip) else 'none'
    selected = sign_mode_option if sign_mode is None else sign_mode
    key = str(selected).strip().lower().replace('-', '_').replace(' ', '_')
    aliases = {
        'block_crossfit': 'block_crossfit',
        'matched_crossfit': 'block_crossfit',
        'rigorous_crossfit': 'block_crossfit',
        'rigorous': 'block_crossfit',
        'split_half': 'split_half',
        'cross_validated': 'split_half',
        'crossvalidation': 'split_half',
        'cv': 'split_half',
        # Old insertion-analysis calls now get the safe replacement by default.
        'unit_nonstim': 'split_half',
        'none': 'none',
        'signed': 'none',
        'legacy': 'legacy',
        'legacy_unit_nonstim': 'legacy',
    }
    if key not in aliases:
        raise ValueError(
            "sign_mode must be 'block_crossfit', 'split_half', 'none', or 'legacy' "
            "('unit_nonstim' is accepted as a split-half alias)"
        )
    return aliases[key]


def _sign_mask(data, sign_window):
    if sign_window is None:
        return None
    mask = _window(data, tuple(sign_window))
    if not np.any(mask):
        t = np.asarray(data['peth_time'], dtype=float)
        raise ValueError(
            f'sign_window={tuple(sign_window)} is empty for peth_time range '
            f'[{t[0]:g}, {t[-1]:g}]'
        )
    return mask


def _oriented_delta_traces(data, idx=None, *, mode=None,
                           sign_mode=sign_mode_option, sign_window=DEFAULT_SIGN_WINDOW,
                           sign_flip=None, scale=1.0,
                           smooth_ms=None, smooth_mode='centered'):
    """Return paired control/stim delta traces with one shared orientation rule.

    ``block_crossfit`` learns a unit's sign from a temporally separate half of
    each bias-block run and applies it to trial-count/time-matched held-out
    control and stim trials. It then swaps halves and averages the folds. This
    is the preferred control-preference-aligned estimator.

    ``split_half`` learns a unit's sign from control split A and applies it to
    held-out control split B and the full stim trace, then swaps A/B and averages
    the folds. Consequently, the control samples used to choose a sign are never
    the samples whose oriented magnitude is plotted. Stim never determines sign.

    ``none`` returns the ordinary signed traces. ``legacy`` reproduces the biased
    same-data control orientation and is intentionally opt-in only.
    """
    if idx is None:
        idx = np.arange(len(data['units']))
    idx = np.asarray(idx, dtype=int)
    active_mode = _active_mode(data, mode)
    resolved_mode = _canonical_sign_mode(sign_mode, sign_flip)
    sign_mask = _sign_mask(data, sign_window) if resolved_mode != 'none' else None

    if resolved_mode == 'block_crossfit':
        if not block_crossfit_traces_available(data, active_mode):
            raise KeyError(
                f"Block-crossfit traces for mode='{active_mode}' are not in this pickle. "
                "Re-run SNr_inhibition_BS_downstream_effect.py with "
                "save_diagnostic_traces=1 (or future-proof sufficient stats), "
                "or use sign_mode='none'."
            )
        control_folds = []
        stim_folds = []
        for fold in ('a', 'b'):
            reference = _stack_key(
                data, f'trace_block_crossfit_reference_{fold}_{active_mode}', idx)
            control_eval = _stack_key(
                data, f'trace_block_crossfit_control_eval_{fold}_{active_mode}', idx)
            stim_eval = _stack_key(
                data, f'trace_block_crossfit_stim_eval_{fold}_{active_mode}', idx)
            sign = _sign_from_reference(reference, sign_mask)
            control_folds.append(sign * control_eval)
            stim_folds.append(sign * stim_eval)
        control = _nanmean(np.stack(control_folds, axis=0), axis=0)
        stim = _nanmean(np.stack(stim_folds, axis=0), axis=0)
    elif resolved_mode == 'split_half':
        if not split_half_traces_available(data, active_mode):
            raise KeyError(
                f"Split-half traces for mode='{active_mode}' are not in this pickle. "
                "Re-run the BS pipeline with save_diagnostic_traces=1, or use "
                "sign_mode='none'."
            )
        a = _stack_key(data, f'trace_nonstim_split_a_{active_mode}', idx)
        b = _stack_key(data, f'trace_nonstim_split_b_{active_mode}', idx)
        stim = _stack_key(data, f'trace_stim_{active_mode}', idx)
        sign_a = _sign_from_reference(a, sign_mask)
        sign_b = _sign_from_reference(b, sign_mask)
        control = _nanmean(np.stack([sign_a * b, sign_b * a], axis=0), axis=0)
        stim = _nanmean(np.stack([sign_a * stim, sign_b * stim], axis=0), axis=0)
    else:
        ctrl_key, stim_key, active_mode = _delta_trace_keys(data, active_mode)
        control = _stack_key(data, ctrl_key, idx)
        stim = _stack_key(data, stim_key, idx)
        if resolved_mode == 'legacy':
            sign = _sign_from_reference(control, sign_mask)
            control, stim = sign * control, sign * stim

    control = np.asarray(control, dtype=float) * float(scale)
    stim = np.asarray(stim, dtype=float) * float(scale)
    control = _maybe_smooth_rows(data, control, smooth_ms, smooth_mode)
    stim = _maybe_smooth_rows(data, stim, smooth_ms, smooth_mode)
    return control, stim, active_mode, resolved_mode


def _maybe_smooth_rows(data, arr, smooth_ms=None, smooth_mode='centered'):
    if smooth_ms is None:
        return arr
    bin_size = float(data.get('bin_size', 0.05))
    bins = max(1, int(round(float(smooth_ms) / 1000.0 / bin_size)))
    return np.vstack([_smooth(row, bins, smooth_mode) for row in arr])


def _paired_trace_summary(data, ns, st, *, pre_window=(-5.0, 0.0), post_window=(0.0, 2.0)):
    pre = _window(data, pre_window)
    post = _window(data, post_window)
    ns_pre_unit = _nanmean(ns[:, pre], axis=1)
    st_pre_unit = _nanmean(st[:, pre], axis=1)
    ns_post_unit = _nanmean(ns[:, post], axis=1)
    st_post_unit = _nanmean(st[:, post], axis=1)
    pre_gap = st_pre_unit - ns_pre_unit
    did = (st_post_unit - st_pre_unit) - (ns_post_unit - ns_pre_unit)
    ok_pre = np.isfinite(pre_gap)
    ok_did = np.isfinite(did)
    summary = {
        'n_units': int(ns.shape[0]),
        'ns_pre': float(_nanmean(ns[:, pre])),
        'st_pre': float(_nanmean(st[:, pre])),
        'pre_gap_st_minus_ns': float(_nanmean(pre_gap)),
        'pre_gap_st_minus_ns_median': float(np.nanmedian(pre_gap)),
        'ns_post': float(_nanmean(ns[:, post])),
        'st_post': float(_nanmean(st[:, post])),
        'post_gap_st_minus_ns': float(_nanmean(st_post_unit - ns_post_unit)),
        'diff_in_diff_st_minus_ns': float(_nanmean(did)),
        'diff_in_diff_st_minus_ns_median': float(np.nanmedian(did)),
        'n_pre_pairs': int(ok_pre.sum()),
        'n_did_pairs': int(ok_did.sum()),
        'pre_gap_t_p': float(stats.ttest_1samp(pre_gap[ok_pre], 0.0).pvalue) if ok_pre.sum() >= 2 else np.nan,
        'did_t_p': float(stats.ttest_1samp(did[ok_did], 0.0).pvalue) if ok_did.sum() >= 2 else np.nan,
    }
    try:
        summary['pre_gap_wilcoxon_p'] = (
            float(stats.wilcoxon(pre_gap[ok_pre]).pvalue) if ok_pre.sum() >= 2 else np.nan)
    except ValueError:
        summary['pre_gap_wilcoxon_p'] = np.nan
    try:
        summary['did_wilcoxon_p'] = (
            float(stats.wilcoxon(did[ok_did]).pvalue) if ok_did.sum() >= 2 else np.nan)
    except ValueError:
        summary['did_wilcoxon_p'] = np.nan
    unit_metrics = pd.DataFrame({
        'pre_gap_st_minus_ns': pre_gap,
        'diff_in_diff_st_minus_ns': did,
        'ns_pre': ns_pre_unit,
        'st_pre': st_pre_unit,
        'ns_post': ns_post_unit,
        'st_post': st_post_unit,
    })
    return summary, unit_metrics


def block_crossfit_sign_diagnostic(data, idx=None, *, mode=None,
                                   sign_window=DEFAULT_SIGN_WINDOW,
                                   pre_window=(-5.0, 0.0), post_window=(0.0, 2.0),
                                   smooth_ms=None, smooth_mode='centered',
                                   return_traces=False, verbose=True):
    """Diagnose the temporally separated, trial-matched block cross-fit.

    Each fold's preference is learned from control trials outside the evaluated
    half-block. Its held-out control and opto trials are matched within the same
    block run before PETH construction. Reference-sign agreement reports how
    consistently the two independent training halves identify unit preference.
    """
    mode = _active_mode(data, mode)
    if not block_crossfit_traces_available(data, mode):
        raise KeyError(
            f"Block-crossfit traces for mode='{mode}' are not in this pickle. "
            "Re-run SNr_inhibition_BS_downstream_effect.py with "
            "save_diagnostic_traces=1."
        )
    ns_cv, st_cv, mode, _ = _oriented_delta_traces(
        data, idx, mode=mode, sign_mode='block_crossfit',
        sign_window=sign_window, scale=100.0,
        smooth_ms=smooth_ms, smooth_mode=smooth_mode,
    )
    summary, unit_metrics = _paired_trace_summary(
        data, ns_cv, st_cv, pre_window=pre_window, post_window=post_window)

    if idx is None:
        idx = np.arange(len(data['units']))
    idx = np.asarray(idx, dtype=int)
    sign_mask = _sign_mask(data, sign_window)
    ref_a = _stack_key(data, f'trace_block_crossfit_reference_a_{mode}', idx)
    ref_b = _stack_key(data, f'trace_block_crossfit_reference_b_{mode}', idx)
    sign_a = _sign_from_reference(ref_a, sign_mask)[:, 0]
    sign_b = _sign_from_reference(ref_b, sign_mask)[:, 0]
    valid_signs = np.isfinite(sign_a) & np.isfinite(sign_b)
    summary.update({
        'diagnostic': 'block_crossfit_sign',
        'mode': mode,
        'sign_window': tuple(sign_window) if sign_window is not None else None,
        'n_reference_sign_pairs': int(valid_signs.sum()),
        'reference_sign_agreement': (
            float(np.mean(sign_a[valid_signs] == sign_b[valid_signs]))
            if np.any(valid_signs) else np.nan),
    })
    if verbose:
        print(pd.Series(summary).to_string())
    if return_traces:
        return summary, unit_metrics, (np.asarray(data['peth_time']), ns_cv, st_cv)
    return summary, unit_metrics


def split_half_sign_diagnostic(data, idx=None, *, mode=None,
                               sign_window=DEFAULT_SIGN_WINDOW,
                               pre_window=(-5.0, 0.0), post_window=(0.0, 2.0),
                               smooth_ms=None, smooth_mode='centered',
                               return_traces=False, verbose=True):
    """Cross-validated sign diagnostic from saved split-half nonstim traces.

    Sign is learned from nonstim split A and applied to held-out nonstim split B
    and the full stim trace; then the folds are swapped and averaged. This avoids
    using the same noisy control trace to both orient and plot the control mean.
    Values are returned in percent baseline units, matching plot_delta_fr.
    """
    mode = _active_mode(data, mode)
    if not split_half_traces_available(data, mode):
        raise KeyError(f"Diagnostic split/matched traces for mode='{mode}' are not in this pickle. "
                       "Re-run the BS pipeline with save_diagnostic_traces=1.")
    ns_cv, st_cv, mode, _ = _oriented_delta_traces(
        data, idx, mode=mode, sign_mode='split_half', sign_window=sign_window,
        scale=100.0, smooth_ms=smooth_ms, smooth_mode=smooth_mode,
    )
    summary, unit_metrics = _paired_trace_summary(
        data, ns_cv, st_cv, pre_window=pre_window, post_window=post_window)
    summary.update({
        'diagnostic': 'split_half_sign',
        'mode': mode,
        'sign_window': tuple(sign_window) if sign_window is not None else None,
    })
    if verbose:
        print(pd.Series(summary).to_string())
    if return_traces:
        return summary, unit_metrics, (np.asarray(data['peth_time']), ns_cv, st_cv)
    return summary, unit_metrics


def trial_count_matched_diagnostic(data, idx=None, *, mode=None, sign_mode='none',
                                   pre_window=(-5.0, 0.0), post_window=(0.0, 2.0),
                                   smooth_ms=None, smooth_mode='centered',
                                   return_traces=False, verbose=True):
    """Compare stim traces to nonstim traces subsampled to stim block counts.

    sign_mode can be 'none', 'nonstim_full', or 'trialmatched'. Use 'none' to
    isolate trial-count effects without adding orientation bias.
    """
    mode = _active_mode(data, mode)
    if not diagnostic_traces_available(data, mode):
        raise KeyError(f"Diagnostic trial-matched traces for mode='{mode}' are not in this pickle. "
                       "Re-run the BS pipeline with save_diagnostic_traces=1.")
    if idx is None:
        idx = np.arange(len(data['units']))
    ns = _stack_key(data, f'trace_nonstim_trialmatched_{mode}', idx, scale=100.0)
    st = _stack_key(data, f'trace_stim_{mode}', idx, scale=100.0)
    ns = _maybe_smooth_rows(data, ns, smooth_ms, smooth_mode)
    st = _maybe_smooth_rows(data, st, smooth_ms, smooth_mode)
    if sign_mode == 'nonstim_full':
        ref = _maybe_smooth_rows(data, _stack_key(data, f'trace_nonstim_{mode}', idx, scale=100.0),
                                 smooth_ms, smooth_mode)
        sign = _sign_from_reference(ref)
        ns, st = sign * ns, sign * st
    elif sign_mode == 'trialmatched':
        sign = _sign_from_reference(ns)
        ns, st = sign * ns, sign * st
    elif sign_mode != 'none':
        raise ValueError("sign_mode must be 'none', 'nonstim_full', or 'trialmatched'")
    summary, unit_metrics = _paired_trace_summary(
        data, ns, st, pre_window=pre_window, post_window=post_window)
    summary.update({'diagnostic': 'trial_count_matched', 'mode': mode, 'sign_mode': sign_mode})
    if verbose:
        print(pd.Series(summary).to_string())
    if return_traces:
        return summary, unit_metrics, (np.asarray(data['peth_time']), ns, st)
    return summary, unit_metrics


def baseline_gap_metrics(data, idx=None, *, mode=None, pre_window=(-5.0, 0.0)):
    """Per-unit pre-laser stim-control delta gap metrics in percent units.

    Useful for finding units whose opto delta baseline is unstable relative to
    nonstim. If split/matched diagnostic traces exist, extra columns quantify the
    gap after cross-validated sign orientation and trial-count matching.
    """
    mode = _active_mode(data, mode)
    if idx is None:
        idx = np.arange(len(data['units']))
    pre = _window(data, pre_window)
    out = data['units'].iloc[idx].copy().reset_index(drop=True)
    ns = _stack_key(data, f'trace_nonstim_{mode}', idx, scale=100.0)
    st = _stack_key(data, f'trace_stim_{mode}', idx, scale=100.0)
    out['pre_gap_st_minus_ns'] = _nanmean(st[:, pre], axis=1) - _nanmean(ns[:, pre], axis=1)
    out['abs_pre_gap_st_minus_ns'] = np.abs(out['pre_gap_st_minus_ns'])
    out['pre_stim_delta_mean'] = _nanmean(st[:, pre], axis=1)
    out['pre_nonstim_delta_mean'] = _nanmean(ns[:, pre], axis=1)

    if diagnostic_traces_available(data, mode):
        a = _stack_key(data, f'trace_nonstim_split_a_{mode}', idx, scale=100.0)
        b = _stack_key(data, f'trace_nonstim_split_b_{mode}', idx, scale=100.0)
        tm = _stack_key(data, f'trace_nonstim_trialmatched_{mode}', idx, scale=100.0)
        out['control_split_pre_gap_b_minus_a'] = _nanmean(b[:, pre], axis=1) - _nanmean(a[:, pre], axis=1)
        out['abs_control_split_pre_gap'] = np.abs(out['control_split_pre_gap_b_minus_a'])
        out['trialmatched_pre_gap_st_minus_ns'] = _nanmean(st[:, pre], axis=1) - _nanmean(tm[:, pre], axis=1)
        cv_summary, cv_units = split_half_sign_diagnostic(
            data, idx, mode=mode, pre_window=pre_window, verbose=False)
        out['cv_pre_gap_st_minus_ns'] = cv_units['pre_gap_st_minus_ns'].to_numpy()
        out['abs_cv_pre_gap_st_minus_ns'] = np.abs(out['cv_pre_gap_st_minus_ns'])
        denom = out['abs_control_split_pre_gap'].replace(0, np.nan)
        out['abs_pre_gap_over_control_split_gap'] = out['abs_pre_gap_st_minus_ns'] / denom
    return out


def _prelaser_delta_gap(data, prelaser_window_s=0.5, mode=None):
    mode = _active_mode(data, mode)
    pre = _prelaser_window(data, prelaser_window_s)
    ns = _stack_key(data, f'trace_nonstim_{mode}', scale=100.0)
    st = _stack_key(data, f'trace_stim_{mode}', scale=100.0)
    return np.abs(_nanmean(st[:, pre], axis=1) - _nanmean(ns[:, pre], axis=1))


def _cv_prelaser_delta_gap(data, prelaser_window_s=0.5, mode=None):
    mode = _active_mode(data, mode)
    _, unit_metrics = split_half_sign_diagnostic(
        data, mode=mode, pre_window=(-abs(prelaser_window_s), 0.0), verbose=False)
    return np.abs(unit_metrics['pre_gap_st_minus_ns'].to_numpy())


def list_pids(data, by=('brain_region_inhibited', 'condition')):
    """Per-PID summary: region, condition, n units, n BS, BS fraction. Use this
    to see whether the effect is general or localized to particular insertions."""
    df = data['units']
    cols = [c for c in by if c in df.columns]
    g = (df.groupby(['pid'] + cols)
           .agg(n_units=('BS_score', 'size'), n_bs=('BS_score', 'sum'))
           .reset_index())
    g['BS_frac'] = g['n_bs'] / g['n_units']
    print(g.to_string(index=False))
    return g


def export_unstable_units(data, out_path, *, prelaser_window_s=0.5,
                          max_prelaser_zdev=2.0, min_prelaser_separation_frac=None,
                          max_prelaser_trace_std=None, restrict=None):
    """Compute baseline-unstable units (failing the pre-laser checks) and save a
    {pid: [clustnum, ...]} pickle that the CD pipeline can load and exclude.

    `restrict` is an optional dict of filter_units kwargs to first narrow the
    population (e.g. {'recorded_region': 'midbrain'}) before flagging.
    """
    import pickle
    df = data['units']
    sub_idx = np.arange(len(df))
    if restrict:
        _, sub_idx = filter_units(data, verbose=False, **restrict)
    zdev, sepfrac, noise = _prelaser_metrics(data, prelaser_window_s)
    bad = np.zeros(len(df), dtype=bool)
    if max_prelaser_zdev is not None:
        bad |= zdev > max_prelaser_zdev
    if min_prelaser_separation_frac is not None:
        bad |= sepfrac < min_prelaser_separation_frac
    if max_prelaser_trace_std is not None:
        bad |= noise > max_prelaser_trace_std
    keep = np.zeros(len(df), dtype=bool)
    keep[sub_idx] = True
    bad &= keep
    out = {}
    bdf = df.iloc[np.where(bad)[0]]
    for pid, grp in bdf.groupby('pid'):
        out[str(pid)] = [int(c) for c in grp['clustnum'].tolist()]
    with open(str(Path(out_path).expanduser()), 'wb') as f:
        pickle.dump(out, f)
    n_bad = int(bad.sum())
    print(f'Flagged {n_bad} baseline-unstable units across {len(out)} insertions '
          f'-> {out_path}')
    print('  Load this in CD_config (unstable_units_path) to exclude them there.')
    return out


FILTER_OPTION_HELP = {
    'brain_region_inhibited': "Metadata inhibited region: 'SNr', 'ZI', 'STN', list, or None.",
    'condition': "Metadata condition/hemisphere relation: 'ipsi', 'contra', list, or None.",
    'recorded_region': "Recorded Allen-region filter: 'midbrain', acronyms, exclude_regions([...]), ['~MRN'], or None.",
    'recorded_region_beryl': "Recorded Beryl-region filter: acronyms, exclude_regions([...]), ['~MRN'], or None.",
    'bs_only': "Keep only units with BS_score == 1.",
    'max_pval_empirical': "Keep units with pval_empirical <= threshold.",
    'exclude_drift_units': "Drop units flagged by the shared monotonic QP drift metric.",
    'exclude_nonstationary_units': "Drop units flagged by shared QP nonstationarity thresholds.",
    'max_qp_fr_segment_range_frac': "Keep units with raw QP segment FR range / median FR <= threshold.",
    'max_qp_resid_drift_range_frac': "Keep units with block-residual QP segment range / median FR <= threshold.",
    'max_qp_resid_drift_cv': "Keep units with block-residual QP segment CV <= threshold.",
    'max_qp_resid_abs_rho_time': "Keep units with abs Spearman(block-residual QP FR, trial order) <= threshold.",
    'max_qp_low_activity_fraction': "Keep units with low-QP-activity trial fraction <= threshold.",
    'max_qp_max_low_activity_run': "Keep units with longest low-QP-activity run <= threshold trials.",
    'min_qp_block_effect_sign_consistency': "Keep units whose segment block-effect sign consistency >= threshold.",
    'max_qp_block_effect_segment_cv': "Keep units with segment block-effect CV <= threshold.",
    'max_qp_block_effect_dominance': "Keep units where no one segment dominates block effect above threshold.",
    'exclude_axonal_units': "Drop units flagged as axonal by waveform classification.",
    'exclude_light_artifact': "Drop units flagged by the laser-locked light-artifact detector.",
    'exclude_amplitude_outliers': "Drop units flagged by waveform-amplitude outlier QC.",
    'IBL_quality_label_threshold': "Keep units with IBL_label >= threshold.",
    'presence_threshold': "Keep units with presence_ratio > threshold.",
    'min_firing_rate': "Keep units with session firing_rate >= threshold Hz.",
    'min_n_per_block': "Require n_80_nonstim and n_20_nonstim >= threshold.",
    'min_n_per_delta_block': "Require plotted control/opto delta-PETH block counts >= threshold.",
    'max_prelaser_delta_gap': "Limit raw stim-control pre-laser delta-FR baseline gap.",
    'max_cv_prelaser_delta_gap': "Limit split-half-sign pre-laser stim-control gap.",
    'max_prelaser_zdev': "Limit pre-laser opto-vs-control block-delta z deviation.",
    'min_prelaser_separation_frac': "Require stable pre-laser control block separation.",
    'max_prelaser_trace_std': "Limit pre-laser opto trace variability.",
    'min_prelaser_baseline_fr': "Require raw stim all-trial pre-laser baseline FR.",
    'baseline_fr_window_s': "Window length for min_prelaser_baseline_fr.",
    'prelaser_window_s': "Window length for pre-laser gap/stability metrics.",
    'pids': "Restrict to specific PID(s).",
    'exclude_pids': "Drop specific PID(s).",
}


def available_filter_options(print_options=True):
    """Return/print valid keys for filter_units(...)/browse(..., restrict=...)."""
    if print_options:
        print('Available BS restrict/filter options:')
        for key, desc in FILTER_OPTION_HELP.items():
            print(f'  {key:<32} {desc}')
    return dict(FILTER_OPTION_HELP)


def _validate_filter_kwargs(opts):
    unknown = sorted(set(opts or {}) - set(FILTER_OPTION_HELP))
    if unknown:
        valid = ', '.join(FILTER_OPTION_HELP)
        raise ValueError(
            f"Unknown restrict/filter option(s): {unknown}. Valid options are: {valid}"
        )


def filter_units(data, *,
                 brain_region_inhibited=None,
                 condition=None,
                 recorded_region=None,
                 recorded_region_beryl=None,
                 bs_only=False,
                 max_pval_empirical=None,
                 exclude_drift_units=False,
                 exclude_nonstationary_units=False,
                 max_qp_fr_segment_range_frac=None,
                 max_qp_resid_drift_range_frac=None,
                 max_qp_resid_drift_cv=None,
                 max_qp_resid_abs_rho_time=None,
                 max_qp_low_activity_fraction=None,
                 max_qp_max_low_activity_run=None,
                 min_qp_block_effect_sign_consistency=None,
                 max_qp_block_effect_segment_cv=None,
                 max_qp_block_effect_dominance=None,
                 exclude_axonal_units=False,
                 exclude_light_artifact=False,
                 exclude_amplitude_outliers=False,
                 IBL_quality_label_threshold=None,
                 presence_threshold=None,
                 min_firing_rate=None,
                 min_n_per_block=None,
                 min_n_per_delta_block=None,
                 max_prelaser_delta_gap=None,
                 max_cv_prelaser_delta_gap=None,
                 max_prelaser_zdev=None,
                 min_prelaser_separation_frac=None,
                 max_prelaser_trace_std=None,
                 min_prelaser_baseline_fr=None,
                 baseline_fr_window_s=1.0,
                 prelaser_window_s=0.5,
                 pids=None,
                 exclude_pids=None,
                 verbose=True):
    """Return (filtered_df, positional_indices).

    `positional_indices` are 0-based positions into the original units table,
    used to index the parallel trace lists for plotting.
    """
    _validate_filter_kwargs({
        'brain_region_inhibited': brain_region_inhibited,
        'condition': condition,
        'recorded_region': recorded_region,
        'recorded_region_beryl': recorded_region_beryl,
        'bs_only': bs_only,
        'max_pval_empirical': max_pval_empirical,
        'exclude_drift_units': exclude_drift_units,
        'exclude_nonstationary_units': exclude_nonstationary_units,
        'max_qp_fr_segment_range_frac': max_qp_fr_segment_range_frac,
        'max_qp_resid_drift_range_frac': max_qp_resid_drift_range_frac,
        'max_qp_resid_drift_cv': max_qp_resid_drift_cv,
        'max_qp_resid_abs_rho_time': max_qp_resid_abs_rho_time,
        'max_qp_low_activity_fraction': max_qp_low_activity_fraction,
        'max_qp_max_low_activity_run': max_qp_max_low_activity_run,
        'min_qp_block_effect_sign_consistency': min_qp_block_effect_sign_consistency,
        'max_qp_block_effect_segment_cv': max_qp_block_effect_segment_cv,
        'max_qp_block_effect_dominance': max_qp_block_effect_dominance,
        'exclude_axonal_units': exclude_axonal_units,
        'exclude_light_artifact': exclude_light_artifact,
        'exclude_amplitude_outliers': exclude_amplitude_outliers,
        'IBL_quality_label_threshold': IBL_quality_label_threshold,
        'presence_threshold': presence_threshold,
        'min_firing_rate': min_firing_rate,
        'min_n_per_block': min_n_per_block,
        'min_n_per_delta_block': min_n_per_delta_block,
        'max_prelaser_delta_gap': max_prelaser_delta_gap,
        'max_cv_prelaser_delta_gap': max_cv_prelaser_delta_gap,
        'max_prelaser_zdev': max_prelaser_zdev,
        'min_prelaser_separation_frac': min_prelaser_separation_frac,
        'max_prelaser_trace_std': max_prelaser_trace_std,
        'min_prelaser_baseline_fr': min_prelaser_baseline_fr,
        'baseline_fr_window_s': baseline_fr_window_s,
        'prelaser_window_s': prelaser_window_s,
        'pids': pids,
        'exclude_pids': exclude_pids,
    })
    df = data['units']
    mask = pd.Series(True, index=df.index)

    def _and(m, label):
        nonlocal mask
        before = int(mask.sum())
        mask = mask & m.reindex(mask.index, fill_value=False)
        if verbose:
            print(f'  {label:<34} {before:5d} -> {int(mask.sum()):5d}')

    if verbose:
        print(f'Filtering {len(df)} units:')

    br = _as_set(brain_region_inhibited)
    if br is not None and 'brain_region_inhibited' in df:
        _and(df['brain_region_inhibited'].isin(br), f'brain_region_inhibited in {sorted(br)}')

    cond = _as_set(condition)
    if cond is not None:
        col = 'condition' if 'condition' in df else 'hemisphere'
        _and(df[col].isin(cond), f'{col} in {sorted(cond)}')

    if recorded_region is not None:
        mode, allen_values = _region_selector_mode_values(recorded_region)
        if mode == 'include' and isinstance(recorded_region, str) and recorded_region.lower() == 'midbrain':
            _and(df['is_midbrain'] == True, "recorded region = midbrain (broad)")  # noqa: E712
        elif mode == 'exclude' and allen_values == {'midbrain'}:
            _and(df['is_midbrain'] != True, "recorded region != midbrain (broad)")  # noqa: E712
        else:
            match, mode, allen = _apply_region_selector_mask(df, 'Allenregion', recorded_region)
            op = 'not in' if mode == 'exclude' else 'in'
            _and(match, f'recorded Allen region {op} {sorted(allen)}')
    if recorded_region_beryl is not None:
        match, mode, beryl = _apply_region_selector_mask(df, 'Berylregion', recorded_region_beryl)
        op = 'not in' if mode == 'exclude' else 'in'
        _and(match, f'recorded Beryl region {op} {sorted(beryl)}')

    if bs_only:
        _and(df['BS_score'] == 1, 'BS_score == 1')
    if max_pval_empirical is not None:
        if 'pval_empirical' in df:
            pvals = pd.to_numeric(df['pval_empirical'], errors='coerce')
            _and(pvals <= float(max_pval_empirical),
                 f'pval_empirical <= {max_pval_empirical}')
        else:
            print('max_pval_empirical requested but this pickle lacks pval_empirical; skipping.')
    if exclude_drift_units and 'drift_unit' in df:
        _and(df['drift_unit'] == 0, 'exclude drift units')
    if exclude_nonstationary_units:
        if 'nonstationary_unit' in df:
            _and(df['nonstationary_unit'] == 0, 'exclude nonstationary units')
        else:
            print('exclude_nonstationary_units requested but this pickle lacks '
                  'nonstationary_unit; skipping.')

    nonstationarity_filters = [
        ('qp_fr_segment_range_frac', max_qp_fr_segment_range_frac, '<=',
         'QP raw segment range/median'),
        ('qp_resid_drift_range_frac', max_qp_resid_drift_range_frac, '<=',
         'QP residual segment range/median'),
        ('qp_resid_drift_cv', max_qp_resid_drift_cv, '<=',
         'QP residual segment CV'),
        ('qp_resid_abs_rho_time', max_qp_resid_abs_rho_time, '<=',
         'QP residual |rho time|'),
        ('qp_low_activity_fraction', max_qp_low_activity_fraction, '<=',
         'QP low-activity fraction'),
        ('qp_max_low_activity_run', max_qp_max_low_activity_run, '<=',
         'QP max low-activity run'),
        ('qp_block_effect_sign_consistency', min_qp_block_effect_sign_consistency, '>=',
         'QP block-effect sign consistency'),
        ('qp_block_effect_segment_cv', max_qp_block_effect_segment_cv, '<=',
         'QP block-effect segment CV'),
        ('qp_block_effect_dominance', max_qp_block_effect_dominance, '<=',
         'QP block-effect dominance'),
    ]
    for col, threshold, op, label in nonstationarity_filters:
        if threshold is None:
            continue
        if col not in df:
            print(f'{col} requested but this pickle lacks the column; skipping.')
            continue
        vals = pd.to_numeric(df[col], errors='coerce')
        if op == '<=':
            _and(vals <= float(threshold), f'{label} <= {threshold}')
        else:
            _and(vals >= float(threshold), f'{label} >= {threshold}')
    if exclude_axonal_units and 'ax_unit' in df:
        _and(df['ax_unit'] == 0, 'exclude axonal units')
    if exclude_light_artifact and 'light_artifact_auto' in df:
        _and(df['light_artifact_auto'] == 0, 'exclude light-artifact units')
    if exclude_amplitude_outliers and 'waveform_amplitude_outlier' in df:
        _and(df['waveform_amplitude_outlier'] == 0, 'exclude amplitude outliers')

    if IBL_quality_label_threshold is not None:
        _and(df['IBL_label'] >= IBL_quality_label_threshold,
             f'IBL_label >= {IBL_quality_label_threshold}')
    if presence_threshold is not None and 'presence_ratio' in df:
        _and(df['presence_ratio'] > presence_threshold,
             f'presence_ratio > {presence_threshold}')
    if min_firing_rate is not None and 'firing_rate' in df:
        _and(df['firing_rate'] >= min_firing_rate, f'firing_rate >= {min_firing_rate}')
    if min_n_per_block is not None and {'n_80_nonstim', 'n_20_nonstim'} <= set(df.columns):
        _and((df['n_80_nonstim'] >= min_n_per_block) & (df['n_20_nonstim'] >= min_n_per_block),
             f'>= {min_n_per_block} trials per block (nonstim)')
    if min_n_per_delta_block is not None:
        needed = {'n_80_nonstim_delta_peth', 'n_20_nonstim_delta_peth',
                  'n_80_inhib_delta_peth', 'n_20_inhib_delta_peth'}
        if needed <= set(df.columns):
            _and((df['n_80_nonstim_delta_peth'] >= min_n_per_delta_block)
                 & (df['n_20_nonstim_delta_peth'] >= min_n_per_delta_block)
                 & (df['n_80_inhib_delta_peth'] >= min_n_per_delta_block)
                 & (df['n_20_inhib_delta_peth'] >= min_n_per_delta_block),
                 f'>= {min_n_per_delta_block} trials per delta-PETH block')
        else:
            print('min_n_per_delta_block requested but this pickle lacks *_delta_peth count columns; skipping.')

    # ---- Pre-laser baseline-stability exclusions -------------------------------
    # Before the laser fires, opto and control trials should be identical, and a
    # unit we can assess should already express its block code. Several options
    # (computed from the stored pre-laser traces over [-prelaser_window_s, 0]):
    #   max_prelaser_delta_gap   : direct stim-control delta-FR gap, no sign flip
    #   max_cv_prelaser_delta_gap: same sanity check after split-half sign
    #                              orientation (requires diagnostic traces)
    #   max_prelaser_zdev          : drop units whose opto-vs-control block-delta
    #                                already differs pre-laser (|mean z| too big).
    #   min_prelaser_separation_frac: keep units whose pre-laser control block
    #                                separation is >= this fraction of their own
    #                                overall separation (i.e. a stable baseline).
    if max_prelaser_delta_gap is not None:
        gap = pd.Series(_prelaser_delta_gap(data, prelaser_window_s), index=df.index)
        _and(gap <= max_prelaser_delta_gap,
             f'pre-laser |stim-control delta| <= {max_prelaser_delta_gap}%')

    if max_cv_prelaser_delta_gap is not None:
        try:
            gap = pd.Series(_cv_prelaser_delta_gap(data, prelaser_window_s), index=df.index)
            _and(gap <= max_cv_prelaser_delta_gap,
                 f'CV pre-laser |stim-control delta| <= {max_cv_prelaser_delta_gap}%')
        except KeyError as exc:
            print(f'max_cv_prelaser_delta_gap requested but unavailable: {exc}')

    if (max_prelaser_zdev is not None or min_prelaser_separation_frac is not None
            or max_prelaser_trace_std is not None):
        zdev, sepfrac, noise = _prelaser_metrics(data, prelaser_window_s)
        zdev = pd.Series(zdev, index=df.index)
        sepfrac = pd.Series(sepfrac, index=df.index)
        noise = pd.Series(noise, index=df.index)
        if max_prelaser_zdev is not None:
            _and(zdev <= max_prelaser_zdev,
                 f'pre-laser |opto-control z| <= {max_prelaser_zdev} (win {prelaser_window_s}s)')
        if min_prelaser_separation_frac is not None:
            _and(sepfrac >= min_prelaser_separation_frac,
                 f'pre-laser block sep >= {min_prelaser_separation_frac} of baseline')
        if max_prelaser_trace_std is not None:
            _and(noise <= max_prelaser_trace_std,
                 f'pre-laser opto-trace std <= {max_prelaser_trace_std}% (win {prelaser_window_s}s)')

    # Low pre-laser baseline-FR exclusion (uses the raw stim PETH, trace_stim_all).
    # A unit whose stim-trial firing rate is near zero before the laser has no
    # reliable baseline to normalise against, so its delta is untrustworthy.
    if min_prelaser_baseline_fr is not None:
        if 'trace_stim_all' in data:
            t = np.asarray(data['peth_time'], dtype=float)
            pre = (t >= -abs(baseline_fr_window_s)) & (t < 0)
            sa = data['trace_stim_all']
            base_fr = np.array([np.nanmean(np.asarray(sa[i], dtype=float)[pre])
                                for i in range(len(df))])
            base_fr = pd.Series(base_fr, index=df.index)
            _and(base_fr >= min_prelaser_baseline_fr,
                 f'pre-laser stim baseline FR >= {min_prelaser_baseline_fr} Hz '
                 f'(win {baseline_fr_window_s}s)')
        else:
            print("min_prelaser_baseline_fr requested but no 'trace_stim_all' in "
                  "this pickle; skipping (re-run the BS pipeline to save it).")

    if pids is not None:
        _and(df['pid'].isin(_as_set(pids)), 'restrict to pids')
    if exclude_pids is not None:
        _and(~df['pid'].isin(_as_set(exclude_pids)), 'exclude pids')

    idx = np.where(mask.values)[0]
    return df.iloc[idx].copy(), idx


# ---------------------------------------------------------------------------
# Plotting (reproduces the original population figures)
# ---------------------------------------------------------------------------
def rolling_window_mean_1d(arr, window_bins):
    """CAUSAL left-padded sliding-window mean. NaN-aware: each window averages
    only its finite values (so zero_2_nan traces survive smoothing); identical to
    the original cumsum version when there are no NaNs."""
    arr = np.asarray(arr, dtype=float)
    if window_bins <= 1:
        return arr
    k = int(window_bins)
    a = np.pad(arr, (k, 0), mode='edge')
    finite = np.isfinite(a)
    a0 = np.where(finite, a, 0.0)
    csum = np.cumsum(a0)
    ccnt = np.cumsum(finite.astype(float))
    win_sum = csum[k:] - csum[:-k]
    win_cnt = ccnt[k:] - ccnt[:-k]
    with np.errstate(invalid='ignore'):
        out = win_sum / win_cnt
    out[win_cnt == 0] = np.nan
    return out


def centered_mean_1d(arr, window_bins):
    """Symmetric (non-causal) moving average; does NOT shift features in time,
    so it is preferable for heavy visual smoothing. NaN-aware (see above)."""
    arr = np.asarray(arr, dtype=float)
    if window_bins <= 1:
        return arr
    k = int(window_bins)
    pad = k // 2
    a = np.pad(arr, (pad, k - pad), mode='edge')
    finite = np.isfinite(a)
    a0 = np.where(finite, a, 0.0)
    csum = np.cumsum(np.insert(a0, 0, 0.0))
    ccnt = np.cumsum(np.insert(finite.astype(float), 0, 0.0))
    win_sum = csum[k:] - csum[:-k]
    win_cnt = ccnt[k:] - ccnt[:-k]
    with np.errstate(invalid='ignore'):
        out = win_sum / win_cnt
    out[win_cnt == 0] = np.nan
    return out[:len(arr)]


def _smooth(arr, bins, mode):
    return centered_mean_1d(arr, bins) if mode == 'centered' else rolling_window_mean_1d(arr, bins)


def _default_plot_time_range(data, time_range=None):
    """Default display window by alignment; user tuple overrides."""
    if time_range is not None:
        return tuple(time_range)
    align = str(data.get('onset_alignment', '')).lower()
    if 'feedback' in align:
        return (-5.0, 2.0)
    if 'go' in align or 'cue' in align:
        return (-5.0, 2.0)
    return (-2.0, 5.0)


def _time_mask_for_plot(t, time_range):
    t = np.asarray(t, dtype=float)
    if time_range is None:
        return np.isfinite(t)
    t0, t1 = tuple(time_range)
    return np.isfinite(t) & (t >= float(t0)) & (t <= float(t1))


def _resolve_plot_time_range(t, time_range):
    t = np.asarray(t, dtype=float)
    finite = t[np.isfinite(t)]
    if finite.size == 0:
        return time_range
    mask = _time_mask_for_plot(t, time_range)
    if not np.any(mask):
        print(f'Plot time_range={time_range} has no overlap with data range [{t[0]:g}, {t[-1]:g}]; showing full range.')
        return (float(finite[0]), float(finite[-1]))
    return tuple(time_range)


def _axis_size_to_figsize(axis_size, n_axes=1):
    if axis_size is None:
        axis_size = (6.5, 4.5)
    return (float(axis_size[0]) * int(n_axes), float(axis_size[1]))


def _stack_traces(data, idx, key, smooth_bins,
                  smooth_mode='causal', baseline_pre_mask=None):
    """Stack, optionally baseline-reference, and smooth selected traces.

    Sign orientation is deliberately handled only by
    :func:`_oriented_delta_traces`, where split-half independence is enforced.
    """
    traces = data[key]
    out = []
    for i in idx:
        tr = np.asarray(traces[i], dtype=float)
        if baseline_pre_mask is not None and baseline_pre_mask.any():
            tr = tr - np.nanmean(tr[baseline_pre_mask])
        out.append(_smooth(tr, smooth_bins, smooth_mode))
    return np.vstack(out) if out else np.empty((0, len(data['peth_time'])))


def plot_delta_fr(data, idx, title='', ax=None, smooth_ms=None,
                  smooth_mode='causal', baseline_subtract=False,
                  baseline_window=(-0.5, 0.0), sign_flip=None,
                  sign_mode=sign_mode_option, sign_window=DEFAULT_SIGN_WINDOW,
                  norm_mode=None, orientation_mode=None,
                  trial_estimator=None,
                  whole_control_min_fr=QP_CONTROL_SCALAR_MIN_FR,
                  time_range=None, axis_size=(6.5, 4.5),
                  y_range=None):
    """Black = control (nonstim), blue = laser (stim); delta FR 80/20 (% baseline),
    mean +/- SEM, with cross-validated control orientation by default.

    smooth_ms        : smoothing window in ms (None -> the pipeline's stored value).
                       Increase (e.g. 800-1500) for much less noisy traces.
    smooth_mode      : 'causal' uses only current/past bins and matches the
                       pipeline; 'centered' is symmetric and non-causal.
    baseline_subtract: reference each unit's traces to their own pre-laser mean
                       over `baseline_window`, removing constant offsets (e.g. an
                       opto trace sitting entirely below control) so only the
                       laser-induced CHANGE is shown.
    sign_mode        : 'none' keeps ordinary signed traces. 'block_crossfit'
                       learns preference from a temporally separate control
                       half and evaluates time/count-matched held-out control and
                       opto trials, then swaps/averages. 'split_half' is the older
                       random-control diagnostic; 'legacy' reproduces the biased
                       same-data sign flip for historical comparisons.
    sign_window      : fixed control window used only to learn direction. Defaults
                       to (-2, 0) seconds for every alignment.
    sign_flip        : compatibility alias. True now means split-half; False means
                       no orientation. Prefer sign_mode in new calls.
    norm_mode        : New orthogonal API: 'raw_hz', 'whole_control_scalar',
                       'qp_control_scalar', or a legacy saved normalization.
    orientation_mode : 'qp_preference', 'independent_absolute', or
                       'signed_80_minus_20'. When supplied, this supersedes
                       sign_mode/sign_flip.
    trial_estimator  : 'all_trials' uses every eligible inhibition-range trial;
                       'matched_crossfit' uses matched held-out evaluation folds.
                       When supplied, this supersedes the estimator implicit in
                       sign_mode.
    time_range       : display/analyze window in seconds. None defaults to
                       (-2, 5) for LaserOnset and (-5, 2) for GoCueOnset.
    axis_size        : figsize for this single axis when ax is None.
    """
    t = np.asarray(data['peth_time'])
    win_ms = smooth_ms if smooth_ms is not None else data.get('smoothing_window_ms', 300)
    pre = ((t >= baseline_window[0]) & (t < baseline_window[1])) if baseline_subtract else None
    use_primary_api = _requests_primary_api(
        norm_mode, orientation_mode, trial_estimator)
    if use_primary_api:
        ns, st, active_mode, resolved_orientation, resolved_estimator = (
            _primary_delta_traces(
                data, idx, norm_mode=norm_mode or 'raw_hz',
                orientation_mode=orientation_mode or 'qp_preference',
                trial_estimator=trial_estimator or 'all_trials',
                whole_control_min_fr=whole_control_min_fr,
            ))
        resolved_sign_mode = resolved_orientation
    else:
        ns, st, active_mode, resolved_sign_mode = _oriented_delta_traces(
            data, idx, mode=norm_mode, sign_mode=sign_mode,
            sign_window=sign_window, sign_flip=sign_flip, scale=100.0,
        )
        resolved_orientation = resolved_sign_mode
        resolved_estimator = (
            'matched_crossfit' if resolved_sign_mode == 'block_crossfit'
            else 'legacy_full_trace')
    if baseline_subtract and not use_primary_api and resolved_sign_mode == 'legacy':
        raise ValueError(
            "baseline_subtract with sign_mode='legacy' is circular: the control "
            "baseline chooses its own sign. Use sign_mode='block_crossfit'.")
    if pre is not None and pre.any():
        ns = ns - _nanmean(ns[:, pre], axis=1)[:, None]
        st = st - _nanmean(st[:, pre], axis=1)[:, None]
    ns = _maybe_smooth_rows(data, ns, win_ms, smooth_mode)
    st = _maybe_smooth_rows(data, st, win_ms, smooth_mode)
    valid_rows = (
        np.any(np.isfinite(ns), axis=1) & np.any(np.isfinite(st), axis=1)
    )
    ns, st = ns[valid_rows], st[valid_rows]
    n = ns.shape[0]
    if n == 0:
        print('No units selected; nothing to plot.')
        return None
    ns_m, st_m = np.nanmean(ns, axis=0), np.nanmean(st, axis=0)
    ns_e = _sem_rows(ns)
    st_e = _sem_rows(st)
    align = data.get('onset_alignment', 'Laser onset')
    time_range = _default_plot_time_range(data, time_range)
    time_range = _resolve_plot_time_range(t, time_range)
    show = _time_mask_for_plot(t, time_range)
    t_plot = t[show]

    if ax is None:
        _, ax = plt.subplots(figsize=axis_size)
    ax.plot(t_plot, st_m[show], color=OPTO_COLOR, linewidth=3, label='laser')
    ax.fill_between(t_plot, st_m[show] - st_e[show], st_m[show] + st_e[show], color=OPTO_COLOR, alpha=0.2)
    ax.plot(t_plot, ns_m[show], color='k', linewidth=3, label='control')
    ax.fill_between(t_plot, ns_m[show] - ns_e[show], ns_m[show] + ns_e[show], color='k', alpha=0.2)
    ax.axvline(0, linestyle='--', color='red')
    ax.set_xlabel(f'Time from {align} (s)')
    relation_label = (
        'absolute block difference'
        if resolved_orientation == 'independent_absolute'
        else 'preferred - nonpreferred'
        if resolved_orientation == 'qp_preference'
        else '80 - 20 blocks')
    ylab = f'Delta FR, {relation_label} ({_delta_value_label(active_mode)})'
    ax.set_ylabel(ylab + ', baseline-subtracted' if baseline_subtract else ylab)
    ax.set_xlim(time_range)
    if y_range is not None:
        ax.set_ylim(y_range)
    ax.set_title(f'{title}, n = {n}')
    return ax


def plot_primary_delta_option_grid(
        data, idx, *,
        norm_modes=('raw_hz', 'whole_control_scalar'),
        orientation_modes=('qp_preference', 'independent_absolute'),
        trial_estimators=('all_trials', 'matched_crossfit'),
        smooth_ms=300, smooth_mode='centered',
        baseline_subtract=False, baseline_window=(-2.0, 0.0),
        whole_control_min_fr=QP_CONTROL_SCALAR_MIN_FR,
        time_range=None, axis_size=(5.8, 3.4), suptitle=''):
    """Plot a coherent grid of the new primary delta-FR analysis options.

    Columns are normalization modes. Rows are every requested
    ``trial_estimator`` × ``orientation_mode`` combination. Each axis is drawn
    by :func:`plot_delta_fr`, so it is numerically identical to the standalone
    primary trace for that configuration. Axes use independent y limits because
    raw Hz and percent-normalized values are not directly commensurate.
    """
    norm_modes = tuple(norm_modes)
    orientation_modes = tuple(orientation_modes)
    trial_estimators = tuple(trial_estimators)
    rows = [
        (estimator, orientation)
        for estimator in trial_estimators
        for orientation in orientation_modes
    ]
    if not norm_modes or not rows:
        raise ValueError('Request at least one norm/orientation/estimator option.')
    fig, axes = plt.subplots(
        len(rows), len(norm_modes), squeeze=False,
        figsize=(axis_size[0] * len(norm_modes), axis_size[1] * len(rows)),
    )
    for row_i, (estimator, orientation) in enumerate(rows):
        for col_i, norm in enumerate(norm_modes):
            title = (
                f'{estimator.replace("_", " ")} | '
                f'{orientation.replace("_", " ")} | '
                f'{norm.replace("_", " ")}')
            plot_delta_fr(
                data, idx, title=title, ax=axes[row_i, col_i],
                smooth_ms=smooth_ms, smooth_mode=smooth_mode,
                baseline_subtract=baseline_subtract,
                baseline_window=baseline_window,
                norm_mode=norm, orientation_mode=orientation,
                trial_estimator=estimator,
                whole_control_min_fr=whole_control_min_fr,
                time_range=time_range, y_range=None,
            )
            axes[row_i, col_i].set_ylabel(
                f'Delta FR ({_delta_value_label(_canonical_primary_norm(data, norm))})')
    fig.suptitle(suptitle or 'Delta-FR primary-analysis option grid', y=0.998)
    fig.tight_layout(rect=(0, 0, 1, 0.985), h_pad=2.2, w_pad=3.2)
    return fig, axes


def _default_effect_window(data, window_s=None):
    if window_s is not None:
        return tuple(window_s)
    align = str(data.get('onset_alignment', '')).lower()
    if 'feedback' in align:
        return (-0.5, 0.0)
    if 'go' in align or 'cue' in align:
        return (-0.5, 0.0)
    return (0.0, 0.5)


def _delta_trace_keys(data, norm_mode=None):
    mode = _active_mode(data, norm_mode)
    kn = f'trace_nonstim_{mode}'
    ks = f'trace_stim_{mode}'
    if kn in data and ks in data:
        return kn, ks, mode
    if norm_mode is None and 'trace_nonstim' in data and 'trace_stim' in data:
        return 'trace_nonstim', 'trace_stim', mode
    raise KeyError(f"Could not find delta-FR traces for norm_mode='{mode}'.")


def _sem_rows(mat):
    mat = np.asarray(mat, dtype=float)
    n = np.sum(np.isfinite(mat), axis=0)
    with np.errstate(invalid='ignore', divide='ignore'):
        se = np.nanstd(mat, axis=0, ddof=1) / np.sqrt(n)
    se[n < 2] = np.nan
    return se


def _smooth_matrix_for_plot(mat, bin_size, smooth_ms=None, smooth_mode='centered'):
    mat = np.asarray(mat, dtype=float)
    if smooth_ms is None or smooth_ms <= 0:
        return mat
    bins = max(1, int(round(float(smooth_ms) / 1000.0 / float(bin_size))))
    return np.vstack([_smooth(row, bins, smooth_mode) for row in mat])


def _alignment_slug(data):
    align = str(data.get('onset_alignment', '')).lower()
    if 'feedback' in align:
        return 'feedback'
    if 'go' in align or 'cue' in align:
        return 'gocue'
    if 'laser' in align:
        return 'laser'
    return 'alignment'


def _shade_effect_window(ax, window_s):
    ax.axvspan(window_s[0], window_s[1], color='0.9', zorder=0)


def _insertion_delta_ylabel(summary, prefix):
    orientation = summary.get('orientation_mode', summary.get('sign_mode'))
    relation = {
        'qp_preference': 'preferred - nonpreferred',
        'independent_absolute': 'absolute block difference',
        'signed_80_minus_20': '80 - 20 blocks',
    }.get(orientation, '80/20 blocks')
    ylabel = (
        f"{prefix}, {relation} "
        f"({_delta_value_label(summary.get('norm_mode'))})")
    if summary.get('baseline_subtract'):
        ylabel += ', change from own baseline'
    return ylabel


def build_insertion_delta_fr_summary(data, idx, *, min_units_per_insertion=10,
                                     window_s=None, norm_mode=None,
                                     sign_mode=sign_mode_option,
                                     sign_window_s=DEFAULT_SIGN_WINDOW,
                                     orientation_mode=None,
                                     trial_estimator=None,
                                     whole_control_min_fr=QP_CONTROL_SCALAR_MIN_FR,
                                     min_control_delta_fr=0.0,
                                     baseline_subtract=False,
                                     baseline_window=(-0.5, 0.0)):
    """Average normalized delta-FR traces within insertion before statistics.

    `idx` should come from filter_units(...). The traces are percent baseline,
    matching plot_delta_fr. ``norm_mode='qp_control_scalar'`` uses the same
    control-only, block-balanced quiescence-period denominator for control and
    opto. Use ``block_crossfit`` for the preferred matched, held-out
    control-preference orientation, ``none`` for ordinary signed traces, or
    ``legacy`` only for historical comparisons. ``min_control_delta_fr`` is
    applied to each insertion's control scalar over ``window_s`` after any
    requested baseline subtraction; use ``None`` to disable this gate.

    New calls should set ``orientation_mode`` and ``trial_estimator`` explicitly.
    These supersede the estimator/orientation bundled into legacy ``sign_mode``.
    """
    if 'pid' not in data['units'].columns:
        raise KeyError("data['units'] needs a 'pid' column for insertion-level averaging.")

    idx = np.asarray(idx, dtype=int)
    if idx.size == 0:
        raise ValueError('No units selected; cannot build insertion-level summary.')

    window_s = _default_effect_window(data, window_s)
    effect_mask = _window(data, window_s)
    if not np.any(effect_mask):
        raise ValueError(f'window_s={window_s} is empty for this peth_time range.')
    if min_control_delta_fr is not None:
        min_control_delta_fr = float(min_control_delta_fr)
        if not np.isfinite(min_control_delta_fr):
            raise ValueError('min_control_delta_fr must be finite or None.')

    use_primary_api = _requests_primary_api(
        norm_mode, orientation_mode, trial_estimator)
    if use_primary_api:
        (ctrl_units, opto_units, active_mode, resolved_orientation,
         resolved_estimator) = _primary_delta_traces(
            data, idx, norm_mode=norm_mode or 'raw_hz',
            orientation_mode=orientation_mode or 'qp_preference',
            trial_estimator=trial_estimator or 'all_trials',
            whole_control_min_fr=whole_control_min_fr,
        )
        resolved_sign_mode = resolved_orientation
    else:
        ctrl_units, opto_units, active_mode, resolved_sign_mode = (
            _oriented_delta_traces(
                data, idx, mode=norm_mode, sign_mode=sign_mode,
                sign_window=sign_window_s, scale=100.0,
            ))
        resolved_orientation = resolved_sign_mode
        resolved_estimator = (
            'matched_crossfit' if resolved_sign_mode == 'block_crossfit'
            else 'legacy_full_trace')

    if baseline_subtract and not use_primary_api and resolved_sign_mode == 'legacy':
        raise ValueError(
            "baseline_subtract with sign_mode='legacy' is circular: the control "
            "baseline chooses its own sign. Use sign_mode='block_crossfit'.")

    if baseline_subtract:
        pre = _window(data, baseline_window)
        if not np.any(pre):
            raise ValueError(
                f'baseline_window={tuple(baseline_window)} is empty for peth_time')
        ctrl_units = ctrl_units - _nanmean(ctrl_units[:, pre], axis=1)[:, None]
        opto_units = opto_units - _nanmean(opto_units[:, pre], axis=1)[:, None]

    selected_units = data['units'].iloc[idx].copy().reset_index(drop=True)
    selected_units['_selected_pos'] = np.arange(idx.size)
    unit_count_rows = []
    removed_rows = []
    rows = []
    ctrl_traces = []
    opto_traces = []
    ctrl_unit_sem = []
    opto_unit_sem = []

    for pid, grp in selected_units.groupby('pid', sort=True):
        selected_positions = grp['_selected_pos'].to_numpy(int)
        n_selected_units = int(len(selected_positions))
        trace_valid = (
            np.any(np.isfinite(ctrl_units[selected_positions]), axis=1) &
            np.any(np.isfinite(opto_units[selected_positions]), axis=1)
        )
        positions = selected_positions[trace_valid]
        n_units = int(len(positions))
        meta = grp.iloc[0]
        count_row = {
            'pid': pid,
            'mouse': meta.get('mouse', np.nan),
            'brain_region_inhibited': meta.get('brain_region_inhibited', np.nan),
            'condition': meta.get('condition', meta.get('hemisphere', np.nan)),
            'n_selected_units': n_selected_units,
            'n_trace_valid_units': n_units,
            'min_units_per_insertion': int(min_units_per_insertion),
            'min_control_delta_fr': (
                min_control_delta_fr
                if min_control_delta_fr is not None else np.nan),
            'control_delta_fr': np.nan,
            'passes_control_delta_threshold': np.nan,
            'included_in_insertion_summary': False,
            'exclusion_reason': '',
        }
        if n_units < int(min_units_per_insertion):
            count_row['exclusion_reason'] = 'below_min_units'
            unit_count_rows.append(count_row)
            removed_rows.append({
                **count_row,
                'opto_delta_fr': np.nan,
                'window_start': window_s[0],
                'window_end': window_s[1],
            })
            continue
        cmat = ctrl_units[positions]
        omat = opto_units[positions]
        ctrace = _nanmean(cmat, axis=0)
        otrace = _nanmean(omat, axis=0)
        c_scalar = float(_nanmean(ctrace[effect_mask]))
        o_scalar = float(_nanmean(otrace[effect_mask]))
        count_row['control_delta_fr'] = c_scalar
        if not np.isfinite(c_scalar):
            exclusion_reason = 'nonfinite_control_delta_fr'
            passes_control_threshold = False
        elif (min_control_delta_fr is not None
              and c_scalar < min_control_delta_fr):
            exclusion_reason = 'control_delta_below_threshold'
            passes_control_threshold = False
        else:
            exclusion_reason = ''
            passes_control_threshold = True
        count_row['passes_control_delta_threshold'] = passes_control_threshold
        count_row['included_in_insertion_summary'] = not exclusion_reason
        count_row['exclusion_reason'] = exclusion_reason
        unit_count_rows.append(count_row)
        if exclusion_reason:
            removed_rows.append({
                **count_row,
                'opto_delta_fr': o_scalar,
                'window_start': window_s[0],
                'window_end': window_s[1],
            })
            continue
        ctrl_traces.append(ctrace)
        opto_traces.append(otrace)
        ctrl_unit_sem.append(_sem_rows(cmat))
        opto_unit_sem.append(_sem_rows(omat))
        rows.append({
            'pid': pid,
            'mouse': meta.get('mouse', np.nan),
            'brain_region_inhibited': meta.get('brain_region_inhibited', np.nan),
            'condition': meta.get('condition', meta.get('hemisphere', np.nan)),
            'n_units': n_units,
            'n_selected_units': n_selected_units,
            'control_delta_fr': c_scalar,
            'opto_delta_fr': o_scalar,
            'control_minus_opto': c_scalar - o_scalar,
            'diff_in_diff_opto_minus_control': (
                o_scalar - c_scalar if baseline_subtract else np.nan),
            'metric': (
                'change_from_own_baseline' if baseline_subtract else 'window_mean'),
            'window_start': window_s[0],
            'window_end': window_s[1],
            'min_control_delta_fr': (
                min_control_delta_fr
                if min_control_delta_fr is not None else np.nan),
            'norm_mode': active_mode,
            'qp_control_scalar_min_fr': (
                float(data.get('_qp_control_scalar_min_fr'))
                if active_mode == 'qp_control_scalar' else np.nan),
            'sign_mode': resolved_sign_mode,
            'orientation_mode': resolved_orientation,
            'trial_estimator': resolved_estimator,
            'analysis_api': 'orthogonal_primary' if use_primary_api else 'legacy_sign_mode',
            'whole_control_min_fr': (
                float(whole_control_min_fr)
                if active_mode == 'whole_control_scalar' else np.nan),
            'sign_window_start': (
                float(sign_window_s[0])
                if not use_primary_api and resolved_sign_mode != 'none'
                and sign_window_s is not None else np.nan),
            'sign_window_end': (
                float(sign_window_s[1])
                if not use_primary_api and resolved_sign_mode != 'none'
                and sign_window_s is not None else np.nan),
            'baseline_subtract': bool(baseline_subtract),
            'baseline_window_start': (
                float(baseline_window[0]) if baseline_subtract else np.nan),
            'baseline_window_end': (
                float(baseline_window[1]) if baseline_subtract else np.nan),
        })

    if not rows:
        reason_counts = pd.Series(
            [row['exclusion_reason'] for row in removed_rows]
        ).value_counts().to_dict()
        raise ValueError(
            'No insertions remained after insertion-level gates. '
            f'Exclusions: {reason_counts}. Lower min_units_per_insertion, lower '
            'min_control_delta_fr, or set min_control_delta_fr=None.'
        )

    removed_columns = [
        'pid', 'mouse', 'brain_region_inhibited', 'condition',
        'n_selected_units', 'n_trace_valid_units',
        'min_units_per_insertion', 'min_control_delta_fr',
        'control_delta_fr', 'opto_delta_fr',
        'passes_control_delta_threshold',
        'included_in_insertion_summary', 'exclusion_reason',
        'window_start', 'window_end',
    ]

    return {
        'time': np.asarray(data['peth_time'], dtype=float),
        'bin_size': float(data.get('bin_size', 0.05)),
        'alignment': data.get('onset_alignment', 'alignment'),
        'window_s': window_s,
        'min_control_delta_fr': min_control_delta_fr,
        'norm_mode': active_mode,
        'qp_control_scalar_min_fr': (
            float(data.get('_qp_control_scalar_min_fr'))
            if active_mode == 'qp_control_scalar' else None),
        'sign_mode': resolved_sign_mode,
        'orientation_mode': resolved_orientation,
        'trial_estimator': resolved_estimator,
        'analysis_api': 'orthogonal_primary' if use_primary_api else 'legacy_sign_mode',
        'whole_control_min_fr': (
            float(whole_control_min_fr)
            if active_mode == 'whole_control_scalar' else None),
        'sign_window_s': (
            tuple(sign_window_s)
            if not use_primary_api and resolved_sign_mode != 'none'
            and sign_window_s is not None else None),
        'baseline_subtract': baseline_subtract,
        'baseline_window': tuple(baseline_window) if baseline_subtract else None,
        'insertion_df': pd.DataFrame(rows),
        'insertion_unit_counts': pd.DataFrame(unit_count_rows),
        'removed_pids_df': pd.DataFrame(removed_rows, columns=removed_columns),
        'control_traces': np.vstack(ctrl_traces),
        'opto_traces': np.vstack(opto_traces),
        'control_unit_sem': np.vstack(ctrl_unit_sem),
        'opto_unit_sem': np.vstack(opto_unit_sem),
    }


def _insertion_delta_unit_payload(data, idx, pid, *, window_s=None,
                                  norm_mode=None,
                                  sign_mode=sign_mode_option,
                                  sign_window_s=DEFAULT_SIGN_WINDOW,
                                  baseline_subtract=False,
                                  baseline_window=(-0.5, 0.0)):
    """Assemble exact and component traces for one insertion diagnostic.

    The ``control`` and ``opto`` arrays returned here use precisely the same
    estimator and trace-validity rule as ``build_insertion_delta_fr_summary``.
    Additional raw/fold/QP fields are diagnostic only and do not alter the
    insertion analysis.
    """
    if 'pid' not in data['units'].columns:
        raise KeyError("data['units'] needs a 'pid' column.")
    idx = np.asarray(idx, dtype=int)
    if idx.size == 0:
        raise ValueError('No selected units were supplied.')
    selected_pid = (
        data['units'].iloc[idx]['pid'].astype(str).to_numpy() == str(pid))
    pid_idx = idx[selected_pid]
    if pid_idx.size == 0:
        available = sorted(data['units'].iloc[idx]['pid'].astype(str).unique())
        raise KeyError(
            f'PID {pid!r} has no units in idx. Selected PIDs include: '
            f'{available[:10]}{" ..." if len(available) > 10 else ""}')

    window_s = _default_effect_window(data, window_s)
    effect_mask = _window(data, window_s)
    if not np.any(effect_mask):
        raise ValueError(f'window_s={window_s} is empty for peth_time.')

    control, opto, active_mode, resolved_sign_mode = _oriented_delta_traces(
        data, pid_idx, mode=norm_mode, sign_mode=sign_mode,
        sign_window=sign_window_s, scale=100.0,
    )
    baseline_mask = None
    if baseline_subtract:
        if resolved_sign_mode == 'legacy':
            raise ValueError(
                "baseline_subtract with sign_mode='legacy' is circular.")
        baseline_mask = _window(data, baseline_window)
        if not np.any(baseline_mask):
            raise ValueError(
                f'baseline_window={tuple(baseline_window)} is empty for peth_time')
        control = control - _nanmean(control[:, baseline_mask], axis=1)[:, None]
        opto = opto - _nanmean(opto[:, baseline_mask], axis=1)[:, None]

    trace_valid = (
        np.any(np.isfinite(control), axis=1) &
        np.any(np.isfinite(opto), axis=1)
    )
    pid_idx = pid_idx[trace_valid]
    control = control[trace_valid]
    opto = opto[trace_valid]
    if pid_idx.size == 0:
        raise ValueError(
            f'PID {pid!r} has no trace-valid units under norm_mode={active_mode!r} '
            f'and sign_mode={resolved_sign_mode!r}.')

    t = np.asarray(data['peth_time'], dtype=float)
    sign_mask = (
        _sign_mask(data, sign_window_s)
        if resolved_sign_mode != 'none' and sign_window_s is not None else None)
    units = data['units'].reset_index(drop=True).iloc[pid_idx].copy()

    # Unoriented full block PETHs: these reveal whether the raw numerator is
    # genuinely small or merely cancels after orientation/held-out evaluation.
    raw_keys = (
        'trace_nonstim_80_raw', 'trace_nonstim_20_raw',
        'trace_stim_80_raw', 'trace_stim_20_raw',
    )
    missing_raw = [key for key in raw_keys if key not in data]
    raw = {}
    if not missing_raw:
        raw = {
            'control80': _stack_key(data, raw_keys[0], pid_idx),
            'control20': _stack_key(data, raw_keys[1], pid_idx),
            'opto80': _stack_key(data, raw_keys[2], pid_idx),
            'opto20': _stack_key(data, raw_keys[3], pid_idx),
        }

    fold_traces = {}
    fold_signs = {
        'a': np.full(pid_idx.size, np.nan),
        'b': np.full(pid_idx.size, np.nan),
    }
    fold_reference_means = {
        'a': np.full(pid_idx.size, np.nan),
        'b': np.full(pid_idx.size, np.nan),
    }
    if (resolved_sign_mode == 'block_crossfit'
            and block_crossfit_traces_available(data, active_mode)):
        for fold in ('a', 'b'):
            reference = _stack_key(
                data, f'trace_block_crossfit_reference_{fold}_{active_mode}', pid_idx)
            control_eval = _stack_key(
                data, f'trace_block_crossfit_control_eval_{fold}_{active_mode}', pid_idx)
            stim_eval = _stack_key(
                data, f'trace_block_crossfit_stim_eval_{fold}_{active_mode}', pid_idx)
            sign = _sign_from_reference(reference, sign_mask)[:, 0]
            fold_signs[fold] = sign
            fold_reference_means[fold] = (
                _nanmean(reference[:, sign_mask], axis=1)
                if sign_mask is not None else _nanmean(reference, axis=1))
            cfold = sign[:, None] * control_eval * 100.0
            ofold = sign[:, None] * stim_eval * 100.0
            if baseline_subtract:
                cfold = cfold - _nanmean(cfold[:, baseline_mask], axis=1)[:, None]
                ofold = ofold - _nanmean(ofold[:, baseline_mask], axis=1)[:, None]
            fold_traces[fold] = {'control': cfold, 'opto': ofold}

    # QP distributions expose the important distinction between the broad
    # control trials used for the BS call and the inhibition-range control
    # trials used by the plotted delta estimator/QP denominator.
    qp_groups = []
    qp_rows = data.get('qp_fr_per_trial')
    metadata = data.get('trial_metadata_by_pid', {})
    meta = metadata.get(str(pid), {}) if isinstance(metadata, dict) else {}
    probability = np.asarray(meta.get('probability_left', []), dtype=float)
    crossfit_trial_numbers = meta.get('crossfit_trial_numbers', {})

    def _combined_crossfit_control_trials(block):
        pieces = []
        if isinstance(crossfit_trial_numbers, dict):
            for fold in ('a', 'b'):
                fold_data = crossfit_trial_numbers.get(fold, {})
                role_data = (
                    fold_data.get('control_eval', {})
                    if isinstance(fold_data, dict) else {})
                values = role_data.get(block, []) if isinstance(role_data, dict) else []
                pieces.append(np.asarray(values, dtype=int))
        nonempty = [piece for piece in pieces if piece.size]
        return np.unique(np.concatenate(nonempty)) if nonempty else np.array([], dtype=int)

    matched_control20 = _combined_crossfit_control_trials('20')
    matched_control80 = _combined_crossfit_control_trials('80')
    for source_i in pid_idx:
        groups = {
            'BS 20': np.array([], dtype=float),
            'BS 80': np.array([], dtype=float),
            'range 20': np.array([], dtype=float),
            'range 80': np.array([], dtype=float),
            'matched 20': np.array([], dtype=float),
            'matched 80': np.array([], dtype=float),
        }
        if qp_rows is not None and int(source_i) < len(qp_rows):
            qp_fr = np.asarray(qp_rows[int(source_i)], dtype=float)

            def _qp_values(trial_ids, block_probability=None):
                trial_ids = np.asarray(trial_ids, dtype=int)
                ok = (trial_ids >= 0) & (trial_ids < qp_fr.size)
                if block_probability is not None and probability.size:
                    ok &= (trial_ids < probability.size)
                    trial_ids = trial_ids[ok]
                    trial_ids = trial_ids[
                        np.isclose(probability[trial_ids], block_probability)]
                else:
                    trial_ids = trial_ids[ok]
                values = qp_fr[trial_ids]
                return values[np.isfinite(values)]

            bs_trials = meta.get('nonstim_trials_bs', [])
            groups['BS 20'] = _qp_values(bs_trials, 0.2)
            groups['BS 80'] = _qp_values(bs_trials, 0.8)
            groups['range 20'] = _qp_values(meta.get('nonstim_20_delta', []))
            groups['range 80'] = _qp_values(meta.get('nonstim_80_delta', []))
            groups['matched 20'] = _qp_values(matched_control20)
            groups['matched 80'] = _qp_values(matched_control80)
        qp_groups.append(groups)

    def _saved_values(column):
        if column not in data:
            return np.full(pid_idx.size, np.nan)
        values = np.asarray(data[column], dtype=float)
        return values[pid_idx]

    qp80 = _saved_values('qp_control_scalar_baseline_80_hz')
    qp20 = _saved_values('qp_control_scalar_baseline_20_hz')
    qpden = _saved_values('qp_control_scalar_baseline_hz')
    control_effect = _nanmean(control[:, effect_mask], axis=1)
    opto_effect = _nanmean(opto[:, effect_mask], axis=1)
    control_sign_window = (
        _nanmean(control[:, sign_mask], axis=1)
        if sign_mask is not None else np.full(pid_idx.size, np.nan))
    opto_sign_window = (
        _nanmean(opto[:, sign_mask], axis=1)
        if sign_mask is not None else np.full(pid_idx.size, np.nan))

    bs_qp_delta = np.asarray([
        (_nanmean(g['BS 80']) - _nanmean(g['BS 20']))
        if len(g['BS 80']) and len(g['BS 20']) else np.nan
        for g in qp_groups
    ])
    range_qp_delta = np.asarray([
        (_nanmean(g['range 80']) - _nanmean(g['range 20']))
        if len(g['range 80']) and len(g['range 20']) else np.nan
        for g in qp_groups
    ])
    matched_qp_delta = np.asarray([
        (_nanmean(g['matched 80']) - _nanmean(g['matched 20']))
        if len(g['matched 80']) and len(g['matched 20']) else np.nan
        for g in qp_groups
    ])
    qp_preference_sign = np.sign(bs_qp_delta)

    full_control_qp_oriented = np.full_like(control, np.nan)
    if raw:
        valid_denominator = np.isfinite(qpden) & (qpden > 0)
        with np.errstate(invalid='ignore', divide='ignore'):
            full_control_qp_oriented[valid_denominator] = (
                qp_preference_sign[valid_denominator, None]
                * (raw['control80'][valid_denominator]
                   - raw['control20'][valid_denominator])
                / qpden[valid_denominator, None] * 100.0
            )

    # Sensitivity views only: keep the actual estimator unchanged, but expose
    # (a) held-out trials oriented by the broad QP BS preference, and (b) a
    # fold-count-weighted version of the actual PETH-crossfit orientation. The
    # latter diagnoses whether a fold with very few trials is receiving a
    # disproportionate 50% weight in the current two-fold average.
    heldout_qp_oriented = np.full_like(control, np.nan)
    count_weighted_control = np.full_like(control, np.nan)
    count_weighted_opto = np.full_like(opto, np.nan)
    crossfit_counts = meta.get('crossfit_counts', {}) if isinstance(meta, dict) else {}
    diagnostic_min_events = int(
        data.get('run_config', {}).get('diagnostic_min_events_per_peth', 1))
    usable_eval_folds = {}
    if fold_traces:
        qp_oriented_folds = []
        actual_control_folds = []
        actual_opto_folds = []
        effective_weights = []
        for fold in ('a', 'b'):
            actual_control_folds.append(fold_traces[fold]['control'])
            actual_opto_folds.append(fold_traces[fold]['opto'])
            qp_oriented_folds.append(
                fold_traces[fold]['control']
                * fold_signs[fold][:, None]
                * qp_preference_sign[:, None])
            n80 = float(crossfit_counts.get(
                f'crossfit_{fold}_control_80', np.nan))
            n20 = float(crossfit_counts.get(
                f'crossfit_{fold}_control_20', np.nan))
            usable_eval_folds[fold] = bool(
                np.any(np.isfinite(fold_traces[fold]['control']))
                and np.any(np.isfinite(fold_traces[fold]['opto'])))
            effective_weights.append(
                1.0 / (1.0 / n80 + 1.0 / n20)
                if np.isfinite(n80) and np.isfinite(n20)
                and n80 >= diagnostic_min_events
                and n20 >= diagnostic_min_events
                and usable_eval_folds[fold] else np.nan)

        heldout_qp_oriented = _nanmean(
            np.stack(qp_oriented_folds, axis=0), axis=0)

        def _weighted_fold_mean(fold_arrays, weights):
            arrays = np.stack(fold_arrays, axis=0)
            weights = np.asarray(weights, dtype=float)[:, None, None]
            finite = np.isfinite(arrays) & np.isfinite(weights)
            numerator = np.nansum(
                np.where(finite, arrays * weights, 0.0), axis=0)
            denominator = np.sum(
                np.where(finite, weights, 0.0), axis=0)
            out = np.full(arrays.shape[1:], np.nan, dtype=float)
            np.divide(numerator, denominator, out=out, where=denominator > 0)
            return out

        count_weighted_control = _weighted_fold_mean(
            actual_control_folds, effective_weights)
        count_weighted_opto = _weighted_fold_mean(
            actual_opto_folds, effective_weights)
    else:
        effective_weights = [np.nan, np.nan]
        usable_eval_folds = {'a': False, 'b': False}

    full_control_qp_effect = _nanmean(
        full_control_qp_oriented[:, effect_mask], axis=1)
    heldout_qp_effect = _nanmean(
        heldout_qp_oriented[:, effect_mask], axis=1)
    count_weighted_control_effect = _nanmean(
        count_weighted_control[:, effect_mask], axis=1)
    count_weighted_opto_effect = _nanmean(
        count_weighted_opto[:, effect_mask], axis=1)

    def _unit_column(name, fallback=np.nan):
        if name in units:
            return pd.to_numeric(units[name], errors='coerce').to_numpy(float)
        return np.full(pid_idx.size, fallback, dtype=float)

    unit_df = pd.DataFrame({
        'source_row': pid_idx,
        'pid': str(pid),
        'clustnum': _unit_column('clustnum'),
        'Allenregion': units.get('Allenregion', pd.Series('', index=units.index)).astype(str).to_numpy(),
        'BS_score': _unit_column('BS_score'),
        'pval_empirical': _unit_column('pval_empirical'),
        'stat_real': _unit_column('stat_real'),
        'control_delta_effect_window': control_effect,
        'opto_delta_effect_window': opto_effect,
        'control_minus_opto_effect_window': control_effect - opto_effect,
        'control_delta_sign_window': control_sign_window,
        'opto_delta_sign_window': opto_sign_window,
        'crossfit_reference_sign_a': fold_signs['a'],
        'crossfit_reference_sign_b': fold_signs['b'],
        'crossfit_reference_mean_a': fold_reference_means['a'],
        'crossfit_reference_mean_b': fold_reference_means['b'],
        'crossfit_reference_sign_agrees': (
            np.isfinite(fold_signs['a']) & np.isfinite(fold_signs['b']) &
            (fold_signs['a'] == fold_signs['b'])
        ),
        'crossfit_eval_fold_a_usable': bool(usable_eval_folds['a']),
        'crossfit_eval_fold_b_usable': bool(usable_eval_folds['b']),
        'bs_qp_delta_hz': bs_qp_delta,
        'delta_range_qp_delta_hz': range_qp_delta,
        'matched_control_qp_delta_hz': matched_qp_delta,
        'qp_control_80_hz': qp80,
        'qp_control_20_hz': qp20,
        'qp_control_denominator_hz': qpden,
        'full_control_qp_oriented_effect_window': full_control_qp_effect,
        'heldout_control_qp_oriented_effect_window': heldout_qp_effect,
        'count_weighted_control_effect_window': count_weighted_control_effect,
        'count_weighted_opto_effect_window': count_weighted_opto_effect,
    })

    def _cancellation_ratio(values):
        values = np.asarray(values, dtype=float)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return np.nan
        denominator = float(np.mean(np.abs(finite)))
        return abs(float(np.mean(finite))) / denominator if denominator > 0 else np.nan

    sign_pair_valid = np.isfinite(fold_signs['a']) & np.isfinite(fold_signs['b'])
    summary = {
        'pid': str(pid),
        'alignment': data.get('onset_alignment', 'alignment'),
        'norm_mode': active_mode,
        'sign_mode': resolved_sign_mode,
        'window_s': tuple(window_s),
        'sign_window_s': (
            tuple(sign_window_s) if sign_mask is not None else None),
        'baseline_subtract': bool(baseline_subtract),
        'baseline_window': (
            tuple(baseline_window) if baseline_subtract else None),
        'n_selected_pid_units': int(selected_pid.sum()),
        'n_trace_valid_units': int(pid_idx.size),
        'control_delta_effect_window_mean': float(_nanmean(control_effect)),
        'opto_delta_effect_window_mean': float(_nanmean(opto_effect)),
        'control_minus_opto_effect_window_mean': float(
            _nanmean(control_effect - opto_effect)),
        'control_effect_window_mean_abs_unit_delta': float(
            _nanmean(np.abs(control_effect))),
        'control_effect_window_cancellation_ratio': _cancellation_ratio(control_effect),
        'control_effect_window_positive_fraction': float(
            np.mean(control_effect[np.isfinite(control_effect)] > 0))
            if np.any(np.isfinite(control_effect)) else np.nan,
        'control_sign_window_mean': float(_nanmean(control_sign_window)),
        'control_sign_window_mean_abs_unit_delta': float(
            _nanmean(np.abs(control_sign_window))),
        'control_sign_window_cancellation_ratio': _cancellation_ratio(
            control_sign_window),
        'n_crossfit_reference_sign_pairs': int(sign_pair_valid.sum()),
        'crossfit_reference_sign_agreement': float(
            np.mean(fold_signs['a'][sign_pair_valid]
                    == fold_signs['b'][sign_pair_valid]))
            if np.any(sign_pair_valid) else np.nan,
        'median_qp_control_denominator_hz': float(np.nanmedian(qpden))
            if np.any(np.isfinite(qpden)) else np.nan,
        'crossfit_fold_a_effective_block_count': float(effective_weights[0]),
        'crossfit_fold_b_effective_block_count': float(effective_weights[1]),
        'diagnostic_min_events_per_peth': diagnostic_min_events,
        'n_usable_crossfit_eval_folds': int(sum(usable_eval_folds.values())),
        'full_control_qp_oriented_effect_window_mean': float(
            _nanmean(full_control_qp_effect)),
        'heldout_control_qp_oriented_effect_window_mean': float(
            _nanmean(heldout_qp_effect)),
        'count_weighted_control_effect_window_mean': float(
            _nanmean(count_weighted_control_effect)),
        'count_weighted_opto_effect_window_mean': float(
            _nanmean(count_weighted_opto_effect)),
        'bs_vs_delta_range_qp_sign_agreement': float(np.mean(
            np.sign(bs_qp_delta[
                np.isfinite(bs_qp_delta) & np.isfinite(range_qp_delta)])
            == np.sign(range_qp_delta[
                np.isfinite(bs_qp_delta) & np.isfinite(range_qp_delta)])))
            if np.any(np.isfinite(bs_qp_delta) & np.isfinite(range_qp_delta)) else np.nan,
        'bs_vs_matched_control_qp_sign_agreement': float(np.mean(
            np.sign(bs_qp_delta[
                np.isfinite(bs_qp_delta) & np.isfinite(matched_qp_delta)])
            == np.sign(matched_qp_delta[
                np.isfinite(bs_qp_delta) & np.isfinite(matched_qp_delta)])))
            if np.any(np.isfinite(bs_qp_delta) & np.isfinite(matched_qp_delta)) else np.nan,
    }
    return {
        'time': t,
        'bin_size': float(data.get('bin_size', 0.05)),
        'unit_indices': pid_idx,
        'control': control,
        'opto': opto,
        'raw': raw,
        'fold_traces': fold_traces,
        'crossfit_counts': crossfit_counts,
        'effective_fold_weights': np.asarray(effective_weights, dtype=float),
        'usable_eval_folds': usable_eval_folds,
        'full_control_qp_oriented': full_control_qp_oriented,
        'heldout_qp_oriented': heldout_qp_oriented,
        'count_weighted_control': count_weighted_control,
        'count_weighted_opto': count_weighted_opto,
        'qp_groups': qp_groups,
        'unit_df': unit_df,
        'summary': summary,
        'effect_mask': effect_mask,
        'sign_mask': sign_mask,
    }


def diagnose_insertion_delta_units(data, idx, pid, *, window_s=None,
                                    norm_mode=None,
                                    sign_mode=sign_mode_option,
                                    sign_window_s=DEFAULT_SIGN_WINDOW,
                                    baseline_subtract=False,
                                    baseline_window=(-0.5, 0.0),
                                    csv_path=None, verbose=True):
    """Quantify unit/fold cancellation for one insertion without changing it.

    Returns a payload containing an insertion summary, one row per contributing
    unit, and the exact/component traces used by
    :func:`browse_insertion_delta_units`. ``csv_path`` optionally saves the
    per-unit audit table.
    """
    payload = _insertion_delta_unit_payload(
        data, idx, pid, window_s=window_s, norm_mode=norm_mode,
        sign_mode=sign_mode, sign_window_s=sign_window_s,
        baseline_subtract=baseline_subtract,
        baseline_window=baseline_window,
    )
    if csv_path is not None:
        csv_path = Path(csv_path).expanduser()
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        payload['unit_df'].to_csv(csv_path, index=False)
        payload['csv_path'] = csv_path
    if verbose:
        print('Insertion unit-level delta diagnostic:')
        print(pd.Series(payload['summary']).to_string())
        print(
            '\nCancellation ratio = |mean unit delta| / mean(|unit delta|). '
            'Values near 0 indicate strong cancellation; values near 1 indicate '
            'a consistent orientation. This is descriptive, not an exclusion test.'
        )
        if csv_path is not None:
            print(f'Per-unit audit -> {csv_path}')
    return payload


def _draw_insertion_delta_unit(payload, position, axes, *, smooth_ms=300,
                               smooth_mode='centered'):
    """Draw one unit from an insertion diagnostic payload onto four axes."""
    position = int(position)
    unit_df = payload['unit_df']
    if position < 0 or position >= len(unit_df):
        raise IndexError(position)
    row = unit_df.iloc[position]
    t = payload['time']
    smooth = lambda trace: _smooth_for_plot(
        {'bin_size': payload['bin_size']}, trace,
        smooth_ms=smooth_ms, smooth_mode=smooth_mode)
    ax_exact, ax_raw, ax_folds, ax_qp = np.asarray(axes).ravel()
    for ax in (ax_exact, ax_raw, ax_folds, ax_qp):
        ax.clear()

    summary = payload['summary']
    for ax in (ax_exact, ax_folds):
        _shade_effect_window(ax, summary['window_s'])
        if summary['sign_window_s'] is not None:
            ax.axvspan(*summary['sign_window_s'], color='0.75', alpha=0.22,
                       zorder=0)
        ax.axvline(0, linestyle='--', color='red', lw=0.9)
        ax.axhline(0, color='0.6', lw=0.7)

    control = smooth(payload['control'][position])
    opto = smooth(payload['opto'][position])
    ax_exact.plot(t, control, color='black', lw=2.1, label='control')
    ax_exact.plot(t, opto, color=OPTO_COLOR, lw=2.1, label='opto')
    ax_exact.set_title(
        'Exact trace entering insertion mean\n'
        f"window: C={row['control_delta_effect_window']:.2f}, "
        f"O={row['opto_delta_effect_window']:.2f}")
    ax_exact.set_ylabel(
        'Delta FR (% control QP baseline)'
        if summary['norm_mode'] == 'qp_control_scalar'
        else 'Delta FR (% baseline)')
    ax_exact.legend(frameon=False, fontsize=8)

    raw = payload['raw']
    if raw:
        ax_raw.plot(t, smooth(raw['control80'][position]), color='black', lw=1.8,
                    label='control 80')
        ax_raw.plot(t, smooth(raw['control20'][position]), color='black', lw=1.8,
                    linestyle='--', label='control 20')
        ax_raw.plot(t, smooth(raw['opto80'][position]), color=OPTO_COLOR, lw=1.8,
                    label='opto 80')
        ax_raw.plot(t, smooth(raw['opto20'][position]), color=OPTO_COLOR, lw=1.8,
                    linestyle='--', label='opto 20')
        ax_raw.axvline(0, linestyle='--', color='red', lw=0.9)
        ax_raw.set_title('Full, unoriented block PETHs')
        ax_raw.set_ylabel('Firing rate (Hz)')
        ax_raw.legend(frameon=False, fontsize=8, ncol=2)
    else:
        ax_raw.text(0.5, 0.5, 'Raw block PETHs unavailable',
                    ha='center', va='center', transform=ax_raw.transAxes)
        ax_raw.set_title('Full, unoriented block PETHs')

    if payload['fold_traces']:
        for fold, linestyle in (('a', '-'), ('b', '--')):
            ax_folds.plot(
                t, smooth(payload['fold_traces'][fold]['control'][position]),
                color='black', alpha=0.85, linestyle=linestyle, lw=1.8,
                label=f'control fold {fold}')
            ax_folds.plot(
                t, smooth(payload['fold_traces'][fold]['opto'][position]),
                color=OPTO_COLOR, alpha=0.85, linestyle=linestyle, lw=1.8,
                label=f'opto fold {fold}')
        sign_a = row['crossfit_reference_sign_a']
        sign_b = row['crossfit_reference_sign_b']
        agree = bool(row['crossfit_reference_sign_agrees'])
        counts = payload.get('crossfit_counts', {})
        count_text = (
            f"eval A 80/20={counts.get('crossfit_a_control_80', '?')}/"
            f"{counts.get('crossfit_a_control_20', '?')} "
            f"({'used' if payload['usable_eval_folds'].get('a') else 'missing'}), "
            f"B 80/20={counts.get('crossfit_b_control_80', '?')}/"
            f"{counts.get('crossfit_b_control_20', '?')} "
            f"({'used' if payload['usable_eval_folds'].get('b') else 'missing'})")
        ax_folds.set_title(
            f'Cross-fit folds: signs A={sign_a:g}, B={sign_b:g}, '
            f'agree={agree}\n{count_text}; count-weighted sensitivity '
            f"C={row['count_weighted_control_effect_window']:.2f}",
            fontsize=9)
        ax_folds.set_ylabel(ax_exact.get_ylabel())
        ax_folds.legend(frameon=False, fontsize=8, ncol=2)
    else:
        ax_folds.text(0.5, 0.5, 'Cross-fit fold traces not used',
                      ha='center', va='center', transform=ax_folds.transAxes)
        ax_folds.set_title(f"sign_mode={summary['sign_mode']}")

    groups = payload['qp_groups'][position]
    labels = [
        'BS 20', 'BS 80', 'range 20', 'range 80',
        'matched 20', 'matched 80']
    colors = ['0.55', 'black'] * 3
    rng = np.random.default_rng(int(row['source_row']))
    for x, (label, color) in enumerate(zip(labels, colors)):
        values = np.asarray(groups[label], dtype=float)
        if values.size:
            jitter = rng.uniform(-0.13, 0.13, size=values.size)
            ax_qp.scatter(x + jitter, values, s=12, color=color, alpha=0.42,
                          edgecolors='none')
            ax_qp.plot([x - 0.18, x + 0.18], [np.mean(values)] * 2,
                       color='crimson', lw=2.0)
    ax_qp.set_xticks(range(len(labels)))
    ax_qp.set_xticklabels(labels, rotation=20, ha='right')
    ax_qp.set_ylabel('Quiescence-period FR (Hz)')
    ax_qp.set_title(
        'QP delta 80-20: '
        f"BS={row['bs_qp_delta_hz']:.2f}; "
        f"range={row['delta_range_qp_delta_hz']:.2f}; "
        f"matched={row['matched_control_qp_delta_hz']:.2f} Hz\n"
        f"control QP normalizer={row['qp_control_denominator_hz']:.2f} Hz",
        fontsize=9)

    for ax in (ax_exact, ax_raw, ax_folds):
        ax.set_xlim(t[0], t[-1])
        ax.set_xlabel(f"Time from {summary['alignment']} (s)")
    pval = row['pval_empirical']
    ptext = f'{pval:.3g}' if np.isfinite(pval) else 'NA'
    return (
        f"PID {str(summary['pid'])[:8]} | unit {position + 1}/{len(unit_df)} | "
        f"row {int(row['source_row'])} | cluster {int(row['clustnum'])} | "
        f"{row['Allenregion']} | BS p={ptext}"
    )


def browse_insertion_delta_units(data, idx, pid, *, window_s=None,
                                 norm_mode=None,
                                 sign_mode=sign_mode_option,
                                 sign_window_s=DEFAULT_SIGN_WINDOW,
                                 baseline_subtract=False,
                                 baseline_window=(-0.5, 0.0),
                                 smooth_ms=300,
                                 smooth_mode='centered',
                                 start=0,
                                 sort_by='pval_empirical',
                                 diagnostics_csv=None,
                                 save_dir=None,
                                 figsize=(13.5, 8.5),
                                 block=True):
    """Interactively browse every trace-valid unit entering one PID mean.

    The upper-left panel is the exact unit trace used by
    ``run_insertion_delta_fr_analysis`` under the supplied settings. The other
    panels show raw block PETHs, the two independently oriented held-out folds,
    and QP activity for the broad BS trial set versus the inhibition-range
    control subset.

    Use Left/Right (or p/n) to navigate, Home/End to jump, ``s`` to save the
    current unit when ``save_dir`` is provided, and q/Escape to close. The
    Previous/Next buttons work in notebook and interactive backends.
    """
    payload = diagnose_insertion_delta_units(
        data, idx, pid, window_s=window_s, norm_mode=norm_mode,
        sign_mode=sign_mode, sign_window_s=sign_window_s,
        baseline_subtract=baseline_subtract,
        baseline_window=baseline_window,
        csv_path=diagnostics_csv, verbose=True,
    )
    unit_df = payload['unit_df']
    if sort_by is not None:
        sort_key = str(sort_by)
        if sort_key == 'abs_control_delta_effect_window':
            order = np.argsort(
                -np.abs(unit_df['control_delta_effect_window'].to_numpy(float)))
        elif sort_key in unit_df.columns:
            values = pd.to_numeric(unit_df[sort_key], errors='coerce').to_numpy(float)
            order = np.argsort(np.where(np.isfinite(values), values, np.inf))
        else:
            raise ValueError(
                f'Unknown sort_by={sort_by!r}; use a unit_df column, '
                "'abs_control_delta_effect_window', or None.")
        for key in (
                'unit_indices', 'control', 'opto',
                'full_control_qp_oriented', 'heldout_qp_oriented',
                'count_weighted_control', 'count_weighted_opto'):
            payload[key] = payload[key][order]
        for key in payload['raw']:
            payload['raw'][key] = payload['raw'][key][order]
        for fold in payload['fold_traces'].values():
            fold['control'] = fold['control'][order]
            fold['opto'] = fold['opto'][order]
        payload['qp_groups'] = [payload['qp_groups'][i] for i in order]
        payload['unit_df'] = unit_df.iloc[order].reset_index(drop=True)
        unit_df = payload['unit_df']

    if len(unit_df) == 0:
        raise ValueError('No trace-valid units to browse.')
    state = {'position': int(np.clip(int(start), 0, len(unit_df) - 1))}
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.subplots_adjust(bottom=0.12, top=0.88, hspace=0.43, wspace=0.27)

    def _redraw():
        title = _draw_insertion_delta_unit(
            payload, state['position'], axes,
            smooth_ms=smooth_ms, smooth_mode=smooth_mode)
        fig.suptitle(title, fontsize=12)
        fig.canvas.draw_idle()

    def _move(step):
        state['position'] = (state['position'] + int(step)) % len(unit_df)
        _redraw()

    def _save_current():
        if save_dir is None:
            print('Set save_dir=... to enable saving with the s key.')
            return
        row = payload['unit_df'].iloc[state['position']]
        directory = Path(save_dir).expanduser()
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / (
            f"insertion_unit_delta_{str(pid)}_cluster{int(row['clustnum'])}.png")
        fig.savefig(path, dpi=180, bbox_inches='tight')
        print(f'Saved current unit -> {path}')

    def _on_key(event):
        if event.key in ('right', 'down', 'n', ' '):
            _move(1)
        elif event.key in ('left', 'up', 'p'):
            _move(-1)
        elif event.key == 'home':
            state['position'] = 0
            _redraw()
        elif event.key == 'end':
            state['position'] = len(unit_df) - 1
            _redraw()
        elif event.key == 's':
            _save_current()
        elif event.key in ('q', 'escape'):
            plt.close(fig)

    from matplotlib.widgets import Button
    previous_ax = fig.add_axes([0.38, 0.025, 0.10, 0.045])
    next_ax = fig.add_axes([0.52, 0.025, 0.10, 0.045])
    previous_button = Button(previous_ax, 'Previous')
    next_button = Button(next_ax, 'Next')
    previous_button.on_clicked(lambda event: _move(-1))
    next_button.on_clicked(lambda event: _move(1))
    fig.canvas.mpl_connect('key_press_event', _on_key)
    # Keep widgets alive for the lifetime of the figure.
    fig._insertion_delta_browser = {
        'previous_button': previous_button,
        'next_button': next_button,
        'payload': payload,
        'state': state,
    }
    _redraw()
    plt.show(block=block)
    return payload


def _plot_insertion_delta_trace(summary, row_i, out_dir, *,
                                smooth_ms=300, smooth_mode='centered'):
    t = summary['time']
    c = _smooth_matrix_for_plot(summary['control_traces'][[row_i]], summary['bin_size'],
                                smooth_ms, smooth_mode)[0]
    o = _smooth_matrix_for_plot(summary['opto_traces'][[row_i]], summary['bin_size'],
                                smooth_ms, smooth_mode)[0]
    c_sem = _smooth_matrix_for_plot(summary['control_unit_sem'][[row_i]], summary['bin_size'],
                                    smooth_ms, smooth_mode)[0]
    o_sem = _smooth_matrix_for_plot(summary['opto_unit_sem'][[row_i]], summary['bin_size'],
                                    smooth_ms, smooth_mode)[0]
    row = summary['insertion_df'].iloc[row_i]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    _shade_effect_window(ax, summary['window_s'])
    ax.plot(t, c, color='black', lw=2.2, label='control')
    ax.plot(t, o, color=OPTO_COLOR, lw=2.2, label='opto')
    if np.isfinite(c_sem).any():
        ax.fill_between(t, c - c_sem, c + c_sem, color='black', alpha=0.18, linewidth=0)
    if np.isfinite(o_sem).any():
        ax.fill_between(t, o - o_sem, o + o_sem, color=OPTO_COLOR, alpha=0.18, linewidth=0)
    ax.axvline(0, linestyle='--', color='red', lw=1.0)
    ax.axhline(0, color='0.6', lw=0.7)
    ax.set_xlim(t[0], t[-1])
    ax.set_xlabel(f"Time from {summary['alignment']} (s)")
    ax.set_ylabel(_insertion_delta_ylabel(summary, 'Delta FR 80/20 blocks'))
    ax.set_title(f"{str(row['pid'])[:8]}, n units={int(row['n_units'])}")
    ax.legend(frameon=False)
    fig.tight_layout()
    out_path = Path(out_dir) / f"insertion_delta_fr_{str(row['pid'])}.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def _stats_from_insertion_df(df):
    c = df['control_delta_fr'].to_numpy(float)
    o = df['opto_delta_fr'].to_numpy(float)
    ok = np.isfinite(c) & np.isfinite(o)
    c, o = c[ok], o[ok]
    diff = c - o
    out = {
        'n_insertions': int(diff.size),
        'control_mean': float(np.nanmean(c)) if diff.size else np.nan,
        'opto_mean': float(np.nanmean(o)) if diff.size else np.nan,
        'control_minus_opto_mean': float(np.nanmean(diff)) if diff.size else np.nan,
        'opto_minus_control_mean': float(np.nanmean(-diff)) if diff.size else np.nan,
        'control_sem': float(np.nanstd(c, ddof=1) / np.sqrt(c.size)) if c.size >= 2 else np.nan,
        'opto_sem': float(np.nanstd(o, ddof=1) / np.sqrt(o.size)) if o.size >= 2 else np.nan,
        'delta_sem': float(np.nanstd(diff, ddof=1) / np.sqrt(diff.size)) if diff.size >= 2 else np.nan,
        'paired_t_p': np.nan,
        'wilcoxon_p': np.nan,
        'cohen_dz': np.nan,
    }
    if diff.size >= 2:
        out['paired_t_p'] = float(stats.ttest_rel(c, o, nan_policy='omit').pvalue)
        try:
            out['wilcoxon_p'] = float(stats.wilcoxon(c, o).pvalue)
        except ValueError:
            out['wilcoxon_p'] = np.nan
        sd = np.nanstd(diff, ddof=1)
        out['cohen_dz'] = float(np.nanmean(diff) / sd) if np.isfinite(sd) and sd > 0 else np.nan
    return pd.DataFrame([out])


def _plot_insertion_population_trace(summary, out_dir, *,
                                     smooth_ms=300, smooth_mode='centered',
                                     title=''):
    t = summary['time']
    cmat = _smooth_matrix_for_plot(summary['control_traces'], summary['bin_size'], smooth_ms, smooth_mode)
    omat = _smooth_matrix_for_plot(summary['opto_traces'], summary['bin_size'], smooth_ms, smooth_mode)
    cmu, omu = _nanmean(cmat, axis=0), _nanmean(omat, axis=0)
    cse, ose = _sem_rows(cmat), _sem_rows(omat)
    n = cmat.shape[0]
    fig, ax = plt.subplots(figsize=(6.8, 4.5))
    _shade_effect_window(ax, summary['window_s'])
    ax.plot(t, cmu, color='black', lw=2.8, label='control')
    ax.plot(t, omu, color=OPTO_COLOR, lw=2.8, label='opto')
    ax.fill_between(t, cmu - cse, cmu + cse, color='black', alpha=0.2, linewidth=0)
    ax.fill_between(t, omu - ose, omu + ose, color=OPTO_COLOR, alpha=0.2, linewidth=0)
    ax.axvline(0, linestyle='--', color='red', lw=1.0)
    ax.axhline(0, color='0.6', lw=0.7)
    ax.set_xlim(t[0], t[-1])
    ax.set_xlabel(f"Time from {summary['alignment']} (s)")
    ax.set_ylabel(_insertion_delta_ylabel(summary, 'Insertion-mean delta FR'))
    title = title or f"{summary['alignment']}: insertion-averaged delta FR"
    ax.set_title(f'{title}, N={n} insertions')
    ax.legend(frameon=False)
    fig.tight_layout()
    out_path = Path(out_dir) / 'population_insertion_delta_fr_trace.png'
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def _plot_insertion_bar(summary, stats_df, out_dir, *, title='',
                        figsize=(2.5, 4.5),
                        show_xticks=False,
                        show_title=False,
                        control_color='black',
                        opto_color=OPTO_COLOR,
                        bar_alpha=0.7,
                        bar_width=0.6,
                        pair_line_color='gray',
                        pair_line_alpha=0.5,
                        pair_linewidth=1.0,
                        pair_marker='o',
                        pair_markersize=4,
                        errorbar_linewidth=2.5,
                        errorbar_capsize=4,
                        xmargin=0.5,
                        ylabel_fontsize=9,
                        xtick_fontsize=8,
                        save_dpi=150,
                        save_tight_bbox=True):
    df = summary['insertion_df']
    c = df['control_delta_fr'].to_numpy(float)
    o = df['opto_delta_fr'].to_numpy(float)
    ok = np.isfinite(c) & np.isfinite(o)
    c, o = c[ok], o[ok]
    fig, ax = plt.subplots(figsize=figsize)
    means = [np.nanmean(c), np.nanmean(o)]
    sems = [
        np.nanstd(c, ddof=1) / np.sqrt(c.size) if c.size >= 2 else np.nan,
        np.nanstd(o, ddof=1) / np.sqrt(o.size) if o.size >= 2 else np.nan,
    ]
    colors = [control_color, opto_color]
    ax.bar([0, 1], means, color=colors, alpha=bar_alpha,
           width=bar_width, zorder=2)
    for x, mean, sem in zip([0, 1], means, sems):
        if np.isfinite(sem):
            ax.errorbar(
                x, mean, yerr=sem, color='black',
                linewidth=errorbar_linewidth, capsize=errorbar_capsize,
                zorder=3,
            )
    for ci, oi in zip(c, o):
        ax.plot([0, 1], [ci, oi], color=pair_line_color,
                alpha=pair_line_alpha, lw=pair_linewidth, zorder=1)
        if pair_marker:
            ax.plot([0, 1], [ci, oi], linestyle='none', marker=pair_marker,
                    markersize=pair_markersize, color='0.2',
                    alpha=0.75, zorder=4)
    ax.axhline(0, color='0.6', lw=0.7)
    ax.set_xlim(-float(xmargin), 1 + float(xmargin))
    if show_xticks:
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Control', 'Stim'], fontsize=xtick_fontsize)
    else:
        ax.set_xticks([])
        ax.tick_params(axis='x', length=0)
    ax.set_ylabel(
        _insertion_delta_ylabel(summary, 'Mean delta FR'),
        fontsize=ylabel_fontsize)
    stat = stats_df.iloc[0]
    window = summary['window_s']
    title = title or f"{summary['alignment']} ({window[0]:g} to {window[1]:g}s)"
    if show_title:
        ax.set_title(
            f"{title}\npaired t p={stat['paired_t_p']:.3g}, "
            f"Wilcoxon p={stat['wilcoxon_p']:.3g}"
        )
    fig.tight_layout()
    out_path = Path(out_dir) / 'insertion_delta_fr_control_vs_opto_bar.pdf'
    save_kwargs = {'dpi': save_dpi}
    if save_tight_bbox:
        save_kwargs['bbox_inches'] = 'tight'
    fig.savefig(out_path, **save_kwargs)
    plt.close(fig)
    return out_path


def run_insertion_delta_fr_analysis(data, idx, *,
                                    out_dir='~/python/saved_figures/BS_insertion_delta_fr',
                                    min_units_per_insertion=10,
                                    window_s=None,
                                    norm_mode=None,
                                    sign_mode=sign_mode_option,
                                    sign_window_s=DEFAULT_SIGN_WINDOW,
                                    orientation_mode=None,
                                    trial_estimator=None,
                                    whole_control_min_fr=QP_CONTROL_SCALAR_MIN_FR,
                                    min_control_delta_fr=0.0,
                                    baseline_subtract=False,
                                    baseline_window=(-0.5, 0.0),
                                    smooth_ms=300,
                                    smooth_mode='centered',
                                    save_insertion_plots=True,
                                    max_insertion_plots=None,
                                    bar_plot_options=None,
                                    title=''):
    """End-to-end insertion-level control-vs-opto delta-FR analysis.

    Saves per-insertion mean traces, a population trace, a paired bar plot, and
    CSV stats. This uses the current saved pickle; it does not require raw trials.
    ``norm_mode='qp_control_scalar'`` applies one control-only, block-balanced
    quiescence-period denominator per unit to both conditions. With
    ``min_control_delta_fr=0`` (the default), insertions whose control scalar is
    negative over ``window_s`` are removed before plotting/statistics; pass
    ``None`` to disable this threshold. Every exclusion is written to the
    removed-PID CSV with its reason. With
    sign_mode='block_crossfit' and baseline_subtract=True, the traces are
    control-preference aligned from independent training trials and the scalar
    comparison is the opto-vs-control difference in change from each condition's
    own baseline (a difference-in-differences presentation).
    Pass bar_plot_options=dict(...) to adjust the saved paired-bar figure
    without changing the scalar values or statistics.
    """
    summary = build_insertion_delta_fr_summary(
        data, idx,
        min_units_per_insertion=min_units_per_insertion,
        window_s=window_s,
        norm_mode=norm_mode,
        sign_mode=sign_mode,
        sign_window_s=sign_window_s,
        orientation_mode=orientation_mode,
        trial_estimator=trial_estimator,
        whole_control_min_fr=whole_control_min_fr,
        min_control_delta_fr=min_control_delta_fr,
        baseline_subtract=baseline_subtract,
        baseline_window=baseline_window,
    )
    if summary.get('analysis_api') == 'orthogonal_primary':
        analysis_slug = (
            f"{_alignment_slug(data)}_{summary['norm_mode']}_"
            f"{summary['trial_estimator']}_{summary['orientation_mode']}")
    else:
        analysis_slug = (
            f"{_alignment_slug(data)}_{summary['norm_mode']}_{summary['sign_mode']}")
    if summary['norm_mode'] == 'qp_control_scalar':
        qp_min = float(summary['qp_control_scalar_min_fr'])
        qp_token = f'{qp_min:g}'.replace('.', 'p')
        analysis_slug += f'_min{qp_token}hz'
    if summary['norm_mode'] == 'whole_control_scalar':
        whole_min = float(summary['whole_control_min_fr'])
        whole_token = f'{whole_min:g}'.replace('.', 'p')
        analysis_slug += f'_min{whole_token}hz'
    if summary['min_control_delta_fr'] is not None:
        ctrl_token = (f"{float(summary['min_control_delta_fr']):g}"
                      .replace('-', 'm').replace('.', 'p'))
        analysis_slug += f'_ctrlmin{ctrl_token}'
    if summary.get('baseline_subtract'):
        analysis_slug += '_baseline_change'
    out_dir = Path(out_dir).expanduser() / analysis_slug
    out_dir.mkdir(parents=True, exist_ok=True)
    insertion_plot_dir = out_dir / 'per_insertion_plots'
    if save_insertion_plots:
        insertion_plot_dir.mkdir(parents=True, exist_ok=True)

    insertion_df = summary['insertion_df']
    unit_counts_df = summary.get('insertion_unit_counts', pd.DataFrame())
    removed_pids_df = summary.get('removed_pids_df', pd.DataFrame())
    stats_df = _stats_from_insertion_df(insertion_df)
    insertion_df.to_csv(out_dir / 'insertion_delta_fr_scalar_values.csv', index=False)
    if len(unit_counts_df):
        unit_counts_df.to_csv(out_dir / 'insertion_delta_fr_selected_unit_counts.csv', index=False)
    removed_pids_path = out_dir / 'insertion_delta_fr_removed_pids.csv'
    removed_pids_df.to_csv(removed_pids_path, index=False)
    stats_df.to_csv(out_dir / 'insertion_delta_fr_stats.csv', index=False)

    saved = []
    if save_insertion_plots:
        n_plot = len(insertion_df) if max_insertion_plots is None else min(len(insertion_df), int(max_insertion_plots))
        for row_i in range(n_plot):
            saved.append(_plot_insertion_delta_trace(
                summary, row_i, insertion_plot_dir,
                smooth_ms=smooth_ms, smooth_mode=smooth_mode,
            ))
    saved.append(_plot_insertion_population_trace(
        summary, out_dir, smooth_ms=smooth_ms, smooth_mode=smooth_mode, title=title,
    ))
    saved.append(_plot_insertion_bar(
        summary, stats_df, out_dir, title=title,
        **(bar_plot_options or {}),
    ))

    print(f"Saved insertion-level delta-FR analysis to: {out_dir}")
    if len(unit_counts_df):
        n_selected_insertions = int(unit_counts_df['pid'].nunique())
        n_included_insertions = int(unit_counts_df['included_in_insertion_summary'].sum())
        print(
            f"Selected insertions: {n_selected_insertions}; "
            f"included after all insertion-level gates: "
            f"{n_included_insertions}"
        )
    if len(removed_pids_df):
        print(f'Removed insertions ({len(removed_pids_df)}):')
        print(removed_pids_df[
            ['pid', 'control_delta_fr', 'min_control_delta_fr',
             'exclusion_reason']
        ].to_string(index=False))
    else:
        print('Removed insertions: none')
    print(f'Removed-PID audit -> {removed_pids_path}')
    print(stats_df.to_string(index=False))
    summary['out_dir'] = out_dir
    summary['saved_paths'] = saved
    summary['removed_pids_path'] = removed_pids_path
    summary['stats_df'] = stats_df
    return summary


# ---------------------------------------------------------------------------
# Nested mouse -> session -> insertion -> unit analysis
# ---------------------------------------------------------------------------
def _stable_number_token(value):
    return f'{float(value):g}'.replace('-', 'm').replace('.', 'p')


def _session_hierarchy_for_units(data, units):
    """Return deterministic PID -> session ids without requiring a ONE query.

    A real ``eid``/``session_id`` is preferred when present. Current sufficient-
    statistics pickles predate that column, but bilateral insertions from one
    session carry identical session-wide trial/event arrays. Hashing those arrays
    therefore reconstructs their shared session membership without loading spikes.
    """
    units = units.copy()
    metadata = data.get('trial_metadata_by_pid', {}) or {}
    pid_rows = units[['pid', 'mouse']].drop_duplicates('pid')
    out = {}
    sources = {}

    for row in pid_rows.itertuples(index=False):
        pid = str(row.pid)
        mouse = str(row.mouse)
        pid_mask = units['pid'].astype(str) == pid
        explicit = None
        explicit_source = None
        for column in ('eid', 'session_id'):
            if column in units:
                values = units.loc[pid_mask, column].dropna().astype(str)
                values = values[~values.str.lower().isin({'', 'nan', 'none'})]
                if len(values):
                    explicit = values.iloc[0]
                    explicit_source = f'units.{column}'
                    break
        meta = metadata.get(pid, {})
        if explicit is None and isinstance(meta, dict):
            for key in ('eid', 'session_id'):
                value = meta.get(key)
                if value is not None and str(value).lower() not in {'', 'nan', 'none'}:
                    explicit = str(value)
                    explicit_source = f'trial_metadata_by_pid.{key}'
                    break
        if explicit is not None:
            out[pid] = str(explicit)
            sources[pid] = explicit_source
            continue

        if isinstance(meta, dict):
            digest = hashlib.sha256()
            digest.update(mouse.encode('utf-8'))
            n_arrays = 0
            for key in (
                    'probability_left', 'go_cue_times', 'feedback_times',
                    'interval_start_times', 'interval_end_times'):
                values = meta.get(key)
                if values is None:
                    continue
                arr = np.asarray(values)
                if arr.size == 0:
                    continue
                if np.issubdtype(arr.dtype, np.number):
                    arr = np.asarray(arr, dtype='<f8')
                    arr = np.nan_to_num(
                        arr, nan=9.87654321e307,
                        posinf=8.76543210e307, neginf=-8.76543210e307)
                else:
                    arr = np.asarray(arr, dtype='U').astype('S')
                digest.update(key.encode('utf-8'))
                digest.update(str(arr.shape).encode('ascii'))
                digest.update(np.ascontiguousarray(arr).tobytes())
                n_arrays += 1
            if n_arrays:
                out[pid] = f'{mouse}_fp_{digest.hexdigest()[:16]}'
                sources[pid] = 'trial_metadata_fingerprint'
                continue

        # Honest fallback: do not merge insertions when session identity cannot
        # be reconstructed. The model audit makes this fallback explicit.
        out[pid] = f'{mouse}_pid_{pid}'
        sources[pid] = 'pid_fallback'
    return out, sources


def build_multilevel_delta_fr_data(
        data, idx, *, window_s=None, norm_mode=None,
        sign_mode=sign_mode_option, sign_window_s=DEFAULT_SIGN_WINDOW,
        min_units_per_insertion=5, min_valid_unit_fraction=None,
        min_reference_sign_agreement=None, min_control_delta_fr=None,
        max_abs_unit_effect=None):
    """Build auditable unit/insertion/session tables for a nested mixed model.

    The response is each unit's paired, cross-fit-oriented mean
    ``control - opto`` delta-FR over ``window_s``. Default exclusions are limited
    to non-finite unit estimates and insertions with fewer than five valid units.
    Outcome-dependent gates are available but deliberately disabled by default.
    """
    idx = np.asarray(idx, dtype=int)
    if idx.size == 0:
        raise ValueError('No units selected; cannot build a multilevel analysis.')
    min_units_per_insertion = int(min_units_per_insertion)
    if min_units_per_insertion < 1:
        raise ValueError('min_units_per_insertion must be >= 1.')
    for name, value in (
            ('min_valid_unit_fraction', min_valid_unit_fraction),
            ('min_reference_sign_agreement', min_reference_sign_agreement)):
        if value is not None and not (0 <= float(value) <= 1):
            raise ValueError(f'{name} must be within [0, 1] or None.')
    if min_control_delta_fr is not None and not np.isfinite(float(min_control_delta_fr)):
        raise ValueError('min_control_delta_fr must be finite or None.')
    if max_abs_unit_effect is not None:
        max_abs_unit_effect = float(max_abs_unit_effect)
        if not np.isfinite(max_abs_unit_effect) or max_abs_unit_effect <= 0:
            raise ValueError('max_abs_unit_effect must be finite and > 0, or None.')

    window_s = _default_effect_window(data, window_s)
    effect_mask = _window(data, window_s)
    if not np.any(effect_mask):
        raise ValueError(f'window_s={window_s} is empty for this peth_time range.')
    control, opto, active_mode, resolved_sign_mode = _oriented_delta_traces(
        data, idx, mode=norm_mode, sign_mode=sign_mode,
        sign_window=sign_window_s, scale=100.0)
    control_scalar = _nanmean(control[:, effect_mask], axis=1)
    opto_scalar = _nanmean(opto[:, effect_mask], axis=1)
    unit_effect = control_scalar - opto_scalar

    selected = data['units'].iloc[idx].copy().reset_index(drop=True)
    selected['source_row'] = idx
    selected['pid'] = selected['pid'].astype(str)
    if 'mouse' not in selected:
        selected['mouse'] = 'unknown'
    selected['mouse'] = selected['mouse'].astype(str)
    session_map, session_sources = _session_hierarchy_for_units(data, selected)
    selected['session_id'] = selected['pid'].map(session_map)
    selected['session_id_source'] = selected['pid'].map(session_sources)
    selected['control_delta_fr'] = control_scalar
    selected['opto_delta_fr'] = opto_scalar
    selected['unit_effect_control_minus_opto'] = unit_effect

    reference_agreement = np.full(len(selected), np.nan, dtype=float)
    reference_strength = np.full(len(selected), np.nan, dtype=float)
    if (resolved_sign_mode == 'block_crossfit'
            and block_crossfit_traces_available(data, active_mode)):
        sign_mask = _sign_mask(data, sign_window_s)
        ref_a = _stack_key(
            data, f'trace_block_crossfit_reference_a_{active_mode}', idx)
        ref_b = _stack_key(
            data, f'trace_block_crossfit_reference_b_{active_mode}', idx)
        sign_a = _sign_from_reference(ref_a, sign_mask)[:, 0]
        sign_b = _sign_from_reference(ref_b, sign_mask)[:, 0]
        valid_pair = np.isfinite(sign_a) & np.isfinite(sign_b)
        reference_agreement[valid_pair] = (
            sign_a[valid_pair] == sign_b[valid_pair]).astype(float)
        ref_a_mean = _nanmean(ref_a[:, sign_mask], axis=1)
        ref_b_mean = _nanmean(ref_b[:, sign_mask], axis=1)
        reference_strength = 50.0 * (np.abs(ref_a_mean) + np.abs(ref_b_mean))
    selected['reference_sign_agreement'] = reference_agreement
    selected['reference_abs_delta_pct_qp'] = reference_strength

    finite = (
        np.isfinite(control_scalar) & np.isfinite(opto_scalar)
        & np.isfinite(unit_effect))
    selected['unit_valid_before_insertion_gate'] = finite
    selected['included_in_model'] = False
    selected['exclusion_reason'] = np.where(finite, '', 'nonfinite_unit_effect')
    if max_abs_unit_effect is not None:
        outlier = finite & (np.abs(unit_effect) > max_abs_unit_effect)
        selected.loc[outlier, 'unit_valid_before_insertion_gate'] = False
        selected.loc[outlier, 'exclusion_reason'] = 'unit_effect_above_abs_limit'

    audit_rows = []
    for pid, grp in selected.groupby('pid', sort=True):
        valid_grp = grp[grp['unit_valid_before_insertion_gate']]
        n_selected = int(len(grp))
        n_valid = int(len(valid_grp))
        valid_fraction = n_valid / n_selected if n_selected else np.nan
        c_mean = float(_nanmean(valid_grp['control_delta_fr'])) if n_valid else np.nan
        o_mean = float(_nanmean(valid_grp['opto_delta_fr'])) if n_valid else np.nan
        effect_mean = float(_nanmean(
            valid_grp['unit_effect_control_minus_opto'])) if n_valid else np.nan
        effect_sd = float(np.nanstd(
            valid_grp['unit_effect_control_minus_opto'], ddof=1)) if n_valid >= 2 else np.nan
        effect_sem = effect_sd / np.sqrt(n_valid) if n_valid >= 2 else np.nan
        ref_values = valid_grp['reference_sign_agreement'].to_numpy(float)
        n_ref_pairs = int(np.isfinite(ref_values).sum())
        ref_agreement = (
            float(np.nanmean(ref_values)) if n_ref_pairs else np.nan)

        reasons = []
        if n_valid < min_units_per_insertion:
            reasons.append('below_min_units')
        if (min_valid_unit_fraction is not None
                and (not np.isfinite(valid_fraction)
                     or valid_fraction < float(min_valid_unit_fraction))):
            reasons.append('below_min_valid_unit_fraction')
        if min_reference_sign_agreement is not None:
            if (not np.isfinite(ref_agreement)
                    or ref_agreement < float(min_reference_sign_agreement)):
                reasons.append('below_min_reference_sign_agreement')
        if min_control_delta_fr is not None:
            if (not np.isfinite(c_mean)
                    or c_mean < float(min_control_delta_fr)):
                reasons.append('control_delta_below_threshold')
        included = len(reasons) == 0
        warning_flags = []
        if np.isfinite(c_mean) and c_mean < 0:
            warning_flags.append('negative_control_mean_retained')
        if np.isfinite(ref_agreement) and ref_agreement < 0.5:
            warning_flags.append('reference_sign_agreement_below_chance')
        first = grp.iloc[0]
        audit_rows.append({
            'pid': pid,
            'mouse': first['mouse'],
            'session_id': first['session_id'],
            'session_id_source': first['session_id_source'],
            'n_selected_units': n_selected,
            'n_valid_units': n_valid,
            'valid_unit_fraction': valid_fraction,
            'control_delta_fr': c_mean,
            'opto_delta_fr': o_mean,
            'effect_control_minus_opto': effect_mean,
            'effect_unit_sd': effect_sd,
            'effect_unit_sem': effect_sem,
            'n_reference_sign_pairs': n_ref_pairs,
            'reference_sign_agreement': ref_agreement,
            'included_in_model': included,
            'exclusion_reason': ';'.join(reasons),
            'warning_flags': ';'.join(warning_flags),
        })
        if included:
            selected.loc[
                (selected['pid'] == pid)
                & selected['unit_valid_before_insertion_gate'],
                'included_in_model'] = True
        else:
            still_blank = (selected['pid'] == pid) & (selected['exclusion_reason'] == '')
            selected.loc[still_blank, 'exclusion_reason'] = (
                'insertion_gate:' + ';'.join(reasons))

    insertion_audit = pd.DataFrame(audit_rows)
    model_units = selected[selected['included_in_model']].copy()
    if model_units.empty:
        raise ValueError(
            'No units remain after multilevel insertion gates. Inspect the '
            'requested criteria or lower min_units_per_insertion.')
    included_insertions = insertion_audit[
        insertion_audit['included_in_model']].copy().reset_index(drop=True)
    session_summary = (
        model_units.groupby(['mouse', 'session_id'], sort=True)
        .agg(
            n_insertions=('pid', 'nunique'),
            n_units=('source_row', 'size'),
            control_delta_fr=('control_delta_fr', 'mean'),
            opto_delta_fr=('opto_delta_fr', 'mean'),
            effect_control_minus_opto=(
                'unit_effect_control_minus_opto', 'mean'),
        ).reset_index()
    )
    return {
        'unit_df': selected,
        'model_unit_df': model_units,
        'insertion_audit': insertion_audit,
        'insertion_df': included_insertions,
        'session_df': session_summary,
        'window_s': tuple(window_s),
        'norm_mode': active_mode,
        'sign_mode': resolved_sign_mode,
        'sign_window_s': (
            tuple(sign_window_s) if sign_window_s is not None else None),
        'min_units_per_insertion': min_units_per_insertion,
        'min_valid_unit_fraction': min_valid_unit_fraction,
        'min_reference_sign_agreement': min_reference_sign_agreement,
        'min_control_delta_fr': min_control_delta_fr,
        'max_abs_unit_effect': max_abs_unit_effect,
    }


def _fit_nested_unit_effect_model(model_units, *, reml=True):
    """Fit control-opto unit effects with nested random intercepts."""
    try:
        import statsmodels.formula.api as smf
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise ImportError(
            'run_multilevel_delta_fr_analysis requires statsmodels.') from exc

    df = model_units.copy()
    df['mouse'] = df['mouse'].astype(str)
    df['session_id'] = df['session_id'].astype(str)
    df['pid'] = df['pid'].astype(str)
    df['unit_effect'] = pd.to_numeric(
        df['unit_effect_control_minus_opto'], errors='coerce')
    df = df[np.isfinite(df['unit_effect'])].copy()
    n_mice = int(df['mouse'].nunique())
    n_sessions = int(df['session_id'].nunique())
    n_insertions = int(df['pid'].nunique())
    if n_mice < 2:
        raise ValueError('The mixed model needs at least two mice.')
    if n_insertions < 2:
        raise ValueError('The mixed model needs at least two insertions.')

    insertions_per_session = df.groupby('session_id')['pid'].nunique()
    n_multi_insertion_sessions = int((insertions_per_session > 1).sum())
    session_component_identifiable = (
        n_sessions < n_insertions and n_multi_insertion_sessions > 0)
    vc_formula = {'insertion': '0 + C(pid)'}
    hierarchy_note = (
        'mouse + session-within-mouse + insertion-within-session + unit'
        if session_component_identifiable else
        'mouse + insertion + unit; session variance omitted because session '
        'and insertion are one-to-one in the included data')
    if session_component_identifiable:
        vc_formula['session'] = '0 + C(session_id)'

    model = smf.mixedlm(
        'unit_effect ~ 1', data=df, groups=df['mouse'],
        re_formula='1', vc_formula=vc_formula)
    fit_warnings = []
    errors = []
    result = None
    optimizer = None
    for candidate in ('lbfgs', 'powell', 'cg'):
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                candidate_result = model.fit(
                    reml=bool(reml), method=candidate,
                    maxiter=2000, disp=False)
            fit_warnings.extend(
                f'{type(item.message).__name__}: {item.message}'
                for item in caught)
            if result is None or bool(candidate_result.converged):
                result = candidate_result
                optimizer = candidate
            if bool(candidate_result.converged):
                break
        except Exception as exc:
            errors.append(f'{candidate}: {type(exc).__name__}: {exc}')
    if result is None:
        raise RuntimeError(
            'All MixedLM optimizers failed: ' + ' | '.join(errors))

    intercept = float(result.fe_params['Intercept'])
    se = float(result.bse_fe['Intercept'])
    pvalue = float(result.pvalues['Intercept'])
    ci = result.conf_int().loc['Intercept'].to_numpy(float)
    fixed_effects = pd.DataFrame([{
        'term': 'population_control_minus_opto',
        'estimate': intercept,
        'std_error': se,
        'ci95_low': float(ci[0]),
        'ci95_high': float(ci[1]),
        'wald_z': intercept / se if np.isfinite(se) and se > 0 else np.nan,
        'wald_p': pvalue,
    }])

    variance_rows = []
    mouse_var = (
        float(result.cov_re.iloc[0, 0])
        if result.cov_re.size else np.nan)
    variance_rows.append({'level': 'mouse', 'variance': mouse_var})
    for name, value in zip(model.exog_vc.names, np.asarray(result.vcomp, float)):
        variance_rows.append({'level': str(name), 'variance': float(value)})
    variance_rows.append({'level': 'unit_residual', 'variance': float(result.scale)})
    variance_components = pd.DataFrame(variance_rows)
    total_variance = float(np.nansum(variance_components['variance']))
    variance_components['fraction_of_total'] = (
        variance_components['variance'] / total_variance
        if total_variance > 0 else np.nan)

    diagnostics = pd.DataFrame([{
        'converged': bool(result.converged),
        'optimizer': optimizer,
        'reml': bool(reml),
        'n_units': int(len(df)),
        'n_insertions': n_insertions,
        'n_sessions': n_sessions,
        'n_mice': n_mice,
        'n_multi_insertion_sessions': n_multi_insertion_sessions,
        'session_component_identifiable': session_component_identifiable,
        'hierarchy_fitted': hierarchy_note,
        'log_likelihood': float(result.llf),
        'scale': float(result.scale),
        'fit_warnings': ' | '.join(dict.fromkeys(fit_warnings)),
        'optimizer_errors': ' | '.join(errors),
        'asymptotic_inference_warning': (
            'Wald CI/p are model-based and asymptotic; few top-level mice '
            'limit population-level generalization.'),
    }])
    return {
        'model': model,
        'result': result,
        'model_df': df,
        'fixed_effects': fixed_effects,
        'variance_components': variance_components,
        'diagnostics': diagnostics,
    }


def _multilevel_point_sizes(n_units, low=28.0, high=115.0):
    n_units = np.asarray(n_units, dtype=float)
    root = np.sqrt(np.maximum(n_units, 1))
    if root.size == 0 or not np.isfinite(root).any():
        return np.asarray([], dtype=float)
    lo, hi = np.nanmin(root), np.nanmax(root)
    if hi <= lo:
        return np.full(root.shape, 0.5 * (low + high))
    return low + (high - low) * (root - lo) / (hi - lo)


def _plot_multilevel_effect_estimation(analysis, model_fit, out_dir, *,
                                       title='', save_dpi=220):
    """Paired insertion view plus an effect forest with the model estimate."""
    from matplotlib.lines import Line2D

    insertion_df = analysis['insertion_df'].copy()
    insertion_df = insertion_df.sort_values(
        ['mouse', 'session_id', 'effect_control_minus_opto'],
        ascending=[True, True, False]).reset_index(drop=True)
    fixed = model_fit['fixed_effects'].iloc[0]
    mice = insertion_df['mouse'].astype(str).unique().tolist()
    cmap = plt.get_cmap('tab10')
    mouse_colors = {mouse: cmap(i % 10) for i, mouse in enumerate(mice)}
    sizes = _multilevel_point_sizes(insertion_df['n_valid_units'])
    n_insertions = len(insertion_df)
    fig_height = max(5.2, 0.27 * n_insertions + 1.8)
    fig, (ax_pair, ax_effect) = plt.subplots(
        1, 2, figsize=(10.2, fig_height),
        gridspec_kw={'width_ratios': [0.9, 1.5]})

    for row_i, row in insertion_df.iterrows():
        color = mouse_colors[str(row['mouse'])]
        size = sizes[row_i]
        ax_pair.plot(
            [0, 1], [row['control_delta_fr'], row['opto_delta_fr']],
            color=color, alpha=0.55, lw=1.2, zorder=1)
        ax_pair.scatter(0, row['control_delta_fr'], s=size,
                        color='black', edgecolor=color, linewidth=1.1, zorder=3)
        ax_pair.scatter(1, row['opto_delta_fr'], s=size,
                        color=OPTO_COLOR, edgecolor=color, linewidth=1.1, zorder=3)
    ax_pair.axhline(0, color='0.7', lw=0.8)
    ax_pair.set_xlim(-0.4, 1.4)
    ax_pair.set_xticks([0, 1])
    ax_pair.set_xticklabels(['Control', 'Opto'])
    ax_pair.set_ylabel(_insertion_delta_ylabel(
        {'norm_mode': analysis['norm_mode']}, 'Insertion mean delta FR'))
    ax_pair.set_title('Each line is one insertion')

    y = np.arange(n_insertions)
    for row_i, row in insertion_df.iterrows():
        color = mouse_colors[str(row['mouse'])]
        sem = float(row['effect_unit_sem'])
        xerr = sem if np.isfinite(sem) else None
        ax_effect.errorbar(
            row['effect_control_minus_opto'], y[row_i], xerr=xerr,
            fmt='none', ecolor=color, elinewidth=1.0, alpha=0.45,
            capsize=2, zorder=1)
        ax_effect.scatter(
            row['effect_control_minus_opto'], y[row_i],
            s=sizes[row_i], color=color, edgecolor='black', linewidth=0.5,
            alpha=0.9, zorder=3)
    ax_effect.axvline(0, color='0.45', lw=0.9)
    ax_effect.axvspan(
        fixed['ci95_low'], fixed['ci95_high'],
        color='seagreen', alpha=0.12, zorder=0)
    ax_effect.axvline(
        fixed['estimate'], color='seagreen', lw=2.3,
        label='multilevel estimate')
    ax_effect.set_yticks(y)
    ax_effect.set_yticklabels([
        f"{str(row.pid)[:8]}  (n={int(row.n_valid_units)})"
        for row in insertion_df.itertuples(index=False)
    ], fontsize=8)
    ax_effect.invert_yaxis()
    baseline_label = (
        '% control QP baseline'
        if analysis['norm_mode'] == 'qp_control_scalar' else '% baseline')
    ax_effect.set_xlabel(f'Control − opto delta FR ({baseline_label})')
    ax_effect.set_title('Insertion effects; bars are within-insertion SEM')

    condition_handles = [
        Line2D([0], [0], marker='o', color='none', markerfacecolor='black',
               markeredgecolor='black', label='Control'),
        Line2D([0], [0], marker='o', color='none', markerfacecolor=OPTO_COLOR,
               markeredgecolor=OPTO_COLOR, label='Opto'),
    ]
    ax_pair.legend(handles=condition_handles, frameon=False, loc='best')
    mouse_handles = [
        Line2D([0], [0], marker='o', color='none', markerfacecolor=color,
               markeredgecolor='black', label=mouse)
        for mouse, color in mouse_colors.items()
    ]
    model_handle = Line2D([0], [0], color='seagreen', lw=2.3,
                          label='Multilevel estimate (95% CI band)')
    ax_effect.legend(
        handles=[model_handle] + mouse_handles, frameon=False,
        fontsize=8, loc='best')

    diagnostics = model_fit['diagnostics'].iloc[0]
    window = analysis['window_s']
    title = title or (
        f"{analysis.get('alignment', 'Alignment')}: nested BS-unit effect")
    fig.suptitle(title, y=0.995, fontsize=13)
    fig.text(
        0.5, 0.008,
        f"Mixed model control − opto = {fixed['estimate']:.2f} "
        f"[{fixed['ci95_low']:.2f}, {fixed['ci95_high']:.2f}]  "
        f"Wald p={fixed['wald_p']:.3g}; window {window[0]:g} to {window[1]:g}s; "
        f"{int(diagnostics['n_units'])} units, "
        f"{int(diagnostics['n_insertions'])} insertions, "
        f"{int(diagnostics['n_sessions'])} sessions, "
        f"{int(diagnostics['n_mice'])} mice. Point area scales with sqrt(n units).",
        ha='center', va='bottom', fontsize=8.5)
    fig.tight_layout(rect=(0, 0.045, 1, 0.975))
    out_dir = Path(out_dir)
    png_path = out_dir / 'multilevel_effect_estimation.png'
    pdf_path = out_dir / 'multilevel_effect_estimation.pdf'
    fig.savefig(png_path, dpi=save_dpi, bbox_inches='tight')
    fig.savefig(pdf_path, bbox_inches='tight')
    plt.close(fig)
    return [png_path, pdf_path]


def run_multilevel_delta_fr_analysis(
        data, idx, *, out_dir='~/python/saved_figures/BS_multilevel_delta_fr',
        window_s=None, norm_mode=None, sign_mode=sign_mode_option,
        sign_window_s=DEFAULT_SIGN_WINDOW, min_units_per_insertion=5,
        min_valid_unit_fraction=None, min_reference_sign_agreement=None,
        min_control_delta_fr=None, max_abs_unit_effect=None,
        reml=True, title=''):
    """Run and save the additional nested unit-effect analysis.

    The model response is each unit's paired control-minus-opto scalar. Random
    intercepts are fitted for mouse and insertion, plus session when session and
    insertion are separately identifiable. Negative control insertions are
    retained and flagged by default; outcome-dependent exclusion gates are opt-in.
    """
    analysis = build_multilevel_delta_fr_data(
        data, idx, window_s=window_s, norm_mode=norm_mode,
        sign_mode=sign_mode, sign_window_s=sign_window_s,
        min_units_per_insertion=min_units_per_insertion,
        min_valid_unit_fraction=min_valid_unit_fraction,
        min_reference_sign_agreement=min_reference_sign_agreement,
        min_control_delta_fr=min_control_delta_fr,
        max_abs_unit_effect=max_abs_unit_effect)
    analysis['alignment'] = data.get('onset_alignment', 'Alignment')
    model_fit = _fit_nested_unit_effect_model(
        analysis['model_unit_df'], reml=reml)

    window = analysis['window_s']
    slug = (
        f"{_alignment_slug(data)}_{analysis['norm_mode']}_{analysis['sign_mode']}"
        f"_multilevel_w{_stable_number_token(window[0])}to"
        f"{_stable_number_token(window[1])}_min{min_units_per_insertion}u")
    if min_control_delta_fr is not None:
        slug += f'_ctrlmin{_stable_number_token(min_control_delta_fr)}'
    out_dir = Path(out_dir).expanduser() / slug
    out_dir.mkdir(parents=True, exist_ok=True)

    analysis['unit_df'].to_csv(
        out_dir / 'multilevel_unit_effects_and_inclusion.csv', index=False)
    analysis['insertion_audit'].to_csv(
        out_dir / 'multilevel_insertion_exclusion_audit.csv', index=False)
    analysis['insertion_df'].to_csv(
        out_dir / 'multilevel_insertion_effects.csv', index=False)
    analysis['session_df'].to_csv(
        out_dir / 'multilevel_session_effects.csv', index=False)
    model_fit['fixed_effects'].to_csv(
        out_dir / 'multilevel_fixed_effect.csv', index=False)
    model_fit['variance_components'].to_csv(
        out_dir / 'multilevel_variance_components.csv', index=False)
    model_fit['diagnostics'].to_csv(
        out_dir / 'multilevel_model_diagnostics.csv', index=False)
    (out_dir / 'multilevel_model_summary.txt').write_text(
        model_fit['result'].summary().as_text() + '\n\n' +
        model_fit['diagnostics'].to_string(index=False) + '\n')
    config = pd.Series({
        'window_start': window[0], 'window_end': window[1],
        'norm_mode': analysis['norm_mode'], 'sign_mode': analysis['sign_mode'],
        'sign_window': analysis['sign_window_s'],
        'min_units_per_insertion': min_units_per_insertion,
        'min_valid_unit_fraction': min_valid_unit_fraction,
        'min_reference_sign_agreement': min_reference_sign_agreement,
        'min_control_delta_fr': min_control_delta_fr,
        'max_abs_unit_effect': max_abs_unit_effect,
        'reml': bool(reml),
    }, name='value')
    config.to_csv(out_dir / 'multilevel_analysis_config.csv', header=True)
    saved_paths = _plot_multilevel_effect_estimation(
        analysis, model_fit, out_dir, title=title)

    fixed = model_fit['fixed_effects'].iloc[0]
    diagnostics = model_fit['diagnostics'].iloc[0]
    excluded = analysis['insertion_audit'][
        ~analysis['insertion_audit']['included_in_model']]
    warned = analysis['insertion_audit'][
        analysis['insertion_audit']['warning_flags'].astype(str) != '']
    print(f'Saved nested multilevel analysis to: {out_dir}')
    print(
        f"Control - opto = {fixed['estimate']:.3f} "
        f"(95% Wald CI {fixed['ci95_low']:.3f} to {fixed['ci95_high']:.3f}; "
        f"p={fixed['wald_p']:.3g})")
    print(
        f"Included: {int(diagnostics['n_units'])} units, "
        f"{int(diagnostics['n_insertions'])} insertions, "
        f"{int(diagnostics['n_sessions'])} sessions, "
        f"{int(diagnostics['n_mice'])} mice")
    print(f"Hierarchy fitted: {diagnostics['hierarchy_fitted']}")
    if len(excluded):
        print('Excluded insertions:')
        print(excluded[['pid', 'n_valid_units', 'exclusion_reason']].to_string(index=False))
    else:
        print('Excluded insertions: none')
    if len(warned):
        print('Retained insertion warnings:')
        print(warned[['pid', 'control_delta_fr', 'reference_sign_agreement',
                      'warning_flags']].to_string(index=False))

    analysis.update(model_fit)
    analysis['out_dir'] = out_dir
    analysis['saved_paths'] = saved_paths
    return analysis


def plot_zscore(data, idx, title='', ax=None, sign_flip=False, smooth=True,
                smooth_ms=None, smooth_mode='causal',
                baseline_subtract=False, baseline_window=(-0.5, 0.0),
                time_range=None, axis_size=(6.5, 4.5)):
    """Mean z-scored delta (stim vs control) +/- SEM.

    This trace remains unsigned by default. An unbiased split-half z-score cannot
    be reconstructed from the saved full-sample z-score because fold-specific
    error terms were not saved. ``sign_flip=True`` is therefore rejected rather
    than silently restoring the biased same-data orientation.
    smooth_ms overrides the window (ms);
    smooth_mode='causal' uses only current/past bins; 'centered' is symmetric
    and non-causal.
    baseline_subtract references each unit to its own pre-laser mean (over
    baseline_window), so the trace shows the laser-induced CHANGE and the
    pre-laser level is 0 by construction.
    time_range defaults to (-2, 5) for LaserOnset and (-5, 2) for GoCueOnset.
    axis_size is the figsize for this single axis when ax is None."""
    t = np.asarray(data['peth_time'])
    if sign_flip:
        raise ValueError(
            'plot_zscore(sign_flip=True) is disabled: the saved z-score has no '
            'fold-specific control error terms for unbiased split-half orientation. '
            'Use the default unsigned z-score or the cross-validated delta-FR plots.'
        )
    if len(idx) == 0:
        print('No units selected; nothing to plot.')
        return None
    bin_size = float(data.get('bin_size', 0.05))
    win_ms = smooth_ms if smooth_ms is not None else data.get('smoothing_window_ms', 300)
    smooth_bins = max(1, int(round(win_ms / 1000.0 / bin_size))) if smooth else 1
    pre = ((t >= baseline_window[0]) & (t < baseline_window[1])) if baseline_subtract else None
    z = _stack_traces(data, idx, 'trace_zscore', smooth_bins,
                      smooth_mode=smooth_mode, baseline_pre_mask=pre)
    n = z.shape[0]
    m = np.nanmean(z, axis=0)
    e = np.nanstd(z, axis=0) / np.sqrt(n)
    align = data.get('onset_alignment', 'Laser onset')
    time_range = _default_plot_time_range(data, time_range)
    time_range = _resolve_plot_time_range(t, time_range)
    show = _time_mask_for_plot(t, time_range)
    t_plot = t[show]
    if ax is None:
        _, ax = plt.subplots(figsize=axis_size)
    ax.plot(t_plot, m[show], color=OPTO_COLOR, linewidth=3)
    ax.fill_between(t_plot, m[show] - e[show], m[show] + e[show], color='k', alpha=0.2)
    ax.axvline(0, linestyle='--', color='red')
    ax.axhline(0, color='gray', linestyle='--', linewidth=1)
    ax.set_xlabel(f'Time from {align} (s)')
    ax.set_ylabel('Delta FR 80/20 blocks (Z-scored)')
    ax.set_xlim(time_range)
    ax.set_title(f'{title}, n = {n}')
    return ax


def plot_stim_psth(data, idx, ax=None, smooth_ms=None, normalize='raw',
                   baseline_window=(-1.0, 0.0), title='', trace_key='trace_stim_all',
                   color=OPTO_COLOR, min_baseline_fr=1.0):
    """Mean PETH across selected units -- a ground-truth laser-alignment check.
    This is the raw firing rate (not a block delta), so if the data are aligned
    to laser onset, an inhibition session shows a laser-locked feature at t=0.

    Requires `trace_key` in the pickle (default 'trace_stim_all'; pass
    'trace_nonstim_all' to plot the control PETH).

    normalize:
        'raw'      -> plot firing rate in Hz (default; robust, no division).
        'whole'    -> divide each unit by its mean over the whole window (robust;
                      makes units of different rates comparable without a fragile
                      pre-laser denominator).
        'baseline' -> divide by the pre-laser mean. NOTE: firing collapses to ~0
                      just before onset in this dataset, so this denominator is
                      tiny and unstable; units with pre-laser FR < min_baseline_fr
                      are dropped (set NaN) rather than allowed to explode.
    """
    if trace_key not in data:
        print(f"No '{trace_key}' in this pickle -- re-run the BS pipeline to save it.")
        return None
    t = np.asarray(data['peth_time'], dtype=float)
    sa = data[trace_key]
    idx = np.asarray(idx, dtype=int)
    bin_size = float(data.get('bin_size', 0.05))
    win_ms = smooth_ms if smooth_ms is not None else 100
    smooth_bins = max(1, int(round(win_ms / 1000.0 / bin_size)))
    pre = (t >= baseline_window[0]) & (t < baseline_window[1])
    rows = []
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        for i in idx:
            tr = np.asarray(sa[i], dtype=float)
            if normalize == 'baseline':
                base = np.nanmean(tr[pre])
                tr = tr / base if (np.isfinite(base) and base >= min_baseline_fr) else tr * np.nan
            elif normalize == 'whole':
                base = np.nanmean(tr)
                tr = tr / base if (np.isfinite(base) and base > 0) else tr * np.nan
            # 'raw' -> leave as Hz
            rows.append(_smooth(tr, smooth_bins, 'centered'))
    if not rows:
        print('No units selected; nothing to plot.')
        return None
    M = np.vstack(rows)
    M[~np.isfinite(M)] = np.nan          # drop any residual inf/huge from the mean
    n = int(np.isfinite(M).any(axis=1).sum())
    m = np.nanmean(M, axis=0)
    e = np.nanstd(M, axis=0) / np.sqrt(max(n, 1))
    align = data.get('onset_alignment', 'Laser onset')
    if ax is None:
        _, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(t, m, color=color, linewidth=2)
    ax.fill_between(t, m - e, m + e, color=color, alpha=0.2)
    ax.axvline(0, linestyle='--', color='red')
    ax.set_xlabel(f'Time from {align} (s)')
    _ylab = {'raw': 'FR (Hz)', 'whole': 'FR (/whole-window mean)',
             'baseline': 'FR (/pre-laser baseline)'}.get(normalize, 'FR')
    ax.set_ylabel(('stim' if 'stim_all' in trace_key else 'control') + '-trial ' + _ylab)
    ax.set_xlim(t[0], t[-1])
    ax.set_title(f'{title or "Mean PETH"}, n = {n}')
    return ax


def _resolve_unit_index(data, unit_index=None, *, pid=None, clustnum=None):
    """Resolve a unit by row index or by stable (pid, clustnum) identity."""
    df = data['units'].reset_index(drop=True)
    if unit_index is not None:
        unit_index = int(unit_index)
        if unit_index < 0 or unit_index >= len(df):
            raise IndexError(f'unit_index {unit_index} outside 0..{len(df)-1}')
        return unit_index
    if pid is None or clustnum is None:
        raise ValueError('Provide unit_index, or both pid and clustnum.')
    pid_text = str(pid)
    pid_values = df['pid'].astype(str)
    pid_match = pid_values == pid_text
    if not pid_match.any():
        prefix_match = pid_values.str.startswith(pid_text)
        prefix_pids = pid_values[prefix_match].unique()
        if len(prefix_pids) == 1:
            pid_match = prefix_match
        elif len(prefix_pids) > 1:
            raise KeyError(
                f'PID prefix {pid_text!r} is ambiguous; matches '
                f'{list(prefix_pids[:5])}')
    m = pid_match & (df['clustnum'].astype(int) == int(clustnum))
    hits = np.where(m.to_numpy())[0]
    if hits.size == 0:
        raise KeyError(f'No unit found for pid={pid}, clustnum={clustnum}')
    return int(hits[0])


def _unit_identity(data, unit_index):
    row = data['units'].reset_index(drop=True).iloc[int(unit_index)]
    return {
        'row_index': int(unit_index),
        'pid': str(row.get('pid', 'unknown')),
        'clustnum': int(row.get('clustnum', -1)) if pd.notna(row.get('clustnum', np.nan)) else -1,
        'mouse': row.get('mouse', 'unknown'),
        'region': row.get('Allenregion', row.get('Berylregion', 'unknown')),
        'bs_score': row.get('BS_score', np.nan),
        'pval_empirical': row.get('pval_empirical', row.get('pval_real', np.nan)),
    }


def _unit_title(data, unit_index, title_prefix=''):
    ident = _unit_identity(data, unit_index)
    bits = [
        f"PID {ident['pid'][:8]}",
        f"cluster {ident['clustnum']}",
        f"row {ident['row_index']}",
        str(ident['region']),
    ]
    if np.isfinite(ident['bs_score']):
        bits.append(f"BS={int(ident['bs_score'])}")
    if np.isfinite(ident['pval_empirical']):
        bits.append(f"p={ident['pval_empirical']:.3g}")
    prefix = f'{title_prefix}: ' if title_prefix else ''
    return prefix + ', '.join(bits)


def _trace_for_unit(data, key, unit_index):
    if key not in data:
        return None
    return np.asarray(data[key][int(unit_index)], dtype=float)


def _unit_peth_style(colors=None, linestyles=None):
    """Return validated, independently mutable example-unit plot styles."""
    palette = dict(UNIT_BLOCK_PETH_COLORS)
    if colors is not None:
        unknown = set(colors) - set(palette)
        if unknown:
            raise ValueError(
                f'Unknown unit-PETH color keys: {sorted(unknown)}; '
                f'use any of {sorted(palette)}')
        palette.update(colors)

    styles = dict(UNIT_BLOCK_PETH_LINESTYLES)
    if linestyles is not None:
        unknown = set(linestyles) - set(styles)
        if unknown:
            raise ValueError(
                f'Unknown unit-PETH linestyle keys: {sorted(unknown)}; '
                f'use any of {sorted(styles)}')
        styles.update(linestyles)
    return palette, styles


_UNIT_BLOCK_PETH_REQUIRED = ('trace_nonstim_80_raw', 'trace_nonstim_20_raw')


def _missing_unit_block_peth_keys(data):
    return [key for key in _UNIT_BLOCK_PETH_REQUIRED if key not in data]


def _require_unit_block_peth_data(label, data):
    missing = _missing_unit_block_peth_keys(data)
    if missing:
        available = sorted(key for key in data if key.startswith('trace_') and key.endswith('_raw'))
        raise KeyError(
            f"{label} pickle is missing {missing}. These block-specific firing-rate "
            "traces are not recoverable from the older postprocessed pickle. Re-run "
            "SNr_inhibition_BS_downstream_effect.py with save_raw_block_peths=1. "
            f"Available raw trace keys: {available}"
        )


def _smooth_for_plot(data, trace, smooth_ms=None, smooth_mode='centered'):
    if trace is None:
        return None
    if smooth_ms is None or smooth_ms <= 0:
        return np.asarray(trace, dtype=float)
    bin_size = float(data.get('bin_size', 0.05))
    smooth_bins = max(1, int(round(float(smooth_ms) / 1000.0 / bin_size)))
    return _smooth(np.asarray(trace, dtype=float), smooth_bins, smooth_mode)


def _plot_line_with_optional_sem(ax, t, mean, sem, *, color, label, alpha=1.0,
                                 sem_alpha=0.20, linewidth=2.2,
                                 linestyle='-', zorder=2):
    if mean is None:
        return False
    ax.plot(t, mean, color=color, alpha=alpha, linewidth=linewidth,
            linestyle=linestyle, label=label, zorder=zorder)
    if sem is not None and sem.shape == mean.shape and np.isfinite(sem).any():
        ax.fill_between(t, mean - sem, mean + sem, color=color,
                        alpha=sem_alpha * alpha, linewidth=0, zorder=zorder - 0.5)
    return True


def _add_unit_peth_legend(ax, **kwargs):
    """Show control first in the legend while preserving opto-behind layering."""
    handles, labels = ax.get_legend_handles_labels()
    preferred = (
        'Control 20% block', 'Control 80% block',
        'Opto 20% block', 'Opto 80% block', 'Opto all trials',
    )
    order = [labels.index(label) for label in preferred if label in labels]
    order.extend(i for i in range(len(labels)) if i not in order)
    return ax.legend(
        [handles[i] for i in order], [labels[i] for i in order],
        **kwargs,
    )


_RASTER_CONTEXT_CACHE = {}
_RASTER_ONE = None
_RASTER_ATLAS = None
_RASTER_STATE_PROBABILITY = None


def _alignment_event_times_for_trials(trials, trial_indices, alignment):
    trial_indices = np.asarray(trial_indices, dtype=int)
    if alignment == 'Laser onset':
        return np.asarray(trials.intervals[trial_indices, 0], dtype=float)
    if alignment == 'Feedback':
        try:
            feedback = np.asarray(trials.feedback_times[trial_indices], dtype=float)
        except Exception:
            feedback = np.asarray(trials.intervals[trial_indices, 1], dtype=float)
        fallback = np.asarray(trials.intervals[trial_indices, 1], dtype=float)
        return np.where(np.isfinite(feedback), feedback, fallback)
    return np.asarray(trials.goCue_times[trial_indices], dtype=float)


def _metadata_trial_spec_for_pid(pid):
    from metadata_optostim_new import insertions as optostim_insertions

    for ins in optostim_insertions:
        if str(ins.get('PID')) == str(pid):
            return ins.get('opto inhibition trials', 'ALL')
    return 'ALL'


def _lazy_raster_one_and_atlas():
    global _RASTER_ONE, _RASTER_ATLAS
    if _RASTER_ONE is None:
        from one.api import ONE
        _RASTER_ONE = ONE(
            base_url='https://alyx.internationalbrainlab.org',
            cache_dir=Path('/Users/natemiska/Downloads/ONE/alyx.internationalbrainlab.org'),
        )
    if _RASTER_ATLAS is None:
        from iblatlas.atlas import AllenAtlas
        _RASTER_ATLAS = AllenAtlas()
    return _RASTER_ONE, _RASTER_ATLAS


def _lazy_glmhmm_helpers():
    global _RASTER_STATE_PROBABILITY
    try:
        import sys
        glmhmm_path = '/Users/natemiska/int-brain-lab/GLM-HMM'
        if glmhmm_path not in sys.path:
            sys.path.append(glmhmm_path)
        from psychometric_utils import get_glmhmm_indices
        if _RASTER_STATE_PROBABILITY is None:
            with open('/Users/natemiska/int-brain-lab/GLM-HMM/all_subject_states.csv', 'rb') as f:
                _RASTER_STATE_PROBABILITY = pickle.load(f)
        return get_glmhmm_indices, _RASTER_STATE_PROBABILITY
    except Exception as exc:
        print(f'Raster GLM-HMM helper unavailable; using non-GLM-HMM trial set: {exc}')
        return None, None


def _raster_context_for_pid(pid, data):
    """Load/cache raw spikes and final analysis trial selection for one PID."""
    pid = str(pid)
    if pid in _RASTER_CONTEXT_CACHE:
        return _RASTER_CONTEXT_CACHE[pid]

    one, ba = _lazy_raster_one_and_atlas()
    from optostim_preprocessing import load_session, prepare_trials, TrialQCParams
    import BS_config as _bs_cfg

    sb = load_session(pid, one, ba, load_waveforms=False)
    params = TrialQCParams.from_config(_bs_cfg)
    run_config = data.get('run_config', {}) or {}
    for key in (
        'beginning_block_trials_remove',
        'remove_stim_trials_preceded_by_stim',
        'use_GLMHMM_engaged_indices',
        'opto_trials_GLMHMM',
        'n_states',
    ):
        if key in run_config and hasattr(params, key):
            setattr(params, key, run_config[key])
    params.save_qc_outputs = 0
    params.figures_path = '/private/tmp'

    glmhmm_fn = None
    glmhmm_state = None
    if int(getattr(params, 'use_GLMHMM_engaged_indices', 0) or 0) == 1:
        glmhmm_fn, glmhmm_state = _lazy_glmhmm_helpers()

    trial_spec = _metadata_trial_spec_for_pid(pid)
    ts = prepare_trials(
        sb, trial_spec, params, Path('/private/tmp'),
        one=one,
        glmhmm_indices_fn=glmhmm_fn,
        glmhmm_state_probability=glmhmm_state,
    )
    if ts is None:
        raise RuntimeError(f'prepare_trials returned None for PID={pid}')

    ctx = {
        'pid': pid,
        'spike_times': np.asarray(sb.spikes.times, dtype=float),
        'spike_clusters': np.asarray(sb.spikes.clusters, dtype=int),
        'trials': sb.trials,
        'trial_indices': np.asarray(ts.inhibition_trials_range, dtype=int),
        'perturbation': np.asarray(ts.perturbation, dtype=bool),
        'block_ids': np.asarray(ts.block_ids, dtype=int),
    }
    _RASTER_CONTEXT_CACHE[pid] = ctx
    return ctx


def _raster_color(is_opto, block_id, color_by='condition', colors=None):
    palette, _ = _unit_peth_style(colors=colors)
    color_by = str(color_by or 'condition').lower()
    if color_by == 'block':
        return palette['control_80'] if int(block_id) == 1 else palette['control_20']
    if color_by in ('block_opto', 'condition_block'):
        if bool(is_opto):
            return palette['opto_80'] if int(block_id) == 1 else palette['opto_20']
        return palette['control_80'] if int(block_id) == 1 else palette['control_20']
    if color_by == 'none':
        return '0.1'
    return palette['opto_all'] if bool(is_opto) else palette['control_80']


def plot_unit_raster_from_raw(data, *, pid, clustnum, alignment, ax,
                              time_range=None, max_trials=300,
                              color_by='condition', raster_lw=0.35,
                              colors=None):
    """Plot an on-demand spike raster for one unit using raw cached session data."""
    ctx = _raster_context_for_pid(pid, data)
    trial_indices = ctx['trial_indices']
    if trial_indices.size == 0:
        raise RuntimeError('No final analysis trials available for raster.')

    t = np.asarray(data['peth_time'], dtype=float)
    time_range = _default_plot_time_range(data, time_range)
    time_range = _resolve_plot_time_range(t, time_range)
    t0, t1 = map(float, time_range)

    if max_trials is not None and trial_indices.size > int(max_trials):
        show_pos = np.linspace(0, trial_indices.size - 1, int(max_trials)).round().astype(int)
    else:
        show_pos = np.arange(trial_indices.size, dtype=int)
    show_trials = trial_indices[show_pos]
    events = _alignment_event_times_for_trials(ctx['trials'], show_trials, alignment)

    unit_mask = ctx['spike_clusters'] == int(clustnum)
    unit_spikes = ctx['spike_times'][unit_mask]
    segments = []
    colors = []
    opto_by_trial = {int(ti): bool(o) for ti, o in zip(trial_indices, ctx['perturbation'])}
    block_by_trial = {int(ti): int(b) for ti, b in zip(trial_indices, ctx['block_ids'])}

    for row, (trial_num, ev) in enumerate(zip(show_trials, events)):
        if not np.isfinite(ev):
            continue
        lo = ev + t0
        hi = ev + t1
        rel = unit_spikes[(unit_spikes >= lo) & (unit_spikes <= hi)] - ev
        if rel.size == 0:
            continue
        y0 = row + 0.08
        y1 = row + 0.92
        segments.extend([[(float(x), y0), (float(x), y1)] for x in rel])
        colors.extend([
            _raster_color(opto_by_trial.get(int(trial_num), False),
                          block_by_trial.get(int(trial_num), 0),
                          color_by=color_by, colors=colors)
            for _ in rel
        ])

    if segments:
        ax.add_collection(LineCollection(segments, colors=colors, linewidths=float(raster_lw)))
    ax.axvline(0, linestyle='--', color='0.35', linewidth=0.8)
    ax.set_xlim(time_range)
    ax.set_ylim(-0.5, max(len(show_pos) - 0.5, 0.5))
    ax.invert_yaxis()
    ax.set_ylabel('Trials')
    ax.set_xlabel(f'Time from {alignment} (s)')
    if show_trials.size:
        ax.set_yticks([0, len(show_trials) - 1])
        ax.set_yticklabels([str(int(show_trials[0])), str(int(show_trials[-1]))])
    ax.set_title(
        f'Raster: {len(show_trials)}/{len(trial_indices)} analysis trials',
        fontsize=9,
    )
    return ax


def plot_unit_block_peth(data, unit_index=None, *, pid=None, clustnum=None,
                         ax=None, smooth_ms=None, smooth_mode='centered',
                         opto_overlay='all', control_sem_alpha=0.22,
                         opto_alpha=0.90, title_prefix='', show_title=True,
                         show_legend=True, time_range=None,
                         axis_size=(7.0, 4.8), colors=None,
                         linestyles=None, event_line_color='0.25',
                         event_line_style='--', event_line_width=1.0):
    """Plot one unit's raw block PETHs from a BS results pickle.

    Control block 1 is the 80% left block (`trace_nonstim_80_raw`) and block 0 is
    the 20% left block (`trace_nonstim_20_raw`). Opto can be overlaid as all stim
    trials (`opto_overlay='all'`, default), block-specific stim traces
    (`'blocks'` or `'block_dotted'`), or hidden (`'none'`). The default
    publication palette is defined once in UNIT_BLOCK_PETH_COLORS and can be
    changed globally or overridden per call with ``colors={...}``. Opto traces
    are always drawn first and behind the control traces.

    True shaded SEM requires a pickle saved after the SEM fields were added. Older
    pickles with only raw means still plot line traces. time_range defaults to
    (-2, 5) for LaserOnset and (-5, 2) for GoCueOnset. axis_size is used only
    when ax is None.
    """
    _require_unit_block_peth_data(data.get('onset_alignment', 'This'), data)
    unit_index = _resolve_unit_index(data, unit_index, pid=pid, clustnum=clustnum)
    t = np.asarray(data['peth_time'], dtype=float)
    time_range = _default_plot_time_range(data, time_range)
    time_range = _resolve_plot_time_range(t, time_range)
    show = _time_mask_for_plot(t, time_range)
    t_plot = t[show]
    if ax is None:
        _, ax = plt.subplots(figsize=axis_size)
    palette, block_styles = _unit_peth_style(
        colors=colors, linestyles=linestyles)

    traces = {
        'ctrl_b1': _smooth_for_plot(data, _trace_for_unit(data, 'trace_nonstim_80_raw', unit_index),
                                    smooth_ms, smooth_mode),
        'ctrl_b0': _smooth_for_plot(data, _trace_for_unit(data, 'trace_nonstim_20_raw', unit_index),
                                    smooth_ms, smooth_mode),
        'ctrl_b1_sem': _smooth_for_plot(data, _trace_for_unit(data, 'trace_nonstim_80_sem_raw', unit_index),
                                        smooth_ms, smooth_mode),
        'ctrl_b0_sem': _smooth_for_plot(data, _trace_for_unit(data, 'trace_nonstim_20_sem_raw', unit_index),
                                        smooth_ms, smooth_mode),
    }
    for key, val in list(traces.items()):
        if val is not None and np.asarray(val).shape == t.shape:
            traces[key] = np.asarray(val, dtype=float)[show]

    # Draw opto first at the lower z-order. This is intentional: when traces
    # overlap, the control data remain visible on top.
    opto_overlay = str(opto_overlay or 'none').lower()
    if opto_overlay == 'all':
        opto_mean = _smooth_for_plot(data, _trace_for_unit(data, 'trace_stim_all', unit_index),
                                     smooth_ms, smooth_mode)
        opto_sem = _smooth_for_plot(data, _trace_for_unit(data, 'trace_stim_all_sem_raw', unit_index),
                                    smooth_ms, smooth_mode)
        if opto_mean is not None and opto_mean.shape == t.shape:
            opto_mean = opto_mean[show]
        if opto_sem is not None and opto_sem.shape == t.shape:
            opto_sem = opto_sem[show]
        _plot_line_with_optional_sem(
            ax, t_plot, opto_mean, opto_sem,
            color=palette['opto_all'], label='Opto all trials',
            alpha=opto_alpha, sem_alpha=0.18, linewidth=2.0,
            linestyle=block_styles['opto_all'], zorder=1.5,
        )
    elif opto_overlay in ('blocks', 'block_dotted', 'block_dots'):
        for key, sem_key, label, color_key, block_key, zorder in (
            ('trace_stim_20_raw', 'trace_stim_20_sem_raw',
             'Opto 20% block', 'opto_20', '20', 1.5),
            ('trace_stim_80_raw', 'trace_stim_80_sem_raw',
             'Opto 80% block', 'opto_80', '80', 1.6),
        ):
            opto_mean = _smooth_for_plot(data, _trace_for_unit(data, key, unit_index),
                                         smooth_ms, smooth_mode)
            opto_sem = _smooth_for_plot(data, _trace_for_unit(data, sem_key, unit_index),
                                        smooth_ms, smooth_mode)
            if opto_mean is not None and opto_mean.shape == t.shape:
                opto_mean = opto_mean[show]
            if opto_sem is not None and opto_sem.shape == t.shape:
                opto_sem = opto_sem[show]
            _plot_line_with_optional_sem(
                ax, t_plot, opto_mean, opto_sem,
                color=palette[color_key], label=label,
                alpha=opto_alpha, sem_alpha=0.12, linewidth=2.2,
                linestyle=block_styles[block_key], zorder=zorder,
            )
    elif opto_overlay != 'none':
        raise ValueError("opto_overlay must be 'all', 'blocks'/'block_dotted', or 'none'")

    _plot_line_with_optional_sem(
        ax, t_plot, traces['ctrl_b0'], traces['ctrl_b0_sem'],
        color=palette['control_20'], label='Control 20% block',
        sem_alpha=control_sem_alpha, linewidth=2.4,
        linestyle=block_styles['20'], zorder=3,
    )
    _plot_line_with_optional_sem(
        ax, t_plot, traces['ctrl_b1'], traces['ctrl_b1_sem'],
        color=palette['control_80'], label='Control 80% block',
        sem_alpha=control_sem_alpha, linewidth=2.4,
        linestyle=block_styles['80'], zorder=4,
    )

    align = data.get('onset_alignment', 'alignment')
    ax.axvline(
        0, linestyle=event_line_style, color=event_line_color,
        linewidth=event_line_width,
    )
    ax.set_xlabel(f'Time from {align} (s)')
    ax.set_ylabel('Firing rate (Hz)')
    ax.set_xlim(time_range)
    if show_title:
        ax.set_title(_unit_title(data, unit_index, title_prefix=title_prefix or align))
    if show_legend:
        _add_unit_peth_legend(ax, frameon=False, loc='best')
    return ax


def _normalise_alignment_datasets(datasets):
    if isinstance(datasets, dict):
        return list(datasets.items())
    out = []
    for item in datasets:
        if (isinstance(item, tuple) and len(item) == 2
                and isinstance(item[1], dict)):
            label, data = item
            out.append((str(label), data))
        else:
            data = item
            out.append((str(data.get('onset_alignment', f'alignment {len(out)+1}')), data))
    return out


def plot_unit_block_peth_alignments(datasets, unit_index=None, *, pid=None,
                                    clustnum=None, smooth_ms=None,
                                    opto_overlay='blocks', save_path=None,
                                    time_range=None, axis_size=(7.0, 4.8),
                                    figsize=None, include_raster=False,
                                    raster_max_trials=300,
                                    raster_color_by='block_opto',
                                    raster_height_fraction=0.42,
                                    colors=None, linestyles=None,
                                    show_title=True):
    """Plot the same unit across one or more alignment pickles.

    `datasets` can be a dict such as {'Laser onset': laser_data,
    'Feedback': feedback_data}. Units are matched across datasets by pid and
    clustnum. By default, opto trials are plotted separately by block using the
    publication palette and redundant 20/80 line styles, matching the browser
    view. time_range defaults
    independently by alignment: (-2, 5) for LaserOnset and (-5, 2) for
    GoCueOnset. axis_size is the size of each trace axis/panel; the total
    figure width is axis_size[0] * n panels. If include_raster=True, an
    on-demand raw-spike raster is added below each alignment trace.
    """
    pairs = _normalise_alignment_datasets(datasets)
    if not pairs:
        raise ValueError('No datasets provided.')
    for label, data in pairs:
        _require_unit_block_peth_data(label, data)

    first_label, first_data = pairs[0]
    first_idx = _resolve_unit_index(first_data, unit_index, pid=pid, clustnum=clustnum)
    ident = _unit_identity(first_data, first_idx)
    pid = ident['pid']
    clustnum = ident['clustnum']

    n = len(pairs)
    if figsize is None:
        if include_raster:
            figsize = (float(axis_size[0]) * n,
                       float(axis_size[1]) * (1.0 + float(raster_height_fraction)))
        else:
            figsize = _axis_size_to_figsize(axis_size, n_axes=n)

    if include_raster:
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(
            2, n,
            height_ratios=[1.0, float(raster_height_fraction)],
            hspace=0.28,
            wspace=0.25,
        )
        trace_axes = np.array([fig.add_subplot(gs[0, i]) for i in range(n)])
        raster_axes = np.array([
            fig.add_subplot(gs[1, i], sharex=trace_axes[i])
            for i in range(n)
        ])
    else:
        fig, axes = plt.subplots(1, n, figsize=figsize, squeeze=False)
        trace_axes = axes.ravel()
        raster_axes = [None] * n

    for ax, rax, (label, data) in zip(trace_axes, raster_axes, pairs):
        idx = _resolve_unit_index(data, pid=pid, clustnum=clustnum)
        plot_unit_block_peth(
            data, idx, ax=ax, smooth_ms=smooth_ms,
            opto_overlay=opto_overlay, title_prefix=label,
            show_title=show_title, show_legend=(ax is trace_axes[-1]),
            time_range=time_range, colors=colors,
            linestyles=linestyles,
        )
        if include_raster:
            ax.set_xlabel('')
            try:
                alignment = str(data.get('onset_alignment', label))
                plot_unit_raster_from_raw(
                    data, pid=pid, clustnum=clustnum,
                    alignment=alignment, ax=rax, time_range=time_range,
                    max_trials=raster_max_trials, color_by=raster_color_by,
                    colors=colors,
                )
            except Exception as exc:
                rax.text(
                    0.5, 0.5,
                    f'Raster unavailable\n{type(exc).__name__}: {exc}',
                    ha='center', va='center', transform=rax.transAxes,
                    fontsize=9,
                )
                rax.set_axis_off()
    fig.tight_layout()
    if save_path is not None:
        save_path = Path(save_path).expanduser()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200)
    return fig


def export_bs_unit_block_peths(
        pid, unit_number, *,
        laser_data_or_path=DEFAULT_BS_UNIT_LASER_RESULTS,
        feedback_data_or_path=DEFAULT_BS_UNIT_FEEDBACK_RESULTS,
        output_dir='~/python/saved_figures/BS_example_units',
        smooth_ms=250, opto_overlay='blocks',
        laser_time_range=None, feedback_time_range=None,
        axis_size=(6.6, 4.5), colors=None, linestyles=None,
        event_line_color='red', event_line_style='--',
        event_line_width=1.5,
        show_title=False, show_legend=True,
        save=True, show=False, close=None, transparent=False):
    """Reproduce one example unit as separate laser/feedback PDF panels.

    Parameters
    ----------
    pid : str
        Full PID or a unique PID prefix.
    unit_number : int
        The unit's saved cluster number (``clustnum``), printed as
        ``unit_number=...`` by :func:`browse_bs_unit_block_peths`. This is not
        the transient DataFrame row index.
    laser_data_or_path, feedback_data_or_path : dict or path-like
        Loaded results dicts or pickle paths. The defaults point to the current
        standard-crossfit output files.
    output_dir : path-like
        Destination for two vector PDFs, one per alignment.
    axis_size : (width, height)
        Exact Matplotlib figure size in inches for each panel.
    colors, linestyles : dict, optional
        Per-call overrides of UNIT_BLOCK_PETH_COLORS and
        UNIT_BLOCK_PETH_LINESTYLES.
    event_line_color, event_line_style, event_line_width
        Alignment-marker styling. Defaults match :func:`plot_delta_fr`.
    save, show : bool
        Save PDFs and/or display the figures. With ``show=False``, figures are
        closed after saving unless ``close=False`` is explicitly requested.

    Returns
    -------
    dict
        ``{'laser': {...}, 'feedback': {...}}`` entries containing each figure,
        axis, resolved unit index, and saved path (or None).
    """
    def _coerce(value, label):
        if isinstance(value, (str, Path)):
            return load_results(value)
        if isinstance(value, dict):
            return value
        raise TypeError(f'{label} must be a loaded results dict or pickle path.')

    laser_data = _coerce(laser_data_or_path, 'laser_data_or_path')
    feedback_data = _coerce(feedback_data_or_path, 'feedback_data_or_path')
    unit_number = int(unit_number)

    # Resolve against both files before producing anything, so a stale PID/unit
    # identifier cannot silently yield just one of the two requested panels.
    laser_idx = _resolve_unit_index(
        laser_data, pid=pid, clustnum=unit_number)
    ident = _unit_identity(laser_data, laser_idx)
    resolved_pid = ident['pid']
    feedback_idx = _resolve_unit_index(
        feedback_data, pid=resolved_pid, clustnum=unit_number)

    if close is None:
        close = not bool(show)
    out_dir = Path(output_dir).expanduser()
    if save:
        out_dir.mkdir(parents=True, exist_ok=True)

    specs = (
        ('laser', 'Laser onset', laser_data, laser_idx, laser_time_range),
        ('feedback', 'Feedback', feedback_data, feedback_idx, feedback_time_range),
    )
    result = {}
    for slug, label, data, unit_idx, time_range in specs:
        fig, ax = plt.subplots(figsize=tuple(map(float, axis_size)))
        plot_unit_block_peth(
            data, unit_idx, ax=ax, smooth_ms=smooth_ms,
            opto_overlay=opto_overlay, title_prefix=label,
            show_title=show_title, show_legend=False,
            time_range=time_range, colors=colors,
            linestyles=linestyles,
            event_line_color=event_line_color,
            event_line_style=event_line_style,
            event_line_width=event_line_width,
        )
        if show_legend:
            _add_unit_peth_legend(
                ax,
                frameon=False, fontsize=6.5, ncol=2, loc='best',
                handlelength=2.0, columnspacing=0.8, handletextpad=0.4,
            )
        fig.tight_layout(pad=0.45)

        save_path = None
        if save:
            save_path = (
                out_dir
                / f'BS_unit_{resolved_pid}_cluster{unit_number}_{slug}.pdf'
            )
            fig.savefig(
                save_path, format='pdf', transparent=bool(transparent),
            )
            print(f'Saved {label} example-unit panel: {save_path}')
        result[slug] = {
            'figure': fig,
            'axis': ax,
            'unit_index': int(unit_idx),
            'path': save_path,
        }

    if show:
        plt.show(block=False)
        for item in result.values():
            item['figure'].canvas.draw_idle()
    if close:
        for item in result.values():
            plt.close(item['figure'])
    return result


def browse_bs_unit_block_peths(data,datasets, idx=None, *, restrict=None,
                               start=0, max_units=None, sort_by='pval_empirical',
                               smooth_ms=None, opto_overlay='blocks', save_dir=None,
                               skip_plot_errors=True, time_range=None,
                               axis_size=(7.0, 4.8), include_raster=True,
                               raster_max_trials=300,
                               print_qc_metrics = True,
                               raster_color_by='block_opto', colors=None,
                               linestyles=None):
    """Interactively scroll through BS-selective units and plot block PETHs.

    Parameters
    ----------
    datasets : dict or list
        One or more loaded BS result dicts. Use a dict for nice panel labels,
        e.g. {'Laser onset': laser_data, 'Go cue onset': gocue_data}.
    idx : array-like, optional
        Positional unit indices from the first dataset. If None, units are
        selected from the first dataset with filter_units(..., bs_only=True).
    restrict : dict, optional
        Extra filter_units kwargs, e.g. {'brain_region_inhibited': 'SNr',
        'condition': 'ipsi', 'recorded_region': 'midbrain'}.
    time_range : (start, stop), optional
        Display window; None uses alignment defaults.
    axis_size : (width, height)
        Size of each alignment panel.
    include_raster : bool
        If True, add raw-spike rasters below each alignment panel. Rasters are
        loaded on demand from the PID's cached raw data, so older postprocessed
        pickles can still be browsed.
    raster_max_trials : int or None
        Maximum final-analysis trials to show per raster; None shows all.
        When capped, trials are sampled evenly across the session to preserve
        session-wide drift/presence structure.
    raster_color_by : {'condition', 'block', 'block_opto', 'none'}
        Raster spike coloring. 'condition' uses black/control and
        blue/opto. ``colors`` overrides UNIT_BLOCK_PETH_COLORS for both PETHs
        and rasters; ``linestyles`` overrides UNIT_BLOCK_PETH_LINESTYLES.
    """
    df = data['units']
    pairs = _normalise_alignment_datasets(datasets)
    first_label, first_data = pairs[0]
    for label, data in pairs:
        _require_unit_block_peth_data(label, data)
    if idx is None:
        opts = {} if restrict is None else dict(restrict)
        _validate_filter_kwargs(opts)
        opts.setdefault('bs_only', True)
        _, idx = filter_units(first_data, verbose=True, **opts)
    idx = np.asarray(idx, dtype=int)
    if sort_by is not None and sort_by in first_data['units'].columns:
        vals = first_data['units'].reset_index(drop=True).iloc[idx][sort_by].to_numpy()
        order = np.argsort(np.where(np.isfinite(vals), vals, np.inf))
        idx = idx[order]
    idx = idx[int(start):]
    if max_units is not None:
        idx = idx[:int(max_units)]

    visited = []
    for count, unit_i in enumerate(idx, start=int(start) + 1):
        ident = _unit_identity(first_data, unit_i)
        visited.append(ident)
        save_path = None
        # if save_dir is not None:
        #     save_path = (
        #         Path(save_dir).expanduser()
        #         / f"BS_unit_block_peth_{ident['pid']}_cluster{ident['clustnum']}.png"
        #     )
        print(
            f"[{count}/{int(start)+len(idx)}] {first_label}: "
            f"row={ident['row_index']} pid={ident['pid']} "
            f"unit_number={ident['clustnum']} (clustnum) "
            f"region={ident['region']} p={ident['pval_empirical']}"
        )
        if print_qc_metrics:
            print('QC metrics:')
            print('qp_fr_median = ' + str(df['qp_fr_median'][unit_i]))
            print('qp_fr_segment_range_frac = ' + str(df['qp_fr_segment_range_frac'][unit_i]))
            print('qp_resid_drift_range_frac = ' + str(df['qp_resid_drift_range_frac'][unit_i]))
            print('qp_resid_drift_cv = ' + str(df['qp_resid_drift_cv'][unit_i]))
            print('qp_resid_abs_rho_time = ' + str(df['qp_resid_abs_rho_time'][unit_i]))
            print('qp_low_activity_fraction = ' + str(df['qp_low_activity_fraction'][unit_i]))
            print('qp_max_low_activity_run = ' + str(df['qp_max_low_activity_run'][unit_i]))
            print('qp_n_block_effect_segments = ' + str(df['qp_n_block_effect_segments'][unit_i]))
            print('qp_block_effect_global = ' + str(df['qp_block_effect_global'][unit_i]))
            print('qp_block_effect_segment_cv = ' + str(df['qp_block_effect_segment_cv'][unit_i]))
            print('qp_block_effect_sign_consistency = ' + str(df['qp_block_effect_sign_consistency'][unit_i]))
            print('qp_block_effect_dominance = ' + str(df['qp_block_effect_dominance'][unit_i]))

        try:
            fig = plot_unit_block_peth_alignments(
                pairs, pid=ident['pid'], clustnum=ident['clustnum'],
                smooth_ms=smooth_ms, opto_overlay=opto_overlay,
                save_path=save_path, time_range=time_range,
                axis_size=axis_size,
                include_raster=include_raster,
                raster_max_trials=raster_max_trials,
                raster_color_by=raster_color_by,
                colors=colors, linestyles=linestyles,
            )
        except Exception as exc:
            if not skip_plot_errors:
                raise
            print(
                f"  Skipping this unit because it could not be plotted: {type(exc).__name__}: {exc}"
            )
            continue
        plt.show(block=False)
        fig.canvas.draw_idle()
        plt.pause(0.05)
        print('Press any key/click in the figure to continue, or close the figure.')
        # try:
        plt.waitforbuttonpress()
        plt.close(fig)
        # finally:
        #     if plt.fignum_exists(fig.number):
        #         plt.close(fig)
    return pd.DataFrame(visited)


def plot_paired_bar(data, idx, ax=None, window_s=None, test='wilcoxon',
                    sign_flip=None, sign_mode=sign_mode_option,
                    sign_window=DEFAULT_SIGN_WINDOW, norm_mode=None,
                    orientation_mode=None, trial_estimator=None,
                    whole_control_min_fr=QP_CONTROL_SCALAR_MIN_FR,
                    baseline_subtract=False, baseline_window=(-2.0, 0.0)):
    """Paired control-vs-laser mean block-delta over a user-defined window,
    computed from the stored per-unit traces (post-hoc -- no re-run, and it
    samples the actual window you ask for rather than the fixed pipeline scalar).

    window_s : (t0, t1) in seconds relative to alignment. If None, defaults to
        (0, 1) for a LaserOnset pickle (the post-laser inhibition window) and
        (-1, 0) for GoCueOnset/Feedback pickles.
    test : 'wilcoxon' (signed-rank, robust to outliers, default) or 'ttest'.
    sign_mode : 'block_crossfit', 'split_half', 'none', or explicit legacy mode.
        ``block_crossfit`` is the preferred control-preference-aligned estimator;
        it matches held-out control and opto trials within block/session time.
    sign_window : control-only direction window; defaults to (-2, 0) seconds.
    sign_flip : compatibility alias. True now requests split-half orientation;
        False requests no orientation.
    baseline_subtract : if True, subtract each condition's own per-unit mean over
        baseline_window before averaging window_s. With an independent sign mode
        such as block_crossfit, the paired comparison is an intuitive time-resolved
        difference-in-differences (change from baseline), without using baseline
        control noise to choose the displayed sign.
    orientation_mode, trial_estimator : Orthogonal primary-analysis options;
        when either is supplied they supersede legacy sign_mode/sign_flip.
    """
    t = np.asarray(data['peth_time'], dtype=float)
    align = str(data.get('onset_alignment', 'Laser onset')).lower()
    if window_s is None:
        window_s = (-1.0, 0.0) if ('go' in align or 'cue' in align or 'feedback' in align) else (0.0, 1.0)
    w = (t >= window_s[0]) & (t < window_s[1])
    if not w.any():
        print(f'window {window_s} empty for peth_time range '
              f'[{t[0]:.2f}, {t[-1]:.2f}]; nothing to plot.')
        return None
    idx = np.asarray(idx, dtype=int)
    use_primary_api = _requests_primary_api(
        norm_mode, orientation_mode, trial_estimator)
    if use_primary_api:
        ns, st, active_mode, resolved_orientation, resolved_estimator = (
            _primary_delta_traces(
                data, idx, norm_mode=norm_mode or 'raw_hz',
                orientation_mode=orientation_mode or 'qp_preference',
                trial_estimator=trial_estimator or 'all_trials',
                whole_control_min_fr=whole_control_min_fr,
            ))
        resolved_sign_mode = resolved_orientation
    else:
        ns, st, active_mode, resolved_sign_mode = _oriented_delta_traces(
            data, idx, mode=norm_mode, sign_mode=sign_mode,
            sign_window=sign_window, sign_flip=sign_flip,
        )
        resolved_orientation = resolved_sign_mode
    if baseline_subtract and not use_primary_api and resolved_sign_mode == 'legacy':
        raise ValueError(
            "baseline_subtract with sign_mode='legacy' is circular: the control "
            "baseline chooses its own sign. Use sign_mode='block_crossfit'.")
    if baseline_subtract:
        pre = _window(data, baseline_window)
        if not np.any(pre):
            raise ValueError(
                f'baseline_window={tuple(baseline_window)} is empty for peth_time')
        ns = ns - _nanmean(ns[:, pre], axis=1)[:, None]
        st = st - _nanmean(st[:, pre], axis=1)[:, None]
    a = _nanmean(ns[:, w], axis=1)
    b = _nanmean(st[:, w], axis=1)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if ax is None:
        _, ax = plt.subplots(figsize=(3.5, 4.5))
    ax.bar(1, np.nanmean(a), yerr=np.nanstd(a) / np.sqrt(max(a.size, 1)),
           color='black', ecolor='black', alpha=0.7, width=0.6)
    ax.bar(2, np.nanmean(b), yerr=np.nanstd(b) / np.sqrt(max(b.size, 1)),
           color=OPTO_COLOR, ecolor='black', alpha=0.7, width=0.6)
    ax.set_xticks([1, 2]); ax.set_xticklabels(['control', 'laser'])
    relation_label = (
        'absolute block difference'
        if resolved_orientation == 'independent_absolute'
        else 'preferred - nonpreferred'
        if resolved_orientation == 'qp_preference'
        else '80 - 20 blocks')
    ylabel = (
        f'mean delta FR, {relation_label} [{_delta_value_label(active_mode)}]\n'
        f'{window_s[0]:g} to {window_s[1]:g}s')
    if baseline_subtract:
        ylabel += '\nchange from own baseline'
    ax.set_ylabel(ylabel)
    if a.size >= 2:
        if test == 'ttest':
            stat, p = stats.ttest_rel(a, b); lbl = 'paired t'
        else:
            try:
                stat, p = stats.wilcoxon(a, b)
            except ValueError:
                stat, p = np.nan, np.nan
            lbl = 'Wilcoxon'
        ax.set_title(f'{lbl}: p = {p:.3g}  (n={a.size})')
        return ax, (stat, p)
    return ax, (np.nan, np.nan)


# ---------------------------------------------------------------------------
# Summary stats (population-level BS-fraction test)
# ---------------------------------------------------------------------------
BS_SPECIFIC_FILTER_KEYS = {'bs_only', 'max_pval_empirical'}


def _eligible_population_filter_options(filter_options):
    """Active filters for the all-unit denominator, excluding BS-call gates."""
    opts = dict(filter_options or {})
    opts['bs_only'] = False
    opts['max_pval_empirical'] = None
    return opts


def summarize(df, alpha=0.05, data=None, filter_options=None):
    """Print BS-unit counts and, when possible, the matching all-unit denominator.

    If `data` and `filter_options` are provided, the denominator is recomputed
    from the same active filters after removing BS-specific gates
    (`bs_only`, `max_pval_empirical`). This keeps anatomical/QC filters such as
    region, drift, nonstationarity, light artifact, firing-rate, and trial-count
    exclusions active for the non-BS count.
    """
    if data is not None and filter_options is not None:
        eligible_options = _eligible_population_filter_options(filter_options)
        eligible_df, _ = filter_units(data, verbose=False, **eligible_options)

        n_selected = int(len(df))
        n_selected_bs = int((df['BS_score'] == 1).sum()) if 'BS_score' in df else np.nan
        n_eligible = int(len(eligible_df))
        n_eligible_bs = int((eligible_df['BS_score'] == 1).sum()) if 'BS_score' in eligible_df else np.nan
        n_eligible_non_bs = (
            int(n_eligible - n_eligible_bs)
            if np.isfinite(n_eligible_bs) else np.nan
        )
        used_frac = n_selected_bs / n_eligible if n_eligible and np.isfinite(n_selected_bs) else np.nan
        eligible_frac = n_eligible_bs / n_eligible if n_eligible and np.isfinite(n_eligible_bs) else np.nan

        print('\nAnalysis unit counts:')
        print(f'  Selected units used by current analysis: {n_selected}')
        print(f'  Selected BS units:                    {n_selected_bs}')
        print(f'  Eligible all units after non-BS filters: {n_eligible}')
        print(f'  Eligible BS_score == 1 units:            {n_eligible_bs}')
        print(f'  Eligible non-BS units:                   {n_eligible_non_bs}')
        print(f'  Selected-BS / eligible-all fraction:     {used_frac:.3f}')
        print(f'  BS_score==1 / eligible-all fraction:     {eligible_frac:.3f}')

        if n_eligible and np.isfinite(n_selected_bs):
            try:
                res = stats.binomtest(int(n_selected_bs), int(n_eligible), alpha,
                                      alternative='greater')
                pval = res.pvalue
            except AttributeError:  # SciPy < 1.7
                pval = stats.binom_test(int(n_selected_bs), int(n_eligible), alpha,
                                        alternative='greater')
            print(f'  Binomial test (selected BS fraction > {alpha:.3f}): p = {pval:.3g}')

        if {'brain_region_inhibited', 'condition'}.issubset(eligible_df.columns) and n_eligible:
            group = (
                eligible_df
                .groupby(['brain_region_inhibited', 'condition'])['BS_score']
                .agg(['size', 'sum'])
                .rename(columns={'size': 'total_units', 'sum': 'bs_score_units'})
            )
            group['non_bs_units'] = group['total_units'] - group['bs_score_units']
            group['BS_frac'] = group['bs_score_units'] / group['total_units']
            print('\nEligible denominator by inhibited region/condition:')
            print(group)

        return {
            'n_selected': n_selected,
            'n_selected_bs': n_selected_bs,
            'n_eligible': n_eligible,
            'n_eligible_bs': n_eligible_bs,
            'n_eligible_non_bs': n_eligible_non_bs,
            'selected_bs_fraction_of_eligible': used_frac,
            'eligible_bs_fraction': eligible_frac,
        }

    # Backward-compatible summary for callers that pass only an already-filtered
    # DataFrame.
    n = len(df)
    n_bs = int((df['BS_score'] == 1).sum()) if 'BS_score' in df else 0
    frac = n_bs / n if n else float('nan')
    print(f'\nUnits: {n} | BS units: {n_bs} | fraction: {frac:.3f} '
          f'(chance = {alpha:.3f})')
    if n:
        try:
            res = stats.binomtest(n_bs, n, alpha, alternative='greater')
            pval = res.pvalue
        except AttributeError:  # SciPy < 1.7
            pval = stats.binom_test(n_bs, n, alpha, alternative='greater')
        print(f'Binomial test (BS fraction > chance): p = {pval:.3g}')
    if 'brain_region_inhibited' in df and n:
        print(df.groupby(['brain_region_inhibited', 'condition'])['BS_score']
              .agg(['size', 'sum', 'mean']).rename(columns={'mean': 'BS_frac'}))
    return {'n': n, 'n_bs': n_bs, 'fraction': frac}

# ---------------------------------------------------------------------------
# Example usage  -- edit OPTIONS and run
# ---------------------------------------------------------------------------
alignment = 'laser'

if __name__ == '__main__':
    # Matches BS_config.figure_prefix='NOGLMHMM_standard'. Change this if the
    # pipeline run uses a different figure_prefix.
    if alignment == 'laser':
        RESULTS_PATH = '~/python/saved_figures/BS_all_insertions_NOGLMHMM_standard_crossfit_LaserOnset.pkl'
    else:
        RESULTS_PATH = '~/python/saved_figures/BS_all_insertions_NOGLMHMM_standard_crossfit_Feedback.pkl'

    # RESULTS_PATH = '~/python/saved_figures/BS_all_insertions_GLMHMM_crossfit_LaserOnset.pkl'
    # RESULTS_PATH = '~/python/saved_figures/BS_all_insertions_GLMHMM_crossfit_Feedback.pkl'

    # RESULTS_PATH = '~/python/saved_figures/BS_all_insertions_ALL_LaserOnset.pkl'
    # RESULTS_PATH = '~/python/saved_figures/BS_all_insertions_ALL_Feedback.pkl'

    # RESULTS_PATH = '~/python/saved_figures/BS_all_insertions_NOGLMHMM_PCC_LaserOnset.pkl'
    # RESULTS_PATH = '~/python/saved_figures/BS_all_insertions_NOGLMHMM_PCC_Feedback.pkl'

    OPTIONS = dict(
        brain_region_inhibited='SNr',     # 'SNr' | 'ZI' | 'STN' | list | None
        condition='ipsi',                 # 'ipsi' | 'contra' | None
        # recorded_region='midbrain',       # 'midbrain' | ['MRN','SCig',...] | None
        recorded_region=['MRN','RR'],
        # recorded_region=['PRNr','NB','P','fp'],       # 'midbrain' | ['MRN','SCig',...] | None
        # recorded_region=['POST','PRE','RSPd1','RSPd2/3'],#,'dhc','bic','cpd','or'],
        # recorded_region_beryl=['MRN','SCm','SCs'],  # SC/MRN. For NOT SC/MRN: exclude_regions(['MRN','SCm','SCs'])
        # recorded_region_beryl=['PRNr'],
        # recorded_region_beryl=exclude_regions(['MRN','SCm','SCs']),
        # recorded_region_beryl=exclude_regions(['root', 'PRNr', 'RN', 'PPN','POST','RR','void','NB','PRE','RSPd','PAG']),
        pids=None, # None=all, or ['pid1','pid2']
        exclude_pids=None,#['e1b4c254-0455-4cd3-9642-0e23892aef85','09ee9be3-3c85-46bb-aed3-3143862ef03d'],#['c9a6b866-2d9b-481c-86ec-0d4937fbd696','68288763-9572-4678-9eb4-3866e3e9fb3d','fc4f446b-177c-4b94-89d2-14c0500374a4','32425853-de5f-4e5d-8a73-fe1285893c7f','9583d73c-ee29-45d1-9aa1-2b5917bcf726'],                # e.g. ['bad_pid']; None disables
        bs_only=True,
        max_pval_empirical=None,#0.05,          # e.g. 0.01 for stricter pseudo/permutation BS units; None disables
        exclude_drift_units=False,
        exclude_nonstationary_units=False, # uses nonstationary_unit; becomes 1 only when BS_config thresholds flag a unit
        # QP nonstationarity filters (available after rerun with the new metrics):
        max_qp_fr_segment_range_frac=None,
        max_qp_resid_drift_range_frac=None,#0.8, too restrictive
        max_qp_resid_drift_cv=None,#0.4, too restrictive
        max_qp_resid_abs_rho_time=None,#0.13,
        max_qp_low_activity_fraction=None,
        max_qp_max_low_activity_run=None,
        min_qp_block_effect_sign_consistency=None, #0.75
        max_qp_block_effect_segment_cv=None,
        max_qp_block_effect_dominance=None,
        exclude_axonal_units=False, #False
        exclude_light_artifact=True,
        exclude_amplitude_outliers=True,
        IBL_quality_label_threshold=0/3, #1/3
        presence_threshold=None, #0.5
        min_firing_rate=1, #1
        min_n_per_block=None,
        min_n_per_delta_block=None,        # e.g. 25/50/100 on the actual plotted delta-PETH trial counts
        # Pre-laser baseline-stability exclusions (off unless set):
        max_prelaser_delta_gap=None,      # e.g. 10.0 -> raw stim-control baseline gap <=10 percentage points
        max_cv_prelaser_delta_gap=None,   # e.g. 10.0 -> split-half-sign baseline gap <=10; needs new pickle
        max_prelaser_zdev=None,           # e.g. 2.0 -> drop units already differing opto vs control
        min_prelaser_separation_frac=None,# e.g. 0.3 -> require a stable baseline block code
        max_prelaser_trace_std=None,      # e.g. 10.0 -> require low pre-laser trace variability
        min_prelaser_baseline_fr=None,    # e.g. 1.0 -> require stim raw baseline FR >= 1 Hz
        baseline_fr_window_s=1.0,
        prelaser_window_s=2,            # window [-this, 0]; toggle up to ~2.0
    )

    data = load_results(RESULTS_PATH)
    df_sel, idx = filter_units(data, **OPTIONS)
    summarize(df_sel, data=data, filter_options=OPTIONS)

    NORM = 'whole_control_scalar'#'raw_hz' #'whole_control_scalar'
    ORIENTATION = 'qp_preference' #'independent_absolute' (not working properly)
    ESTIMATOR = 'all_trials' #'matched_crossfit'

    if alignment == 'laser':
        window_vals = (0,2)
        time_range = (-2,5)
    else:
        window_vals = (-2,0)
        time_range = (-5,2)

    plot_delta_fr(
        data, idx,
        norm_mode=NORM,
        orientation_mode=ORIENTATION,
        trial_estimator=ESTIMATOR,
        smooth_ms=300,
        time_range=time_range,
        y_range=None,
    )
    plt.show()

    plot_paired_bar(
        data, idx,
        window_s=window_vals,
        norm_mode=NORM,
        orientation_mode=ORIENTATION,
        trial_estimator=ESTIMATOR,
    )
    plt.show()

    summary = run_insertion_delta_fr_analysis(
        data, idx,
        min_units_per_insertion=10,
        window_s=window_vals,
        norm_mode=NORM,
        orientation_mode=ORIENTATION,
        trial_estimator=ESTIMATOR,
        min_control_delta_fr=None,
        smooth_ms=300,
    )

    # # Common, alignment-invariant denominator: duration-weighted control QP
    # # firing, balanced equally across 80/20 blocks. Units below qp_min_fr are
    # # marked missing rather than divided by a floor.
    # use_norm(data, 'qp_control_scalar', qp_min_fr=0.5)
    # audit_futureproof_payload(data)

    # # See whether the effect is general or localized to particular insertions:
    # list_pids(data)


    # Report the un-baselined held-out gap. The plotted traces below use the same
    # fixed control-QP denominator and do not receive an additional time-baseline
    # subtraction.
    block_crossfit_sign_diagnostic(
        data, idx, mode='qp_control_scalar', sign_window=(-2, 0),
        pre_window=(-2, 0), post_window=(0, 2), smooth_ms=500,
    )

    # mask_low_fr_bins(data, threshold_hz = 0.1)


    # ------------------------------------------------------------------
    # New orthogonal primary-analysis API (examples; uncomment as needed)
    # ------------------------------------------------------------------
    # Compare all 2 normalization x 2 orientation x 2 estimator combinations:
    # fig, axes = plot_primary_delta_option_grid(
    #     data, idx,
    #     norm_modes=('raw_hz', 'whole_control_scalar'),
    #     orientation_modes=('qp_preference', 'independent_absolute'),
    #     trial_estimators=('all_trials', 'matched_crossfit'),
    #     smooth_ms=300, time_range=(-2, 5),
    #     whole_control_min_fr=0.5,
    #     suptitle='SNr ipsi BS midbrain: analysis-option comparison',
    # ); plt.show()

    # Recommended simple primary trace: raw Hz, QP-defined preference, all
    # eligible inhibition-range trials.
    # plot_delta_fr(
    #     data, idx, title='Raw Hz | QP preference | all trials',
    #     norm_mode='raw_hz', orientation_mode='qp_preference',
    #     trial_estimator='all_trials', smooth_ms=300,
    #     time_range=(-2, 5), y_range=None,
    # ); plt.show()

    # Requested absolute-difference version (condition-specific |80-20|):
    # plot_delta_fr(
    #     data, idx, title='Raw Hz | independent absolute | all trials',
    #     norm_mode='raw_hz', orientation_mode='independent_absolute',
    #     trial_estimator='all_trials', smooth_ms=300,
    #     time_range=(-2, 5), y_range=None,
    # ); plt.show()

    # Normalized version using one block-balanced whole-control-trace scalar
    # per unit. Swap orientation_mode between the two choices above.
    # plot_delta_fr(
    #     data, idx, title='Whole-control normalized | QP preference',
    #     norm_mode='whole_control_scalar',
    #     orientation_mode='qp_preference', trial_estimator='all_trials',
    #     whole_control_min_fr=0.5, smooth_ms=300,
    #     time_range=(-2, 5), y_range=None,
    # ); plt.show()

    # Retain the matched held-out estimator by changing only this dimension:
    # plot_delta_fr(
    #     data, idx, title='Raw Hz | QP preference | matched held-out',
    #     norm_mode='raw_hz', orientation_mode='qp_preference',
    #     trial_estimator='matched_crossfit', smooth_ms=300,
    #     time_range=(-2, 5), y_range=None,
    # ); plt.show()

    # Use the SAME three options for the unit-level paired bar and insertion
    # summary so traces, scalar values, and statistics have identical meaning:
    # plot_paired_bar(
    #     data, idx, window_s=(0, 2),
    #     norm_mode='raw_hz', orientation_mode='qp_preference',
    #     trial_estimator='all_trials',
    # ); plt.show()
    # primary_insertion_summary = run_insertion_delta_fr_analysis(
    #     data, idx,
    #     out_dir='~/python/saved_figures/BS_insertion_delta_fr',
    #     min_units_per_insertion=10, window_s=(0, 2),
    #     norm_mode='raw_hz', orientation_mode='qp_preference',
    #     trial_estimator='all_trials', min_control_delta_fr=None,
    #     smooth_ms=300, save_insertion_plots=True,
    # )
    # plot_zscore(data, idx, title='SNr ipsi, BS midbrain'); plt.show()
    # Window defaults to (0,1)s for LaserOnset, (-1,0)s for GoCueOnset; override
    # with window_s=(t0, t1). Uses Wilcoxon signed-rank by default.
    # plot_paired_bar(
    #     data, idx, window_s=(0, 2),
    #     norm_mode='qp_control_scalar',
    #     sign_mode='block_crossfit', sign_window=(-2, 0),
    #     baseline_subtract=False, baseline_window=(-2, 0),
    # ); plt.show()

    # Insertion-level delta-FR analysis. This is the preferred first-pass
    # inferential view over treating every unit as an independent observation:
    # units are averaged within PID, insertions below min_units_per_insertion
    # are dropped, and paired control-vs-opto stats are run across insertions.
    # Window defaults are (0, 0.5)s for LaserOnset and (-0.5, 0)s for GoCueOnset.
    # INSERTION_BAR_PLOT_OPTIONS = dict(
    #     figsize=(2.5, 4.5),
    #     show_xticks=False,
    #     show_title=False,
    #     control_color='black',
    #     opto_color=OPTO_COLOR,
    #     bar_alpha=0.7,
    #     bar_width=0.6,
    #     pair_line_color='gray',
    #     pair_line_alpha=0.5,
    #     pair_linewidth=1.0,
    #     pair_marker='o',
    #     pair_markersize=4,
    #     errorbar_linewidth=2.5,
    #     errorbar_capsize=4,
    #     xmargin=0.5,
    #     ylabel_fontsize=9,
    #     xtick_fontsize=8,
    #     save_dpi=150,
    #     save_tight_bbox=True,
    # )
    # insertion_summary = run_insertion_delta_fr_analysis(
    #     data, idx,
    #     out_dir='~/python/saved_figures/BS_insertion_delta_fr',
    #     min_units_per_insertion=10,
    #     window_s=(0, 2),
    #     norm_mode=None,#'qp_control_scalar',
    #     sign_mode=sign_mode_option,
    #     sign_window_s=(-2, 0),
    #     min_control_delta_fr=None,#0.0,  # set None to retain negative-control PIDs
    #     baseline_subtract=False,
    #     baseline_window=(-2, 0),
    #     smooth_ms=300,
    #     save_insertion_plots=True,
    #     bar_plot_options=INSERTION_BAR_PLOT_OPTIONS,
    # )

    # Diagnose a suspicious insertion unit by unit. This reproduces the exact
    # estimator/settings above and adds raw block PETHs, separate cross-fit
    # folds, and broad-BS versus inhibition-range QP activity. Replace PID as
    # needed; arrows or Previous/Next scroll through contributing units.
    insertion_unit_browser = browse_insertion_delta_units(
        data, idx,
        pid='09ee9be3-3c85-46bb-aed3-3143862ef03d',
        window_s=(0, 2),
        norm_mode='qp_control_scalar',
        sign_mode='block_crossfit', sign_window_s=(-2, 0),
        baseline_subtract=False, baseline_window=(-2, 0),
        smooth_ms=300,
        diagnostics_csv=(
            '~/python/saved_figures/BS_insertion_delta_fr/'
            '930adc32_unit_diagnostics.csv'),
    )

    # Additional publication-oriented multilevel analysis. The paired unit
    # effect is control - opto over window_s. Mouse, session, and insertion are
    # retained as nested random-intercept levels whenever separately
    # identifiable. Only non-finite unit effects and insertions with <5 valid
    # units are excluded by default; negative-control insertions are retained
    # and explicitly flagged in the audit.
    # multilevel_summary = run_multilevel_delta_fr_analysis(
    #     data, idx,
    #     out_dir='~/python/saved_figures/BS_multilevel_delta_fr',
    #     window_s=(0, 2),
    #     norm_mode='qp_control_scalar',
    #     sign_mode='block_crossfit',
    #     sign_window_s=(-2, 0),
    #     min_units_per_insertion=5,
    #     min_valid_unit_fraction=None,
    #     min_reference_sign_agreement=None,
    #     min_control_delta_fr=None,
    #     max_abs_unit_effect=None,
    #     reml=True,
    #     title='SNr ipsi BS midbrain: nested control - opto effect',
    # )

    # New rerun-pickle diagnostics:
    # use_norm(data, 'zero_2_nan')
    # use_norm(data, 'qp_control_scalar', qp_min_fr=1.0)  # stricter sensitivity
    # split_half_sign_diagnostic(data, idx, mode='zero_2_nan', sign_window=(-2, 0), smooth_ms=1500)
    # trial_count_matched_diagnostic(data, idx, mode='zero_2_nan', sign_mode='none', smooth_ms=1500)
    # gap_df = baseline_gap_metrics(data, idx, mode='zero_2_nan', pre_window=(-5.0, 0.0))

    # Export baseline-unstable units for the CD pipeline to exclude:
    # export_unstable_units(data, '~/python/saved_figures/unstable_units.pkl',
    #                       max_prelaser_zdev=2.0, prelaser_window_s=0.5,
    #                       restrict={'recorded_region': 'midbrain'})

    # Browse individual bias-selective units once LaserOnset and GoCueOnset
    # pickles have been generated with save_raw_block_peths=1:
    laser_data = load_results('~/python/saved_figures/BS_all_insertions_NOGLMHMM_standard_crossfit_LaserOnset.pkl')
    feedback_data = load_results('~/python/saved_figures/BS_all_insertions_NOGLMHMM_standard_crossfit_Feedback.pkl')
    browse_bs_unit_block_peths(laser_data,
        {'Laser onset': laser_data, 'Feedback': feedback_data},
        restrict=dict(brain_region_inhibited='SNr',
                        condition='ipsi',
                        recorded_region='midbrain',
                        min_n_per_block=100),
        opto_overlay='blocks',  # four block-specific traces (the new default)
        smooth_ms=250,
        # Colors can be changed globally in UNIT_BLOCK_PETH_COLORS or per call:
        # colors=dict(control_20='0.6', control_80='black',
        #             opto_20='lightskyblue', opto_80='deepskyblue'),
        # time_range=None,       # defaults: Laser (-2,5), Feedback (-5,2)
        # axis_size=(5.0, 3.5),  # size of each Laser/Feedback axis
        include_raster=True,
        raster_max_trials=300,
        raster_color_by='block_opto',
    )

    # Reproduce a chosen example later using the PID and unit_number printed by
    # the browser. This saves separate, publication-sized vector PDFs. Set
    # save=False, show=True to inspect without writing files.
    exported = export_bs_unit_block_peths(
        pid='77c33d3e-8b71-43f9-9a9c-b7dc49a25e30',
        unit_number=286,
        laser_data_or_path=laser_data,
        feedback_data_or_path=feedback_data,
        output_dir='~/python/saved_figures/BS_example_units',
        save=True,
        show=False,
        # colors=dict(opto_20='#8FD3FF', opto_80='#008FD5'),
    )

    # df_sel, idx = filter_units(data, **OPTIONS)
    # use_norm(data, 'per_bin')
    # plot_delta_fr(data, idx, title='unsigned', sign_mode='none', smooth_ms=1500); plt.show()


def detection_audit(data, alpha=0.05, restrict=None):
    """Quantify perm-test under-detection WITHOUT re-running. Compares each
    unit's raw block ranksum p (`pval_real`, independent of the pseudo) to its
    permutation call (`BS_score`). Many units with a clearly significant
    `pval_real` but BS_score == 0 is the signature of the all-trials-pseudo bug
    (fixed by pseudo_restrict_to_nonstim=True in isbiasblockselective_perm_vector).
    After re-running with the fix, the 'missed' count should drop sharply."""
    df = data['units']
    if restrict:
        df, _ = filter_units(data, verbose=False, **restrict)
    if 'pval_real' not in df.columns:
        print('pval_real not in pickle; re-run the pipeline to enable this audit.')
        return None
    sig_raw = df['pval_real'] < alpha
    called = df['BS_score'] == 1
    missed = sig_raw & ~called
    n = len(df)
    print(f'Detection audit on {n} units (alpha={alpha}):')
    print(f'  raw ranksum significant (pval_real<{alpha}): {int(sig_raw.sum())} '
          f'({sig_raw.mean()*100:.1f}%)')
    print(f'  called BS by permutation test:               {int(called.sum())} '
          f'({called.mean()*100:.1f}%)')
    print(f'  significant-but-MISSED by permutation:       {int(missed.sum())} '
          f'({missed.mean()*100:.1f}%)')
    if sig_raw.sum():
        print(f'  -> {missed.sum()/max(sig_raw.sum(),1)*100:.0f}% of clearly block-tuned '
              f'units are missed by the current permutation null.')
    return {'n': n, 'n_sig_raw': int(sig_raw.sum()),
            'n_called': int(called.sum()), 'n_missed': int(missed.sum())}


# ---------------------------------------------------------------------------
# Midbrain recorded-region inventory helper
# ---------------------------------------------------------------------------
def _region_inventory_table(df, region_col, include_unknown=False):
    """Count units/insertions/mice for one region-acronym column."""
    if region_col not in df.columns:
        return pd.DataFrame(columns=[
            region_col, 'n_units', 'n_bs_units', 'bs_fraction',
            'n_insertions', 'n_mice', 'inhibited_region_conditions',
        ])

    work = df.copy()
    work['_region'] = work[region_col].map(
        lambda value: _clean_region_acronym(value, include_unknown=include_unknown)
    )
    work = work.dropna(subset=['_region'])

    rows = []
    for region, sub in work.groupby('_region', sort=True):
        conditions = []
        if {'brain_region_inhibited', 'condition'}.issubset(sub.columns):
            condition_df = sub[['brain_region_inhibited', 'condition']].drop_duplicates()
            conditions = sorted(
                f"{row.brain_region_inhibited}-{row.condition}"
                for row in condition_df.itertuples(index=False)
            )
        n_units = int(len(sub))
        n_bs = int((sub['BS_score'] == 1).sum()) if 'BS_score' in sub else 0
        rows.append({
            region_col: region,
            'n_units': n_units,
            'n_bs_units': n_bs,
            'bs_fraction': n_bs / n_units if n_units else np.nan,
            'n_insertions': int(sub['pid'].nunique()) if 'pid' in sub else np.nan,
            'n_mice': int(sub['mouse'].nunique()) if 'mouse' in sub else np.nan,
            'inhibited_region_conditions': ', '.join(conditions),
        })

    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values(['n_units', region_col], ascending=[False, True]).reset_index(drop=True)
    return out


def _allen_beryl_pair_inventory_table(df, include_unknown=False):
    """Count observed Allen/Beryl region pairs for midbrain units."""
    if 'Allenregion' not in df.columns or 'Berylregion' not in df.columns:
        return pd.DataFrame(columns=[
            'Allenregion', 'Berylregion', 'n_units', 'n_bs_units',
            'bs_fraction', 'n_insertions', 'n_mice',
        ])

    work = df.copy()
    work['Allenregion'] = work['Allenregion'].map(
        lambda value: _clean_region_acronym(value, include_unknown=include_unknown)
    )
    work['Berylregion'] = work['Berylregion'].map(
        lambda value: _clean_region_acronym(value, include_unknown=include_unknown)
    )
    work = work.dropna(subset=['Allenregion', 'Berylregion'])

    rows = []
    for (allen, beryl), sub in work.groupby(['Allenregion', 'Berylregion'], sort=True):
        n_units = int(len(sub))
        n_bs = int((sub['BS_score'] == 1).sum()) if 'BS_score' in sub else 0
        rows.append({
            'Allenregion': allen,
            'Berylregion': beryl,
            'n_units': n_units,
            'n_bs_units': n_bs,
            'bs_fraction': n_bs / n_units if n_units else np.nan,
            'n_insertions': int(sub['pid'].nunique()) if 'pid' in sub else np.nan,
            'n_mice': int(sub['mouse'].nunique()) if 'mouse' in sub else np.nan,
        })

    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values(['n_units', 'Allenregion', 'Berylregion'],
                              ascending=[False, True, True]).reset_index(drop=True)
    return out


def list_midbrain_recorded_regions(
        data_or_path='~/python/saved_figures/BS_all_insertions_ALL_LaserOnset.pkl',
        *,
        brain_region_inhibited=None,
        condition=None,
        pids=None,
        exclude_pids=None,
        bs_only=False,
        max_pval_empirical=None,
        include_unknown=False,
        save_csv=True,
        out_prefix='~/python/saved_figures/BS_midbrain_recorded_region_inventory',
        max_print_rows=40):
    """Print and optionally save all midbrain Allen/Beryl regions in a BS pickle.

    This is a convenience inventory for choosing `recorded_region=[...]` or
    `recorded_region_beryl=[...]` in the OPTIONS block above. By default it
    uses all broad-midbrain units in BS_all_insertions_ALL_LaserOnset.pkl. Pass
    `brain_region_inhibited='SNr'` and/or `condition='ipsi'` if you want the
    inventory to match a specific comparison.
    """
    if isinstance(data_or_path, (str, Path)):
        data = load_results(data_or_path)
        source = str(Path(data_or_path).expanduser())
    else:
        data = data_or_path
        source = '<loaded data>'

    df_mid, _ = filter_units(
        data,
        brain_region_inhibited=brain_region_inhibited,
        condition=condition,
        recorded_region='midbrain',
        bs_only=bs_only,
        max_pval_empirical=max_pval_empirical,
        pids=pids,
        exclude_pids=exclude_pids,
        verbose=False,
    )

    allen_table = _region_inventory_table(df_mid, 'Allenregion', include_unknown=include_unknown)
    beryl_table = _region_inventory_table(df_mid, 'Berylregion', include_unknown=include_unknown)
    pair_table = _allen_beryl_pair_inventory_table(df_mid, include_unknown=include_unknown)

    allen_regions = sorted(allen_table['Allenregion'].tolist()) if len(allen_table) else []
    beryl_regions = sorted(beryl_table['Berylregion'].tolist()) if len(beryl_table) else []

    print('\nMidbrain recorded-region inventory')
    print(f'  Source: {source}')
    print(f'  Units: {len(df_mid)}')
    if 'pid' in df_mid:
        print(f'  Insertions: {df_mid["pid"].nunique()}')
    if 'mouse' in df_mid:
        print(f'  Mice: {df_mid["mouse"].nunique()}')
    print('\nAllen regions, copy/paste for recorded_region:')
    print(f'  {allen_regions}')
    print('\nBeryl regions, copy/paste for recorded_region_beryl:')
    print(f'  {beryl_regions}')

    if len(allen_table):
        print('\nAllen region counts:')
        print(allen_table.head(max_print_rows).to_string(index=False))
    if len(beryl_table):
        print('\nBeryl region counts:')
        print(beryl_table.head(max_print_rows).to_string(index=False))

    saved = {}
    if save_csv:
        prefix = Path(out_prefix).expanduser()
        if prefix.suffix:
            prefix = prefix.with_suffix('')
        try:
            prefix.parent.mkdir(parents=True, exist_ok=True)
            saved['allen'] = prefix.parent / f'{prefix.name}_allen.csv'
            saved['beryl'] = prefix.parent / f'{prefix.name}_beryl.csv'
            saved['allen_beryl_pairs'] = prefix.parent / f'{prefix.name}_allen_beryl_pairs.csv'
            allen_table.to_csv(saved['allen'], index=False)
            beryl_table.to_csv(saved['beryl'], index=False)
            pair_table.to_csv(saved['allen_beryl_pairs'], index=False)
        except OSError as exc:
            print(f'\nCould not save CSVs to {prefix.parent}: {exc}')
            fallback = Path.cwd() / prefix.name
            print(f'Falling back to current directory: {fallback.parent}')
            saved['allen'] = fallback.parent / f'{fallback.name}_allen.csv'
            saved['beryl'] = fallback.parent / f'{fallback.name}_beryl.csv'
            saved['allen_beryl_pairs'] = fallback.parent / f'{fallback.name}_allen_beryl_pairs.csv'
            allen_table.to_csv(saved['allen'], index=False)
            beryl_table.to_csv(saved['beryl'], index=False)
            pair_table.to_csv(saved['allen_beryl_pairs'], index=False)
        print('\nSaved region inventory CSVs:')
        for label, path in saved.items():
            print(f'  {label}: {path}')

    return {
        'midbrain_units': df_mid,
        'allen_regions': allen_regions,
        'beryl_regions': beryl_regions,
        'allen_table': allen_table,
        'beryl_table': beryl_table,
        'allen_beryl_pair_table': pair_table,
        'saved': saved,
    }


# Quick-use snippet:
# from BS_postprocess import list_midbrain_recorded_regions
# inv = list_midbrain_recorded_regions()
if __name__ == '__main__':
    inv_snr_ipsi = list_midbrain_recorded_regions(
        brain_region_inhibited='SNr',
        condition='ipsi',
    )
