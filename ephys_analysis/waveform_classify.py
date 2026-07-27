"""
Axonal unit classification via waveform feature analysis
=========================================================
Classifies spike-sorted units as putative somatic or axonal based on
extracellular waveform shape, using the classical criterion:

  A unit is classified as axonal if BOTH:
    1. The peak (global max) occurs BEFORE the trough (global min)
    2. |peak amplitude| > |trough amplitude|  (PT ratio > 1)

Waveform input:
  This module consumes the IBL canonical waveform templates loaded via:

      waveforms = ssl.load_spike_sorting_object('waveforms')

  See: https://docs.internationalbrainlab.org/loading_examples/loading_spike_waveforms.html

  IMPORTANT NOTE ON SHAPE:
  `waveforms['templates']` has shape (n_clusters, n_channels, n_samples) —
  channels axis first, samples axis second. This matches the IBL/double_wiggle
  convention (also used by raw waveforms after spike averaging). The shape
  is auto-verified by treating the shorter of the two non-cluster axes as
  channels, since for Neuropixels n_channels (~32) < n_samples (~60-120).

Metrics computed per unit:
  - pt_ratio              : |peak| / |trough|, using GLOBAL extrema
  - peak_to_trough_ms     : absolute time between peak and trough (ms)
  - peak_before_trough    : bool, True if peak (global max) precedes trough
  - spatial_spread        : number of channels with amplitude > 25% of peak
                            channel max (auxiliary diagnostic)

Usage:
  from waveform_classify import classify_and_plot_axonal_units

  waveforms = ssl.load_spike_sorting_object('waveforms')
  axonal_IDs, metrics_df = classify_and_plot_axonal_units(
      thresholded_cluster_IDs, waveforms,
      save_path='figures/', prefix='my_session'
  )
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize


# ========================== CONFIGURATION ==================================

# Neuropixels AP band sampling rate (Hz)
SAMPLING_RATE_HZ = 30_000

# Number of baseline samples at the start of waveform to subtract
N_BASELINE_SAMPLES = 15

# Canonical IBL voltage scaling factor: raw * UV_SCALE -> microvolts
UV_SCALE = 1e6 / 80.0

# Fraction of peak-channel amplitude to define "active" for spatial spread
SPATIAL_SPREAD_FRACTION = 0.25

# Classification threshold (PT ratio above which a peak-first unit is axonal)
PT_RATIO_THRESHOLD = 1.0

# Waveform grid
WF_GRID_COLS = 12
PLOT_WF_GRID = True


# ========================== CORE FUNCTIONS =================================


def _extract_cluster_template_uv(templates, cluster_id):
    """
    Extract a single cluster's template as a 2D array of shape
    (n_samples, n_channels) in microvolts.

    Auto-detects axis orientation by treating the shorter of the two
    non-cluster axes as channels (for Neuropixels, n_channels ~32 is
    much smaller than n_samples ~60-120). Returns None if cluster_id
    is out of range or the template is empty/all-zero.
    """
    if cluster_id >= templates.shape[0]:
        return None

    tmpl = templates[cluster_id]  # shape (?, ?)
    if not np.any(np.isfinite(tmpl)) or np.all(tmpl == 0):
        return None

    # Channels axis is the shorter of the two
    if tmpl.shape[0] < tmpl.shape[1]:
        # (n_channels, n_samples) — transpose to (n_samples, n_channels)
        tmpl_2d = tmpl.T
    else:
        # Already (n_samples, n_channels)
        tmpl_2d = tmpl

    return tmpl_2d * UV_SCALE


def compute_waveform_metrics(wf_avg, wf_avg_all_channels=None,
                             sampling_rate_hz=SAMPLING_RATE_HZ):
    """
    Compute waveform shape metrics for a single unit using GLOBAL extrema.

    Parameters
    ----------
    wf_avg : 1D array, shape (n_timepoints,)
        Baseline-subtracted mean waveform on the peak channel (in µV).
    wf_avg_all_channels : 2D array, shape (n_timepoints, n_channels), optional
        Baseline-subtracted mean waveforms across all available channels.
    sampling_rate_hz : float

    Returns
    -------
    metrics : dict
    """
    sample_period_ms = 1000.0 / sampling_rate_hz

    peak_idx = int(np.argmax(wf_avg))
    trough_idx = int(np.argmin(wf_avg))
    peak_amp = float(wf_avg[peak_idx])
    trough_amp = float(wf_avg[trough_idx])

    if abs(trough_amp) > 0:
        pt_ratio = abs(peak_amp) / abs(trough_amp)
    else:
        pt_ratio = np.nan

    peak_to_trough_ms = abs(peak_idx - trough_idx) * sample_period_ms
    peak_before_trough = peak_idx < trough_idx

    spatial_spread = np.nan
    if wf_avg_all_channels is not None and wf_avg_all_channels.ndim == 2:
        peak_chan_max = np.max(np.abs(wf_avg))
        if peak_chan_max > 0:
            threshold = SPATIAL_SPREAD_FRACTION * peak_chan_max
            chan_max_amps = np.max(np.abs(wf_avg_all_channels), axis=0)
            spatial_spread = int(np.sum(chan_max_amps > threshold))

    return {
        'pt_ratio': float(pt_ratio),
        'peak_to_trough_ms': float(peak_to_trough_ms),
        'peak_before_trough': bool(peak_before_trough),
        'spatial_spread': spatial_spread,
        'peak_idx': peak_idx,
        'trough_idx': trough_idx,
        'peak_amplitude': peak_amp,
        'trough_amplitude': trough_amp,
    }


def compute_metrics_for_population(cluster_ids, waveforms):
    """
    Compute waveform metrics for all units in a population using IBL templates.

    Parameters
    ----------
    cluster_ids : array-like of int
    waveforms : dict
        Output of ssl.load_spike_sorting_object('waveforms'). Must contain
        key 'templates' with shape (n_clusters, n_channels, n_samples).

    Returns
    -------
    df : pd.DataFrame
    waveforms_dict : dict cluster_id -> baseline-subtracted peak-channel
                     mean waveform (1D, in µV)
    skipped_ids : list of cluster IDs skipped due to missing/empty template
    """
    rows = []
    skipped = []
    waveforms_dict = {}

    templates = waveforms['templates']

    # Sanity check: print detected orientation on first call
    if templates.ndim != 3:
        raise ValueError(f"Expected 3D templates array, got shape {templates.shape}")

    sample_template = _extract_cluster_template_uv(templates, int(cluster_ids[0]))
    if sample_template is not None:
        print(f'  Templates shape: {templates.shape} '
              f'-> per-cluster waveform: {sample_template.shape} (n_samples, n_channels)')

    for j in cluster_ids:
        wf_template_uv = _extract_cluster_template_uv(templates, int(j))
        if wf_template_uv is None:
            skipped.append(int(j))
            continue

        # Find peak channel by max absolute amplitude across channels
        chan_max_amps = np.max(np.abs(wf_template_uv), axis=0)
        peak_chan = int(np.argmax(chan_max_amps))

        # Mean waveform on peak channel, baseline-subtracted
        wf_avg_peak = wf_template_uv[:, peak_chan]
        wf_avg_peak = wf_avg_peak - np.mean(wf_avg_peak[:N_BASELINE_SAMPLES])

        # All channels, baseline-subtracted per channel
        wf_avg_all = wf_template_uv.copy()
        for ch in range(wf_avg_all.shape[1]):
            wf_avg_all[:, ch] -= np.mean(wf_avg_all[:N_BASELINE_SAMPLES, ch])

        metrics = compute_waveform_metrics(wf_avg_peak, wf_avg_all)
        metrics['cluster_id'] = int(j)
        metrics['peak_channel'] = peak_chan
        rows.append(metrics)
        waveforms_dict[int(j)] = wf_avg_peak

    df = pd.DataFrame(rows)
    return df, waveforms_dict, skipped


def classify_axonal_classical(df, pt_ratio_threshold=PT_RATIO_THRESHOLD):
    """
    Classical axonal criterion:
        peak_before_trough AND pt_ratio > pt_ratio_threshold
    """
    valid = np.isfinite(df['pt_ratio'])
    df['is_axonal'] = False
    df.loc[valid, 'is_axonal'] = (
        df.loc[valid, 'peak_before_trough'] &
        (df.loc[valid, 'pt_ratio'] > pt_ratio_threshold)
    )

    n_axonal = df['is_axonal'].sum()
    n_valid = valid.sum()
    print(f'  Classical axonal criterion (peak BEFORE trough AND '
          f'PT ratio > {pt_ratio_threshold}): '
          f'{n_valid - n_axonal} somatic, {n_axonal} axonal '
          f'(of {n_valid} valid units)')

    return df


# ========================== PLOTTING =======================================


def plot_feature_space(df, save_path=None, prefix='', title=None,
                       pt_ratio_threshold=PT_RATIO_THRESHOLD):
    """
    Diagnostic scatter plot using a signed peak-to-trough latency.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    valid = np.isfinite(df['pt_ratio']) & np.isfinite(df['peak_to_trough_ms'])
    df_valid = df[valid].copy()

    df_valid['signed_latency_ms'] = np.where(
        df_valid['peak_before_trough'],
        df_valid['peak_to_trough_ms'],
        -df_valid['peak_to_trough_ms']
    )

    # --- Panel 1: Scatter coloured by spatial spread ---
    ax = axes[0]
    spread = df_valid['spatial_spread'].values
    has_spread = np.isfinite(spread)

    if has_spread.any():
        sc = ax.scatter(
            df_valid.loc[has_spread, 'signed_latency_ms'].values,
            df_valid.loc[has_spread, 'pt_ratio'].values,
            c=spread[has_spread],
            cmap='plasma', s=30, alpha=0.7, edgecolors='k', linewidths=0.3,
            norm=Normalize(vmin=np.nanmin(spread[has_spread]),
                           vmax=np.nanmax(spread[has_spread]))
        )
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label('Spatial spread (n channels)', fontsize=10)

    if (~has_spread).any():
        ax.scatter(
            df_valid.loc[~has_spread, 'signed_latency_ms'].values,
            df_valid.loc[~has_spread, 'pt_ratio'].values,
            c='grey', s=20, alpha=0.4, edgecolors='k', linewidths=0.2,
        )

    ax.axhline(pt_ratio_threshold, color='red', linewidth=1, linestyle='--',
               alpha=0.7, label=f'PT = {pt_ratio_threshold}')
    ax.axvline(0, color='blue', linewidth=1, linestyle='--',
               alpha=0.7, label='Peak/trough order')
    ax.set_xlabel('Peak-to-trough latency (ms)\n← peak after trough  |  peak before trough →',
                  fontsize=10)
    ax.set_ylabel('Peak / Trough ratio', fontsize=11)
    ax.set_title('Feature space (colour = spatial spread)')
    ax.legend(fontsize=8, loc='upper right')

    # --- Panel 2: Classification result ---
    ax = axes[1]

    somatic = df_valid[~df_valid['is_axonal']]
    axonal = df_valid[df_valid['is_axonal']]

    ax.scatter(somatic['signed_latency_ms'].values, somatic['pt_ratio'].values,
               c='steelblue', s=30, alpha=0.7, edgecolors='k',
               linewidths=0.3, label=f'Somatic (n={len(somatic)})')
    ax.scatter(axonal['signed_latency_ms'].values, axonal['pt_ratio'].values,
               c='tomato', s=40, alpha=0.8, edgecolors='k',
               linewidths=0.3, label=f'Axonal (n={len(axonal)})')

    x_lim = ax.get_xlim()
    y_lim = ax.get_ylim()
    ax.fill_between(
        [0, max(x_lim[1], 0.01)], pt_ratio_threshold, y_lim[1],
        color='tomato', alpha=0.07, zorder=0
    )
    ax.axhline(pt_ratio_threshold, color='red', linewidth=1, linestyle='--', alpha=0.5)
    ax.axvline(0, color='blue', linewidth=1, linestyle='--', alpha=0.5)
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)

    ax.set_xlabel('Peak-to-trough latency (ms)\n← peak after trough  |  peak before trough →',
                  fontsize=10)
    ax.set_ylabel('Peak / Trough ratio', fontsize=11)
    ax.set_title('Classical criterion classification')
    ax.legend(fontsize=9, loc='upper right')

    if title:
        fig.suptitle(title, fontsize=13, fontweight='bold')

    plt.tight_layout()

    if save_path:
        fname = f'{save_path}/{prefix}_axonal_classification.png'
        fig.savefig(fname, dpi=200, bbox_inches='tight')
        print(f'  Saved feature space plot: {fname}')
        plt.close()
    else:
        plt.show()
        plt.close()

    return fig


def plot_waveform_grid(df, waveforms_dict, save_path=None, prefix='',
                       title=None, n_cols=WF_GRID_COLS,
                       sampling_rate_hz=SAMPLING_RATE_HZ):
    """
    Grid of mean waveforms (µV), sorted by classification then PT ratio.
    """
    valid_mask = (
        df['cluster_id'].isin(waveforms_dict.keys()) &
        np.isfinite(df['pt_ratio']) &
        np.isfinite(df['peak_to_trough_ms'])
    )
    df_plot = df[valid_mask].copy()

    if len(df_plot) == 0:
        print('  No valid waveforms to plot.')
        return None

    df_plot['sort_key'] = (
        df_plot['is_axonal'].astype(int) * 1000 + df_plot['pt_ratio'].fillna(0)
    )
    df_plot = df_plot.sort_values('sort_key', ascending=False).reset_index(drop=True)

    n_units = len(df_plot)
    n_rows = int(np.ceil(n_units / n_cols))

    sample_period_ms = 1000.0 / sampling_rate_hz

    fig = plt.figure(figsize=(n_cols * 1.6, n_rows * 1.5))
    gs = gridspec.GridSpec(n_rows, n_cols, hspace=0.6, wspace=0.3)

    if title:
        fig.suptitle(title, fontsize=13, fontweight='bold', y=1.01)

    for i, (_, row) in enumerate(df_plot.iterrows()):
        r = i // n_cols
        c = i % n_cols
        ax = fig.add_subplot(gs[r, c])

        cid = int(row['cluster_id'])
        wf = waveforms_dict[cid]
        time_ms = np.arange(len(wf)) * sample_period_ms

        is_ax = row['is_axonal']
        color = 'tomato' if is_ax else 'steelblue'
        label_str = 'AX' if is_ax else 'SM'

        ax.plot(time_ms, wf, color=color, linewidth=1.0)

        trough_idx = int(row['trough_idx'])
        peak_idx = int(row['peak_idx'])
        ax.plot(time_ms[trough_idx], wf[trough_idx], 'v', color='black',
                markersize=4, zorder=5)
        ax.plot(time_ms[peak_idx], wf[peak_idx], '^', color='black',
                markersize=4, zorder=5)

        pt = row['pt_ratio']
        lat = row['peak_to_trough_ms']
        sign = '+' if row['peak_before_trough'] else '-'
        ax.set_title(f'{label_str} #{cid}\nPT={pt:.2f}  Δt={sign}{lat:.2f}',
                     fontsize=5.5, color=color, fontweight='bold', pad=2)

        for spine in ax.spines.values():
            spine.set_color(color)
            spine.set_linewidth(1.5)

        ax.set_xticks([])
        ax.set_yticks([])
        ax.margins(x=0.05, y=0.15)

    for i in range(n_units, n_rows * n_cols):
        r = i // n_cols
        c = i % n_cols
        ax = fig.add_subplot(gs[r, c])
        ax.set_visible(False)

    plt.tight_layout()

    if save_path:
        fname = f'{save_path}/{prefix}_waveform_grid.png'
        fig.savefig(fname, dpi=200, bbox_inches='tight')
        print(f'  Saved waveform grid: {fname}')
        plt.close()
    else:
        plt.show()
        plt.close()

    return fig


# ========================== PIPELINE INTEGRATION ===========================


def classify_and_plot_axonal_units(cluster_ids, waveforms,
                                    save_path=None, prefix='',
                                    title=None,
                                    pt_ratio_threshold=PT_RATIO_THRESHOLD,
                                    plot_wf_grid=PLOT_WF_GRID):
    """
    End-to-end axonal classification using IBL canonical waveform templates.

    Parameters
    ----------
    cluster_ids : array-like of int
    waveforms : dict from ssl.load_spike_sorting_object('waveforms')
    save_path : str or Path, optional
    prefix : str, optional
    title : str, optional
    pt_ratio_threshold : float
    plot_wf_grid : bool

    Returns
    -------
    axonal_IDs : np.ndarray of int
    metrics_df : pd.DataFrame
    """
    print('  Computing waveform metrics...')
    df, waveforms_dict, skipped = compute_metrics_for_population(
        cluster_ids, waveforms
    )

    if len(df) == 0:
        print('  No units with valid templates.')
        return np.array([], dtype=int), pd.DataFrame()

    if len(skipped) > 0:
        print(f'  Skipped {len(skipped)} clusters (no valid template)')

    df = classify_axonal_classical(df, pt_ratio_threshold=pt_ratio_threshold)

    plot_feature_space(df, save_path=save_path, prefix=prefix,
                       title=title, pt_ratio_threshold=pt_ratio_threshold)

    if plot_wf_grid:
        plot_waveform_grid(df, waveforms_dict, save_path=save_path,
                           prefix=prefix,
                           title=(title or '') + ' — Waveform grid')

    axonal_IDs = df.loc[df['is_axonal'], 'cluster_id'].values.astype(int)
    return axonal_IDs, df
