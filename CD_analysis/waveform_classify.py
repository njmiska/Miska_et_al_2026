"""
Axonal unit classification via waveform feature analysis
=========================================================
Classifies spike-sorted units as putative somatic or axonal based on
extracellular waveform shape features, using simple threshold criteria
in a 2D feature space.

Features:
  1. Peak-to-trough ratio (PT ratio):
       abs(positive peak) / abs(negative trough) on the peak channel.
       Axonal waveforms tend to have PT ratio closer to or above 1,
       while somatic waveforms are trough-dominant (PT ratio << 1).

  2. Trough-to-peak duration (ms):
       Time from the negative trough to the subsequent positive peak.
       Axonal spikes are narrow (< ~0.3 ms), somatic spikes are broader.
       Reference: Barthó et al. (2004) J Neurophysiol.

  3. Spatial spread (auxiliary diagnostic metric):
       Number of channels on which the waveform amplitude exceeds a
       fraction of the peak-channel amplitude. Axonal spikes propagate
       across many channels; somatic spikes are spatially localized.

Classification rule (default):
  A unit is classified as axonal if:
    trough_to_peak_ms < T2P_THRESHOLD  AND  pt_ratio > PT_RATIO_THRESHOLD

Usage:
  from waveform_classify import classify_and_plot_axonal_units

  axonal_IDs, metrics_df = classify_and_plot_axonal_units(
      thresholded_cluster_IDs, spike_wfs, wf_clusterIDs,
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

# Minimum number of spikes per cluster required for waveform analysis
MIN_SPIKES_FOR_WF = 5

# Fraction of peak-channel amplitude to define "active" for spatial spread
SPATIAL_SPREAD_FRACTION = 0.25

# Classification thresholds
T2P_THRESHOLD = 0.4     # ms — units with T2P below this are narrow (axonal-like)
PT_RATIO_THRESHOLD = 0.5  # units with PT ratio above this are positive-dominant (axonal-like)

# Waveform grid
WF_GRID_COLS = 12         # Number of columns in the waveform grid figure
PLOT_WF_GRID = True       # Whether to generate the waveform grid figure


# ========================== CORE FUNCTIONS =================================


def compute_waveform_metrics(wf_avg, wf_avg_all_channels=None,
                             sampling_rate_hz=SAMPLING_RATE_HZ):
    """
    Compute waveform shape metrics for a single unit.

    Parameters
    ----------
    wf_avg : 1D array, shape (n_timepoints,)
        Baseline-subtracted mean waveform on the peak channel.
    wf_avg_all_channels : 2D array, shape (n_timepoints, n_channels), optional
        Baseline-subtracted mean waveforms across all available channels.
        If provided, spatial spread is computed.
    sampling_rate_hz : float
        Sampling rate in Hz (default: 30000 for Neuropixels).

    Returns
    -------
    metrics : dict
    """
    sample_period_ms = 1000.0 / sampling_rate_hz

    # Find trough (most negative point)
    trough_idx = np.argmin(wf_avg)
    trough_amp = wf_avg[trough_idx]

    # Find peak AFTER trough (repolarization peak)
    post_trough = wf_avg[trough_idx:]
    if len(post_trough) > 1:
        peak_idx_relative = np.argmax(post_trough)
        peak_idx = trough_idx + peak_idx_relative
        peak_amp = wf_avg[peak_idx]
    else:
        peak_idx = trough_idx
        peak_amp = trough_amp

    # PT ratio
    if abs(trough_amp) > 0:
        pt_ratio = abs(peak_amp) / abs(trough_amp)
    else:
        pt_ratio = np.nan

    # Trough-to-peak duration
    trough_to_peak_ms = (peak_idx - trough_idx) * sample_period_ms

    # Spatial spread
    spatial_spread = np.nan
    if wf_avg_all_channels is not None and wf_avg_all_channels.ndim == 2:
        peak_chan_max = np.max(np.abs(wf_avg))
        if peak_chan_max > 0:
            threshold = SPATIAL_SPREAD_FRACTION * peak_chan_max
            chan_max_amps = np.max(np.abs(wf_avg_all_channels), axis=0)
            spatial_spread = int(np.sum(chan_max_amps > threshold))

    return {
        'pt_ratio': float(pt_ratio),
        'trough_to_peak_ms': float(trough_to_peak_ms),
        'spatial_spread': spatial_spread,
        'trough_idx': int(trough_idx),
        'peak_idx': int(peak_idx),
        'trough_amplitude': float(trough_amp),
        'peak_amplitude': float(peak_amp),
    }


def compute_metrics_for_population(cluster_ids, spike_wfs, wf_clusterIDs,
                                   min_spikes=MIN_SPIKES_FOR_WF):
    """
    Compute waveform metrics for all units in a population.

    Returns
    -------
    df : pd.DataFrame with one row per cluster
    waveforms_dict : dict mapping cluster_id -> baseline-subtracted mean waveform (1D)
    skipped_ids : list of cluster IDs skipped
    """
    rows = []
    skipped = []
    waveforms_dict = {}

    for j in cluster_ids:
        wf_idx = np.where(wf_clusterIDs == j)[0]
        if len(wf_idx) < min_spikes:
            skipped.append(j)
            continue

        wfs = spike_wfs['waveforms'][wf_idx, :, :]

        # Mean waveform on peak channel (channel 0), baseline-subtracted
        wf_avg_peak = np.mean(wfs[:, :, 0], axis=0)
        wf_avg_peak = wf_avg_peak - np.mean(wf_avg_peak[:N_BASELINE_SAMPLES])

        # Mean waveforms across all channels, baseline-subtracted per channel
        wf_avg_all = np.mean(wfs, axis=0)  # (n_timepoints, n_channels)
        for ch in range(wf_avg_all.shape[1]):
            wf_avg_all[:, ch] -= np.mean(wf_avg_all[:N_BASELINE_SAMPLES, ch])

        metrics = compute_waveform_metrics(wf_avg_peak, wf_avg_all)
        metrics['cluster_id'] = int(j)
        rows.append(metrics)
        waveforms_dict[int(j)] = wf_avg_peak

    df = pd.DataFrame(rows)
    return df, waveforms_dict, skipped


def classify_axonal_threshold(df, t2p_threshold=T2P_THRESHOLD,
                               pt_ratio_threshold=PT_RATIO_THRESHOLD):
    """
    Classify units as axonal using simple thresholds.

    Rule: axonal if trough_to_peak_ms < t2p_threshold AND pt_ratio > pt_ratio_threshold

    Parameters
    ----------
    df : pd.DataFrame with 'pt_ratio' and 'trough_to_peak_ms' columns.
    t2p_threshold : float, ms
    pt_ratio_threshold : float

    Returns
    -------
    df : modified in place with 'is_axonal' column
    """
    valid = np.isfinite(df['pt_ratio']) & np.isfinite(df['trough_to_peak_ms'])
    df['is_axonal'] = False
    df.loc[valid, 'is_axonal'] = (
        (df.loc[valid, 'trough_to_peak_ms'] < t2p_threshold) &
        (df.loc[valid, 'pt_ratio'] > pt_ratio_threshold)
    )

    n_axonal = df['is_axonal'].sum()
    n_valid = valid.sum()
    print(f'  Threshold classification (T2P < {t2p_threshold} ms, '
          f'PT > {pt_ratio_threshold}): '
          f'{n_valid - n_axonal} somatic, {n_axonal} axonal '
          f'(of {n_valid} valid units)')

    return df


# ========================== PLOTTING =======================================


def plot_feature_space(df, save_path=None, prefix='', title=None,
                       t2p_threshold=T2P_THRESHOLD,
                       pt_ratio_threshold=PT_RATIO_THRESHOLD):
    """
    Diagnostic scatter plot: PT ratio vs trough-to-peak duration,
    coloured by spatial spread, with threshold lines.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    valid = np.isfinite(df['pt_ratio']) & np.isfinite(df['trough_to_peak_ms'])
    df_valid = df[valid]

    # --- Panel 1: Scatter coloured by spatial spread ---
    ax = axes[0]
    spread = df_valid['spatial_spread'].values
    has_spread = np.isfinite(spread)

    if has_spread.any():
        sc = ax.scatter(
            df_valid.loc[has_spread, 'trough_to_peak_ms'].values,
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
            df_valid.loc[~has_spread, 'trough_to_peak_ms'].values,
            df_valid.loc[~has_spread, 'pt_ratio'].values,
            c='grey', s=20, alpha=0.4, edgecolors='k', linewidths=0.2,
        )

    ax.axhline(pt_ratio_threshold, color='red', linewidth=1, linestyle='--',
               alpha=0.7, label=f'PT = {pt_ratio_threshold}')
    ax.axvline(t2p_threshold, color='blue', linewidth=1, linestyle='--',
               alpha=0.7, label=f'T2P = {t2p_threshold} ms')
    ax.set_xlabel('Trough-to-peak duration (ms)', fontsize=11)
    ax.set_ylabel('Peak / Trough ratio', fontsize=11)
    ax.set_title('Feature space (colour = spatial spread)')
    ax.legend(fontsize=8, loc='upper right')

    # --- Panel 2: Classification result ---
    ax = axes[1]

    somatic = df_valid[~df_valid['is_axonal']]
    axonal = df_valid[df_valid['is_axonal']]

    ax.scatter(somatic['trough_to_peak_ms'].values, somatic['pt_ratio'].values,
               c='steelblue', s=30, alpha=0.7, edgecolors='k',
               linewidths=0.3, label=f'Somatic (n={len(somatic)})')
    ax.scatter(axonal['trough_to_peak_ms'].values, axonal['pt_ratio'].values,
               c='tomato', s=40, alpha=0.8, edgecolors='k',
               linewidths=0.3, label=f'Axonal (n={len(axonal)})')

    # Shade the axonal region
    x_lim = ax.get_xlim()
    y_lim = ax.get_ylim()
    ax.fill_between(
        [0, t2p_threshold], pt_ratio_threshold, y_lim[1],
        color='tomato', alpha=0.07, zorder=0
    )
    ax.axhline(pt_ratio_threshold, color='red', linewidth=1, linestyle='--', alpha=0.5)
    ax.axvline(t2p_threshold, color='blue', linewidth=1, linestyle='--', alpha=0.5)
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)

    ax.set_xlabel('Trough-to-peak duration (ms)', fontsize=11)
    ax.set_ylabel('Peak / Trough ratio', fontsize=11)
    ax.set_title('Threshold classification')
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
    Plot a grid of individual unit waveforms, annotated with metrics,
    sorted by classification (axonal first) then by PT ratio.

    Each subplot shows the mean waveform on the peak channel, with:
      - Trough and peak marked
      - PT ratio and T2P duration annotated
      - Border colour: red = axonal, blue = somatic
    """
    # Filter to units with waveforms and valid metrics
    valid_mask = (
        df['cluster_id'].isin(waveforms_dict.keys()) &
        np.isfinite(df['pt_ratio']) &
        np.isfinite(df['trough_to_peak_ms'])
    )
    df_plot = df[valid_mask].copy()

    if len(df_plot) == 0:
        print('  No valid waveforms to plot.')
        return None

    # Sort: axonal first (by PT ratio desc), then somatic (by T2P asc)
    df_plot['sort_key'] = df_plot['is_axonal'].astype(int) * 1000 - df_plot['trough_to_peak_ms']
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

        # Plot waveform
        ax.plot(time_ms, wf, color=color, linewidth=1.0)

        # Mark trough and peak
        trough_idx = int(row['trough_idx'])
        peak_idx = int(row['peak_idx'])
        ax.plot(time_ms[trough_idx], wf[trough_idx], 'v', color='black',
                markersize=3, zorder=5)
        ax.plot(time_ms[peak_idx], wf[peak_idx], '^', color='black',
                markersize=3, zorder=5)

        # Annotate
        pt = row['pt_ratio']
        t2p = row['trough_to_peak_ms']
        ax.set_title(f'{label_str} #{cid}\nPT={pt:.2f} T2P={t2p:.2f}',
                     fontsize=5.5, color=color, fontweight='bold', pad=2)

        # Border colour
        for spine in ax.spines.values():
            spine.set_color(color)
            spine.set_linewidth(1.5)

        ax.set_xticks([])
        ax.set_yticks([])
        ax.margins(x=0.05, y=0.15)

    # Hide empty subplots
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


def classify_and_plot_axonal_units(cluster_ids, spike_wfs, wf_clusterIDs,
                                    save_path=None, prefix='',
                                    title=None, min_spikes=MIN_SPIKES_FOR_WF,
                                    t2p_threshold=T2P_THRESHOLD,
                                    pt_ratio_threshold=PT_RATIO_THRESHOLD,
                                    plot_wf_grid=PLOT_WF_GRID):
    """
    End-to-end axonal classification: compute metrics, classify, plot, return IDs.

    Parameters
    ----------
    cluster_ids : array-like of int
        Cluster IDs to analyse.
    spike_wfs : dict-like
        Must have key 'waveforms' with shape (n_spikes, n_timepoints, n_channels).
    wf_clusterIDs : 1D array
        Cluster ID for each spike in spike_wfs.
    save_path : str or Path, optional
        Directory to save diagnostic plots. None = show interactively.
    prefix : str, optional
        Filename prefix for saved plots.
    title : str, optional
        Title for plots.
    min_spikes : int
        Minimum spikes required per cluster.
    t2p_threshold : float
        Trough-to-peak threshold in ms (units below this are axonal candidates).
    pt_ratio_threshold : float
        PT ratio threshold (units above this are axonal candidates).
    plot_wf_grid : bool
        Whether to generate the waveform grid figure.

    Returns
    -------
    axonal_IDs : np.ndarray of int
        Cluster IDs classified as axonal.
    metrics_df : pd.DataFrame
        Full metrics table with classification results.
    """
    print('  Computing waveform metrics...')
    df, waveforms_dict, skipped = compute_metrics_for_population(
        cluster_ids, spike_wfs, wf_clusterIDs, min_spikes=min_spikes
    )

    if len(df) == 0:
        print('  No units with sufficient waveforms.')
        return np.array([], dtype=int), pd.DataFrame()

    if len(skipped) > 0:
        print(f'  Skipped {len(skipped)} clusters (< {min_spikes} spikes)')

    # Classify
    df = classify_axonal_threshold(df, t2p_threshold=t2p_threshold,
                                    pt_ratio_threshold=pt_ratio_threshold)

    # Feature space scatter plot
    plot_feature_space(df, save_path=save_path, prefix=prefix,
                       title=title, t2p_threshold=t2p_threshold,
                       pt_ratio_threshold=pt_ratio_threshold)

    # Waveform grid
    if plot_wf_grid:
        plot_waveform_grid(df, waveforms_dict, save_path=save_path,
                           prefix=prefix,
                           title=(title or '') + ' — Waveform grid')

    axonal_IDs = df.loc[df['is_axonal'], 'cluster_id'].values.astype(int)
    return axonal_IDs, df
