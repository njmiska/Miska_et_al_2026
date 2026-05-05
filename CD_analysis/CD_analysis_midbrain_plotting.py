import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import ttest_rel

# --- CONFIGURATION ---
figures_path = '/Users/natemiska/Desktop/cd_figures'
with open(figures_path + '/SNr.pkl', 'rb') as f:
    pop_data = pickle.load(f)


# Colors
COLOR_CTRL = 'black'
COLOR_OPTO = 'blue'  # Requested Blue

# Quick approximation using your existing 'res' object if X is missing:
# You can't perfectly reconstruct X_orth from 'res' alone. 
# You would need to add this calculation to your main pipeline loop.

# --- PLOT 1: Grand Average Separation Traces ---
def get_group_stats(data_list, group_name, metric_key_path):
    collected_traces = []
    common_time = data_list[0]['time']
    for session in data_list:
        if session['hemisphere'] != group_name: continue
        val = session
        for k in metric_key_path: val = val[k]
        collected_traces.append(val)
    
    if not collected_traces: return None, None, None
    arr = np.array(collected_traces)
    return common_time, np.nanmean(arr, axis=0), np.nanstd(arr, axis=0)/np.sqrt(arr.shape[0])

def plot_shaded(ax, x, mean, sem, color, label):
    ax.plot(x, mean, color=color, label=label, lw=2)
    ax.fill_between(x, mean-sem, mean+sem, color=color, alpha=0.2)

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
conditions = ['Ipsi', 'Contra']

for i, hemi in enumerate(conditions):
    ax = axes[i]
    time, mu_c, se_c = get_group_stats(pop_data, hemi, ['separation', 'ctrl'])
    _,    mu_o, se_o = get_group_stats(pop_data, hemi, ['separation', 'opto'])
    
    if time is None: continue

    plot_shaded(ax, time, mu_c, se_c, COLOR_CTRL, 'Control')
    plot_shaded(ax, time, mu_o, se_o, COLOR_OPTO, 'SNr Inhibition')
    
    ax.set_title(f"{hemi}lateral SNr Inhibition")
    ax.set_xlabel("Time from Laser Onset (s)")
    ax.axvline(0, color='k', linestyle='--', alpha=0.5)
    ax.axhline(0, color='k', linewidth=0.5)

axes[0].set_ylabel("CD Separation (Block L - Block R)")
axes[0].legend()
plt.tight_layout()
plt.savefig(figures_path + '/GrandAverage_Separation_Blue.png')
plt.show()

# Define the integration window (in seconds)
WINDOW_START = 0.1
WINDOW_END   = 0.5

# --- 1. EXTRACT DATA (Windowed Average) ---
stats_data = []

for session in pop_data:
    # Find indices corresponding to the window
    t = session['time']
    mask = (t >= WINDOW_START) & (t <= WINDOW_END)
    
    if np.sum(mask) == 0: continue # Skip if no timepoints found
    
    # Average the separation traces over this window
    sep_ctrl = np.nanmean(session['separation']['ctrl'][mask])
    sep_opto = np.nanmean(session['separation']['opto'][mask])
    
    stats_data.append({
        'PID': session['pid'],
        'Hemisphere': session['hemisphere'],
        'Condition': 'Control',
        'Separation': sep_ctrl
    })
    stats_data.append({
        'PID': session['pid'],
        'Hemisphere': session['hemisphere'],
        'Condition': 'Opto',
        'Separation': sep_opto
    })

df = pd.DataFrame(stats_data)
df = df[df['Hemisphere'].isin(['Ipsi', 'Contra'])]

# --- 2. PLOTTING ---
plt.figure(figsize=(7, 6))

# A. The Bar Plot (Averages)
# We save the axes object 'ax' to draw lines on it later
ax = sns.barplot(
    data=df, x='Hemisphere', y='Separation', hue='Condition',
    palette={'Control': 'gray', 'Opto': 'dodgerblue'},
    errorbar='se', capsize=0.1, alpha=0.9, err_kws={'linewidth': 2}
)

# B. The Paired Lines (Red Spaghetti)
# This requires knowing the X-coordinates of the bars. 
# In a 2-hue barplot, the centers are usually at: [Tick - width/2, Tick + width/2]
# Default width is 0.8, so offset is +/- 0.2.
bar_width = 0.4 # Approx offset
x_centers = {'Ipsi': 0, 'Contra': 1}
offsets   = {'Control': -0.2, 'Opto': 0.2}

# Loop through each PID to draw lines
pids = df['PID'].unique()
for pid in pids:
    subset = df[df['PID'] == pid]
    if len(subset) != 2: continue # Skip if missing one condition
    
    hemi = subset['Hemisphere'].iloc[0]
    
    # Get Y values
    y_ctrl = subset[subset['Condition'] == 'Control']['Separation'].values[0]
    y_opto = subset[subset['Condition'] == 'Opto']['Separation'].values[0]
    
    # Calculate X coordinates
    x_ctrl = x_centers[hemi] + offsets['Control']
    x_opto = x_centers[hemi] + offsets['Opto']
    
    # Plot the Line
    plt.plot([x_ctrl, x_opto], [y_ctrl, y_opto], color='red', linewidth=1, alpha=0.5, zorder=10)
    # Plot the dots on top of the lines
    plt.scatter([x_ctrl, x_opto], [y_ctrl, y_opto], color='navy', s=20, zorder=11, edgecolors='white', linewidth=0.5)

# Decoration
plt.axhline(0, color='k', linewidth=0.5)
plt.ylabel(f"Mean CD Separation ({WINDOW_START}-{WINDOW_END}s)")
plt.title(f"Effect of SNr Inhibition (Integrated {WINDOW_START}-{WINDOW_END}s)")
plt.legend(loc='upper right')
plt.tight_layout()

plt.savefig(figures_path + '/Quantification_PairedLines.png')
plt.show()

# --- 3. STATISTICS ---
print(f"--- Paired Statistics ({WINDOW_START}-{WINDOW_END}s) ---")
for hemi in ['Ipsi', 'Contra']:
    subset = df[df['Hemisphere'] == hemi]
    pids_hemi = subset['PID'].unique()
    
    # Align pairs
    pairs = []
    for pid in pids_hemi:
        c = subset[(subset['Condition']=='Control') & (subset['PID']==pid)]['Separation'].values
        o = subset[(subset['Condition']=='Opto') & (subset['PID']==pid)]['Separation'].values
        if len(c)==1 and len(o)==1:
            pairs.append((c[0], o[0]))
    
    if not pairs: continue
    
    c_arr = np.array([p[0] for p in pairs])
    o_arr = np.array([p[1] for p in pairs])
    
    t, p = ttest_rel(c_arr, o_arr)
    print(f"{hemi} (N={len(c_arr)}): Control={c_arr.mean():.3f}, Opto={o_arr.mean():.3f}, p={p:.4f}")

# import pickle
# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
# import seaborn as sns
# from scipy.stats import ttest_rel

# # --- SETUP ---
# figures_path = '/Users/natemiska/Documents/Lab_Data/LabMeetingDec2025/CD_analysis'
# with open(figures_path + '/CD_population_results_IBLGOOD_final_ZI.pkl', 'rb') as f:
#     pop_data = pickle.load(f)


# # --- PLOT 2: Updated Quantification (Grouped Bars) ---
# target_time = 0.2
# long_form_data = []

# for session in pop_data:
#     t_idx = np.argmin(np.abs(session['time'] - target_time))
    
#     # Append TWO rows per session (Long format for Seaborn)
#     # Row 1: Control
#     long_form_data.append({
#         'PID': session['pid'],
#         'Hemisphere': session['hemisphere'],
#         'Condition': 'Control',
#         'Separation': session['separation']['ctrl'][t_idx]
#     })
#     # Row 2: Opto
#     long_form_data.append({
#         'PID': session['pid'],
#         'Hemisphere': session['hemisphere'],
#         'Condition': 'Opto',
#         'Separation': session['separation']['opto'][t_idx]
#     })

# df_long = pd.DataFrame(long_form_data)
# # Filter for only Ipsi/Contra (ignore 'Other' if any)
# df_long = df_long[df_long['Hemisphere'].isin(['Ipsi', 'Contra'])]

# # Plotting
# plt.figure(figsize=(7, 6))

# # Define palette manually to ensure Control=Black, Opto=Blue
# palette = {'Control': 'gray', 'Opto': 'dodgerblue'}

# sns.barplot(
#     data=df_long, 
#     x='Hemisphere', 
#     y='Separation', 
#     hue='Condition', 
#     palette=palette,
#     errorbar='se', 
#     capsize=0.1,
#     alpha=0.9
# )

# # Add swarmplot/stripplot for individual points
# sns.stripplot(
#     data=df_long, 
#     x='Hemisphere', 
#     y='Separation', 
#     hue='Condition', 
#     palette={'Control': 'black', 'Opto': 'darkblue'}, # Darker dots for visibility
#     dodge=True, # Important: aligns dots with the grouped bars
#     jitter=True,
#     marker='o',
#     alpha=0.6,
#     legend=False
# )

# plt.axhline(0, color='k', linewidth=0.5)
# plt.ylabel(f"CD Separation at {target_time}s")
# plt.title("Effect of SNr Inhibition on Coding Direction")
# plt.tight_layout()
# plt.savefig(figures_path + '/Quantification_500ms_Grouped.png')
# plt.show()

# # --- STATISTICS ---
# print(f"--- Paired Statistics at {target_time}s ---")
# for hemi in ['Ipsi', 'Contra']:
#     subset = df_long[df_long['Hemisphere'] == hemi]
#     ctrl_vals = subset[subset['Condition'] == 'Control']['Separation'].values
#     opto_vals = subset[subset['Condition'] == 'Opto']['Separation'].values
    
#     # Ensure they are aligned by PID (Seaborn handles this, but for T-test we must be sure)
#     # Assuming the list order preserved pairing, but let's be safe:
#     pids = subset['PID'].unique()
#     pairs = []
#     for pid in pids:
#         c = subset[(subset['Condition']=='Control') & (subset['PID']==pid)]['Separation'].values
#         o = subset[(subset['Condition']=='Opto') & (subset['PID']==pid)]['Separation'].values
#         if len(c)==1 and len(o)==1:
#             pairs.append((c[0], o[0]))
            
#     c_arr = np.array([p[0] for p in pairs])
#     o_arr = np.array([p[1] for p in pairs])
    
#     t, p = ttest_rel(c_arr, o_arr)
#     print(f"{hemi}: Control Mean={c_arr.mean():.3f}, Opto Mean={o_arr.mean():.3f}")
#     print(f"      p-val = {p:.5f} (N={len(c_arr)})")