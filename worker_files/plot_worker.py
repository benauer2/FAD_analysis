import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from config import get_step_at_time
import pandas as pd

# Set the global font family to sans-serif and prioritize Calibri
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Calibri']
plt.rcParams['legend.handletextpad'] = 0.2

'''
FORMIC ACID PLOTTER
'''
def plot_formic_acid(rates_df, x_interval=8.0, y_interval=2.0, figsize=(11, 5), shade_colors=None):
    if rates_df is None or rates_df.empty or "Time (hr)" not in rates_df.columns or "FA mol frac" not in rates_df.columns:
        fig, ax = plt.subplots(figsize=figsize)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(
        rates_df["Time (hr)"], 
        rates_df["FA mol frac"], 
        linestyle='None', 
        marker='o', 
        markersize=6, 
        label='Formic Acid (Reactor)', 
        markerfacecolor='purple', 
        markeredgewidth=1, 
        markeredgecolor='black'
    )

    max_rate = rates_df["FA mol frac"].max()
    max_time = rates_df["Time (hr)"].max()

    y_upper = max_rate * 1.2 if np.isfinite(max_rate) and max_rate > 0 else 1.0
    x_upper = max_time if np.isfinite(max_time) and max_time > 0 else 1.0

    ax.set_ylim(0, y_upper)
    ax.set_xlim(0, x_upper)

    plt.xlabel("Time (hr)\n\n", labelpad=7, fontsize=18)
    plt.ylabel("FA mole fraction (%)", labelpad=7, fontsize=20)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(x_interval))
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=8, prune=None))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

    ax.tick_params(axis='both', which='major', direction='inout', length=8, width=1.5, labelsize=16)
    ax.tick_params(axis='both', which='minor', direction='in', length=4, width=1.0)

    for spine in ['top', 'bottom', 'left', 'right']:
        ax.spines[spine].set_linewidth(1.8)
        ax.spines[spine].set_zorder(5) 

    species_handles, species_labels = ax.get_legend_handles_labels()
    if species_handles:
        species_legend = ax.legend(handles=species_handles, labels=species_labels,bbox_to_anchor=(0.5, -0.15), 
                                loc='upper center', columnspacing=0.7, fontsize=13, ncol=len(species_labels), frameon=False)
        ax.add_artist(species_legend)

    return fig, ax

def update_axis_ticks(ax, max_time):
    """Helper function to cleanly scale major and minor axis ticks."""
    x_interval = max(2.0, np.ceil(max_time / 10))
    ax.xaxis.set_major_locator(ticker.MultipleLocator(x_interval))
    # Using AutoMinorLocator prevents tick explosion on large time datasets
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(5))


'''
REACTION RATE PLOTTER
'''
def plot_gas_rates(rates_df, df_electronics, x_interval=6.0, y_interval=5.0, figsize=(11, 5), shade_colors=None, rxn_schedule=None): 
    fig, ax = plt.subplots(figsize=figsize)
    shade_colors = shade_colors or {}
    
    has_rates = rates_df is not None and not rates_df.empty and "Time (hr)" in rates_df.columns
    has_elec = df_electronics is not None and not df_electronics.empty and "Time (hr)" in df_electronics.columns

    if not has_rates and not has_elec:
        return fig, ax

    if has_rates:
        for col, marker, color, label in [("CO2 (nmol/min)", 's', 'orange', r'CO$_2$'),
                                          ("CO (nmol/min)", '^', 'blue', 'CO'),
                                          ("H2 (nmol/min)", 'D', 'red', r'H$_2$')]:
            if col in rates_df.columns:
                ax.plot(rates_df["Time (hr)"], rates_df[col], linestyle='None',
                        marker=marker, markersize=6, label=label, markerfacecolor=color,
                        markeredgewidth=1, markeredgecolor='black', zorder=3)


    if has_elec and "current/2 (nmol/min)" in df_electronics.columns:
        ax.plot(df_electronics["Time (hr)"], df_electronics["current/2 (nmol/min)"],
                linestyle='-', linewidth=2, label='Rate of Electron Transfer (|e|/2)', color='orange', zorder=4)
    
    all_times = []
    if has_rates: all_times.extend(rates_df["Time (hr)"].dropna().values)
    if has_elec: all_times.extend(df_electronics["Time (hr)"].dropna().values)
    max_time = max(all_times) if all_times else 1.0
    ax.set_xlim(0, max_time)
    
    y_vals = []
    cutoff_hr = 4.0
    if has_rates:
        for c in ["CO2 (nmol/min)", "CO (nmol/min)", "H2 (nmol/min)"]:
            if c in rates_df.columns:
                if c == "H2 (nmol/min)":
                    if "Time (hr)" in rates_df.columns:
                        h2_filtered = rates_df[rates_df["Time (hr)"] >= cutoff_hr]["H2 (nmol/min)"].dropna()
                        y_vals.extend(h2_filtered.values)
                    else:
                        y_vals.extend(rates_df[c].dropna().values)
                else:
                    y_vals.extend(rates_df[c].dropna().values)
    y_min = min(y_vals) if y_vals else 0.0
    y_max = max(y_vals) if y_vals else 1.0
    y_span = y_max - y_min
    padding = y_span * 0.20 if y_span > 0 else 0.1
    ax.set_ylim(0, y_max + padding)

    plt.xlabel("Time (hr)\n\n", labelpad=7, fontsize=18)
    plt.ylabel("Extensive Rate (nmol/min)", labelpad=7, fontsize=20)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(x_interval))
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=8, prune=None))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

    ax.tick_params(axis='both', which='major', direction='inout', length=8, width=1.5, labelsize=16)
    ax.tick_params(axis='both', which='minor', direction='in', length=4, width=1.0)

    for spine in ['top', 'bottom', 'left', 'right']:
        ax.spines[spine].set_linewidth(1.8)
        ax.spines[spine].set_zorder(5) 

    has_bias_data = (rates_df is not None) and (rxn_schedule is not None) and ("Time (hr)" in rates_df.columns)
    colors = shade_colors if shade_colors is not None else {}



    if has_bias_data:
        bias_values = rates_df["Bias (V)"]
        time_values = rates_df['Time (hr)'].tolist()

        if len(time_values) > 0:
            dt = time_values[1] - time_values[0] if len(time_values) > 1 else 0
            start = time_values[0]
            current_bias = bias_values[0]

            for i in range(1, len(bias_values)):
                if bias_values[i] != current_bias:
                    ax.axvspan(start, time_values[i], color=shade_colors.get(current_bias, 'white'), alpha=1.0, zorder=1, lw=0)
                    start = time_values[i]
                    current_bias = bias_values[i]
    # Draw final block stretch
            ax.axvspan(start, time_values[-1] + dt, color=shade_colors.get(current_bias, 'white'), alpha=1.0, zorder=1, lw=0)

    # 5. Draw double legends
    species_handles, species_labels = ax.get_legend_handles_labels()

    if species_handles:
        species_legend = ax.legend(handles=species_handles, labels=species_labels,bbox_to_anchor=(0.5, -0.15), 
                        loc='upper center', columnspacing=0.7, fontsize=13, ncol=len(species_labels), frameon=False)
        ax.add_artist(species_legend)

    if has_bias_data:
        unique_biases = sorted(set(bias_values), key=lambda x: (isinstance(x, str), x)) 

        # Use clean patches with thin borders for the background key
        bias_patches = [Patch(
                            facecolor=shade_colors[b], 
                            edgecolor='black', 
                            label=f'{b} V' if isinstance(b, (int, float)) else f'{b}'
                            ) 
                        for b in unique_biases
        ]

        # Bias legend sitting above the top axis line
        bias_legend = ax.legend(handles=bias_patches, bbox_to_anchor=(0.5, 1.00), loc="lower center", 
                                columnspacing=1.0, fontsize=12, ncol=len(unique_biases), frameon=False)



    return fig, ax

'''
#############
MFC PLOTTER
#############
'''

def plot_mfc_profile(mfc_df, figsize=(11, 4)):
    fig, ax = plt.subplots(figsize=figsize)
    
    time_col = 'Elapsed_Time_hr' if 'Elapsed_Time_hr' in mfc_df.columns else ('Time (hr)' if 'Time (hr)' in mfc_df.columns else mfc_df.columns[0])
    max_time = pd.to_numeric(mfc_df[time_col], errors='coerce').max()
    if pd.isna(max_time) or max_time <= 0:
        max_time = 1.0
    
    # Calculate dynamic x-interval
    x_interval = max(0.5, np.ceil(max_time / 10.0))
    
    n2_cols = [c for c in mfc_df.columns if 'n2' in c.lower() and 'flow' in c.lower()]
    h2_cols = [c for c in mfc_df.columns if 'h2' in c.lower() and 'flow' in c.lower()]
    co_cols = [c for c in mfc_df.columns if 'co' in c.lower() and 'flow' in c.lower() and 'co2' not in c.lower()]
    co2_cols = [c for c in mfc_df.columns if 'co2' in c.lower() and 'flow' in c.lower()]
    
    plotted_cols = set()
    
    if n2_cols:
        ax.plot(mfc_df[time_col], pd.to_numeric(mfc_df[n2_cols[0]], errors='coerce'), color='navy', lw=1.5, label=r'N2 Flow (sccm)')
        plotted_cols.add(n2_cols[0])
    if h2_cols:
        ax.plot(mfc_df[time_col], pd.to_numeric(mfc_df[h2_cols[0]], errors='coerce'), color='forestgreen', lw=1.5, label=r'H2 Flow (sccm)')
        plotted_cols.add(h2_cols[0])
    if co_cols:
        ax.plot(mfc_df[time_col], pd.to_numeric(mfc_df[co_cols[0]], errors='coerce'), color='darkorange', lw=1.8, label=r'CO Flow (sccm)')
        plotted_cols.add(co_cols[0])
    if co2_cols:
        ax.plot(mfc_df[time_col], pd.to_numeric(mfc_df[co2_cols[0]], errors='coerce'), color='crimson', lw=1.5, label=r'CO2 Flow (sccm)')
        plotted_cols.add(co2_cols[0])
        
    other_flow_cols = [c for c in mfc_df.columns if 'flow' in c.lower() and 'total' not in c.lower() and c not in plotted_cols]
    palette = ['purple', 'teal', 'magenta', 'brown', 'goldenrod']
    for idx, col in enumerate(other_flow_cols):
        c_color = palette[idx % len(palette)]
        clean_name = col.replace('_', ' ').replace('(sccm)', '').strip()
        ax.plot(mfc_df[time_col], pd.to_numeric(mfc_df[col], errors='coerce'), color=c_color, lw=1.5, label=f'{clean_name} (sccm)')
        
    ax.set_xlabel("Time (hr)", labelpad=7, fontsize=14)
    ax.set_ylabel("Flow Rate (sccm)", labelpad=7, fontsize=14)
    ax.set_xlim(0, max_time)
    
    ax.xaxis.set_major_locator(ticker.MultipleLocator(x_interval))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(x_interval / 5))
    ax.tick_params(axis='both', which='major', direction='inout', length=6, width=1.2, labelsize=12)
    
    for spine in ['top', 'bottom', 'left', 'right']:
        ax.spines[spine].set_linewidth(1.5)
        
    ax.legend(loc='best', frameon=False, fontsize=11)
    return fig, ax

'''
#####################
TEMPERATURE PLOTTER
#####################
'''

def plot_temp_profile(temp_df, figsize=(11, 4)):
    fig, ax = plt.subplots(figsize=figsize)
    
    time_col = 'Elapsed_Hours' if 'Elapsed_Hours' in temp_df.columns else 'Time (hr)'
    max_time = temp_df[time_col].max()
    
    # Calculate a dynamic x-interval based on length of run
    x_interval = max(2.0, np.ceil(max_time / 10))

    ax.plot(temp_df[time_col], temp_df['Temperature_C'], color='crimson', lw=1.5, label='Measured T (C)')
    if 'Setpoint_C' in temp_df.columns:
        ax.plot(temp_df[time_col], temp_df['Setpoint_C'], color='black', linestyle='--', lw=1.5, label='Setpoint (C)')
        
    ax.set_xlabel("Time (hr)", labelpad=7, fontsize=14)
    ax.set_ylabel("Temperature (C)", labelpad=7, fontsize=14)
    ax.set_xlim(0, max_time)
    
    ax.xaxis.set_major_locator(ticker.MultipleLocator(x_interval))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(x_interval / 5))
    ax.tick_params(axis='both', which='major', direction='inout', length=6, width=1.2, labelsize=12)
    
    for spine in ['top', 'bottom', 'left', 'right']:
        ax.spines[spine].set_linewidth(1.5)
        
    ax.legend(loc='upper right', frameon=False, fontsize=11)
    return fig, ax


'''
Current vs time plotter
'''


def plot_current_time(rates_df=None, df_electronics=None, x_interval=8.0, y_interval=None, figsize=(11, 5), shade_colors=None, rxn_schedule=None):
    """
    Plots extensive rates and electronic data over time with a clean layout,
    custom 'inout' bold ticks, dynamic background bias shading, and stacked legends.
    """
    fig, ax = plt.subplots(figsize=figsize)

    if df_electronics is None:
        if rates_df is not None:
            df_electronics = rates_df
            rates_df = None
        else:
            df_electronics = pd.DataFrame()

    has_data = (
        not df_electronics.empty and 
        "Time (hr)" in df_electronics.columns and 
        "Current (uA)" in df_electronics.columns and
        not df_electronics["Time (hr)"].dropna().empty
    )

    if not has_data:
        max_time = 1.0
        if rxn_schedule and isinstance(rxn_schedule, list) and len(rxn_schedule) > 0 and "time_window" in rxn_schedule[-1]:
            max_time = rxn_schedule[-1]["time_window"][1]
        elif rates_df is not None and not rates_df.empty and "Time (hr)" in rates_df.columns:
            max_time = max(rates_df["Time (hr)"].dropna().max(), 1.0)
        ax.set_xlim(0, max_time)
        ax.set_ylim(-1, 1)
        ax.text(0.5, 0.5, "No Current / Electronics Data Available", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=14)
        return fig, ax

    # 1. Plot Data with strict Z-ordering (Lines/Markers on top of background shading)
    ax.plot(df_electronics["Time (hr)"], df_electronics["Current (uA)"], 
            linestyle='-', linewidth=2, label='Current', color='black', zorder=4)
    
    # 2. Set Y-axis limit based on current
    y_min = df_electronics["Current (uA)"].min()
    y_max = df_electronics["Current (uA)"].max()

    if np.isfinite(y_min) and np.isfinite(y_max):
        if y_min == y_max:
            ax.set_ylim(y_min - 1.0, y_max + 1.0)
        else:
            span = y_max - y_min
            ax.set_ylim(y_min - 0.1 * span, y_max + 0.1 * span)

    max_t = df_electronics["Time (hr)"].dropna().max()
    if not np.isfinite(max_t) or max_t <= 0:
        max_t = 1.0
    ax.set_xlim(0, max_t)

    # 3. Labeling and Axis Formatting
    plt.xlabel("Time (hr)\n\n", labelpad=7, fontsize=18)
    plt.ylabel("Current (uA)", labelpad=7, fontsize=20)

    # Tick marks configuration
    ax.xaxis.set_major_locator(ticker.MultipleLocator(x_interval))
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))

    if y_interval is None:
        ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=6, steps=[1,2,5,10]))
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    else:
        ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=6, steps=[1,2,5,10]))
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))


    ax.tick_params(axis='both', which='major', direction='inout', length=8, width=1.5, labelsize=16)
    ax.tick_params(axis='both', which='minor', direction='in', length=4, width=1.0)

    # Thick borders around the plot frame
    for spine in ['top', 'bottom', 'left', 'right']:
        ax.spines[spine].set_linewidth(1.8)
        ax.spines[spine].set_zorder(5) # Keeps borders on top of shaded background

    has_bias_data = (rates_df is not None) and (rxn_schedule is not None) and ("Time (hr)" in rates_df.columns)
    colors = shade_colors if shade_colors is not None else {}

    if has_bias_data:
        bias_values = rates_df["Bias (V)"]
        time_values = rates_df['Time (hr)'].tolist()

        if len(time_values) > 0:
            dt = time_values[1] - time_values[0] if len(time_values) > 1 else 0
            start = time_values[0]
            current_bias = bias_values[0]

            for i in range(1, len(bias_values)):
                if bias_values[i] != current_bias:
                    ax.axvspan(start, time_values[i], color=shade_colors.get(current_bias, 'white'), alpha=1.0, zorder=1, lw=0)
                    start = time_values[i]
                    current_bias = bias_values[i]
    # Draw final block stretch
            ax.axvspan(start, time_values[-1] + dt, color=shade_colors.get(current_bias, 'white'), alpha=1.0, zorder=1, lw=0)

    # 5. Draw double legends
    species_handles, species_labels = ax.get_legend_handles_labels()

    if species_handles:
        species_legend = ax.legend(handles=species_handles, labels=species_labels,bbox_to_anchor=(0.5, -0.15), 
                        loc='upper center', columnspacing=0.7, fontsize=13, ncol=len(species_labels), frameon=False)
        ax.add_artist(species_legend)

    if has_bias_data:
        unique_biases = sorted(set(bias_values), key=lambda x: (isinstance(x, str), x)) 

        # Use clean patches with thin borders for the background key
        bias_patches = [Patch(
                            facecolor=shade_colors[b], 
                            edgecolor='black', 
                            label=f'{b} V' if isinstance(b, (int, float)) else f'{b}'
                            ) 
                        for b in unique_biases
        ]

        # Bias legend sitting above the top axis line
        bias_legend = ax.legend(handles=bias_patches, bbox_to_anchor=(0.5, 1.00), loc="lower center", 
                                columnspacing=1.0, fontsize=12, ncol=len(unique_biases), frameon=False)
    return fig, ax


def plot_OCP(rates_df, df_electronics, figsize=(11, 5), shade_colors=None, rxn_schedule=None):
    """
    Plots extensive rates and electronic data over time with a clean layout,
    custom 'inout' bold ticks, dynamic background bias shading, and stacked legends.
    
    Parameters:
    -----------
    rates_df : pandas.DataFrame
        DataFrame containing gas rate columns, 'Time (hr)', and 'Bias (V)'.
    df_electronics : pandas.DataFrame
        DataFrame containing 'Time (hr)' and 'current/2 (nmol/min)'.
    x_interval : float, default 6.0
        The step sizing for major X-axis tick lines.
    y_interval : float, default 5.0
        The step sizing for major Y-axis tick lines.
    figsize : tuple, default (11, 5)
        The total dimensions of the output figure canvas window.
    """
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # 1. Plot Data with strict Z-ordering (Lines/Markers on top of background shading)
    ax.plot(df_electronics["Time (hr)"], df_electronics["Working Electrode (V)"], 
            linestyle='-', linewidth=2, label='WE (Pt)', color='red', zorder=4)

    ax.plot(df_electronics["Time (hr)"], df_electronics["Counter Electrode (V)"], 
            linestyle='-', linewidth=2, label='CE (Au)', color='blue', zorder=4)
    
    #ax.set_ylim(df_electronics["Working Electrode (V)"].min()*1.2, df_electronics["Working Electrode (V)"].max()*1.2)
    ax.set_xlim(0, rates_df["Time (hr)"].max())

    # Works perfectly whether data is [-0.5, -0.1], [0.001, 0.005], or [10, 100]
    y_min, y_max = df_electronics["Working Electrode (V)"].min(), df_electronics["Working Electrode (V)"].max()

    # 1. Calculate the total span of the data
    y_span = y_max - y_min

    # 2. Define a fraction for your margin (e.g., 10% padding on top and bottom)
    # If data is completely flat (span is 0), fallback to a small default step
    padding = y_span * 0.10 if y_span != 0 else 0.1

    # 3. SUBTRACT from the min, ADD to the max
    ymin_padded = y_min - padding
    ymax_padded = y_max + padding

    # 4. Apply to your plot
    ax.set_ylim(ymin_padded, ymax_padded)

    # 3. Labeling and Axis Formatting
    plt.xlabel("Time (hr)\n\n", labelpad=7, fontsize=18)
    plt.ylabel("Potential (V)", labelpad=7, fontsize=20)

    # Tick marks configuration
    #ax.xaxis.set_major_locator(ticker.MultipleLocator(x_interval))
    #ax.xaxis.set_minor_locator(ticker.MultipleLocator(x_interval/6))
    #ax.yaxis.set_major_locator(ticker.MultipleLocator(y_interval))
    #ax.yaxis.set_minor_locator(ticker.MultipleLocator(y_interval/4))

    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=6, prune=None))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f')) # Standardizes decimals
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

    ax.tick_params(axis='both', which='major', direction='inout', length=8, width=1.5, labelsize=16)
    ax.tick_params(axis='both', which='minor', direction='in', length=4, width=1.0)

    # Thick borders around the plot frame
    for spine in ['top', 'bottom', 'left', 'right']:
        ax.spines[spine].set_linewidth(1.8)
        ax.spines[spine].set_zorder(5) # Keeps borders on top of shaded background

    # 4. Generate Background Bias Shading
    if "Bias (V)" in rates_df.columns:
        bias_values = rates_df["Bias (V)"].tolist()
    else:
        bias_values = rates_df['Time (hr)'].apply(
            lambda t: get_step_at_time(t, rxn_schedule).get('we_volt', 0.0) if isinstance(get_step_at_time(t, rxn_schedule), dict) else 0.0
        ).tolist()
    time_values = rates_df['Time (hr)'].tolist()

    dt = time_values[1] - time_values[0] if len(time_values) > 1 else 0
    start = time_values[0]
    current_bias = bias_values[0]

    for i in range(1, len(bias_values)):
        if bias_values[i] != current_bias:
            ax.axvspan(start, time_values[i], color=shade_colors.get(current_bias, 'white'), alpha=1.0, zorder=1, lw=0)
            start = time_values[i]
            current_bias = bias_values[i]
    # Draw final block stretch
    ax.axvspan(start, time_values[-1] + dt, color=shade_colors.get(current_bias, 'white'), alpha=1.0, zorder=1, lw=0)

    # 5. Draw double legends
    species_handles, species_labels = ax.get_legend_handles_labels()

    unique_biases = sorted(set(bias_values), key=lambda x: (isinstance(x, str), x))

    # Use clean patches with thin borders for the background key
    bias_patches = [Patch(
                        facecolor=shade_colors.get(b, 'white'), 
                        edgecolor='black', 
                        label=f'{b} V' if isinstance(b, (int, float)) else f'{b}'
                        ) 
                    for b in unique_biases
    ]

    # Species legend sitting below the bottom axis line
    species_legend = ax.legend(handles=species_handles, labels=species_labels,bbox_to_anchor=(0.5, -0.15), 
                               loc='upper center', columnspacing=0.7, fontsize=13, ncol=len(species_labels), frameon=False)
    ax.add_artist(species_legend)

    # Bias legend sitting above the top axis line
    bias_legend = ax.legend(handles=bias_patches, bbox_to_anchor=(0.5, 1.00), loc="lower center", 
                            columnspacing=1.0, fontsize=12, ncol=len(unique_biases), frameon=False)


    return fig, ax

def cap_plot(cv_segments, x_interval=3.0, y_interval=None, figsize=(11, 5), shade_colors=None, rxn_schedule=None, target_sweep_rate=1.0):
    """
    Plots capacitance vs time profile for a specified sweep rate (e.g. 1.0 V/s or 0.02 V/s).
    """
    runs = []
    capacitances = []
    times = []

    for cv_id, cv_df in cv_segments.items():
        current_rate = cv_df['sweep_rate'].iloc[0]
        
        # Match target rate within tolerance or match exactly
        if target_sweep_rate is None or abs(current_rate - target_sweep_rate) < 0.005 or (target_sweep_rate >= 0.1 and abs(current_rate - target_sweep_rate) < 0.05):
            runs.append(cv_id)
            times.append(cv_df['Time (hr)'].iloc[0])
            capacitances.append(cv_df['capacitance (uF)'].iloc[0])

    fig, ax = plt.subplots(figsize=figsize)

    if shade_colors is None:
        shade_colors = {
            0: "#ffffff",
            1: '#d0e1f9',
            -1: '#f9d0d0',
        }

    cum_time = 0.0
    unique_biases = []

    if rxn_schedule:
        for block in rxn_schedule:
            if not isinstance(block, dict):
                continue
            v_bias = block.get("we_volt", block.get("voltage", 0.0))
            duration = block.get("dur_hrs", block.get("duration_hours", 0.0))
            if "time_window" in block and isinstance(block["time_window"], (list, tuple)) and len(block["time_window"]) >= 2:
                start_t, end_t = block["time_window"][0], block["time_window"][1]
                cum_time = end_t
            else:
                start_t = cum_time
                end_t = cum_time + duration
                cum_time = end_t

            color = shade_colors.get(v_bias, "#f0f0f0")
            ax.axvspan(start_t, end_t, color=color, alpha=1.0, zorder=1, lw=0)

            if v_bias not in unique_biases:
                unique_biases.append(v_bias)

    plot_color = '#b30000' if (target_sweep_rate is not None and target_sweep_rate < 0.1) else 'black'
    plot_marker = 's' if (target_sweep_rate is not None and target_sweep_rate < 0.1) else 'o'
    ax.plot(times, capacitances, marker=plot_marker, linestyle='None', color=plot_color, linewidth=2, markersize=7)

    max_time = max(cum_time, max(times) if times else 0)
    ax.set_xlim(0, max_time)

    rate_str = f" ({target_sweep_rate:.2f} V/s)" if (target_sweep_rate is not None and target_sweep_rate < 0.1) else (f" ({target_sweep_rate:.1f} V/s)" if target_sweep_rate is not None else "")
    ax.set_xlabel('Time (hr)', fontsize=18)
    ax.set_ylabel(f'Capacitance{rate_str} (uF)', fontsize=18)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(x_interval))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(x_interval/6))

    if y_interval is None:
        ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=6, steps=[1,2,5,10]))
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(4))
    else:
        ax.yaxis.set_major_locator(ticker.MultipleLocator(y_interval))
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(y_interval/4))

    ax.tick_params(axis='both', which='major', direction='inout', length=8, width=1.5, labelsize=16)
    ax.tick_params(axis='both', which='minor', direction='in', length=4, width=1.0)

    for spine in ['top', 'bottom', 'left', 'right']:
        ax.spines[spine].set_linewidth(1.8)
        ax.spines[spine].set_zorder(5)

    bias_patches = [
        Patch(facecolor=shade_colors.get(b, '#f0f0f0'), edgecolor='black', linewidth=0.8,
              label=f'{b} V' if isinstance(b, (int, float)) else f'{b}')
              for b in unique_biases
    ]

    ax.legend(
        handles=bias_patches,
        bbox_to_anchor=(0.5,1.00),
        loc="lower center",
        columnspacing=1.0,
        fontsize=12,
        ncol=len(unique_biases),
        frameon=False
    )

    return fig, ax


def cv_plot(cv_segments, rxn_schedule=None, max_per_page=6, rate_filter='fast'):
    """
    Plots CV data in square subplots.
    rate_filter:
      'fast': sweeps with sweep_rate >= 0.1 V/s (e.g. 0.5, 1.0, 1.5 V/s)
      'slow': sweeps with sweep_rate < 0.1 V/s (e.g. 0.02 V/s)
      'all' or None: all sweeps
    """
    cv_items = list(cv_segments.items())
    if not cv_items:
        return []

    # Filter items according to rate_filter
    if rate_filter == 'fast':
        filtered_items = [item for item in cv_items if item[1]['sweep_rate'].iloc[0] >= 0.1]
    elif rate_filter == 'slow':
        filtered_items = [item for item in cv_items if item[1]['sweep_rate'].iloc[0] < 0.1]
    else:
        filtered_items = cv_items

    if not filtered_items:
        return []

    # Group by block_id if present, otherwise group by chunking
    if "block_id" in filtered_items[0][1].columns:
        cv_chunks = []
        current_block = None
        current_chunk = []
        for item in filtered_items:
            b_id = item[1]["block_id"].iloc[0]
            if current_block is None or b_id == current_block:
                current_chunk.append(item)
                current_block = b_id
            else:
                cv_chunks.append(current_chunk)
                current_chunk = [item]
                current_block = b_id
        if current_chunk:
            cv_chunks.append(current_chunk)
    else:
        chunk_size = 3 if rate_filter == 'fast' else 1
        cv_chunks = [filtered_items[i:i + chunk_size] for i in range(0, len(filtered_items), chunk_size)]

    page_chunks = [cv_chunks[i : i + max_per_page]
                   for i in range(0, len(cv_chunks), max_per_page)]
    
    figs = []
    if rate_filter == 'slow':
        colors = ['#b30000', '#d95f02', '#7570b3']
    else:
        colors = ['#85b8e0', '#0077c8', '#08306b', '#31a354']
        
    first_cv_id = filtered_items[0][0] if filtered_items else None
    cols = 2

    for page_idx, current_page_chunks in enumerate(page_chunks):
        rows = int(np.ceil(len(current_page_chunks) / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(9.5, 4.8 * rows), squeeze=False)
        axes = axes.flatten()

        for group_idx, chunk in enumerate(current_page_chunks):
            ax = axes[group_idx]
            ax.set_box_aspect(1)
            
            for idx, (cv_id, cv_df) in enumerate(chunk):
                voltage = cv_df.iloc[:, 2]
                current = cv_df.iloc[:, 3] * 1E6
                
                sweep_rate = cv_df['sweep_rate'].iloc[0]
                color = colors[idx % len(colors)]
                label_text = f'{sweep_rate:.2f} V/s' if sweep_rate < 0.1 else f'{sweep_rate:.1f} V/s'
                
                ax.plot(
                    voltage, 
                    current, 
                    color=color, 
                    marker='o', 
                    markersize=2.5, 
                    linewidth=1.5, 
                    label=label_text
                )

            first_df = chunk[0][1]
            raw_elapsed_hr = first_df['Time (hr)'].iloc[0]
            elapsed_hr = round(raw_elapsed_hr, 1)
            
            step = get_step_at_time(raw_elapsed_hr, rxn_schedule) if rxn_schedule else None
            if isinstance(step, dict) and "we_volt" in step:
                volt_val = step["we_volt"]
                cond_str = f"{volt_val} V" if isinstance(volt_val, (int, float)) else str(volt_val)
                in_plot_label = f"{cond_str} window\n{elapsed_hr:.1f} hr"
            elif chunk[0][0] == first_cv_id or raw_elapsed_hr < 0.1:
                in_plot_label = f"Initial Setup\n{elapsed_hr:.1f} hr"
            else:
                in_plot_label = f"{elapsed_hr:.1f} hr"

            # 1. Spines setup (clean black box)
            for spine in ['left', 'right', 'top', 'bottom']:
                ax.spines[spine].set_linewidth(1.2)
                ax.spines[spine].set_color('black')

            # 2. Outward tick marks and tick locators
            ax.tick_params(axis='both', which='major', direction='out', length=5, width=1.2, labelsize=10)
            ax.tick_params(axis='both', which='minor', direction='out', length=3, width=0.8)
            ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
            ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))

            # 3. Axis labels
            ax.set_xlabel('Voltage (V)', fontsize=11, fontweight='normal', labelpad=4)
            ax.set_ylabel(r'Current ($\mu$A)', fontsize=11, fontweight='normal', labelpad=4)

            # 4. Bold Panel Letter (A, B, C...)
            panel_letter = chr(65 + page_idx * max_per_page + group_idx)
            ax.text(
                0.0, 1.05, 
                panel_letter, 
                transform=ax.transAxes, 
                fontsize=13, 
                fontweight='bold', 
                va='bottom', 
                ha='left'
            )

            # 5. Inset Condition & Time Label inside upper-left of plot area
            ax.text(
                0.05, 0.94, 
                in_plot_label, 
                transform=ax.transAxes, 
                fontsize=10, 
                fontweight='bold', 
                va='top', 
                ha='left'
            )

            # 6. Horizontal Legend above the plot box
            ax.legend(
                loc='lower right',
                bbox_to_anchor=(1.0, 1.01),
                ncol=min(4, len(chunk)),
                fontsize=9,
                frameon=False,
                handlelength=0.8,
                handletextpad=0.3,
                columnspacing=0.8,
                markerscale=1.5
            )

        # Hide unused axes
        for j in range(group_idx + 1, len(axes)):
            axes[j].axis('off')
        
        plt.subplots_adjust(hspace=0.42, wspace=0.32, top=0.92, bottom=0.08, left=0.10, right=0.95)
        figs.append(fig)

    return figs

def plot_cv_sweep_rate_slope(cv_segments, summary_df, fit_details, title_prefix=""):
    """
    Generates a two-panel plot: 
    1. Multi-sweep rate CV overlays.
    2. Linear regression fit of capacitive current vs sweep rate.
    """
    if fit_details is None or not fit_details:
        print("No valid fit details available for plotting.")
        return None, None

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Panel 1: CV overlay
    ax_cv = axes[0]
    for cv_id, cv_df in cv_segments.items():
        v_col = 'Working Electrode (V)' if 'Working Electrode (V)' in cv_df.columns else cv_df.columns[2]
        c_col = 'Current (uA)' if 'Current (uA)' in cv_df.columns else 'Current (A)'
        
        v = cv_df[v_col].values
        i = cv_df[c_col].values if 'Current (uA)' in cv_df.columns else cv_df[c_col].values * 1e6
        rate = cv_df['sweep_rate'].iloc[0] if 'sweep_rate' in cv_df.columns else 1.0

        ax_cv.plot(v, i, label=f"{rate} V/s", linewidth=1.8)

    eval_v = summary_df["Evaluation Potential (V)"].iloc[0]
    ax_cv.axvline(eval_v, color='black', linestyle='--', alpha=0.6, label=f"Eval V ({eval_v}V)")
    ax_cv.set_xlabel("Working Electrode (V)", fontsize=12)
    ax_cv.set_ylabel(r"Current ($\mu$A)", fontsize=12)
    ax_cv.set_title(f"{title_prefix} Multi-Sweep CVs", fontsize=13)
    ax_cv.legend(frameon=False)

    # Panel 2: Slope linear regression
    ax_slope = axes[1]
    rates = fit_details["sweep_rates"]
    i_avg = fit_details["currents_avg_uA"]
    slope = fit_details["slope"]
    intercept = fit_details["intercept"]
    r2 = fit_details["r_squared"]

    x_fit = np.linspace(min(rates) * 0.8, max(rates) * 1.2, 50)
    y_fit = slope * x_fit + intercept

    ax_slope.plot(rates, i_avg, 'o', color="#A10601", markersize=8, label="$Delta i_{cap}/2$ Data")
    ax_slope.plot(x_fit, y_fit, '--', color="black", linewidth=2, 
                   label=f"Fit: C = {abs(slope):.2f} uF\n$R^2$ = {r2:.4f}")

    ax_slope.set_xlabel("Sweep Rate (V/s)", fontsize=12)
    ax_slope.set_ylabel(r"Capacitive Current ($\mu$A)", fontsize=12)
    ax_slope.set_title(f"Capacitance Slope Extraction @ {eval_v} V", fontsize=13)
    ax_slope.legend(frameon=False)

    fig.subplots_adjust(left=0.08, right=0.95, top=0.90, bottom=0.15, wspace=0.3)
    return fig, axes

import matplotlib.pyplot as plt
import numpy as np

def plot_all_cv_chunks_grid(fit_details_list, max_cols=3):
    """
    Plots all CV chunk slope fits (i vs. v) into a single compact multi-panel grid 
    so they fit on a single PDF page.
    """
    n_chunks = len(fit_details_list)
    if n_chunks == 0:
        return None, None

    cols = min(n_chunks, max_cols)
    rows = int(np.ceil(n_chunks / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(3.5 * cols, 2.8 * rows), squeeze=False)
    axes_flat = axes.flatten()

    for idx, detail in enumerate(fit_details_list):
        ax = axes_flat[idx]
        rates = detail["sweep_rates"]
        i_avg = detail["currents_avg_uA"]
        slope = detail["slope"]
        intercept = detail["intercept"]
        r2 = detail["r_squared"]
        chunk_id = detail["chunk_id"]
        time_hr = detail["time_hr"]

        # Raw data points
        ax.plot(rates, i_avg, 'o', color="#2b5c8f", markersize=5, label="Data")

        # Fitted regression line
        x_fit = np.linspace(min(rates) * 0.8, max(rates) * 1.2, 20)
        y_fit = slope * x_fit + intercept
        ax.plot(x_fit, y_fit, '--', color="#d95f02", linewidth=1.5, label="Fit")

        # Subplot Annotations & Labels
        ax.set_title(f"Chunk {chunk_id} ({time_hr}h)\nC = {abs(slope):.2f} µF | R² = {r2:.3f}", fontsize=8, fontweight='bold')
        ax.set_xlabel("v (V/s)", fontsize=7)
        ax.set_ylabel(r"$i_{cap}$ ($\mu$A)", fontsize=7)
        ax.tick_params(axis='both', which='major', labelsize=7)
        ax.grid(True, linestyle=':', alpha=0.6)

    # Hide unused subplots in the grid
    for j in range(n_chunks, len(axes_flat)):
        fig.delaxes(axes_flat[j])

    fig.subplots_adjust(hspace=0.55, wspace=0.35, left=0.08, right=0.95, top=0.92, bottom=0.08)
    return fig, axes


def plot_10CV(df, title_name, ax, panel_letter=None):
    """
    Plots an overlay of all sweep rates onto a single provided axis (ax) using
    publication-ready square formatting.
    """
    ax.set_box_aspect(1)
    unique_steps = sorted(df["Step number"].unique()) if "Step number" in df.columns else [1]

    # 10 custom hex codes transitioning smoothly from Dark Red -> Purple -> Deep Blue
    red_to_blue = [
        "#b30000",  # 0.1 V/s - Deep Red
        "#e34a33",  # 0.2 V/s - Crimson
        "#fc8d59",  # 0.3 V/s - Coral/Orange-Red
        "#fdbb84",  # 0.4 V/s - Muted Peach
        "#b89bc7",  # 0.5 V/s - Soft Purple
        "#8c96c6",  # 0.6 V/s - Lavender Blue
        "#8c6bb1",  # 0.7 V/s - Medium Violet
        "#41b6c4",  # 0.8 V/s - Teal/Cyan Blue
        "#1d91c0",  # 0.9 V/s - Bright Cerulean Blue
        "#081d58",  # 1.0 V/s - Deep Navy Blue
    ]

    sweep_rates = [step * 0.1 for step in unique_steps]

    for step, color, rate in zip(unique_steps, red_to_blue, sweep_rates):
        step_data = df[df["Step number"] == step].sort_values("Elapsed Time (s)") if ("Step number" in df.columns and "Elapsed Time (s)" in df.columns) else (df[df["Step number"] == step] if "Step number" in df.columns else df)

        v_col = "Working Electrode (V)" if "Working Electrode (V)" in step_data.columns else ("Potential (V)" if "Potential (V)" in step_data.columns else step_data.columns[4 if len(step_data.columns) > 4 else 2])
        c_col = "Current (A)" if "Current (A)" in step_data.columns else ("Current (uA)" if "Current (uA)" in step_data.columns else step_data.columns[3])

        v_vals = step_data[v_col].values
        c_vals = step_data[c_col].values
        if "uA" not in str(c_col) and "\u03bcA" not in str(c_col):
            c_vals = c_vals * 1e6

        ax.plot(
            v_vals,
            c_vals,
            linestyle="-",
            marker='o',
            markersize=2.5,
            linewidth=1.2,
            label=f"{rate:.1f} V/s",
            color=color,
        )

    # 1. Spines setup (clean black box)
    for spine in ["top", "bottom", "left", "right"]:
        ax.spines[spine].set_linewidth(1.2)
        ax.spines[spine].set_color("black")

    # 2. Outward tick marks
    ax.tick_params(axis='both', which='major', direction='out', length=5, width=1.2, labelsize=10)
    ax.tick_params(axis='both', which='minor', direction='out', length=3, width=0.8)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))

    # 3. Axis labels
    ax.set_xlabel("Voltage (V)", fontsize=11, fontweight='normal', labelpad=4)
    ax.set_ylabel(r"Current ($\mu$A)", fontsize=11, fontweight='normal', labelpad=4)

    # 4. Panel Letter (A, B, C...) above top-left
    if panel_letter:
        ax.text(
            0.0, 1.05, 
            panel_letter, 
            transform=ax.transAxes, 
            fontsize=13, 
            fontweight='bold', 
            va='bottom', 
            ha='left'
        )

    # 5. Inset condition/folder title inside upper-left of plot area
    ax.text(
        0.05, 0.94, 
        title_name, 
        transform=ax.transAxes, 
        fontsize=10, 
        fontweight='bold', 
        va='top', 
        ha='left'
    )

    # 6. Horizontal Legend above the plot box
    ax.legend(
        loc='lower right',
        bbox_to_anchor=(1.0, 1.01),
        ncol=min(len(unique_steps), 5),
        fontsize=7.5,
        frameon=False,
        handlelength=0.8,
        handletextpad=0.3,
        columnspacing=0.6,
        markerscale=1.2
    )

def plot_background_rates(rates_df, x_interval=6.0, y_interval=None, figsize=(11, 5)):
    fig, ax1 = plt.subplots(figsize=figsize)
    ax2 = ax1.twinx()

    relative_time = rates_df["Time (hr)"] - rates_df["Time (hr)"].min()
    
    # 1. Plot Data
    ax1.plot(relative_time, rates_df["CO2 (nmol/min)"], linestyle='None',
            marker='s', markersize=6, label=r'CO$_2$', markerfacecolor='orange', 
            markeredgewidth=1, markeredgecolor='black', zorder=3)
    
    ax1.plot(relative_time, rates_df["H2 (nmol/min)"], linestyle='None', 
            marker='D', markersize=6, label='H2', markerfacecolor='red', 
            markeredgewidth=1, markeredgecolor='black', zorder=3)

    ax2.plot(relative_time, rates_df["FA mol frac"], linestyle='None', 
            marker='o', markersize=6, label='Formic Acid (Reactor)', markerfacecolor='purple', 
            markeredgewidth=1, markeredgecolor='black', zorder=4)

    # 2. Filter H2 rate data to only consider t > 4 hours for y-axis upper limit
    h2_after_4h = rates_df.loc[relative_time > 4.0, "H2 (nmol/min)"]
    h2_max = h2_after_4h.max() if not h2_after_4h.empty else 0.0

    max_rate = max(
        rates_df["CO2 (nmol/min)"].max() if "CO2 (nmol/min)" in rates_df.columns else 0.0,
        rates_df["CO (nmol/min)"].max() if "CO (nmol/min)" in rates_df.columns else 0.0,
        h2_max
    )
    
    # Handle edge case if max_rate is zero or NaN
    if not np.isfinite(max_rate) or max_rate <= 0:
        max_rate = rates_df["H2 (nmol/min)"].max() if "H2 (nmol/min)" in rates_df.columns else 10.0
    if not np.isfinite(max_rate) or max_rate <= 0:
        max_rate = 10.0

    t_max = relative_time.max() if (not relative_time.empty and np.isfinite(relative_time.max())) else 1.0
    if t_max <= 0:
        t_max = 1.0
    ax1.set_xlim(0, t_max)
    ax1.set_ylim(0, max_rate * 1.25)

    # 3. Dynamic y_interval scaling to prevent "tick blob" on ax1
    if y_interval is None or (max_rate / y_interval) > 10 or (max_rate / y_interval) < 2:
        raw_step = (max_rate * 1.25) / 5.0
        magnitude = 10 ** np.floor(np.log10(raw_step)) if raw_step > 0 else 1.0
        y_interval = np.ceil(raw_step / magnitude) * magnitude

    # 4. Axis Labels
    ax1.set_xlabel("Time (hr)\n\n", labelpad=7, fontsize=18)
    ax1.set_ylabel("Extensive Rate (nmol/min)", labelpad=7, fontsize=20)
    ax2.set_ylabel('Formic Acid (%)', labelpad=7, fontsize=20)

    # 5. Locators
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(x_interval))
    ax1.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))

    ax1.yaxis.set_major_locator(ticker.MaxNLocator(nbins=6, prune=None))
    ax1.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))

    max_fa = rates_df["FA mol frac"].max() if "FA mol frac" in rates_df.columns else 1.0
    ax2.set_ylim(0, max(4.0, max_fa * 1.15))
    ax2.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5))

    # 6. Ticks & Styling
    ax1.tick_params(axis='both', which='major', direction='inout', length=8, width=1.5, labelsize=16)
    ax1.tick_params(axis='both', which='minor', direction='in', length=4, width=1.0)
    
    ax2.yaxis.tick_right()
    ax2.tick_params(axis='y', which='major', direction='inout', length=8, width=1.5, labelsize=16, colors='black')
    ax2.tick_params(axis='y', which='minor', direction='in', length=4, width=1.0, colors='black')

    for ax in [ax1, ax2]:
        for spine in ['top', 'bottom', 'left', 'right']:
            ax.spines[spine].set_linewidth(1.8)
            ax.spines[spine].set_visible(True)
            ax.spines[spine].set_zorder(5)

    # 7. Legend
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()

    species_legend = ax1.legend(
        handles=handles1 + handles2, 
        labels=labels1 + labels2, 
        bbox_to_anchor=(0.5, -0.15), 
        loc='upper center', 
        columnspacing=0.8, 
        fontsize=13, 
        ncol=4, 
        frameon=False, 
        handletextpad=0.2
    )
    ax1.add_artist(species_legend)

    return fig, ax1, ax2


import numpy as np
import matplotlib.pyplot as plt



def plot_arrhenius(arrhenius_results, figsize=(8, 5), use_ci95=False):
    """
    Plots ln(Rate) vs 1000/T with rate error bars and linear regression fit.
    
    Parameters:
        use_ci95 (bool): If True, displays 95% CI in the legend. If False, displays standard error (1-sigma).
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = {"CO2_avg": "#D9534F", "H2_avg": "#0275D8"}
    labels = {"CO2_avg": r"$\mathrm{CO_2}$ Production", "H2_avg": r"$\mathrm{H_2}$ Production"}
    
    for species, res in arrhenius_results.items():
        x_1000T = res["inv_T"] * 1000.0  # Convert 1/K to 1000/K
        y_ln = res["ln_rate"]
        
        # Propagate rate SD into log space: sigma_ln(Rate) = sigma_Rate / Rate
        if res.get("rate_std") is not None:
            rates = np.exp(y_ln)
            y_err = res["rate_std"] / rates
        else:
            y_err = None

        # Scatter data points with error bars
        ax.errorbar(
            x_1000T, y_ln, 
            yerr=y_err,
            fmt='o',
            color=colors.get(species, "black"), 
            ecolor=colors.get(species, "black"),
            capsize=3,
            capthick=1,
            elinewidth=1.2,
            markersize=6, 
            zorder=3, 
            label=labels.get(species, species)
        )
        
        # Linear fit line
        x_fit = np.linspace(min(x_1000T) * 0.95, max(x_1000T) * 1.05, 100)
        y_fit = (res["slope"] * (x_fit / 1000.0)) + res["intercept"]
        
        # Select error type for legend string
        err_key = "Ea_ci95_kJ" if use_ci95 else "Ea_err_kJ"
        err_suffix = " (95% CI)" if use_ci95 else ""
        
        if err_key in res and np.isfinite(res[err_key]):
            ea_label = f"{res['Ea_kJ']:.1f} ± {res[err_key]:.1f}{err_suffix}"
        else:
            ea_label = f"{res['Ea_kJ']:.1f}"
            
        fit_label = f"{labels.get(species, species)} Fit ($E_a$ = {ea_label} kJ/mol, $R^2$ = {res['R2']:.3f})"
        
        ax.plot(
            x_fit, y_fit, 
            linestyle="--", 
            color=colors.get(species, "black"), 
            linewidth=1.5, 
            label=fit_label
        )

    # Styling & Labels
    ax.set_xlabel(r"1000 / T ($\mathrm{K^{-1}}$)", fontsize=11, labelpad=8)
    ax.set_ylabel(r"$\ln(\mathrm{Rate \ / \ nmol \cdot min^{-1}})$", fontsize=11, labelpad=8)
    ax.set_title("Arrhenius Plot", fontsize=13, fontweight="bold", pad=14)
    
    ax.tick_params(axis='both', which='major', direction='inout', length=6, width=1.2, labelsize=10)
    for spine in ['top', 'bottom', 'left', 'right']:
        ax.spines[spine].set_linewidth(1.2)
        
    ax.legend(loc="best", frameon=False, fontsize=9)
    fig.tight_layout()
    
    return fig, ax

def plot_voltage_rates_barchart(summary_rates_df, shade_colors=None, figsize=(11, 5)):
    """
    Plots a grouped bar chart of averaged gas evolution rates per voltage window.
    Bar colors dynamically match the applied bias window shade from shade_colors.
    Species are distinguished by fill style: CO2 (solid) vs H2 (hatched texture).
    """
    if summary_rates_df is None or summary_rates_df.empty:
        fig, ax = plt.subplots(figsize=figsize)
        return fig, ax

    fig, ax = plt.subplots(figsize=figsize)
    shade_colors = shade_colors or {}

    # Format x-axis tick labels: "Voltage (V)\n(Start-End h)"
    x_labels = [
        f"{row.get('WE_Voltage', row.get('Voltage (V)', 0.0)):.1f}" if isinstance(row.get('WE_Voltage', row.get('Voltage (V)', 0.0)), (int, float)) else str(row.get('WE_Voltage', row.get('Voltage (V)', '0.0')))
        for _, row in summary_rates_df.iterrows()
    ]
    
    x = np.arange(len(x_labels))
    width = 0.30  # Adjusted width for 2 primary gas species (CO2 and H2)

    # Extract species data and uncertainties
    co2_means = summary_rates_df["CO2_avg"].values if "CO2_avg" in summary_rates_df.columns else summary_rates_df.get("CO2_mean", pd.Series(0.0, index=summary_rates_df.index)).values
    co2_errs = summary_rates_df["CO2_std"].values if "CO2_std" in summary_rates_df.columns else summary_rates_df.get("CO2_err", pd.Series(0.0, index=summary_rates_df.index)).values

    h2_means = summary_rates_df["H2_avg"].values if "H2_avg" in summary_rates_df.columns else summary_rates_df.get("H2_mean", pd.Series(0.0, index=summary_rates_df.index)).values
    h2_errs = summary_rates_df["H2_std"].values if "H2_std" in summary_rates_df.columns else summary_rates_df.get("H2_err", pd.Series(0.0, index=summary_rates_df.index)).values

    # Determine per-window background colors matching bias voltage
    window_colors = []
    for _, row in summary_rates_df.iterrows():
        v = row.get("WE_Voltage", row.get("Voltage (V)", 0.0))
        # Handle floating point key lookup or default soft blue/gray
        color = shade_colors.get(v, shade_colors.get(round(v, 2) if isinstance(v, (int, float)) else v, "#d0e1f9"))
        window_colors.append(color)

    # Plot CO2 Bars: Solid Filled
    for i in range(len(x)):
        ax.bar(
            x[i] - width/2, co2_means[i], width, yerr=co2_errs[i],
            color=window_colors[i], edgecolor='black', linewidth=1.2,
            capsize=4, error_kw={'elinewidth': 1.2, 'capthick': 1.2},
            zorder=3
        )

    # Plot H2 Bars: Hatched Texture (///) with matching voltage color
    for i in range(len(x)):
        ax.bar(
            x[i] + width/2, h2_means[i], width, yerr=h2_errs[i],
            color=window_colors[i], edgecolor='black', linewidth=1.2,
            hatch='///', capsize=4, error_kw={'elinewidth': 1.2, 'capthick': 1.2},
            zorder=3
        )

    # Axis Labels and Formatting
    ax.set_ylabel("Extensive Rate (nmol/min)", labelpad=7, fontsize=18)
    ax.set_xlabel("Applied Voltage Window (V)", labelpad=7, fontsize=18)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=12)

    # Y-axis auto-scaling with 20% overhead padding
    max_val = max(
        np.max(co2_means + co2_errs) if len(co2_means) > 0 else 0,
        np.max(h2_means + h2_errs) if len(h2_means) > 0 else 0
    )
    y_upper = max_val * 1.20 if max_val > 0 else 1.0
    ax.set_ylim(0, y_upper)

    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=6))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.tick_params(axis='both', which='major', direction='inout', length=8, width=1.5, labelsize=14)
    ax.tick_params(axis='both', which='minor', direction='in', length=4, width=1.0)

    # Frame Spine Styling
    for spine in ['top', 'bottom', 'left', 'right']:
        ax.spines[spine].set_linewidth(1.8)
        ax.spines[spine].set_zorder(5)

    # Custom Legend indicating texture representation for species
    legend_patches = [
        Patch(facecolor='lightgray', edgecolor='black', label=r'CO$_2$'),
        Patch(facecolor='lightgray', edgecolor='black', hatch='///', label=r'H$_2$')
    ]
    ax.legend(handles=legend_patches, loc='upper right', frameon=False, fontsize=13, ncol=2)
    fig.tight_layout()

    return fig, ax


def plot_selectivity_barchart(summary_rates_df, shade_colors=None, figsize=(11, 5)):
    """
    Plots a bar chart of CO2 / CO selectivity (ratio) per voltage window.
    Bar colors dynamically match the applied bias window shade.
    """
    if summary_rates_df is None or summary_rates_df.empty:
        fig, ax = plt.subplots(figsize=figsize)
        return fig, ax

    fig, ax = plt.subplots(figsize=figsize)
    shade_colors = shade_colors or {}

    x_labels = [
        f"{row.get('WE_Voltage', row.get('Voltage (V)', 0.0)):.1f}" if isinstance(row.get('WE_Voltage', row.get('Voltage (V)', 0.0)), (int, float)) else str(row.get('WE_Voltage', row.get('Voltage (V)', '0.0')))
        for _, row in summary_rates_df.iterrows()
    ]
    
    x = np.arange(len(x_labels))
    width = 0.40

    sel_means = summary_rates_df["Selectivity_mean"].values if "Selectivity_mean" in summary_rates_df.columns else pd.Series(0.0, index=summary_rates_df.index).values
    sel_errs = summary_rates_df["Selectivity_err"].values if "Selectivity_err" in summary_rates_df.columns else pd.Series(0.0, index=summary_rates_df.index).values

    window_colors = [
        shade_colors.get(row.get("WE_Voltage", row.get("Voltage (V)", 0.0)), shade_colors.get(round(row.get("WE_Voltage", row.get("Voltage (V)", 0.0)), 2) if isinstance(row.get("WE_Voltage", row.get("Voltage (V)", 0.0)), (int, float)) else 0, "#d0e1f9"))
        for _, row in summary_rates_df.iterrows()
    ]

    for i in range(len(x)):
        ax.bar(
            x[i], sel_means[i], width, yerr=sel_errs[i],
            color=window_colors[i], edgecolor='black', linewidth=1.2,
            capsize=5, error_kw={'elinewidth': 1.2, 'capthick': 1.2},
            zorder=3
        )

    ax.set_ylabel(r"Selectivity ($\mathrm{CO_2}$ Rate / CO Rate)", labelpad=7, fontsize=18)
    ax.set_xlabel("Applied Voltage Window (V)", labelpad=7, fontsize=18)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=12)

    max_val = np.max(sel_means + sel_errs) if len(sel_means) > 0 else 1.0
    ax.set_ylim(0, max_val * 1.25 if max_val > 0 else 1.0)

    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=6))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.tick_params(axis='both', which='major', direction='inout', length=8, width=1.5, labelsize=14)
    ax.tick_params(axis='both', which='minor', direction='in', length=4, width=1.0)

    for spine in ['top', 'bottom', 'left', 'right']:
        ax.spines[spine].set_linewidth(1.8)
        ax.spines[spine].set_zorder(5)

    fig.tight_layout()
    return fig, ax


'''
#############
MFC PLOTTER
#############
'''

def plot_mfc_profile(mfc_df, figsize=(11, 4)):
    fig, ax = plt.subplots(figsize=figsize)
    
    time_col = 'Elapsed_Time_hr' if 'Elapsed_Time_hr' in mfc_df.columns else ('Time (hr)' if 'Time (hr)' in mfc_df.columns else mfc_df.columns[0])
    max_time = pd.to_numeric(mfc_df[time_col], errors='coerce').max()
    if pd.isna(max_time) or max_time <= 0:
        max_time = 1.0
    
    x_interval = max(0.5, np.ceil(max_time / 10.0))
    
    n2_cols = [c for c in mfc_df.columns if 'n2' in c.lower() and 'flow' in c.lower()]
    h2_cols = [c for c in mfc_df.columns if 'h2' in c.lower() and 'flow' in c.lower()]
    co_cols = [c for c in mfc_df.columns if 'co' in c.lower() and 'flow' in c.lower() and 'co2' not in c.lower()]
    co2_cols = [c for c in mfc_df.columns if 'co2' in c.lower() and 'flow' in c.lower()]
    
    plotted_cols = set()
    
    if n2_cols:
        ax.plot(mfc_df[time_col], pd.to_numeric(mfc_df[n2_cols[0]], errors='coerce'), color='navy', lw=1.5, label=r'N2 Flow (sccm)')
        plotted_cols.add(n2_cols[0])
    if h2_cols:
        ax.plot(mfc_df[time_col], pd.to_numeric(mfc_df[h2_cols[0]], errors='coerce'), color='forestgreen', lw=1.5, label=r'H2 Flow (sccm)')
        plotted_cols.add(h2_cols[0])
    if co_cols:
        ax.plot(mfc_df[time_col], pd.to_numeric(mfc_df[co_cols[0]], errors='coerce'), color='darkorange', lw=1.8, label=r'CO Flow (sccm)')
        plotted_cols.add(co_cols[0])
    if co2_cols:
        ax.plot(mfc_df[time_col], pd.to_numeric(mfc_df[co2_cols[0]], errors='coerce'), color='crimson', lw=1.5, label=r'CO2 Flow (sccm)')
        plotted_cols.add(co2_cols[0])
        
    other_flow_cols = [c for c in mfc_df.columns if 'flow' in c.lower() and 'total' not in c.lower() and c not in plotted_cols]
    palette = ['purple', 'teal', 'magenta', 'brown', 'goldenrod']
    for idx, col in enumerate(other_flow_cols):
        c_color = palette[idx % len(palette)]
        clean_name = col.replace('_', ' ').replace('(sccm)', '').strip()
        ax.plot(mfc_df[time_col], pd.to_numeric(mfc_df[col], errors='coerce'), color=c_color, lw=1.5, label=f'{clean_name} (sccm)')
        
    ax.set_xlabel("Time (hr)", labelpad=7, fontsize=14)
    ax.set_ylabel("Flow Rate (sccm)", labelpad=7, fontsize=14)
    ax.set_xlim(0, max_time)
    
    ax.xaxis.set_major_locator(ticker.MultipleLocator(x_interval))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(x_interval / 5))
    ax.tick_params(axis='both', which='major', direction='inout', length=6, width=1.2, labelsize=12)
    
    for spine in ['top', 'bottom', 'left', 'right']:
        ax.spines[spine].set_linewidth(1.5)
        
    ax.legend(loc='best', frameon=False, fontsize=11)
    return fig, ax

'''
#####################
TEMPERATURE PLOTTER
#####################
'''

def plot_temp_profile(temp_df, figsize=(11, 4)):
    fig, ax = plt.subplots(figsize=figsize)
    
    time_col = 'Elapsed_Hours' if 'Elapsed_Hours' in temp_df.columns else 'Time (hr)'
    max_time = temp_df[time_col].max()
    
    # Calculate a dynamic x-interval based on length of run
    x_interval = max(2.0, np.ceil(max_time / 10))

    ax.plot(temp_df[time_col], temp_df['Temperature_C'], color='crimson', lw=1.5, label='Measured T (C)')
    if 'Setpoint_C' in temp_df.columns:
        ax.plot(temp_df[time_col], temp_df['Setpoint_C'], color='black', linestyle='--', lw=1.5, label='Setpoint (C)')
        
    ax.set_xlabel("Time (hr)", labelpad=7, fontsize=14)
    ax.set_ylabel("Temperature (C)", labelpad=7, fontsize=14)
    ax.set_xlim(0, max_time)
    
    ax.xaxis.set_major_locator(ticker.MultipleLocator(x_interval))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(x_interval / 5))
    ax.tick_params(axis='both', which='major', direction='inout', length=6, width=1.2, labelsize=12)
    
    for spine in ['top', 'bottom', 'left', 'right']:
        ax.spines[spine].set_linewidth(1.5)
        
    ax.legend(loc='upper right', frameon=False, fontsize=11)
    return fig, ax


'''
##########################
SYRINGE PUMP PLOTTER
##########################
'''

def plot_pump_profile(pump_df, figsize=(11, 4)):
    """
    Plots Syringe Pump Infusion Rate and Dispensed Volume over time.
    """
    fig, ax1 = plt.subplots(figsize=figsize)
    if pump_df is None or pump_df.empty:
        return fig, ax1

    import re
    time_col = 'Elapsed_Hours' if 'Elapsed_Hours' in pump_df.columns else ('Time (hr)' if 'Time (hr)' in pump_df.columns else pump_df.columns[0])
    time_hours = pd.to_numeric(pump_df[time_col], errors='coerce').fillna(0.0)
    max_time = time_hours.max()
    if max_time <= 0:
        max_time = 1.0

    x_interval = max(0.5, np.ceil(max_time / 10.0))

    def _parse_rate_ul(val):
        if pd.isna(val):
            return 0.0
        s = str(val).lower()
        m = re.search(r'([0-9]*\.?[0-9]+)', s)
        if not m:
            return 0.0
        num = float(m.group(1))
        if 'ml/min' in s:
            return num * 1000.0
        elif 'nl/min' in s:
            return num / 1000.0
        return num

    def _parse_vol_ml(val):
        if pd.isna(val):
            return 0.0
        s = str(val).lower()
        m = re.search(r'([0-9]*\.?[0-9]+)', s)
        if not m:
            return 0.0
        num = float(m.group(1))
        if 'ul' in s or 'µl' in s:
            return num / 1000.0
        elif 'nl' in s:
            return num / 1e6
        return num

    rate_col = 'Target_Rate' if 'Target_Rate' in pump_df.columns else pump_df.columns[2]
    vol_col = 'Dispensed_Volume' if 'Dispensed_Volume' in pump_df.columns else pump_df.columns[3]

    rates_ul = pump_df[rate_col].apply(_parse_rate_ul)
    vols_ml = pump_df[vol_col].apply(_parse_vol_ml)

    line1 = ax1.plot(time_hours, rates_ul, color='darkviolet', lw=1.8, label='FA Infusion Rate (uL/min)')
    ax1.set_xlabel("Time (hr)", labelpad=7, fontsize=14)
    ax1.set_ylabel("FA Infusion Rate (uL/min)", color='darkviolet', fontsize=14)
    ax1.tick_params(axis='y', labelcolor='darkviolet', labelsize=12)
    ax1.set_xlim(0, max_time)
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(x_interval))
    ax1.xaxis.set_minor_locator(ticker.MultipleLocator(x_interval / 5))
    ax1.tick_params(axis='x', which='major', direction='inout', length=6, width=1.2, labelsize=12)

    ax2 = ax1.twinx()
    line2 = ax2.plot(time_hours, vols_ml, color='teal', linestyle='--', lw=1.6, label='Dispensed Volume (mL)')
    ax2.set_ylabel("Dispensed Volume (mL)", color='teal', fontsize=14)
    ax2.tick_params(axis='y', labelcolor='teal', labelsize=12)

    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left', frameon=False, fontsize=11)

    for spine in ['top', 'bottom', 'left', 'right']:
        ax1.spines[spine].set_linewidth(1.5)

    fig.tight_layout()
    return fig, ax1