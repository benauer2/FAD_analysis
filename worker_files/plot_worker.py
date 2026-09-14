import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from config import get_step_at_time

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
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(x_interval/6.0))
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
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(x_interval/6.0))
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
    
    time_col = 'Elapsed_Time_hr' if 'Elapsed_Time_hr' in mfc_df.columns else 'Time (hr)'
    max_time = mfc_df[time_col].max()
    
    # Calculate dynamic x-interval
    x_interval = max(2.0, np.ceil(max_time / 10))
    
    n2_col = [c for c in mfc_df.columns if 'N2' in c and 'Flow' in c]
    h2_col = [c for c in mfc_df.columns if 'H2' in c and 'Flow' in c]
    
    if n2_col:
        ax.plot(mfc_df[time_col], mfc_df[n2_col[0]], color='navy', lw=1.5, label=r'N2 Flow (sccm)')
    if h2_col:
        ax.plot(mfc_df[time_col], mfc_df[h2_col[0]], color='forestgreen', lw=1.5, label=r'H2 Flow (sccm)')
        
    ax.set_xlabel("Time (hr)", labelpad=7, fontsize=14)
    ax.set_ylabel("Flow Rate (sccm)", labelpad=7, fontsize=14)
    ax.set_xlim(0, max_time)
    
    ax.xaxis.set_major_locator(ticker.MultipleLocator(x_interval))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(x_interval / 5))
    ax.tick_params(axis='both', which='major', direction='inout', length=6, width=1.2, labelsize=12)
    
    for spine in ['top', 'bottom', 'left', 'right']:
        ax.spines[spine].set_linewidth(1.5)
        
    ax.legend(loc='upper right', frameon=False, fontsize=11)
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

    if df_electronics is None:
        if rates_df is not None:
            df_electronics = rates_df
            rates_df = None
        else:
            raise ValueError("df_electronics must be provided to generate the plot")

    fig, ax = plt.subplots(figsize=figsize)

    # 1. Plot Data with strict Z-ordering (Lines/Markers on top of background shading)
    ax.plot(df_electronics["Time (hr)"], df_electronics["Current (uA)"], 
            linestyle='-', linewidth=2, label='Current', color='black', zorder=4)
    
    # 2. Set Y-axis limit based on CO2 and CO rates (ignores H2)

    # Calculate min and max
    y_min = df_electronics["Current (uA)"].min()
    y_max = df_electronics["Current (uA)"].max()

    # Check if the values are valid numbers (not NaN or Inf) and that the dataframe isn't empty
    if np.isfinite(y_min) and np.isfinite(y_max):
        # Only scale if they aren't both zero to avoid flat-line issues
        ax.set_ylim(y_min * 1.2, y_max * 1.2)

    ax.set_xlim(0, df_electronics["Time (hr)"].max())

    # 3. Labeling and Axis Formatting
    plt.xlabel("Time (hr)\n\n", labelpad=7, fontsize=18)
    plt.ylabel("Current (uA)", labelpad=7, fontsize=20)

    # Tick marks configuration
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
    bias_values = rates_df['Time (hr)'].apply(
        lambda t: get_step_at_time(t, rxn_schedule))
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
                        facecolor=shade_colors[b], 
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

def cap_plot(cv_segments,  x_interval=3.0, y_interval=None, figsize=(11, 5), shade_colors=None, rxn_schedule=None):
        # 1. Specify the sweep rate you want to isolate (e.g., 0.5, 1.0, or 1.5 V/s)
    target_sweep_rate = 1.0 

    # 2. Extract the run ID and capacitance for the target sweep rate
    runs = []
    capacitances = []
    times = []

    for cv_id, cv_df in cv_segments.items():
        # Grab the sweep rate from the first row of this segment
        current_rate = cv_df['sweep_rate'].iloc[0]
        
        # Only collect data if it matches the specified sweep rate
        if current_rate == target_sweep_rate:
            runs.append(cv_id)
            times.append(cv_df['Time (hr)'].iloc[0])
            # Grab the calculated capacitance from the first row
            capacitances.append(cv_df['capacitance (uF)'].iloc[0])

    # 3. Create the plot
    fig, ax = plt.subplots(figsize=(11, 5))

    if shade_colors is None:
        shade_colors = {
            0: "#ffffff",    # Neutral soft gray (0 V)
            1: '#d0e1f9',   # Soft blue (-1 V)
            -1: '#f9d0d0',    # Soft red (+1 V)
        }

    cum_time = 0.0
    unique_biases = []

    for block in rxn_schedule:
        v_bias = block["voltage"]
        duration = block["duration_hours"]
        start_t = cum_time
        end_t = cum_time + duration
        cum_time = end_t

        color = shade_colors.get(v_bias, "#f0f0f0")
        ax.axvspan(start_t, end_t, color=color, alpha=1.0, zorder=1, lw=0)

        if v_bias not in unique_biases:
            unique_biases.append(v_bias)

    ax.plot(times, capacitances, marker='o', linestyle='None', color='black', linewidth=2, markersize=8)

    max_time = max(cum_time, max(times) if times else 0)
    ax.set_xlim(0, max_time)

    # 4. Format and style the plot
    ax.set_xlabel('Time (hr)', fontsize=18)
    ax.set_ylabel('Capacitance (uF)', fontsize=18)

        # Tick marks configuration
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

    # Thick borders around the plot frame
    for spine in ['top', 'bottom', 'left', 'right']:
        ax.spines[spine].set_linewidth(1.8)
        ax.spines[spine].set_zorder(5) # Keeps borders on top of shaded background

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
    '''
    # Add data labels next to the points so you can easily read exact values
    for i, txt in enumerate(capacitances):
        plt.annotate(f'{txt:.3}', (times[i], capacitances[i]), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)
    '''
    

    return fig, ax


def cv_plot(cv_segments, rxn_schedule=None, max_per_page=7):
    cv_items = list(cv_segments.items())
    chunk_size = 3
    cv_chunks = [cv_items[i:i + chunk_size] for i in range(0, len(cv_items), chunk_size)]

    page_chunks = [cv_chunks[i : i + max_per_page]
                   for i in range(0, len(cv_chunks), max_per_page)]
    
    figs = []
    colors = ['crimson', 'dodgerblue', 'forestgreen']
    first_cv_id = cv_items[0][0] if cv_items else None

    cols = 2

    for page_idx, current_page_chunks in enumerate(page_chunks):
        rows = int(np.ceil(len(current_page_chunks) / cols)) # Fixed to handle odd chunk counts gracefully
        fig, axes = plt.subplots(rows, cols, figsize=(11, 2.2 * rows), squeeze=False)
        axes = axes.flatten()

        for group_idx, chunk in enumerate(current_page_chunks):
            ax = axes[group_idx]
            
            for idx, (cv_id, cv_df) in enumerate(chunk):
                voltage = cv_df.iloc[:, 2]
                current = cv_df.iloc[:, 3] * 1E6
                
                sweep_rate = cv_df['sweep_rate'].iloc[0]
                color = colors[idx % len(colors)]
                label_text = f'{sweep_rate} V/s'
                
                ax.plot(voltage, current, color=color, lw=1.5, label=label_text)

            first_df = chunk[0][1]
            raw_elapsed_hr = first_df['Time (hr)'].iloc[0]
            elapsed_hr = round(raw_elapsed_hr, 1)
            
            if chunk[0][0] == first_cv_id:
                prior_condition = "Initial Setup"
                title_text = f"Time: {elapsed_hr: .1f} hr, Initial Setup CVs"
            else:
                prior_condition = get_step_at_time(max(raw_elapsed_hr - 0.5, 0), rxn_schedule)
                title_text = f"Time: {elapsed_hr: .1f} hr, Post {prior_condition} window"

            # 1. Title and basic spine setup
            start_id = chunk[0][0]
            end_id = chunk[-1][0]
            ax.set_title(title_text, fontsize=11, fontweight='bold')
            
            for spine in ['left', 'right', 'top', 'bottom']:
                ax.spines[spine].set_linewidth(1.5)
                ax.spines[spine].set_color('black')

            # 2. Outer Border Ticks Setup (Labels stay out here!)
            ax.tick_params(axis='both', direction='in', length=0, width=1.5, labelsize=9)
            ax.set_xlabel('Potential (V)', fontsize=11)
            ax.set_ylabel(r'Current ($\mu$A)', fontsize=11) # Cleaned up micro symbol

            # 3. Dynamic layout update to ensure limits are locked before we draw center ticks
            ax.autoscale(enable=True, axis='both')
            fig.canvas.draw() # Forces matplotlib to calculate the actual data limits
            
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()

            # 4. Draw bold center crosshairs
            ax.axhline(0, color='black', linewidth=1.5, zorder=1)
            ax.axvline(0, color='black', linewidth=1.5, zorder=1)

            # 5. GENERATE CENTER TICKS (Labels outside, tick marks inside!)
            # Define tick intervals based on your data scale
            x_ticks = ax.get_xticks()
            y_ticks = ax.get_yticks()
            
            # Calculate a proportional tick length for the crosshairs
            tick_len_x = (ylim[1] - ylim[0]) * 0.02 
            tick_len_y = (xlim[1] - xlim[0]) * 0.02

            # Draw ticks along the X-centerline (Vertical ticks on the horizontal axis)
            for tick in x_ticks:
                if tick != 0 and xlim[0] <= tick <= xlim[1]:
                    tick_line = Line2D([tick, tick], [-tick_len_x, tick_len_x], color='black', linewidth=1.5, zorder=2)
                    ax.add_line(tick_line)

            # Draw ticks along the Y-centerline (Horizontal ticks on the vertical axis)
            for tick in y_ticks:
                if tick != 0 and ylim[0] <= tick <= ylim[1]:
                    tick_line = Line2D([-tick_len_y, tick_len_y], [tick, tick], color='black', linewidth=1.5, zorder=2)
                    ax.add_line(tick_line)

            ax.legend(loc='best', fontsize=9, frameon=False)

        # Hide unused axes
        for j in range(group_idx + 1, len(axes)):
            axes[j].axis('off')
        
        plt.subplots_adjust(hspace=0.45, wspace=0.3)
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


def plot_10CV(df, title_name, ax):
    """
    Plots an overlay of all 10 sweep rates onto a single provided axis (ax).
    """
    # 1. Get unique steps from the passed dataframe
    unique_steps = sorted(df["Step number"].unique())

    # 2. 10 custom hex codes transitioning smoothly from Dark Red -> Purple -> Deep Blue
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

    # 3. Calculate sweep rates assuming steps 1-10 map to 0.1-1.0 V/s
    sweep_rates = [step * 0.1 for step in unique_steps]

    # 4. Loop and plot ALL sweeps onto the SAME ax
    for step, color, rate in zip(unique_steps, red_to_blue, sweep_rates):
        step_data = df[df["Step number"] == step]
        step_data = step_data.sort_values("Elapsed Time (s)")

        ax.plot(
            step_data["Working Electrode (V)"],
            step_data["Current (A)"],
            linestyle="-",
            linewidth=1.5,
            label=f"{rate:.1f} V/s",
            color=color,
        )

    ax.set_xlim(-1.5, 1.5)
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))

    for spine in ["top", "bottom", "left", "right"]:
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_color("black")
        ax.spines[spine].set_linewidth(1.0)

    ax.axhline(0, color="black", linewidth=1.0, zorder=1)
    ax.axvline(0, color="black", linewidth=1.0, zorder=1)

    # Create dummy secondary axes fixed at y=0 and x=0 to draw the inner tick marks
    secx = ax.secondary_xaxis(location=0)
    secy = ax.secondary_yaxis(location=0)

    # Enable inner tick marks across the zero lines, but suppress their numbers/labels
    secx.tick_params(
        axis="x", direction="inout", length=6, width=1.0, labelbottom=False
    )
    secy.tick_params(
        axis="y", direction="inout", length=6, width=1.0, labelleft=False
    )

    # Hide the bounding spines of the dummy secondary axes so only ticks remain
    for s in secx.spines.values():
        s.set_visible(False)
    for s in secy.spines.values():
        s.set_visible(False)


    # Style the single subplot axis container
    ax.set_title(title_name, fontsize=11, fontweight='bold', pad=8)
    ax.set_xlabel("Voltage (V)", fontsize=9, labelpad=5)
    ax.set_ylabel("Current (A)", fontsize=9, labelpad=5)
    
    # Optional small legend for individual subplots if needed; 
    # remove or comment out if it clutters the grid.
    ax.legend(title="Sweep Rate", fontsize=7, title_fontsize=8, loc="best", frameon=False)

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
        rates_df["CO2 (nmol/min)"].max(),
        h2_max
    )
    
    # Handle edge case if max_rate is zero or NaN
    if not np.isfinite(max_rate) or max_rate <= 0:
        max_rate = 10.0

    ax1.set_ylim(0, max_rate * 1.15)
    ax1.set_xlim(0, relative_time.max())

    # 3. Dynamic y_interval scaling to prevent "tick blob" on ax1
    if y_interval is None or (max_rate / y_interval) > 10:
        raw_step = max_rate / 5.0
        magnitude = 10 ** np.floor(np.log10(raw_step)) if raw_step > 0 else 1.0
        y_interval = np.ceil(raw_step / magnitude) * magnitude

    # 4. Axis Labels
    ax1.set_xlabel("Time (hr)\n\n", labelpad=7, fontsize=18)
    ax1.set_ylabel("Extensive Rate (nmol/min)", labelpad=7, fontsize=20)
    ax2.set_ylabel('Formic Acid (%)', labelpad=7, fontsize=20)

    # 5. Locators
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(x_interval))
    ax1.xaxis.set_minor_locator(ticker.MultipleLocator(x_interval / 6))

    ax1.yaxis.set_major_locator(ticker.MultipleLocator(y_interval))
    ax1.yaxis.set_minor_locator(ticker.MultipleLocator(y_interval / 2))

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
        f"{row['WE_Voltage']:.1f}"
        for _, row in summary_rates_df.iterrows()
    ]
    
    x = np.arange(len(x_labels))
    width = 0.30  # Adjusted width for 2 primary gas species (CO2 and H2)

    # Extract species data and uncertainties
    co2_means = summary_rates_df["CO2_avg"].values
    co2_errs = summary_rates_df["CO2_std"].values

    h2_means = summary_rates_df["H2_avg"].values
    h2_errs = summary_rates_df["H2_std"].values

    # Determine per-window background colors matching bias voltage
    window_colors = []
    for _, row in summary_rates_df.iterrows():
        v = row["WE_Voltage"]
        # Handle floating point key lookup or default soft blue/gray
        color = shade_colors.get(v, shade_colors.get(round(v, 2), "#d0e1f9"))
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
        f"{row['Voltage (V)']:.1f}"
        for _, row in summary_rates_df.iterrows()
    ]
    
    x = np.arange(len(x_labels))
    width = 0.40

    sel_means = summary_rates_df["Selectivity_mean"].values
    sel_errs = summary_rates_df["Selectivity_err"].values

    window_colors = [
        shade_colors.get(row["Voltage (V)"], shade_colors.get(round(row["Voltage (V)"], 2), "#d0e1f9"))
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

def plot_rate_vs_capacitance(window_avg_corr_df, figsize=(9, 4.5)):
    fig, ax = plt.subplots(figsize=figsize)

    if window_avg_corr_df is None or window_avg_corr_df.empty:
        return fig, ax

    species_config = [
        ("CO2_mean", "CO2_err", 's', 'orange', r'CO$_2$'),
        ("H2_mean", "H2_err", 'D', 'red', r'H$_2$'),
        ("CO_mean", "CO_err", '^', 'blue', 'CO')
    ]

    # Draw tie-lines connecting capacitance points within the same voltage window
    for win_id, win_group in window_avg_corr_df.groupby("Window"):
        if len(win_group) > 1:
            x_vals = win_group["Capacitance (uF)"].values
            for mean_col, _, _, _, _ in species_config:
                if mean_col in win_group.columns:
                    y_val = win_group[mean_col].iloc[0]
                    ax.plot(x_vals, [y_val] * len(x_vals), color='gray', linestyle=':', alpha=0.5, zorder=1)

    # Plot rate vs capacitance scatter points with error bars
    for mean_col, err_col, marker, color, label in species_config:
        if mean_col in window_avg_corr_df.columns:
            ax.errorbar(
                window_avg_corr_df["Capacitance (uF)"], 
                window_avg_corr_df[mean_col], 
                yerr=window_avg_corr_df[err_col] if err_col in window_avg_corr_df.columns else None,
                fmt=marker, color=color, ecolor='black',
                markeredgecolor='black', markeredgewidth=1.0,
                markersize=7, capsize=3, elinewidth=1.0,
                label=label, zorder=3
            )

    # Annotate points with Voltage Bias from the window schedule
    for _, row in window_avg_corr_df.iterrows():
        ax.annotate(
            f"{row['Voltage (V)']:.2f}V",
            (row["Capacitance (uF)"], row["CO2_mean"]),
            textcoords="offset points",
            xytext=(0, 6),
            ha='center',
            fontsize=8,
            alpha=0.85
        )

    ax.set_xlabel(r"Capacitance ($\mu$F)", fontsize=14, labelpad=8)
    ax.set_ylabel("Average Rate (nmol/min)", fontsize=14, labelpad=8)
    ax.set_title("Average Reaction Rates vs. Window Capacitances (via Bias Schedule)", fontsize=12, fontweight='bold', pad=10)

    # Formatting
    ax.tick_params(axis='both', which='major', direction='inout', length=7, width=1.3, labelsize=11)
    ax.tick_params(axis='both', which='minor', direction='in', length=3.5, width=0.8)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

    for spine in ['top', 'bottom', 'left', 'right']:
        ax.spines[spine].set_linewidth(1.5)

    ax.grid(True, linestyle='--', alpha=0.4, zorder=1)
    ax.legend(frameon=False, fontsize=11, loc='best')

    fig.tight_layout()
    return fig, ax

