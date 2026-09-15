import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from arrhenius_worker_files.arrhenius_config import get_step_at_time

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

def plot_gas_rates(rates_df, df_electronics, x_interval=6.0, y_interval=5.0, 
                   figsize=(11, 5), shade_colors=None, RXN_SCHEDULE=None, bias_schedule=None): 
    
    # Backwards-compatibility for parameter naming
    schedule = RXN_SCHEDULE if RXN_SCHEDULE is not None else bias_schedule
    
    fig, ax = plt.subplots(figsize=figsize)
    shade_colors = shade_colors or {}
    
    has_rates = rates_df is not None and not rates_df.empty and "Time (hr)" in rates_df.columns
    has_elec = df_electronics is not None and not df_electronics.empty and "Time (hr)" in df_electronics.columns

    if not has_rates and not has_elec:
        return fig, ax

    # 1. Plot Gas Reaction Rates
    if has_rates:
        for col, marker, color, label in [("CO2 (nmol/min)", 's', 'orange', r'CO$_2$'),
                                          ("CO (nmol/min)", '^', 'blue', 'CO'),
                                          ("H2 (nmol/min)", 'D', 'red', r'H$_2$')]:
            if col in rates_df.columns:
                ax.plot(rates_df["Time (hr)"], rates_df[col], linestyle='None',
                        marker=marker, markersize=6, label=label, markerfacecolor=color,
                        markeredgewidth=1, markeredgecolor='black', zorder=3)

    # 2. Plot Electron Transfer Rate from Electronics
    if has_elec and "current/2 (nmol/min)" in df_electronics.columns:
        ax.plot(df_electronics["Time (hr)"], df_electronics["current/2 (nmol/min)"],
                linestyle='-', linewidth=2, label='Rate of Electron Transfer (|e|/2)', color='orange', zorder=4)
    
    # 3. Axis Limits & Formatting
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

    # 4. Background Shading using time_window boundaries
    unique_biases = []
    if schedule:
        for step in schedule:
            if not isinstance(step, dict) or "time_window" not in step:
                continue
                
            start_hr, end_hr = step["time_window"]
            bias_val = step.get("we_volt", 0.0)
            
            # Lookup color matching float or string representation
            color = shade_colors.get(bias_val, shade_colors.get(str(bias_val), 'white'))
            
            # Draw vertical shaded block across full step duration
            ax.axvspan(start_hr, end_hr, color=color, alpha=1.0, zorder=1, lw=0)
            
            if bias_val not in unique_biases:
                unique_biases.append(bias_val)

    # 5. Draw Double Legends
    species_handles, species_labels = ax.get_legend_handles_labels()

    if species_handles:
        species_legend = ax.legend(handles=species_handles, labels=species_labels, bbox_to_anchor=(0.5, -0.15), 
                                   loc='upper center', columnspacing=0.7, fontsize=13, ncol=len(species_labels), frameon=False)
        ax.add_artist(species_legend)

    if unique_biases:
        bias_patches = [
            Patch(
                facecolor=shade_colors.get(b, shade_colors.get(str(b), 'white')), 
                edgecolor='black', 
                label=f"{b} V" if isinstance(b, (int, float)) else f"{b}"
            ) 
            for b in unique_biases
        ]

        # Bias legend sitting above top axis boundary
        ax.legend(handles=bias_patches, bbox_to_anchor=(0.5, 1.00), loc="lower center", 
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


import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def plot_arrhenius_by_voltage(arrhenius_results, target_voltage, plots_dir):
    """
    Generates and saves an Arrhenius plot for a single target voltage bias (including OCP).
    """
    # Normalize voltage key lookup
    if isinstance(target_voltage, str) and target_voltage.upper() == "OCP":
        v_key = "OCP"
        v_label = "OCP"
        filename_str = "OCP"
    else:
        try:
            v_key = float(target_voltage)
            v_label = f"{v_key:.1f} V"
            filename_str = f"{v_key:.1f}V"
        except (ValueError, TypeError):
            v_key = str(target_voltage)
            v_label = str(target_voltage)
            filename_str = str(target_voltage)

    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    if v_key not in arrhenius_results or not arrhenius_results[v_key]:
        return None, None, None

    sp_dict = arrhenius_results[v_key]

    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    
    species_colors = {"CO2": "tab:blue", "H2": "tab:orange", "CO": "tab:green"}
    species_markers = {"CO2": "o", "H2": "s", "CO": "^"}

    has_data = False
    for species, fit in sp_dict.items():
        if "inv_T" not in fit or "ln_rate" not in fit:
            continue
            
        inv_T = fit["inv_T"]
        ln_rate = fit["ln_rate"]
        color = species_colors.get(species, "tab:grey")
        marker = species_markers.get(species, "o")

        ax.scatter(
            inv_T * 1000, 
            ln_rate, 
            label=f"{species} (R² = {fit['R2']:.3f})", 
            color=color, 
            marker=marker, 
            s=50, 
            zorder=3
        )

        x_line = np.linspace(min(inv_T), max(inv_T), 50)
        y_line = fit["slope"] * x_line + fit["intercept"]
        ax.plot(x_line * 1000, y_line, color=color, linestyle="--", linewidth=1.5, alpha=0.8, zorder=2)
        
        has_data = True

    if not has_data:
        plt.close(fig)
        return None, None, None

    ax.set_xlabel("1000 / T (1/K)", fontsize=10, fontweight="bold")
    ax.set_ylabel(f"ln(Rate / nmol min^{-1})", fontsize=10, fontweight="bold")
    ax.set_title(f"Arrhenius Plot at {v_label} WE Bias", fontsize=11, fontweight="bold")
    ax.legend(loc="best", fontsize=9, frameon=False)
    plt.tight_layout()

    plot_file_path = plots_dir / f"arrhenius_{filename_str}.png"
    fig.savefig(plot_file_path, dpi=300, bbox_inches="tight")
    
    return fig, ax, plot_file_path
