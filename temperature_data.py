import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from background_config import TEMP_SCHEDULE
import scipy.stats as stats
import matplotlib.ticker as ticker

# Read structural reaction datasets using the newly selected paths
t_filepath = 'background_run_20260615_170057.csv'
raw_temperature_df = pd.read_csv(t_filepath)
raw_temperature_df['Time (hr)'] = raw_temperature_df['Elapsed_Minutes'] / 60

exp_length = raw_temperature_df['Step'].unique()
step_df = raw_temperature_df[raw_temperature_df['Step'] == 2]

def temp_plotter(df, x_interval=10, y_interval=6, figsize=(11,5)):
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(df["Time (hr)"], df["Temperature_C"], linestyle='None',
            marker='s', markersize=6, label=r'T (C)', markerfacecolor='red', 
            markeredgewidth=1, markeredgecolor='black', zorder=3)
    ax.plot(df["Time (hr)"], df["Setpoint_C"], linestyle='-',
        color = 'black', label=r'Setpoint (C)', zorder=3
    )

    # 2. Set Y-axis limit based on Max temp
    max_temp = df["Temperature_C"].max()
    ax.set_ylim(0, max_temp*1.2)
    ax.set_xlim(0, df["Time (hr)"].max())

    # 3. Labeling and Axis Formatting
    plt.xlabel("Time (hr)\n\n", labelpad=7, fontsize=18)
    plt.ylabel("Temperature (C)", labelpad=7, fontsize=20)

# Tick marks configuration
    ax.xaxis.set_major_locator(ticker.MultipleLocator(x_interval))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(x_interval/5))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(y_interval))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(y_interval/5))

    ax.tick_params(axis='both', which='major', direction='inout', length=8, width=1.5, labelsize=16)
    ax.tick_params(axis='both', which='minor', direction='in', length=4, width=1.0)

    # Thick borders around the plot frame
    for spine in ['top', 'bottom', 'left', 'right']:
        ax.spines[spine].set_linewidth(1.8)
        ax.spines[spine].set_zorder(5) # Keeps borders on top of shaded background

    # 5. Draw double legends
    species_handles, species_labels = ax.get_legend_handles_labels()

    # Species legend sitting below the bottom axis line
    species_legend = ax.legend(handles=species_handles, labels=species_labels,bbox_to_anchor=(0.5, -0.15), 
                               loc='upper center', columnspacing=0.7, fontsize=13, ncol=len(species_labels), frameon=False)
    ax.add_artist(species_legend)


    plt.show()
    return fig, ax





print(step_df.head())
print(exp_length)
fig, ax = temp_plotter(step_df)
