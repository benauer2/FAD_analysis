"""
PDF Report Generator for Formic Acid Decomposition Trials (OOP Refactored Test Version)
Author: Ben Auer
Some code adapted from Justin Hopkins.
"""
from datetime import date
from pathlib import Path
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
from worker_files.electronics_worker import (
    EIS_cap_pre,
    EIS_cap_rxn,
    calc_cap_from_cv,
    calc_cap_from_cv_grouped,
    get_cv,
    getcstats,
)
from worker_files.file_utils import select_folder_and_load_data
from worker_files.GC_worker import calc_rxn_rates, compute_schedule_summary
from worker_files.plot_worker import (
    cap_plot,
    cv_plot,
    plot_10CV,
    plot_current_time,
    plot_formic_acid,
    plot_gas_rates,
    plot_OCP,
    plot_voltage_rates_barchart,
    plot_mfc_profile,
    plot_temp_profile,
    plot_pump_profile
)
from worker_files.pdf_classes import ReactionReport, safe_str

# ---------------------------------------------------------
# DATA ACQUISITION & IMPORTING
# ---------------------------------------------------------
dfs_dictionary, pre_reaction_df, selected_dir = select_folder_and_load_data()
if dfs_dictionary is None:
    sys.exit("Data loading failed. Exiting script.")

GC_status = "FAD_df" in dfs_dictionary and dfs_dictionary["FAD_df"] is not None
electronics_status = "electronics_df" in dfs_dictionary and dfs_dictionary["electronics_df"] is not None
eis_status = "EIS_df" in dfs_dictionary and dfs_dictionary["EIS_df"] is not None
pre_rxn_status = isinstance(pre_reaction_df, dict) and len(pre_reaction_df) > 0

GC_df = dfs_dictionary.get("FAD_df")
raw_electronics_df = dfs_dictionary.get("electronics_df")
eis_df = dfs_dictionary.get("EIS_df")
temp_df = dfs_dictionary.get("temp_df")
mfc_df  = dfs_dictionary.get("mfc_df")
pump_df = dfs_dictionary.get("pump_df")

# Load configuration from YAML & sequence CSV
config.handle_yaml(selected_dir)
BASE_DIR = config.BASE_DIR
notes = config.notes
analysis = config.analysis
SHADE_COLORS = config.SHADE_COLORS
TRIAL_INFO = config.TRIAL_INFO
GC_METHOD_INFO = config.GC_METHOD_INFO
RXN_SCHEDULE = config.RXN_SCHEDULE

plots_dir = selected_dir / "plots"
data_dir = selected_dir / "data"
plots_dir.mkdir(parents=True, exist_ok=True)
data_dir.mkdir(parents=True, exist_ok=True)

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Calibri']

# ---------------------------------------------------------
# SYSTEM PARAMETERS
# ---------------------------------------------------------
p_reac_f = float(TRIAL_INFO.get("P reactor (psig)", 0.0))
v_box_t = float(GC_METHOD_INFO.get("Valve box T (deg C)", 100.0))

# ---------------------------------------------------------
# PROCESSING OF GC AND ELECTRONICS DATA
# ---------------------------------------------------------
cv_status = True
capacitance = pd.DataFrame()
df_electronics = getcstats(df_raw=raw_electronics_df, file_path=None, data_dir=None, save_to_data_dir=False, rxn_schedule=RXN_SCHEDULE)
if df_electronics is not None and not df_electronics.empty:
    df_electronics.to_csv(data_dir / 'current_vs_time.csv', index=False)

try:
    df_cv, cv_seg = get_cv(raw_electronics_df)
    if cv_seg:
        capacitance = calc_cap_from_cv(cv_seg, rxn_schedule=RXN_SCHEDULE)
        if capacitance.empty:
            cv_status = False
    else:
        print("[WARNING] No CV segments extracted from raw_electronics_df.")
        cv_status = False
except Exception as e:
    print(f"[ERROR] Failed to process CV data: {e}")
    cv_status = False

# ---------------------------------------------------------
# PROCESSING OF IN-SITU REACTION EIS DATA
# ---------------------------------------------------------
rxn_eis_plots = {}
rxn_eis_summary = pd.DataFrame()
if eis_status and eis_df is not None and not eis_df.empty:
    try:
        rxn_eis_plots, rxn_eis_summary = EIS_cap_rxn(eis_df, "Reaction EIS", rxn_schedule=RXN_SCHEDULE)
        if not rxn_eis_summary.empty:
            rxn_eis_summary.to_csv(data_dir / 'rxn_eis_summary.csv', index=False)
    except Exception as e:
        print(f"[WARNING] In-situ reaction EIS processing encountered: {e}")

OCP_status = (df_electronics['Step name'] == 'Open Circuit Potential').any() if (df_electronics is not None and 'Step name' in df_electronics.columns) else False

rates_df = calc_rxn_rates(GC_df=GC_df, RXN_SCHEDULE=RXN_SCHEDULE, v_box_t=v_box_t, p_reac_f=p_reac_f, GC_METHOD_INFO=GC_METHOD_INFO)
rates_df.to_csv(data_dir / "rates_vs_time.csv", index=False)

summary_rates_df = compute_schedule_summary(rates_df, RXN_SCHEDULE)
summary_rates_df.to_csv(data_dir / 'avg_rates.csv', index=False)

# ---------------------------------------------------------
# PRE-REACTION CAPACITANCE EXTRACTION (LATEST CV IN N2)
# ---------------------------------------------------------
target_key = None
max_version = -1

if pre_rxn_status:
    for key in pre_reaction_df.keys():
        if "CV" in key.upper() and "N2" in key.upper():
            match = re.search(r'_v(\d+)$', key)
            if match:
                version_num = int(match.group(1))
                if version_num > max_version:
                    max_version = version_num
                    target_key = key
            elif max_version == -1:
                target_key = key

if target_key is not None:
    cv_summary_df = calc_cap_from_cv_grouped({target_key: pre_reaction_df[target_key]})
    if not cv_summary_df.empty:
        sweep_1_df = cv_summary_df[cv_summary_df["Sweep Rate (V/s)"] == 1.0]
        if not sweep_1_df.empty:
            pre_rxn_cap = sweep_1_df["Capacitance (uF)"].iloc[-1]
        else:
            pre_rxn_cap = cv_summary_df["Capacitance (uF)"].iloc[-1]
    else:
        pre_rxn_cap = 0.0
else:
    print("Warning: No pre-reaction CV folder in N2 to pull capacitance from.")
    pre_rxn_cap = 0.0

# ==========================================
# REPORT COMPILATION INFRASTRUCTURE (OOP)
# ==========================================
report = ReactionReport(
    target_dir=selected_dir,
    trial_info=TRIAL_INFO,
    gc_method_info=GC_METHOD_INFO
)

# ---------------------------------------------------------
# PAGE 1: KINETICS & RATE DATA
# ---------------------------------------------------------
fig_rate, ax_rate = plot_gas_rates(
    rates_df=rates_df,
    df_electronics=df_electronics,
    rxn_schedule=RXN_SCHEDULE,
    shade_colors=SHADE_COLORS,
    x_interval=8
)

report.build_metadata_page(
    notes=notes,
    analysis=analysis,
    pre_rxn_cap=pre_rxn_cap,
    rate_fig=fig_rate
)

# ---------------------------------------------------------
# PAGE 2: FORMIC ACID & CURRENT DATA
# ---------------------------------------------------------
report.add_page()
fig_fa, _ = plot_formic_acid(rates_df, shade_colors=SHADE_COLORS)
report.add_plot_figure(fig_fa, filename="FA_data.png", title="Formic Acid Concentration Plot", width=180, ln_after=105)

fig_curr, _ = plot_current_time(rates_df, df_electronics, rxn_schedule=RXN_SCHEDULE, shade_colors=SHADE_COLORS)
report.add_plot_figure(fig_curr, filename="current_data.png", title="Current vs. Time Profile", width=180, ln_after=0)

if OCP_status:
    report.add_page()
    fig_ocp, _ = plot_OCP(rates_df, df_electronics, shade_colors=SHADE_COLORS, rxn_schedule=RXN_SCHEDULE)
    report.add_plot_figure(fig_ocp, filename="OCP_data.png", title="Open Circuit Potential (OCP) Plot", width=180, ln_after=0)

# ---------------------------------------------------------
# PAGE 3: VOLTAGE WINDOW RATES BAR CHART & SUMMARY TABLE
# ---------------------------------------------------------
if not summary_rates_df.empty:
    report.add_page()
    fig_bar, _ = plot_voltage_rates_barchart(summary_rates_df, shade_colors=SHADE_COLORS)
    report.add_plot_figure(fig_bar, filename="voltage_rates_barchart.png", title="Voltage Window Reaction Rate Averages", width=180, ln_after=100)

    report.add_title_block("Steady-State Rate Summary & Schedule Table")
    
    col_widths = [10, 22, 22, 16, 22, 22, 22, 22, 22]
    headers = ["Run", "Rxr T (C)", "FA (uL/m)", "WE (V)", "Avg Win", "Run Time", "CO2 (nmol)", "H2 (nmol)", "CO (nmol)"]
    
    render_summary = pd.DataFrame()
    render_summary["Run"] = summary_rates_df.get('Run', '')
    render_summary["Rxr T (C)"] = summary_rates_df.get('Rxr_T_C', summary_rates_df.get('Reactor_T_C', ''))
    render_summary["FA (uL/m)"] = summary_rates_df.get('FA_flow_ul.min', summary_rates_df.get('FA_flow_ul_min', ''))
    render_summary["WE (V)"] = summary_rates_df.get('WE_Voltage', summary_rates_df.get('Voltage (V)', ''))
    render_summary["Avg Win"] = summary_rates_df.get('Avg_Window', '')
    render_summary["Run Time"] = summary_rates_df.get('Run_Time', '')
    render_summary["CO2 (nmol)"] = summary_rates_df.get('CO2 (nmol/min)', summary_rates_df.get('CO2_avg', ''))
    render_summary["H2 (nmol)"] = summary_rates_df.get('H2 (nmol/min)', summary_rates_df.get('H2_avg', ''))
    render_summary["CO (nmol)"] = summary_rates_df.get('CO (nmol/min)', summary_rates_df.get('CO_avg', ''))

    report.add_table(render_summary, col_widths=col_widths, headers=headers)

# ---------------------------------------------------------
# PAGE 4: IMPEDANCE (EIS) DATA (IN-SITU & PRE-REACTION)
# ---------------------------------------------------------
if not rxn_eis_summary.empty and rxn_eis_plots:
    for run_id, (circuit, fig, params) in rxn_eis_plots.items():
        report.add_page()
        step_name_s = rxn_eis_summary.loc[rxn_eis_summary["EIS Run"] == run_id, "Step Name"].values
        step_title = step_name_s[0] if len(step_name_s) > 0 else f"Run {run_id}"

        report.add_title_block(f"Reaction Impedance (EIS) Data - {step_title}")
        report.add_plot_figure(fig, filename=f"Reaction_EIS_Run_{run_id}.png", width=180, ln_after=78)

        run_row = rxn_eis_summary[rxn_eis_summary["EIS Run"] == run_id].iloc[0]
        metrics = {
            "Series Resistance (wire, contact)": f"{run_row['r0_value']:.2f} Ohms",
            "Capacitor Leakage Resistance (R1)": f"{run_row['r1_value']:.2f} Ohms",
            "Device Capacitance (CPE1)": f"{run_row['cpe1_0']:.4f} uF",
            "Operating Condition": f"DC Bias: {run_row['DC Working Electrode (V)']} V | Time: {run_row['Time (hr)']} hr | Preceding: {run_row['Preceding Condition']}"
        }
        report.add_metrics_card("Equivalent Circuit Model Parameters", metrics, model_name="R0-p(R1, CPE1)")

if pre_rxn_status:
    eis_keys = [k for k in pre_reaction_df.keys() if "EIS" in k]
    if eis_keys:
        for key in eis_keys:
            circuit, (fig, axes), fitted_params, EIS_summary = EIS_cap_pre(pre_reaction_df[key], key)
            report.add_page()
            report.add_title_block(f"Pre-Reaction Impedance (EIS) Data - {key}")
            report.add_plot_figure(fig, filename=f"EIS_{key}.png", width=180, ln_after=78)

            metrics = {
                "Series Resistance (wire, contact)": f"{EIS_summary['r0_value'].iloc[0]:.2f} Ohms",
                "Capacitor Leakage Resistance": f"{EIS_summary['r1_value'].iloc[0]:.2f} Ohms",
                "Device Capacitance": f"{EIS_summary['cpe1_0'].iloc[0]:.2f} uF",
                "DC Working Electrode (V)": f"{EIS_summary['DC Working Electrode (V)'].iloc[0]}"
            }
            report.add_metrics_card("Equivalent Circuit Model Parameters", metrics, model_name="R0-p(R1, CPE1)")

# ---------------------------------------------------------
# PAGE 5: REACTION CYCLIC VOLTAMMETRY (CV) DATA & CAPACITANCE
# ---------------------------------------------------------
if cv_status and cv_seg and isinstance(capacitance, pd.DataFrame) and not capacitance.empty:
    # 1. Fast CV Sweeps
    figs_fast = cv_plot(cv_seg, rxn_schedule=RXN_SCHEDULE, max_per_page=6, rate_filter='fast')
    for i, fig in enumerate(figs_fast, start=1):
        report.add_page()
        report.add_plot_figure(fig, filename=f"CV_fast_data_page_{i}.png", title=f"Reaction CV Data - Fast Sweeps (Page {i})", width=160, ln_after=0)

    # 2. Slow CV Sweeps
    has_slow = any(df['sweep_rate'].iloc[0] < 0.1 for df in cv_seg.values() if 'sweep_rate' in df.columns)
    if has_slow:
        figs_slow = cv_plot(cv_seg, rxn_schedule=RXN_SCHEDULE, max_per_page=6, rate_filter='slow')
        for i, fig in enumerate(figs_slow, start=1):
            report.add_page()
            report.add_plot_figure(fig, filename=f"CV_slow_data_page_{i}.png", title=f"Reaction CV Data - Slow Sweep 0.02 V/s (Page {i})", width=160, ln_after=0)

    # 3. Capacitance summary table & trend
    report.add_page()
    render_df = pd.DataFrame()
    render_df["CV Run"] = capacitance["CV Run"].astype(str)
    render_df["Sweep Rate (V/s)"] = capacitance["Sweep Rate (V/s)"].apply(lambda r: f"{r:.2f}" if r < 0.1 else f"{r:.1f}")
    render_df["Capacitance (uF)"] = capacitance["Capacitance (uF)"].map("{:.4f}".format)
    render_df["Time (hr)"] = capacitance["Time (hr)"].map("{:.1f}".format)
    render_df["Preceding Condition"] = capacitance["Preceding Condition"].astype(str)

    report.add_title_block("Transient Capacitance Summary Table")
    report.add_table(render_df, col_widths=[25, 45, 40, 45, 35])

    fig_cap, _ = cap_plot(cv_seg, x_interval=12, rxn_schedule=RXN_SCHEDULE, shade_colors=SHADE_COLORS, target_sweep_rate=1.0)
    if report.pdf.get_y() > 180:
        report.add_page()
    report.add_plot_figure(fig_cap, filename="capacitance_data_1_0Vs.png", title="Transient Capacitance Trend (1.0 V/s)", width=180, ln_after=0)

    if has_slow:
        fig_slow_cap, _ = cap_plot(cv_seg, x_interval=12, rxn_schedule=RXN_SCHEDULE, shade_colors=SHADE_COLORS, target_sweep_rate=0.02)
        report.add_page()
        report.add_plot_figure(fig_slow_cap, filename="capacitance_data_0_02Vs.png", title="Transient Capacitance Trend (0.02 V/s)", width=180, ln_after=0)

    df_cv.to_csv(data_dir / 'rxn_cv_data.csv', index=False)
    render_df.to_csv(data_dir / 'rxn_capacitcance_data.csv', index=False)

# ---------------------------------------------------------
# PAGE 6: PRE-REACTION CV DIAGNOSTICS GRID
# ---------------------------------------------------------
if pre_rxn_status:
    cv_keys = [key for key in pre_reaction_df.keys() if "CV" in key]
    if cv_keys:
        cols = 2
        max_per_page = 4
        for page_idx, chunk_start in enumerate(range(0, len(cv_keys), max_per_page), start=1):
            chunk_keys = cv_keys[chunk_start:chunk_start + max_per_page]
            n_plots = len(chunk_keys)
            rows = int(np.ceil(n_plots / cols))
            
            fig, axes = plt.subplots(rows, cols, figsize=(8.5, 4.0 * rows), squeeze=False)
            axes_flat = axes.flatten()
            for i, key in enumerate(chunk_keys):
                panel_letter = chr(65 + chunk_start + i)
                plot_10CV(pre_reaction_df[key], title_name=key, ax=axes_flat[i], panel_letter=panel_letter)
                
            for j in range(n_plots, len(axes_flat)):
                axes_flat[j].axis('off')
                
            plt.subplots_adjust(hspace=0.42, wspace=0.32, top=0.92, bottom=0.08, left=0.10, right=0.95)
            report.add_page()
            page_title = "Pre-Reaction Electrochemical Diagnostics" if len(cv_keys) <= max_per_page else f"Pre-Reaction Electrochemical Diagnostics (Page {page_idx})"
            report.add_plot_figure(fig, filename=f"CV_Diagnostics_Grid_page_{page_idx}.png", title=page_title, width=155, ln_after=0)

# ---------------------------------------------------------
# PAGE 7: METADATA & SPECIFICATIONS
# ---------------------------------------------------------
report.add_page()
report.add_title_block("Trial Parameters Metadata")
report.add_key_value_pairs(TRIAL_INFO, font_size=9, label_bold=True)
report.pdf.ln(3)

report.add_title_block("Chromatography Method (GC) Specifications")
report.add_key_value_pairs(GC_METHOD_INFO, font_size=9, label_bold=True)

if RXN_SCHEDULE:
    report.add_page()
    report.add_title_block("Reaction Schedule Specification")
    
    has_co = any(step.get("co_flow", 0.0) > 0 for step in RXN_SCHEDULE)
    sched_df = pd.DataFrame(RXN_SCHEDULE)
    if has_co:
        col_w = [10, 14, 18, 18, 18, 18, 20, 20, 20]
        headers = ["Run", "Dur (h)", "T (C)", "N2 (sccm)", "H2 (sccm)", "CO (sccm)", "FA (uL/m)", "Tot (sccm)", "WE (V)"]
        render_sched = pd.DataFrame()
        render_sched["Run"] = sched_df.get("run", "")
        render_sched["Dur (h)"] = sched_df.get("dur_hrs", 0).map("{:.1f}".format)
        render_sched["T (C)"] = sched_df.get("rxr_t", 0).map("{:.1f}".format)
        render_sched["N2 (sccm)"] = sched_df.get("n2_flow", 0).map("{:.1f}".format)
        render_sched["H2 (sccm)"] = sched_df.get("h2_flow", 0).map("{:.1f}".format)
        render_sched["CO (sccm)"] = sched_df.get("co_flow", 0).map("{:.1f}".format)
        render_sched["FA (uL/m)"] = sched_df.get("fa_flow", 0).map("{:.2f}".format)
        render_sched["Tot (sccm)"] = sched_df.get("tot_flow", 0).map("{:.2f}".format)
        render_sched["WE (V)"] = sched_df.get("we_volt", "")
    else:
        col_w = [12, 16, 20, 22, 22, 22, 22, 24]
        headers = ["Run", "Dur (h)", "T (C)", "N2 (sccm)", "FA (uL/m)", "Tot (sccm)", "FA (%)", "WE (V)"]
        render_sched = pd.DataFrame()
        render_sched["Run"] = sched_df.get("run", "")
        render_sched["Dur (h)"] = sched_df.get("dur_hrs", 0).map("{:.1f}".format)
        render_sched["T (C)"] = sched_df.get("rxr_t", 0).map("{:.1f}".format)
        render_sched["N2 (sccm)"] = sched_df.get("n2_flow", 0).map("{:.1f}".format)
        render_sched["FA (uL/m)"] = sched_df.get("fa_flow", 0).map("{:.2f}".format)
        render_sched["Tot (sccm)"] = sched_df.get("tot_flow", 0).map("{:.2f}".format)
        render_sched["FA (%)"] = sched_df.get("fa_pct", 0).map("{:.2f}".format)
        render_sched["WE (V)"] = sched_df.get("we_volt", "")

    report.add_table(render_sched, col_widths=col_w, headers=headers)

# ---------------------------------------------------------
# PAGE 8: EQUIPMENT & REAL-TIME PROFILES
# ---------------------------------------------------------
if temp_df is not None and not temp_df.empty:
    report.add_page()
    fig_temp, _ = plot_temp_profile(temp_df=temp_df)
    report.add_plot_figure(fig_temp, filename="temp_data.png", title="Temperature Profile", width=180, ln_after=0)

if mfc_df is not None and not mfc_df.empty:
    report.add_page()
    fig_mfc, _ = plot_mfc_profile(mfc_df=mfc_df)
    report.add_plot_figure(fig_mfc, filename="mfc_data.png", title="MFC Gas Flow Profile", width=180, ln_after=0)

if pump_df is not None and not pump_df.empty:
    report.add_page()
    fig_pump, _ = plot_pump_profile(pump_df=pump_df)
    report.add_plot_figure(fig_pump, filename="pump_data.png", title="Syringe Pump (FA Flow) Profile", width=180, ln_after=0)

# Export PDF
output_pdf_path = report.save("reaction_data.pdf")
