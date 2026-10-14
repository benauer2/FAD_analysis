"""
PDF Report Generator for Background Testing (OOP Refactored Test Version)
Author: Ben Auer
Some code adapted from Justin Hopkins.
"""
import sys
from pathlib import Path
from datetime import date
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import background_config
from worker_files.file_utils import select_folder_and_load_data
from worker_files.plot_worker import plot_background_rates, plot_temp_profile, plot_mfc_profile, plot_arrhenius
from worker_files.GC_worker import calc_rxn_rates, compute_schedule_summary, perform_arrhenius_analysis
from worker_files.pdf_classes import BackgroundReport, safe_str

dfs_dictionary, pre_reaction_df, selected_dir = select_folder_and_load_data(selected_dir=None)
if dfs_dictionary is None or selected_dir is None:
    sys.exit("Data loading failed or was cancelled.")

GC_df   = dfs_dictionary.get("FAD_df")
temp_df = dfs_dictionary.get("temp_df")
mfc_df  = dfs_dictionary.get("mfc_df")

background_config.handle_yaml(selected_dir)
TRIAL_INFO     = background_config.TRIAL_INFO
GC_METHOD_INFO = background_config.GC_METHOD_INFO
notes          = background_config.notes
analysis       = background_config.analysis
RXN_SCHEDULE   = getattr(background_config, 'RXN_SCHEDULE', getattr(background_config, 'TEMP_SCHEDULE', []))

plots_dir = selected_dir / "plots"
data_dir = selected_dir / "data"
plots_dir.mkdir(parents=True, exist_ok=True)
data_dir.mkdir(parents=True, exist_ok=True)

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Calibri']

# ---------------------------------------------------------
# SYSTEM PARAMETERS & RATE CALCULATIONS
# ---------------------------------------------------------
p_reac_f = float(TRIAL_INFO.get("P reactor (psig)", 0.0))
v_box_t  = float(GC_METHOD_INFO.get('Valve box T (deg C)', 100.0))

summary_df = pd.DataFrame()
arrhenius_res = {}
rates_df = pd.DataFrame()

if GC_df is not None and not GC_df.empty:
    rates_df = calc_rxn_rates(
        GC_df=GC_df,
        RXN_SCHEDULE=RXN_SCHEDULE,
        v_box_t=v_box_t,
        p_reac_f=p_reac_f,
        GC_METHOD_INFO=GC_METHOD_INFO
    )
    rates_df.to_csv(data_dir / "rates_vs_time.csv", index=False)

    summary_df = compute_schedule_summary(rates_df, RXN_SCHEDULE)
    summary_df.to_csv(data_dir / "avg_rates.csv", index=False)

if summary_df.empty:
    raise ValueError("summary_df is empty. Check that GC_df was loaded correctly and RXN_SCHEDULE is defined.")

# ---------------------------------------------------------
# REPORT COMPILATION INFRASTRUCTURE (OOP)
# ---------------------------------------------------------
report = BackgroundReport(target_dir=selected_dir)

# --- PAGE 1: OVERVIEW & TABULATED SUMMARY ---
report.add_page()
kv_meta = {
    "Date": TRIAL_INFO.get('Date', ''),
    "Trial": TRIAL_INFO.get('Trial', '')
}
report.add_key_value_pairs(kv_meta, font_size=11)
report.pdf.ln(2)

# Render Overview Rates Plot
fig_rates, ax1, _ = plot_background_rates(rates_df)
report.add_plot_figure(fig_rates, filename="overview_rates.png", width=165, ln_after=4)

# SUMMARY DATA TABLE
report.add_title_block("Summary Table (Steady-State Averages per Condition Window)")

col_widths = [12, 22, 22, 24, 38, 38, 38]
headers = ["Run", "Temp (C)", "N2 (sccm)", "Window", "CO2 (nmol/min)", "H2 (nmol/min)", "CO (nmol/min)"]

render_summary = pd.DataFrame()
render_summary["Run"] = summary_df.get('Run', '')
render_summary["Temp (C)"] = summary_df.get('Rxr_T_C', summary_df.get('Temp_C', ''))
render_summary["N2 (sccm)"] = summary_df.get('N2_flow_sccm', summary_df.get('N2_sccm', ''))
render_summary["Window"] = summary_df.get('Avg_Window', summary_df.get('Window_hr', ''))
render_summary["CO2 (nmol/min)"] = summary_df.apply(lambda r: r.get('CO2 (nmol/min)', f"{r.get('CO2_avg', 0.0):.2f} +/- {r.get('CO2_std', 0.0):.2f}"), axis=1)
render_summary["H2 (nmol/min)"]  = summary_df.apply(lambda r: r.get('H2 (nmol/min)', f"{r.get('H2_avg', 0.0):.2f} +/- {r.get('H2_std', 0.0):.2f}"), axis=1)
render_summary["CO (nmol/min)"]  = summary_df.apply(lambda r: r.get('CO (nmol/min)', f"{r.get('CO_avg', 0.0):.2f} +/- {r.get('CO_std', 0.0):.2f}"), axis=1)

report.add_table(render_summary, col_widths=col_widths, headers=headers)

# --- PAGE 2: ARRHENIUS ANALYSIS (IF PRESENT) ---
if arrhenius_res:
    report.add_page()
    report.add_title_block("Arrhenius Kinetic Analysis")
    fig_arr, _ = plot_arrhenius(arrhenius_res, use_ci95=True)
    report.add_plot_figure(fig_arr, filename="arrhenius_plot.png", width=155, ln_after=6)

    arr_metrics = {}
    for species, res in arrhenius_res.items():
        sp_name = "CO2" if "CO2" in species else ("H2" if "H2" in species else ("CO" if "CO" in species else species))
        ea_val = res.get('Ea_kJ', np.nan)
        ea_ci95 = res.get('Ea_ci95_kJ', np.nan)
        a_val = res.get('A', np.nan)
        r2 = res.get('R2', np.nan)
        arr_metrics[f"{sp_name} Activation Energy (Ea)"] = f"{ea_val:.2f} +/- {ea_ci95:.2f} kJ/mol | ln(A) = {a_val:.2f} | R2 = {r2:.4f}"
    
    report.add_metrics_card("Calculated Kinetic Parameters", arr_metrics)

# --- PAGE 3: REAL-TIME PROFILES ---
if temp_df is not None and not temp_df.empty:
    report.add_page()
    fig_temp, _ = plot_temp_profile(temp_df)
    report.add_plot_figure(fig_temp, filename="temp_profile.png", title="Reactor Temperature Profile", width=165, ln_after=0)

if mfc_df is not None and not mfc_df.empty:
    report.add_page()
    fig_mfc, _ = plot_mfc_profile(mfc_df)
    report.add_plot_figure(fig_mfc, filename="mfc_profile.png", title="MFC Flow Rates Profile", width=165, ln_after=0)

# --- METADATA & SCHEDULE PAGE ---
report.add_page()
report.add_title_block("Trial Parameters Metadata")
report.add_key_value_pairs(TRIAL_INFO, font_size=9)
report.pdf.ln(3)

report.add_title_block("Chromatography Method (GC) Specifications")
report.add_key_value_pairs(GC_METHOD_INFO, font_size=9)
report.pdf.ln(3)

if notes:
    report.add_text_section("Reaction Notes", notes)
if analysis:
    report.add_text_section("Analysis", analysis)

output_pdf_path = report.save("background_report.pdf")
