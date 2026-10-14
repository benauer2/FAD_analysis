"""
PDF Report Generator for Residence Time Distribution (RTD) Analysis (OOP Refactored Test Version)
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
from worker_files.plot_worker import plot_temp_profile, plot_mfc_profile
from worker_files.GC_worker import calc_rxn_rates
from worker_files.rtd_worker import (
    prompt_step_change_parameters,
    calculate_rtd_step_response,
    plot_rtd_analysis,
    compute_rtd_summary_dataframe
)
from worker_files.pdf_classes import RTDReport, safe_str

# ---------------------------------------------------------
# DATA ACQUISITION & IMPORTING
# ---------------------------------------------------------
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
# RATE / CONCENTRATION CALCULATIONS
# ---------------------------------------------------------
p_reac_f = float(TRIAL_INFO.get("P reactor (psig)", 0.0))
v_box_t  = float(GC_METHOD_INFO.get('Valve box T (deg C)', 100.0))

if GC_df is None or GC_df.empty:
    raise ValueError("No GC chromatogram data found in the selected folder.")

rates_df = calc_rxn_rates(
    GC_df=GC_df,
    RXN_SCHEDULE=RXN_SCHEDULE,
    v_box_t=v_box_t,
    p_reac_f=p_reac_f,
    GC_METHOD_INFO=GC_METHOD_INFO
)
rates_df.to_csv(data_dir / "rates_vs_time.csv", index=False)

# ---------------------------------------------------------
# INTERACTIVE RTD STEP CONFIGURATION
# ---------------------------------------------------------
rtd_cfg = prompt_step_change_parameters(
    RXN_SCHEDULE=RXN_SCHEDULE, 
    rates_df=rates_df, 
    default_species="CO"
)

t_step_hr = float(rtd_cfg.get("step_time_hr", 0.5))
tracer_col = str(rtd_cfg.get("tracer_col", "CO (nmol/min)"))
step_dir = str(rtd_cfg.get("step_direction", "auto"))
base_pts = int(rtd_cfg.get("baseline_points", 2))

print(f"\n[RTD Analysis] Running step change analysis on '{tracer_col}' at t = {t_step_hr:.3f} hr (direction: {step_dir})...")

rxr_t_val = float(TRIAL_INFO.get('T reactor (deg C)', 125.0)) if str(TRIAL_INFO.get('T reactor (deg C)', '')).replace('.', '').isdigit() else 125.0
v_geom_val = float(TRIAL_INFO.get('Reactor volume (cm^3)', TRIAL_INFO.get('Reactor volume (mL)', 0.0))) if TRIAL_INFO.get('Reactor volume (cm^3)') or TRIAL_INFO.get('Reactor volume (mL)') else None

rtd_results = calculate_rtd_step_response(
    df=rates_df,
    tracer_col=tracer_col,
    t_step_hr=t_step_hr,
    step_direction=step_dir,
    baseline_pts=base_pts,
    p_reac_psig=p_reac_f,
    rxr_t_c=rxr_t_val,
    v_geom_cm3=v_geom_val
)

generated_plots = plot_rtd_analysis(rtd_results, plots_dir)

df_summary_metrics, df_curve_data = compute_rtd_summary_dataframe(rtd_results)
df_summary_metrics.to_csv(data_dir / "rtd_summary_metrics.csv", index=False)
df_curve_data.to_csv(data_dir / "rtd_curve_data.csv", index=False)

# ---------------------------------------------------------
# PDF REPORT COMPILATION (OOP)
# ---------------------------------------------------------
report = RTDReport(target_dir=selected_dir)

# =========================================================
# PAGE 1: EXECUTIVE SUMMARY & STEP PROFILE
# =========================================================
report.add_page()
kv_meta = {
    "Date": TRIAL_INFO.get('Date', ''),
    "Trial": TRIAL_INFO.get('Trial', 'RTD Step Test'),
    "Tracer Gas": f"{tracer_col} (Step at {t_step_hr:.2f} hr)"
}
report.add_key_value_pairs(kv_meta, font_size=11)
report.pdf.ln(2)

if "step_profile" in generated_plots:
    report.add_plot_image(generated_plots["step_profile"], width=170, ln_after=4)

report.add_title_block("Key Residence Time Distribution Parameters:")

col_w = [84, 52, 44]
headers = ["Parameter", "Value (+/- Error)", "Units"]

key_rows = [
    ("Step Change Injection Time (t0)", f"{rtd_results['t_step_hr']:.3f}", "hr"),
    ("Mean Residence Time (t_mean)", f"{rtd_results['t_mean_min']:.2f} +/- {rtd_results['t_mean_err_min']:.2f}", "min"),
    ("Variance (sigma^2)", f"{rtd_results['sigma2_min2']:.2f} +/- {rtd_results['sigma2_err_min2']:.2f}", "min^2"),
    ("Dimensionless Variance (sigma_theta^2)", f"{rtd_results['sigma_theta2']:.4f} +/- {rtd_results['sigma_theta2_err']:.4f}", "-"),
    ("Equivalent Tanks-in-Series (N)", f"{rtd_results['N_tanks']:.2f} +/- {rtd_results['N_tanks_err']:.2f}", "tanks"),
    ("Fitted Tanks-in-Series (N_fit)", f"{rtd_results['N_tanks_fit']:.2f} +/- {rtd_results['N_tanks_fit_err']:.2f}", "tanks"),
    ("Fitted Space Time (tau_fit)", f"{rtd_results['tau_fit_min']:.1f} +/- {rtd_results['tau_fit_err_min']:.1f}", "min"),
    ("Tanks-in-Series Model Fit (R^2)", f"{rtd_results['r2_tis']:.4f}", "-"),
    ("Vessel Dispersion Number (D / uL)", f"{rtd_results['dispersion_number']:.4f} +/- {rtd_results['dispersion_err']:.4f}", "-"),
    ("Peclet Number (Pe = uL / D)", f"{rtd_results['peclet_number']:.2f} +/- {rtd_results['peclet_err']:.2f}", "-"),
    ("Total Gas Flow Rate (Standard)", f"{rtd_results['tot_flow_sccm']:.2f} +/- {rtd_results['tot_flow_err_sccm']:.2f}", "sccm"),
    ("Line Volumetric Flow Rate (20 deg C)", f"{rtd_results['q_ambient_ml_min']:.2f} +/- {rtd_results['q_ambient_err_ml_min']:.2f}", "mL/min"),
    ("Heated Reactor Flow Rate (at T_rxr)", f"{rtd_results['q_rxr_hot_ml_min']:.2f} +/- {rtd_results['q_rxr_hot_err_ml_min']:.2f}", "mL/min"),
    ("Line / System Volume (at 20 deg C)", f"{rtd_results['v_ambient_ml']:.3f} +/- {rtd_results['v_ambient_err_ml']:.3f} mL", "cm^3 (mL)"),
    ("Line / System Volume (in uL)", f"{rtd_results['v_ambient_ul']:.1f} +/- {rtd_results['v_ambient_err_ul']:.1f} uL", "uL"),
    ("Heated Reactor Zone Volume (at T_rxr)", f"{rtd_results['v_hot_rxr_ml']:.3f} +/- {rtd_results['v_hot_rxr_err_ml']:.3f} mL", "cm^3 (mL)"),
    ("Standard Fluid Volume (V_std)", f"{rtd_results['v_standard_scc']:.3f} +/- {rtd_results['v_standard_err_scc']:.3f}", "scc"),
    ("Model Fitted Ambient Volume (V_model)", f"{rtd_results['v_model_ml']:.3f} +/- {rtd_results['v_model_err_ml']:.3f}", "mL")
]

if pd.notna(rtd_results.get("dead_volume_pct")):
    key_rows.append(("Geometric Reactor Volume (V_geom)", f"{rtd_results['v_geom_cm3']:.3f}", "cm^3"))
    key_rows.append(("Dead Space Volume Fraction", f"{rtd_results['dead_volume_pct']:.1f} +/- {rtd_results['dead_volume_err_pct']:.1f}", "%"))
    key_rows.append(("Gas Hourly Space Velocity (GHSV)", f"{rtd_results['ghsv_hr']:.1f} +/- {rtd_results['ghsv_err_hr']:.1f}", "hr^-1"))

rtd_table_df = pd.DataFrame(key_rows, columns=headers)
report.add_table(rtd_table_df, col_widths=col_w, headers=headers, font_size=8.0, row_height=4.6)

# =========================================================
# PAGE 2: F-CURVE & E-CURVE DISTRIBUTION ANALYSIS
# =========================================================
report.add_page()
report.add_title_block("RTD Step Response F(t) & Differential Density E(t)")

if "f_curve" in generated_plots:
    report.add_plot_image(generated_plots["f_curve"], width=155, ln_after=4)

if "e_curve" in generated_plots:
    report.add_plot_image(generated_plots["e_curve"], width=155, ln_after=4)

# =========================================================
# PAGE 3: DIMENSIONLESS RTD & TABULATED CURVE DATA
# =========================================================
report.add_page()
report.add_title_block("Dimensionless RTD & Experimental Curve Data")

if "dimensionless" in generated_plots:
    report.add_plot_image(generated_plots["dimensionless"], width=155, ln_after=4)

report.add_title_block("Experimental RTD Step Data (Sample Points):")

n_total = len(df_curve_data)
if n_total > 20:
    idx_sample = np.linspace(0, n_total - 1, 16, dtype=int)
    table_curve = df_curve_data.iloc[idx_sample].copy()
else:
    table_curve = df_curve_data.copy()

col_w_curve = [24, 24, 28, 28, 28, 28, 28]
report.add_table(table_curve, col_widths=col_w_curve, font_size=7.5, row_height=4.2)

# =========================================================
# PAGE 4: REAL-TIME OPERATIONAL PROFILES
# =========================================================
if temp_df is not None and not temp_df.empty:
    report.add_page()
    fig_temp, _ = plot_temp_profile(temp_df)
    report.add_plot_figure(fig_temp, filename="temp_profile.png", title="Reactor Temperature Profile", width=165, ln_after=0)

if mfc_df is not None and not mfc_df.empty:
    report.add_page()
    fig_mfc, _ = plot_mfc_profile(mfc_df)
    report.add_plot_figure(fig_mfc, filename="mfc_profile.png", title="MFC Gas Flow Rates Profile", width=165, ln_after=0)

# =========================================================
# PAGE 5: METADATA & EXPERIMENT NOTES
# =========================================================
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

output_pdf_path = report.save("rtd_report.pdf")
