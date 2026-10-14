"""
PDF Report Generator for Residence Time Distribution (RTD) Analysis.
Author: Ben Auer
Some code adapted from Justin Hopkins.
"""
import sys
from pathlib import Path
from datetime import date
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fpdf import FPDF, XPos, YPos, enums

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

def safe_str(text):
    if text is None:
        return ""
    replacements = {
        "•": "-",
        "°": " deg ",
        "±": "+/-",
        "—": "-",
        "–": "-",
        "µ": "u",
        "τ": "tau",
        "θ": "theta",
        "σ": "sigma"
    }
    s = str(text)
    for key, val in replacements.items():
        s = s.replace(key, val)
    return s.encode("latin-1", "ignore").decode("latin-1")

# ---------------------------------------------------------
# DATA ACQUISITION & IMPORTING
# ---------------------------------------------------------
dfs_dictionary, pre_reaction_df, selected_dir = select_folder_and_load_data(selected_dir=None)
if dfs_dictionary is None or selected_dir is None:
    sys.exit("Data loading failed or was cancelled.")

GC_df   = dfs_dictionary.get("FAD_df")
temp_df = dfs_dictionary.get("temp_df")
mfc_df  = dfs_dictionary.get("mfc_df")

# Load configuration metadata from YAML & CSV sequence
background_config.handle_yaml(selected_dir)
TRIAL_INFO     = background_config.TRIAL_INFO
GC_METHOD_INFO = background_config.GC_METHOD_INFO
notes          = background_config.notes
analysis       = background_config.analysis
RXN_SCHEDULE   = getattr(background_config, 'RXN_SCHEDULE', getattr(background_config, 'TEMP_SCHEDULE', []))

# Configure output directories
plots_dir = selected_dir / "plots"
data_dir = selected_dir / "data"
plots_dir.mkdir(parents=True, exist_ok=True)
data_dir.mkdir(parents=True, exist_ok=True)

# Global styling
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
# Default to CO or first available active species
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

# ---------------------------------------------------------
# RESIDENCE TIME DISTRIBUTION (RTD) CALCULATIONS
# ---------------------------------------------------------
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

# Generate plots
generated_plots = plot_rtd_analysis(rtd_results, plots_dir)

# Compute and save summary tables
df_summary_metrics, df_curve_data = compute_rtd_summary_dataframe(rtd_results)
df_summary_metrics.to_csv(data_dir / "rtd_summary_metrics.csv", index=False)
df_curve_data.to_csv(data_dir / "rtd_curve_data.csv", index=False)

tod = date.today().strftime("%m. %d. %Y.")

# ---------------------------------------------------------
# PDF REPORT COMPILATION
# ---------------------------------------------------------
class ReportPDF(FPDF):
    def header(self):
        self.set_font('Times', 'B', 12)
        self.cell(0, 8, safe_str('Residence Time Distribution (RTD) Report'), align='C', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(3)
        
    def footer(self):
        self.set_y(-15)
        self.set_font('Times', 'I', 10)
        self.cell(0, 10, safe_str(f'Page {self.page_no()} | Auer | {tod}'), align='R', border=0)

pdf = ReportPDF()
pdf.set_auto_page_break(auto=True, margin=15)

# =========================================================
# PAGE 1: EXECUTIVE SUMMARY & STEP PROFILE
# =========================================================
pdf.add_page()
pdf.set_font('Times', 'B', 11)

pdf.cell(30, 6, "Date:", new_x=XPos.RIGHT, new_y=YPos.LAST)
pdf.set_font('Times', '', 11)
pdf.cell(0, 6, safe_str(TRIAL_INFO.get('Date', '')), new_x=XPos.LMARGIN, new_y=YPos.NEXT)

pdf.set_font('Times', 'B', 11)
pdf.cell(30, 6, "Trial:", new_x=XPos.RIGHT, new_y=YPos.LAST)
pdf.set_font('Times', '', 11)
pdf.cell(0, 6, safe_str(TRIAL_INFO.get('Trial', 'RTD Step Test')), new_x=XPos.LMARGIN, new_y=YPos.NEXT)

pdf.set_font('Times', 'B', 11)
pdf.cell(30, 6, "Tracer Gas:", new_x=XPos.RIGHT, new_y=YPos.LAST)
pdf.set_font('Times', '', 11)
pdf.cell(0, 6, safe_str(f"{tracer_col} (Step at {t_step_hr:.2f} hr)"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
pdf.ln(3)

# Embed Tracer Step Profile Plot
if "step_profile" in generated_plots:
    pdf.image(generated_plots["step_profile"], x=enums.Align.C, w=170)
    pdf.ln(4)

# Tabulated Key RTD Metrics Table
pdf.set_font('Times', 'B', 11)
pdf.cell(0, 6, safe_str("Key Residence Time Distribution Parameters:"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
pdf.ln(1)

col_w = [84, 52, 44]
pdf.set_font('Times', 'B', 9)
pdf.cell(col_w[0], 5.5, safe_str("Parameter"), border=1, align='L')
pdf.cell(col_w[1], 5.5, safe_str("Value (+/- Error)"), border=1, align='C')
pdf.cell(col_w[2], 5.5, safe_str("Units"), border=1, align='C')
pdf.ln()

pdf.set_font('Times', '', 8.5)
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
    # Flow & Volume Parameters
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

for p_name, p_val, p_unit in key_rows:
    pdf.cell(col_w[0], 4.6, safe_str(f"  {p_name}"), border=1, align='L')
    pdf.cell(col_w[1], 4.6, safe_str(p_val), border=1, align='C')
    pdf.cell(col_w[2], 4.6, safe_str(p_unit), border=1, align='C')
    pdf.ln()

pdf.ln(4)

# =========================================================
# PAGE 2: F-CURVE & E-CURVE DISTRIBUTION ANALYSIS
# =========================================================
pdf.add_page()
pdf.set_font('Times', 'B', 13)
pdf.cell(0, 8, safe_str("RTD Step Response F(t) & Differential Density E(t)"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
pdf.ln(2)

if "f_curve" in generated_plots:
    pdf.image(generated_plots["f_curve"], x=enums.Align.C, w=155)
    pdf.ln(4)

if "e_curve" in generated_plots:
    pdf.image(generated_plots["e_curve"], x=enums.Align.C, w=155)
    pdf.ln(4)

# =========================================================
# PAGE 3: DIMENSIONLESS RTD & TABULATED CURVE DATA
# =========================================================
pdf.add_page()
pdf.set_font('Times', 'B', 13)
pdf.cell(0, 8, safe_str("Dimensionless RTD & Experimental Curve Data"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
pdf.ln(2)

if "dimensionless" in generated_plots:
    pdf.image(generated_plots["dimensionless"], x=enums.Align.C, w=150)
    pdf.ln(4)

# Tabulated F(t) and E(t) points
pdf.set_font('Times', 'B', 10)
pdf.cell(0, 6, safe_str("Experimental Step Response Table (Sampled Data Points):"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
pdf.ln(1)

table_widths = [26, 26, 32, 32, 32, 32]
table_headers = ["Time (min)", "Time (hr)", "Tracer Signal", "F(t) Exp", "E(t) (1/min)", "F(t) Model"]

pdf.set_font('Times', 'B', 8.5)
for w, h in zip(table_widths, table_headers):
    pdf.cell(w, 5, safe_str(h), border=1, align='C')
pdf.ln()

pdf.set_font('Times', '', 8.5)
# Show up to first 25 points or sampled evenly
max_pts = min(25, len(df_curve_data))
for _, row in df_curve_data.head(max_pts).iterrows():
    pdf.cell(table_widths[0], 4.5, safe_str(f"{row['Time_Relative_min']:.1f}"), border=1, align='C')
    pdf.cell(table_widths[1], 4.5, safe_str(f"{row['Time_Relative_hr']:.3f}"), border=1, align='C')
    pdf.cell(table_widths[2], 4.5, safe_str(f"{row['Tracer_Signal']:.2f}"), border=1, align='C')
    pdf.cell(table_widths[3], 4.5, safe_str(f"{row['F_experimental']:.4f}"), border=1, align='C')
    pdf.cell(table_widths[4], 4.5, safe_str(f"{row['E_experimental_min-1']:.5f}"), border=1, align='C')
    pdf.cell(table_widths[5], 4.5, safe_str(f"{row['F_model_TIS']:.4f}"), border=1, align='C')
    pdf.ln()

pdf.ln(4)

# =========================================================
# PAGE 4: TEMPERATURE & MFC PROFILES
# =========================================================
if (temp_df is not None and not temp_df.empty) or (mfc_df is not None and not mfc_df.empty):
    pdf.add_page()
    if temp_df is not None and not temp_df.empty:
        fig_t, _ = plot_temp_profile(temp_df=temp_df)
        p_temp = plots_dir / "temp_data.png"
        fig_t.savefig(p_temp, dpi=300, bbox_inches="tight")
        plt.close(fig_t)
        pdf.set_font("Times", "B", 12)
        pdf.cell(0, 10, safe_str("Reactor Temperature Profile"), new_x=XPos.RIGHT, new_y=YPos.NEXT)
        pdf.image(p_temp, enums.Align.C, y=pdf.get_y(), w=180)
        pdf.ln(105)

    if mfc_df is not None and not mfc_df.empty:
        fig_m, _ = plot_mfc_profile(mfc_df=mfc_df)
        p_mfc = plots_dir / "mfc_data.png"
        fig_m.savefig(p_mfc, dpi=300, bbox_inches="tight")
        plt.close(fig_m)
        pdf.set_font("Times", "B", 12)
        pdf.cell(0, 10, safe_str("MFC Gas Flow Profile"), new_x=XPos.RIGHT, new_y=YPos.NEXT)
        pdf.image(p_mfc, enums.Align.C, y=pdf.get_y(), w=180)
        pdf.ln(105)

# =========================================================
# PAGE 5: METADATA
# =========================================================
pdf.add_page()
pdf.set_font('Times', 'B', 14)
pdf.cell(0, 7, safe_str("Trial Parameters Metadata"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
pdf.set_font('Times', '', 9)
for key, value in TRIAL_INFO.items():
    pdf.cell(0, 5, safe_str(f"{key}: {value}"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
pdf.ln(3)

if notes:
    pdf.set_font('Times', 'B', 12)
    pdf.cell(0, 6, safe_str("Notes:"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font('Times', '', 10)
    pdf.multi_cell(0, 5, safe_str(notes))
    pdf.ln(3)

if analysis:
    pdf.set_font('Times', 'B', 12)
    pdf.cell(0, 6, safe_str("Analysis:"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font('Times', '', 10)
    pdf.multi_cell(0, 5, safe_str(analysis))
    pdf.ln(3)

pdf.set_font('Times', 'B', 14)
pdf.cell(0, 7, safe_str("Chromatography Method (GC) Specifications"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
pdf.set_font('Times', '', 9)
for key, value in GC_METHOD_INFO.items():
    pdf.cell(0, 5, safe_str(f"{key}: {value}"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)

# Save Final PDF
output_path = selected_dir / 'rtd_data.pdf'
pdf.output(str(output_path))
print(f"\n[Success] RTD Report successfully generated at: {output_path}")
