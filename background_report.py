"""
PDF Report Generator for Background Testing.
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
from scipy.stats import t

import background_config
from worker_files.file_utils import select_folder_and_load_data
from worker_files.plot_worker import plot_background_rates, plot_temp_profile, plot_mfc_profile, plot_arrhenius
from worker_files.GC_worker import calc_rxn_rates, compute_schedule_summary, perform_arrhenius_analysis

def safe_str(text):
    if text is None:
        return ""
    replacements = {
        "•": "-",
        "°": " deg ",
        "±": "+/-",
        "—": "-",
        "–": "-",
        "µ": "u"
    }
    s = str(text)
    for key, val in replacements.items():
        s = s.replace(key, val)
    # Encode to latin-1, ignoring unencodable characters as a fail-safe
    return s.encode("latin-1", "ignore").decode("latin-1")

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

# CONFIGURE OUTPUT FOLDER LOCATIONS
plots_dir = selected_dir / "plots"
data_dir = selected_dir / "data"
plots_dir.mkdir(parents=True, exist_ok=True)
data_dir.mkdir(parents=True, exist_ok=True)

# GLOBAL CONFIGURATION FOR TEXT
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Calibri']

'''
########################################
SYSTEM PARAMETERS & RATE CALCULATIONS
########################################
'''
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

    #arrhenius_res = perform_arrhenius_analysis(summary_df)

if summary_df.empty:
    raise ValueError("summary_df is empty. Check that GC_df was loaded correctly and RXN_SCHEDULE is defined.")

tod = date.today().strftime("%m. %d. %Y.")

'''
##############
BUILD OUT PDF
##############
'''
class ReportPDF(FPDF):
    def header(self):
        self.set_font('Times', 'B', 12)
        self.cell(0, 8, safe_str('Formic Acid Decomposition Background Report'), align='C', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(4)
        
    def footer(self):
        self.set_y(-15)
        self.set_font('Times', 'I', 10)
        self.cell(0, 10, safe_str(f'Page {self.page_no()} | Auer | {tod}'), align='R', border=0)

pdf = ReportPDF()
pdf.set_auto_page_break(auto=True, margin=15)

# --- PAGE 1: OVERVIEW & TABULATED SUMMARY ---
pdf.add_page()
pdf.set_font('Times', 'B', 11)

pdf.cell(30, 6, "Date:", new_x=XPos.RIGHT, new_y=YPos.LAST)
pdf.set_font('Times', '', 11)
pdf.cell(0, 6, safe_str(TRIAL_INFO.get('Date', '')), new_x=XPos.LMARGIN, new_y=YPos.NEXT)

pdf.set_font('Times', 'B', 11)
pdf.cell(30, 6, "Trial:", new_x=XPos.RIGHT, new_y=YPos.LAST)
pdf.set_font('Times', '', 11)
pdf.cell(0, 6, safe_str(TRIAL_INFO.get('Trial', '')), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
pdf.ln(4)

# Render Overview Rates Plot
fig, ax1, _ = plot_background_rates(rates_df)
plot_path = plots_dir / "overview_rates.png"
fig.savefig(plot_path, dpi=300, bbox_inches="tight")
plt.close(fig)

pdf.image(plot_path, x=enums.Align.C, w=165)
pdf.ln(4)

# SUMMARY DATA TABLE
pdf.set_font('Times', 'B', 11)
pdf.cell(0, 6, safe_str("Summary Table (Steady-State Averages per Condition Window):"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
pdf.ln(2)

# Table Header
col_widths = [12, 22, 22, 24, 38, 38, 38]
headers = ["Run", "Temp (C)", "N2 (sccm)", "Window", "CO2 (nmol/min)", "H2 (nmol/min)", "CO (nmol/min)"]

pdf.set_font('Times', 'B', 9)
for w, h in zip(col_widths, headers):
    pdf.cell(w, 6, safe_str(h), border=1, align='C')
pdf.ln()

# Table Rows
pdf.set_font('Times', '', 9)
for _, row in summary_df.iterrows():
    run_val = str(row.get('Run', ''))
    temp_val = str(row.get('Rxr_T_C', row.get('Temp_C', '')))
    n2_val = str(row.get('N2_flow_sccm', row.get('N2_sccm', '')))
    win_val = str(row.get('Avg_Window', row.get('Window_hr', '')))
    co2_str = row.get('CO2 (nmol/min)', f"{row.get('CO2_avg', 0.0):.2f} +/- {row.get('CO2_std', 0.0):.2f}")
    h2_str  = row.get('H2 (nmol/min)',  f"{row.get('H2_avg', 0.0):.2f} +/- {row.get('H2_std', 0.0):.2f}")
    co_str  = row.get('CO (nmol/min)',  f"{row.get('CO_avg', 0.0):.2f} +/- {row.get('CO_std', 0.0):.2f}")

    pdf.cell(col_widths[0], 6, safe_str(run_val), border=1, align='C')
    pdf.cell(col_widths[1], 6, safe_str(temp_val), border=1, align='C')
    pdf.cell(col_widths[2], 6, safe_str(n2_val), border=1, align='C')
    pdf.cell(col_widths[3], 6, safe_str(win_val), border=1, align='C')
    pdf.cell(col_widths[4], 6, safe_str(co2_str), border=1, align='C')
    pdf.cell(col_widths[5], 6, safe_str(h2_str), border=1, align='C')
    pdf.cell(col_widths[6], 6, safe_str(co_str), border=1, align='C')
    pdf.ln()

pdf.ln(6)

# --- PAGE 2: ARRHENIUS ANALYSIS ---
if arrhenius_res:
    pdf.add_page()
    pdf.set_font('Times', 'B', 13)
    pdf.cell(0, 8, safe_str("Arrhenius Kinetic Analysis"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(2)

    USE_95_CI = True

    fig_arr, _ = plot_arrhenius(arrhenius_res, use_ci95=USE_95_CI)
    arr_path = plots_dir / "arrhenius_plot.png"
    fig_arr.savefig(arr_path, dpi=300, bbox_inches="tight")
    plt.close(fig_arr)

    pdf.image(arr_path, x=enums.Align.C, w=155)
    pdf.ln(6)

    pdf.set_font('Times', 'B', 11)
    pdf.cell(0, 6, safe_str("Calculated Kinetic Parameters:"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font('Times', '', 10)
    
    for species, res in arrhenius_res.items():
        sp_name = "CO2" if "CO2" in species else ("H2" if "H2" in species else ("CO" if "CO" in species else species))
        
        ea_val = res.get('Ea_kJ', np.nan)
        ea_err = res.get('Ea_err_kJ', np.nan)      # 1-sigma SE
        ea_ci95 = res.get('Ea_ci95_kJ', np.nan)   # 95% CI
        
        a_val = res.get('A', np.nan)
        a_err = res.get('A_err', np.nan)          # 1-sigma SE
        a_ci95 = res.get('A_ci95', np.nan)        # 95% CI
        
        r2_val = res.get('R2', np.nan)
        
        ea_str = f"{ea_val:.2f} +/- {ea_err:.2f} (SE) [95% CI: +/- {ea_ci95:.2f}]" if np.isfinite(ea_err) else f"{ea_val:.2f}"
        a_str = f"{a_val:.2e} +/- {a_err:.2e} (SE)" if np.isfinite(a_err) else f"{a_val:.2e}"
        
        pdf.cell(
            0, 5, 
            safe_str(f"  - {sp_name}: Ea = {ea_str} kJ/mol  |  A = {a_str}  |  R2 = {r2_val:.4f}"), 
            new_x=XPos.LMARGIN, 
            new_y=YPos.NEXT
        )

# --- DYNAMIC CONDITION STEP PAGES ---
for _, row in summary_df.iterrows():
    run_idx = row.get("Run", 1)
    temp_val = row.get("Rxr_T_C", row.get("Temp_C", ""))
    n2_val = row.get("N2_flow_sccm", row.get("N2_sccm", ""))
    avg_win = row.get("Avg_Window", row.get("Window_hr", ""))
    
    step_label = f"Step {run_idx}: {temp_val} deg C - {n2_val} sccm N2"
    sub_df = rates_df[rates_df["Run"] == run_idx] if "Run" in rates_df.columns else rates_df
    
    if sub_df.empty:
        continue

    pdf.add_page()
    pdf.set_font("Times", "B", 13)
    pdf.cell(0, 8, safe_str(f"Condition Analysis: {step_label}"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(2)

    max_rate = max(sub_df["CO2 (nmol/min)"].max(), sub_df["CO (nmol/min)"].max(), sub_df["H2 (nmol/min)"].max()) * 1.3 if not sub_df.empty else 10.0
    calc_y_int = max(1.0, np.ceil(max_rate / 4)) if max_rate > 0 else 5.0

    fig, ax1, _ = plot_background_rates(sub_df, y_interval=calc_y_int)
    ax1.set_title(safe_str(f"Gas Rates - {step_label}"), fontsize=12, fontweight="bold", pad=12)

    sanitized_name = f"step_{run_idx}_{temp_val}C_{n2_val}sccm".replace(" ", "_")
    stage_plot_path = plots_dir / f"rates_{sanitized_name}.png"
    fig.savefig(stage_plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    pdf.image(stage_plot_path, x=enums.Align.C, w=165)
    pdf.ln(6)

    # Statistics Summary Block
    co2_stat = row.get('CO2 (nmol/min)', f"{row.get('CO2_avg', 0.0):.2f} +/- {row.get('CO2_std', 0.0):.2f}")
    h2_stat  = row.get('H2 (nmol/min)',  f"{row.get('H2_avg', 0.0):.2f} +/- {row.get('H2_std', 0.0):.2f}")
    co_stat  = row.get('CO (nmol/min)',  f"{row.get('CO_avg', 0.0):.2f} +/- {row.get('CO_std', 0.0):.2f}")

    pdf.set_font('Times', 'B', 11)
    pdf.cell(0, 6, safe_str(f"Averaging Window Used: {avg_win} (Relative Step Time)"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font('Times', '', 11)
    pdf.cell(0, 5, safe_str(f"  - CO2 Rate: {co2_stat} nmol/min"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.cell(0, 5, safe_str(f"  - H2 Rate:  {h2_stat} nmol/min"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.cell(0, 5, safe_str(f"  - CO Rate:  {co_stat} nmol/min"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)

# --- TEMPERATURE & MFC PROFILES ---
if temp_df is not None and not temp_df.empty:
    pdf.add_page()
    fig, ax = plot_temp_profile(temp_df=temp_df)
    plot_file_path = plots_dir / "temp_data.png"
    fig.savefig(plot_file_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    pdf.set_font("Times", "B", 12)
    pdf.cell(0, 10, safe_str("Temperature Profile Plot"), new_x=XPos.RIGHT, new_y=YPos.NEXT)
    pdf.image(plot_file_path, enums.Align.C, y=pdf.get_y(), w=180)
    pdf.ln(105)

if mfc_df is not None and not mfc_df.empty:
    fig, ax = plot_mfc_profile(mfc_df=mfc_df)
    plot_file_path = plots_dir / "mfc_data.png"
    fig.savefig(plot_file_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    pdf.set_font("Times", "B", 12)
    pdf.cell(0, 10, safe_str("MFC Flow Profile Plot"), new_x=XPos.RIGHT, new_y=YPos.NEXT)
    pdf.image(plot_file_path, enums.Align.C, y=pdf.get_y(), w=180)
    pdf.ln(105)

# --- METADATA PAGE ---
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
output_path = selected_dir / 'background_data.pdf'
pdf.output(str(output_path))
print(f"Report successfully generated at: {output_path}")
