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
from worker_files.file_utils import(select_folder_and_load_data)
from worker_files.plot_worker import plot_background_rates, plot_temp_profile, plot_mfc_profile, plot_arrhenius
from worker_files.GC_worker import calc_rxn_rates_dynamic, compute_schedule_summary, perform_arrhenius_analysis

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

GC_df   = dfs_dictionary.get("FAD_df")
temp_df = dfs_dictionary.get("temp_df")
mfc_df  = dfs_dictionary.get("mfc_df")

background_config.handle_yaml(selected_dir)
TRIAL_INFO     = background_config.TRIAL_INFO
GC_METHOD_INFO = background_config.GC_METHOD_INFO
notes          = background_config.notes
analysis       = background_config.analysis
RXN_SCHEDULE = getattr(background_config, 'RXN_SCHEDULE', getattr(background_config, 'TEMP_SCHEDULE', []))

#CONFIGURES OUTPUT FOLDER LOCATIONS
plots_dir = selected_dir / "plots"
data_dir = selected_dir / "data"
plots_dir.mkdir(parents=True, exist_ok=True)
data_dir.mkdir(parents=True, exist_ok=True)

#GLOBAL CONFIGURATION FOR TEXT
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Calibri']

'''
########################################
METADATA VALUE GENERATION BASED ON YAML
########################################
'''

summary_df = pd.DataFrame()
arrhenius_res = {}
#EXTRACT VALUES TO CONVERT FA FLOW FROM uL/min TO sccm
if GC_df is not None:
    if "Time (hr)" not in GC_df.columns:
        GC_df['Time (min)'] = [i * 30 for i in range(len(GC_df))]
        GC_df['Time (hr)']  = GC_df['Time (min)'] / 60

        rates_df = calc_rxn_rates_dynamic(
            GC_df=GC_df,
            RXN_SCHEDULE=RXN_SCHEDULE,
            TRIAL_INFO=TRIAL_INFO,
            GC_METHOD_INFO=GC_METHOD_INFO
        )
        summary_df = compute_schedule_summary(rates_df, RXN_SCHEDULE)
        arrhenius_res = perform_arrhenius_analysis(summary_df)
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
        self.cell(0, 8, safe_str('Formic Acid Decomposition Trial Report'), align='C', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
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
col_widths = [22, 22, 24, 38, 38, 38]
headers = ["Temp (C)", "N2 (sccm)", "Window", "CO2 (nmol/min)", "H2 (nmol/min)", "CO (nmol/min)"]

pdf.set_font('Times', 'B', 9)
for w, h in zip(col_widths, headers):
    pdf.cell(w, 6, safe_str(h), border=1, align='C')
pdf.ln()

# Table Rows
pdf.set_font('Times', '', 9)
for _, row in summary_df.iterrows():
    pdf.cell(col_widths[0], 6, safe_str(f"{row['Temp_C']}"), border=1, align='C')
    pdf.cell(col_widths[1], 6, safe_str(f"{row['N2_sccm']}"), border=1, align='C')
    pdf.cell(col_widths[2], 6, safe_str(f"{row['Window_hr']}"), border=1, align='C')
    pdf.cell(col_widths[3], 6, safe_str(f"{row['CO2_avg']:.2f} +/- {row['CO2_std']:.2f}"), border=1, align='C')
    pdf.cell(col_widths[4], 6, safe_str(f"{row['H2_avg']:.2f} +/- {row['H2_std']:.2f}"), border=1, align='C')
    pdf.cell(col_widths[5], 6, safe_str(f"{row['CO_avg']:.2f} +/- {row['CO_std']:.2f}"), border=1, align='C')
    pdf.ln()

pdf.ln(6)

# --- PAGE 2: ARRHENIUS ANALYSIS ---
if arrhenius_res:
    pdf.add_page()
    pdf.set_font('Times', 'B', 13)
    pdf.cell(0, 8, safe_str("Arrhenius Kinetic Analysis"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(2)

    # Set use_ci95=True if you prefer 95% CIs on the plot legend, or False for SE
    USE_95_CI = True  # Toggle as desired

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
        sp_name = "CO2" if "CO2" in species else "H2"
        
        ea_val = res.get('Ea_kJ', np.nan)
        ea_err = res.get('Ea_err_kJ', np.nan)      # 1-sigma SE
        ea_ci95 = res.get('Ea_ci95_kJ', np.nan)   # 95% CI
        
        a_val = res.get('A', np.nan)
        a_err = res.get('A_err', np.nan)          # 1-sigma SE
        a_ci95 = res.get('A_ci95', np.nan)        # 95% CI
        
        r2_val = res.get('R2', np.nan)
        
        # Display both SE and 95% CI explicitly in the text summary
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
    step_label = f"{row['Temp_C']} deg C - {row['N2_sccm']} sccm N2"
    sub_df = rates_df[rates_df["Condition_Step"] == f"{row['Temp_C']}°C - {row['N2_sccm']} sccm N2"]
    
    if sub_df.empty:
        continue

    pdf.add_page()
    pdf.set_font("Times", "B", 13)
    pdf.cell(0, 8, safe_str(f"Condition Analysis: {step_label}"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(2)

    max_rate = max(sub_df["CO2 (nmol/min)"].max(), sub_df["CO (nmol/min)"].max()) * 1.3
    calc_y_int = max(1.0, np.ceil(max_rate / 4)) if max_rate > 0 else 5.0

    fig, ax1, _ = plot_background_rates(rates_df, y_interval=None)

    ax1.set_title(safe_str(f"Gas Rates - {step_label}"), fontsize=12, fontweight="bold", pad=12)

    sanitized_name = step_label.replace(" ", "_").replace("deg_C", "C")
    stage_plot_path = plots_dir / f"rates_{sanitized_name}.png"
    fig.savefig(stage_plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    pdf.image(stage_plot_path, x=enums.Align.C, w=165)
    pdf.ln(6)

    # Statistics Summary Block
    pdf.set_font('Times', 'B', 11)
    pdf.cell(0, 6, safe_str(f"Averaging Window Used: {row['Window_hr']} (Relative Step Time)"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font('Times', '', 11)
    pdf.cell(0, 5, safe_str(f"  - CO2 Rate: {row['CO2_avg']:.2f} +/- {row['CO2_std']:.2f} nmol/min"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.cell(0, 5, safe_str(f"  - H2 Rate:  {row['H2_avg']:.2f} +/- {row['H2_std']:.2f} nmol/min"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.cell(0, 5, safe_str(f"  - CO Rate:  {row['CO_avg']:.2f} +/- {row['CO_std']:.2f} nmol/min"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)

# Save Final PDF
output_path = selected_dir / 'background_data.pdf'
pdf.output(str(output_path))
print(f"Report successfully generated at: {output_path}")