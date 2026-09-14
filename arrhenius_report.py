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

import arrhenius_worker_files.arrhenius_config as arrhenius_config
from arrhenius_worker_files.file_utils_arrhenius import(select_folder_and_load_data)
from arrhenius_worker_files.plot_worker_arrhenius import(plot_temp_profile, plot_mfc_profile, plot_gas_rates, plot_formic_acid, plot_arrhenius_by_voltage)
from arrhenius_worker_files.arrhenius_electronics_worker import(getcstats)
from arrhenius_worker_files.arrhenius_GC_worker import(calc_rxn_rates, compute_schedule_summary, perform_arrhenius_analysis)
'''
from worker_files.plot_worker import plot_background_rates, plot_temp_profile, plot_mfc_profile, plot_arrhenius
from worker_files.GC_worker import calc_rxn_rates_dynamic, compute_schedule_summary, perform_arrhenius_analysis
'''
dfs_dictionary, pre_reaction_df, selected_dir = select_folder_and_load_data(selected_dir="S:/projmon/velma_formic-acid-decomposition/Data/[BA12] Device #1 IDT (50um) IG(Silica + EDMIM TFSI; 30;1, 4um)-Au (50 nm)-SiO2(20 nm) + Pt (50 nm)/Arrhenius - 2 - 09.01.2026")

GC_df   = dfs_dictionary.get("FAD_df")
temp_df = dfs_dictionary.get("temp_df")
mfc_df  = dfs_dictionary.get("mfc_df")
raw_electronics_df = dfs_dictionary.get("electronics_df")

# Process YAML Configurations
arrhenius_config.handle_yaml(selected_dir)
BASE_DIR = arrhenius_config.BASE_DIR
JSON_PARAMS_FID = arrhenius_config.JSON_PARAMS_FID
JSON_PARAMS_TCD = arrhenius_config.JSON_PARAMS_TCD
notes = arrhenius_config.notes
analysis = arrhenius_config.analysis
SHADE_COLORS = arrhenius_config.SHADE_COLORS
BIAS_SCHEDULE = arrhenius_config.BIAS_SCHEDULE
TRIAL_INFO = arrhenius_config.TRIAL_INFO
GC_METHOD_INFO = arrhenius_config.GC_METHOD_INFO
RXN_SCHEDULE = arrhenius_config.RXN_SCHEDULE

n2_flow = float(TRIAL_INFO.get("N2 flow in setup (sccm)", 0.0))
fa_pct = float(TRIAL_INFO.get("Percent (%) FA in vapor stream", 4.0))
# Configure Output Paths
plots_dir = selected_dir / "plots"
data_dir = selected_dir / "data"
plots_dir.mkdir(parents=True, exist_ok=True)
data_dir.mkdir(parents=True, exist_ok=True)

# Font & Matplotlib Setup
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Calibri']


'''
########################################
METADATA VALUE GENERATION BASED ON YAML
########################################
'''
TRIAL_INFO["FA flow (uL/min)"] = f"{n2_flow * (fa_pct / 100) / (1 - (fa_pct / 100)) * 0.000001 * 101325 / 8.314 / 293 * 46.025 / 1.22 * 1000:.4g}"

expected_values = {
    "Percent (%) FA in vapor stream": fa_pct,
    "Total flow (sccm)": f"{(n2_flow + float(TRIAL_INFO.get('H2 flow (sccm)', 0.0))) / (1 - (fa_pct / 100)):.2f}",
    "Pt sites estimation (sites/cm^2)": f"{1E15:.1e}",
    "mol Pt sites": f"{float(TRIAL_INFO.get('Catalyst area (cm^2)', 1.0)) / 6.02E23 * 1E15:.2e}"
}

tot_flow_f = float(expected_values["Total flow (sccm)"])
fa_pct_f = float(expected_values["Percent (%) FA in vapor stream"])
p_reac_f = float(TRIAL_INFO.get("P reactor (psig)", 0.0))
v_box_t = float(GC_METHOD_INFO.get('Valve box T (deg C)', 100.0))
fid_calib = float(GC_METHOD_INFO.get('CO2 calibration factor (FID, umol/(pA s))', 1.0))
mol_pt_s = float(expected_values['mol Pt sites'])

expected_values["Expected FA flow (sccm)"] = f"{fa_pct_f / 100 * tot_flow_f:.4g}" 
expected_values["Expected FA concentration (uL/mL)"] = f"{fa_pct_f / 100 * (p_reac_f + 14.696) / 14.696 * 101325 / 8.314 / (273.15 + v_box_t):.4g}" 
expected_values["Expected FA peak areas (pA s)"] = f"{float(expected_values['Expected FA concentration (uL/mL)']) / fid_calib:.4g}" 

df_electronics = getcstats(df_raw=raw_electronics_df, file_path=None, data_dir=None, save_to_data_dir=False)
df_electronics.to_csv(data_dir / 'current_vs_time.csv', index=False)


rates_df = calc_rxn_rates(GC_df=GC_df, RXN_SCHEDULE=RXN_SCHEDULE, tot_flow_f=tot_flow_f, v_box_t=v_box_t, p_reac_f=p_reac_f, GC_METHOD_INFO=GC_METHOD_INFO)

summary_df = compute_schedule_summary(rates_df, RXN_SCHEDULE)
summary_df.to_csv(data_dir / 'avg_rates.csv', index=False)

rates_df.to_csv(data_dir / "rates_vs_time.csv", index=False)

arrhenius_results = perform_arrhenius_analysis(summary_df)
#arrhenius_results.to_csv(data_dir / "arrhenius.csv", index=False)

tod = date.today().strftime("%m. %d. %Y.")

'''
##############
BUILD OUT PDF
##############
'''
class ReportPDF(FPDF):
    def header(self):
        self.set_font('Times', 'B', 12)
        self.cell(0, 8, 'Formic Acid Decomposition Trial Report', align='C', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(4)
        
    def footer(self):
        self.set_y(-15)
        self.set_font('Times', 'I', 10)
        self.cell(0, 10, f'Page {self.page_no()} | Auer | {tod}', align='R', border=0)

pdf = ReportPDF()
pdf.set_auto_page_break(auto=True, margin=15)

# --- PAGE 1: OVERVIEW & TABULATED SUMMARY ---
pdf.add_page()
pdf.set_font('Times', 'B', 11)

pdf.cell(30, 6, "Date:", new_x=XPos.RIGHT, new_y=YPos.LAST)
pdf.set_font('Times', '', 11)
pdf.cell(0, 6, "Test", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

pdf.set_font('Times', 'B', 11)
pdf.cell(30, 6, "Trial:", new_x=XPos.RIGHT, new_y=YPos.LAST)
pdf.set_font('Times', '', 11)
pdf.cell(0, 6, "Test 2", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
pdf.ln(4)

fig, ax = plot_gas_rates(rates_df=rates_df, df_electronics=df_electronics, bias_schedule=BIAS_SCHEDULE, shade_colors=SHADE_COLORS, x_interval=8)
plot_file_path = plots_dir / "rate_data.png"
fig.savefig(plot_file_path, dpi=300, bbox_inches="tight")
plt.close(fig)

pdf.set_font("Times", "B", 12)
pdf.cell(0, 10, "Reaction Kinetics Plot", new_x=XPos.RIGHT, new_y=YPos.NEXT)
pdf.image(plot_file_path, enums.Align.C, y=pdf.get_y(), w=180)
pdf.ln(105)

line_height = 6
pdf.set_font('Times', 'B', 12)
pdf.write(line_height, "Reaction Notes: ")
pdf.set_font('Times', '', 12)
pdf.write(line_height, f"{notes}\n\n")

pdf.set_font('Times', 'B', 12)
pdf.write(line_height, "Analysis: ")
pdf.set_font('Times', '', 12)
pdf.write(line_height, f"{analysis}\n\n")

pdf.add_page()
# SUMMARY DATA TABLE
pdf.set_font('Times', 'B', 11)
pdf.cell(0, 6, "Summary Table (Steady-State Averages per Condition Window):", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
pdf.ln(2)

# Table Header
col_widths = [10, 25, 20, 13, 20, 25, 25, 25, 25]
headers = ["Run", "Rxr Temp (C)", "FA (ul/min)", "WE (V)", "Avg Window", "Run Time (hrs)", "CO2 (nmol/min)", "H2 (nmol/min)", "CO (nmol/min)"]

pdf.set_font('Times', 'B', 9)
for w, h in zip(col_widths, headers):
    pdf.cell(w, 6, h, border=1, align='C')
pdf.ln()

# Table Rows
pdf.set_font('Times', '', 9)
for _, row in summary_df.iterrows():
    pdf.cell(col_widths[0], 6, f"{row['Run']}", border=1, align='C')
    pdf.cell(col_widths[1], 6, f"{row['Rxr_T_C']}", border=1, align='C')
    pdf.cell(col_widths[2], 6, f"{row['FA_flow_ul.min']}", border=1, align='C')
    pdf.cell(col_widths[3], 6, f"{row['WE_Voltage']}", border=1, align='C')
    pdf.cell(col_widths[4], 6, f"{row['Avg_Window']}", border=1, align='C')
    pdf.cell(col_widths[5], 6, f"{row['Run_Time']}", border=1, align='C')
    pdf.cell(col_widths[6], 6, f"{row['CO2 (nmol/min)']}", border=1, align='C')
    pdf.cell(col_widths[7], 6, f"{row['H2 (nmol/min)']}", border=1, align='C')
    pdf.cell(col_widths[8], 6, f"{row['CO (nmol/min)']}", border=1, align='C')
    pdf.ln()

pdf.ln(6)

# --- ARRHENIUS ANALYSIS PAGES (DEDICATED PAGE PER VOLTAGE BIAS / OCP) ---
arrhenius_results = perform_arrhenius_analysis(summary_df)

for voltage, sp_dict in arrhenius_results.items():
    if not sp_dict:
        continue

    # Format header/label string
    v_title_str = "OCP" if voltage == "OCP" else f"{voltage} V"

    fig, ax, plot_file_path = plot_arrhenius_by_voltage(
        arrhenius_results=arrhenius_results, 
        target_voltage=voltage, 
        plots_dir=plots_dir
    )
    
    if plot_file_path is None:
        continue

    plt.close(fig)

    pdf.add_page()
    pdf.set_font("Times", "B", 12)
    pdf.cell(0, 10, f"Arrhenius Kinetics Analysis ({v_title_str} WE Bias)", new_x=XPos.RIGHT, new_y=YPos.NEXT)

    # Embed Voltage Plot
    pdf.image(plot_file_path, enums.Align.C, y=pdf.get_y(), w=160)
    pdf.ln(95)

    # Table Header
    pdf.set_font('Times', 'B', 11)
    pdf.cell(0, 6, f"Activation Energy & Fitting Parameters ({v_title_str} Bias):", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(2)

    arr_widths = [30, 50, 50, 30]
    arr_headers = ["Species", "Ea (kJ/mol)", "A Factor", "R2"]

    pdf.set_font('Times', 'B', 9)
    for w, h in zip(arr_widths, arr_headers):
        pdf.cell(w, 6, h, border=1, align='C')
    pdf.ln()

    # Table Rows
    pdf.set_font('Times', '', 9)
    for species, fit in sp_dict.items():
        ea_str = f"{fit['Ea_kJ']:.2f} +/- {fit['Ea_err_kJ']:.2f}" if pd.notna(fit['Ea_kJ']) else "N/A"
        a_str = f"{fit['A']:.2e}" if pd.notna(fit['A']) else "N/A"
        r2_str = f"{fit['R2']:.4f}" if pd.notna(fit['R2']) else "N/A"

        pdf.cell(arr_widths[0], 6, f"{species}", border=1, align='C')
        pdf.cell(arr_widths[1], 6, ea_str, border=1, align='C')
        pdf.cell(arr_widths[2], 6, a_str, border=1, align='C')
        pdf.cell(arr_widths[3], 6, r2_str, border=1, align='C')
        pdf.ln()

    pdf.ln(6)


# --- ADDITIONAL EXPERIMENTAL PROFILES ---
pdf.add_page()
fig, ax = plot_formic_acid(rates_df, shade_colors=SHADE_COLORS)
plot_file_path = plots_dir / "FA_data.png"
fig.savefig(plot_file_path, dpi=300, bbox_inches="tight")
plt.close(fig)

pdf.set_font("Times", "B", 12)
pdf.cell(0, 10, "Formic Acid Concentration Plot", new_x=XPos.RIGHT, new_y=YPos.NEXT)
pdf.image(plot_file_path, enums.Align.C, y=pdf.get_y(), w=180)
pdf.ln(105)

fig, ax = plot_temp_profile(temp_df=temp_df)
plot_file_path = plots_dir / "temp_data.png"
fig.savefig(plot_file_path, dpi=300, bbox_inches="tight")
plt.close(fig)
pdf.set_font("Times", "B", 12)
pdf.cell(0, 10, "Temperature Plot", new_x=XPos.RIGHT, new_y=YPos.NEXT)
pdf.image(plot_file_path, enums.Align.C, y=pdf.get_y(), w=180)
pdf.ln(105)

fig, ax = plot_mfc_profile(mfc_df=mfc_df)
plot_file_path = plots_dir / "mfc_data.png"
fig.savefig(plot_file_path, dpi=300, bbox_inches="tight")
plt.close(fig)
pdf.set_font("Times", "B", 12)
pdf.cell(0, 10, "MFC Plot", new_x=XPos.RIGHT, new_y=YPos.NEXT)
pdf.image(plot_file_path, enums.Align.C, y=pdf.get_y(), w=180)
pdf.ln(105)

'''
####################
METADATA PAGE
####################
'''
pdf.add_page()
pdf.set_font('Times', 'B', 14)
pdf.cell(0, 7, "Trial Parameters Metadata", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
pdf.set_font('Times', '', 9)
for key, value in TRIAL_INFO.items():
    pdf.cell(0, 5, f"{key}: {value}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
pdf.ln(3)

pdf.set_font('Times', 'B', 14)
pdf.cell(0, 7, "Expected Thermochemical Values", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
pdf.set_font('Times', '', 9)
for key, value in expected_values.items():
    pdf.cell(0, 5, f"{key}: {value}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
pdf.ln(3)

pdf.set_font('Times', 'B', 14)
pdf.cell(0, 7, "Chromatography Method (GC) Specifications", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
pdf.set_font('Times', '', 9)
for key, value in GC_METHOD_INFO.items():
    pdf.cell(0, 5, f"{key}: {value}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

# Save Final PDF
output_path = selected_dir / 'arhhenius_data.pdf'
pdf.output(str(output_path))
print(f"Report successfully generated at: {output_path}")
