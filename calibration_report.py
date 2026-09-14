"""
PDF Report Generator for Formic Acid Decomposition Trials
Author: Ben Auer
Some code adapted from Justin Hopkins.
"""
import sys
from pathlib import Path
from datetime import date
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fpdf import FPDF, XPos, YPos, enums

import config
from config import get_bias_at_time
from worker_files.file_utils import select_folder_and_load_data, projmon_loc
from worker_files.plot_worker import (
    plot_current_time,
    plot_formic_acid,
    plot_gas_rates,
    plot_OCP,
    cap_plot,
    cv_plot,
    plot_10CV,
    plot_voltage_rates_barchart,
    plot_selectivity_barchart,
    plot_cv_sweep_rate_slope,
    plot_all_cv_chunks_grid,
    plot_rate_vs_capacitance
)
from worker_files.GC_worker import (
    calc_rxn_rates,
    compute_voltage_window_summary,
    GC_calibration
)
from worker_files.electronics_worker import (
    get_cv,
    calc_cap_from_cv,
    EIS_cap_pre,
    reaction_EIS_parser,
    EIS_cap_rxn,
    getcstats,
    calc_cap_from_cv_grouped,
    charge_transfer,
    calculate_gc_faradaic_efficiency,
    calc_cap_from_sweep_rate_slope,
    correlate_rates_and_capacitance_by_schedule
)

dfs_dictionary, pre_reaction_df, selected_dir = select_folder_and_load_data()
if dfs_dictionary is None:
    sys.exit("Data loading failed. Exiting script.")

# Check loaded datasets
GC_status = "FAD_df" in dfs_dictionary and dfs_dictionary["FAD_df"] is not None
TEMP_status = "temp_df" in dfs_dictionary and dfs_dictionary["temp_df"] is not None
MFC_status = "mfc_df" in dfs_dictionary and dfs_dictionary["mfc_df"] is not None

print('GC status: ', GC_status)
print(dfs_dictionary["FAD_df"])
print('TEMP status: ', TEMP_status)
print(dfs_dictionary["temp_df"])
print('MFC status: ', MFC_status)
print(dfs_dictionary["mfc_df"])


GC_df = dfs_dictionary.get("FAD_df")
temp_df = dfs_dictionary.get("temp_df")
mfc_df = dfs_dictionary.get("mfc_df")

'''
# Process YAML Configurations
config.handle_yaml(selected_dir)
BASE_DIR = config.BASE_DIR
JSON_PARAMS_FID = config.JSON_PARAMS_FID
JSON_PARAMS_TCD = config.JSON_PARAMS_TCD
notes = config.notes
analysis = config.analysis
SHADE_COLORS = config.SHADE_COLORS
BIAS_SCHEDULE = config.BIAS_SCHEDULE
TRIAL_INFO = config.TRIAL_INFO
GC_METHOD_INFO = config.GC_METHOD_INFOs
'''

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
t_reac_f = float(TRIAL_INFO.get('T reactor (deg C)', 100.0))
mol_pt_s = float(expected_values['mol Pt sites'])

expected_values["Expected FA flow (sccm)"] = f"{fa_pct_f / 100 * tot_flow_f:.4g}" 
expected_values["Expected FA concentration (uL/mL)"] = f"{fa_pct_f / 100 * (p_reac_f + 14.696) / 14.696 * 101325 / 8.314 / (273.15 + v_box_t):.4g}" 
expected_values["Expected FA peak areas (pA s)"] = f"{float(expected_values['Expected FA concentration (uL/mL)']) / fid_calib:.4g}" 
expected_values["Expected Activity (mol_CO2/mol_Pt/min)"] = f"{np.exp(26.163) * np.exp(-104.3332 * 1000 / 8.314 / (273.15 + t_reac_f)) * 60 * (fa_pct_f / 100 * 760 / 5)**0.43 * (0.000001 * 760 / 15)**-0.4:.4g}"

exp_act_f = float(expected_values["Expected Activity (mol_CO2/mol_Pt/min)"])
expected_values["Expected conversion (%)"] = f"{exp_act_f * mol_pt_s / (0.0000013064 * (tot_flow_f * (273.15 + t_reac_f) / 293) + exp_act_f * mol_pt_s) * 100:.4g}"

exp_conv_f = float(expected_values["Expected conversion (%)"])
exp_fa_flow_f = float(expected_values["Expected FA flow (sccm)"])
expected_values["Expected CO2 flow rate (nmol/min)"] = f"{exp_conv_f / 100 * exp_fa_flow_f * 0.000001 * (101325 * (1 + p_reac_f / 14.696) / 8.314 / 293) * 1000000000:.4g}"

exp_co2_fr = float(expected_values["Expected CO2 flow rate (nmol/min)"])
expected_values["Expected CO2 mol frac (ppm)"] = f"{exp_co2_fr * 1E-9 / (101325 / 14.696 * (p_reac_f + 14.696)) * 8.314 * (273.15 + t_reac_f) * 1E6 / (tot_flow_f * (273.15 + t_reac_f) / 300) * 1E6:.4g}"
expected_values["Expected CO2 peak area (pA s)"] = f"{exp_conv_f / 100 * float(expected_values['Expected FA peak areas (pA s)']):.5g}"

'''

cal = GC_calibration((GC_df, flow_sched, p_reac_f, concs))



# ==========================================
# REPORT COMPILATION INFRASTRUCTURE (PDF)
# ==========================================

class MyPDF(FPDF):
    def header(self):
        self.set_font('Times', 'B', 12)
        self.cell(0, 10, 'Formic Acid Decomposition Trial Report', align='C')
        self.ln(10)
        
    def footer(self):
        self.set_y(-15)
        self.set_font('Times', 'I', 12)
        self.cell(0, 10, f'Page {self.page_no()} | Auer | {tod}', align='R', border=0)

pdf = MyPDF()

'''
#############################
PAGE 1: KINETICS & RATE DATA
#############################
'''
pdf.add_page()
pdf.set_font('Times', 'B', 12)
pdf.cell(pdf.get_string_width("Device: ") + 2, 10, "Device: ", new_x=XPos.RIGHT, new_y=YPos.LAST)
pdf.set_font('Times', '', 12)
pdf.cell(0, 10, f"{TRIAL_INFO.get('Device', 'N/A')}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
'''
pdf.set_font('Times', 'B', 12)
pdf.cell(pdf.get_string_width("Date: ") + 2, 10, "Date: ", new_x=XPos.RIGHT, new_y=YPos.LAST)
pdf.set_font('Times', '', 12)
pdf.cell(0, 10, f"{TRIAL_INFO.get('Date', 'N/A')}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

pdf.set_font('Times', 'B', 12)
pdf.cell(pdf.get_string_width("Reactor Temperature: ") + 2, 10, "Reactor Temperature: ", new_x=XPos.RIGHT, new_y=YPos.LAST)
pdf.set_font('Times', '', 12)
pdf.cell(0, 10, f"{TRIAL_INFO.get('T reactor (deg C)', 'N/A')} \u00b0C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

pdf.set_font('Times', 'B', 12)
pdf.cell(pdf.get_string_width("N2 flow (sccm): ") + 2, 10, "N2 flow (sccm): ", new_x=XPos.RIGHT, new_y=YPos.LAST)
pdf.set_font('Times', '', 12)
pdf.cell(0, 10, f"{TRIAL_INFO.get('N2 flow in setup (sccm)', 'N/A')}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

pdf.set_font('Times', 'B', 12)
pdf.cell(pdf.get_string_width("% FA in vapor stream: ") + 2, 10, "% FA in vapor stream: ", new_x=XPos.RIGHT, new_y=YPos.LAST)
pdf.set_font('Times', '', 12)
pdf.cell(0, 10, f"{TRIAL_INFO.get('Percent (%) FA in vapor stream', 'N/A')}%", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

pdf.set_font('Times', 'B', 12)
pdf.cell(pdf.get_string_width("Pre-rxn Capacitance (EIS, N2): ") + 2, 10, "Pre-rxn Capacitance (EIS): ", new_x=XPos.RIGHT, new_y=YPos.LAST)
pdf.set_font('Times', '', 12)
pdf.cell(0, 10, f"{pre_rxn_cap:.2f} uF", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

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
'''