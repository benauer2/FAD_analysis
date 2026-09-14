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
    compute_schedule_summary
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

'''
#############################
DATA ACQUISITION & IMPORTING
#############################
'''
dfs_dictionary, pre_reaction_df, selected_dir = select_folder_and_load_data("S:/projmon/velma_formic-acid-decomposition/Data/[BA12] Device #1 IDT (50um) IG(Silica + EDMIM TFSI; 30;1, 4um)-Au (50 nm)-SiO2(20 nm) + Pt (50 nm)/Rxn - 3 - 08.04.2026")
if dfs_dictionary is None:
    sys.exit("Data loading failed. Exiting script.")

# Check loaded datasets
GC_status = "FAD_df" in dfs_dictionary and dfs_dictionary["FAD_df"] is not None
electronics_status = "electronics_df" in dfs_dictionary and dfs_dictionary["electronics_df"] is not None
eis_status = "EIS_df" in dfs_dictionary and dfs_dictionary["EIS_df"] is not None
pre_rxn_status = isinstance(pre_reaction_df, dict) and len(pre_reaction_df) > 0


GC_df = dfs_dictionary.get("FAD_df")
raw_electronics_df = dfs_dictionary.get("electronics_df")
eis_df = dfs_dictionary.get("EIS_df")


# Process YAML Configurations
config.handle_yaml(selected_dir)
BASE_DIR = config.BASE_DIR
JSON_PARAMS_FID = config.JSON_PARAMS_FID
JSON_PARAMS_TCD = config.JSON_PARAMS_TCD
notes = config.notes
analysis = config.analysis
SHADE_COLORS = config.SHADE_COLORS
TRIAL_INFO = config.TRIAL_INFO
GC_METHOD_INFO = config.GC_METHOD_INFO
RXN_SCHEDULE = config.RXN_SCHEDULE

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

# Have the trial info say varied if that is the case for FA and Temps
# Figure out another way to incorporate it into the calucaltions? Should be almost easier?

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
#expected_values["Expected CO2 mol frac (ppm)"] = f"{exp_co2_fr * 1E-9 / (101325 / 14.696 * (p_reac_f + 14.696)) * 8.314 * (273.15 + t_reac_f) * 1E6 / (tot_flow_f * (273.15 + t_reac_f) / 300) * 1E6:.4g}"
#expected_values["Expected CO2 peak area (pA s)"] = f"{exp_conv_f / 100 * float(expected_values['Expected FA peak areas (pA s)']):.5g}"

'''
######################################
PROCESSING OF GC AND ELECTRONICS DATA
######################################
'''
cv_status = True
df_electronics = getcstats(df_raw=raw_electronics_df, file_path=None, data_dir=None, save_to_data_dir=False)
df_electronics.to_csv(data_dir / 'current_vs_time.csv', index=False)


try:
    df_cv, cv_seg = get_cv(raw_electronics_df)
    if cv_seg:
        capacitance = calc_cap_from_cv(cv_seg, rxn_schedule=RXN_SCHEDULE)
    else:
        print("[WARNING] No CV segments extracted from raw_electronics_df.")
        cv_status = False
except Exception as e:
    print(f"[ERROR] Failed to process CV data: {e}")
    cv_status = False

OCP_status = (df_electronics['Step name'] == 'Open Circuit Potential').any()

rates_df = calc_rxn_rates(GC_df=GC_df, RXN_SCHEDULE=RXN_SCHEDULE, tot_flow_f=tot_flow_f, v_box_t=v_box_t, p_reac_f=p_reac_f, GC_METHOD_INFO=GC_METHOD_INFO)
rates_df.to_csv(data_dir / "rates_vs_time.csv", index=False)
summary_rates_df = compute_schedule_summary(rates_df, RXN_SCHEDULE)
summary_rates_df.to_csv(data_dir / 'avg_rates.csv', index=False)

if electronics_status and df_electronics is not None and not df_electronics.empty:
    try:
        rates_df_fe = calculate_gc_faradaic_efficiency(
            rates_df=rates_df,
            df_electronics=df_electronics,
            n_electrons=2,
            gc_sampling_window_sec=60
        )
        rates_df_fe.to_csv(data_dir / 'rates_with_faradaic_efficiency.csv', index=False)
    except Exception as e:
        print(f"[WARNING] Faradaic efficiency calculation skipped: {e}")

    try:
        df_charge = charge_transfer(df_electronics)
        df_charge.to_csv(data_dir / 'charge_transfer_summary.csv', index=False)
    except Exception as e:
        print(f"[WARNING] Charge transfer analysis skipped: {e}")

tod = date.today().strftime("%m. %d. %Y.")

'''
#########################################
HANDLE THE PRE REACTION CAPACITANCE DATA
#########################################
'''
target_key = None
max_version = -1

if pre_rxn_status:
    for key in pre_reaction_df.keys():
        if "EIS" in key and "N2" in key and "C" in key:
            match = re.search(r'_v(\d+)$', key)
            if match:
                version_num = int(match.group(1))
                if version_num > max_version:
                    max_version = version_num
                    target_key = key
            elif max_version == -1:
                target_key = key

if target_key is not None:
    circuit, (fig, axes), fitted_params, EIS_summary = EIS_cap_pre((pre_reaction_df[target_key]), target_key)
    pre_rxn_cap = EIS_summary['cpe1_0'].iloc[0]
else:
    print("Warning: No pre-reaction EIS folder to pull capacitance from.")
    pre_rxn_cap = 0.0

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

fig, ax = plot_gas_rates(rates_df=rates_df, df_electronics=df_electronics, rxn_schedule=RXN_SCHEDULE, shade_colors=SHADE_COLORS, x_interval=8)
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
#############################
PAGE 2: FORMIC ACID & CURRENT DATA
#############################
'''
pdf.add_page()
fig, ax = plot_formic_acid(rates_df, shade_colors=SHADE_COLORS)
plot_file_path = plots_dir / "FA_data.png"
fig.savefig(plot_file_path, dpi=300, bbox_inches="tight")
plt.close(fig)

pdf.set_font("Times", "B", 12)
pdf.cell(0, 10, "Formic Acid Concentration Plot", new_x=XPos.RIGHT, new_y=YPos.NEXT)
pdf.image(plot_file_path, enums.Align.C, y=pdf.get_y(), w=180)
pdf.ln(105)

fig, ax = plot_current_time(rates_df, df_electronics, rxn_schedule=RXN_SCHEDULE, shade_colors=SHADE_COLORS)
plot_file_path = plots_dir / "current_data.png"
fig.savefig(plot_file_path, dpi=300, bbox_inches="tight")
plt.close(fig)

pdf.set_font("Times", "B", 12)
pdf.cell(0, 10, "Current vs. Time Profile", new_x=XPos.RIGHT, new_y=YPos.NEXT)
pdf.image(plot_file_path, enums.Align.C, y=pdf.get_y(), w=180)

# OCP PLOT IF AVAILABLE
if OCP_status:
    pdf.add_page()
    fig, ax = plot_OCP(rates_df, df_electronics, shade_colors=SHADE_COLORS, rxn_schedule=RXN_SCHEDULE)
    plot_file_path = plots_dir / "OCP_data.png"
    fig.savefig(plot_file_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    pdf.set_font("Times", "B", 12)
    pdf.cell(0, 10, "Open Circuit Potential (OCP) Plot", new_x=XPos.RIGHT, new_y=YPos.NEXT)
    pdf.image(plot_file_path, enums.Align.C, y=pdf.get_y(), w=180)



'''
#############################################
VOLTAGE WINDOW RATES BAR CHART & SUMMARY
#############################################
'''
if not summary_rates_df.empty:
    # --- PAGE A: GAS RATES BAR CHART & SUMMARY TABLE ---
    pdf.add_page()
    
    fig, ax = plot_voltage_rates_barchart(summary_rates_df, shade_colors=SHADE_COLORS)
    bar_plot_path = plots_dir / "voltage_rates_barchart.png"
    fig.savefig(bar_plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    pdf.set_font("Times", "B", 14)
    pdf.cell(0, 10, "Voltage Window Reaction Rate Averages", new_x=XPos.RIGHT, new_y=YPos.NEXT)
    pdf.image(bar_plot_path, enums.Align.C, y=pdf.get_y(), w=180)
    pdf.ln(100)

    pdf.set_font("Times", "B", 12)
    pdf.cell(0, 8, "Steady-State Rate Summary & Uncertainty Analysis", new_x=XPos.RIGHT, new_y=YPos.NEXT)
    pdf.ln(2)

    headers = ["Win", "Bias (V)", "Time (hr)", "CO2 Rate (nmol/min)", "H2 Rate (nmol/min)", "CO Rate (nmol/min)", "CO2/CO Ratio"]
    table_data = [headers]
    
    # Table Header
    col_widths = [10, 25, 20, 13, 20, 25, 25, 25, 25]
    headers = ["Run", "Rxr Temp (C)", "FA (ul/min)", "WE (V)", "Avg Window", "Run Time (hrs)", "CO2 (nmol/min)", "H2 (nmol/min)", "CO (nmol/min)"]

    pdf.set_font('Times', 'B', 9)
    for w, h in zip(col_widths, headers):
        pdf.cell(w, 6, h, border=1, align='C')
    pdf.ln()

    # Table Rows
    pdf.set_font('Times', '', 9)
    for _, row in summary_rates_df.iterrows():
        pdf.cell(col_widths[0], 6, f"{row.get('Run', '')}", border=1, align='C')
        pdf.cell(col_widths[1], 6, f"{row.get('Rxr_T_C', row.get('Reactor_T_C', ''))}", border=1, align='C')
        pdf.cell(col_widths[2], 6, f"{row.get('FA_flow_ul.min', row.get('FA_flow_ul_min', ''))}", border=1, align='C')
        pdf.cell(col_widths[3], 6, f"{row.get('WE_Voltage', row.get('Voltage (V)', ''))}", border=1, align='C')
        pdf.cell(col_widths[4], 6, f"{row.get('Avg_Window', '')}", border=1, align='C')
        pdf.cell(col_widths[5], 6, f"{row.get('Run_Time', '')}", border=1, align='C')
        pdf.cell(col_widths[6], 6, f"{row.get('CO2 (nmol/min)', row.get('CO2_avg', ''))}", border=1, align='C')
        pdf.cell(col_widths[7], 6, f"{row.get('H2 (nmol/min)', row.get('H2_avg', ''))}", border=1, align='C')
        pdf.cell(col_widths[8], 6, f"{row.get('CO (nmol/min)', row.get('CO_avg', ''))}", border=1, align='C')
        pdf.ln()
    '''
    for _, r in summary_rates_df.iterrows():
        time_str = f"{r['Start (hr)']:.1f}-{r['End (hr)']:.1f}h"
        co2_str = f"{r['CO2_avg']:.2f} \u00b1 {r['CO2_err']:.2f}"
        h2_str = f"{r['H2_avg']:.2f} \u00b1 {r['H2_err']:.2f}"
        co_str = f"{r['CO_avg']:.2f} \u00b1 {r['CO_std']:.2f}"
        sel_str = f"{r['Selectivity_mean']:.2f} \u00b1 {r['Selectivity_err']:.2f}"
        
        table_data.append([
            str(int(r["Window"])),
            f"{r['Voltage (V)']:.2f}",
            time_str,
            co2_str,
            h2_str,
            co_str,
            sel_str
        ])

    with pdf.table(col_widths=(10, 18, 24, 32, 32, 32, 32), text_align=("C", "C", "C", "C", "C", "C", "C")) as table:
        for r_idx, row_values in enumerate(table_data):
            row = table.row()
            pdf.set_font("Times", "B" if r_idx == 0 else "", 8)
            for datum in row_values:
                row.cell(str(datum))
    '''
'''
    # --- PAGE B: SELECTIVITY BAR CHART ---
    pdf.add_page()
    fig, ax = plot_selectivity_barchart(summary_rates_df, shade_colors=SHADE_COLORS)
    sel_plot_path = plots_dir / "selectivity_barchart.png"
    fig.savefig(sel_plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    pdf.set_font("Times", "B", 14)
    pdf.cell(0, 10, "CO2 / CO Selectivity Ratio by Voltage Window", new_x=XPos.RIGHT, new_y=YPos.NEXT)
    pdf.image(sel_plot_path, enums.Align.C, y=pdf.get_y(), w=180)
'''

'''
######################################
PAGE 3: IMPEDANCE & CAPACITANCE DATA
######################################
'''
if pre_rxn_status:
    pdf.add_page()
    pdf.set_font('Times', 'B', 14)
    pdf.cell(0, 7, "Impedance Data", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(3)

    pre_rxn_eis_rows = []
    insitu_eis_rows = []

    pdf.set_font('Times', 'B', 14)
    pdf.cell(0, 7, "Pre-reaction Data", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font('Times', '', 9)
    pdf.ln(3)

    for key in pre_reaction_df.keys():
        if "EIS" in key:
            circuit, (fig, axes), fitted_params, EIS_summary = EIS_cap_pre((pre_reaction_df[key]), key)
            EIS_plot_file_path = plots_dir / f"EIS_{key}.png"
            fig.savefig(EIS_plot_file_path, dpi=300, bbox_inches="tight")
            plt.close(fig)

            pre_rxn_eis_rows.append({
                "EIS_Run_ID": str(key),
                "Series Resistance (wire, contact)": f"{EIS_summary['r0_value'].iloc[0]:.2f}",
                "Capacitor Leakage Resistance": f"{EIS_summary['r1_value'].iloc[0]:.2f}",
                "Device Capacitance": f"{EIS_summary['cpe1_0'].iloc[0]:.2f}",
                "DC Working Electrode (V)": f"{EIS_summary['DC Working Electrode (V)'].iloc[0]}"
            })

            if pdf.get_y() > 220:
                pdf.add_page()
            pdf.image(EIS_plot_file_path, enums.Align.C, y=pdf.get_y(), w=180)
            pdf.ln(75)
            pdf.cell(w=0, h=7, text=f"Series Resistance (wire, contact): {EIS_summary['r0_value'].iloc[0]:.2f} Ohms", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.cell(w=0, h=7, text=f"Capacitor Leakage Resistance: {EIS_summary['r1_value'].iloc[0]:.2f} Ohms", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.cell(w=0, h=7, text=f"Device Capacitance: {EIS_summary['cpe1_0'].iloc[0]:.2f} uF", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.cell(w=0, h=7, text=f"DC Working Electrode (V): {EIS_summary['DC Working Electrode (V)'].iloc[0]}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.ln(5)

    if eis_status:
        pdf.add_page()
        pdf.set_font('Times', 'B', 14)
        pdf.cell(0, 7, "In-Situ Data", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_font('Times', '', 9)
        pdf.ln(3)
        eis_parsed_df = reaction_EIS_parser(eis_df)
        for unique_value, chunk in eis_parsed_df.groupby('EIS_Run_ID', sort=False, dropna=True):
            circuit, (fig, axes), fitted_params, EIS_summary = EIS_cap_rxn(chunk, unique_value, RXN_SCHEDULE)
            EIS_plot_file_path = plots_dir / f"EIS_{unique_value}.png"
            fig.savefig(EIS_plot_file_path, dpi=300, bbox_inches="tight")
            plt.close(fig)

            cond_val = EIS_summary['Preceding Condition'].iloc[0]
            cond_str = str(cond_val) if str(cond_val).endswith("V") or str(cond_val) == "Initial Setup" or str(cond_val) == "N/A" else f"{cond_val} V"

            insitu_eis_rows.append({
                "Time (hr)": f"{EIS_summary['Time (hr)'].iloc[0]:.1f}",
                "Preceding Condition": cond_str,
                "Series Resistance (wire, contact)": f"{EIS_summary['r0_value'].iloc[0]:.2f}",
                "Capacitor Leakage Resistance": f"{EIS_summary['r1_value'].iloc[0]:.2f}",
                "Device Capacitance": f"{EIS_summary['cpe1_0'].iloc[0]:.2f}",
                "DC Working Electrode (V)": f"{EIS_summary['DC Working Electrode (V)'].iloc[0]}",
            })

            if pdf.get_y() > 220:
                pdf.add_page()
            pdf.image(EIS_plot_file_path, enums.Align.C, y=pdf.get_y(), w=180)
            pdf.ln(75)
            pdf.cell(w=0, h=7, text=f"Series Resistance (wire, contact): {EIS_summary['r0_value'].iloc[0]:.2f} Ohms", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.cell(w=0, h=7, text=f"Capacitor Leakage Resistance: {EIS_summary['r1_value'].iloc[0]:.2f} Ohms", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.cell(w=0, h=7, text=f"Device Capacitance: {EIS_summary['cpe1_0'].iloc[0]:.2f} uF", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.cell(w=0, h=7, text=f"Preceding Condition: {cond_str}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.cell(w=0, h=7, text=f"DC Working Electrode (V): {EIS_summary['DC Working Electrode (V)'].iloc[0]}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.cell(w=0, h=7, text=f"Time (hr): {EIS_summary['Time (hr)'].iloc[0]:.1f}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.ln(5)
'''
######################################
TABULATED EIS SUMMARY TABLE
######################################

pdf.add_page()
pdf.set_font("Times", "B", 14)
pdf.cell(0, 10, "EIS Equivalent Circuit Parameter Tabulation", new_x=XPos.RIGHT, new_y=YPos.NEXT)
pdf.ln(4)

pdf.set_font("Times", "B", 11)
pdf.cell(0, 7, "Pre-Reaction EIS Summary Metrics", new_x=XPos.RIGHT, new_y=YPos.NEXT)
pdf.ln(2)

df_pre_table = pd.DataFrame(pre_rxn_eis_rows)
if not df_pre_table.empty:
    pre_table_data = [df_pre_table.columns.tolist()] + df_pre_table.values.tolist()
    pdf.set_font("Times", size=9)
    with pdf.table(col_widths=(60, 30, 30, 30, 30), text_align=("L", "C", "C", "C", "C")) as table:
        for r_idx, data_row in enumerate(pre_table_data):
            row = table.row()
            pdf.set_font("Times", "B" if r_idx == 0 else "", 9)
            for datum in data_row:
                row.cell(str(datum))
else:
    pdf.set_font("Times", "I", 9)
    pdf.cell(0, 5, "No pre-reaction EIS data collected.", new_x=XPos.RIGHT, new_y=YPos.NEXT)
pdf.ln(10)

pdf.set_font("Times", "B", 11)
pdf.cell(0, 7, "In-Situ In-Operando EIS Summary Metrics", new_x=XPos.RIGHT, new_y=YPos.NEXT)
pdf.ln(2)

df_insitu_table = pd.DataFrame(insitu_eis_rows)
if not df_insitu_table.empty:
    insitu_table_data = [df_insitu_table.columns.tolist()] + df_insitu_table.values.tolist()
    pdf.set_font("Times", size=9)
    with pdf.table(col_widths=(30, 30, 30, 30, 35, 25), text_align=("C", "C", "C", "C", "C", "C")) as table:
        for r_idx, data_row in enumerate(insitu_table_data):
            row = table.row()
            pdf.set_font("Times", "B" if r_idx == 0 else "", 9)
            for datum in data_row:
                row.cell(str(datum))
else:
    pdf.set_font("Times", "I", 9)
    pdf.cell(0, 5, "No in-situ EIS data collected.", new_x=XPos.RIGHT, new_y=YPos.NEXT)
pdf.ln(5)
'''

'''
###########################################
PRE-REACTION CYCLIC VOLTAMMETRY (CV) DATA
###########################################
'''
if cv_status:
    # Use max_per_page=4 or 6 so subplots fit cleanly on a single PDF page
    figs = cv_plot(cv_seg, rxn_schedule=RXN_SCHEDULE, max_per_page=14)
    
    for i, fig in enumerate(figs, start=1):
        pdf.add_page()
        # FIX 1: Unique file name per figure page so plots don't overwrite each other
        plot_file_path = plots_dir / f"CV_data_page_{i}.png"
        fig.savefig(plot_file_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        pdf.set_font("Times", "B", 12)
        pdf.cell(0, 10, f"Reaction CV Data (Page {i})", new_x=XPos.RIGHT, new_y=YPos.NEXT)
        
        # FIX 2: Fit image neatly on page without triggering fpdf auto-pagebreak
        pdf.image(plot_file_path, enums.Align.C, y=pdf.get_y(), w=175)

    # Move to next page for the Capacitance summary table
    pdf.add_page()
    pdf.set_font("Times", size=10)
    
    render_df = pd.DataFrame()
    render_df["CV Run"] = capacitance["CV Run"].astype(str)
    render_df["Sweep Rate (V/s)"] = capacitance["Sweep Rate (V/s)"].map("{:.1f}".format)
    render_df["Capacitance (uF)"] = capacitance["Capacitance (uF)"].map("{:.4f}".format)
    render_df["Time (hr)"] = capacitance["Time (hr)"].map("{:.1f}".format)
    render_df["Preceding Condition"] = capacitance["Preceding Condition"].astype(str)

    table_data = [render_df.columns.tolist()] + render_df.values.tolist()

    with pdf.table(col_widths=(25, 45, 40, 45, 35), text_align=("C", "C", "C", "C", "C")) as table:
        for data_row in table_data:
            row = table.row()
            for datum in data_row:
                row.cell(datum)

    fig, ax = cap_plot(cv_seg, x_interval=12, rxn_schedule=RXN_SCHEDULE, shade_colors=SHADE_COLORS)
    plot_file_path = plots_dir / "capacitance_data.png"
    fig.savefig(plot_file_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    pdf.ln(8)
    if pdf.get_y() > 180:
        pdf.add_page()
    pdf.set_font("Times", "B", 12)
    pdf.cell(0, 10, "Transient Capacitance Trend", new_x=XPos.RIGHT, new_y=YPos.NEXT)
    pdf.image(plot_file_path, enums.Align.C, y=pdf.get_y(), w=180)

    df_cv.to_csv(data_dir / 'rxn_cv_data.csv', index=False)
    render_df.to_csv(data_dir / 'rxn_capacitcance_data.csv', index=False)


'''
##############################################
PRE-REACTION CYCLIC VOLTAMMETRY DIAGNOSTICS
##############################################
'''
pdf.add_page()
pdf.set_font('Times', 'B', 14)
pdf.cell(0, 10, "Pre-Reaction Electrochemical Diagnostics", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
pdf.ln(5)

cv_keys = [key for key in pre_reaction_df.keys() if "CV" in key] if pre_rxn_status else []

if cv_keys:
    cols = 2
    rows = int(np.ceil(len(cv_keys) / cols))
    
    fig, axes = plt.subplots(rows, cols, figsize=(12, 3.5 * rows), squeeze=False)
    axes = axes.flatten()
    
    for i, key in enumerate(cv_keys):
        plot_10CV(pre_reaction_df[key], title_name=key, ax=axes[i])
        
    for j in range(len(cv_keys), len(axes)):
        fig.delaxes(axes[j])
        
    fig.tight_layout()
    cv_grid_path = plots_dir / "CV_Diagnostics_Grid.png"
    fig.savefig(cv_grid_path, dpi=400, bbox_inches="tight")
    plt.close(fig)
        
    pdf.image(cv_grid_path, enums.Align.C, y=pdf.get_y(), w=180)
else:
    pdf.set_font('Times', 'I', 11)
    pdf.cell(0, 10, "No pre-reaction diagnostic CV folders available to analyze.", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

'''
#############################################
TARGETED CAPACITANCE EVALUATION
#############################################
'''
if pre_rxn_status:
    pdf.add_page()
    cv_summary_df = calc_cap_from_cv_grouped(pre_reaction_df)

    if not cv_summary_df.empty:
        render_df = pd.DataFrame()
        render_df["Folder Source"] = cv_summary_df["Folder Source"].astype(str)
        render_df["Sweep Rate (V/s)"] = cv_summary_df["Sweep Rate (V/s)"].map("{:.1f}".format)
        render_df["Voltage Delta (V)"] = cv_summary_df["Voltage Delta (V)"].map("{:.1f}".format)
        render_df["Capacitance (uF)"] = cv_summary_df["Capacitance (uF)"].map("{:.4f}".format)

        table_data = [render_df.columns.tolist()] + render_df.values.tolist()

        pdf.set_font("Times", "B", 14)
        pdf.cell(0, 12, "CV Capacitance Summary Table", new_x=XPos.RIGHT, new_y=YPos.NEXT)
        pdf.ln(2)

        with pdf.table(
            col_widths=(60, 30, 35, 40, 25),
            text_align=("L", "C", "C", "C", "C"),
        ) as table:
            for row_idx, data_row in enumerate(table_data):
                is_header = (row_idx == 0)
                is_sweep_1 = (not is_header and data_row[1] == "1.0")
                if is_header or is_sweep_1:
                    row = table.row()
                    for datum in data_row:
                        pdf.set_font("Times", "B" if is_header else "", 10 if is_header else 8)
                        row.cell(datum)

        pdf.set_font("Times", "", 10)
        pdf.ln(10)

'''
#######################################################
CV SWEEP RATE SLOPE CAPACITANCE EXTRACTION (i vs. v)
#######################################################

if cv_status and cv_seg:
    eval_pot = 0.0  # Evaluation potential in Volts
    slope_summary_df, slope_fit_details = calc_cap_from_sweep_rate_slope(
        cv_seg, 
        eval_potential=eval_pot, 
        rxn_schedule=RXN_SCHEDULE,
        sweeps_per_chunk=3
    )

    if not slope_summary_df.empty:
        # PAGE 1: Capacitance vs Time Trend + Summary Table
        pdf.add_page()
        pdf.set_font("Times", "B", 14)
        pdf.cell(0, 10, "Capacitance Over Reaction Time (Slope Method)", new_x=XPos.RIGHT, new_y=YPos.NEXT)
        pdf.ln(2)

        fig, ax = plt.subplots(figsize=(8, 3.8))
        ax.plot(slope_summary_df["Time (hr)"], slope_summary_df["Capacitance (uF)"], 'o-', color="#2b5c8f", linewidth=2, markersize=6)
        ax.set_xlabel("Time (hr)", fontsize=11)
        ax.set_ylabel(r"Capacitance (uF)", fontsize=11)
        ax.set_title(f"Chunk-wise Capacitance Extraction @ {eval_pot} V", fontsize=12, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.5)
        
        fig.subplots_adjust(left=0.12, right=0.95, top=0.88, bottom=0.18)
        slope_plot_path = plots_dir / "cv_slope_capacitance_time.png"
        fig.savefig(slope_plot_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        pdf.image(slope_plot_path, enums.Align.C, y=pdf.get_y(), w=170)
        pdf.ln(90)

        pdf.set_font("Times", "B", 12)
        pdf.cell(0, 8, "Chunk-wise Capacitance Summary (i = C * v + i_leak)", new_x=XPos.RIGHT, new_y=YPos.NEXT)
        pdf.ln(2)

        slope_table_headers = ["Chunk", "Time (hr)", "Preceding Condition", "Capacitance (uF)", "Leakage (uA)", "R-Squared (R2)"]
        
        table_rows = [slope_table_headers]
        for _, row in slope_summary_df.iterrows():
            table_rows.append([
                str(int(row["Chunk ID"])),
                f"{row['Time (hr)']:.1f}",
                str(row["Preceding Condition"]),
                f"{row['Capacitance (uF)']:.4f}",
                f"{row['Leakage Current (uA)']:.2f}",
                f"{row['R-squared']:.4f}"
            ])

        with pdf.table(col_widths=(18, 22, 40, 35, 35, 30), text_align=("C", "C", "C", "C", "C", "C")) as table:
            for r_idx, row_values in enumerate(table_rows):
                row = table.row()
                pdf.set_font("Times", "B" if r_idx == 0 else "", 8)
                for datum in row_values:
                    row.cell(datum)

        # PAGE 2: All individual chunk fits on a single grid page
        pdf.add_page()
        pdf.set_font("Times", "B", 14)
        pdf.cell(0, 10, "Individual Chunk Slope Fits Grid", new_x=XPos.RIGHT, new_y=YPos.NEXT)
        pdf.ln(2)

        grid_fig, _ = plot_all_cv_chunks_grid(slope_fit_details, max_cols=3)
        if grid_fig:
            grid_plot_path = plots_dir / "cv_chunks_grid.png"
            grid_fig.savefig(grid_plot_path, dpi=300, bbox_inches="tight")
            plt.close(grid_fig)

            pdf.image(grid_plot_path, enums.Align.C, y=pdf.get_y(), w=180)

        slope_summary_df.to_csv(data_dir / 'cv_slope_capacitance_summary.csv', index=False)
'''

'''
#######################################################
AVERAGE RATES VS CAPACITANCE CORRELATION ANALYSIS
#######################################################
'''
if 'summary_rates_df' in locals() and not summary_rates_df.empty and 'capacitance' in locals() and not capacitance.empty:
    
    # Run safe schedule correlation
    window_avg_corr_df = correlate_rates_and_capacitance_by_schedule(
        summary_rates_df, capacitance, rxn_schedule=RXN_SCHEDULE
    )

    if not window_avg_corr_df.empty:
        # PDF & Plot generation
        pdf.add_page()
        pdf.set_font("Times", "B", 14)
        pdf.cell(0, 10, "Average Rates vs. Capacitance (Window Aligned)", new_x=XPos.RIGHT, new_y=YPos.NEXT)
        pdf.ln(2)

        fig, ax = plot_rate_vs_capacitance(window_avg_corr_df, figsize=(9, 4.2))
        rate_cap_plot_path = plots_dir / "rate_vs_capacitance.png"
        fig.savefig(rate_cap_plot_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        pdf.image(rate_cap_plot_path, enums.Align.C, y=pdf.get_y(), w=175)
        pdf.ln(95)

        # PDF Table Output
        corr_headers = ["Win (#)", "Bias (V)", "Cap (uF)", "CO2 Rate", "H2 Rate", "CO Rate"]
        corr_rows = [corr_headers]

        for _, r in window_avg_corr_df.iterrows():
            corr_rows.append([
                f"{int(r['Window'])}.{int(r['CV_Measurement'])}",
                f"{r['Voltage (V)']:.2f}" if isinstance(r['Voltage (V)'], (int, float)) else str(r['Voltage (V)']),
                f"{r['Capacitance (uF)']:.2f}",
                f"{r['CO2_mean']:.2f} \u00b1 {r['CO2_err']:.2f}",
                f"{r['H2_mean']:.2f} \u00b1 {r['H2_err']:.2f}",
                f"{r['CO_mean']:.2f} \u00b1 {r['CO_err']:.2f}"
            ])

        with pdf.table(col_widths=(18, 22, 25, 40, 40, 35), text_align=("C", "C", "C", "C", "C", "C")) as table:
            for r_idx, row_values in enumerate(corr_rows):
                row = table.row()
                pdf.set_font("Times", "B" if r_idx == 0 else "", 8)
                for datum in row_values:
                    row.cell(datum)

        window_avg_corr_df.to_csv(data_dir / 'window_avg_rate_vs_capacitance.csv', index=False)


'''
####################
EQUIPMENT DATA PAGE
####################
'''
pdf.add_page()
pdf.set_font('Times', 'B', 14)
pdf.cell(0, 7, "Equipment Data", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
pdf.set_font('Times', '', 9)



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

# Export compiled report PDF
output_pdf_path = selected_dir / 'reaction_data.pdf'
pdf.output(str(output_pdf_path))
print(f"Report generated successfully at: {output_pdf_path}")


