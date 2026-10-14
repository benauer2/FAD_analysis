"""
PDF Report Generator for Formic Acid Decomposition Trials
Author: Ben Auer
Some code adapted from Justin Hopkins.
"""
from datetime import date
from pathlib import Path
import re
import sys

from fpdf import FPDF, XPos, YPos, enums
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

tod = date.today().strftime("%m. %d. %Y.")

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

# ---------------------------------------------------------
# PAGE 1: KINETICS & RATE DATA
# ---------------------------------------------------------
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

fa_pct_disp = f"{RXN_SCHEDULE[0].get('fa_pct', 0.0):.1f}%" if (RXN_SCHEDULE and len(RXN_SCHEDULE) == 1) else (f"{RXN_SCHEDULE[0].get('fa_pct', 0.0):.1f}% (Initial)" if RXN_SCHEDULE else f"{TRIAL_INFO.get('Percent (%) FA in vapor stream', 'N/A')}%")
pdf.set_font('Times', 'B', 12)
pdf.cell(pdf.get_string_width("% FA in vapor stream: ") + 2, 10, "% FA in vapor stream: ", new_x=XPos.RIGHT, new_y=YPos.LAST)
pdf.set_font('Times', '', 12)
pdf.cell(0, 10, fa_pct_disp, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

pdf.set_font('Times', 'B', 12)
pdf.cell(pdf.get_string_width("Pre-rxn Capacitance (CV, N2): ") + 2, 10, "Pre-rxn Capacitance (CV, N2): ", new_x=XPos.RIGHT, new_y=YPos.LAST)
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

# ---------------------------------------------------------
# PAGE 2: FORMIC ACID & CURRENT DATA
# ---------------------------------------------------------
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

if OCP_status:
    pdf.add_page()
    fig, ax = plot_OCP(rates_df, df_electronics, shade_colors=SHADE_COLORS, rxn_schedule=RXN_SCHEDULE)
    plot_file_path = plots_dir / "OCP_data.png"
    fig.savefig(plot_file_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    pdf.set_font("Times", "B", 12)
    pdf.cell(0, 10, "Open Circuit Potential (OCP) Plot", new_x=XPos.RIGHT, new_y=YPos.NEXT)
    pdf.image(plot_file_path, enums.Align.C, y=pdf.get_y(), w=180)

# ---------------------------------------------------------
# PAGE 3: VOLTAGE WINDOW RATES BAR CHART & SUMMARY TABLE
# ---------------------------------------------------------
if not summary_rates_df.empty:
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
    pdf.cell(0, 8, "Steady-State Rate Summary & Schedule Table", new_x=XPos.RIGHT, new_y=YPos.NEXT)
    pdf.ln(2)

    col_widths = [10, 22, 22, 16, 22, 22, 22, 22, 22]
    headers = ["Run", "Rxr T (C)", "FA (uL/m)", "WE (V)", "Avg Win", "Run Time", "CO2 (nmol)", "H2 (nmol)", "CO (nmol)"]

    pdf.set_font('Times', 'B', 8.5)
    for w, h in zip(col_widths, headers):
        pdf.cell(w, 6, h, border=1, align='C')
    pdf.ln()

    pdf.set_font('Times', '', 8.5)
    for _, row in summary_rates_df.iterrows():
        pdf.cell(col_widths[0], 6, str(row.get('Run', '')), border=1, align='C')
        pdf.cell(col_widths[1], 6, str(row.get('Rxr_T_C', row.get('Reactor_T_C', ''))), border=1, align='C')
        pdf.cell(col_widths[2], 6, str(row.get('FA_flow_ul.min', row.get('FA_flow_ul_min', ''))), border=1, align='C')
        pdf.cell(col_widths[3], 6, str(row.get('WE_Voltage', row.get('Voltage (V)', ''))), border=1, align='C')
        pdf.cell(col_widths[4], 6, str(row.get('Avg_Window', '')), border=1, align='C')
        pdf.cell(col_widths[5], 6, str(row.get('Run_Time', '')), border=1, align='C')
        pdf.cell(col_widths[6], 6, str(row.get('CO2 (nmol/min)', row.get('CO2_avg', ''))), border=1, align='C')
        pdf.cell(col_widths[7], 6, str(row.get('H2 (nmol/min)', row.get('H2_avg', ''))), border=1, align='C')
        pdf.cell(col_widths[8], 6, str(row.get('CO (nmol/min)', row.get('CO_avg', ''))), border=1, align='C')
        pdf.ln()

# ---------------------------------------------------------
# PAGE 4: IMPEDANCE (EIS) DATA (IN-SITU & PRE-REACTION)
# ---------------------------------------------------------
if not rxn_eis_summary.empty and rxn_eis_plots:
    for run_id, (circuit, fig, params) in rxn_eis_plots.items():
        pdf.add_page()
        step_name_s = rxn_eis_summary.loc[rxn_eis_summary["EIS Run"] == run_id, "Step Name"].values
        step_title = step_name_s[0] if len(step_name_s) > 0 else f"Run {run_id}"

        eis_plot_file_path = plots_dir / f"Reaction_EIS_Run_{run_id}.png"
        fig.savefig(eis_plot_file_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        pdf.set_font('Times', 'B', 14)
        pdf.cell(0, 10, f"Reaction Impedance (EIS) Data - {step_title}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(2)
        pdf.image(eis_plot_file_path, enums.Align.C, y=pdf.get_y(), w=180)
        pdf.ln(78)

        run_row = rxn_eis_summary[rxn_eis_summary["EIS Run"] == run_id].iloc[0]
        pdf.set_font('Times', 'B', 10)
        pdf.cell(w=0, h=5, text="Equivalent Circuit Model: R0-p(R1, CPE1)", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_font('Times', '', 10)
        pdf.cell(w=0, h=5, text=f"Series Resistance (wire, contact): {run_row['r0_value']:.2f} Ohms", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.cell(w=0, h=5, text=f"Capacitor Leakage Resistance (R1): {run_row['r1_value']:.2f} Ohms", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.cell(w=0, h=5, text=f"Device Capacitance (CPE1): {run_row['cpe1_0']:.4f} uF", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.cell(w=0, h=5, text=f"DC Bias: {run_row['DC Working Electrode (V)']} V | Time: {run_row['Time (hr)']} hr | Preceding: {run_row['Preceding Condition']}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(4)

if pre_rxn_status:
    eis_keys = [k for k in pre_reaction_df.keys() if "EIS" in k]
    if eis_keys:
        for key in eis_keys:
            circuit, (fig, axes), fitted_params, EIS_summary = EIS_cap_pre(pre_reaction_df[key], key)
            EIS_plot_file_path = plots_dir / f"EIS_{key}.png"
            fig.savefig(EIS_plot_file_path, dpi=300, bbox_inches="tight")
            plt.close(fig)

            pdf.add_page()
            pdf.set_font('Times', 'B', 14)
            pdf.cell(0, 10, f"Pre-Reaction Impedance (EIS) Data - {key}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.ln(2)
            pdf.image(EIS_plot_file_path, enums.Align.C, y=pdf.get_y(), w=180)
            pdf.ln(78)

            pdf.set_font('Times', 'B', 10)
            pdf.cell(w=0, h=5, text="Equivalent Circuit Model: R0-p(R1, CPE1)", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.set_font('Times', '', 10)
            pdf.cell(w=0, h=5, text=f"Series Resistance (wire, contact): {EIS_summary['r0_value'].iloc[0]:.2f} Ohms", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.cell(w=0, h=5, text=f"Capacitor Leakage Resistance: {EIS_summary['r1_value'].iloc[0]:.2f} Ohms", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.cell(w=0, h=5, text=f"Device Capacitance: {EIS_summary['cpe1_0'].iloc[0]:.2f} uF", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.cell(w=0, h=5, text=f"DC Working Electrode (V): {EIS_summary['DC Working Electrode (V)'].iloc[0]}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.ln(4)

# ---------------------------------------------------------
# PAGE 5: REACTION CYCLIC VOLTAMMETRY (CV) DATA & CAPACITANCE
# ---------------------------------------------------------
if cv_status and cv_seg and isinstance(capacitance, pd.DataFrame) and not capacitance.empty:
    # 1. Fast CV Sweeps (0.5, 1.0, 1.5 V/s)
    figs_fast = cv_plot(cv_seg, rxn_schedule=RXN_SCHEDULE, max_per_page=6, rate_filter='fast')
    for i, fig in enumerate(figs_fast, start=1):
        pdf.add_page()
        plot_file_path = plots_dir / f"CV_fast_data_page_{i}.png"
        fig.savefig(plot_file_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        pdf.set_font("Times", "B", 12)
        pdf.cell(0, 10, f"Reaction CV Data - Fast Sweeps (Page {i})", new_x=XPos.RIGHT, new_y=YPos.NEXT)
        pdf.image(plot_file_path, enums.Align.C, y=pdf.get_y(), w=160)

    # 2. Slow CV Sweeps (e.g. 0.02 V/s)
    has_slow = any(df['sweep_rate'].iloc[0] < 0.1 for df in cv_seg.values() if 'sweep_rate' in df.columns)
    if has_slow:
        figs_slow = cv_plot(cv_seg, rxn_schedule=RXN_SCHEDULE, max_per_page=6, rate_filter='slow')
        for i, fig in enumerate(figs_slow, start=1):
            pdf.add_page()
            plot_file_path = plots_dir / f"CV_slow_data_page_{i}.png"
            fig.savefig(plot_file_path, dpi=300, bbox_inches="tight")
            plt.close(fig)

            pdf.set_font("Times", "B", 12)
            pdf.cell(0, 10, f"Reaction CV Data - Slow Sweep 0.02 V/s (Page {i})", new_x=XPos.RIGHT, new_y=YPos.NEXT)
            pdf.image(plot_file_path, enums.Align.C, y=pdf.get_y(), w=160)

    # 3. Capacitance summary table & trend
    pdf.add_page()
    pdf.set_font("Times", size=10)
    
    render_df = pd.DataFrame()
    render_df["CV Run"] = capacitance["CV Run"].astype(str)
    render_df["Sweep Rate (V/s)"] = capacitance["Sweep Rate (V/s)"].apply(lambda r: f"{r:.2f}" if r < 0.1 else f"{r:.1f}")
    render_df["Capacitance (uF)"] = capacitance["Capacitance (uF)"].map("{:.4f}".format)
    render_df["Time (hr)"] = capacitance["Time (hr)"].map("{:.1f}".format)
    render_df["Preceding Condition"] = capacitance["Preceding Condition"].astype(str)

    table_data = [render_df.columns.tolist()] + render_df.values.tolist()

    with pdf.table(col_widths=(25, 45, 40, 45, 35), text_align=("C", "C", "C", "C", "C")) as table:
        for data_row in table_data:
            row = table.row()
            for datum in data_row:
                row.cell(datum)

    fig, ax = cap_plot(cv_seg, x_interval=12, rxn_schedule=RXN_SCHEDULE, shade_colors=SHADE_COLORS, target_sweep_rate=1.0)
    plot_file_path = plots_dir / "capacitance_data_1_0Vs.png"
    fig.savefig(plot_file_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    pdf.ln(8)
    if pdf.get_y() > 180:
        pdf.add_page()
    pdf.set_font("Times", "B", 12)
    pdf.cell(0, 10, "Transient Capacitance Trend (1.0 V/s)", new_x=XPos.RIGHT, new_y=YPos.NEXT)
    pdf.image(plot_file_path, enums.Align.C, y=pdf.get_y(), w=180)

    if has_slow:
        fig_slow_cap, ax_sc = cap_plot(cv_seg, x_interval=12, rxn_schedule=RXN_SCHEDULE, shade_colors=SHADE_COLORS, target_sweep_rate=0.02)
        plot_file_path_slow = plots_dir / "capacitance_data_0_02Vs.png"
        fig_slow_cap.savefig(plot_file_path_slow, dpi=300, bbox_inches="tight")
        plt.close(fig_slow_cap)

        pdf.add_page()
        pdf.set_font("Times", "B", 12)
        pdf.cell(0, 10, "Transient Capacitance Trend (0.02 V/s)", new_x=XPos.RIGHT, new_y=YPos.NEXT)
        pdf.image(plot_file_path_slow, enums.Align.C, y=pdf.get_y(), w=180)

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
            cv_grid_path = plots_dir / f"CV_Diagnostics_Grid_page_{page_idx}.png"
            fig.savefig(cv_grid_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
                
            pdf.add_page()
            pdf.set_font('Times', 'B', 14)
            page_title = "Pre-Reaction Electrochemical Diagnostics" if len(cv_keys) <= max_per_page else f"Pre-Reaction Electrochemical Diagnostics (Page {page_idx})"
            pdf.cell(0, 10, page_title, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.ln(2)
            pdf.image(cv_grid_path, enums.Align.C, y=pdf.get_y(), w=155)

# ---------------------------------------------------------
# PAGE 7: METADATA & SPECIFICATIONS
# ---------------------------------------------------------
pdf.add_page()
pdf.set_font('Times', 'B', 14)
pdf.cell(0, 7, "Trial Parameters Metadata", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
pdf.set_font('Times', '', 9)
for key, value in TRIAL_INFO.items():
    pdf.cell(0, 5, f"{key}: {value}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
pdf.ln(3)

pdf.set_font('Times', 'B', 14)
pdf.cell(0, 7, "Chromatography Method (GC) Specifications", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
pdf.set_font('Times', '', 9)
for key, value in GC_METHOD_INFO.items():
    pdf.cell(0, 5, f"{key}: {value}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)


if RXN_SCHEDULE:
    pdf.add_page()
    pdf.set_font('Times', 'B', 14)
    pdf.cell(0, 7, "Reaction Schedule Specification", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(2)
    
    has_co = any(step.get("co_flow", 0.0) > 0 for step in RXN_SCHEDULE)
    if has_co:
        col_w = [10, 14, 18, 18, 18, 18, 20, 20, 20]
        sched_headers = ["Run", "Dur (h)", "T (C)", "N2 (sccm)", "H2 (sccm)", "CO (sccm)", "FA (uL/m)", "Tot (sccm)", "WE (V)"]
    else:
        col_w = [12, 16, 20, 22, 22, 22, 22, 24]
        sched_headers = ["Run", "Dur (h)", "T (C)", "N2 (sccm)", "FA (uL/m)", "Tot (sccm)", "FA (%)", "WE (V)"]

    pdf.set_font('Times', 'B', 8.5)
    for w, h in zip(col_w, sched_headers):
        pdf.cell(w, 6, h, border=1, align='C')
    pdf.ln()
    pdf.set_font('Times', '', 8.5)
    for step in RXN_SCHEDULE:
        if has_co:
            pdf.cell(col_w[0], 6, str(step.get("run", "")), border=1, align='C')
            pdf.cell(col_w[1], 6, f"{step.get('dur_hrs', 0):.1f}", border=1, align='C')
            pdf.cell(col_w[2], 6, f"{step.get('rxr_t', 0):.1f}", border=1, align='C')
            pdf.cell(col_w[3], 6, f"{step.get('n2_flow', 0):.1f}", border=1, align='C')
            pdf.cell(col_w[4], 6, f"{step.get('h2_flow', 0):.1f}", border=1, align='C')
            pdf.cell(col_w[5], 6, f"{step.get('co_flow', 0):.1f}", border=1, align='C')
            pdf.cell(col_w[6], 6, f"{step.get('fa_flow', 0):.2f}", border=1, align='C')
            pdf.cell(col_w[7], 6, f"{step.get('tot_flow', 0):.2f}", border=1, align='C')
            pdf.cell(col_w[8], 6, str(step.get("we_volt", "")), border=1, align='C')
        else:
            pdf.cell(col_w[0], 6, str(step.get("run", "")), border=1, align='C')
            pdf.cell(col_w[1], 6, f"{step.get('dur_hrs', 0):.1f}", border=1, align='C')
            pdf.cell(col_w[2], 6, f"{step.get('rxr_t', 0):.1f}", border=1, align='C')
            pdf.cell(col_w[3], 6, f"{step.get('n2_flow', 0):.1f}", border=1, align='C')
            pdf.cell(col_w[4], 6, f"{step.get('fa_flow', 0):.2f}", border=1, align='C')
            pdf.cell(col_w[5], 6, f"{step.get('tot_flow', 0):.2f}", border=1, align='C')
            pdf.cell(col_w[6], 6, f"{step.get('fa_pct', 0):.2f}", border=1, align='C')
            pdf.cell(col_w[7], 6, str(step.get("we_volt", "")), border=1, align='C')
        pdf.ln()
    pdf.ln(4)

# ---------------------------------------------------------
# PAGE 8: EQUIPMENT & REAL-TIME PROFILES
# ---------------------------------------------------------
if temp_df is not None and not temp_df.empty:
    pdf.add_page()
    fig, ax = plot_temp_profile(temp_df=temp_df)
    plot_file_path = plots_dir / "temp_data.png"
    fig.savefig(plot_file_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    pdf.set_font("Times", "B", 12)
    pdf.cell(0, 10, "Temperature Profile", new_x=XPos.RIGHT, new_y=YPos.NEXT)
    pdf.image(plot_file_path, enums.Align.C, y=pdf.get_y(), w=180)
    pdf.ln(105)

if mfc_df is not None and not mfc_df.empty:
    pdf.add_page()
    fig, ax = plot_mfc_profile(mfc_df=mfc_df)
    plot_file_path = plots_dir / "mfc_data.png"
    fig.savefig(plot_file_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    pdf.set_font("Times", "B", 12)
    pdf.cell(0, 10, "MFC Gas Flow Profile", new_x=XPos.RIGHT, new_y=YPos.NEXT)
    pdf.image(plot_file_path, enums.Align.C, y=pdf.get_y(), w=180)
    pdf.ln(105)

if pump_df is not None and not pump_df.empty:
    pdf.add_page()
    fig, ax = plot_pump_profile(pump_df=pump_df)
    plot_file_path = plots_dir / "pump_data.png"
    fig.savefig(plot_file_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    pdf.set_font("Times", "B", 12)
    pdf.cell(0, 10, "Syringe Pump (FA Flow) Profile", new_x=XPos.RIGHT, new_y=YPos.NEXT)
    pdf.image(plot_file_path, enums.Align.C, y=pdf.get_y(), w=180)
    pdf.ln(105)


# Export compiled report PDF
output_pdf_path = selected_dir / 'reaction_data.pdf'
pdf.output(str(output_pdf_path))
print(f"Report generated successfully at: {output_pdf_path}")
