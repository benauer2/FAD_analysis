
from statsmodels.graphics import _regressionplots_doc
import yaml
from pathlib import Path
import shutil
import warnings
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import simpson
from scipy import stats
import os
import tkinter as tk
from tkinter import filedialog
from pathlib import Path
import pandas as pd
from worker_files.file_utils import load_selected_data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import pandas as pd
from fpdf import FPDF, XPos, YPos, enums
from datetime import date

class VelmaReactor:
    def __init__(
        self, 
        selected_dir=None, 
        filepaths=None, 
        current_df=None, 
        GC_df=None, 
        temp_df=None, 
        mfc_df=None, 
        pump_df=None, 
        eis_df=None, 
        schedule=None, 
        formic_acid=None, 
        trial_info=None, 
        gc_method_info=None, 
        shade_colors=None, 
        notes=None, 
        analysis=None, 
        gc_filepaths=None, 
        gc_method_length=None, 
        valve_box_t=None
    ):
        self.projmon_loc = Path('S:/projmon/velma_formic-acid-decomposition/Data')
        self.selected_dir = Path("S:/projmon/velma_formic-acid-decomposition/Data/[BA19] Device #3 IDT (50um) IG(Silica + EDMIM TFSI, 4;1, 4um)-Au (100 nm)-SiO2(20 nm) + Pt (100 nm)/Rxn - 4 - 11.02.2026")
        self.filepaths = load_selected_data("S:/projmon/velma_formic-acid-decomposition/Data/[BA19] Device #3 IDT (50um) IG(Silica + EDMIM TFSI, 4;1, 4um)-Au (100 nm)-SiO2(20 nm) + Pt (100 nm)/Rxn - 4 - 11.02.2026")
        self.formic_acid = {
            'density':1.220, #g/cm3
            'mol w':46.025 #g/mol
        }
        self.rgas = 8.314 #J/molK
        self.schedule = pd.DataFrame()

        self.handle_yaml()
        self.load_schedule_from_csv()

    def sort_files(self):
        """Utilizes the self.filepaths to scrape all of the data in the raw_data folder. Loads each relevant file type into its own pandas dataframe."""
        for file in self.filepaths:
                filename_lower = file.name.lower()

                try:
                    if "fad" in filename_lower:
                        print(f"Processing 'GC' file: {file.name}")
                        header_line = 0
                        with open(file, 'r', errors='ignore') as f:
                            for idx, line in enumerate(f):
                                if "folder name" in line.lower() or "line" in line.lower():
                                    if line.count(",") >= 2:
                                        header_line = idx
                                        break 
                        self.GC_df = pd.read_csv(file, skiprows=header_line)
                    elif "eis" in filename_lower and "pre" not in filename_lower:
                        print(f"Processing 'EIS' file: {file.name}")    
                        self.eis_df = pd.read_csv(file)
                    elif "current_data" in filename_lower:
                        print(f"Processing 'Electronics' file: {file.name}")
                        self.current_df = pd.read_csv(file)
                    elif "temp_data" in filename_lower:
                        print(f"Processing 'Temperature' file: {file.name}")
                        self.temp_df = pd.read_csv(file)
                    elif "mfc_data" in filename_lower:
                        print(f"Processing 'MFC' file: {file.name}")
                        self.mfc_df = pd.read_csv(file)
                    elif "pump_data" in filename_lower or "pump" in filename_lower:
                        print(f"Processing 'Pump' file: {file.name}")
                        self.pump_df = pd.read_csv(file)
                except Exception as e:
                    print(f"Error reading root CSV {file.name}: {e}")


        for subfolder in self.filepaths:
            if not subfolder.is_dir() or subfolder.name.startswith('.'):
                continue
            
            folder_name = subfolder.name
            csv_files = list(subfolder.glob("*.csv"))
            folder_dfs = []

            print(f"Processing subfolder: [{folder_name}]")

            for file_path in csv_files:
                filename = file_path.name.lower()

                if "quiettime" in filename:
                    continue

                try:
                    df = pd.read_csv(file_path)
                    if not df.empty:
                        folder_dfs.append(df)
                except Exception as e:
                    print(f"Error reading {file_path.name}: {e}")
                    """
            if folder_dfs:
                merged_df = pd.concat(folder_dfs, ignore_index=True)
                self.pre_reaction_df[folder_name] = merged_df
        """
    
    
    """
    ################################
    Importing schedule, yaml file
    ################################
    """
    def parse_avg_window(self, val):
        """Converts a string like '6, 16' or '[6, 16]' into a list of floats [6.0, 16.0]."""
        if pd.isna(val) or val is None:
            return [0.0, 0.0]
        val_str = str(val).strip("[]() ")
        parts = val_str.split(",")
        return [float(p.strip()) for p in parts]
    
    def parse_voltage(self, val):
        """Safely converts voltage inputs to float or standard string 'OCP'."""
        if pd.isna(val) or val == "":
            return 0.0
        val_str = str(val).strip()
        if val_str.upper() == "OCP":
            return "OCP"
        try:
            return float(val_str)
        except ValueError:
            return "OCP"
    

    def load_schedule_from_csv(self):
        """Parses sequence CSVs (sequence_default, arrhenius_sequence, etc.) into a standardized schedule dictionary format."""
        sched = []
        
        csv_path = self.selected_dir / "reaction_sequence.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Could not find sequence/schedule CSV in {self.selected_dir}")
        df = pd.read_csv(csv_path)

        current_time = 0
        
        for idx, (_, row) in enumerate(df.iterrows(), start=1):
            step_run = row.get("Run")
            rxr_t = float(row.get("Rxr_T_C"))
            n2_flow_sccm = float(row.get("N2_flow_sccm"))
            h2_flow_sccm = float(row.get("H2_flow_sccm"))
            co_flow_sccm = float(row.get("CO_flow_sccm"))
            fa_flow_ul = float(row.get("FA_flow_ul_min"))

            fa_flow_sccm = fa_flow_ul * 1E3 / 1E6 * self.formic_acid["density"] / self.formic_acid["mol w"] * 22.414 * 1000
            tot_flow_sccm = n2_flow_sccm + h2_flow_sccm + co_flow_sccm + fa_flow_sccm
            fa_vap_p = (fa_flow_sccm / tot_flow_sccm) * 100

            we_volt = self.parse_voltage(row.get("WE_Voltage"))
            avg_window = self.parse_avg_window(row.get("Avg_Window"))
            dur_hrs = float(row.get("Duration_hrs"))

            start_time = current_time
            end_time = current_time + float(dur_hrs)
            time_window = [start_time, end_time]
            current_time = end_time

            step = {
                "run": step_run,
                "rxr_t_c": rxr_t,
                "n2_flow": n2_flow_sccm,
                "h2_flow": h2_flow_sccm,
                "co_flow": co_flow_sccm,
                "fa_flow_ul": fa_flow_ul,
                "fa_flow_sccm": fa_flow_sccm,
                "tot_flow_sccm": tot_flow_sccm,
                "we_volt": we_volt,
                "avg_window": avg_window,
                "dur_hrs": dur_hrs,
                "time_window": time_window
            }

            sched.append(step)
        self.schedule = pd.DataFrame(sched)
    
        return

    def handle_yaml(self):
        """Loads metadata & shade colors from YAML and the reaction sequence from CSV."""        
        yaml_files = list(self.selected_dir.glob("*.yaml"))
        if yaml_files:
            yaml_path = yaml_files[0]
            print(f"[Config] Loading metadata parameters from: {yaml_path.name}")
            with open(yaml_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
                
            self.trial_info = cfg.get("trial_info", {})
            self.gc_info = cfg.get("gc_method_info", {})
            self.notes = cfg.get("notes", "")
            self.analysis = cfg.get("analysis", "")
            self.gc_filepaths = cfg.get("file_paths", {})

            self.gc_method_length = self.trial_info.get("GC method length")
            self.valve_box_t = self.gc_info.get("Valve box T (deg C)")
            
            # EXTRACT & MAP SHADE_COLORS
            raw_shades = cfg.get("shade_colors", {})
            self.shade_colors = {}
            for k, v in raw_shades.items():
                # Support numeric values (floats/ints) and string lookups (e.g. "OCP")
                try:
                    self.shade_colors[float(k)] = v
                except (ValueError, TypeError):
                    self.shade_colors[str(k)] = v
        



    """
    ##########
    Calculations
    #############
    """

    def calc_rxn_rates(self):
        self.GC_df["Time (min)"] = np.arange(len(self.GC_df)) * self.gc_method_length
        self.GC_df["Time (hr)"] = self.GC_df["Time (min)"] / 60

        C_cal = self.gc_info.get("C calibration factor (FID, umol/(pA s))")
        H_cal = self.gc_info.get("H2 calibration factor (TCD, umol/(uV s))")

        self.GC_df["Step"] = self.GC_df["Time (hr)"].apply(self.get_step_at_time)

        denom_p = 14.696 + float(0)
        if denom_p == 0:
            denom_p = 14.696

        total_flow_map = dict(zip(self.schedule['run'], self.schedule['tot_flow_sccm']))
        self.GC_df['tot_flow_sccm'] = self.GC_df['Step'].map(total_flow_map)

        volt_map = dict(zip(self.schedule['run'], self.schedule['we_volt']))
        self.GC_df['Bias (V)'] = self.GC_df['Step'].map(volt_map)

        temp_corr = (273.15 + float(self.valve_box_t)) / 300.0
        press_corr = 14.696 / denom_p
        
        self.GC_df['norm_vol_factor'] = (self.GC_df['tot_flow_sccm'] * temp_corr * press_corr) * 1000

        for species in ['CO2', 'FA', 'H2', 'CO']:
            if species in self.GC_df.columns:
                self.GC_df[species] = pd.to_numeric(self.GC_df[species], errors='coerce').fillna(0.0)
            else:
                self.GC_df[species] = 0.0
            
        self.GC_df['CO2 (nmol/min)'] = (C_cal*self.GC_df['CO2']) * self.GC_df['norm_vol_factor']
        self.GC_df['H2 (nmol/min)'] = (H_cal*self.GC_df['H2']) * self.GC_df['norm_vol_factor']
        self.GC_df['CO (nmol/min)'] =(C_cal*self.GC_df['CO']) * self.GC_df['norm_vol_factor']

        fa_denom = (denom_p / 14.696 * 101325 / 8.314 / (273.15 + float(self.valve_box_t)))
        self.GC_df['FA mol frac'] = 100 * (C_cal*self.GC_df['FA'])/((1/press_corr)*101325/self.rgas/(273.15+self.valve_box_t))


        return

    def get_step_at_time(self, time_hrs):
        #valid_schedule = self.schedule[self.schedule["time_window"].notna()]
        current_step = 0
        for entry in self.schedule["time_window"]:
            start_win = entry[0]
            end_win = entry[1]
            if start_win <= float(time_hrs) < end_win:
                return current_step + 1
            else:
                current_step +=1
            
        return current_step

    def current_stats(self):
        #std_cols = ['Time (hr)', 'Current (uA)', 'Current (A)', 'Working Electrode (V)', 'current/2 (nmol/min)', 'Bias (V)', 'Step_Index']
        self.current_df["Step"] = self.current_df["Overall_Elapsed_Time_hr"].apply(self.get_step_at_time)
        self.current_df["current/2 (nmol/min)"] = (abs(self.current_df['Current (uA)']) * 1e-6 / 96485 * 60 / 2 * 1e9)



    """
    Plotting
    """

    def plot_gas_rates(self, cutoff_hr=2.0, x_interval=8.0, figsize=(11,5)):
        fig, ax = plt.subplots(figsize=figsize)
        #ax = fig.add_subplot(1,1,1)

        has_rates = self.GC_df is not None and not self.GC_df.empty and "Time (hr)" in self.GC_df.columns
        has_elec = self.current_df is not None and not self.current_df.empty and "Overall_Elapsed_Time_hr" in self.current_df.columns

        if has_rates:
            for col, marker, color, label in [("CO2 (nmol/min)", 's', 'orange', r'CO$_2$'),
                                          ("CO (nmol/min)", '^', 'blue', 'CO'),
                                          ("H2 (nmol/min)", 'D', 'red', r'H$_2$')]:
                if col in self.GC_df.columns:
                    ax.plot(self.GC_df["Time (hr)"], self.GC_df[col], linestyle='None',
                            marker=marker, markersize=6, label=label, markerfacecolor=color,
                            markeredgewidth=1, markeredgecolor='black', zorder=3)

        if has_elec and "current/2 (nmol/min)" in self.current_df.columns:
                ax.plot(self.current_df["Overall_Elapsed_Time_hr"], self.current_df["current/2 (nmol/min)"],
                        linestyle='-', linewidth=2, label='Rate of Electron Transfer (|e|/2)', color='orange', zorder=4)
            
        all_times = []
        if has_rates: all_times.extend(self.GC_df["Time (hr)"].dropna().values)
        if has_elec: all_times.extend(self.current_df["Overall_Elapsed_Time_hr"].dropna().values)
        max_time = max(all_times) if all_times else 1.0
        ax.set_xlim(0, max_time)

        y_vals = []
        if has_rates:
            for c in ["CO2 (nmol/min)", "CO (nmol/min)", "H2 (nmol/min)"]:
                if c in self.GC_df.columns:
                    if c == "H2 (nmol/min)":
                        if "Time (hr)" in self.GC_df.columns:
                            h2_filtered = self.GC_df[self.GC_df["Time (hr)"] >= cutoff_hr]["H2 (nmol/min)"].dropna()
                            y_vals.extend(h2_filtered.values)
                        else:
                            y_vals.extend(self.GC_df[c].dropna().values)
                    else:
                        y_vals.extend(self.GC_df[c].dropna().values)
        y_min = min(y_vals) if y_vals else 0.0
        y_max = max(y_vals) if y_vals else 1.0
        y_span = y_max - y_min
        padding = y_span * 0.20 if y_span > 0 else 0.1
        ax.set_ylim(0, y_max + padding)

        plt.xlabel("Time (hr)\n\n", labelpad=7, fontsize=18)
        plt.ylabel("Extensive Rate (nmol/min)", labelpad=7, fontsize=20)

        ax.xaxis.set_major_locator(ticker.MultipleLocator(x_interval))
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
        ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=8, prune=None))
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

        ax.tick_params(axis='both', which='major', direction='inout', length=8, width=1.5, labelsize=16)
        ax.tick_params(axis='both', which='minor', direction='in', length=4, width=1.0)

        for spine in ['top', 'bottom', 'left', 'right']:
            ax.spines[spine].set_linewidth(1.8)
            ax.spines[spine].set_zorder(5) 

        has_bias_data = (self.GC_df is not None) and (self.schedule is not None) and ("Time (hr)" in self.GC_df.columns)
        colors = self.shade_colors if self.shade_colors is not None else {}



        if has_bias_data:
            bias_values = self.GC_df["Bias (V)"]
            time_values = self.GC_df['Time (hr)'].tolist()

            if len(time_values) > 0:
                dt = time_values[1] - time_values[0] if len(time_values) > 1 else 0
                start = time_values[0]
                current_bias = bias_values[0]

                for i in range(1, len(bias_values)):
                    if bias_values[i] != current_bias:
                        ax.axvspan(start, time_values[i], color=self.shade_colors.get(current_bias, 'white'), alpha=1.0, zorder=1, lw=0)
                        start = time_values[i]
                        current_bias = bias_values[i]
        # Draw final block stretch
                ax.axvspan(start, time_values[-1] + dt, color=self.shade_colors.get(current_bias, 'white'), alpha=1.0, zorder=1, lw=0)

        # 5. Draw double legends
        species_handles, species_labels = ax.get_legend_handles_labels()

        if species_handles:
            species_legend = ax.legend(handles=species_handles, labels=species_labels,bbox_to_anchor=(0.5, -0.15), 
                            loc='upper center', columnspacing=0.7, fontsize=13, ncol=len(species_labels), frameon=False)
            ax.add_artist(species_legend)

        if has_bias_data:
            unique_biases = sorted(set(bias_values), key=lambda x: (isinstance(x, str), x)) 

            # Use clean patches with thin borders for the background key
            bias_patches = [Patch(
                                facecolor=self.shade_colors[b], 
                                edgecolor='black', 
                                label=f'{b} V' if isinstance(b, (int, float)) else f'{b}'
                                ) 
                            for b in unique_biases
            ]

            # Bias legend sitting above the top axis line
            bias_legend = ax.legend(handles=bias_patches, bbox_to_anchor=(0.5, 1.00), loc="lower center", 
                                    columnspacing=1.0, fontsize=12, ncol=len(unique_biases), frameon=False)


        #plt.show()
        return fig, ax

    def plot_formic_acid(self, figsize=(11,5), x_interval=8):
        if self.GC_df is None or self.GC_df.empty or "Time (hr)" not in self.GC_df.columns or "FA mol frac" not in self.GC_df.columns:
            fig, ax = plt.subplots(figsize=figsize)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        ax.plot(
            self.GC_df["Time (hr)"], 
            self.GC_df["FA mol frac"], 
            linestyle='None', 
            marker='o', 
            markersize=6, 
            label='Formic Acid (Reactor)', 
            markerfacecolor='purple', 
            markeredgewidth=1, 
            markeredgecolor='black'
        )

        max_rate = self.GC_df["FA mol frac"].max()
        max_time = self.GC_df["Time (hr)"].max()

        y_upper = max_rate * 1.2 if np.isfinite(max_rate) and max_rate > 0 else 1.0
        x_upper = max_time if np.isfinite(max_time) and max_time > 0 else 1.0

        ax.set_ylim(0, y_upper)
        ax.set_xlim(0, x_upper)

        plt.xlabel("Time (hr)\n\n", labelpad=7, fontsize=18)
        plt.ylabel("FA mole fraction (%)", labelpad=7, fontsize=20)

        ax.xaxis.set_major_locator(ticker.MultipleLocator(x_interval))
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
        ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=8, prune=None))
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

        ax.tick_params(axis='both', which='major', direction='inout', length=8, width=1.5, labelsize=16)
        ax.tick_params(axis='both', which='minor', direction='in', length=4, width=1.0)

        for spine in ['top', 'bottom', 'left', 'right']:
            ax.spines[spine].set_linewidth(1.8)
            ax.spines[spine].set_zorder(5) 

        species_handles, species_labels = ax.get_legend_handles_labels()
        if species_handles:
            species_legend = ax.legend(handles=species_handles, labels=species_labels,bbox_to_anchor=(0.5, -0.15), 
                                    loc='upper center', columnspacing=0.7, fontsize=13, ncol=len(species_labels), frameon=False)
            ax.add_artist(species_legend)

        return fig, ax

class MyPDF(FPDF):
    
    def header(self):
        self.set_font('Times', 'B', 12)
        self.cell(0, 10, 'Formic Acid Decomposition Trial Report', align='C')
        self.ln(10)
        
    def footer(self):
        tod = date.today().strftime("%m. %d. %Y.")
        self.set_y(-15)
        self.set_font('Times', 'I', 12)
        self.cell(0, 10, f'Page {self.page_no()} | Auer | {tod}', align='R', border=0)



class ReactionReport:
    """Runs the reaction report.
    """
    def __init__(self, VelmaReactor):
        self.VelmaReactor = VelmaReactor
        self.VelmaReactor.sort_files()
        self.VelmaReactor.current_stats()
        self.VelmaReactor.calc_rxn_rates()
        self.VelmaReactor.plot_gas_rates()
        self.generate_pdf()
        

    def generate_pdf(self):

        pdf = MyPDF()

        pdf.add_page()

        """
        pdf.set_font('Times', 'B', 12)
        pdf.cell(pdf.get_string_width("Device: ") + 2, 10, "Device: ", new_x=XPos.RIGHT, new_y=YPos.LAST)
        pdf.set_font('Times', '', 12)
        #pdf.cell(0, 10, f"{TRIAL_INFO.get('Device', 'N/A')}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        pdf.set_font('Times', 'B', 12)
        pdf.cell(pdf.get_string_width("Date: ") + 2, 10, "Date: ", new_x=XPos.RIGHT, new_y=YPos.LAST)
        pdf.set_font('Times', '', 12)
        pdf.cell(0, 10, f"{TRIAL_INFO.get('Date', 'N/A')}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        pdf.set_font('Times', 'B', 12)
        pdf.cell(pdf.get_string_width("GC Method: ") + 2, 10, "GC Method: ", new_x=XPos.RIGHT, new_y=YPos.LAST)
        pdf.set_font('Times', '', 12)
        pdf.cell(0, 10, f"{TRIAL_INFO.get('GC method', TRIAL_INFO.get('GC Method', TRIAL_INFO.get('gc_method', 'N/A')))}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

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
        """
        plots_dir = Path("C:/Users/cemshauerlocal/Documents/GitHub/FAD_analysis/testing_results")
        fig, ax = self.VelmaReactor.plot_gas_rates()
        plot_file_path = plots_dir / "rate_data.png"
        fig.savefig(plot_file_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        pdf.set_font("Times", "B", 12)
        pdf.cell(0, 10, "Reaction Kinetics Plot", new_x=XPos.RIGHT, new_y=YPos.NEXT)
        pdf.image(plot_file_path, enums.Align.C, y=pdf.get_y(), w=180)
        pdf.ln(105)

        fig, ax = self.VelmaReactor.plot_formic_acid()
        plot_file_path = plots_dir / "fa_data.png"
        fig.savefig(plot_file_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        pdf.set_font("Times", "B", 12)
        pdf.cell(0, 10, "Formic Acid Plot", new_x=XPos.RIGHT, new_y=YPos.NEXT)
        pdf.image(plot_file_path, enums.Align.C, y=pdf.get_y(), w=180)
        pdf.ln(105)

        """
        line_height = 6
        pdf.set_font('Times', 'B', 12)
        pdf.write(line_height, "Reaction Notes: ")
        pdf.set_font('Times', '', 12)
        pdf.write(line_height, f"{notes}\n\n")

        pdf.set_font('Times', 'B', 12)
        pdf.write(line_height, "Analysis: ")
        pdf.set_font('Times', '', 12)
        pdf.write(line_height, f"{analysis}\n\n")
        """

        # ---------------------------------------------------------
        # PAGE 2: FORMIC ACID & CURRENT DATA
        # ---------------------------------------------------------
        """
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

        """
        output_pdf_path = plots_dir / 'reaction_data.pdf'
        pdf.output(str(output_pdf_path))
        print(f"Report generated successfully at: {output_pdf_path}")
            

            

                



