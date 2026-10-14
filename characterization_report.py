"""
PDF Characterization Report Generator for Electrochemical Data (EIS & CV)
Processes single/multiple EIS and CV data files and subfolders in a directory to produce
a comprehensive characterization PDF report with Nyquist/Bode plots, multi-sweep CV slope regressions,
and publication-ready voltammogram overlays.
Author: Ben Auer
"""
import sys
import os
import re
from pathlib import Path
from datetime import date
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy import stats
import tkinter as tk
from tkinter import filedialog

from worker_files.file_utils import safe_read_csv, projmon_loc, get_dialog_root
from worker_files.electronics_worker import (
    calculate_loop_area,
    calc_cap_from_cv_grouped,
    _fit_and_plot_eis
)
from worker_files.plot_worker import plot_10CV
from worker_files.pdf_classes import CharacterizationReport, safe_str

# Global font configuration
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Calibri']


def scan_characterization_folder(target_dir=None):
    """
    Scans a folder and its subfolders for EIS and CV characterization data.
    Returns:
        cv_data_dict: dict {key: DataFrame} for all CV runs
        eis_data_dict: dict {key: DataFrame} for all EIS runs
        other_data_dict: dict {key: DataFrame} for other electronics files
        target_path: Path
    """
    if target_dir is None:
        if len(sys.argv) > 1 and Path(sys.argv[1]).exists():
            target_path = Path(sys.argv[1])
        else:
            root = get_dialog_root()
            selected = filedialog.askdirectory(
                title="Select Characterization / Reaction Folder",
                initialdir=projmon_loc if projmon_loc.exists() else Path.cwd()
            )
            root.destroy()
            if not selected:
                print("No folder selected.")
                return None, None, None, None
            target_path = Path(selected)
    else:
        target_path = Path(target_dir)

    # If raw_data folder exists, scan inside it as well as root
    search_dirs = [target_path]
    if (target_path / "raw_data").exists():
        search_dirs.append(target_path / "raw_data")

    cv_data_dict = {}
    eis_data_dict = {}
    other_data_dict = {}

    print(f"\nScanning folder: {target_path.resolve()}")

    # 1. Process Subfolders (Squidstat individual characterization folders)
    for base_d in search_dirs:
        for subfolder in sorted(base_d.iterdir()):
            if not subfolder.is_dir() or subfolder.name.startswith('.'):
                continue
            if subfolder.name.lower() in ["plots", "data", "raw_data"]:
                continue

            folder_name = subfolder.name
            csv_files = sorted(subfolder.glob("*.csv"))
            folder_dfs = []

            for file_path in csv_files:
                if "quiettime" in file_path.name.lower():
                    continue
                try:
                    df = safe_read_csv(file_path)
                    if df is not None and not df.empty:
                        folder_dfs.append(df)
                except Exception as e:
                    print(f"Error reading {file_path.name}: {e}")

            if folder_dfs:
                merged_df = pd.concat(folder_dfs, ignore_index=True)
                folder_lower = folder_name.lower()
                if "eis" in folder_lower:
                    eis_data_dict[folder_name] = merged_df
                    print(f"  [EIS Subfolder] Found: {folder_name} ({len(merged_df)} rows)")
                elif "cv" in folder_lower or "voltamm" in folder_lower:
                    cv_data_dict[folder_name] = merged_df
                    print(f"  [CV Subfolder]  Found: {folder_name} ({len(merged_df)} rows)")
                else:
                    other_data_dict[folder_name] = merged_df
                    print(f"  [Other Subfolder] Found: {folder_name} ({len(merged_df)} rows)")

    # 2. Process loose CSV files directly in target_path or raw_data
    for base_d in search_dirs:
        for file_path in sorted(base_d.glob("*.csv")):
            if "quiettime" in file_path.name.lower():
                continue
            file_key = file_path.stem
            if file_key in cv_data_dict or file_key in eis_data_dict or file_key in other_data_dict:
                continue

            file_lower = file_path.name.lower()
            try:
                df = safe_read_csv(file_path)
                if df is None or df.empty:
                    continue
                if "eis" in file_lower:
                    eis_data_dict[file_key] = df
                    print(f"  [EIS File] Found: {file_path.name}")
                elif "cv" in file_lower:
                    cv_data_dict[file_key] = df
                    print(f"  [CV File]  Found: {file_path.name}")
                elif "raw_data" in file_lower or "electronics" in file_lower:
                    other_data_dict[file_key] = df
                    print(f"  [Electronics File] Found: {file_path.name}")
            except Exception as e:
                print(f"Error reading root CSV {file_path.name}: {e}")

    return cv_data_dict, eis_data_dict, other_data_dict, target_path


def process_eis_dataset(df, title_name):
    """
    Cleans EIS data, converts |Z| and Phase to complex impedance if needed,
    and performs equivalent circuit fit.
    """
    clean_df = df.copy()
    clean_df.columns = clean_df.columns.str.strip()

    freq_col = None
    for cand in ["Frequency (Hz)", "Frequency", "Freq (Hz)", "freq", "Frequency_Hz"]:
        if cand in clean_df.columns:
            freq_col = cand
            break
    if freq_col is None and len(clean_df.columns) > 0:
        freq_col = clean_df.columns[0]

    # Ensure complex impedance columns exist
    if "Z' (Ohms)" not in clean_df.columns or '-Z" (Ohms)' not in clean_df.columns:
        if "|Z| (Ohms)" in clean_df.columns and "Phase (deg)" in clean_df.columns:
            clean_df["Z' (Ohms)"] = clean_df["|Z| (Ohms)"] * np.cos(np.radians(clean_df["Phase (deg)"]))
            clean_df['-Z" (Ohms)'] = -clean_df["|Z| (Ohms)"] * np.sin(np.radians(clean_df["Phase (deg)"]))
        elif "Z_real" in clean_df.columns and "Z_imag" in clean_df.columns:
            clean_df["Z' (Ohms)"] = clean_df["Z_real"]
            clean_df['-Z" (Ohms)'] = -clean_df["Z_imag"]

    clean_df = clean_df.dropna(subset=[freq_col, "Z' (Ohms)", '-Z" (Ohms)'])
    f = pd.to_numeric(clean_df[freq_col], errors='coerce').values
    z_real = pd.to_numeric(clean_df["Z' (Ohms)"], errors='coerce').values
    z_imag = pd.to_numeric(clean_df['-Z" (Ohms)'], errors='coerce').values

    valid_mask = (f > 0) & np.isfinite(f) & np.isfinite(z_real) & np.isfinite(z_imag)
    f = f[valid_mask]
    Z = z_real[valid_mask] - 1j * z_imag[valid_mask]

    if len(f) < 4:
        print(f"Warning: Insufficient EIS points for {title_name}")
        return None, (None, None), {}, pd.DataFrame()

    circuit, fig_axes, params = _fit_and_plot_eis(f, Z, title_name)

    dc_we_col = None
    for cand in ["DC Working Electrode (V)", "Working Electrode (V)", "Potential (V)", "DC_WE_V"]:
        if cand in clean_df.columns:
            dc_we_col = cand
            break
    dc_we = clean_df[dc_we_col].iloc[0] if dc_we_col else np.nan

    r0_val = getattr(circuit, 'parameters_', [np.nan] * 4)[0]
    r1_val = getattr(circuit, 'parameters_', [np.nan] * 4)[1]
    cpe1_0_val = getattr(circuit, 'parameters_', [np.nan] * 4)[2]
    cpe1_n_val = getattr(circuit, 'parameters_', [np.nan] * 4)[3] if len(getattr(circuit, 'parameters_', [])) > 3 else np.nan

    summary_df = pd.DataFrame([{
        "Characterization": title_name,
        "R0 (Ohms)": float(r0_val),
        "R1 (Ohms)": float(r1_val),
        "CPE1 (uF)": float(cpe1_0_val) * 1e6 if np.isfinite(cpe1_0_val) else np.nan,
        "CPE1_n": float(cpe1_n_val) if np.isfinite(cpe1_n_val) else np.nan,
        "DC Bias (V)": f"{dc_we:.2f}" if pd.notna(dc_we) else "N/A"
    }])

    return circuit, fig_axes, params, summary_df


def process_multi_sweep_regression(df, title_name, eval_potential=0.0):
    """
    Analyzes a multi-sweep CV dataset (e.g. 10 CV sweeps from 0.1 to 1.0 V/s):
    1. Extracts capacitive currents at eval_potential (i_pos - i_neg) / 2.
    2. Performs linear regression: i_cap (uA) = C (uF) * v (V/s) + i_leak (uA).
    3. Produces a 2-panel figure: Multi-sweep overlay + Capacitance regression slope.
    """
    step_col = "Step number" if "Step number" in df.columns else None
    unique_steps = sorted(df[step_col].unique()) if step_col else [1]

    if len(unique_steps) < 2:
        return None, None, {}

    v_col = ("Working Electrode (V)" if "Working Electrode (V)" in df.columns 
             else ("Potential (V)" if "Potential (V)" in df.columns else df.columns[4 if len(df.columns) > 4 else 2]))
    c_col = ("Current (A)" if "Current (A)" in df.columns 
             else ("Current (uA)" if "Current (uA)" in df.columns else df.columns[3]))

    sweep_rates = []
    currents_cap_uA = []
    step_data_dict = {}

    # Check potential range to ensure eval_potential is valid, or fallback to center potential
    all_v = pd.to_numeric(df[v_col], errors='coerce').dropna()
    if not all_v.empty:
        v_min, v_max = all_v.min(), all_v.max()
        if not (v_min < eval_potential < v_max):
            eval_potential = round((v_min + v_max) / 2.0, 2)

    for step in unique_steps:
        step_df = df[df[step_col] == step].sort_values("Elapsed Time (s)") if ("Elapsed Time (s)" in df.columns and step_col) else (df[df[step_col] == step] if step_col else df)
        if len(step_df) < 4:
            continue

        raw_v = pd.to_numeric(step_df[v_col], errors="coerce").values
        raw_c = pd.to_numeric(step_df[c_col], errors="coerce").values
        valid_idx = np.isfinite(raw_v) & np.isfinite(raw_c)
        v_vals, c_vals = raw_v[valid_idx], raw_c[valid_idx]

        if len(v_vals) < 4:
            continue

        if "uA" not in str(c_col) and "\u03bcA" not in str(c_col):
            c_vals = c_vals * 1e6

        rate = step * 0.1 if len(unique_steps) > 1 else 1.0
        step_data_dict[step] = (v_vals, c_vals, rate)

        peak_idx = np.argmax(v_vals)
        v_fwd, c_fwd = v_vals[:peak_idx], c_vals[:peak_idx]
        v_rev, c_rev = v_vals[peak_idx:], c_vals[peak_idx:]

        if len(v_fwd) > 1 and len(v_rev) > 1:
            i_pos = np.interp(eval_potential, v_fwd, c_fwd)
            i_neg = np.interp(eval_potential, v_rev[::-1], c_rev[::-1])
            if pd.notna(i_pos) and pd.notna(i_neg):
                sweep_rates.append(rate)
                currents_cap_uA.append(abs(i_pos - i_neg) / 2.0)

    if len(sweep_rates) < 3:
        return None, None, {}

    sweep_rates = np.array(sweep_rates)
    currents_cap_uA = np.array(currents_cap_uA)

    slope, intercept, r_value, p_value, std_err = stats.linregress(sweep_rates, currents_cap_uA)
    r_squared = r_value ** 2

    # Two-panel figure
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax_cv, ax_slope = axes[0], axes[1]

    # Color palette
    red_to_blue = [
        "#b30000", "#e34a33", "#fc8d59", "#fdbb84", "#b89bc7",
        "#8c96c6", "#8c6bb1", "#41b6c4", "#1d91c0", "#081d58"
    ]

    for idx, (step, (v_vals, c_vals, rate)) in enumerate(step_data_dict.items()):
        color = red_to_blue[idx % len(red_to_blue)]
        ax_cv.plot(v_vals, c_vals, label=f"{rate:.1f} V/s", color=color, linewidth=1.2)

    ax_cv.axvline(eval_potential, color='black', linestyle='--', alpha=0.6, label=f"Eval V ({eval_potential} V)")
    ax_cv.set_xlabel("Working Electrode (V)", fontsize=11)
    ax_cv.set_ylabel(r"Current ($\mu$A)", fontsize=11)
    ax_cv.set_title(f"{title_name}\nMulti-Sweep Overlays", fontsize=12, fontweight='bold')
    ax_cv.legend(frameon=False, fontsize=8, ncol=min(len(step_data_dict), 4), loc='upper center', bbox_to_anchor=(0.5, -0.15))

    # Slope regression plot
    x_fit = np.linspace(min(sweep_rates) * 0.8, max(sweep_rates) * 1.15, 30)
    y_fit = slope * x_fit + intercept

    ax_slope.plot(sweep_rates, currents_cap_uA, 'o', color="#A10601", markersize=6, label=r"Measured $\Delta i_{cap}/2$")
    ax_slope.plot(x_fit, y_fit, '--', color="black", linewidth=1.8, 
                 label=f"Linear Fit\n$C_{{slope}} = {abs(slope):.2f}\ \mu$F\n$i_{{leak}} = {intercept:.2f}\ \mu$A\n$R^2 = {r_squared:.4f}$")

    ax_slope.set_xlabel("Sweep Rate (V/s)", fontsize=11)
    ax_slope.set_ylabel(r"Capacitive Current ($\mu$A)", fontsize=11)
    ax_slope.set_title(f"Capacitance Slope Fit @ {eval_potential} V", fontsize=12, fontweight='bold')
    ax_slope.legend(frameon=False, fontsize=9.5)

    for ax in [ax_cv, ax_slope]:
        for spine in ["top", "bottom", "left", "right"]:
            ax.spines[spine].set_linewidth(1.2)
        ax.tick_params(axis='both', which='major', direction='out', length=5, width=1.2, labelsize=10)
        ax.tick_params(axis='both', which='minor', direction='out', length=3, width=0.8)

    fig.subplots_adjust(left=0.08, right=0.95, top=0.88, bottom=0.22, wspace=0.28)

    metrics = {
        "Dataset": title_name,
        "Eval Potential (V)": eval_potential,
        "Capacitance (uF)": abs(slope),
        "Leakage Current (uA)": intercept,
        "R_squared": r_squared,
        "Std Err (uF)": std_err
    }

    return fig, axes, metrics


def generate_characterization_report(target_dir=None):
    """
    Main execution pipeline: loads characterizations, runs analyses, and generates PDF.
    """
    cv_dict, eis_dict, other_dict, target_path = scan_characterization_folder(target_dir)
    if target_path is None:
        return

    if not cv_dict and not eis_dict and not other_dict:
        print(f"No electrochemical data found in {target_path}")
        return

    folder_title = target_path.name
    report = CharacterizationReport(target_dir=target_path, folder_title=folder_title)

    # ---------------------------------------------------------
    # 1. PROCESS EIS DATA
    # ---------------------------------------------------------
    eis_summaries = []
    eis_plot_paths = []

    for name, df in eis_dict.items():
        print(f"Processing EIS fit for [{name}]...")
        circuit, fig_axes, params, sum_df = process_eis_dataset(df, name)
        if sum_df is not None and not sum_df.empty:
            eis_summaries.append(sum_df)
            fig, axes = fig_axes
            if fig is not None:
                img_path = report.plots_dir / f"EIS_{re.sub(r'[^a-zA-Z0-9_-]', '_', name)}.png"
                fig.savefig(img_path, dpi=300, bbox_inches="tight")
                plt.close(fig)
                eis_plot_paths.append((name, img_path, sum_df.iloc[0]))

    all_eis_summary_df = pd.concat(eis_summaries, ignore_index=True) if eis_summaries else pd.DataFrame()
    if not all_eis_summary_df.empty:
        all_eis_summary_df.to_csv(report.data_dir / "eis_characterization_summary.csv", index=False)

    # ---------------------------------------------------------
    # 2. PROCESS CV DATA: MULTI-SWEEP SLOPES & LOOP CAPACITANCE
    # ---------------------------------------------------------
    multi_sweep_metrics = []
    multi_sweep_plot_paths = []

    for name, df in cv_dict.items():
        step_col = "Step number" if "Step number" in df.columns else None
        n_steps = df[step_col].nunique() if step_col else 1
        if n_steps >= 2 or "10 cv" in name.lower() or "10_cv" in name.lower():
            print(f"Processing Multi-Sweep Regression for [{name}]...")
            fig, axes, metrics = process_multi_sweep_regression(df, name)
            if fig is not None:
                img_path = report.plots_dir / f"CV_Slope_{re.sub(r'[^a-zA-Z0-9_-]', '_', name)}.png"
                fig.savefig(img_path, dpi=300, bbox_inches="tight")
                plt.close(fig)
                multi_sweep_plot_paths.append((name, img_path, metrics))
                multi_sweep_metrics.append(metrics)

    multi_sweep_summary_df = pd.DataFrame(multi_sweep_metrics) if multi_sweep_metrics else pd.DataFrame()
    if not multi_sweep_summary_df.empty:
        multi_sweep_summary_df.to_csv(report.data_dir / "cv_multi_sweep_slope_summary.csv", index=False)

    all_cv_summary_df = pd.DataFrame()
    if cv_dict:
        print("Calculating CV loop capacitance metrics...")
        all_cv_summary_df = calc_cap_from_cv_grouped(cv_dict)
        if not all_cv_summary_df.empty:
            all_cv_summary_df.to_csv(report.data_dir / "cv_capacitance_summary.csv", index=False)

    # ---------------------------------------------------------
    # 3. BUILD PDF REPORT
    # ---------------------------------------------------------
    # PAGE 1: OVERVIEW & EXECUTIVE SUMMARY
    report.add_page()
    report.add_title_block("Electrochemical Characterization Summary")

    kv_info = {
        "Folder / Experiment": folder_title,
        "Directory Path": str(target_path),
        "Characterizations Found": f"{len(eis_dict)} EIS datasets, {len(cv_dict)} CV datasets ({len(multi_sweep_metrics)} multi-sweep sets)"
    }
    report.add_key_value_pairs(kv_info)
    report.pdf.ln(4)

    # EIS Summary Table on Page 1
    if not all_eis_summary_df.empty:
        report.add_title_block(
            "Impedance Spectroscopy (EIS) Equivalent Circuit Fits",
            subtitle="Model: R0-p(R1, CPE1) -> Series Resistance (R0), Leakage/CT Resistance (R1), Admittance (CPE1)"
        )
        col_w = [70, 28, 28, 28, 20, 16]
        headers = ["Dataset / Condition", "R0 (Ohm)", "R1 (Ohm)", "CPE1 (uF)", "CPE1_n", "DC (V)"]
        report.add_table(all_eis_summary_df, col_widths=col_w, headers=headers)
        report.pdf.ln(3)

    # Multi-Sweep Linear Regression Summary Table
    if not multi_sweep_summary_df.empty:
        report.add_title_block(
            "Multi-Sweep CV Linear Regression Slopes (i = C*v + i_leak)",
            subtitle="Extracts true capacitance (C_slope) and leakage current (i_leak) across 0.1 to 1.0 V/s"
        )
        col_w_ms = [70, 26, 32, 32, 30]
        headers_ms = ["Dataset / Condition", "Eval V (V)", "C_slope (uF)", "i_leak (uA)", "R-squared"]
        report.add_table(multi_sweep_summary_df, col_widths=col_w_ms, headers=headers_ms)
        report.pdf.ln(3)

    # CV Loop Summary Table
    if not all_cv_summary_df.empty:
        if report.pdf.get_y() > 210:
            report.add_page()

        report.add_title_block(
            "Cyclic Voltammetry (CV) Loop Area Capacitance Table",
            subtitle="Calculated via Shoelace Loop Area: C = Area / (2 * v * Delta_V)"
        )
        col_w_cv = [65, 20, 26, 26, 28, 25]
        headers_cv = ["Dataset / Condition", "Step #", "Rate (V/s)", "Span (V)", "Cap (uF)", "Time (hr)"]
        report.add_table(all_cv_summary_df, col_widths=col_w_cv, headers=headers_cv)
        report.pdf.ln(3)

    # ---------------------------------------------------------
    # EIS DETAILED PLOTS (BODE & NYQUIST PAGES)
    # ---------------------------------------------------------
    for name, img_path, sum_row in eis_plot_paths:
        report.add_page()
        report.add_title_block(f"Impedance Spectroscopy (EIS) - {name}")
        report.add_plot_image(img_path, width=180, ln_after=78)

        metrics = {
            "Series Resistance (R0)": f"{sum_row['R0 (Ohms)']:.2f} Ohms",
            "Charge Transfer / Leakage Resistance (R1)": f"{sum_row['R1 (Ohms)']:.2f} Ohms",
            "Constant Phase Element Admittance (CPE1)": f"{sum_row['CPE1 (uF)']:.4f} uF (n = {sum_row.get('CPE1_n', np.nan):.2f})",
            "DC Bias Working Electrode": f"{sum_row['DC Bias (V)']} V"
        }
        report.add_metrics_card(
            title="Equivalent Circuit Fit Parameters",
            metrics_dict=metrics,
            model_name="R0-p(R1, CPE1)"
        )

    # ---------------------------------------------------------
    # MULTI-SWEEP CV LINEAR REGRESSION PAGES
    # ---------------------------------------------------------
    for name, img_path, metrics in multi_sweep_plot_paths:
        report.add_page()
        report.add_title_block(f"Multi-Sweep CV Scan Rate Diagnostics - {name}")
        report.add_plot_image(img_path, width=180, ln_after=76)

        slope_metrics = {
            "Slope Capacitance (C_slope)": f"{metrics['Capacitance (uF)']:.4f} uF",
            "Leakage Current Intercept (i_leak)": f"{metrics['Leakage Current (uA)']:.4f} uA",
            "Regression Coefficient (R^2)": f"{metrics['R_squared']:.4f}",
            "Evaluation Potential": f"{metrics['Eval Potential (V)']:.2f} V"
        }
        report.add_metrics_card(
            title="Scan Rate Linear Regression Fit",
            metrics_dict=slope_metrics,
            model_name="i_cap = C * v + i_leak"
        )

    # ---------------------------------------------------------
    # CV DETAILED OVERLAYS (2x2 GRID PANELS)
    # ---------------------------------------------------------
    cv_keys = list(cv_dict.keys())
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
                plot_10CV(cv_dict[key], title_name=key, ax=axes_flat[i], panel_letter=panel_letter)

            for j in range(n_plots, len(axes_flat)):
                axes_flat[j].axis('off')

            plt.subplots_adjust(hspace=0.42, wspace=0.32, top=0.92, bottom=0.08, left=0.10, right=0.95)
            cv_grid_path = report.plots_dir / f"CV_Characterizations_Grid_page_{page_idx}.png"
            fig.savefig(cv_grid_path, dpi=300, bbox_inches="tight")
            plt.close(fig)

            report.add_page()
            page_title = "Cyclic Voltammetry (CV) Panels" if len(cv_keys) <= max_per_page else f"Cyclic Voltammetry (CV) Panels (Page {page_idx})"
            report.add_title_block(page_title)
            report.add_plot_image(cv_grid_path, width=160, ln_after=0)

    # Export PDF
    output_pdf_path = report.save(f"{folder_title}_characterization_report.pdf")
    return output_pdf_path


if __name__ == "__main__":
    generate_characterization_report()
