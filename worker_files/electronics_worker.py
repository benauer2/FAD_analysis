from pathlib import Path
import shutil
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import simpson
from scipy import stats
from impedance.models.circuits import CustomCircuit

from config import get_step_at_time

def assign_voltage_windows(df, rxn_schedule):
    df['Voltage_Window'] = df['Time (hr)'].apply(
        lambda t: get_step_at_time(t, rxn_schedule).get('we_volt', None)
    )
    return df


#------------------------------------------------------------------
# CV ANALYSIS FUNCTIONS
#------------------------------------------------------------------
def get_cv(df_raw, sweep_rate_map=None):
    'GETS CV DATA FROM REACTION ELECTRONICS DF. DIVIDES INTO INDIVIDUAL CHUNKS AND ASSIGNS SWEEP RATES TO THE SCANS.'
    if df_raw is None or df_raw.empty:
        return df_raw, {}
    
    if sweep_rate_map is None:
        sweep_rate_map = {0: 0.5, 1: 1.0, 2: 1.5}

    df_cv = df_raw.copy()
    df_cv.columns = df_cv.columns.str.strip()

    if "Step name" not in df_cv.columns or "Cyclic Voltammetry" not in df_cv['Step name'].values:
        return None, {}
    
    df_cv['Time (hr)'] = df_cv['Elapsed Time (s)'] / 3600.0
    df_cv['Current (uA)'] = df_cv['Current (A)'] * 1e6
    df_cv['Step name'] = df_cv['Step name'].astype(str).str.strip()

    relevant_steps = df_cv[df_cv["Step name"].isin(["Cyclic Voltammetry", "Constant Potential"])].copy()
    is_cp = relevant_steps["Step name"] == "Constant Potential"
    relevant_steps["major_block_id"] = (is_cp != is_cp.shift()).cumsum()

    cv_only_df = relevant_steps[relevant_steps["Step name"] == "Cyclic Voltammetry"].copy()
    if cv_only_df.empty:
        return df_cv, {}

    cv_segments = {}
    segment_count = 1

    for _, chunk_group in cv_only_df.groupby("major_block_id"):
        chunk_group = chunk_group.copy()
        chunk_group["dt"] = chunk_group["Elapsed Time (s)"].diff()
        chunk_group["is_new_sweep"] = chunk_group["dt"] > 3.0
        chunk_group["sweep_in_chunk"] = (chunk_group["is_new_sweep"].fillna(False).cumsum())

        for sweep_idx, (_, sweep_group) in enumerate(chunk_group.groupby("sweep_in_chunk")):
            clean_sweep = sweep_group.drop(
                columns=["dt","is_new_sweep","sweep_in_chunk","major_block_id",], errors='ignore').copy()

            position = sweep_idx % len(sweep_rate_map)
            clean_sweep["sweep_rate"] = sweep_rate_map.get(position, 1.0)
            cv_segments[segment_count] = clean_sweep
            segment_count += 1

    return df_cv, cv_segments

def get_cv_rachel(df_raw):
    'ANALYSIS FOR RACHEL'
    if df_raw is None or df_raw.empty:
        return df_raw, {}

    df_cv = df_raw.copy()
    df_cv.columns = df_cv.columns.str.strip()

    targ = "Cyclic Voltammetry"
    if 'Step name' not in df_cv.columns or targ not in df_cv['Step name'].values:
        return None, None

    df_cv['Time (hr)'] = df_cv['Elapsed Time (s)'] / 3600.0
    df_cv['Current (uA)'] = df_cv['Current (A)'] * 1E6
    df_cv['Step name'] = df_cv['Step name'].astype(str).str.strip()

    cv_only_df = df_cv[df_cv['Step name'].str.contains('Cyclic', case=False)].copy()
    if cv_only_df.empty:
        return df_cv, {}

    v = cv_only_df['Working Electrode (V)'].values
    dv = np.diff(v)
    sign_change = np.where(np.diff(np.sign(dv)) != 0)[0] + 1
    
    cycle_boundaries = [0] + list(sign_change[1::2]) + [len(v)]
    
    cv_segments = {}
    expected_rates = [0.5, 1.0, 1.5]
    cycle_count = 1

    for i in range(len(cycle_boundaries) - 1):
        start_idx, end_idx = cycle_boundaries[i], cycle_boundaries[i+1]
        c_df = cv_only_df.iloc[start_idx:end_idx].copy()
        
        if len(c_df) > 10: 
            dt = c_df['Elapsed Time (s)'].iloc[-1] - c_df['Elapsed Time (s)'].iloc[0]
            v_span = c_df['Working Electrode (V)'].max() - c_df['Working Electrode (V)'].min()
            
            calc_rate = (2 * v_span) / dt if dt > 0 else 0.5
            assigned_rate = min(expected_rates, key=lambda x: abs(x - calc_rate))
            
            c_df['sweep_rate'] = assigned_rate
            cv_segments[cycle_count] = c_df
            cycle_count += 1
            
    return df_cv, cv_segments

def calculate_loop_area(voltage, current):
    'CALCULATES ENCLOSED AREA OF CV USING THE SHOELACE FORMULA (SIMPSON METHOD)'
    if len(voltage) < 3 or len(current) < 3:
        return 0.0
    return abs(simpson(y=current, x=voltage))

def calc_cap_from_cv(cv_segments, bias_schedule=None, delta_V=2.0):
    'CALCULATES ENCLOSED AREA OF CV USING THE SHOELACE FORMULA (SIMPSON METHOD)'
    if not cv_segments:
        return pd.DataFrame(columns=["CV Run", "Sweep Rate (V/s)", "Preceding Condition", "Capacitance (uF)", "Time (hr)"])

    for cv_id, cv_df in cv_segments.items():
        voltage = cv_df['Working Electrode (V)'].values if 'Working Electrode (V)' in cv_df else cv_df.iloc[:, 2].values
        current = cv_df['Current (A)'].values if 'Current (A)' in cv_df else cv_df.iloc[:, 3].values
        
        enclosed_area = calculate_loop_area(voltage, current)
        scan_rate = cv_df['sweep_rate'].iloc[0] if 'sweep_rate' in cv_df.columns else 1.0
        
        capacitance = abs(enclosed_area / (2 * scan_rate * delta_V)) if (scan_rate > 0 and delta_V > 0) else 0.0
        cv_df['capacitance (uF)'] = capacitance * 1e6

    summary_rows = []
    first_cv_id = min(cv_segments.keys()) if cv_segments else None

    for cv_id, cv_df in cv_segments.items():
        if 'sweep_rate' not in cv_df.columns:
            continue

        rate = cv_df['sweep_rate'].iloc[0]
        cap = cv_df['capacitance (uF)'].iloc[0]
        raw_t = cv_df['Time (hr)'].iloc[0]
        
        if rate == 1.0:
            if cv_id == first_cv_id:
                prior_condition = "Initial Setup"
            else:
                prior_condition = get_bias_at_time(max(raw_t - 0.5, 0), bias_schedule)

            summary_rows.append({
                "CV Run": cv_id,
                "Sweep Rate (V/s)": rate,
                "Preceding Condition": prior_condition,
                "Capacitance (uF)": cap,
                "Time (hr)": round(raw_t, 1)
            })

    columns = ["CV Run", "Sweep Rate (V/s)", "Preceding Condition", "Capacitance (uF)", "Time (hr)"]
    return pd.DataFrame(summary_rows, columns=columns)

def calc_cap_from_cv_grouped(dfs_dict):
    'CALCULATES CAPACITANCE FROM PRE REACTION CV RUNS.'
    summary_rows = []

    for folder_name, df in dfs_dict.items():
        if "cv" not in folder_name.lower() or "10 " in folder_name:
            continue

        step_col = "Step number" if "Step number" in df.columns else None
        unique_steps = sorted(df[step_col].unique()) if step_col else [1]

        if not step_col:
            df = df.assign(**{"Step number": 1})

        for step in unique_steps:
            step_data = df[df["Step number"] == step].sort_values("Elapsed Time (s)")
            if len(step_data) < 3:
                continue

            v_col = ("Working Electrode (V)" if "Working Electrode (V)" in step_data.columns else step_data.columns[4])
            c_col = ("Current (A)" if "Current (A)" in step_data.columns else step_data.columns[3])

            raw_voltage = pd.to_numeric(step_data[v_col], errors="coerce")
            raw_current = pd.to_numeric(step_data[c_col], errors="coerce")

            valid_idx = raw_voltage.notna() & raw_current.notna()
            voltage, current = raw_voltage[valid_idx].values, raw_current[valid_idx].values

            if len(voltage) < 3:
                continue

            enclosed_area = calculate_loop_area(voltage, current)
            scan_rate = 1.0 if "1 V-s" in folder_name else step * 0.1
            delta_V = voltage.max() - voltage.min()

            capacitance_F = abs(enclosed_area / (2 * scan_rate * delta_V)) if (scan_rate > 0 and delta_V > 0) else 0.0
            t = round(step_data["Time (hr)"].iloc[0], 1) if "Time (hr)" in step_data.columns else 0.0

            summary_rows.append(
                {
                    "Folder Source": folder_name,
                    "Step Number": step,
                    "Sweep Rate (V/s)": round(scan_rate, 2),
                    "Voltage Delta (V)": round(delta_V, 3),
                    "Capacitance (uF)": capacitance_F * 1e6,
                    "Time (hr)": t,
                }
            )

    return pd.DataFrame(summary_rows)

def calc_cap_from_sweep_rate_slope(cv_segments, eval_potential=0.0, bias_schedule=None, sweeps_per_chunk=3):
    """
    Groups CV sweeps into discrete chunks, extracts capacitance via linear regression (i = C * v),
    and subtracts non-zero y-intercept leakage current (i_leak).
    """
    if not cv_segments:
        return pd.DataFrame(), []

    summary_rows = []
    fit_details_list = []

    segment_ids = sorted(cv_segments.keys())
    chunk_groups = [segment_ids[i:i + sweeps_per_chunk] for i in range(0, len(segment_ids), sweeps_per_chunk)]

    for chunk_idx, chunk_ids in enumerate(chunk_groups, start=1):
        sweep_rates = []
        currents_avg_uA = []
        time_hrs = []

        for cv_id in chunk_ids:
            cv_df = cv_segments[cv_id]
            v_col = 'Working Electrode (V)' if 'Working Electrode (V)' in cv_df.columns else cv_df.columns[2]
            c_col = 'Current (uA)' if 'Current (uA)' in cv_df.columns else 'Current (A)'
            
            v_vals = cv_df[v_col].values
            c_vals = cv_df[c_col].values if 'Current (uA)' in cv_df.columns else cv_df[c_col].values * 1e6
            
            if 'sweep_rate' not in cv_df.columns or len(v_vals) < 4:
                continue

            rate = cv_df['sweep_rate'].iloc[0]
            time_hrs.append(cv_df['Time (hr)'].iloc[0] if 'Time (hr)' in cv_df.columns else 0.0)

            peak_idx = np.argmax(v_vals)
            v_fwd, c_fwd = v_vals[:peak_idx], c_vals[:peak_idx]
            v_rev, c_rev = v_vals[peak_idx:], c_vals[peak_idx:]

            i_pos = np.interp(eval_potential, v_fwd, c_fwd) if len(v_fwd) > 1 else np.nan
            i_neg = np.interp(eval_potential, v_rev[::-1], c_rev[::-1]) if len(v_rev) > 1 else np.nan

            if pd.notna(i_pos) and pd.notna(i_neg):
                sweep_rates.append(rate)
                currents_avg_uA.append((i_pos - i_neg) / 2.0)

        if len(sweep_rates) < 2:
            continue

        sweep_rates = np.array(sweep_rates)
        currents_avg_uA = np.array(currents_avg_uA)

        # Linear Regression: i (uA) = C (uF) * v (V/s) + i_leak
        slope, intercept, r_value, p_value, std_err = stats.linregress(sweep_rates, currents_avg_uA)

        # Leakage-corrected capacitive currents
        currents_corrected_uA = currents_avg_uA - intercept

        avg_t = np.mean(time_hrs) if time_hrs else 0.0
        prior_condition = "Initial Setup"
        if bias_schedule and avg_t > 0.1:
            prior_condition = get_bias_at_time(avg_t, bias_schedule)

        summary_rows.append({
            "Chunk ID": chunk_idx,
            "Evaluation Potential (V)": eval_potential,
            "Capacitance (uF)": abs(slope),
            "Leakage Current (uA)": intercept,
            "R-squared": r_value**2,
            "Slope Std Err": std_err,
            "Preceding Condition": prior_condition,
            "Time (hr)": round(avg_t, 2)
        })

        fit_details_list.append({
            "chunk_id": chunk_idx,
            "sweep_rates": sweep_rates,
            "currents_avg_uA": currents_avg_uA,
            "currents_corrected_uA": currents_corrected_uA,
            "slope": slope,
            "intercept": intercept,
            "r_squared": r_value**2,
            "time_hr": round(avg_t, 2)
        })

    return pd.DataFrame(summary_rows), fit_details_list







# ==============================================================================
# EIS ANALYSIS FUNCTIONS
# ==============================================================================
def reaction_EIS_parser(df, max_freq=2000000.0):
    'SEGMENTS RAW EIS CONTINUOUS DTAFRAME INTO DISCRETE EIS RUN IDs'
    df_parsed = df.copy()
    df_parsed["is_new_run"] = (df_parsed["Frequency (Hz)"] == max_freq).fillna(True)
    df_parsed["EIS_Run_ID"] = df_parsed["is_new_run"].cumsum()
    return df_parsed

def _fit_and_plot_eis(f, Z, title_name):
    'HELPER FUNCTION TO FIT RANDLES-LIKE CIRCUIT AND PLOT BODE/NYQUIST PLOTS curves'
    circuit = CustomCircuit(circuit='R0-p(R1,CPE1)', initial_guess=[1000.0, 200000.0, 1e-6, 0.85])
    try:
        circuit.fit(f, Z)
        fitted_params = dict(zip(circuit.get_param_names()[0], circuit.parameters_))
        Z_fit = circuit.predict(f)
    except Exception as e:
        warnings.warn(f"EIS Fit failed for {title_name}: {e}")
        fitted_params = {'R0': np.nan, 'R1': np.nan, 'CPE1_0': np.nan, 'CPE1_1': np.nan}
        Z_fit = np.full_like(Z, np.nan + 1j * np.nan)

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 6))
    ax_bode_mag, ax_nyquist = axes[0], axes[1]

    # Bode Plot
    ax_bode_mag.loglog(f, np.abs(Z), 'D', color="#2740CF", label='Magnitude |Z|', markersize=5)
    ax_bode_mag.loglog(f, np.abs(Z_fit), '-', color="#2740CF", label='Fit |Z|', linewidth=2)
    ax_bode_mag.set_xlabel("Frequency (Hz)", fontsize=16)
    ax_bode_mag.set_ylabel(r"Magnitude |Z| ($\Omega$)", fontsize=16)

    ax_bode_phase = ax_bode_mag.twinx()
    ax_bode_phase.semilogx(f, np.angle(Z, deg=True), '^', color="#6E84FF", label='Phase Angle', markersize=5)
    ax_bode_phase.semilogx(f, np.angle(Z_fit, deg=True), '-', color="#6E84FF", label='Fit Phase', linewidth=2)
    ax_bode_phase.set_ylabel(r"Phase Angle ($\degree$)", fontsize=16)

    lines1, labels1 = ax_bode_mag.get_legend_handles_labels()
    lines2, labels2 = ax_bode_phase.get_legend_handles_labels()
    ax_bode_mag.legend(lines1[:1] + lines2[:1], labels1[:1] + labels2[:1], loc='center right', frameon=False)
    ax_bode_mag.set_title("Bode Plot: Magnitude & Phase", fontsize=18)

    # Nyquist Plot
    ax_nyquist.plot(Z.real, -Z.imag, 'o', color="#A10601", label='Experimental Data', markersize=5)
    ax_nyquist.plot(Z_fit.real, -Z_fit.imag, '-', color='black', label='Fitted Model', linewidth=2)
    ax_nyquist.set_xlabel(r"Z' ($\Omega$)", fontsize=16)
    ax_nyquist.set_ylabel(r"-Z'' ($\Omega$)", fontsize=16)
    ax_nyquist.legend(loc='center right', frameon=False)
    ax_nyquist.set_title("Nyquist Plot", fontsize=18)

    fig.suptitle(f"EIS Analysis: R0-p(R1,CPE1) - {title_name}", fontsize=18, y=0.98)
    fig.tight_layout()

    return circuit, (fig, axes), fitted_params

def EIS_cap_rxn(df, title_name, bias_schedule=None):
    'PROCESSES FITS, AND PLOTS MULTI-RUN REACTION EIS MEASUREMENTS'
    if "EIS_Run_ID" not in df.columns:
        df = reaction_EIS_parser(df)

    working_df = df.dropna(subset=["Frequency (Hz)","|Z| (Ohms)", "Phase (deg)"]).copy()
    working_df["Z' (Ohms)"] = working_df["|Z| (Ohms)"] * np.cos(np.radians(working_df["Phase (deg)"]))  
    working_df['-Z" (Ohms)'] = -working_df["|Z| (Ohms)"] * np.sin(np.radians(working_df["Phase (deg)"]))

    sum_rows  = []
    last_circuit, last_fig_axes, last_params = None, None, None

    for run_id, chunk in working_df.groupby('EIS_Run_ID', sort=False):
        f = chunk['Frequency (Hz)'].values
        complex_array = chunk["Z' (Ohms)"].values - 1j * chunk['-Z" (Ohms)'].values
        valid_mask = (f > 0) & np.isfinite(complex_array)

        if np.count_nonzero(valid_mask) < 4:
            continue

        f_clean, Z_clean = f[valid_mask], complex_array[valid_mask]
        circuit, fig_axes, params = _fit_and_plot_eis(f_clean, Z_clean, f"{title_name} Run {run_id}")

        if last_fig_axes is not None:
            plt.close(last_fig_axes[0])

        last_circuit, last_fig_axes, last_params = circuit, fig_axes, params

        raw_t = chunk['Elapsed Time (s)'].iloc[0] / 3600
        dc_WE = df['DC Working Electrode (V)'].iloc[0] if 'DC Working Electrode (V)' in chunk.columns else np.nan
        
        prior_condition = None
        if bias_schedule:
            cum_time = 0
            for entry in bias_schedule:
                start, end = cum_time, cum_time + entry["duration_hours"]
                if start <= raw_t <= (end + 0.2):
                    prior_condition = "Initial Setup" if (start == 0 and raw_t < 0.1) else entry["voltage"]
                    break
                cum_time = end
            if prior_condition is None:
                prior_condition = "Initial Setup" if raw_t < 0.1 else bias_schedule[-1]["voltage"]

        r0_val = getattr(circuit, 'parameters_', [np.nan]*3)[0]
        r1_val = getattr(circuit, 'parameters_', [np.nan]*3)[1]
        cpe1_0__val = getattr(circuit, 'parameters_', [np.nan]*3)[2]

        sum_rows.append({
            "EIS Run": run_id,
            "Preceding Condition": prior_condition,
            "r0_value": float(r0_val),
            "r1_value": float(r1_val),
            "cpe1_0": float(cpe1_0__val) * 1e6,
            "Time (hr)": round(raw_t, 1),
            "DC Working Electrode (V)": f"{dc_WE:.1f}" if pd.notna(dc_WE) else "N/A"
        })

    return last_circuit, last_fig_axes, last_params, pd.DataFrame(sum_rows)

def EIS_cap_pre(df, title_name):
    'PROCESSES A SINGLE EIS SWEEP DATAFRAME AND RETURNS FIT METRICS'
    clean_df = df.dropna(subset=["Frequency (Hz)", "Z' (Ohms)", '-Z" (Ohms)'])
    f = clean_df['Frequency (Hz)'].values
    Z = clean_df["Z' (Ohms)"].values -1j * clean_df['-Z" (Ohms)'].values

    circuit, fig_axes, params = _fit_and_plot_eis(f, Z, title_name)
    dc_WE = clean_df['DC Working Electrode (V)'].iloc[0] if 'DC Working Electrode (V)' in clean_df.columns else np.nan
    
    r0_val = getattr(circuit, 'parameters_', [np.nan]*3)[0]
    r1_val = getattr(circuit, 'parameters_', [np.nan]*3)[1]
    cpe1_0__val = getattr(circuit, 'parameters_', [np.nan]*3)[2]

    summary_df = pd.DataFrame([{
        "EIS Run": title_name,
        "r0_value": float(r0_val),
        "r1_value": float(r1_val),
        "cpe1_0": float(cpe1_0__val) * 1e6,
        "DC Working Electrode (V)": f"{dc_WE:.1f}" if pd.notna(dc_WE) else "N/A"
    }])
    
    return circuit, fig_axes, params, summary_df

# ---------------------------------------------------
# CHARGE TRANSFER ANALYSIS
# ---------------------------------------------------

def charge_transfer(df_electronics, ss_fraction = 0.25):
    'Integrates transient charge behavior across constant potential'
    df = df_electronics.copy()

    if 'Current (A)' not in df.columns:
        if 'Current (uA)' in df.columns:
            df['Current (A)'] = df['Current (uA)'] / 1e6
        else:
            raise KeyError("DataFrame must contain 'Current (A)' or 'Current (uA)'.")
    
    v_col = ('Working Electrode (V)' if 'Working Electrode (V)' in df.columns else 'DC Working Electrode (V)')
    potential_rounded = df[v_col].round(2)
    step_change = (df['Step name'] != df['Step name'].shift()) | (potential_rounded != potential_rounded.shift())
    df['window_id'] = step_change.cumsum()

    summary_rows = []

    for window_id, group in df.groupby('window_id'):
        if len(group) < 4:
            continue
        group = group.sort_values('Elapsed Time (s)')

        time_sec = group['Elapsed Time (s)'].values
        current_amp = group['Current (A)'].values

        tail_start_idx = int(len(current_amp) * (1.0 - ss_fraction))
        i_ss_amp = np.mean(current_amp[tail_start_idx:])
        transient_current_amp = current_amp - i_ss_amp

        raw_charge_C = simpson(y=current_amp, x=time_sec)
        transient_charge_C = simpson(y=transient_current_amp, x=time_sec)
        
        start_time_hr = group['Time (hr)'].iloc[0] if 'Time (hr)' in group.columns else time_sec[0] / 3600.0
        end_time_hr = group['Time (hr)'].iloc[-1] if 'Time (hr)' in group.columns else time_sec[-1] / 3600.0

        summary_rows.append({
            "Window ID": window_id,
            "Step Name": group['Step name'].iloc[0],
            "Applied Voltage (V)": round(group[v_col].mean(), 2),
            "Start Time (hr)": round(start_time_hr, 2),
            "End Time (hr)": round(end_time_hr, 2),
            "Duration (hr)": round(end_time_hr - start_time_hr, 2),
            "Baseline Current (uA)": i_ss_amp * 1e6,
            "Transient Charge (mC)": transient_charge_C * 1e3,
            "Abs Transient Charge (mC)": abs(transient_charge_C) * 1e3,
            "Total Charge (mC)": raw_charge_C * 1e3,
            "Avg Current (uA)": group['Current (uA)'].mean() if 'Current (uA)' in group.columns else (current_amp.mean() * 1e6)
        })

    return pd.DataFrame(summary_rows)

# ---------------------------------------------------
# GENERAL UTILITIES
# ---------------------------------------------------
def _to_scalar(val, default=0):
    if val is None:
        return default
    if isinstance(val, (list, tuple, np.ndarray)):
        val = val[0] if len(val) > 0 else default
    return float(val)

def calc_stats(values):
    'Calculates the mean and 95% confidence interval uncertainties for the array/list "values"'
    if values is None:
        return np.nan, np.nan

    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]

    if len(arr) <= 1:
        return (arr[0], 0.0) if len(arr) == 1 else (np.nan, np.nan)
        
    avg = float(np.mean(arr))
    sem = float(stats.sem(arr))

    if sem == 0 or np.isnan(sem):
        return avg, 0.0

    try:
        ci = stats.t.interval(0.95, df=len(arr)-1, loc=avg, scale=sem)
        return avg, avg - ci[0]
    except Exception:
        return avg, 0.0

def getcstats(df_raw=None, file_path=None, data_dir=None, save_to_data_dir=True):
    'Loads CSV and extracts/calculates general reaction electronics metadata'
    source_path = Path(file_path) if file_path else None

    if df_raw is None:
        if source_path:
            df_raw = pd.read_csv(source_path)
        elif data_dir:
            data_dir = Path(data_dir)
            if not data_dir.exists():
                raise FileNotFoundError(f"data_dir does not exist: {data_dir}")
            csv_files = sorted(data_dir.glob('*.csv'), key=lambda p: p.stat().st_mtime, reverse=True)
            if not csv_files:
                raise FileNotFoundError(f"No CSV files found in {data_dir}")
            source_path = csv_files[0]
            df_raw = pd.read_csv(source_path)
        else:
            raise ValueError('Provide df_raw or file_path or data_dir')

    if save_to_data_dir and source_path and data_dir:
        data_dir = Path(data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)
        target_path = data_dir / source_path.name
        if source_path.resolve() != target_path.resolve():
            try:
                shutil.copy2(source_path, target_path)
            except Exception as e:
                warnings.warn(f"Could not copy {source_path}: {e}")

    df_master = df_raw.copy()
    df_master.columns = df_master.columns.str.strip()

    expected_cols = ['Step name', 'Elapsed Time (s)', 'Working Electrode (V)', 'Current (A)', 'Counter Electrode (V)']
    if not all(col in df_master.columns for col in expected_cols):
        raise ValueError('Error: CSV file columns do not match expected layout.')
    
    df_master['Time (hr)'] = df_master['Elapsed Time (s)'] / 3600.0
    df_master['Current (uA)'] = df_master['Current (A)'] * 1e6

    target_steps = ['Constant Potential', 'Open Circuit Potential']
    df_electronics = df_master[df_master['Step name'].isin(target_steps)].copy()

    df_electronics["current/2 (nmol/min)"] = (abs(df_electronics['Current (uA)']) * 1e-6 / 96485 * 60 / 2 * 1e9)

    return df_electronics

# ---------------------------------
# Faradaic Efficiency Calculation
# ---------------------------------
def calculate_gc_faradaic_efficiency(rates_df, df_electronics, n_electrons=2, gc_sampling_window_sec=60):
    """
    Aligns GC reaction rates (from calc_rxn_rates/calc_rxn_rates_dynamic) with
    electrochemical current data to calculate instantaneous Faradaic Efficiency (FE %).

    Parameters:
    -----------
    rates_df : DataFrame
        Processed GC DataFrame containing 'Time (hr)', 'CO2 (nmol/min)', 'H2 (nmol/min)', etc.
    df_electronics : DataFrame
        Electrochemical DataFrame from getcstats() containing 'Time (hr)' and 'Current (A)'.
    n_electrons : int
        Electrons per product molecule (n=2 for HCOOH -> CO2 + 2H+ + 2e-).
    gc_sampling_window_sec : float
        Time window (seconds) over which the GC sample loop is filled prior to injection.

    Returns:
    --------
    rates_df_fe : DataFrame
        Updated rates DataFrame containing FE % columns for CO2, H2, and total FE.
    """
    FARADAY = 96485.0  # C / mol
    
    df_result = rates_df.copy()
    
    co2_fe_list, h2_fe_list, total_fe_list, avg_current_mA_list = [], [], [], []

    for _, row in df_result.iterrows():
        t_gc = row['Time (hr)']
        
        # 1. Define sampling time window prior to GC injection
        t_start = t_gc - (gc_sampling_window_sec / 3600.0)
        
        # 2. Extract corresponding current points from electronics data
        mask = (df_electronics['Time (hr)'] >= t_start) & (df_electronics['Time (hr)'] <= t_gc)
        matching_currents = df_electronics.loc[mask, 'Current (A)'].abs()
        
        # Fallback to the nearest electrochemical point if GC window is sparse
        if matching_currents.empty:
            nearest_idx = (df_electronics['Time (hr)'] - t_gc).abs().idxmin()
            mean_current_A = abs(df_electronics.loc[nearest_idx, 'Current (A)'])
        else:
            mean_current_A = matching_currents.mean()

        avg_current_mA_list.append(mean_current_A * 1e3)

        # 3. Calculate FE (%) for each species:
        # Rate (mol/s) = Rate (nmol/min) * 1e-9 / 60
        # Theoretical Current I_theo = Rate (mol/s) * n * F
        if mean_current_A > 0:
            # CO2 Faradaic Efficiency
            co2_rate_nmol_min = row.get('CO2 (nmol/min)', 0.0)
            co2_i_theo_A = (co2_rate_nmol_min * 1e-9 / 60.0) * n_electrons * FARADAY
            co2_fe = (co2_i_theo_A / mean_current_A) * 100.0
            
            # H2 Faradaic Efficiency
            h2_rate_nmol_min = row.get('H2 (nmol/min)', 0.0)
            h2_i_theo_A = (h2_rate_nmol_min * 1e-9 / 60.0) * n_electrons * FARADAY
            h2_fe = (h2_i_theo_A / mean_current_A) * 100.0
        else:
            co2_fe = 0.0
            h2_fe = 0.0

        co2_fe_list.append(round(co2_fe, 2))
        h2_fe_list.append(round(h2_fe, 2))
        total_fe_list.append(round(co2_fe + h2_fe, 2))

    df_result['Avg Current (mA)'] = avg_current_mA_list
    df_result['FE_CO2 (%)'] = co2_fe_list
    df_result['FE_H2 (%)'] = h2_fe_list
    df_result['FE_Total (%)'] = total_fe_list

    return df_result

def correlate_rates_and_capacitance_by_schedule(summary_rates_df, cv_capacitance_df, bias_schedule_df):
    """
    Safely correlates window average rates with CV capacitances using bias_schedule.
    Includes fallback matching to ensure data is never lost due to column naming or time-units.
    """
    if summary_rates_df is None or summary_rates_df.empty:
        print("[Warning] summary_rates_df is empty.")
        return pd.DataFrame()
    if cv_capacitance_df is None or cv_capacitance_df.empty:
        print("[Warning] cv_capacitance_df is empty.")
        return pd.DataFrame()

    # Identify time column in CV data dynamically
    time_col = next((c for c in cv_capacitance_df.columns if 'time' in c.lower() or 'hr' in c.lower()), cv_capacitance_df.columns[0])
    cap_col = next((c for c in cv_capacitance_df.columns if 'cap' in c.lower()), cv_capacitance_df.columns[1])

    correlated_rows = []

    # Loop through each row of the summary rates (which already correspond to windows)
    for idx, rate_row in summary_rates_df.iterrows():
        win_num = rate_row.get('Window', idx + 1)
        
        # Try to pull timing from rate_row or bias_schedule_df
        t_start = rate_row.get('Start (hr)', None)
        t_end = rate_row.get('End (hr)', None)
        
        if (t_start is None or pd.isna(t_start)) and bias_schedule_df is not None and not bias_schedule_df.empty:
            if idx < len(bias_schedule_df):
                sched_row = bias_schedule_df.iloc[idx]
                t_start = sched_row.get('Start (hr)', sched_row.get('start_time', None))
                t_end = sched_row.get('End (hr)', sched_row.get('end_time', None))

        # Filter CV capacitances within the window range
        if t_start is not None and t_end is not None:
            cap_matches = cv_capacitance_df[
                (cv_capacitance_df[time_col] >= t_start) & 
                (cv_capacitance_df[time_col] <= t_end)
            ]
        else:
            cap_matches = pd.DataFrame()

        # Fallback: If timing matching finds nothing, chunk the CV dataframe sequentially
        if cap_matches.empty:
            num_windows = len(summary_rates_df)
            chunk_size = max(1, len(cv_capacitance_df) // num_windows)
            start_i = int(idx * chunk_size)
            end_i = int((idx + 1) * chunk_size) if idx < num_windows - 1 else len(cv_capacitance_df)
            cap_matches = cv_capacitance_df.iloc[start_i:end_i]

        # Append individual rows for each capacitance point found in this window
        for cap_idx, (_, cap_row) in enumerate(cap_matches.iterrows(), start=1):
            correlated_rows.append({
                "Window": int(win_num),
                "CV_Measurement": cap_idx,
                "Voltage (V)": rate_row.get("Voltage (V)", rate_row.get("Bias (V)", 0.0)),
                "Capacitance (uF)": cap_row[cap_col],
                "CO2_mean": rate_row.get("CO2_mean", 0.0),
                "CO2_err": rate_row.get("CO2_err", 0.0),
                "H2_mean": rate_row.get("H2_mean", 0.0),
                "H2_err": rate_row.get("H2_err", 0.0),
                "CO_mean": rate_row.get("CO_mean", 0.0),
                "CO_err": rate_row.get("CO_err", 0.0),
            })

    result_df = pd.DataFrame(correlated_rows)
    print(f"[Success] Matched {len(result_df)} total CV capacitance points across {len(summary_rates_df)} windows.")
    return result_df

