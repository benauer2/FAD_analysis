from pathlib import Path
import shutil
import warnings
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import simpson
from scipy import stats
from impedance.models.circuits import CustomCircuit

from config import get_step_at_time, get_voltage_at_time

def assign_voltage_windows(df, rxn_schedule):
    """
    Annotates the electronics DataFrame with schedule properties based on hard step boundaries:
    Bias (V), Temp_C, N2_sccm, FA_flow_ul_min, Step_Index, Step_Start_hr, Step_End_hr, Step_Rel_Time_hr, Avg_Window.
    """
    if df is None or df.empty:
        return df

    if 'Time (hr)' not in df.columns and 'Elapsed Time (s)' in df.columns:
        df['Time (hr)'] = pd.to_numeric(df['Elapsed Time (s)'], errors='coerce') / 3600.0
    
    if not rxn_schedule:
        df['Voltage_Window'] = 0.0
        df['Bias (V)'] = 0.0
        df['Temp_C'] = 0.0
        df['N2_sccm'] = 0.0
        df['FA_flow_ul_min'] = 0.0
        df['Step_Index'] = 1
        df['Step_Start_hr'] = 0.0
        df['Step_End_hr'] = df['Time (hr)'].max() if ('Time (hr)' in df.columns and not df['Time (hr)'].empty and np.isfinite(df['Time (hr)'].max())) else 0.0
        df['Step_Rel_Time_hr'] = df['Time (hr)'] if 'Time (hr)' in df.columns else 0.0
        return df

    # If Step_Index column is already explicitly present (from hardware orchestrator)
    if "Step_Index" in df.columns and df["Step_Index"].dropna().nunique() > 0:
        def match_by_step_or_time(row):
            s_idx = row.get("Step_Index")
            try:
                s_int = int(s_idx)
                if 1 <= s_int <= len(rxn_schedule):
                    return rxn_schedule[s_int - 1]
            except (ValueError, TypeError):
                pass
            t_val = row.get("Time (hr)", 0.0)
            return get_step_at_time(t_val, RXN_SCHEDULE=rxn_schedule)

        step_series = df.apply(match_by_step_or_time, axis=1)
    else:
        step_series = df['Time (hr)'].apply(
            lambda t: get_step_at_time(t, RXN_SCHEDULE=rxn_schedule)
        )

    df['Voltage_Window'] = step_series.apply(
        lambda s: s["we_volt"] if isinstance(s, dict) and "we_volt" in s else 0.0
    )
    df['Bias (V)'] = df['Voltage_Window']
    df['Temp_C'] = step_series.apply(
        lambda s: s["rxr_t"] if isinstance(s, dict) and "rxr_t" in s else 0.0
    )
    df['N2_sccm'] = step_series.apply(
        lambda s: s["n2_flow"] if isinstance(s, dict) and "n2_flow" in s else 0.0
    )
    df['FA_flow_ul_min'] = step_series.apply(
        lambda s: s["fa_flow"] if isinstance(s, dict) and "fa_flow" in s else 0.0
    )
    df['Step_Index'] = step_series.apply(
        lambda s: s["run"] if isinstance(s, dict) and "run" in s else 1
    )
    df['Step_Start_hr'] = step_series.apply(
        lambda s: s["time_window"][0] if isinstance(s, dict) and "time_window" in s else 0.0
    )
    df['Step_End_hr'] = step_series.apply(
        lambda s: s["time_window"][1] if isinstance(s, dict) and "time_window" in s else 0.0
    )
    df['Step_Rel_Time_hr'] = df['Time (hr)'] - df['Step_Start_hr']
    df['Avg_Window'] = step_series.apply(
        lambda s: s["avg_window"] if isinstance(s, dict) and "avg_window" in s else [0.0, 0.0]
    )

    return df


#------------------------------------------------------------------
# CV ANALYSIS FUNCTIONS
#------------------------------------------------------------------
def detect_sweep_rate(df_sweep, default_map=None, sweep_idx=0):
    """
    Detects CV sweep rate from dataframe notes/columns or directly from empirical dV/dt slope.
    """
    # 1. Check explicit columns
    rate_col_candidates = [
        "Sweep_Rate_V_s", "Sweep_rate_v_s", "Scan Rate (V/s)", "Sweep Rate (V/s)", "Scan Rate", "Sweep Rate",
        "Scan_Rate", "Sweep_Rate", "rate", "Rate", "scan_rate", "sweep_rate"
    ]
    for col in rate_col_candidates:
        if col in df_sweep.columns:
            val = pd.to_numeric(df_sweep[col], errors="coerce").dropna()
            if not val.empty and val.iloc[0] > 0:
                return float(val.iloc[0])

    # 2. Check notes / comment columns for text patterns
    text_col_candidates = ["Notes", "Note", "Comment", "Comments", "Step description", "Step name", "Step_Name", "Tag", "Label"]
    for col in text_col_candidates:
        if col in df_sweep.columns:
            text_val = str(df_sweep[col].iloc[0])
            m_mv = re.search(r'([0-9]*\.?[0-9]+)\s*m[vV]/s', text_val)
            if m_mv:
                return float(m_mv.group(1)) / 1000.0
            m_v = re.search(r'([0-9]*\.?[0-9]+)\s*[vV]/s', text_val)
            if m_v:
                return float(m_v.group(1))

    # 3. Calculate empirical dV/dt from potential and elapsed time data
    v_col = None
    for candidate in ["Working Electrode (V)", "Working_Electrode_V", "Potential (V)", "Voltage (V)", "E (V)"]:
        if candidate in df_sweep.columns:
            v_col = candidate
            break
    if v_col is None and len(df_sweep.columns) > 2:
        v_col = df_sweep.columns[2]

    t_col = None
    for candidate in ["Elapsed Time (s)", "Elapsed_Time_s", "Time (s)", "t (s)"]:
        if candidate in df_sweep.columns:
            t_col = candidate
            break
    if t_col is None and len(df_sweep.columns) > 0:
        t_col = df_sweep.columns[0]

    if v_col and t_col:
        v_vals = pd.to_numeric(df_sweep[v_col], errors="coerce").values
        t_vals = pd.to_numeric(df_sweep[t_col], errors="coerce").values
        valid_mask = np.isfinite(v_vals) & np.isfinite(t_vals)
        v_vals = v_vals[valid_mask]
        t_vals = t_vals[valid_mask]

        if len(v_vals) > 5:
            dt = np.diff(t_vals)
            dv = np.diff(v_vals)
            valid_steps = dt > 0.0005
            if np.any(valid_steps):
                inst_rates = np.abs(dv[valid_steps] / dt[valid_steps])
                filtered_rates = inst_rates[inst_rates > 0.001]
                if len(filtered_rates) > 0:
                    emp_rate = float(np.median(filtered_rates))
                    
                    # Snap to closest known standard rate if within 18% tolerance
                    std_rates = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 5.0]
                    closest_std = min(std_rates, key=lambda x: abs(x - emp_rate))
                    if abs(closest_std - emp_rate) / max(closest_std, 1e-6) < 0.18:
                        return closest_std
                    return round(emp_rate, 3)

    # 4. Fallback to default mapping if provided
    if default_map:
        pos = sweep_idx % len(default_map)
        return default_map.get(pos, 1.0)

    return 1.0


def get_cv(df_raw, sweep_rate_map=None):
    """
    Extracts CV data from reaction electronics DataFrame, partitions into individual sweeps,
    and assigns sweep rates. Fully supports both legacy and new Squidstat output schemas.
    """
    if df_raw is None or df_raw.empty:
        return df_raw, {}
    
    if sweep_rate_map is None:
        sweep_rate_map = {0: 0.5, 1: 1.0, 2: 1.5, 3: 0.02}

    df_cv = df_raw.copy()
    df_cv.columns = df_cv.columns.str.strip()

    # Normalize column names for both schemas
    if "Step name" not in df_cv.columns:
        if "Step_Name" in df_cv.columns:
            df_cv["Step name"] = df_cv["Step_Name"]
        elif "Element_Type" in df_cv.columns:
            df_cv["Step name"] = df_cv["Element_Type"]
        else:
            df_cv["Step name"] = ""

    elem_col = "Element_Type" if "Element_Type" in df_cv.columns else "Step name"
    
    is_cv = (
        df_cv[elem_col].astype(str).str.contains("Cyclic|CV", case=False, na=False) |
        df_cv["Step name"].astype(str).str.contains("Cyclic|CV", case=False, na=False)
    )
    
    if not is_cv.any():
        return df_cv, {}
    
    t_col = "Elapsed Time (s)" if "Elapsed Time (s)" in df_cv.columns else ("Elapsed_Time_s" if "Elapsed_Time_s" in df_cv.columns else df_cv.columns[0])
    df_cv['Time (hr)'] = pd.to_numeric(df_cv[t_col], errors='coerce') / 3600.0
    
    i_col = "Current (A)" if "Current (A)" in df_cv.columns else ("Current" if "Current" in df_cv.columns else df_cv.columns[3])
    df_cv['Current (uA)'] = pd.to_numeric(df_cv[i_col], errors='coerce') * 1e6

    # Extract CV rows
    cv_only_df = df_cv[is_cv].copy()
    if cv_only_df.empty:
        return df_cv, {}

    # Determine major block IDs (e.g. per reaction step or contiguous runs of CV)
    if "Step_Index" in cv_only_df.columns and cv_only_df["Step_Index"].dropna().nunique() > 0:
        cv_only_df["block_id"] = cv_only_df["Step_Index"]
    else:
        # Group contiguous runs of CV
        is_cv_all = is_cv.values
        block_ids = np.zeros(len(df_cv), dtype=int)
        b_count = 0
        in_cv = False
        for idx_row, flag in enumerate(is_cv_all):
            if flag and not in_cv:
                b_count += 1
                in_cv = True
            elif not flag:
                in_cv = False
            block_ids[idx_row] = b_count
        df_cv["_cv_block"] = block_ids
        cv_only_df["block_id"] = df_cv.loc[cv_only_df.index, "_cv_block"]

    cv_segments = {}
    segment_count = 1

    for block_val, chunk_group in cv_only_df.groupby("block_id"):
        chunk_group = chunk_group.copy()
        t_vals = pd.to_numeric(chunk_group[t_col], errors='coerce').fillna(0).values
        
        # Detect new sweep start
        is_new_sweep = np.zeros(len(chunk_group), dtype=bool)
        is_new_sweep[0] = True  # first row of block is start of sweep 0

        # Step name or sweep rate changes
        if "Step_Name" in chunk_group.columns:
            step_changed = (chunk_group["Step_Name"] != chunk_group["Step_Name"].shift(1)).values
            is_new_sweep = is_new_sweep | step_changed

        if "Sweep_Rate_V_s" in chunk_group.columns:
            rate_changed = (chunk_group["Sweep_Rate_V_s"] != chunk_group["Sweep_Rate_V_s"].shift(1)).values
            is_new_sweep = is_new_sweep | rate_changed

        # Timestamp reset (e.g. dt < -0.5s or dt > 3.0s)
        for k in range(1, len(t_vals)):
            dt = t_vals[k] - t_vals[k-1]
            if dt < -0.5 or dt > 3.0:
                is_new_sweep[k] = True

        chunk_group["sweep_in_chunk"] = is_new_sweep.cumsum()

        for sweep_idx, (_, sweep_group) in enumerate(chunk_group.groupby("sweep_in_chunk")):
            clean_sweep = sweep_group.drop(
                columns=["sweep_in_chunk", "block_id", "_cv_block"], errors='ignore').copy()

            clean_sweep["block_id"] = block_val
            clean_sweep["sweep_idx"] = sweep_idx
            clean_sweep["sweep_rate"] = detect_sweep_rate(clean_sweep, default_map=sweep_rate_map, sweep_idx=sweep_idx)
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

def calc_cap_from_cv(cv_segments, rxn_schedule=None, delta_V=2.0, bias_schedule=None):
    'CALCULATES ENCLOSED AREA OF CV USING THE SHOELACE FORMULA (SIMPSON METHOD)'
    schedule = rxn_schedule or bias_schedule
    if not cv_segments:
        return pd.DataFrame(columns=["CV Run", "Sweep Rate (V/s)", "Preceding Condition", "Capacitance (uF)", "Time (hr)"])

    for cv_id, cv_df in cv_segments.items():
        voltage = cv_df['Working Electrode (V)'].values if 'Working Electrode (V)' in cv_df else cv_df.iloc[:, 2].values
        current = cv_df['Current (A)'].values if 'Current (A)' in cv_df else cv_df.iloc[:, 3].values
        
        enclosed_area = calculate_loop_area(voltage, current)
        scan_rate = cv_df['sweep_rate'].iloc[0] if 'sweep_rate' in cv_df.columns else 1.0
        
        span_v = voltage.max() - voltage.min() if len(voltage) > 1 else delta_V
        if span_v <= 0:
            span_v = delta_V
            
        capacitance = abs(enclosed_area / (2 * scan_rate * span_v)) if (scan_rate > 0 and span_v > 0) else 0.0
        cv_df['capacitance (uF)'] = capacitance * 1e6

    summary_rows = []
    first_cv_id = min(cv_segments.keys()) if cv_segments else None

    for cv_id, cv_df in cv_segments.items():
        if 'sweep_rate' not in cv_df.columns:
            continue

        rate = cv_df['sweep_rate'].iloc[0]
        cap = cv_df['capacitance (uF)'].iloc[0]
        raw_t = cv_df['Time (hr)'].iloc[0]
        
        if cv_id == first_cv_id or raw_t < 0.1:
            prior_condition = "Initial Setup"
        elif schedule:
            step = get_step_at_time(max(raw_t - 0.5, 0), schedule)
            if isinstance(step, dict) and "we_volt" in step:
                volt_val = step["we_volt"]
                prior_condition = f"{volt_val} V" if isinstance(volt_val, (int, float)) else str(volt_val)
            else:
                prior_condition = "N/A"
        else:
            prior_condition = "Initial Setup"

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
        if "cv" not in folder_name.lower():
            continue

        step_col = "Step number" if "Step number" in df.columns else None
        unique_steps = sorted(df[step_col].unique()) if step_col else [1]

        if not step_col:
            df = df.assign(**{"Step number": 1})

        for step in unique_steps:
            step_data = df[df["Step number"] == step].sort_values("Elapsed Time (s)") if "Elapsed Time (s)" in df.columns else df[df["Step number"] == step]
            if len(step_data) < 3:
                continue

            v_col = ("Working Electrode (V)" if "Working Electrode (V)" in step_data.columns 
                     else ("Potential (V)" if "Potential (V)" in step_data.columns else step_data.columns[4 if len(step_data.columns) > 4 else 2]))
            c_col = ("Current (A)" if "Current (A)" in step_data.columns 
                     else ("Current (uA)" if "Current (uA)" in step_data.columns else step_data.columns[3]))

            raw_voltage = pd.to_numeric(step_data[v_col], errors="coerce")
            raw_current = pd.to_numeric(step_data[c_col], errors="coerce")

            valid_idx = raw_voltage.notna() & raw_current.notna()
            voltage, current = raw_voltage[valid_idx].values, raw_current[valid_idx].values

            if len(voltage) < 3:
                continue

            if "uA" in str(c_col) or "\u03bcA" in str(c_col):
                current = current * 1e-6

            enclosed_area = calculate_loop_area(voltage, current)
            scan_rate = 1.0 if ("1 V-s" in folder_name or len(unique_steps) == 1) else step * 0.1
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

def calc_cap_from_sweep_rate_slope(cv_segments, eval_potential=0.0, rxn_schedule=None, sweeps_per_chunk=3, bias_schedule=None):
    """
    Groups CV sweeps into discrete chunks, extracts capacitance via linear regression (i = C * v),
    and subtracts non-zero y-intercept leakage current (i_leak).
    """
    schedule = rxn_schedule or bias_schedule
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
        if avg_t <= 0.1 or chunk_idx == 1:
            prior_condition = "Initial Setup"
        elif schedule:
            step = get_step_at_time(max(avg_t - 0.5, 0), schedule)
            if isinstance(step, dict) and "we_volt" in step:
                volt_val = step["we_volt"]
                prior_condition = f"{volt_val} V" if isinstance(volt_val, (int, float)) else str(volt_val)
            else:
                prior_condition = "N/A"
        else:
            prior_condition = "Initial Setup"

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
def reaction_EIS_parser(df, max_freq=1000000.0):
    """
    Segments raw EIS continuous dataframe into discrete EIS Run IDs.
    Supports Step_Index, Step_Name, frequency sweep resets, and elapsed time jumps.
    """
    df_parsed = df.copy()
    if "Step_Index" in df_parsed.columns and df_parsed["Step_Index"].nunique() > 1:
        df_parsed["EIS_Run_ID"] = (df_parsed["Step_Index"] != df_parsed["Step_Index"].shift()).cumsum()
    elif "Step_Name" in df_parsed.columns and df_parsed["Step_Name"].nunique() > 1:
        df_parsed["EIS_Run_ID"] = (df_parsed["Step_Name"] != df_parsed["Step_Name"].shift()).cumsum()
    else:
        freq_col = "Frequency (Hz)" if "Frequency (Hz)" in df_parsed.columns else df_parsed.columns[0]
        freqs = pd.to_numeric(df_parsed[freq_col], errors="coerce").fillna(0)
        is_new = (freqs.diff() > 0) | (freqs == freqs.max())
        if "Elapsed Time (s)" in df_parsed.columns:
            dt = pd.to_numeric(df_parsed["Elapsed Time (s)"], errors="coerce").diff()
            is_new = is_new | (dt > 10.0)
        df_parsed["EIS_Run_ID"] = is_new.fillna(True).cumsum()
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

def EIS_cap_rxn(df, title_name, rxn_schedule=None, bias_schedule=None):
    """
    Processes, fits, and plots multi-run in-situ reaction EIS measurements.
    Returns:
        dict: {run_id: (circuit, fig, params)}
        DataFrame: summary table of R0, R1, CPE1, DC Bias, and Time
    """
    schedule = rxn_schedule or bias_schedule
    if df is None or df.empty:
        return {}, pd.DataFrame()

    if "EIS_Run_ID" not in df.columns:
        df = reaction_EIS_parser(df)

    working_df = df.dropna(subset=["Frequency (Hz)", "|Z| (Ohms)", "Phase (deg)"]).copy()
    working_df["Z' (Ohms)"] = working_df["|Z| (Ohms)"] * np.cos(np.radians(working_df["Phase (deg)"]))  
    working_df['-Z" (Ohms)'] = -working_df["|Z| (Ohms)"] * np.sin(np.radians(working_df["Phase (deg)"]))

    sum_rows = []
    run_results = {}

    for run_id, chunk in working_df.groupby('EIS_Run_ID', sort=False):
        f = chunk['Frequency (Hz)'].values
        complex_array = chunk["Z' (Ohms)"].values - 1j * chunk['-Z" (Ohms)'].values
        valid_mask = (f > 0) & np.isfinite(complex_array)

        if np.count_nonzero(valid_mask) < 4:
            continue

        f_clean, Z_clean = f[valid_mask], complex_array[valid_mask]
        step_lbl = chunk['Step_Name'].iloc[0] if 'Step_Name' in chunk.columns else f"{title_name} Run {run_id}"
        circuit, fig_axes, params = _fit_and_plot_eis(f_clean, Z_clean, step_lbl)

        run_results[run_id] = (circuit, fig_axes[0], params)

        raw_t = chunk['Elapsed Time (s)'].iloc[0] / 3600.0 if 'Elapsed Time (s)' in chunk.columns else 0.0
        dc_WE = chunk['DC Working Electrode (V)'].iloc[0] if 'DC Working Electrode (V)' in chunk.columns else np.nan
        
        if raw_t < 0.1:
            prior_condition = "Initial Setup"
        elif schedule:
            step = get_step_at_time(max(raw_t - 0.2, 0), schedule)
            if isinstance(step, dict) and "we_volt" in step:
                volt_val = step["we_volt"]
                prior_condition = f"{volt_val} V" if isinstance(volt_val, (int, float)) else str(volt_val)
            else:
                prior_condition = "N/A"
        else:
            prior_condition = "Initial Setup"

        r0_val = getattr(circuit, 'parameters_', [np.nan]*4)[0]
        r1_val = getattr(circuit, 'parameters_', [np.nan]*4)[1]
        cpe1_0__val = getattr(circuit, 'parameters_', [np.nan]*4)[2]

        sum_rows.append({
            "EIS Run": run_id,
            "Step Name": step_lbl,
            "Preceding Condition": prior_condition,
            "r0_value": float(r0_val),
            "r1_value": float(r1_val),
            "cpe1_0": float(cpe1_0__val) * 1e6,
            "Time (hr)": round(raw_t, 2),
            "DC Working Electrode (V)": f"{dc_WE:.2f}" if pd.notna(dc_WE) else "N/A"
        })

    return run_results, pd.DataFrame(sum_rows)

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

def charge_transfer(df_electronics, rxn_schedule=None, ss_fraction=0.25):
    'Integrates transient charge behavior across constant potential steps'
    if df_electronics is None or df_electronics.empty:
        return pd.DataFrame()

    df = df_electronics.copy()

    if 'Current (A)' not in df.columns:
        if 'Current (uA)' in df.columns:
            df['Current (A)'] = df['Current (uA)'] / 1e6
        else:
            raise KeyError("DataFrame must contain 'Current (A)' or 'Current (uA)'.")
    
    if 'Time (hr)' not in df.columns and 'Elapsed Time (s)' in df.columns:
        df['Time (hr)'] = df['Elapsed Time (s)'] / 3600.0

    v_col = ('Working Electrode (V)' if 'Working Electrode (V)' in df.columns else 'DC Working Electrode (V)') if ('Working Electrode (V)' in df.columns or 'DC Working Electrode (V)' in df.columns) else None

    summary_rows = []

    # If rxn_schedule is provided, use scheduled step hard boundaries
    if rxn_schedule:
        for phase in rxn_schedule:
            if not isinstance(phase, dict) or "time_window" not in phase:
                continue
            start_t, end_t = phase["time_window"]
            group = df[(df['Time (hr)'] >= start_t) & (df['Time (hr)'] <= end_t)]
            
            # Filter for Constant Potential steps if column exists
            if 'Step name' in group.columns:
                cp_group = group[group['Step name'] == 'Constant Potential']
                if not cp_group.empty:
                    group = cp_group

            if len(group) < 4:
                continue
            group = group.sort_values('Elapsed Time (s)' if 'Elapsed Time (s)' in group.columns else 'Time (hr)')

            time_sec = group['Elapsed Time (s)'].values if 'Elapsed Time (s)' in group.columns else group['Time (hr)'].values * 3600.0
            current_amp = group['Current (A)'].values

            tail_start_idx = int(len(current_amp) * (1.0 - ss_fraction))
            i_ss_amp = np.mean(current_amp[tail_start_idx:])
            transient_current_amp = current_amp - i_ss_amp

            raw_charge_C = simpson(y=current_amp, x=time_sec)
            transient_charge_C = simpson(y=transient_current_amp, x=time_sec)
            
            step_name = group['Step name'].iloc[0] if 'Step name' in group.columns else "Constant Potential"
            appl_v = phase.get("we_volt", group[v_col].mean() if v_col else 0.0)

            summary_rows.append({
                "Window ID": phase.get("run", len(summary_rows) + 1),
                "Step Name": step_name,
                "Applied Voltage (V)": round(appl_v, 2) if isinstance(appl_v, (int, float)) else appl_v,
                "Start Time (hr)": round(start_t, 2),
                "End Time (hr)": round(end_t, 2),
                "Duration (hr)": round(end_t - start_t, 2),
                "Baseline Current (uA)": i_ss_amp * 1e6,
                "Transient Charge (mC)": transient_charge_C * 1e3,
                "Abs Transient Charge (mC)": abs(transient_charge_C) * 1e3,
                "Total Charge (mC)": raw_charge_C * 1e3,
                "Avg Current (uA)": group['Current (uA)'].mean() if 'Current (uA)' in group.columns else (current_amp.mean() * 1e6)
            })

    else:
        # Fallback grouping by potential step change
        potential_rounded = df[v_col].round(2) if v_col else pd.Series(0, index=df.index)
        step_change = (df['Step name'] != df['Step name'].shift()) if 'Step name' in df.columns else (potential_rounded != potential_rounded.shift())
        if v_col:
            step_change = step_change | (potential_rounded != potential_rounded.shift())
        df['window_id'] = step_change.cumsum()

        for window_id, group in df.groupby('window_id'):
            if len(group) < 4:
                continue
            group = group.sort_values('Elapsed Time (s)' if 'Elapsed Time (s)' in group.columns else 'Time (hr)')

            time_sec = group['Elapsed Time (s)'].values if 'Elapsed Time (s)' in group.columns else group['Time (hr)'].values * 3600.0
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
                "Step Name": group['Step name'].iloc[0] if 'Step name' in group.columns else "Constant Potential",
                "Applied Voltage (V)": round(group[v_col].mean(), 2) if v_col else 0.0,
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

def getcstats(df_raw=None, file_path=None, data_dir=None, save_to_data_dir=True, rxn_schedule=None):
    'Loads CSV and extracts/calculates general reaction electronics metadata'
    std_cols = ['Time (hr)', 'Current (uA)', 'Current (A)', 'Working Electrode (V)', 'current/2 (nmol/min)', 'Bias (V)', 'Step_Index']
    
    source_path = Path(file_path) if file_path else None

    if df_raw is None:
        if source_path:
            df_raw = pd.read_csv(source_path)
        elif data_dir:
            data_dir = Path(data_dir)
            if not data_dir.exists():
                return pd.DataFrame(columns=std_cols)
            csv_files = sorted(data_dir.glob('*.csv'), key=lambda p: p.stat().st_mtime, reverse=True)
            if not csv_files:
                return pd.DataFrame(columns=std_cols)
            source_path = csv_files[0]
            df_raw = pd.read_csv(source_path)
        else:
            return pd.DataFrame(columns=std_cols)

    if df_raw is None or df_raw.empty:
        return pd.DataFrame(columns=std_cols)

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

    # Normalize column names for both legacy and new schemas
    if "Step name" not in df_master.columns:
        if "Step_Name" in df_master.columns:
            df_master["Step name"] = df_master["Step_Name"]
        elif "Element_Type" in df_master.columns:
            df_master["Step name"] = df_master["Element_Type"]
        else:
            df_master["Step name"] = "Constant Potential"

    elem_type = df_master["Element_Type"] if "Element_Type" in df_master.columns else df_master["Step name"]

    t_col = "Elapsed Time (s)" if "Elapsed Time (s)" in df_master.columns else ("Elapsed_Time_s" if "Elapsed_Time_s" in df_master.columns else df_master.columns[0])
    
    # Calculate continuous monotonically increasing time across elements
    t_vals = pd.to_numeric(df_master[t_col], errors='coerce').fillna(0).values
    if len(t_vals) > 1:
        cum_t = np.zeros(len(t_vals), dtype=float)
        offset = 0.0
        cum_t[0] = t_vals[0]
        
        step_changed = np.zeros(len(t_vals), dtype=bool)
        if "Step_Name" in df_master.columns:
            step_changed = (df_master["Step_Name"] != df_master["Step_Name"].shift(1)).values
            step_changed[0] = False
        elif "Element_Type" in df_master.columns:
            step_changed = (df_master["Element_Type"] != df_master["Element_Type"].shift(1)).values
            step_changed[0] = False

        for i in range(1, len(t_vals)):
            if (t_vals[i] < t_vals[i-1] - 0.5) or (step_changed[i] and t_vals[i] < 2.0 and t_vals[i-1] > 1.0):
                offset += t_vals[i-1]
            cum_t[i] = t_vals[i] + offset
            
        df_master['Time (hr)'] = cum_t / 3600.0
    else:
        df_master['Time (hr)'] = t_vals / 3600.0

    i_col = "Current (A)" if "Current (A)" in df_master.columns else ("Current" if "Current" in df_master.columns else df_master.columns[3])
    df_master['Current (uA)'] = pd.to_numeric(df_master[i_col], errors='coerce') * 1e6
    df_master['Current (A)'] = pd.to_numeric(df_master[i_col], errors='coerce')

    v_col = "Working Electrode (V)" if "Working Electrode (V)" in df_master.columns else ("Working_Electrode_V" if "Working_Electrode_V" in df_master.columns else df_master.columns[2])
    df_master['Working Electrode (V)'] = pd.to_numeric(df_master[v_col], errors='coerce')

    # Identify Constant Potential and Open Circuit steps
    is_target = (
        elem_type.astype(str).str.contains("Constant Potential|Open Circuit", case=False, na=False) |
        df_master["Step name"].astype(str).str.contains("Constant Potential|Open Circuit", case=False, na=False)
    )

    if not is_target.any():
        df_electronics = df_master.copy()
    else:
        df_electronics = df_master[is_target].copy()

    df_electronics["current/2 (nmol/min)"] = (abs(df_electronics['Current (uA)']) * 1e-6 / 96485 * 60 / 2 * 1e9)

    if rxn_schedule:
        df_electronics = assign_voltage_windows(df_electronics, rxn_schedule)

    return df_electronics

# ---------------------------------
# Faradaic Efficiency Calculation
# ---------------------------------
def calculate_gc_faradaic_efficiency(
    rates_df, 
    df_electronics, 
    n_electrons=2, 
    gc_sampling_window_sec=60,
    baseline_rate=0.0,
    thermal_baseline=None,
    baseline_rate_co2=None,
    baseline_rate_h2=None,
    min_current_A=1e-7
):
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
    baseline_rate : float, int, dict, or Series, optional
        Thermal baseline rate(s) (in nmol/min) to subtract from extensive rates.
        - Scalar (float/int): subtracted from extensive rates.
        - Dict of species: e.g. {'CO2': 15.0, 'H2': 10.0}.
        - Dict of temperatures: e.g. {80: 5.0, 100: 15.0, 120: 35.0}.
        - Nested dict: e.g. {'CO2': {80: 5.0, 100: 15.0}, 'H2': {80: 4.0, 100: 12.0}}.
    thermal_baseline : float, int, dict, or Series, optional
        Alias for baseline_rate. If provided, takes precedence over baseline_rate.
    baseline_rate_co2 : float, optional
        Explicit thermal baseline rate for CO2 (nmol/min). Overrides baseline_rate for CO2.
    baseline_rate_h2 : float, optional
        Explicit thermal baseline rate for H2 (nmol/min). Overrides baseline_rate for H2.
    min_current_A : float, optional
        Minimum current threshold in Amperes (default 1e-7 A / 0.1 uA) below which FE is 0%
        to avoid division by zero/open-circuit noise.

    Returns:
    --------
    rates_df_fe : DataFrame
        Updated rates DataFrame containing FE % columns for CO2, H2, and total FE,
        as well as baseline and net electrochemical rates.
    """
    FARADAY = 96485.0  # C / mol
    
    if rates_df is None or rates_df.empty:
        return pd.DataFrame()

    df_result = rates_df.copy()
    
    # Determine base thermal baseline input
    eff_baseline = thermal_baseline if thermal_baseline is not None else baseline_rate

    def _get_baseline_val(spec, species_name, row_data):
        if spec is None:
            return 0.0
        if isinstance(spec, (int, float, np.number)):
            return float(spec)
        if isinstance(spec, dict):
            # 1. Check for direct species key
            for key in [species_name, f"{species_name} (nmol/min)", species_name.lower(), species_name.upper()]:
                if key in spec:
                    val = spec[key]
                    if isinstance(val, (int, float, np.number)):
                        return float(val)
                    elif isinstance(val, dict):
                        # Nested temperature dict e.g. {'CO2': {80: 5.0, 100: 15.0}}
                        temp_val = row_data.get('Temp_C')
                        if temp_val is not None:
                            try:
                                t_float = float(temp_val)
                                for tk, tv in val.items():
                                    if abs(float(tk) - t_float) < 1.0:
                                        return float(tv)
                            except (ValueError, TypeError):
                                pass
            # 2. Check for temperature-keyed dict e.g. {80: 5.0, 100: 15.0}
            temp_val = row_data.get('Temp_C')
            if temp_val is not None:
                try:
                    t_float = float(temp_val)
                    for tk, tv in spec.items():
                        try:
                            if abs(float(tk) - t_float) < 1.0:
                                if isinstance(tv, dict):
                                    return float(tv.get(species_name, 0.0))
                                return float(tv)
                        except (ValueError, TypeError):
                            pass
                except (ValueError, TypeError):
                    pass
            return 0.0
        if callable(spec):
            return float(spec(row_data, species_name))
        if isinstance(spec, str) and spec in row_data:
            return float(row_data[spec])
        return 0.0

    if df_electronics is None or df_electronics.empty:
        df_result['Avg Current (mA)'] = 0.0
        df_result['FE_CO2 (%)'] = 0.0
        df_result['FE_H2 (%)'] = 0.0
        df_result['FE_Total (%)'] = 0.0
        return df_result

    co2_fe_list, h2_fe_list, total_fe_list, avg_current_mA_list = [], [], [], []
    co2_base_list, h2_base_list = [], []
    co2_electro_list, h2_electro_list = [], []

    for _, row in df_result.iterrows():
        t_gc = row['Time (hr)']
        
        # 1. Define sampling time window prior to GC injection
        t_start = t_gc - (gc_sampling_window_sec / 3600.0)
        
        # 2. Extract corresponding current points from electronics data
        mask = (df_electronics['Time (hr)'] >= t_start) & (df_electronics['Time (hr)'] <= t_gc)
        matching_currents = df_electronics.loc[mask, 'Current (A)'].abs() if 'Current (A)' in df_electronics.columns else pd.Series(dtype=float)
        
        # Fallback to the nearest electrochemical point if GC window is sparse
        if matching_currents.empty:
            if 'Time (hr)' in df_electronics.columns and 'Current (A)' in df_electronics.columns and not df_electronics['Current (A)'].dropna().empty:
                nearest_idx = (df_electronics['Time (hr)'] - t_gc).abs().idxmin()
                mean_current_A = abs(df_electronics.loc[nearest_idx, 'Current (A)'])
            else:
                mean_current_A = 0.0
        else:
            mean_current_A = matching_currents.mean()

        avg_current_mA_list.append(mean_current_A * 1e3)

        # 3. Determine thermal baseline to subtract from extensive rate
        co2_base = float(baseline_rate_co2) if baseline_rate_co2 is not None else _get_baseline_val(eff_baseline, 'CO2', row)
        h2_base = float(baseline_rate_h2) if baseline_rate_h2 is not None else _get_baseline_val(eff_baseline, 'H2', row)

        co2_rate_nmol_min = row.get('CO2 (nmol/min)', 0.0)
        h2_rate_nmol_min = row.get('H2 (nmol/min)', 0.0)

        # Net electrochemical rate (extensive rate minus thermal baseline)
        co2_rate_electro = co2_rate_nmol_min - co2_base
        h2_rate_electro = h2_rate_nmol_min - h2_base

        co2_base_list.append(round(co2_base, 3))
        h2_base_list.append(round(h2_base, 3))
        co2_electro_list.append(round(co2_rate_electro, 3))
        h2_electro_list.append(round(h2_rate_electro, 3))

        # 4. Calculate FE (%) for each species:
        # Rate (mol/s) = Rate_electro (nmol/min) * 1e-9 / 60
        # Theoretical Current I_theo = Rate_electro (mol/s) * n * F
        if mean_current_A > min_current_A:
            # CO2 Faradaic Efficiency
            co2_i_theo_A = (co2_rate_electro * 1e-9 / 60.0) * n_electrons * FARADAY
            co2_fe = (co2_i_theo_A / mean_current_A) * 100.0
            
            # H2 Faradaic Efficiency
            h2_i_theo_A = (h2_rate_electro * 1e-9 / 60.0) * n_electrons * FARADAY
            h2_fe = (h2_i_theo_A / mean_current_A) * 100.0
        else:
            co2_fe = 0.0
            h2_fe = 0.0

        co2_fe_list.append(round(co2_fe, 2))
        h2_fe_list.append(round(h2_fe, 2))
        total_fe_list.append(round(co2_fe + h2_fe, 2))

    df_result['Avg Current (mA)'] = avg_current_mA_list
    df_result['CO2_Baseline (nmol/min)'] = co2_base_list
    df_result['H2_Baseline (nmol/min)'] = h2_base_list
    df_result['CO2_Electro (nmol/min)'] = co2_electro_list
    df_result['H2_Electro (nmol/min)'] = h2_electro_list
    df_result['FE_CO2 (%)'] = co2_fe_list
    df_result['FE_H2 (%)'] = h2_fe_list
    df_result['FE_Total (%)'] = total_fe_list

    return df_result

