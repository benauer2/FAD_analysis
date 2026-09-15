import pandas as pd
import numpy as np
from scipy.stats import linregress, t
import matplotlib.pyplot as plt

from arrhenius_worker_files.arrhenius_config import get_step_at_time

import statsmodels.api as sm
import numpy as np

R_GAS = 8.314  # J / (mol K)

'''
#################################################################################################################
CALCULATES REACTION RATES BASED ON CONCENTRATIONS IN GC_df. GETS THEM ALIGNED WITH BIAS SCHEDULE FROM YAML FILE.
#################################################################################################################
'''
def calc_rxn_rates(GC_df, RXN_SCHEDULE, tot_flow_f, v_box_t, p_reac_f, GC_METHOD_INFO):
    #TEMPORARY SHORTCUT
    #C_1 = 0.00000000001457
    #C_2 = 0.000007638
    #H_1 = 0.0001665609

    C_1 = GC_METHOD_INFO["CO2 calibration factor_1 (FID, umol/(pA s))"]
    C_2 = GC_METHOD_INFO["CO2 calibration factor_2 (FID, umol/(pA s))"]
    H_1 = GC_METHOD_INFO["H2 calibration factor (TCD, umol/(uV s))"]


    if GC_df is None or GC_df.empty:
        return pd.DataFrame()
    
    rates_df = GC_df.copy()

    # Time tracking setup
    rates_df['Time (min)'] = [i * 30 for i in range(len(rates_df))]
    rates_df['Time (hr)'] = rates_df['Time (min)'] / 60
    rates_df['Trial data point'] = [f"Flow {i}" for i in range(len(rates_df))]

    # Map RXN_SCHEDULE properties dynamically based on time_window
    if RXN_SCHEDULE:
        step_series = rates_df['Time (hr)'].apply(
            lambda t: get_step_at_time(t, RXN_SCHEDULE=RXN_SCHEDULE)
        )
        
        # Safely check isinstance(s, dict) before accessing keys
        rates_df['Bias (V)'] = step_series.apply(
            lambda s: s["we_volt"] if isinstance(s, dict) and "we_volt" in s else 0.0
        )
        rates_df['Temp_C'] = step_series.apply(
            lambda s: s["rxr_t"] if isinstance(s, dict) and "rxr_t" in s else 0.0
        )
        rates_df['N2_sccm'] = step_series.apply(
            lambda s: s["n2_flow"] if isinstance(s, dict) and "n2_flow" in s else 0.0
        )
        rates_df['Step_Start_hr'] = step_series.apply(
            lambda s: s["time_window"][0] if isinstance(s, dict) and "time_window" in s else 0.0
        )
        
        # Relative time inside current condition step
        rates_df['Step_Rel_Time_hr'] = rates_df['Time (hr)'] - rates_df['Step_Start_hr']
        
        # Label string matching compute_schedule_summary requirements
        def format_label(s):
            if isinstance(s, dict) and "rxr_t" in s and "n2_flow" in s:
                t_val = s["rxr_t"]
                f_val = s["n2_flow"]
                t_str = int(t_val) if isinstance(t_val, (int, float)) and float(t_val).is_integer() else t_val
                f_str = int(f_val) if isinstance(f_val, (int, float)) and float(f_val).is_integer() else f_val
                return f"{t_str}°C - {f_str} sccm N2"
            return "0°C - 0 sccm N2"

        rates_df['Condition_Step'] = step_series.apply(format_label)
    else:
        rates_df['Bias (V)'] = 0.0
        rates_df['Temp_C'] = 0.0
        rates_df['N2_sccm'] = 0.0
        rates_df['Step_Rel_Time_hr'] = rates_df['Time (hr)']
        rates_df['Condition_Step'] = "0°C - 0 sccm N2"

    # Calculate normalization parameters
    denom_p = 14.696 + float(p_reac_f)
    if denom_p == 0:
        denom_p = 14.696

    norm_vol_factor = (tot_flow_f * (273.1 + float(v_box_t)) / 300 * 14.696 / denom_p) * 1000

    # Ensure species columns are numeric
    for species in ['CO2', 'FA', 'H2', 'CO']:
        if species in rates_df.columns:
            rates_df[species] = pd.to_numeric(rates_df[species], errors='coerce').fillna(0.0)
        else:
            rates_df[species] = 0.0
        
    rates_df['CO2 (nmol/min)'] = (C_1*rates_df['CO2']**2 +  C_2*rates_df['CO2']) * norm_vol_factor
    rates_df['H2 (nmol/min)'] = H_1*rates_df['H2'] * norm_vol_factor
    rates_df['CO (nmol/min)'] =(C_1*rates_df['CO']**2 +  C_2*rates_df['CO']) * norm_vol_factor

    fa_denom = (denom_p / 14.696 * 101325 / 8.314 / (273.15 + float(v_box_t)))
    rates_df['FA mol frac'] = 100 * (C_1*rates_df['FA']**2 +  C_2*rates_df['FA']) / (fa_denom) if fa_denom != 0 else 0.0

    return rates_df


def assign_schedule_parameters(df_gc, RXN_SCHEDULE):
    """
    Assigns step boundaries, dynamic N2 flow, relative step time, 
    and averaging windows to each GC data row.
    """
    if df_gc is None or df_gc.empty:
        return pd.DataFrame()

    df = df_gc.copy()
    if "Time (hr)" not in df.columns:
        df["Time (hr)"] = [i * 0.5 for i in range(len(df))]
    
    if not RXN_SCHEDULE:
        df["N2_flow_sccm"] = 0.0
        df["Condition_Step"] = "Default"
        df["Step_Rel_Time_hr"] = df["Time (hr)"]
        df["Avg_Window"] = [[0,1]] * len(df)
        return df

    current_time = 0.0
    schedule_intervals = []
    
    for phase in RXN_SCHEDULE:
        duration = float(phase.get("duration_hours", 0.0))
        start_time = current_time
        end_time = current_time + duration
        schedule_intervals.append({
            "start": start_time,
            "end": end_time,
            "temp": phase.get("temperature", 0.0),
            "n2_flow": float(phase.get("mfc_rate", 0.0)),
            "avg_window": phase.get("avg_window", [0, duration]),
            "label": f"{phase.get('temperature', 0.0)}°C - {phase.get('mfc_rate', 0.0)} sccm N2"
        })
        current_time = end_time

    n2_flows, condition_steps, rel_times, avg_windows = [], [], [], []
    
    for _, row in df.iterrows():
        t = row["Time (hr)"]
        matched = False
        for interval in schedule_intervals:
            if interval["start"] <= t <= interval["end"]:
                n2_flows.append(interval["n2_flow"])
                condition_steps.append(interval["label"])
                rel_times.append(t - interval["start"])
                avg_windows.append(interval["avg_window"])
                matched = True
                break
        if not matched:
            target = schedule_intervals[0] if t < schedule_intervals[0]["start"] else schedule_intervals[-1]
            n2_flows.append(target["n2_flow"])
            condition_steps.append(target["label"])
            rel_times.append(t - target["start"])
            avg_windows.append(target["avg_window"])

    df["N2_flow_sccm"] = n2_flows
    df["Condition_Step"] = condition_steps
    df["Step_Rel_Time_hr"] = rel_times
    df["Avg_Window"] = avg_windows
    return df

def calc_rxn_rates_dynamic(GC_df, RXN_SCHEDULE, TRIAL_INFO, GC_METHOD_INFO):
    """
    Calculates gas evolution rates, FA mol fraction, and retains relative timing columns.
    """
    if GC_df is None or GC_df.empty:
        return pd.DataFrame()
    
    TRIAL_INFO = TRIAL_INFO or {}
    GC_METHOD_INFO = GC_METHOD_INFO or {}

    rates_df = assign_schedule_parameters(GC_df, RXN_SCHEDULE)
    
    fa_pct = float(TRIAL_INFO.get("Percent (%) FA in vapor stream", 4.0))
    h2_flow = float(TRIAL_INFO.get("H2 flow (sccm)", 0.0))
    p_reac_f = float(TRIAL_INFO.get("P reactor (psig)", 0.0))
    v_box_t = float(GC_METHOD_INFO.get("Valve box T (deg C)", 100.0))

    fa_frac_denom = 1.0 - (fa_pct / 100.0)
    if fa_frac_denom <= 0:
        fa_frac_denom = 0.01

    denom_p = p_reac_f + 14.696
    if denom_p == 0:
        denom_p = 14.696

    rates_df["Tot_Flow_sccm"] = (rates_df["N2_flow_sccm"] + h2_flow) / fa_frac_denom
    rates_df["norm_vol_factor"] = (
        rates_df["Tot_Flow_sccm"] * (273.15 + v_box_t) / 300.0 * 14.696 / denom_p
    ) * 1000.0

    for col in ["CO2", "H2"]:
        if col in rates_df.columns:
            series_num = pd.to_numeric(rates_df[col], errors="coerce").fillna(0.0)
            rates_df[f"{col} (nmol/min)"] = series_num * rates_df["norm_vol_factor"]
        else:
            rates_df[f"{col} (nmol/min)"] = 0.0

    if "FA" in rates_df.columns:
        fa_series = pd.to_numeric(rates_df["FA"], errors="coerce").fillna(0.0)
        denom = (denom_p / 14.696 * 101325.0 / 8.314 / (273.15 + v_box_t))
        rates_df["FA mol frac"] = (100.0 * fa_series / denom) if denom != 0 else 0.0
    else:
        rates_df["FA mol frac"] = 0.0

    return rates_df


import pandas as pd
import numpy as np

def compute_schedule_summary(rates_df, RXN_SCHEDULE):
    if rates_df is None or rates_df.empty or not RXN_SCHEDULE:
        return pd.DataFrame()

    summary_data = []
    
    for phase in RXN_SCHEDULE:
        if not isinstance(phase, dict) or "time_window" not in phase:
            continue
            
        temp_c = phase["rxr_t"]
        mfc_rate = phase["n2_flow"]
        if pd.isna(temp_c):
            continue

        start_time_hr, end_time_hr = phase["time_window"]
        avg_win_start, avg_win_end = phase["avg_window"]

        sub_df = rates_df[
            (rates_df["Time (hr)"] >= start_time_hr) & 
            (rates_df["Time (hr)"] <= end_time_hr)
        ]

        if not sub_df.empty:
            step_rel_time = sub_df["Time (hr)"] - start_time_hr
            steady_df = sub_df[(step_rel_time >= avg_win_start) & (step_rel_time <= avg_win_end)]
            if steady_df.empty:
                steady_df = sub_df
        else:
            steady_df = pd.DataFrame()

        def _get_mean_std(df_in, col):
            if col in df_in.columns and not df_in[col].empty:
                s = pd.to_numeric(df_in[col], errors="coerce").dropna()
                if len(s) > 0:
                    mean_val = float(s.mean())
                    std_val = float(s.std()) if len(s) > 1 else 0.0
                    return mean_val, (0.0 if np.isnan(std_val) else std_val)
            return 0.0, 0.0
            
        co2_avg, co2_std = _get_mean_std(steady_df, "CO2 (nmol/min)")
        h2_avg, h2_std = _get_mean_std(steady_df, "H2 (nmol/min)")
        co_avg, co_std = _get_mean_std(steady_df, "CO (nmol/min)")

        summary_data.append({
            "Run": phase["run"],
            "Rxr_T_C": temp_c,
            "N2_flow_sccm": mfc_rate,
            "FA_flow_ul.min": phase["fa_flow"],
            "WE_Voltage": phase["we_volt"],
            "Avg_Window": f"{avg_win_start:.1f}-{avg_win_end:.1f}h",
            "Run_Time": f"[{start_time_hr:.1f}, {end_time_hr:.1f}]",
            # Numeric columns for Arrhenius calculations
            "CO2_avg": co2_avg, "CO2_std": co2_std,
            "H2_avg": h2_avg,   "H2_std": h2_std,
            "CO_avg": co_avg,   "CO_std": co_std,
            # String columns for clean UI table display
            "CO2 (nmol/min)": f"{co2_avg:.2f} +/- {co2_std:.2f}",
            "H2 (nmol/min)": f"{h2_avg:.2f} +/- {h2_std:.2f}",
            "CO (nmol/min)": f"{co_avg:.2f} +/- {co_std:.2f}",
        })
        
    return pd.DataFrame(summary_data)



import numpy as np
import pandas as pd
from scipy.stats import linregress, t

R_GAS = 8.31446261815324

def _parse_voltage_key(val):
    if pd.isna(val) or val == "":
        return 0.0
    val_str = str(val).strip()
    if val_str.upper() == "OCP":
        return "OCP"
    try:
        return float(val_str)
    except ValueError:
        return "OCP"

def perform_arrhenius_analysis(summary_df):
    if summary_df is None or summary_df.empty:
        return {}

    df = summary_df.copy()

    df["Temp_C"] = pd.to_numeric(df["Rxr_T_C"], errors="coerce")
    df["Temp_K"] = df["Temp_C"] + 273.15
    df["WE_Voltage_Key"] = df["WE_Voltage"].apply(_parse_voltage_key)

    results = {}

    for voltage_key, v_df in df.groupby("WE_Voltage_Key"):
        results[voltage_key] = {}

        for species in ["CO2", "H2", "CO"]:
            avg_col = f"{species}_avg"
            std_col = f"{species}_std"

            if avg_col not in v_df.columns:
                continue

            valid_df = v_df[
                (v_df[avg_col] > 0) & 
                (v_df["Temp_C"] != 80) & 
                (v_df["Temp_K"] > 0)
            ].copy()

            if len(valid_df) < 2:
                continue

            inv_T = 1.0 / valid_df["Temp_K"].values
            ln_rate = np.log(valid_df[avg_col].values)
            rate_std = valid_df[std_col].values if std_col in valid_df.columns else None

            finite_mask = np.isfinite(inv_T) & np.isfinite(ln_rate)
            n_pts = np.count_nonzero(finite_mask)
            if n_pts < 2:
                continue

            res = linregress(inv_T[finite_mask], ln_rate[finite_mask])
            slope, intercept, r_val, p_val, slope_err = (
                res.slope, res.intercept, res.rvalue, res.pvalue, res.stderr
            )
            intercept_err = getattr(res, 'intercept_stderr', np.nan)

            Ea_kJ = -slope * R_GAS / 1000.0
            A_factor = np.exp(intercept)
            r2 = r_val ** 2

            Ea_err_kJ = slope_err * R_GAS / 1000.0
            A_err = A_factor * intercept_err if intercept_err is not None and np.isfinite(intercept_err) else np.nan

            df_deg = n_pts - 2
            if df_deg > 0:
                t_crit = t.ppf(0.975, df=df_deg)
                Ea_ci95_kJ = Ea_err_kJ * t_crit
                A_ci95 = A_err * t_crit
            else:
                Ea_ci95_kJ = np.nan
                A_ci95 = np.nan

            results[voltage_key][species] = {
                "Ea_kJ": Ea_kJ,
                "Ea_err_kJ": Ea_err_kJ,
                "Ea_ci95_kJ": Ea_ci95_kJ,
                "A": A_factor,
                "A_err": A_err,
                "A_ci95": A_ci95,
                "R2": r2,
                "p_val": p_val,
                "slope": slope,
                "slope_err": slope_err,
                "intercept": intercept,
                "intercept_err": intercept_err,
                "inv_T": inv_T[finite_mask],
                "ln_rate": ln_rate[finite_mask],
                "rate_std": rate_std[finite_mask] if rate_std is not None else None,
                "Temp_C": valid_df["Temp_C"].values[finite_mask]
            }

    return results




def background_stats(df):
    """Calculates mean and standard deviation for key gas evolution rates."""
    if df is None or df.empty:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    
    def _col_stats(col_name):
        if col_name in df.columns:
            s = pd.to_numeric(df[col_name], errors='coerce').dropna()
            if len(s) > 0:
                m=float(s.mean())
                std = float(s.std()) if len(s) > 1 else 0.0
                return m, (0.0 if np.isnan(std) else std)
        return 0.0, 0.0
    
    avg_co2, std_co2 = _col_stats('CO2 (nmol/min)')
    avg_h2, std_h2   = _col_stats('H2 (nmol/min)')
    avg_co, std_co   = _col_stats('CO (nmol/min)')
    return avg_co2, std_co2, avg_h2, std_h2, avg_co, std_co


