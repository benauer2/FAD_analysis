"""
Residence Time Distribution (RTD) Analysis Module for Reactor & Tracer Step Testing.
Author: Ben Auer
"""
import tkinter as tk
from tkinter import simpledialog, ttk
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.special import gammainc, gamma
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

def prompt_step_change_parameters(RXN_SCHEDULE=None, rates_df=None, default_species="CO"):
    """
    Prompts the user directly in the terminal to configure or confirm RTD step change parameters:
    - Step change time (in hours)
    - Tracer chemical species / channel
    - Step direction (Up, Down, Auto)
    """
    detected_step_hr = 0.5
    
    # Auto-detect step time from RXN_SCHEDULE
    if RXN_SCHEDULE and len(RXN_SCHEDULE) >= 2:
        step2 = RXN_SCHEDULE[1]
        if isinstance(step2, dict) and "time_window" in step2:
            detected_step_hr = float(step2["time_window"][0])
        elif isinstance(step2, dict) and "dur_hrs" in RXN_SCHEDULE[0]:
            detected_step_hr = float(RXN_SCHEDULE[0]["dur_hrs"])

    # Collect available candidate columns
    available_species = []
    if rates_df is not None and not rates_df.empty:
        candidate_cols = [
            "H2 (nmol/min)", "CO (nmol/min)", "CO2 (nmol/min)", "FA mol frac",
            "H2", "CO", "CO2", "FA", "Signal"
        ]
        available_species = [c for c in candidate_cols if c in rates_df.columns]
    
    if not available_species:
        available_species = ["H2 (nmol/min)", "CO (nmol/min)", "CO2 (nmol/min)", "FA mol frac"]

    # Pick the best default species (look for species with largest signal variation)
    best_spec = available_species[0]
    max_range = -1.0
    for s in available_species:
        if rates_df is not None and s in rates_df.columns:
            s_num = pd.to_numeric(rates_df[s], errors="coerce").dropna()
            if len(s_num) > 1:
                rng = s_num.max() - s_num.min()
                if rng > max_range:
                    max_range = rng
                    best_spec = s

    # Print Interactive Terminal Menu
    print("\n" + "=" * 70)
    print("        RESIDENCE TIME DISTRIBUTION (RTD) - STEP CONFIGURATION")
    print("=" * 70)
    print(f"  [Auto-Detected Step Time]: {detected_step_hr:.3f} hr ({detected_step_hr * 60:.1f} min)")
    print("\n  Available Tracer Channels:")
    for idx, sp in enumerate(available_species, start=1):
        is_default = " (Recommended / Active)" if sp == best_spec else ""
        print(f"    [{idx}] {sp}{is_default}")
    print("-" * 70)

    # 1. Step time prompt
    try:
        raw_t = input(f"  >> Step change time in hours [Press Enter for {detected_step_hr:.3f} hr]: ").strip()
        step_time_hr = float(raw_t) if raw_t else detected_step_hr
    except (ValueError, EOFError, KeyboardInterrupt):
        step_time_hr = detected_step_hr

    # 2. Species prompt
    try:
        raw_sp = input(f"  >> Select tracer channel [1-{len(available_species)}, default: '{best_spec}']: ").strip()
        if raw_sp.isdigit() and 1 <= int(raw_sp) <= len(available_species):
            tracer_col = available_species[int(raw_sp) - 1]
        elif raw_sp in available_species:
            tracer_col = raw_sp
        else:
            tracer_col = best_spec
    except (EOFError, KeyboardInterrupt):
        tracer_col = best_spec

    # 3. Direction prompt
    try:
        raw_dir = input("  >> Step direction [1: Auto-Detect, 2: Step-Up, 3: Step-Down] [default: Auto]: ").strip()
        if raw_dir == "2":
            step_direction = "up"
        elif raw_dir == "3":
            step_direction = "down"
        else:
            step_direction = "auto"
    except (EOFError, KeyboardInterrupt):
        step_direction = "auto"

    print("=" * 70)
    print(f"  [Confirmed]: Step Time = {step_time_hr:.3f} hr | Tracer = '{tracer_col}' | Direction = {step_direction.capitalize()}")
    print("=" * 70 + "\n")

    return {
        "step_time_hr": step_time_hr,
        "tracer_col": tracer_col,
        "step_direction": step_direction,
        "baseline_points": 2,
        "status": "ok"
    }

    return user_inputs


def calculate_rtd_step_response(
    df, 
    tracer_col, 
    t_step_hr=0.5, 
    step_direction="auto", 
    baseline_pts=2,
    tot_flow_sccm=None,
    rxr_t_c=None,
    p_reac_psig=0.0,
    v_geom_cm3=None
):
    """
    Computes F-curve F(t), E-curve E(t), Mean Residence Time, Variance, 
    Reactor Volume (actual & standard), and Dispersion parameters.
    """
    if df is None or df.empty or tracer_col not in df.columns:
        raise ValueError(f"Valid data frame containing '{tracer_col}' is required for RTD analysis.")

    data = df.copy()
    if "Time (hr)" not in data.columns:
        data["Time (hr)"] = [i * 0.5 for i in range(len(data))]
    
    data = data.sort_values("Time (hr)").reset_index(drop=True)
    time_hr = data["Time (hr)"].values
    conc_raw = pd.to_numeric(data[tracer_col], errors="coerce").fillna(0.0).values

    # Determine Pre-Step Baseline C0 and uncertainty
    pre_mask = time_hr <= t_step_hr
    if np.sum(pre_mask) > 0:
        c0_pts = conc_raw[pre_mask][-baseline_pts:]
        c0 = float(np.mean(c0_pts))
        c0_err = float(np.std(c0_pts, ddof=1)) if len(c0_pts) > 1 else float(0.01 * c0)
    else:
        c0 = float(conc_raw[0])
        c0_err = float(0.01 * c0)

    # Post-Step steady state plateau C_inf and uncertainty
    post_mask = time_hr >= t_step_hr
    if np.sum(post_mask) < 2:
        raise ValueError(f"Insufficient data points after step change time ({t_step_hr:.2f} hr).")

    post_time = time_hr[post_mask]
    post_conc = conc_raw[post_mask]

    # Plateau estimation from last points
    tail_pts = max(1, min(3, len(post_conc)))
    c_inf_pts = post_conc[-tail_pts:]
    c_inf = float(np.mean(c_inf_pts))
    c_inf_err = float(np.std(c_inf_pts, ddof=1)) if len(c_inf_pts) > 1 else float(0.01 * c_inf)

    # Detect step direction if auto
    if step_direction == "auto":
        step_direction = "up" if c_inf >= c0 else "down"

    # Compute F(t)
    delta_c = c_inf - c0
    if abs(delta_c) < 1e-12:
        delta_c = 1.0

    if step_direction == "up":
        f_raw = (post_conc - c0) / delta_c
    else:
        f_raw = 1.0 - ((post_conc - c_inf) / (-delta_c))

    # Clean & Bound F(t) between [0, 1]
    f_curve = np.clip(f_raw, 0.0, 1.0)
    
    # Force F(t=0) = 0 for step up
    t_rel_hr = post_time - t_step_hr
    t_rel_min = t_rel_hr * 60.0

    # Guarantee monotonic initial point at t=0
    if t_rel_min[0] > 0:
        t_rel_min = np.insert(t_rel_min, 0, 0.0)
        t_rel_hr = np.insert(t_rel_hr, 0, 0.0)
        f_curve = np.insert(f_curve, 0, 0.0)
        post_conc_extended = np.insert(post_conc, 0, c0)
    else:
        f_curve[0] = 0.0
        post_conc_extended = post_conc

    # Compute E(t) via numerical gradient & spline
    e_curve = np.gradient(f_curve, t_rel_min)
    e_curve = np.clip(e_curve, 0.0, None)
    
    # Normalize E(t) so area = 1
    trapz_area = np.trapezoid(e_curve, t_rel_min) if hasattr(np, 'trapezoid') else np.trapz(e_curve, t_rel_min)
    if trapz_area > 0:
        e_curve = e_curve / trapz_area

    # Calculate Moments: Mean Residence Time (t_mean) and Variance (sigma^2)
    t_mean_min = np.trapezoid(t_rel_min * e_curve, t_rel_min) if hasattr(np, 'trapezoid') else np.trapz(t_rel_min * e_curve, t_rel_min)
    if t_mean_min <= 0 or not np.isfinite(t_mean_min):
        t_mean_min = np.trapezoid(1.0 - f_curve, t_rel_min) if hasattr(np, 'trapezoid') else np.trapz(1.0 - f_curve, t_rel_min)

    t_mean_hr = t_mean_min / 60.0

    # Variance: sigma^2 = int((t - t_mean)^2 * E(t) dt)
    sigma2_min2 = np.trapezoid((t_rel_min - t_mean_min)**2 * e_curve, t_rel_min) if hasattr(np, 'trapezoid') else np.trapz((t_rel_min - t_mean_min)**2 * e_curve, t_rel_min)
    sigma2_min2 = max(1e-6, sigma2_min2)
    sigma_min = np.sqrt(sigma2_min2)

    # Uncertainty estimation for Mean Residence Time & Moments
    dt_sample_min = np.median(np.diff(t_rel_min)) if len(t_rel_min) > 1 else 5.0
    t_discretization_err_min = dt_sample_min / np.sqrt(12.0)  # Uniform quantization error
    n_pts_trans = max(2, int(np.sum((f_curve > 0.02) & (f_curve < 0.98))))
    t_mean_se_min = sigma_min / np.sqrt(n_pts_trans)
    t_mean_err_min = float(np.sqrt(t_mean_se_min**2 + t_discretization_err_min**2))
    t_mean_err_hr = t_mean_err_min / 60.0

    sigma2_err_min2 = float(sigma2_min2 * (2.0 * t_mean_err_min / max(0.1, t_mean_min)))
    sigma_err_min = float(0.5 * (sigma2_err_min2 / max(1e-4, sigma_min)))

    # Dimensionless Variance: sigma_theta^2 = sigma^2 / t_mean^2
    sigma_theta2 = sigma2_min2 / (t_mean_min**2) if t_mean_min > 0 else 1.0
    sigma_theta2_err = float(sigma_theta2 * np.sqrt((sigma2_err_min2 / max(1e-6, sigma2_min2))**2 + (2.0 * t_mean_err_min / max(0.1, t_mean_min))**2))

    # Equivalent Tanks-in-Series (TIS) parameter: N = 1 / sigma_theta^2
    N_tanks = 1.0 / sigma_theta2 if sigma_theta2 > 0 else 1.0
    N_tanks_err = float(N_tanks * (sigma_theta2_err / max(1e-4, sigma_theta2)))

    # Vessel Dispersion Number: D/(uL) ~ sigma_theta2 / 2 for small dispersion
    dispersion_number = sigma_theta2 / 2.0
    dispersion_err = float(dispersion_number * (sigma_theta2_err / max(1e-4, sigma_theta2)))
    peclet_number = 1.0 / dispersion_number if dispersion_number > 0 else 0.0
    peclet_err = float(peclet_number * (dispersion_err / max(1e-4, dispersion_number))) if dispersion_number > 0 else 0.0

    # Non-linear Curve Fitting: Tanks In Series F(t) Model
    def tis_model(t, N, tau):
        return gammainc(N, N * t / tau)

    try:
        popt, pcov = curve_fit(
            tis_model, 
            t_rel_min, 
            f_curve, 
            p0=[max(1.0, N_tanks), max(1.0, t_mean_min)],
            bounds=([0.5, 0.1], [500.0, 10000.0])
        )
        n_fit, tau_fit = float(popt[0]), float(popt[1])
        perr = np.sqrt(np.diag(pcov)) if pcov is not None else [0.0, 0.0]
        n_fit_err = float(perr[0]) if np.isfinite(perr[0]) and perr[0] > 0 else float(n_fit * 0.05)
        tau_fit_err = float(perr[1]) if np.isfinite(perr[1]) and perr[1] > 0 else float(tau_fit * 0.05)
        f_tis_fit = tis_model(t_rel_min, n_fit, tau_fit)
        
        ss_res = np.sum((f_curve - f_tis_fit)**2)
        ss_tot = np.sum((f_curve - np.mean(f_curve))**2)
        r2_tis = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0
    except Exception:
        n_fit, tau_fit = float(N_tanks), float(t_mean_min)
        n_fit_err = float(N_tanks_err)
        tau_fit_err = float(t_mean_err_min)
        f_tis_fit = tis_model(t_rel_min, n_fit, tau_fit)
        r2_tis = np.nan

    # -------------------------------------------------------------
    # REACTOR & SYSTEM VOLUME CALCULATIONS WITH ERROR PROPAGATION
    # -------------------------------------------------------------
    if tot_flow_sccm is None or tot_flow_sccm <= 0:
        if "Total_Flow_sccm" in data.columns and (data.loc[post_mask, "Total_Flow_sccm"] > 0).any():
            tot_flow_sccm = float(data.loc[post_mask, "Total_Flow_sccm"].mean())
            tot_flow_err_sccm = float(data.loc[post_mask, "Total_Flow_sccm"].std())
        elif "N2_sccm" in data.columns:
            tot_flow_sccm = float(data.loc[post_mask, "N2_sccm"].mean())
            tot_flow_err_sccm = float(data.loc[post_mask, "N2_sccm"].std())
        else:
            tot_flow_sccm = 15.0
            tot_flow_err_sccm = 0.15
    else:
        tot_flow_err_sccm = float(tot_flow_sccm * 0.01)

    if not np.isfinite(tot_flow_err_sccm) or tot_flow_err_sccm <= 0:
        tot_flow_err_sccm = float(max(0.05, tot_flow_sccm * 0.01))

    if rxr_t_c is None:
        if "Temp_C" in data.columns and (data.loc[post_mask, "Temp_C"] > 0).any():
            rxr_t_c = float(data.loc[post_mask, "Temp_C"].mean())
            rxr_t_err_c = float(data.loc[post_mask, "Temp_C"].std())
        else:
            rxr_t_c = 125.0
            rxr_t_err_c = 0.5
    else:
        rxr_t_c = float(rxr_t_c)
        rxr_t_err_c = 0.5

    if not np.isfinite(rxr_t_err_c) or rxr_t_err_c <= 0:
        rxr_t_err_c = 0.5

    t_lines_c = 20.0  # Room temperature for MFCs, connective tubing, and valve manifolds
    t_lines_err_c = 1.0  # Room temp fluctuation +/- 1.0 deg C

    denom_p = float(p_reac_psig) + 14.696
    if denom_p <= 0:
        denom_p = 14.696

    # 1. Ambient / Line conditions (20 deg C, room temperature ~ 15 mL/min)
    q_ambient_ml_min = tot_flow_sccm * ((273.15 + t_lines_c) / 293.15) * (14.696 / denom_p)
    q_ambient_rel_err = np.sqrt((tot_flow_err_sccm / max(0.1, tot_flow_sccm))**2 + (t_lines_err_c / (273.15 + t_lines_c))**2)
    q_ambient_err_ml_min = float(q_ambient_ml_min * q_ambient_rel_err)

    v_ambient_ml = q_ambient_ml_min * t_mean_min
    v_ambient_rel_err = np.sqrt(q_ambient_rel_err**2 + (t_mean_err_min / max(0.01, t_mean_min))**2)
    v_ambient_err_ml = float(v_ambient_ml * v_ambient_rel_err)
    v_ambient_ul = v_ambient_ml * 1000.0
    v_ambient_err_ul = v_ambient_err_ml * 1000.0

    # 2. Hot reactor zone conditions (at T_rxr, e.g. 125 deg C ~ 20 mL/min)
    q_rxr_hot_ml_min = tot_flow_sccm * ((273.15 + float(rxr_t_c)) / 293.15) * (14.696 / denom_p)
    q_rxr_hot_rel_err = np.sqrt((tot_flow_err_sccm / max(0.1, tot_flow_sccm))**2 + (rxr_t_err_c / (273.15 + float(rxr_t_c)))**2)
    q_rxr_hot_err_ml_min = float(q_rxr_hot_ml_min * q_rxr_hot_rel_err)

    v_hot_rxr_ml = q_rxr_hot_ml_min * t_mean_min
    v_hot_rxr_rel_err = np.sqrt(q_rxr_hot_rel_err**2 + (t_mean_err_min / max(0.01, t_mean_min))**2)
    v_hot_rxr_err_ml = float(v_hot_rxr_ml * v_hot_rxr_rel_err)
    v_hot_rxr_ul = v_hot_rxr_ml * 1000.0
    v_hot_rxr_err_ul = v_hot_rxr_err_ml * 1000.0

    # 3. Standard fluid volume (scc)
    v_standard_scc = tot_flow_sccm * t_mean_min
    v_standard_err_scc = float(v_standard_scc * np.sqrt((tot_flow_err_sccm / max(0.1, tot_flow_sccm))**2 + (t_mean_err_min / max(0.01, t_mean_min))**2))

    # 4. Model fitted volume at ambient line conditions
    v_model_ml = q_ambient_ml_min * tau_fit
    v_model_err_ml = float(v_model_ml * np.sqrt(q_ambient_rel_err**2 + (tau_fit_err / max(0.01, tau_fit))**2))

    # Geometric / Dead space comparison if geometric volume is supplied
    dead_volume_pct = np.nan
    dead_volume_err_pct = np.nan
    ghsv_hr = np.nan
    ghsv_err_hr = np.nan
    if v_geom_cm3 is not None and v_geom_cm3 > 0:
        dead_volume_pct = max(0.0, (1.0 - (v_ambient_ml / v_geom_cm3)) * 100.0)
        dead_volume_err_pct = float((v_ambient_err_ml / v_geom_cm3) * 100.0)
        ghsv_hr = (tot_flow_sccm * 60.0) / v_geom_cm3
        ghsv_err_hr = (tot_flow_err_sccm * 60.0) / v_geom_cm3

    rtd_results = {
        "tracer_col": tracer_col,
        "t_step_hr": t_step_hr,
        "step_direction": step_direction,
        "c0": c0,
        "c0_err": c0_err,
        "c_inf": c_inf,
        "c_inf_err": c_inf_err,
        "time_hr_full": time_hr,
        "conc_full": conc_raw,
        "t_rel_min": t_rel_min,
        "t_rel_hr": t_rel_hr,
        "post_conc": post_conc_extended,
        "f_curve": f_curve,
        "e_curve": e_curve,
        "t_mean_min": t_mean_min,
        "t_mean_err_min": t_mean_err_min,
        "t_mean_hr": t_mean_hr,
        "t_mean_err_hr": t_mean_err_hr,
        "sigma2_min2": sigma2_min2,
        "sigma2_err_min2": sigma2_err_min2,
        "sigma_min": sigma_min,
        "sigma_err_min": sigma_err_min,
        "sigma_theta2": sigma_theta2,
        "sigma_theta2_err": sigma_theta2_err,
        "N_tanks": N_tanks,
        "N_tanks_err": N_tanks_err,
        "N_tanks_fit": n_fit,
        "N_tanks_fit_err": n_fit_err,
        "tau_fit_min": tau_fit,
        "tau_fit_err_min": tau_fit_err,
        "r2_tis": r2_tis,
        "dispersion_number": dispersion_number,
        "dispersion_err": dispersion_err,
        "peclet_number": peclet_number,
        "peclet_err": peclet_err,
        "tis_fit_curve": f_tis_fit,
        # Flow rates and Volumes with error
        "tot_flow_sccm": tot_flow_sccm,
        "tot_flow_err_sccm": tot_flow_err_sccm,
        "t_lines_c": t_lines_c,
        "t_lines_err_c": t_lines_err_c,
        "rxr_t_c": rxr_t_c,
        "rxr_t_err_c": rxr_t_err_c,
        "p_reac_psig": p_reac_psig,
        "q_ambient_ml_min": q_ambient_ml_min,
        "q_ambient_err_ml_min": q_ambient_err_ml_min,
        "q_rxr_hot_ml_min": q_rxr_hot_ml_min,
        "q_rxr_hot_err_ml_min": q_rxr_hot_err_ml_min,
        "q_actual_ml_min": q_ambient_ml_min,
        "q_actual_err_ml_min": q_ambient_err_ml_min,
        "v_ambient_ml": v_ambient_ml,
        "v_ambient_err_ml": v_ambient_err_ml,
        "v_ambient_ul": v_ambient_ul,
        "v_ambient_err_ul": v_ambient_err_ul,
        "v_actual_ml": v_ambient_ml,
        "v_actual_err_ml": v_ambient_err_ml,
        "v_actual_ul": v_ambient_ul,
        "v_actual_err_ul": v_ambient_err_ul,
        "v_hot_rxr_ml": v_hot_rxr_ml,
        "v_hot_rxr_err_ml": v_hot_rxr_err_ml,
        "v_hot_rxr_ul": v_hot_rxr_ul,
        "v_hot_rxr_err_ul": v_hot_rxr_err_ul,
        "v_standard_scc": v_standard_scc,
        "v_standard_err_scc": v_standard_err_scc,
        "v_model_ml": v_model_ml,
        "v_model_err_ml": v_model_err_ml,
        "v_geom_cm3": v_geom_cm3,
        "dead_volume_pct": dead_volume_pct,
        "dead_volume_err_pct": dead_volume_err_pct,
        "ghsv_hr": ghsv_hr,
        "ghsv_err_hr": ghsv_err_hr
    }

    return rtd_results


def plot_rtd_analysis(rtd_results, plots_dir):
    """
    Generates high-resolution RTD plots:
    1. Overall Step Change & Raw Tracer Concentration vs Time
    2. Cumulative Step Response F-curve F(t) with Tanks-in-Series Model Fit
    3. Residence Time Distribution Density E-curve E(t)
    4. Dimensionless RTD E(theta) vs theta
    """
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    generated_plots = {}

    tracer = rtd_results["tracer_col"]
    t_step_hr = rtd_results["t_step_hr"]
    t_mean_min = rtd_results["t_mean_min"]
    
    # -----------------------------------------------------------------
    # Plot 1: Tracer Concentration Step Profile
    # -----------------------------------------------------------------
    fig1, ax1 = plt.subplots(figsize=(10, 4.5))
    ax1.plot(
        rtd_results["time_hr_full"], 
        rtd_results["conc_full"], 
        marker="o", markersize=6, color="navy", markeredgecolor="black", markeredgewidth=1,
        linestyle="-", lw=1.8, label=f"Measured {tracer}"
    )
    ax1.axvline(t_step_hr, color="crimson", linestyle="--", lw=2, label=f"Step Change (t = {t_step_hr:.2f} h)")
    
    # Baseline & Plateau with uncertainty bands
    c0_val = rtd_results["c0"]
    c0_err = rtd_results.get("c0_err", 0.0)
    cinf_val = rtd_results["c_inf"]
    cinf_err = rtd_results.get("c_inf_err", 0.0)

    ax1.axhline(c0_val, color="gray", linestyle=":", lw=1.2, label=f"C0 Baseline ({c0_val:.2f} ± {c0_err:.2f})")
    if c0_err > 0:
        ax1.axhspan(c0_val - c0_err, c0_val + c0_err, color="gray", alpha=0.15)

    ax1.axhline(cinf_val, color="forestgreen", linestyle=":", lw=1.2, label=f"C_inf Plateau ({cinf_val:.2f} ± {cinf_err:.2f})")
    if cinf_err > 0:
        ax1.axhspan(cinf_val - cinf_err, cinf_val + cinf_err, color="forestgreen", alpha=0.15)

    ax1.set_xlabel("Elapsed Time (hr)", fontsize=13, labelpad=6)
    ax1.set_ylabel(f"Tracer Concentration / Rate ({tracer})", fontsize=13, labelpad=6)
    ax1.set_title(f"Tracer Step Response Experiment ({tracer})", fontsize=14, fontweight="bold", pad=10)
    ax1.legend(loc="best", frameon=False, fontsize=10)
    ax1.tick_params(axis="both", which="major", labelsize=11)
    for spine in ["top", "bottom", "left", "right"]:
        ax1.spines[spine].set_linewidth(1.3)
    fig1.tight_layout()
    
    p1 = plots_dir / "rtd_tracer_step_profile.png"
    fig1.savefig(p1, dpi=300, bbox_inches="tight")
    plt.close(fig1)
    generated_plots["step_profile"] = p1

    # -----------------------------------------------------------------
    # Plot 2: F-Curve F(t) & TIS Model Fit
    # -----------------------------------------------------------------
    fig2, ax2 = plt.subplots(figsize=(8, 4.5))
    t_min = rtd_results["t_rel_min"]
    f_exp = rtd_results["f_curve"]
    t_mean_err_min = rtd_results.get("t_mean_err_min", 0.0)
    
    ax2.plot(t_min, f_exp, "s", markersize=6, color="#D9534F", markeredgecolor="black", label="Experimental F(t)")
    
    # Model Curve
    t_fine = np.linspace(0, max(t_min) * 1.05, 200)
    n_fit = rtd_results["N_tanks_fit"]
    n_fit_err = rtd_results.get("N_tanks_fit_err", 0.0)
    tau_fit = rtd_results["tau_fit_min"]
    tau_fit_err = rtd_results.get("tau_fit_err_min", 0.0)
    f_model_fine = gammainc(n_fit, n_fit * t_fine / tau_fit)
    
    ax2.plot(
        t_fine, f_model_fine, "-", color="#0275D8", lw=2, 
        label=f"TIS Model Fit (N = {n_fit:.2f}±{n_fit_err:.2f}, $\\tau$ = {tau_fit:.1f}±{tau_fit_err:.1f} min, $R^2$ = {rtd_results['r2_tis']:.3f})"
    )
    ax2.axvline(t_mean_min, color="black", linestyle="--", lw=1.2, label=f"Mean $\\bar{{t}}$ = {t_mean_min:.1f} ± {t_mean_err_min:.1f} min")
    if t_mean_err_min > 0:
        ax2.axvspan(max(0, t_mean_min - t_mean_err_min), t_mean_min + t_mean_err_min, color="black", alpha=0.08)
    
    ax2.set_xlabel("Time Since Step Change (min)", fontsize=13, labelpad=6)
    ax2.set_ylabel("Dimensionless F(t) Response", fontsize=13, labelpad=6)
    ax2.set_ylim(-0.05, 1.1)
    ax2.set_xlim(0, max(t_min) * 1.02)
    ax2.set_title("Step Response F-Curve & Tanks-in-Series Fit", fontsize=14, fontweight="bold", pad=10)
    ax2.legend(loc="lower right", frameon=False, fontsize=9.5)
    ax2.tick_params(axis="both", which="major", labelsize=11)
    for spine in ["top", "bottom", "left", "right"]:
        ax2.spines[spine].set_linewidth(1.3)
    fig2.tight_layout()

    p2 = plots_dir / "rtd_f_curve.png"
    fig2.savefig(p2, dpi=300, bbox_inches="tight")
    plt.close(fig2)
    generated_plots["f_curve"] = p2

    # -----------------------------------------------------------------
    # Plot 3: E-Curve E(t) Distribution
    # -----------------------------------------------------------------
    fig3, ax3 = plt.subplots(figsize=(8, 4.5))
    e_exp = rtd_results["e_curve"]
    
    ax3.plot(t_min, e_exp, "o-", markersize=5, color="#5CB85C", markeredgecolor="black", lw=1.8, label="Experimental E(t) = dF/dt")
    
    # Model E-curve: E(t) = N*(N*t/tau)^(N-1)/(gamma(N)*tau) * exp(-N*t/tau)
    log_e_model = (
        np.log(n_fit / tau_fit) + 
        (n_fit - 1.0) * np.log(np.maximum(1e-12, n_fit * t_fine / tau_fit)) - 
        np.log(gamma(n_fit)) - 
        (n_fit * t_fine / tau_fit)
    )
    e_model_fine = np.exp(np.clip(log_e_model, -50, 50))
    ax3.plot(t_fine, e_model_fine, "--", color="darkgreen", lw=2, label=f"TIS E(t) Model (N = {n_fit:.2f})")
    
    ax3.axvline(t_mean_min, color="crimson", linestyle=":", lw=1.5, label=f"Mean $\\bar{{t}}$ = {t_mean_min:.1f} ± {t_mean_err_min:.1f} min")
    if t_mean_err_min > 0:
        ax3.axvspan(max(0, t_mean_min - t_mean_err_min), t_mean_min + t_mean_err_min, color="crimson", alpha=0.08)
    
    ax3.set_xlabel("Time Since Step Change (min)", fontsize=13, labelpad=6)
    ax3.set_ylabel("RTD Distribution E(t) ($min^{-1}$)", fontsize=13, labelpad=6)
    ax3.set_title("Residence Time Distribution Density E(t)", fontsize=14, fontweight="bold", pad=10)
    ax3.legend(loc="upper right", frameon=False, fontsize=10)
    ax3.tick_params(axis="both", which="major", labelsize=11)
    for spine in ["top", "bottom", "left", "right"]:
        ax3.spines[spine].set_linewidth(1.3)
    fig3.tight_layout()

    p3 = plots_dir / "rtd_e_curve.png"
    fig3.savefig(p3, dpi=300, bbox_inches="tight")
    plt.close(fig3)
    generated_plots["e_curve"] = p3

    # -----------------------------------------------------------------
    # Plot 4: Dimensionless RTD E(theta) vs Theta
    # -----------------------------------------------------------------
    fig4, ax4 = plt.subplots(figsize=(8, 4.5))
    theta = t_min / t_mean_min if t_mean_min > 0 else t_min
    e_theta = e_exp * t_mean_min
    theta_fine = t_fine / t_mean_min if t_mean_min > 0 else t_fine
    e_theta_model = e_model_fine * t_mean_min

    ax4.plot(theta, e_theta, "D-", markersize=5, color="#F0AD4E", markeredgecolor="black", lw=1.8, label=r"Experimental $E(\theta)$")
    ax4.plot(theta_fine, e_theta_model, "--", color="#D9534F", lw=2, label=r"Model $E(\theta)$ ($N = " + f"{n_fit:.2f}" + r"$)")
    ax4.axvline(1.0, color="black", linestyle="--", lw=1.2, label=r"Ideal $\theta = 1.0$")

    ax4.set_xlabel(r"Dimensionless Time $\theta = t / \bar{t}$", fontsize=13, labelpad=6)
    ax4.set_ylabel(r"Dimensionless RTD $E(\theta) = \bar{t} \cdot E(t)$", fontsize=13, labelpad=6)
    ax4.set_title("Dimensionless Residence Time Distribution", fontsize=14, fontweight="bold", pad=10)
    ax4.legend(loc="upper right", frameon=False, fontsize=10)
    ax4.tick_params(axis="both", which="major", labelsize=11)
    for spine in ["top", "bottom", "left", "right"]:
        ax4.spines[spine].set_linewidth(1.3)
    fig4.tight_layout()

    p4 = plots_dir / "rtd_dimensionless.png"
    fig4.savefig(p4, dpi=300, bbox_inches="tight")
    plt.close(fig4)
    generated_plots["dimensionless"] = p4

    return generated_plots


def compute_rtd_summary_dataframe(rtd_results):
    """
    Creates a detailed summary dataframe of the computed RTD parameters and experimental curve points.
    """
    summary_metrics = [
        {"Parameter": "Tracer Species", "Value": str(rtd_results["tracer_col"]), "Uncertainty (+/-)": "-", "Units": "-"},
        {"Parameter": "Step Change Time (t0)", "Value": f"{rtd_results['t_step_hr']:.3f}", "Uncertainty (+/-)": "-", "Units": "hr"},
        {"Parameter": "Step Direction", "Value": rtd_results["step_direction"].capitalize(), "Uncertainty (+/-)": "-", "Units": "-"},
        {"Parameter": "Baseline Concentration (C0)", "Value": f"{rtd_results['c0']:.3f}", "Uncertainty (+/-)": f"{rtd_results.get('c0_err', 0.0):.3f}", "Units": "Signal / Rate"},
        {"Parameter": "Plateau Concentration (C_inf)", "Value": f"{rtd_results['c_inf']:.3f}", "Uncertainty (+/-)": f"{rtd_results.get('c_inf_err', 0.0):.3f}", "Units": "Signal / Rate"},
        {"Parameter": "Mean Residence Time (t_mean)", "Value": f"{rtd_results['t_mean_min']:.2f}", "Uncertainty (+/-)": f"{rtd_results.get('t_mean_err_min', 0.0):.2f}", "Units": "min"},
        {"Parameter": "Mean Residence Time (t_mean)", "Value": f"{rtd_results['t_mean_hr']:.4f}", "Uncertainty (+/-)": f"{rtd_results.get('t_mean_err_hr', 0.0):.4f}", "Units": "hr"},
        {"Parameter": "Variance (sigma^2)", "Value": f"{rtd_results['sigma2_min2']:.2f}", "Uncertainty (+/-)": f"{rtd_results.get('sigma2_err_min2', 0.0):.2f}", "Units": "min^2"},
        {"Parameter": "Standard Deviation (sigma)", "Value": f"{rtd_results['sigma_min']:.2f}", "Uncertainty (+/-)": f"{rtd_results.get('sigma_err_min', 0.0):.2f}", "Units": "min"},
        {"Parameter": "Dimensionless Variance (sigma_theta^2)", "Value": f"{rtd_results['sigma_theta2']:.4f}", "Uncertainty (+/-)": f"{rtd_results.get('sigma_theta2_err', 0.0):.4f}", "Units": "-"},
        {"Parameter": "Equivalent Tanks-in-Series (N)", "Value": f"{rtd_results['N_tanks']:.2f}", "Uncertainty (+/-)": f"{rtd_results.get('N_tanks_err', 0.0):.2f}", "Units": "tanks"},
        {"Parameter": "Fitted Tanks-in-Series (N_fit)", "Value": f"{rtd_results['N_tanks_fit']:.2f}", "Uncertainty (+/-)": f"{rtd_results.get('N_tanks_fit_err', 0.0):.2f}", "Units": "tanks"},
        {"Parameter": "Fitted Space Time (tau_fit)", "Value": f"{rtd_results['tau_fit_min']:.2f}", "Uncertainty (+/-)": f"{rtd_results.get('tau_fit_err_min', 0.0):.2f}", "Units": "min"},
        {"Parameter": "TIS Model Fit Quality (R^2)", "Value": f"{rtd_results['r2_tis']:.4f}", "Uncertainty (+/-)": "-", "Units": "-"},
        {"Parameter": "Vessel Dispersion Number (D/uL)", "Value": f"{rtd_results['dispersion_number']:.4f}", "Uncertainty (+/-)": f"{rtd_results.get('dispersion_err', 0.0):.4f}", "Units": "-"},
        {"Parameter": "Peclet Number (Pe)", "Value": f"{rtd_results['peclet_number']:.2f}", "Uncertainty (+/-)": f"{rtd_results.get('peclet_err', 0.0):.2f}", "Units": "-"},
        # Flow and Volume Metrics
        {"Parameter": "Total Flow Rate (Standard)", "Value": f"{rtd_results['tot_flow_sccm']:.2f}", "Uncertainty (+/-)": f"{rtd_results.get('tot_flow_err_sccm', 0.0):.2f}", "Units": "sccm"},
        {"Parameter": "Line / MFC Temperature", "Value": f"{rtd_results['t_lines_c']:.1f}", "Uncertainty (+/-)": f"{rtd_results.get('t_lines_err_c', 0.0):.1f}", "Units": "deg C"},
        {"Parameter": "Line Volumetric Flow Rate (20 C)", "Value": f"{rtd_results['q_ambient_ml_min']:.2f}", "Uncertainty (+/-)": f"{rtd_results.get('q_ambient_err_ml_min', 0.0):.2f}", "Units": "mL/min"},
        {"Parameter": "Furnace / Reactor Temperature", "Value": f"{rtd_results['rxr_t_c']:.1f}", "Uncertainty (+/-)": f"{rtd_results.get('rxr_t_err_c', 0.0):.1f}", "Units": "deg C"},
        {"Parameter": "Heated Flow Rate (at T_rxr)", "Value": f"{rtd_results['q_rxr_hot_ml_min']:.2f}", "Uncertainty (+/-)": f"{rtd_results.get('q_rxr_hot_err_ml_min', 0.0):.2f}", "Units": "mL/min"},
        {"Parameter": "Line / System Volume (at 20 C)", "Value": f"{rtd_results['v_ambient_ml']:.3f}", "Uncertainty (+/-)": f"{rtd_results.get('v_ambient_err_ml', 0.0):.3f}", "Units": "cm^3 (mL)"},
        {"Parameter": "Line / System Volume (at 20 C)", "Value": f"{rtd_results['v_ambient_ul']:.1f}", "Uncertainty (+/-)": f"{rtd_results.get('v_ambient_err_ul', 0.0):.1f}", "Units": "uL"},
        {"Parameter": "Heated Zone Volume (at T_rxr)", "Value": f"{rtd_results['v_hot_rxr_ml']:.3f}", "Uncertainty (+/-)": f"{rtd_results.get('v_hot_rxr_err_ml', 0.0):.3f}", "Units": "cm^3 (mL)"},
        {"Parameter": "Heated Zone Volume (at T_rxr)", "Value": f"{rtd_results['v_hot_rxr_ul']:.1f}", "Uncertainty (+/-)": f"{rtd_results.get('v_hot_rxr_err_ul', 0.0):.1f}", "Units": "uL"},
        {"Parameter": "Standard Fluid Volume (V_std)", "Value": f"{rtd_results['v_standard_scc']:.3f}", "Uncertainty (+/-)": f"{rtd_results.get('v_standard_err_scc', 0.0):.3f}", "Units": "scc"},
        {"Parameter": "Model Fitted Ambient Volume (V_model)", "Value": f"{rtd_results['v_model_ml']:.3f}", "Uncertainty (+/-)": f"{rtd_results.get('v_model_err_ml', 0.0):.3f}", "Units": "mL"}
    ]
    if pd.notna(rtd_results.get("dead_volume_pct")):
        summary_metrics.append({"Parameter": "Geometric Reactor Volume", "Value": f"{rtd_results['v_geom_cm3']:.3f}", "Uncertainty (+/-)": "-", "Units": "cm^3"})
        summary_metrics.append({"Parameter": "Dead Volume Fraction", "Value": f"{rtd_results['dead_volume_pct']:.1f}", "Uncertainty (+/-)": f"{rtd_results.get('dead_volume_err_pct', 0.0):.1f}", "Units": "%"})
        summary_metrics.append({"Parameter": "GHSV", "Value": f"{rtd_results['ghsv_hr']:.1f}", "Uncertainty (+/-)": f"{rtd_results.get('ghsv_err_hr', 0.0):.1f}", "Units": "hr^-1"})
    df_metrics = pd.DataFrame(summary_metrics)

    # Point by point table
    df_curve = pd.DataFrame({
        "Time_Relative_min": np.round(rtd_results["t_rel_min"], 2),
        "Time_Relative_hr": np.round(rtd_results["t_rel_hr"], 4),
        "Tracer_Signal": np.round(rtd_results["post_conc"], 4),
        "F_experimental": np.round(rtd_results["f_curve"], 4),
        "E_experimental_min-1": np.round(rtd_results["e_curve"], 6),
        "F_model_TIS": np.round(rtd_results["tis_fit_curve"], 4)
    })

    return df_metrics, df_curve
