import yaml
from pathlib import Path
import shutil

BASE_DIR = None
JSON_PARAMS_FID = None
JSON_PARAMS_TCD = None
notes = ""
analysis = ""
SHADE_COLORS = {}
BIAS_SCHEDULE = []
TRIAL_INFO = {}
GC_METHOD_INFO = {}

def parse_avg_window(val):
    """Converts a string like '6, 16' or '[6, 16]' into a list of floats [6.0, 16.0]."""
    if pd.isna(val) or val is None:
        return [0.0, 0.0]
    val_str = str(val).strip("[]() ")
    parts = val_str.split(",")
    return [float(p.strip()) for p in parts]

def parse_voltage(val):
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



from pathlib import Path
import pandas as pd
import numpy as np

def load_schedule_from_csv(csv_path: Path) -> list:
    """Parses sequence CSVs (sequence_default, arrhenius_sequence, etc.) into a standardized schedule dictionary format."""
    df = pd.read_csv(csv_path)
    
    # Strip whitespace and hidden UTF-8 BOM characters from column headers
    df.columns = [str(c).strip().lstrip('\ufeff') for c in df.columns]
    
    def _find_col(df_in, candidates):
        cols_lower = {c.lower().replace("_", "").replace(" ", "").replace("(", "").replace(")", "").replace(".", ""): c for c in df_in.columns}
        for cand in candidates:
            cleaned = cand.lower().replace("_", "").replace(" ", "").replace("(", "").replace(")", "").replace(".", "")
            if cleaned in cols_lower:
                return cols_lower[cleaned]
        return None

    temp_col = _find_col(df, ["Rxr_T_C", "Reactor_T_C", "Reactor_T", "Temp_C", "Temperature_C", "Temp"])
    run_col = _find_col(df, ["Run", "Step", "Step_Index", "Step_ID", "Index"])
    n2_col = _find_col(df, ["N2_flow_sccm", "N2_Flow_sccm", "N2_flow", "N2_sccm", "n2_flow"])
    fa_col = _find_col(df, ["FA_flow_ul_min", "FA_Flow_ul_min", "FA_flow", "fa_flow"])
    volt_col = _find_col(df, ["WE_Voltage", "Target_Voltage_V", "Target_Voltage", "Voltage_V", "Voltage", "we_volt", "Bias_V", "Bias"])
    dur_col = _find_col(df, ["Duration_hrs", "Duration_hr", "Duration (hrs)", "Duration", "dur_hrs", "Duration_hours"])
    avg_col = _find_col(df, ["Avg_Window", "avg_window", "Average_Window"])

    # Drop empty trailing rows
    if temp_col and temp_col in df.columns:
        df = df.dropna(subset=[temp_col])
    elif dur_col and dur_col in df.columns:
        df = df.dropna(subset=[dur_col])
    elif volt_col and volt_col in df.columns:
        df = df.dropna(subset=[volt_col])
    else:
        df = df.dropna(how="all")
    
    schedule = []
    current_time = 0.0
    for idx, (_, row) in enumerate(df.iterrows(), start=1):
        dur_raw = row.get(dur_col) if dur_col else None
        avg_raw = row.get(avg_col) if avg_col else None
        avg_window = parse_avg_window(avg_raw) if avg_raw is not None else [0.0, 0.0]
        
        try:
            dur_hrs = float(dur_raw) if pd.notna(dur_raw) else (avg_window[1] if len(avg_window) >= 2 and avg_window[1] > 0 else 8.0)
        except (ValueError, TypeError):
            dur_hrs = avg_window[1] if len(avg_window) >= 2 and avg_window[1] > 0 else 8.0

        if avg_window == [0.0, 0.0] or (len(avg_window) >= 2 and avg_window[1] == 0.0):
            avg_window = [max(0.0, dur_hrs - 2.0), dur_hrs]

        start_time = current_time
        end_time = current_time + dur_hrs
        time_window = [start_time, end_time]
        current_time = end_time

        step_run = int(row.get(run_col, idx)) if run_col and pd.notna(row.get(run_col)) else idx
        rxr_t = float(row.get(temp_col, 0.0)) if temp_col and pd.notna(row.get(temp_col)) else 0.0
        n2_flow = float(row.get(n2_col, 0.0)) if n2_col and pd.notna(row.get(n2_col)) else 0.0
        fa_flow = float(row.get(fa_col, 0.0)) if fa_col and pd.notna(row.get(fa_col)) else 0.0
        we_volt = parse_voltage(row.get(volt_col, 0.0)) if volt_col else 0.0

        step = {
            "run": step_run,
            "rxr_t": rxr_t,
            "n2_flow": n2_flow,
            "fa_flow": fa_flow,
            "we_volt": we_volt,
            "avg_window": avg_window,
            "dur_hrs": dur_hrs,
            "time_window": time_window
        }
        schedule.append(step)
        
    return schedule


def handle_yaml(selected_dir: Path):
    """Loads metadata & shade colors from YAML and the reaction sequence from CSV."""
    global TRIAL_INFO, GC_METHOD_INFO, RXN_SCHEDULE, TEMP_SCHEDULE, BIAS_SCHEDULE, SHADE_COLORS, notes, analysis, file_paths
    
    if selected_dir is None:
        raise ValueError("Selected directory path cannot be None.")
        
    selected_dir = Path(selected_dir)
    
    # 1. Load YAML for metadata, paths, and shade colors
    yaml_files = list(selected_dir.glob("*.yaml")) + list(selected_dir.glob("*.yml"))
    if yaml_files:
        yaml_path = yaml_files[0]
        print(f"[Config] Loading metadata parameters from: {yaml_path.name}")
        with open(yaml_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
            
        TRIAL_INFO = cfg.get("trial_info", {})
        GC_METHOD_INFO = cfg.get("gc_method_info", {})
        notes = cfg.get("notes", "")
        analysis = cfg.get("analysis", "")
        file_paths = cfg.get("file_paths", {})
        
        # EXTRACT & MAP SHADE_COLORS
        raw_shades = cfg.get("shade_colors", {})
        SHADE_COLORS = {}
        for k, v in raw_shades.items():
            # Support numeric values (floats/ints) and string lookups (e.g. "OCP")
            try:
                SHADE_COLORS[float(k)] = v
            except (ValueError, TypeError):
                SHADE_COLORS[str(k)] = v

    # 2. Load RXN_SCHEDULE from sequence CSV
    csv_path = selected_dir / "arrhenius_sequence.csv"
    if not csv_path.exists():
        csv_path = selected_dir / "sequence_default.csv"
    if not csv_path.exists():
        csv_files = list(selected_dir.glob("*sequence*.csv")) + list(selected_dir.glob("*schedule*.csv"))
        if csv_files:
            csv_path = csv_files[0]
        else:
            raise FileNotFoundError(f"Could not find sequence/schedule CSV in {selected_dir}")
            
    RXN_SCHEDULE = load_schedule_from_csv(csv_path)
    TEMP_SCHEDULE = RXN_SCHEDULE
    BIAS_SCHEDULE = RXN_SCHEDULE
    print(f"Successfully loaded schedule from {csv_path.name} ({len(RXN_SCHEDULE)} steps).")


def get_step_at_time(time_hours, RXN_SCHEDULE, t_start=0.0):
    """
    Locates the active RXN_SCHEDULE step dictionary for a given elapsed time.
    Safely handles invalid or non-dict schedule elements.
    """
    if not RXN_SCHEDULE or not isinstance(RXN_SCHEDULE, list):
        return None

    # Filter schedule to ensure elements are valid dictionaries
    valid_schedule = [s for s in RXN_SCHEDULE if isinstance(s, dict) and "time_window" in s]
    if not valid_schedule:
        return None

    elapsed = time_hours - t_start

    # Handle pre-experiment or negative times
    if elapsed < 0:
        return valid_schedule[0]

    for entry in valid_schedule:
        win = entry.get("time_window")
        if isinstance(win, (list, tuple)) and len(win) >= 2:
            start_win, end_win = float(win[0]), float(win[1])
            # Half-open interval [start_win, end_win)
            if start_win <= elapsed < end_win:
                return entry

    # Return last scheduled step if experiment outlasted duration schedule
    return valid_schedule[-1]


def get_voltage_at_time(time_hours, RXN_SCHEDULE, t_start=0.0):
    """
    Returns the target working electrode voltage (float or string like 'OCP') for a given elapsed time.
    """
    step = get_step_at_time(time_hours, RXN_SCHEDULE, t_start=t_start)
    if step and isinstance(step, dict) and "we_volt" in step:
        return step["we_volt"]
    return 0.0



