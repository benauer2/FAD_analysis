import yaml
from pathlib import Path
import shutil
import pandas as pd

BASE_DIR = None
JSON_PARAMS_FID = None
JSON_PARAMS_TCD = None
notes = ""
analysis = ""
SHADE_COLORS = {}
BIAS_SCHEDULE = []
RXN_SCHEDULE = []
TEMP_SCHEDULE = []
TRIAL_INFO = {}
GC_METHOD_INFO = {}
file_paths = {}
_config = {}

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
    """Parses arrhenius_sequence.csv strictly into your requested dictionary format."""
    df = pd.read_csv(csv_path)
    
    
    # Strip whitespace and hidden UTF-8 BOM characters from column headers
    df.columns = [str(c).strip().lstrip('\ufeff') for c in df.columns]
    # Drop empty trailing rows (e.g. row 9 in the screenshot)
    df = df.dropna(subset=["Rxr_T_C"])
    
    schedule = []
    current_time = 0.0
    for _, row in df.iterrows():
        avg_window = parse_avg_window(row.get("Avg_Window", "0,0"))
        dur_raw = row.get("Duration_hrs")
        
        try:
            dur_hrs = float(dur_raw) if pd.notna(dur_raw) else (avg_window[1] if len(avg_window) >= 2 else 0.0)
        except (ValueError, TypeError):
            dur_hrs = avg_window[1] if len(avg_window) >= 2 else 0.0

        start_time = current_time
        end_time = current_time + dur_hrs
        time_window = [start_time, end_time]

        current_time = end_time


        step = {
            "run": int(row.get("Run", 0)),
            "rxr_t": float(row.get("Rxr_T_C", 0.0)),
            "n2_flow": float(row.get("N2_flow_sccm", 0.0)),
            "fa_flow": float(row.get("FA_flow_ul_min", 0.0)),
            "we_volt": parse_voltage(row.get("WE_Voltage", 0.0)),
            "avg_window": avg_window,
            "dur_hrs": dur_hrs,
            "time_window" : time_window
        }
        schedule.append(step)
        
    return schedule


def handle_yaml(selected_dir: Path):
    """Loads metadata & shade colors from YAML and the reaction sequence from CSV."""
    global TRIAL_INFO, GC_METHOD_INFO, RXN_SCHEDULE, TEMP_SCHEDULE, SHADE_COLORS, notes, analysis, file_paths
    
    if selected_dir is None:
        raise ValueError("Selected directory path cannot be None.")
        
    selected_dir = Path(selected_dir)
    
    # 1. Load YAML for metadata, paths, and shade colors
    yaml_files = list(selected_dir.glob("*.yaml")) + list(selected_dir.glob("*.yml"))
    if yaml_files:
        yaml_path = yaml_files[0]
        print(f"[Config] Loading metadata parameters from: {yaml_path.name}")
        with open(yaml_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
            
        TRIAL_INFO = config.get("trial_info", {})
        GC_METHOD_INFO = config.get("gc_method_info", {})
        notes = config.get("notes", "")
        analysis = config.get("analysis", "")
        file_paths = config.get("file_paths", {})
        
        # EXTRACT & MAP SHADE_COLORS
        raw_shades = config.get("shade_colors", {})
        SHADE_COLORS = {}
        for k, v in raw_shades.items():
            # Support both numeric values (floats/ints) and string lookups (e.g. "OCP")
            try:
                SHADE_COLORS[float(k)] = v
            except (ValueError, TypeError):
                SHADE_COLORS[str(k)] = v

    # 2. Load RXN_SCHEDULE from arrhenius_sequence.csv
    csv_path = selected_dir / "arrhenius_sequence.csv"
    if not csv_path.exists():
        csv_files = list(selected_dir.glob("*sequence*.csv")) + list(selected_dir.glob("*schedule*.csv"))
        if csv_files:
            csv_path = csv_files[0]
        else:
            raise FileNotFoundError(f"Could not find 'arrhenius_sequence.csv' in {selected_dir}")
            
    RXN_SCHEDULE = load_schedule_from_csv(csv_path)
    TEMP_SCHEDULE = RXN_SCHEDULE
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


