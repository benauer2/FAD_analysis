import yaml
from pathlib import Path
import shutil
import pandas as pd
from config import load_schedule_from_csv, get_step_at_time, get_voltage_at_time

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
_config = {}

def handle_yaml(selected_dir):
    global BASE_DIR, JSON_PARAMS_FID, JSON_PARAMS_TCD, notes, analysis, RXN_SCHEDULE, TEMP_SCHEDULE, BIAS_SCHEDULE, SHADE_COLORS, TRIAL_INFO, GC_METHOD_INFO, _config
    reaction_path = Path(selected_dir)
    yaml_path = reaction_path / "background_meta.yaml"

    if not yaml_path.exists():
        yaml_files = list(reaction_path.glob("*.yaml")) + list(reaction_path.glob("*.yml"))
        if yaml_files:
            yaml_path = yaml_files[0]
        else:
            print(f"[Config] No existing 'background_meta.yaml' found in {reaction_path}. Using default.")
            yaml_path = Path(__file__).parent / "background_meta_default.yaml"
            save_dir = reaction_path / "background_meta.yaml"
            
            if not yaml_path.exists():
                raise FileNotFoundError(f"Neither local folder config nor global default yaml was found.")
            
            shutil.copy2(yaml_path, save_dir)

    print(f"[Config] Loading metadata parameters from: {yaml_path.name}")
    with open(yaml_path, "r", encoding="utf-8") as f:
        _config = yaml.safe_load(f) or {}

    # HANDLE FILE PATHS
    file_paths = _config.get("file_paths", {})
    if "base_dir" in file_paths:
        BASE_DIR = Path(file_paths["base_dir"])
        if "json_params_fid" in file_paths:
            JSON_PARAMS_FID = BASE_DIR / file_paths["json_params_fid"]
        if "json_params_tcd" in file_paths:
            JSON_PARAMS_TCD = BASE_DIR / file_paths["json_params_tcd"]

    # SHADE COLORS
    raw_shades = _config.get("shade_colors", {})
    SHADE_COLORS = {}
    for k, v in raw_shades.items():
        try:
            SHADE_COLORS[float(k)] = v
        except (ValueError, TypeError):
            SHADE_COLORS[str(k)] = v

    # TEXT BLOCKS
    notes = _config.get("notes", "")
    analysis = _config.get("analysis", "")

    # TRIAL AND INSTRUMENT METADATA
    TRIAL_INFO = _config.get("trial_info", {})
    GC_METHOD_INFO = _config.get("gc_method_info", {})

    # LOAD REACTION / TEMPERATURE SCHEDULE FROM CSV
    csv_candidates = [
        reaction_path / "background_sequence.csv",
        reaction_path / "sequence_default.csv",
        reaction_path / "arrhenius_sequence.csv",
        reaction_path / "reaction_sequence.csv"
    ]
    csv_path = None
    for cand in csv_candidates:
        if cand.exists():
            csv_path = cand
            break
            
    if not csv_path:
        csv_files = list(reaction_path.glob("*sequence*.csv")) + list(reaction_path.glob("*schedule*.csv"))
        if csv_files:
            csv_path = csv_files[0]

    if csv_path and csv_path.exists():
        RXN_SCHEDULE = load_schedule_from_csv(csv_path)
        TEMP_SCHEDULE = RXN_SCHEDULE
        BIAS_SCHEDULE = RXN_SCHEDULE
        print(f"Successfully loaded schedule from {csv_path.name} ({len(RXN_SCHEDULE)} steps).")
    else:
        # Fallback to YAML schedule if no CSV exists
        if "rxn_schedule" in _config:
            RXN_SCHEDULE = _config["rxn_schedule"]
            TEMP_SCHEDULE = RXN_SCHEDULE
            BIAS_SCHEDULE = RXN_SCHEDULE
        elif "temp_schedule" in _config:
            raw_sched = _config["temp_schedule"]
            standardized = []
            curr_time = 0.0
            for idx, item in enumerate(raw_sched, start=1):
                dur = float(item.get("duration_hours", item.get("dur_hrs", 6.0)))
                temp = float(item.get("temperature", item.get("rxr_t", 0.0)))
                n2 = float(item.get("n2_flow", 15.0))
                fa_flow = float(item.get("fa_flow", 0.0))
                fa_vapor_sccm = fa_flow * (1.220 * 8.314 * 293.15 * 1000) / (46.025 * 101325)
                tot_flow = n2 + fa_vapor_sccm
                fa_pct = (fa_vapor_sccm / tot_flow * 100.0) if tot_flow > 0 else 0.0
                standardized.append({
                    "run": idx,
                    "rxr_t": temp,
                    "n2_flow": n2,
                    "h2_flow": 0.0,
                    "fa_flow": fa_flow,
                    "fa_vapor_sccm": fa_vapor_sccm,
                    "tot_flow": tot_flow,
                    "fa_pct": fa_pct,
                    "we_volt": 0.0,
                    "avg_window": [max(0.0, dur - 2.0), dur],
                    "dur_hrs": dur,
                    "time_window": [curr_time, curr_time + dur]
                })
                curr_time += dur
            RXN_SCHEDULE = standardized
            TEMP_SCHEDULE = RXN_SCHEDULE
            BIAS_SCHEDULE = RXN_SCHEDULE
            print(f"Loaded schedule from YAML temp_schedule ({len(RXN_SCHEDULE)} steps).")
        else:
            RXN_SCHEDULE = []
            TEMP_SCHEDULE = []
            BIAS_SCHEDULE = []

    return _config