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
RXN_SCHEDULE = []
TEMP_SCHEDULE = []
TRIAL_INFO = {}
GC_METHOD_INFO = {}
_config = {}

#Load the yaml file
def handle_yaml(selected_dir):
    global BASE_DIR, JSON_PARAMS_FID, JSON_PARAMS_TCD, notes, analysis, RXN_SCHEDULE, TEMP_SCHEDULE, TRIAL_INFO, GC_METHOD_INFO, _config
    reaction_path = Path(selected_dir)
    yaml_path = reaction_path / "background_meta.yaml"

    if not yaml_path.exists():
        print(f"[Config] No existing 'background_meta.yaml' found in {reaction_path}. Using default.")
        yaml_path = Path(__file__).parent / "background_meta_default.yaml"
        save_dir = selected_dir / "background_meta.yaml"
        
        if not yaml_path.exists():
            raise FileNotFoundError(f"Neither local folder config nor global default yaml was found.")
        
        shutil.copy2(yaml_path, save_dir)

    print(f"[Config] Loading metadata parameters from: {yaml_path.name}")
    with open(yaml_path, "r") as f:
            _config = yaml.safe_load(f)

    #HANDLE FILE PATHS
    BASE_DIR = Path(_config["file_paths"]["base_dir"])
    JSON_PARAMS_FID = BASE_DIR / _config["file_paths"]["json_params_fid"]
    JSON_PARAMS_TCD = BASE_DIR / _config["file_paths"]["json_params_tcd"]

    #MAP PLOTTING & REACTION PARAMETERS

    if "rxn_schedule" in _config:
        RXN_SCHEDULE = _config["rxn_schedule"]
        TEMP_SCHEDULE = RXN_SCHEDULE  # Maintain alias
    elif "temp_schedule" in _config:
        TEMP_SCHEDULE = _config["temp_schedule"]
        RXN_SCHEDULE = TEMP_SCHEDULE
    else:
        RXN_SCHEDULE = []
        TEMP_SCHEDULE = []

    #TEXT BLOCKS
    notes = _config.get("notes", "")
    analysis = _config.get("analysis", "")

    #TRIAL AND INSTRUMENT METADATA
    TRIAL_INFO = _config["trial_info"]
    GC_METHOD_INFO = _config["gc_method_info"]

    return _config