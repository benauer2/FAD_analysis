import os
import tkinter as tk
from tkinter import filedialog
from pathlib import Path
import pandas as pd

projmon_loc = Path('S:/projmon/velma_formic-acid-decomposition/Data')

def get_dialog_root():
    root = tk.Tk()
    root.withdraw()
    root.wm_attributes('-topmost', 1)
    return root

def select_file(title, initial_dir=None, filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]):
    root = get_dialog_root()
    path = filedialog.askopenfilename(title=title, initialdir=initial_dir, filetypes=filetypes)
    root.destroy()
    return Path(path) if path else None

def select_directory(title, initial_dir=None):
    root = get_dialog_root()
    path = filedialog.askdirectory(title=title, initialdir=initial_dir)
    root.destroy()
    return Path(path) if path else None

def get_extended_path(file_path):
    """
    Returns an extended-length path string for Windows (handling drive letters and UNC network paths).
    """
    p_str = str(Path(file_path).resolve())
    if os.name == 'nt' and not p_str.startswith('\\\\?\\'):
        if p_str.startswith('\\\\'):
            p_str = '\\\\?\\UNC\\' + p_str.lstrip('\\')
        else:
            p_str = '\\\\?\\' + p_str
    return p_str

def safe_read_csv(file_path, **kwargs):
    """
    Reads a CSV file safely, prepending \\?\\ or \\?\\UNC\\ on Windows for extended-length path support.
    """
    return pd.read_csv(get_extended_path(file_path), **kwargs)

def select_folder_and_load_data(selected_dir=None):
    # Get the directory from the user if not provided
    if selected_dir is None:
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        selected_dir = filedialog.askdirectory(
            title="Select Reaction Folder",
            initialdir=projmon_loc if projmon_loc.exists() else Path.cwd()
            )
    # Handle the case where the user cancels the selection
    if not selected_dir:
        print("No folder selected.")
        return None, None, None
    
    reaction_path = Path(selected_dir)
    # Route into the raw_data folder if it exists, otherwise search reaction_path directly
    if (reaction_path / "raw_data").exists():
        parent_path = reaction_path / "raw_data"
    else:
        parent_path = reaction_path

    all_df = {}
    pre_reaction_df = {}
        
    '''
    ##########################################
    PROCESS CSV FILES INSIDE DATA FOLDER
    ##########################################
    '''
    # Search both parent_path and reaction_path root for csv files
    csv_search_paths = list(parent_path.glob('*.csv'))
    if parent_path != reaction_path:
        for f in reaction_path.glob('*.csv'):
            if f not in csv_search_paths:
                csv_search_paths.append(f)

    for file in csv_search_paths:
        filename_lower = file.name.lower()

        try:
            if "fad" in filename_lower:
                print(f"Processing 'GC' file: {file.name}")
                # Robust reader: Skip metadata lines dynamically by seeking the true header row
                header_line = 0
                with open(get_extended_path(file), 'r', errors='ignore') as f:
                    for idx, line in enumerate(f):
                        if "folder name" in line.lower() or "line" in line.lower():
                            if line.count(",") >= 2:
                                header_line = idx
                                break 
                all_df["FAD_df"] = safe_read_csv(file, skiprows=header_line)
            elif "eis" in filename_lower and "pre" not in filename_lower:
                print(f"Processing 'EIS' file: {file.name}")    
                all_df["EIS_df"] = safe_read_csv(file)
            elif "raw_data" in filename_lower:
                print(f"Processing 'Electronics' file: {file.name}")
                all_df["electronics_df"] = safe_read_csv(file)
            elif "temp_data" in filename_lower:
                print(f"Processing 'Temperature' file: {file.name}")
                all_df["temp_df"] = safe_read_csv(file)
            elif "mfc_data" in filename_lower:
                print(f"Processing 'MFC' file: {file.name}")
                all_df["mfc_df"] = safe_read_csv(file)
            elif "pump_data" in filename_lower or "pump" in filename_lower:
                print(f"Processing 'Pump' file: {file.name}")
                all_df["pump_df"] = safe_read_csv(file)
        except Exception as e:
            print(f"Error reading root CSV {file.name}: {e}")

    '''
    ##############################################
    PROCESS SUBFOLDERS (PRE-REACTION ELECTRONICS)
    ##############################################
    '''
    for subfolder in parent_path.iterdir():
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
                df = safe_read_csv(file_path)
                if not df.empty:
                    folder_dfs.append(df)
            except Exception as e:
                print(f"Error reading {file_path.name}: {e}")

        if folder_dfs:
            merged_df = pd.concat(folder_dfs, ignore_index=True)
            pre_reaction_df[folder_name] = merged_df

    return all_df, pre_reaction_df, reaction_path