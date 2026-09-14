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

def select_folder_and_load_data(selected_dir=None):
    #Get the directory from the user if not provided
    if selected_dir is None:
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        selected_dir = filedialog.askdirectory(
            title="Select Reaction Folder",
            initialdir=projmon_loc if projmon_loc.exists() else Path.cwd()
            )
    #Handle the case where the user cancels the selection
    if not selected_dir:
        print("No folder selected.")
        return None, None, None
    
    reaction_path = Path(selected_dir)
    #Route directly into the raw_data folder if it exists
    parent_path = reaction_path / "raw_data"

    if not parent_path.exists():
        print(f"Error: Expected 'raw_data' folder not found inside {reaction_path}")
        return None, None, None

    all_df = {}
    pre_reaction_df = {}
        
    '''
    ##########################################
    PROCESS CSV FILES INSIDE RAW_DATA FOLDER
    ##########################################
    '''
    for file in parent_path.glob('*.csv'):
        filename_lower = file.name.lower()

        if "fad" in filename_lower:
            print(f"Processing 'GC' file: {file.name}")
            # Robust reader: Skip metadata lines dynamically by seeking the true header row
            header_line = 0
            with open(file, 'r') as f:
                for idx, line in enumerate(f):
                    # Adjust this keyword to match an expected column name in your GC output (e.g., "Time", "Signal", "Peak")
                    if "folder name" in line.lower() or "line" in line.lower() in line:
                        # Double check that this line actually looks like a data header row
                        if line.count(",") >= 2:
                            header_line = idx
                            break 
            all_df["FAD_df"] = pd.read_csv(file, skiprows=header_line)
        elif "eis" in filename_lower:
            print(f"Processing 'EIS' file: {file.name}")    
            all_df["EIS_df"] = pd.read_csv(file)
        elif "raw_data" in filename_lower:
            print(f"Processing 'Electronics' file: {file.name}")
            all_df["electronics_df"] = pd.read_csv(file)
        elif "temp_data" in filename_lower:
            print(f"Processing 'Temperature' file: {file.name}")
            all_df["temp_df"] = pd.read_csv(file)
        elif "mfc_data" in filename_lower:
            print(f"Processing 'MFC' file: {file.name}")
            all_df["mfc_df"] = pd.read_csv(file)
        

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
                df = pd.read_csv(file_path)
                if not df.empty:
                    folder_dfs.append(df)
            except Exception as e:
                print(f"Error reading {file_path.name}: {e}")

        if folder_dfs:
            merged_df = pd.concat(folder_dfs, ignore_index=True)
            pre_reaction_df[folder_name] = merged_df

    return all_df, pre_reaction_df, reaction_path