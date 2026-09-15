from pathlib import Path
import shutil
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import simpson
from scipy import stats
from impedance.models.circuits import CustomCircuit

from arrhenius_worker_files.arrhenius_config import get_step_at_time

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



