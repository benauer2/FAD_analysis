'''
This code converts a CH file to a CSV file with a user input name
'''

import csv
from GC_handler.ch_parser import CHFile
import tkinter as tk
from tkinter import filedialog
import sys
import numpy as np

# Create a root window but keep it hidden
root = tk.Tk()
root.withdraw() # Hides the small tkinter window

# Open the file dialog
# The 'filetypes' argument filters for specific file extensions
CH_file = filedialog.askopenfilename(
    title="Select a CH file",
    filetypes=(("CH files", "*.ch"), ("All files", "*.*"))
)

# Check if the user selected a file or cancelled the dialog
if not CH_file:
    print("No CH file was selected.")
    sys.exit(0)

ch_obj = CHFile(CH_file)

GC_t = np.array(ch_obj.times()) # Time data from the chromatograph
GC_s = np.array(ch_obj.values) # Signal data from the chromotograph

GC_vals = np.zeros(shape=(len(GC_t),2))
for i in range(len(GC_t)):
    GC_vals[i,0] = GC_t[i]
    GC_vals[i,1] = GC_s[i]

title = input("\nInput CSV file name: ")

file_name = title + ".csv"

np.savetxt(file_name, GC_vals, delimiter=",", header="Time (min), Signal")
