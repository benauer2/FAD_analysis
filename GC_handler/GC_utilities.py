import numpy as np
from .ch_parser import CHFile
import tkinter as tk
from tkinter import filedialog
import sys
from pathlib import Path
import json
import matplotlib.pyplot as plt
from tkinter import filedialog
import os
import csv
from datetime import datetime
import pygetwindow
import pyautogui
import time
from scipy.optimize import curve_fit

'''
This utilities script automates Agilent OpenLab ChemStation control by screen
clicking and also automates integration(which consists of one function for
individual line integration, one for sequence integration) takes a .ch
Agilent GC file, extracts the signal vs. time data, runs an integration, and 
outputs two lists: (1) the list of chemicals and (2) the list of chemical
compositions (converted from peak areas using calibration factors) Additionally,
this code requires a JSON file that contains the integration parameters.
Important note if you use multiple detectors: the JSON parameters should be for
the type of CH file that you select.

A JSON file template example:
{
    "events": [
            {
                "detector": "FID",
                "chemicals": ["FA"],
                "_comment_chemicals": "The chemicals must appear in chronological order",
                "cal_factor": [0.00000414598210598295],
                "_comment_cal_factor": "The calibration factor must be in (your desired units)/(peak area)",
                "integration_algorithm": "simple peak",
                "_comment_algorithm": "Options: simple peak, simple split, simple skim",
                "time_stamps": [13.5,20.5],
                "_comment_time_stamps": "times in minutes, single peaks need two time stamps, simple splits need >=three, simple skims need four",
                "ignored_chemicals": [],
                "_comment_ignored_chemicals": "This tells the integrator not to save the value for that chemical",
                "meta_output_units": "umol/mL",
                "meta_user": "Justin Hopkins",
                "meta_cal_date": "02/15/2025"
            },
            ...
    ]
}
***********************************
'''

class GCController:
    '''
    Class that handles GC control during experiments
    '''

    def __init__(self, name:str, GC_directory:str=None, detectors:list="",
                 JSON_files:list=None):
        '''
        :param name: GC Name listed in the ChemStation Control Panel (e.g., Rivian)
        :type name: str
        :param GC_directory: File path for GC data, typically 
            .../Chemstation/.../1/Data
        :type GC_directory: str
        :param detectors: List of detector names for scanning files
        :type detectors: list of str
        :param JSON_files: Str of file locations for JSON files that contain
            integration parameters. The JSONs must correspond to the order that
            the detectors were listed in
        :type: list of str
        '''

        self.name = name

        self.data_directory = GC_directory # None if nothing is passed

        self.detectors = detectors # None if nothing is passed
        
        self.integration_params = {keys:"" for keys in detectors}

        for i in range(len(detectors)):
            self.integration_params[detectors[i]] = JSON_files[i] # None if nothing is passed


    def run_sequence(self):
        '''
        Controls the mouse in order to run the currently selected sequence in 
        ChemStation. After clicking, it will return to the original screen
        '''

        current_window = pygetwindow.getActiveWindow()
        window_title = f"{self.name} (online): Method and Run Control"

        try:
                # getWindowsWithTitle() returns a list, so we take the first element [0].
                target_window = pyautogui.getWindowsWithTitle(window_title)[0]

                # Bring the window to the front.
                if target_window:
                    print(f"Found '{window_title}', activating it now...")
                    target_window.activate()
                    target_window.maximize()
                else:
                    print(f"Window with title '{window_title}' not found.")

        except IndexError:
            print(f"Error: Could not find any window with the title '{window_title}'.")
            print("Please make sure the program is running and the title is correct.")

        # --- Example: Move and Click ---
        print("Moving to 'Run Control' in 3 seconds...")
        time.sleep(3) # A short pause to give you time to switch windows

        # Move the mouse to the absolute coordinates (800, 500) over 1 second.
        RunControl = pyautogui.locateCenterOnScreen("RunControl.png",
                                                    confidence=0.90,
                                                    grayscale=True)
        pyautogui.moveTo(RunControl.x, RunControl.y, duration=1)

        # Perform a single left-button click.
        pyautogui.click()

        # An even simpler way to do it is to pass the coordinates directly to the click function.
        print("\nMoving and clicking at 'Run Sequence' in 3 seconds...")
        time.sleep(3)
        RunSequence = pyautogui.locateCenterOnScreen("RunSequence.png",
                                                     confidence=0.90,
                                                     grayscale=True)
        pyautogui.moveTo(RunSequence.x, RunSequence.y, duration=1)
        pyautogui.click()

        # Go back to original window
        current_window.activate()


    def run_method(self):
        '''
        Controls the mouse in order to run the currently selected method in 
        ChemStation. After clicking, it will return to the original screen
        '''

        current_window = pygetwindow.getActiveWindow()
        window_title = f"{self.name} (online): Method and Run Control"

        try:
                # getWindowsWithTitle() returns a list, so we take the first element [0].
                target_window = pyautogui.getWindowsWithTitle(window_title)[0]

                # Bring the window to the front.
                if target_window:
                    print(f"Found '{window_title}', activating it now...")
                    target_window.activate()
                    target_window.maximize()
                else:
                    print(f"Window with title '{window_title}' not found.")

        except IndexError:
            print(f"Error: Could not find any window with the title '{window_title}'.")
            print("Please make sure the program is running and the title is correct.")

        # Get the size of your primary monitor.
        # Note that the "Run Control" tab is at X=88, Y=38
        # The "Run Sequence" selection is at X=106, Y = 136
        screenWidth, screenHeight = pyautogui.size()
        print(f"Your screen resolution is: {screenWidth}x{screenHeight}")

        # --- Example: Move and Click ---
        print("Moving to 'Run Control' in 3 seconds...")
        time.sleep(3) # A short pause to give you time to switch windows

        # Move the mouse to the absolute coordinates (800, 500) over 1 second.
        RunControl = pyautogui.locateCenterOnScreen("RunControl.png",
                                                    confidence=0.90,
                                                    grayscale=True)
        pyautogui.moveTo(RunControl.x, RunControl.y, duration=1)

        # Perform a single left-button click.
        pyautogui.click()

        # An even simpler way to do it is to pass the coordinates directly to the click function.
        print("\nMoving and clicking at 'Run Sequence' in 3 seconds...")
        time.sleep(3)
        RunSequence = pyautogui.locateCenterOnScreen("RunSequence.png",
                                                     confidence=0.90,
                                                     grayscale=True)
        pyautogui.moveTo(RunSequence.x, RunSequence.y, duration=1)
        pyautogui.click()

        # Go back to original window
        current_window.activate()
    

    def stop_run(self):
        '''
        Method to stop the GC method or sequence that is currently going
        '''
        current_window = pygetwindow.getActiveWindow()
        window_title = f"{self.name} (online): Method and Run Control"

        try:
                # getWindowsWithTitle() returns a list, so we take the first element [0].
                target_window = pyautogui.getWindowsWithTitle(window_title)[0]

                # Bring the window to the front.
                if target_window:
                    print(f"Found '{window_title}', activating it now...")
                    target_window.activate()
                    target_window.maximize()
                else:
                    print(f"Window with title '{window_title}' not found.")

        except IndexError:
            print(f"Error: Could not find any window with the title '{window_title}'.")
            print("Please make sure the program is running and the title is correct.")

        # --- Example: Move and Click ---
        print("Moving to 'Run Control' in 3 seconds...")
        time.sleep(3) # A short pause to give you time to switch windows

        # Move the mouse to the absolute coordinates (800, 500) over 1 second.
        RunControl = pyautogui.locateCenterOnScreen("RunControl.png",
                                                    confidence=0.90,
                                                    grayscale=True)
        pyautogui.moveTo(RunControl.x, RunControl.y, duration=1)

        # Perform a single left-button click.
        pyautogui.click()

        # An even simpler way to do it is to pass the coordinates directly to the click function.
        print("\nMoving and clicking at 'Stop Run' in 3 seconds...")
        time.sleep(3)
        RunSequence = pyautogui.locateCenterOnScreen("StopRun.png",
                                                     confidence=0.90,
                                                     grayscale=True)
        pyautogui.moveTo(RunSequence.x, RunSequence.y, duration=1)
        pyautogui.click()

        # Go back to original window
        current_window.activate()


    def CHIntegrator(self, CH_file:str=None, detector:str=None,
            show_integration:bool=False):
        '''
        This function is built to be called by other python scripts

        This functions reads the CH_file passed. If no CH_file was
        passed, it defaults to a value of 0 and the user will be prompted to
        select one. The CH file used will be integrated by the JSON files passed
        to the **GCController** object. It consists of a block to extract the
        .ch file's time and value data. It then returns a list of chemicals and 
        a list of chemical compositions for each of those chemicals.

        :param CH_file: Filepath (string) for the .ch file to integrate. If not 
            passed, user will be prompted to select one
        :type CH_file: str
        :param detector: Detector to integrate. Used if multiple detectors were
            described in initialization
        :param show_integration: True/False parameter that determines if the 
            data and integrations will be shown. Defaults to false
        :return chemicals: List of chemicals in the order they appear from the
            integration
        :rtype chemicals: list of str
        :return composition: List of compositions. If no calibration factors were
            passed in the JSON parameters file, then the units will be in peak
            area. If calibration factors were passed, then they will have the
            corresponding units.
        :rtype composition: list of floats
        '''
        ### ___________________________________________________________________________________________________
        ### Code block 1: integration algorithms ###
    
        def simple_peak(times, signals, plotting):
            '''
            Function inputs: a time and signal array where the first data points are the start of the integration
            and the last data points are the end of the integration

            Output: a single area value
            '''
            area = []

            # Subtract the baseline created by the start and end points
            s_adj = []

            for i in range(len(times)):

                # Calculate the interpolated baseline value and subtract it off of the s_select value
                s_interp = (signals[-1] - signals[0])/(times[-1] - times[0]) * (times[i] - times[0]) + signals[0]
                s_adj.append(signals[i] - s_interp)

            # Calculate the areas using the trapezoid rule and append it to the area array
            traps = np.trapezoid(s_adj,times) # Note: np.trapezoid is y-data first, then x-data
            area.append(traps*60) # The area from traps will be in min, we need it in s

            if plotting == True:
                # Draw baseline
                t_line = [times[0], times[-1]]
                s_line = [signals[0], signals[-1]]

                plt.plot(t_line,s_line, 'b-')       

            return area

        def simple_split(split, times, signals, plotting):
            '''
            Function inputs: a time split(s) (which can be an array) to decide where to split the chromatograph, the time array for the peak in question,
            and the signal array for the peak in question

            Output: a single area value
            '''
            
            area = [] # Initialize area list

            # Subtract the baseline created by the start and end points
            s_adj = []

            for i in range(len(times)):

                # Calculate the interpolated baseline value and subtract it off of the s_select value
                s_interp = (signals[-1] - signals[0])/(times[-1] - times[0]) * (times[i] - times[0]) + signals[0]
                s_adj.append(signals[i] - s_interp)

            # Initialize lists to split up the parts of the peak (if there is one split time, split the data into two parts)
            t_split = [[] for _ in range(len(split)+1)] 
            s_split= [[] for _ in range(len(split)+1)]

            # Split chromatograph up
            split_index = np.zeros(len(split)) # Initialize list for split indices
            for i in range(len(times)):
                check = False # Reinitialize check variable to determine if it goes into any of the "if times[i] ..." loops
                for j in range(len(split)):
                    if times[i] < split[j]: # This time is in the first part of the chromatograph

                        t_split[j].append(times[i])
                        s_split[j].append(s_adj[i])
                        split_index[j] = i # Just keep increasing this until it is not true, that gives the split index

                        check = True # Mark the check variable
                        break # No need to parse through the "for j in..." loop for this value of i
                
                if check == False: # It is the last section of chromatograph

                    t_split[-1].append(times[i])
                    s_split[-1].append(s_adj[i])

            # Integrate each section of the chromatograph
            for i in range(len(split)+1):
                traps = np.trapezoid(s_split[i],t_split[i]) # Note: np.trapezoid is y-data first, then x-data
                area.append(traps*60) 
                    
            if plotting == True:
                # Draw baseline
                t_line = [times[0], times[-1]]
                s_line = [signals[0], signals[-1]]

                plt.plot(t_line,s_line, 'b-') 

                # Draw split lines
                for i in range(len(split)):
                    t_line = [t_split[i][-1], t_split[i][-1]]
                    s_interp = (signals[-1] - signals[0])/(times[-1] - times[0]) * (t_split[i][-1] - times[0]) + signals[0]
                    s_line = [s_interp, signals[int(split_index[i])]]

                    plt.plot(t_line,s_line,'b-')
            
            return area

        def simple_skim(split, times, signals, plotting):
            '''
            Function inputs: a time stamps(s) (which must be an array of size 2) to decide where to split the chromatograph, the time array for the peak in question,
            and the signal array for the peak in question

            Output: a single area value
            '''

            if len(split) != 2:
                print("Error: number of split times passed to simple_skim is not 2.\n")
                sys.exit(0)

            area = [] # Initialize area list for two sets of peaks

            # Subtract the baseline created by the start and end points
            s_adj = []

            for i in range(len(times)):

                # Calculate the interpolated baseline value and subtract it off of the s_select value
                s_interp = (signals[-1] - signals[0])/(times[-1] - times[0]) * (times[i] - times[0]) + signals[0]
                s_adj.append(signals[i] - s_interp)

            # Integrate overall peak
            traps = np.trapezoid(s_adj,times) # Note: np.trapezoid is y-data first, then x-data
            area.append(traps*60)

            # Identify the signals for the skim
            t_skim = []
            s_skim = []

            for i in range(len(times)):
                if (times[i] >= split[0]) and (times[i] <= split[1]): # in the part to skim
                    t_skim.append(times[i])
                    s_skim.append(s_adj[i])
                elif times[i] > split[1]:
                    break # Exit the for loop

            # Subtract baseline from skim portion
            s_skim_adj = []

            for i in range(len(t_skim)):

                # Calculate the interpolated baseline value and subtract it off of the s_select value
                s_interp = (s_skim[-1] - s_skim[0])/(t_skim[-1] - t_skim[0]) * (t_skim[i] - t_skim[0]) + s_skim[0]
                s_skim_adj.append(s_skim[i] - s_interp)

            # Calculate area of skim and subtract it off of the first area
            traps_skim = np.trapezoid(s_skim_adj,t_skim)
            area[0] -= (traps_skim*60) # Subtracts the skim area off of the original area
            area.append(traps_skim*60)

            if plotting == True:
                # Draw baseline
                t_line = [times[0], times[-1]]
                s_line = [signals[0], signals[-1]]

                plt.plot(t_line,s_line,"b-") 

                # Draw skim line
                t_line = [t_skim[0], t_skim[-1]]
                s_line = [s_skim[0], s_skim[-1]]

                plt.plot(t_line,s_line,"b-")

            return area

        def exp_skim(skim, times, signals, plotting):
            '''
            Function inputs: a time stamps(s) (which must be an array of size 4)
            to decide where to skim the chromatograph and what to fit to the
            exponential skim, the time array for the peak in question,
            and the signal array for the peak in question

            Output: two area values: one for the main peak, one for the 
            exponentially skimmed peak
            '''

            def exp_model(x, scale, gain, offset):
                y = scale * np.exp(gain*x) + offset
                return y


            if len(skim) != 4:
                print("Error: number of skim times passed to exponential_skim is not 4.\n")
                sys.exit(0)

            area = [] # Initialize area list for two sets of peaks

            # Subtract the baseline created by the start and end points
            s_adj = []

            for i in range(len(times)):

                # Calculate the interpolated baseline value and subtract it off of the s_select value
                s_interp = (signals[-1] - signals[0])/(times[-1] - times[0]) * (times[i] - times[0]) + signals[0]
                s_adj.append(signals[i] - s_interp)

            # Integrate overall peak
            traps = np.trapezoid(s_adj,times) # Note: np.trapezoid is y-data first, then x-data
            area.append(traps*60)

            # Extract the portion of the data between the skim

            t_skim = []
            s_skim = []

            for i in range(len(times)):
                if (times[i] >= skim[2]) and (times[i] <= skim[3]): # in the part to skim
                    t_skim.append(times[i])
                    s_skim.append(s_adj[i])
                elif times[i] > skim[3]:
                    break # Exit the for loop

            # Extract portion to fit the exponential to and try to fit it
            try:
                t_fit = []
                s_fit = []

                for i in range(len(times)):
                    if (times[i] >= skim[0]) and (times[i] <= skim[1]): # in the part to fit
                        if (times[i] <= skim[2]) or (times[i] >= skim[3]): # exclude if in skim region
                            t_fit.append(times[i])
                            s_fit.append(s_adj[i])
                    elif times[i] > skim[1]:
                        break # Exit the for loop

                # Fit these data using the model provided above
                scale_guess = s_fit[0] - s_adj[-1]
                gain_guess = -1
                offset_guess = s_adj[-1]
                p0 = [scale_guess, gain_guess, offset_guess]

                popt, pcov = curve_fit(exp_model, t_fit, s_fit, p0=p0) # Don't save the stats
                scale = popt[0]
                gain = popt[1]
                offset = popt[2]

                # Calculate the signals for the exponential baseline
                s_fit_skim = []

                for t in t_skim:
                    s_fit_skim.append(exp_model(t, scale=scale, gain=gain, offset=offset))

            except Exception as e:
                print(f"Exponential Fitting failed with code: {e}")
                print(f"Fitting a straight line instead")

                s_fit_skim = []

                for t in t_skim:
                    temp = (s_skim[-1] - s_skim[0])/(t_skim[-1] - t_skim[0]) * (t - t_skim[0]) + s_skim[0]
                    s_fit_skim.append(temp)

            finally: # Execute whether the fit worked or not

                # Subtract exponential baseline from skim portion
                s_skim_adj = []

                for i in range(len(t_skim)):

                    s_skim_adj.append(s_skim[i] - s_fit_skim[i])

                # Calculate area of skim and subtract it off of the first area
                traps_skim = np.trapezoid(s_skim_adj,t_skim)
                area[0] -= (traps_skim*60) # Subtracts the skim area off of the original area
                area.append(traps_skim*60)

                if plotting == True:
                    # Draw baseline
                    t_line = [times[0], times[-1]]
                    s_line = [signals[0], signals[-1]]

                    plt.plot(t_line,s_line,"b-") 

                    # Add baseline to skim

                    t_plot = []
                    s_plot = []

                    for i in range(len(times)):
                        for j in range(len(t_skim)):
                            if times[i] == t_skim[j]:
                                t_plot.append(t_skim[j])
                                s_interp = (signals[-1] - signals[0])/(times[-1] - times[0]) * (times[i] - times[0]) + signals[0]
                                s_plot.append(s_interp + s_fit_skim[j])

                    # Draw skim line
                    plt.plot(t_plot,s_plot,"b-")

            return area

        def flat_base(base_time, times, signals, plotting):
            '''
            Function inputs: a time split(s) (which can be an array) to decide which point to 
            draw the flat baseline from and the times to start and end the peak, the time array for the peak in question,
            and the signal array for the peak in question

            Output: a single area value
            '''

            area = []

            # Find the signal value at the time to draw the baseline from (middle time input)
            s_base = []

            for i in range(len(times)):

                if times[i] >= base_time:

                    s_base = signals[i]
                    break # Break out of the for loop

            # Calculate the areas using the trapezoid rule and append it to the area array
            traps = np.trapezoid(signals-s_base,times) # Note: np.trapezoid is y-data first, then x-data
            area.append(traps*60) # The area from traps will be in min, we need it in s

            if plotting == True:
                # Draw baseline
                
                s_line = [s_base for i in range(len(times))]

                plt.plot(times,s_line, 'b-')       

            return area

        ### ___________________________________________________________________________________________________
        ### Code block 2: read the .ch file ###

        # If no CH_file was passed, it will default to 0 which will trigger the user to find one
        if CH_file == None:
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
        GC_s_0 = np.array(ch_obj.values) # Signal data from the chromotograph

        # Smooth the data through moving average so that integration starting points are less dependent of noise
        window_size = 30 # How many neighboring points to average with, Justin checked this and it retains features well
        window = np.ones(window_size) / window_size # Create a normalized array for the moving avg filter
        GC_s = np.convolve(GC_s_0, window, "same") # Creates an array of the same size but smoothed

        if show_integration == True:

            plt.figure(figsize=(8,5))
            plt.plot(GC_t,GC_s_0,"k-", label="Raw Data") # Plot raw signal
            plt.plot(GC_t,GC_s,"r-", linewidth = 0.8, label="Smoothed Data") # Plot smoothed signal
            plt.xlabel("Time (min)")
            plt.ylabel("Signal (a.u.)")

        ### ___________________________________________________________________________________________________
        ### Code block 3: read the JSON file and extract the integration parameters ###

        # If no JSON_file was passed, it will default to 0 which will trigger the user to find one
        if self.integration_params == None:
            # Create a root window but keep it hidden
            root = tk.Tk()
            root.withdraw() # Hides the small tkinter window

            # Open the file dialog
            # The 'filetypes' argument filters for specific file extensions
            JSON_file = filedialog.askopenfilename(
                title="Select a JSON file",
                filetypes=(("JSON files", "*.json"), ("All files", "*.*"))
            )

            # Check if the user selected a file or cancelled the dialog
            if not JSON_file:
                print("No JSON file was selected.")
                sys.exit(0)
        elif len(self.integration_params) > 1 and detector==None:
            detector = input(f"Which detector do you want to use: {self.detectors}: ")
            JSON_file = self.integration_params[detector]
        else: # Then both a detector and set of JSON files were passed
            JSON_file = self.integration_params[detector]

        # Read JSON file
        with open(JSON_file, "r") as f:
            json_data = json.load(f) # Load the JSON file into a Python dictionary

        peaks = json_data["events"] # Creates a list in which each peak event is one entry, the dictionary key is the same as the JSON file
        num_peaks = len(peaks) # Store the number of peak events


        ### ___________________________________________________________________________________________________
        ### Code block 4: Call the relevant integrators ###

        chemicals = [] # Initialize chemicals list
        composition = [] # Initialize composition list

        for i in range(num_peaks):
            
            ignored_peaks = [] # Reinitialize ingored peak on each loop

            # Append the chemicals to integrate
            for j in range(len(peaks[i]["chemicals"])):

                # If chemical name is not on the list of ignroed chemicals, append it to the chemicals list
                for m in range(len(peaks[i]["ignored_chemicals"])):
                    if peaks[i]["chemicals"][j] != peaks[i]["ignored_chemicals"][m]:
                        chemicals.append(peaks[i]["chemicals"][j])
                    else:
                        ignored_peaks.append(j)

            # Identify the subset of the GC data to integrate
            t_subset = [] # Initialize times subset array
            s_subset = [] # Initialize signals subset array
            for j in range(len(GC_t)):       
                if (GC_t[j] >= peaks[i]["time_stamps"][0]) and (GC_t[j] <= peaks[i]["time_stamps"][-1]): # In the write time range
                    t_subset.append(GC_t[j])
                    s_subset.append(GC_s[j])
                elif GC_t[j] > peaks[i]["time_stamps"][-1]: # Past the time
                    break # Exit the for loop

            # Identify which algorithm to use
            if peaks[i]["integration_algorithm"] == "simple peak":

                area = simple_peak(t_subset, s_subset, show_integration)
                composition.append(float(area[0]*peaks[i]["cal_factor"][0])) # Append composition to the list  

            elif peaks[i]["integration_algorithm"] == "simple split":

                area = simple_split(peaks[i]["time_stamps"][1:-1], t_subset, s_subset, show_integration)

                # Remove ignored chemicals from the peaks
                for j in range(len(ignored_peaks)):
                    area.pop(ignored_peaks[j])

                # Add the desired areas to output
                for j in range(len(area)):
                    composition.append(float(area[j] * peaks[i]["cal_factor"][j])) # Append composition to the list

            elif peaks[i]["integration_algorithm"] == "simple skim":

                area = simple_skim(peaks[i]["time_stamps"][1:3], t_subset, s_subset, show_integration)
                    # Note: in python indexing, the last digit is non-inclusive, meaning 1:3 indexes the 2nd and 3rd positions

                for j in range(len(peaks[i]["chemicals"])):
                    if peaks[i]["chemicals"][j] not in ignored_peaks:
                        composition.append(float(area[j] * peaks[i]["cal_factor"][j])) # Append composition to the list
                        
            elif peaks[i]["integration_algorithm"] == "exponential skim":

                area = exp_skim(peaks[i]["time_stamps"][1:5], t_subset, s_subset, show_integration)

                for j in range(len(peaks[i]["chemicals"])):
                    if peaks[i]["chemicals"][j] not in ignored_peaks:
                        composition.append(float(area[j] * peaks[i]["cal_factor"][j])) # Append composition to the list
            
            elif peaks[i]["integration_algorithm"] == "flat base":
                ''' For the flat base algorithm the time stamps are start time,
                time to set as the baseline point, and end time in that order

                No ignored chemicals allowed, only integrates one chemical
                '''

                area = flat_base(peaks[i]["time_stamps"][1], t_subset, s_subset, show_integration)
                composition.append(float(area[0]*peaks[i]["cal_factor"][0])) # Append composition to the list  

            else:
                print("Error: unknown integration algorithm for peak event {i}.\n")
                sys.exit(0)
            
        # Finalize the plot if it was requested
        if show_integration == True:
            plt.legend()
            plt.show()
            

        return chemicals, composition
    

    def SequenceIntegrator(self, sequence_folder=None, save_folder=None):
        '''
        Integrates a full sequence of GC data.

        :param sequence_folder: Folder that contains the sequence data (e.g.,
            a .S file)
        :type sequence_folder: string
        :param save_folder: Folder to save a CSV in. If a folder isn't passed, then
            the header for the CSV and compositions lists will be returned as outputs
        :rtype chemicals: list of str
        :return composition: List of compositions. If no calibration factors were
            passed in the JSON parameters file, then the units will be in peak
            area. If calibration factors were passed, then they will have the
            corresponding units.
        :rtype composition: list of floats
        '''

        ### ______________________________________________________________________________________________________________
        ### Code block 1: Obtain the necessary JSON files to handle the integrations from the user ###
        # If no JSON_file was passed, it will default to 0 which will trigger the user to find one

        #### THE FILE DIALOG FOR THIS NEXT SECTION IS BUGGY FOR SOME REASON???
        
        if self.integration_params == None: # No JSONs passed

            # Determine if the user needs both TCD and FID signals:
            detector_count = int(input("Are there one or two detectors in use on the GC (1 or 2): "))
            JSON_files = [] # Initialize array to store JSON files in
            detector_names = [] # Initialize detector names

            for i in range(detector_count):

                detector_names.append(input("Type of Detector (FID or TCD): "))

                # Create a root window but keep it hidden
                root = tk.Tk()
                root.withdraw() # Hides the small tkinter window
                
                # Open the file dialog
                # The 'filetypes' argument filters for specific file extensions
                temp = filedialog.askopenfilename(
                    title="Select a JSON file for the {detector_names[-1]}",
                    filetypes=(("JSON files", "*.json"), ("All files", "*.*"))
                )

                # Check if the user selected a file or cancelled the dialog
                if not temp:
                    print("No JSON file was selected.")
                    sys.exit(0)
                
                JSON_files.append(temp)
        else:
            detector_names = self.detectors
            detector_count = len(self.detectors)
            JSON_files = []
            for i in self.detectors:
                JSON_files.append(self.integration_params[i])

        events = [[] for _ in range(detector_count)] # Initialize the events list for the JSON files

        for i in range(detector_count):
            with open(JSON_files[i], "r") as f:
                json_data = json.load(f) # Load the JSON file into a Python dictionary

            events[i] = json_data["events"] # Creates a list in which each peak event is one entry, the dictionary key is the same as the JSON file
        
        ### ______________________________________________________________________________________________________________
        ### Code block 2: Obtain a sequence folder ###

        if sequence_folder == None: # No folder passed
            sequence_folder = filedialog.askdirectory(title="Select a GC Sequence Folder")

            # Check if a folder was selected (the user didn't click "Cancel")
            if sequence_folder:
                # Update the label to show the selected folder path
                print(f"The user selected the folder: {sequence_folder}")
            else:
                print("The user canceled the dialog.")

        folder_contents = os.listdir(sequence_folder) # Obtain a list of contents from that folder


        ### ______________________________________________________________________________________________________________
        ### Code block 3: Iterate through each folder looking for a .D file, count the number of .ch files there are, iteratively integrate each ###

        line_count = 0 # Initialize line count

        # Count number of lines
        for i in range(len(folder_contents)):
            if (".D" in folder_contents[i]) and ("post" not in folder_contents[i].lower()) and ("shutdown" not in folder_contents[i].lower()):
                line_count += 1 # Increment counter

        header = ["Line #"] # Initialize header for CSV
        header.append("Folder name")

        for i in range(detector_count): # For each detector
            for j in range(len(events[i])): # For each peak event in that dector's JSON file
                for m in range(len(events[i][j]["chemicals"])): # For each list of chemical in that peak event list
                    for n in range(len(events[i][j]["ignored_chemicals"])):
                        if events[i][j]["chemicals"][m] != events[i][j]["ignored_chemicals"][n]:
                            header.append(events[i][j]["chemicals"][m]) # Append each chemical name onto the list


        compositions = [] # Initialize composition list

        # Search for the folder containing: (1) .D extension, (2) the right line number, e.g., for line 3: 003
        for i in range(line_count):
            # Search for "...00{i+1}... .D"

            for j in range(len(folder_contents)):

                if (".D" in folder_contents[j]) and (f"F-{i+1:0{3}d}" in folder_contents[j]): 
                                    
                    # Change directory to that folder, obtain folder contents
                    d_folder = os.path.join(sequence_folder, folder_contents[j])
                    d_contents = os.listdir(d_folder)

                    c = [i+1] # Start creating a list for the current composition, start with the line number
                    c.append(d_folder)
                    
                    # Iterate through detectors
                    for m in range(detector_count):                  
                        
                        check = 0 # Set checker variable
                        for n in range(len(d_contents)):
                            
                            if (detector_names[m] in d_contents[n]) and (".ch" in d_contents[n]): # Found a .CH file that matches the detector

                                ch = os.path.join(d_folder, d_contents[n])
                                check = 1 # Mark the checker

                                # Pass .CH file to the integrator
                                a, b = self.CHIntegrator(CH_file = ch,
                                        detector=detector_names[m]) # a = chemicals list (don't use), b = composition list
                                for p in range(len(b)):
                                    c.append(b[p])

                                break # Detector found, bypass the rest of "for n in ..." loop

                        if check == 0 : # No detectors were found with that name

                            print("Error: no matching detector names found in .CH files")
                            sys.exit(0)

                    compositions.append(c) # Append the row of composition data

                    break # Don't continue to search for the current line #


        ### ______________________________________________________________________________________________________________
        ### Code block 4: Create a CSV and have the user select a location to save it ###

        if save_folder == None: # Output data as a return

            return header,compositions

        
        else: # Save the CSV here
            
            sequence_name = os.path.basename(sequence_folder)
            timestamp_str = datetime.now().strftime("%Y%m%d %H_%M_%S")
            csv_file_name = sequence_name + ".csv" # CSV file name
            save_filepath = Path(save_folder) / csv_file_name

            with open(save_filepath, "w", newline = "") as f:
                writer = csv.writer(f, delimiter=",")
                

                # Write meta data based on this code

                writer.writerow([f"# Sequence file chosen: {sequence_name}"])
                writer.writerow([f"# Processing date: {timestamp_str}"])
                writer.writerow([f"# "])

                # Write meta data based on the JSON files

                header2= ["# Events", "JSON File", "Detector", "Chemicals passed","Calibration factors (units/peak area)",
                    "Integration algorithm", "Time Stamps", "Ignored Chemicals", "Output Units", "User",
                    "Calibration Date"]
                writer.writerow(header2)
                event_count = 0
                for i in range(len(events)):
                    for j in range(len(events[i])):
                        event_count += 1

                        event_str = [event_count]
                        event_str.append(os.path.basename(JSON_files[i]))
                        event_str.append(events[i][j]["detector"])
                        event_str.append(events[i][j]["chemicals"])
                        event_str.append(events[i][j]["cal_factor"])
                        event_str.append(events[i][j]["integration_algorithm"])
                        event_str.append(events[i][j]["time_stamps"])
                        event_str.append(events[i][j]["ignored_chemicals"])
                        event_str.append(events[i][j]["meta_output_units"])
                        event_str.append(events[i][j]["meta_user"])
                        event_str.append(events[i][j]["meta_cal_date"])

                        writer.writerow(event_str)

                writer.writerow("") # Insert blank row

                writer.writerow(header)

                for i in range(line_count):
                    writer.writerow(compositions[i])



            