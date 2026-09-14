from GC_handler.GC_utilities import GCController
from tkinter import filedialog

JSON_files = []
JSON_files.append(r"C:\Users\cemshauerlocal\Documents\GitHub\REACTION-REPORT-GENERATOR\GC_handler\FAD_IntegrationParams_FID.json")
JSON_files.append(r"C:\Users\cemshauerlocal\Documents\GitHub\REACTION-REPORT-GENERATOR\GC_handler\FAD_IntegrationParams_TCD.json")


save_folder = filedialog.askdirectory(title="Select a Folder to save the CSV output")

# Check if a folder was selected (the user didn't click "Cancel")
if save_folder:
    # Update the label to show the selected folder path
    print(f"The user selected the folder: {save_folder}")
else:
    print("The user canceled the dialog.")

GC = GCController(name="Batman", detectors=["FID", "TCD"], JSON_files=JSON_files)


GC.SequenceIntegrator(save_folder=save_folder)


