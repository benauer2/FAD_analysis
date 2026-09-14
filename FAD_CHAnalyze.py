from GC_handler.GC_utilities import GCController

JSON_files = []
# JSON_files.append(r"C:\Users\cemshauerlocal\Documents\GitHub\velma-reactor\FAD_IntegrationParams_FID.json")
JSON_files.append(r"C:\Users\cemshauerlocal\Documents\GitHub\REACTION-REPORT-GENERATOR\GC_handler\FAD_IntegrationParams_FID_cal.json")
JSON_files.append(r"C:\Users\cemshauerlocal\Documents\GitHub\REACTION-REPORT-GENERATOR\GC_handler\FAD_IntegrationParams_TCD_cal.json")


GC = GCController(name="Batman", detectors=["FID", "TCD"], JSON_files=JSON_files)

chemicals, compositions = GC.CHIntegrator(detector="FID", show_integration=True)

print(chemicals, compositions)


