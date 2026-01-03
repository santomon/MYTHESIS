import pydicom
import json

PARAMS_PATH = "./data/linear_params.json"
DICOM_PATH = r"./data/T1_CMR1001.5 T1_CMR1001.5/Anonymous Study/MR Dynamic Stress VP 120dyn SSFP_SAX_b1s/MR000022.dcm"  

with open(PARAMS_PATH, "r") as f:
    params = json.load(f)

dicom = pydicom.dcmread(DICOM_PATH)
pixel_array = dicom.pixel_array






