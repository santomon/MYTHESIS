import json

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from sklearn.linear_model import LinearRegression
from utils import generate_gradient_map
from env import env
from utils import load_nifti_as_2d

IMAGE_PATH = env.proton_density_nifti_path
MASK_PATH = env.proton_density_mask_nifti_path
PARAMS_PATH = env.linear_params_path

# Load NIfTI files
print("Loading image and mask...")

image = load_nifti_as_2d(IMAGE_PATH)
mask = load_nifti_as_2d(MASK_PATH)

with open(PARAMS_PATH, "r") as f:
    params = json.load(f)



