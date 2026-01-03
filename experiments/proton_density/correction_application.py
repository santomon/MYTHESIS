import json
from copy import deepcopy

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from sklearn.linear_model import LinearRegression
from utils import generate_gradient_map
from env import env
from utils import load_nifti_as_2d, generate_gradient_map

IMAGE_PATH = env.perfusion_test_nifti_path
MASK_PATH = env.perfusion_test_mask_nifti_path
PARAMS_PATH = env.linear_params_path

# Load NIfTI files
print("Loading image and mask...")

image = load_nifti_as_2d(IMAGE_PATH)
mask = load_nifti_as_2d(MASK_PATH)

with open(PARAMS_PATH, "r") as f:
    params = json.load(f)

bin_mask = mask == 1

mask_y, mask_x = np.where(bin_mask)
center_x = np.mean(mask_x)
center_y = np.mean(mask_y)

shape = bin_mask.shape


gradient_image = generate_gradient_map(shape, center_x=center_x, center_y=center_y, **params)

gradient_image_non_zero_mask = gradient_image != 0

image = image.astype(float)
normalized_image = deepcopy(image)
# normalized_image[gradient_image_non_zero_mask] = image[gradient_image_non_zero_mask] / gradient_image[gradient_image_non_zero_mask]
normalized_image = image / gradient_image
normalized_image[mask == 0] = 0




print(normalized_image)

plt.imshow(normalized_image)
plt.show()
