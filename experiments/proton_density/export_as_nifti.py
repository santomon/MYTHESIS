import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.widgets import PolygonSelector, Button
from matplotlib.path import Path
import pydicom
from env import env


def dicom_to_nifti(dicom_path: str, output_path: str):
    """Convert DICOM to NIfTI format for ITK-SNAP"""

    # Load DICOM
    dcm = pydicom.dcmread(dicom_path)
    image_data = dcm.pixel_array

    # Get voxel spacing if available
    try:
        pixel_spacing = dcm.PixelSpacing  # [row_spacing, col_spacing]
        slice_thickness = dcm.SliceThickness
        spacing = [pixel_spacing[0], pixel_spacing[1], slice_thickness]
        print(f"Voxel spacing: {spacing} mm")
    except AttributeError:
        spacing = [1.0, 1.0, 1.0]
        print("Warning: No spacing information found, using [1.0, 1.0, 1.0]")

    # Handle 2D image (add third dimension if needed)
    if image_data.ndim == 2:
        image_data = image_data[:, :, np.newaxis]
        print(f"2D image detected, shape: {image_data.shape}")

    # Create affine matrix (basic, can be refined with orientation info)
    affine = np.diag([spacing[0], spacing[1], spacing[2], 1.0])

    # Create NIfTI image
    nifti_img = nib.Nifti1Image(image_data, affine)

    # Save
    nib.save(nifti_img, output_path)
    print(f"✓ NIfTI file saved to: {output_path}")
    print(f"  Image shape: {image_data.shape}")
    print(f"  Data type: {image_data.dtype}")


if __name__ == "__main__":
    dicom_to_nifti(env.proton_density_dicom_path, env.proton_density_nifti_path)
    dicom_to_nifti(env.perfusion_test_dicom_path, env.perfusion_test_nifti_path)
    print("\nYou can now load this file in ITK-SNAP!")
