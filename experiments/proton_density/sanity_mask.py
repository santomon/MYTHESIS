import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.widgets import PolygonSelector, Button
from matplotlib.path import Path
import pydicom

IMAGE_PATH = "./data/proton_density_b1s_1.nii.gz" 
MASK_PATH = "./data/proton_density_b1s_1_mask.nii.gz" 


def auto_contrast(image, percentile_low=2, percentile_high=98):
    """Apply auto-contrasting using percentile clipping"""
    vmin = np.percentile(image, percentile_low)
    vmax = np.percentile(image, percentile_high)
    image_clipped = np.clip(image, vmin, vmax)
    # Normalize to 0-1
    image_normalized = (image_clipped - vmin) / (vmax - vmin)
    return image_normalized

def plot_image_with_mask(image_path, mask_path):
    """Plot NIfTI image with mask contour overlay"""
    
    # Load images
    img_nii = nib.load(image_path)
    mask_nii = nib.load(mask_path)
    
    img_data = img_nii.get_fdata()
    mask_data = mask_nii.get_fdata()
    
    print(f"Image shape: {img_data.shape}")
    print(f"Mask shape: {mask_data.shape}")
    
    # Handle 3D volumes - take middle slice
    if img_data.ndim == 3:
        mid_slice = img_data.shape[2] // 2
        img_slice = img_data[:, :, mid_slice]
        mask_slice = mask_data[:, :, mid_slice]
        print(f"Displaying slice {mid_slice} of {img_data.shape[2]}")
    else:
        img_slice = img_data.squeeze()
        mask_slice = mask_data.squeeze()
    
    # Apply auto-contrast
    img_contrasted = auto_contrast(img_slice)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Display image
    ax.imshow(img_contrasted, cmap='gray', interpolation='nearest')
    
    # Draw mask contours
    # Find contours at mask boundary
    contours = ax.contour(mask_slice, levels=[0.5], colors='red', linewidths=2)
    
    ax.set_title('Image with Mask Contour Overlay')
    ax.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print("\nMask statistics:")
    print(f"  Mask coverage: {np.sum(mask_slice > 0)} pixels ({100*np.mean(mask_slice > 0):.2f}%)")

if __name__ == "__main__":
    plot_image_with_mask(IMAGE_PATH, MASK_PATH)
