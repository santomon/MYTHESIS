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

# Load NIfTI files
print("Loading image and mask...")

image = load_nifti_as_2d(IMAGE_PATH)
mask = load_nifti_as_2d(MASK_PATH)

mask_bool = mask.astype(bool)

# Get coordinates
if image.ndim == 2:
    h, w = image.shape
    y_coords, x_coords = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    
    # Calculate center of mask (centroid of masked region)
    mask_y, mask_x = np.where(mask_bool)
    center_x = np.mean(mask_x)
    center_y = np.mean(mask_y)
    
    print(f"\nMask center: ({center_x:.2f}, {center_y:.2f})")
    print(f"Number of masked pixels: {len(mask_x)}")
    
    # Select only masked pixels
    intensities = image[mask_bool]
    
    # Prepare features: coordinates relative to center
    X = np.column_stack([mask_x - center_x, mask_y - center_y])
    
    # Fit linear model
    print("Fitting linear model...")
    model = LinearRegression()
    model.fit(X, intensities)
    
    # Extract gradient and value at center
    gradient_x, gradient_y = model.coef_
    value_at_center = model.intercept_
    
    print("\n=== Fitted Parameters ===")
    print(f"Center point: ({center_x:.2f}, {center_y:.2f})")
    print(f"Value at center: {value_at_center:.4f}")
    print(f"Gradient vector: ({gradient_x:.6f}, {gradient_y:.6f})")
    print(f"Gradient magnitude: {np.sqrt(gradient_x**2 + gradient_y**2):.6f}")
    print(f"Gradient direction (degrees): {np.degrees(np.arctan2(gradient_y, gradient_x)):.2f}")


    params = {
            "gradient_x": gradient_x,
            "gradient_y": gradient_y,
            "value_at_center": value_at_center
            }

    with open(env.linear_params_path, "w") as f:
        json.dump(params, f)
    
    # Calculate R² score
    r2 = model.score(X, intensities)
    print(f"R² score: {r2:.6f}")
    
    
    # Generate bias map for current image
    bias_map = generate_gradient_map(
        (h, w), center_x, center_y, gradient_x, gradient_y, value_at_center
    )
    
    # Plot
    plt.figure(figsize=(15, 5))
    
    # Original image with center point
    plt.subplot(1, 3, 1)
    plt.imshow(image, cmap='gray')
    plt.plot(center_x, center_y, 'r+', markersize=15, markeredgewidth=2)
    plt.title('Original Image\n(red + = mask center)')
    plt.colorbar()
    plt.axis('off')
    
    # Mask
    plt.subplot(1, 3, 2)
    plt.imshow(mask, cmap='gray')
    plt.plot(center_x, center_y, 'r+', markersize=15, markeredgewidth=2)
    plt.title('Mask')
    plt.colorbar()
    plt.axis('off')
    
    # Bias map with gradient arrow
    plt.subplot(1, 3, 3)
    plt.imshow(bias_map, cmap='gray')
    plt.plot(center_x, center_y, 'r+', markersize=15, markeredgewidth=2)
    
    # Draw gradient arrow
    arrow_scale = 50  # Scale for visualization
    plt.arrow(center_x, center_y, 
             gradient_x * arrow_scale, gradient_y * arrow_scale,
             color='red', width=2, head_width=8, head_length=8, alpha=0.7)
    
    plt.title(f'Gradient Map\nCenter: ({center_x:.1f}, {center_y:.1f})\n'
             f'Gradient: ({gradient_x:.4f}, {gradient_y:.4f})\n'
             f'Value@center: {value_at_center:.2f}, R²={r2:.4f}')
    plt.colorbar()
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('out/gradient_map.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved as 'out/gradient_map.png'")
    plt.show()
    
    # Save parameters for reuse
    params = {
        'center_x': center_x,
        'center_y': center_y,
        'gradient_x': gradient_x,
        'gradient_y': gradient_y,
        'value_at_center': value_at_center,
        'r2': r2
    }
    
    print("\n=== To apply to another image ===")
    print("Use generate_gradient_map() with the new mask center:")
    print("new_center = np.mean(np.where(new_mask), axis=1)")
    print("bias_map = generate_gradient_map(new_image.shape, new_center[1], new_center[0],")
    print(f"                                 {gradient_x:.6f}, {gradient_y:.6f}, {value_at_center:.4f})")

else:
    print("Error: Expected 2D image. For 3D images, please select a slice first.")
