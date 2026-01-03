import numpy as np
import nibabel as nib


def generate_gradient_map(
    shape: tuple,
    center_x: tuple[float, float],
    center_y: tuple[float, float],
    gradient_x: tuple[float, float],
    gradient_y: tuple[float, float],
    value_at_center: float,
):
    """Generate a gradient map for any image shape and center location."""
    h, w = shape
    y, x = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")

    # Calculate intensity based on distance from center
    dx = x - center_x
    dy = y - center_y
    gradient_map = value_at_center + gradient_x * dx + gradient_y * dy

    return gradient_map



def load_nifti_as_2d(pth: str):
    data = nib.load(pth)
    fdata = data.get_fdata()
    if fdata.ndim == 3 and fdata.shape[2] > 1:
        raise ValueError(f"3D fdata detected with shape {fdata.shape}, while trying to load as 2D")
    if fdata.ndim == 3 and fdata.shape[2] == 1:
        fdata = fdata[:, :, 0]
        print("3D of size 1... treating as 2D")
    return fdata
