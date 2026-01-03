import numpy as np


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
