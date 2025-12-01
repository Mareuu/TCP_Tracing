"""
TCP Object Detection Utilities

Object detection for ARC grids, providing functionality to identify and
analyze objects (connected components) in grids.
"""

import numpy as np
from typing import List, Dict, Any
from tcp_utils import Color, find_connected_components


def extract_objects_from_grid(grid: np.ndarray, background: int = Color.BLACK,
                               connectivity: int = 4) -> List[Dict[str, Any]]:
    """
    Extract objects (connected components) from a grid.

    This function identifies distinct objects in an ARC grid by finding
    connected components of non-background pixels.

    Args:
        grid: 2D numpy array representing the grid
        background: Color value to treat as background (default: Color.BLACK)
        connectivity: 4 or 8, for 4-way or 8-way connectivity (default: 4)

    Returns:
        List of dictionaries, each representing an object with properties:
        - 'id': Unique identifier for the object
        - 'size': Number of pixels in the object
        - 'colors': List of colors present in the object
        - 'shape': Type descriptor (always 'connected_component' in this implementation)
    """
    objects = []

    try:
        # Find all connected components (monochromatic)
        components = find_connected_components(
            grid,
            background=background,
            connectivity=connectivity,
            monochromatic=True
        )

        # Extract properties for each component
        for i, component in enumerate(components):
            # Get all non-background pixels
            non_background_mask = component != background
            non_background_pixels = component[non_background_mask]

            if len(non_background_pixels) > 0:
                # Get unique colors in this object
                colors = list(set(non_background_pixels.flatten()))

                objects.append({
                    'id': i,
                    'size': len(non_background_pixels),
                    'colors': colors,
                    'shape': 'connected_component'
                })

    except Exception as e:
        # Return empty list if extraction fails
        print(f"Warning: Object extraction failed: {e}")
        return []

    return objects


def get_object_bounding_box(grid: np.ndarray, background: int = Color.BLACK) -> tuple:
    """
    Get the bounding box of non-background pixels in a grid.

    Args:
        grid: 2D numpy array
        background: Color value to treat as background

    Returns:
        Tuple of (min_row, min_col, max_row, max_col)
    """
    non_background_coords = np.argwhere(grid != background)

    if len(non_background_coords) == 0:
        return None

    min_row = non_background_coords[:, 0].min()
    max_row = non_background_coords[:, 0].max()
    min_col = non_background_coords[:, 1].min()
    max_col = non_background_coords[:, 1].max()

    return (min_row, min_col, max_row, max_col)


def get_object_dimensions(grid: np.ndarray, background: int = Color.BLACK) -> tuple:
    """
    Get the dimensions (height, width) of the bounding box containing
    all non-background pixels.

    Args:
        grid: 2D numpy array
        background: Color value to treat as background

    Returns:
        Tuple of (height, width) or (0, 0) if no non-background pixels
    """
    bbox = get_object_bounding_box(grid, background)

    if bbox is None:
        return (0, 0)

    min_row, min_col, max_row, max_col = bbox
    height = max_row - min_row + 1
    width = max_col - min_col + 1

    return (height, width)
