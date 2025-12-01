"""
TCP Utility Functions

Utility functions for ARC task evaluation and feedback generation.
"""

import numpy as np
from scipy.ndimage import label


class Color:
    """
    Enum for ARC colors

    Provides color constants for the 10 colors used in ARC puzzles.
    """
    BLACK = 0
    BLUE = 1
    RED = 2
    GREEN = 3
    YELLOW = 4
    GREY = 5
    GRAY = 5
    PINK = 6
    ORANGE = 7
    TEAL = 8
    MAROON = 9
    TRANSPARENT = 0  # Alias for BLACK
    BACKGROUND = 0   # Alias for BLACK

    ALL_COLORS = [BLACK, BLUE, RED, GREEN, YELLOW, GREY, PINK, ORANGE, TEAL, MAROON]
    NOT_BLACK = [BLUE, RED, GREEN, YELLOW, GREY, PINK, ORANGE, TEAL, MAROON]


def find_connected_components(
    grid, background=Color.BLACK, connectivity=4, monochromatic=True
):
    """
    Find the connected components in the grid.

    Args:
        grid: 2D numpy array representing the grid
        background: Color value to treat as background (default: Color.BLACK)
        connectivity: 4 or 8, for 4-way or 8-way connectivity
        monochromatic: If True, each component has only one color. If False,
                      components can include multiple colors.

    Returns:
        List of connected components, where each component is a numpy array
    """
    if connectivity == 4:
        structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    elif connectivity == 8:
        structure = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    else:
        raise ValueError("Connectivity must be 4 or 8.")

    if not monochromatic:
        # Allow multiple colors in a connected component
        labeled, n_objects = label(grid != background, structure)
        connected_components = []
        for i in range(n_objects):
            connected_component = grid * (labeled == i + 1) + background * (labeled != i + 1)
            connected_components.append(connected_component)
        return connected_components
    else:
        # Only allow one color per connected component
        connected_components = []
        for color in set(grid.flatten()) - {background}:
            labeled, n_objects = label(grid == color, structure)
            for i in range(n_objects):
                connected_component = grid * (labeled == i + 1) + background * (labeled != i + 1)
                connected_components.append(connected_component)
        return connected_components
