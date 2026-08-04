from __future__ import annotations
import numpy as np

def generate_biological_grid_responses(
    coords: np.ndarray, 
    num_cells: int = 128, 
    seed: int = 42
) -> np.ndarray:
    """
    Simulates population firing rates of biological grid cells across 2D spatial coordinates.
    """
    np.random.seed(seed)
    responses = []
    
    # Random spatial frequencies, orientations, and phases for grid cell units
    frequencies = np.random.uniform(0.8, 2.5, num_cells)
    orientations = np.random.uniform(0, 2 * np.pi, num_cells)
    phases = np.random.uniform(0, 2 * np.pi, (num_cells, 2))

    for x, y in coords:
        cell_firings = []
        pos = np.array([x, y])
        
        for i in range(num_cells):
            f = frequencies[i]
            theta = orientations[i]
            
            # 3 wave vectors rotated at 60 degrees (120° apart) to form hexagonal grids
            k1 = f * np.array([np.cos(theta), np.sin(theta)])
            k2 = f * np.array([np.cos(theta + np.pi/3), np.sin(theta + np.pi/3)])
            k3 = f * np.array([np.cos(theta + 2*np.pi/3), np.sin(theta + 2*np.pi/3)])
            
            # Biological firing rate approximation
            val = (np.cos(np.dot(k1, pos) + phases[i, 0]) + 
                   np.cos(np.dot(k2, pos) + phases[i, 1]) + 
                   np.cos(np.dot(k3, pos)))
            
            # Rectified firing (neurons cannot have negative firing rates)
            cell_firings.append(np.maximum(0.0, val))
            
        responses.append(cell_firings)
        
    return np.array(responses)  # Shape: [Num_Locations, Num_Neurons]