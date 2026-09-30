"""Simulated entorhinal grid-cell population + spatial reference RDMs.

IMPORTANT: this is a model of grid cells, not recorded neural data. Any conclusion is about
this idealised code, not about real entorhinal cortex.
"""
import warnings

import numpy as np
from scipy.spatial.distance import pdist, squareform


def grid_coords(grid_size):
    """All integer (x, y) points on a grid_size x grid_size arena -> (N, 2)."""
    xs, ys = np.meshgrid(np.arange(grid_size), np.arange(grid_size), indexing="ij")
    return np.stack([xs.ravel(), ys.ravel()], axis=1)


def euclidean_rdm(coords):
    """Pure Euclidean-distance RDM: the 'any spatial layout' reference model."""
    return squareform(pdist(coords.astype(float), metric="euclidean"))


def make_grid_population(coords, n_modules=4, cells_per_module=50,
                         base_scale=3.0, scale_ratio=1.42,
                         noise_sd=0.05, seed=0):
    """Firing rates of a multi-module grid-cell population at each coordinate.

    Each module has its own grid spacing (lambda, in arena units) and random
    orientation; cells within a module differ by spatial phase.
    Returns an array of shape (N_positions, n_modules * cells_per_module).
    """
    rng = np.random.default_rng(seed)
    coords = coords.astype(float)
    rates = []
    for m in range(n_modules):
        lam = base_scale * scale_ratio ** m
        theta = rng.uniform(0, np.pi / 3)
        k_mag = 4 * np.pi / (np.sqrt(3) * lam)
        ks = np.array([[np.cos(theta + j * np.pi / 3), np.sin(theta + j * np.pi / 3)]
                       for j in range(3)]) * k_mag                      # (3, 2)
        a1 = lam * np.array([np.cos(theta), np.sin(theta)])             # lattice vectors
        a2 = lam * np.array([np.cos(theta + np.pi / 3), np.sin(theta + np.pi / 3)])
        for _ in range(cells_per_module):
            u, v = rng.uniform(0, 1, size=2)
            phase = u * a1 + v * a2                                     # phase inside unit cell
            g = np.cos((coords - phase) @ ks.T).sum(axis=1)             # in [-1.5, 3]
            rates.append((g + 1.5) / 4.5)                               # -> [0, 1]
    rates = np.stack(rates, axis=1)
    if noise_sd > 0:
        rates = rates + rng.normal(0, noise_sd, size=rates.shape)
    return rates


def real_rate_maps_to_rdm(rate_maps, grid_size):
    """RDM from REAL recorded grid-cell rate maps, on the same grid_size x grid_size positions.

    rate_maps: array (n_cells, H, W) of firing rates over a 2D open-field arena;
    axis 1 is treated as x and axis 2 as y; NaN marks unvisited bins. Maps are
    block-averaged down to grid_size x grid_size so positions line up with the
    GPT-2 stimuli (x-major order, same as grid_coords). NaN bins are filled with
    that cell's mean rate (a warning reports how many).

    You must preprocess your dataset into this array yourself. The mapping from the
    arena to digits 0..grid_size-1 is arbitrary, and real grid spacings relative to
    arena size differ from the simulation, so treat results as exploratory.
    """
    from rsa_geometry import correlation_rdm
    rm = np.asarray(rate_maps, dtype=float)
    if rm.ndim != 3:
        raise ValueError("rate_maps must have shape (n_cells, H, W)")
    n_cells, H, W = rm.shape
    if H < grid_size or W < grid_size:
        raise ValueError("rate maps are smaller than the target grid")
    xi, yi = np.array_split(np.arange(H), grid_size), np.array_split(np.arange(W), grid_size)
    binned = np.full((n_cells, grid_size, grid_size), np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for a, xs in enumerate(xi):
            for b_, ys in enumerate(yi):
                binned[:, a, b_] = np.nanmean(rm[:, xs][:, :, ys], axis=(1, 2))
        flat = binned.reshape(n_cells, -1)
        cell_mean = np.nanmean(flat, axis=1)
    keep = ~np.isnan(cell_mean)
    flat, cell_mean = flat[keep], cell_mean[keep]
    nan_frac = float(np.isnan(flat).mean())
    flat = np.where(np.isnan(flat), cell_mean[:, None], flat)
    flat = flat[flat.std(axis=1) > 0]
    print(f"Real data: {flat.shape[0]} usable cells (of {n_cells}); "
          f"{nan_frac:.1%} of bins were unvisited and filled with the cell mean.")
    return correlation_rdm(flat.T)