from __future__ import annotations
import numpy as np
import torch
from transformer_lens import HookedTransformer
from src.biological_grid import generate_biological_grid_responses
from src.artificial_world import extract_transformer_spatial_activations
from src.rsa_geometry import compute_rdm, compare_geometries

# Define a 4x4 spatial grid coordinate map
GRID_LOCATIONS = [
    (x, y) for x in range(4) for y in range(4)
]

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using compute device: {device}")

    # 1. Generate Biological Grid Cell Manifold
    print("1. Simulating biological entorhinal grid cell responses...")
    coords = np.array(GRID_LOCATIONS)
    bio_firings = generate_biological_grid_responses(coords, num_cells=256)
    rdm_bio = compute_rdm(bio_firings)

    # 2. Extract Artificial Language Model Spatial Manifold
    print("\n2. Extracting spatial representations from GPT-2...")
    model = HookedTransformer.from_pretrained("gpt2-small", device=device)
    art_activations = extract_transformer_spatial_activations(model, GRID_LOCATIONS, layer=8)
    rdm_art = compute_rdm(art_activations)

    # 3. Representational Similarity Analysis (RSA)
    print("\n3. Executing Representational Similarity Analysis (RSA)...")
    rsa_correlation, p_value = compare_geometries(rdm_bio, rdm_art)

    print("\n--- Project Analysis: Unified Geometry of Intelligence ---")
    print(f"Biological Grid Cell RDM Shape : {rdm_bio.shape}")
    print(f"Artificial Transformer RDM Shape: {rdm_art.shape}")
    print(f"\nRepresentational Geometric Alignment (RSA Score): {rsa_correlation:.4f}")
    print(f"Statistical Significance (p-value): {p_value:.4e}")

    if rsa_correlation > 0.2:
        print("\nConclusion: Conserved World Model Geometry detected! Artificial and biological networks share a similar spatial metric space.")
    else:
        print("\nConclusion: Divergent spatial geometries between artificial and biological manifolds.")

if __name__ == "__main__":
    main()