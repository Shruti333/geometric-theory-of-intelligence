from __future__ import annotations
import numpy as np
import torch
from transformer_lens import HookedTransformer

def extract_transformer_spatial_activations(
    model: HookedTransformer, 
    locations: list[tuple[int, int]], 
    layer: int = 8
) -> np.ndarray:
    """
    Harvests residual stream representation vectors for spatial prompts in GPT-2.
    """
    hook_name = f"blocks.{layer}.hook_resid_post"
    activations = []

    print(f"Harvesting spatial activations from GPT-2 Layer {layer}...")
    for x, y in locations:
        prompt = f"Target grid location position point coordinates ({x}, {y})"
        
        with torch.no_grad():
            _, cache = model.run_with_cache(prompt, names_filter=[hook_name])
            
            # Extract final token residual stream vector: [d_model]
            act = cache[hook_name][0, -1, :].cpu().numpy()
            activations.append(act)

    return np.array(activations)  # Shape: [Num_Locations, d_model]