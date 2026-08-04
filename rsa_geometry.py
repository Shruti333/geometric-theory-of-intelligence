from __future__ import annotations
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr

def compute_rdm(activations: np.ndarray) -> np.ndarray:
    """
    Computes a Representational Dissimilarity Matrix (RDM) using 1 - Pearson Correlation.
    RDM[i, j] measures how dissimilar stimulus i is from stimulus j in neural space.
    """
    return squareform(pdist(activations, metric='correlation'))

def compare_geometries(rdm_bio: np.ndarray, rdm_art: np.ndarray) -> tuple[float, float]:
    """
    Evaluates the geometric alignment between biological and artificial RDMs 
    using Spearman rank correlation of the upper triangular distance vectors.
    """
    # Take upper triangular indices excluding the main diagonal
    triu_idx = np.triu_indices_from(rdm_bio, k=1)
    
    vector_bio = rdm_bio[triu_idx]
    vector_art = rdm_art[triu_idx]
    
    corr, p_value = spearmanr(vector_bio, vector_art)
    return corr, p_value