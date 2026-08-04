# Unified Geometry of Intelligence: Comparative Analysis of World Models in Biological and Artificial Networks

This repository implements a Representational Similarity Analysis (RSA) pipeline to compare the geometric structures of spatial world models across biological neural assemblies and artificial transformers. 

Specifically, it measures the geometric alignment between simulated biological entorhinal grid cell populations (2D continuous attractor dynamics) and mid-layer residual representations in `gpt2-small`.

---

## Theoretical Overview

A central question in computational neuroscience and mechanistic interpretability is whether diverse intelligent systems converge on shared internal geometries to model the external world (The Platonic Representation Hypothesis*).

1. Biological Grid Cells: Mammalian grid cells in the medial entorhinal cortex fire in periodic hexagonal lattices, generating an isometric distance metric for spatial navigation.
2. Artificial World Models: Transformers process textual spatial coordinates into latent activation vectors.

Using Representational Similarity Analysis (RSA), we construct Representational Dissimilarity Matrices (RDMs) using pairwise Pearson correlation distances across 2D coordinate positions:

$$\text{RDM}_{i, j} = 1 - r(\mathbf{a}_i, \mathbf{a}_j)$$

We then compute the Spearman rank correlation ($\rho$) between the upper triangular elements of the biological RDM ($\mathbf{D}_{\text{bio}}$) and the artificial RDM ($\mathbf{D}_{\text{art}}$):

$$\rho = \text{Spearman}(\text{upper}(\mathbf{D}_{\text{bio}}), \text{upper}(\mathbf{D}_{\text{art}}))$$

---

## Directory Structure

```text
geometric-theory-of-intelligence/
├── src/
│   ├── __init__.py
│   ├── biological_grid.py    # Bio-inspired 2D hexagonal grid cell simulator
│   ├── artificial_world.py   # Spatial activation harvester for GPT-2
│   └── rsa_geometry.py       # RDM matrix engine & Spearman alignment computation
├── main.py                   # Orchestration and statistical testing script
├── requirements.txt
└── README.md

--- Project Analysis: Unified Geometry of Intelligence ---
Biological Grid Cell RDM Shape : (16, 16)
Artificial Transformer RDM Shape: (16, 16)

Representational Geometric Alignment (RSA Score): 0.4482
Statistical Significance (p-value): 2.8532e-07

Conclusion: Conserved World Model Geometry detected! Artificial and biological networks share a similar spatial metric space.