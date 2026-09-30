# Does GPT-2 represent 2D space like a grid-cell code?

An RSA study comparing GPT-2's internal representations of 2D positions with a **simulated** grid-cell population, with controls for the obvious confounds.

**Question.** Do hidden states for prompts like "The object is at x = 3, y = 5." share representational geometry with a multi-module grid-cell code, *beyond what plain Euclidean space already explains*, and does training create it?

## Method
- 8x8 arena (64 positions, 2016 RDM pairs); coordinates are single digits so all prompts have equal token counts.
- **Bio model:** 4 grid modules x 50 cells (three-cosine rate maps, random orientation/phase, noise). Simulated, not recorded data. Main analysis uses seed 0; robustness uses 30 seeds.
- **Models:** GPT-2 small (12 layers) and GPT-2 medium (24 layers), hidden states at every layer, last-token pooling.
- **Prompts:** main analysis averages RDMs over 3 templates; robustness uses 8 templates (different wording and final tokens).
- **RDM:** correlation distance after centering each feature across positions.
- **Statistics:** Spearman RSA with a label-permutation (Mantel-style) test, Bonferroni-corrected across layers. RDM entries are not independent, so the ordinary Spearman p-value is not used. Corrected p-values have a floor set by the 5,000 permutations.
- **Controls:** (1) untrained model with identical architecture, (2) lexical baseline from digit input embeddings, (3) partial RSA of the grid RDM controlling for the Euclidean-distance RDM, (4) robustness over 30 grid-population seeds and 8 prompt templates at layers 1-4 (fixed in advance) and, for GPT-2 medium, at layers 18, 20, 22 (chosen after seeing the seed-0 late-layer residual, so post hoc).

## Run
```
pip install -r requirements.txt
python main.py --models gpt2 gpt2-medium --n-grid-seeds 30
python main.py --models gpt2-medium --seed-layers 18 20 22 --n-grid-seeds 30 --n-perm 2000   # late-layer check
```
Writes `results/results.json` plus figures. Optional `--real-rates file.npy` (real rate maps, shape `(n_cells, H, W)`) is implemented and tested on synthetic data only; no real recordings have been analysed here.

## Results

**Main analysis (seed-0 grid population, prompts averaged).** rho = Spearman RSA; partial = grid RDM controlling for Euclidean distance; p = Bonferroni-corrected across layers.

| Model / layer | rho (grid) | rho (Euclid) | partial |
|---|---|---|---|
| GPT-2, layer 1 | 0.164 | 0.603 | 0.080 (p = 0.018) |
| GPT-2, layer 4 | 0.161 | 0.662 | 0.067 (p = 0.023) |
| GPT-2, layer 10 | 0.136 | 0.676 | 0.031 (n.s.) |
| GPT-2, layer 12 | 0.107 | 0.594 | 0.009 (n.s.) |
| GPT-2 untrained, layer 1 | 0.077 | 0.247 | 0.037 (n.s.) |
| GPT-2 untrained, layer 8 | 0.061 | 0.246 | 0.020 (n.s.) |
| GPT-2 medium, layer 3 | 0.158 | 0.619 | 0.070 (n.s.) |
| GPT-2 medium, layer 9 | 0.153 | 0.703 | 0.050 (n.s.) |
| GPT-2 medium, layer 22 | 0.154 | 0.544 | 0.075 (p ~ 0.04) |
| GPT-2 medium, layer 24 | 0.112 | 0.415 | 0.048 (n.s.) |
| GPT-2 medium untrained, layer 1 | 0.079 | 0.217 | 0.044 (n.s.) |
| GPT-2 medium untrained, layer 12 | 0.073 | 0.191 | 0.043 (n.s.) |
| Lexical baseline, GPT-2 | 0.139 | 0.730 | 0.025 (n.s.) |
| Lexical baseline, GPT-2 medium | 0.137 | 0.732 | 0.021 (n.s.) |

**Robustness: partial(grid | Euclid) over 30 grid-population seeds** (mean +/- sd; "wins" = seeds where trained beats the comparison, paired by seed). Lexical baseline: 0.031 +/- 0.039 (GPT-2), 0.027 +/- 0.040 (medium).

| Model, layer | trained | untrained | trained - untrained (wins) | trained - lexical (wins) |
|---|---|---|---|---|
| GPT-2, 1 | 0.064 +/- 0.055 | 0.054 +/- 0.042 | +0.009 +/- 0.049 (15/30) | +0.033 +/- 0.044 (21/30) |
| GPT-2, 2 | 0.053 +/- 0.051 | 0.059 +/- 0.042 | -0.006 +/- 0.055 (13/30) | +0.022 +/- 0.044 (19/30) |
| GPT-2, 3 | 0.043 +/- 0.055 | 0.058 +/- 0.045 | -0.015 +/- 0.060 (12/30) | +0.012 +/- 0.046 (19/30) |
| GPT-2, 4 | 0.044 +/- 0.055 | 0.056 +/- 0.046 | -0.012 +/- 0.061 (12/30) | +0.013 +/- 0.046 (19/30) |
| Medium, 1 | 0.026 +/- 0.025 | 0.069 +/- 0.044 | -0.042 +/- 0.043 (5/30) | -0.001 +/- 0.040 (15/30) |
| Medium, 2 | 0.033 +/- 0.026 | 0.070 +/- 0.044 | -0.037 +/- 0.043 (7/30) | +0.006 +/- 0.040 (17/30) |
| Medium, 3 | 0.043 +/- 0.052 | 0.057 +/- 0.046 | -0.014 +/- 0.058 (13/30) | +0.016 +/- 0.045 (19/30) |
| Medium, 4 | 0.036 +/- 0.050 | 0.064 +/- 0.045 | -0.028 +/- 0.058 (11/30) | +0.009 +/- 0.043 (18/30) |
| Medium, 18 (post hoc) | 0.073 +/- 0.025 | 0.072 +/- 0.045 | +0.001 +/- 0.047 (15/30) | +0.046 +/- 0.041 (24/30) |
| Medium, 20 (post hoc) | 0.080 +/- 0.025 | 0.068 +/- 0.045 | +0.012 +/- 0.045 (18/30) | +0.053 +/- 0.041 (25/30) |
| Medium, 22 (post hoc) | 0.091 +/- 0.025 | 0.068 +/- 0.044 | +0.023 +/- 0.043 (22/30) | +0.064 +/- 0.041 (28/30) |

**Robustness across 8 prompt templates** (each averaged over the 30 seeds): trained beats untrained in 6/4/1/2 of 8 templates at layers 1/2/3/4 for GPT-2 (mean differences +0.007, -0.004, -0.008, -0.006) and in 1/0/3/3 of 8 for GPT-2 medium (mean differences -0.028, -0.026, -0.005, -0.012). Trained beats the lexical mean in 8/8, 8/8, 7/8, 7/8 templates for GPT-2 and 3/8, 5/8, 8/8, 8/8 for GPT-2 medium. At GPT-2 medium layers 18/20/22, trained beats untrained in only 2 of 8 templates at each layer (per-template mean differences -0.009, -0.006, -0.005), driven almost entirely by one phrasing ("Position ({x}, {y}) on the map.", +0.05 to +0.06); trained beats the lexical mean in 8/8, 8/8, 6/8 templates.

**Findings**
1. **Euclidean structure is strong but is not built by the network.** Trained models track Euclidean distance at every layer (GPT-2 rho 0.59-0.68; GPT-2 medium 0.42-0.70, peaking at layer 9) versus ~0.19-0.25 untrained, but the digit input embeddings alone reach 0.73 in both models, higher than any layer.
2. **Raw grid alignment is modest and mostly Euclidean.** rho is about 0.14-0.16 across most layers in both models (untrained about 0.05-0.08; lexical 0.14), and it falls sharply after controlling for Euclidean distance.
3. **The early-layer grid-specific residual seen with seed 0 does not hold up.** Across 30 grid populations and 8 prompts, trained GPT-2's partial correlation at layers 1-4 is not distinguishable from an untrained network's (differences +0.009 to -0.015, GPT-2 winning about half the seeds), and GPT-2 medium is below its untrained counterpart. The modest edge over the lexical baseline in GPT-2 (+0.01 to +0.03, about 19-21 of 30 seeds) is small relative to seed-to-seed spread.
4. **GPT-2 medium's late layers show a weak, prompt-dependent residual.** The partial correlation is about 0.07 at layers 17-23 for seed 0 (corrected p about 0.025-0.05 at layers 17-19, 21, 22; two runs with different permutation counts agree on 18, 21, 22, while 17 and 19 are borderline; untrained: none significant). Across 30 grid populations at layers 18/20/22 it reliably exceeds the lexical baseline (+0.046 to +0.064; 24-28 of 30 seeds), rises slightly with depth (0.073 to 0.091) and is more consistent across seeds than the untrained network's (sd 0.025 vs 0.045). But its edge over the untrained network is small (+0.001, +0.012, +0.023; wins 15, 18, 22 of 30) and depends on the prompt (2 of 8 phrasings). These layers were picked after seeing the seed-0 result. I read this as a possible weak signal, not evidence that training builds grid-like geometry.
5. The raw grid correlation itself depends on the simulated population: at the post-hoc best layer, rho = 0.127 +/- 0.048 (GPT-2 layer 1) and 0.112 +/- 0.045 (medium layer 3), ranges 0.027-0.208 and 0.031-0.194 over 30 draws.

An earlier version of this project (16 positions, no baselines, naive p-value) reported rho = 0.45 and called it "conserved geometry". With more positions, valid statistics and controls the effect is about a third as large, mostly explained by Euclidean structure already in the input embeddings, and the leftover grid-specific signal is not robust. That claim is withdrawn.

Layer 0 is NaN by design (last-token pooling gives identical vectors for every prompt at the embedding layer).

## Limitations
- The grid code is idealised and simulated; nothing here is about real entorhinal recordings.
- Only two models, both GPT-2 (same tokenizer and training data); conclusions may not transfer to other model families.
- Grid parameters (module scales, cell counts) were not varied; prompt phrasings all use digit coordinates.
- 64 positions from one small task; RSA is correlational, with no causal test (e.g. patching).
- Seed and prompt robustness cover layers 1-4 for both models and layers 18, 20, 22 for GPT-2 medium; the late-layer choice was post hoc (made after seeing the seed-0 result), and no other layers or larger models were checked.
- The best layer for the raw-grid seed check is chosen post hoc; Bonferroni counts 13/25 layers including the degenerate layer 0 (slightly conservative). No preregistration.

## Files
`biological_grid.py` grid population, reference RDMs, real-data loader | `artificial_world.py` activation extraction | `rsa_geometry.py` RSA and permutation tests | `main.py` full pipeline