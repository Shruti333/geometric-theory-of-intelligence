"""Do transformer representations of 2D positions share geometry with a grid-cell code
-- beyond what plain Euclidean space explains?

Examples:
  python main.py                                   # gpt2, 8x8 arena, 5000 permutations
  python main.py --n-grid-seeds 30                 # more grid-population seeds
  python main.py --models gpt2 gpt2-medium         # several models
  python main.py --real-rates my_rate_maps.npy     # also test against REAL rate maps
"""
import argparse
import json
import os

import numpy as np

from biological_grid import (euclidean_rdm, grid_coords, make_grid_population,
                             real_rate_maps_to_rdm)
from rsa_geometry import (correlation_rdm, mantel_spearman, partial_mantel_spearman,
                          partial_spearman, spearman_rsa)


def analyse_model(label, layer_rdms, target_rdm, euc_rdm, n_perm):
    """layer_rdms: per-layer RDMs. target_rdm: grid (simulated) or real RDM."""
    rows = []
    for layer, rdm in enumerate(layer_rdms):
        g = mantel_spearman(rdm, target_rdm, n_perm, seed=layer)
        e = mantel_spearman(rdm, euc_rdm, n_perm, seed=100 + layer)
        pg = partial_mantel_spearman(rdm, target_rdm, euc_rdm, n_perm, seed=200 + layer)
        rows.append(dict(model=label, layer=layer,
                         rho_grid=g["rho"], p_grid=g["p"], null95_grid=g["null95"],
                         rho_euclid=e["rho"], p_euclid=e["p"],
                         rho_partial=pg["rho"], p_partial=pg["p"]))
    n_tests = len(rows)                       # Bonferroni across layers
    for r in rows:
        for k in ("p_grid", "p_euclid", "p_partial"):
            r[k + "_bonf"] = None if np.isnan(r[k]) else min(1.0, r[k] * n_tests)
    return rows


def print_table(rows, title=None):
    print(f"\n=== {title or rows[0]['model']} ===")
    print("layer | rho(target) p_bonf | rho(euclid) p_bonf | partial(target|euclid) p_bonf")
    f = lambda v: "  nan " if v is None or np.isnan(v) else f"{v:6.3f}"
    for r in rows:
        print(f"{r['layer']:5d} | {f(r['rho_grid'])} {f(r['p_grid_bonf'])} | "
              f"{f(r['rho_euclid'])} {f(r['p_euclid_bonf'])} | "
              f"{f(r['rho_partial'])} {f(r['p_partial_bonf'])}")


def seed_partial_analysis(euc_rdm, rdm_store, grid_rdms, layers, name):
    """Robustness to the random draw of the simulated grid cells (fixed layers, no p-value:
    the number of seeds is arbitrary, so a seed-count p-value could be made as small as
    you like). Look at paired differences and the fraction of seeds GPT-2 wins."""
    n_seeds = len(grid_rdms)
    lex = np.array([partial_spearman(rdm_store[name + "_lexical"], g, euc_rdm) for g in grid_rdms])
    out = dict(n_seeds=n_seeds, lexical=lex.tolist(), layers={})
    print(f"\n=== {name}: partial(grid|euclid) across {n_seeds} grid-population seeds ===")
    print(f"lexical baseline: {np.nanmean(lex):.3f} +/- {np.nanstd(lex):.3f}")
    for l in layers:
        gp = np.array([partial_spearman(rdm_store[name][l], g, euc_rdm) for g in grid_rdms])
        rn = np.array([partial_spearman(rdm_store[name + "_random_init"][l], g, euc_rdm)
                       for g in grid_rdms])
        d_lex, d_rnd = gp - lex, gp - rn
        print(f"layer {l}: trained {np.nanmean(gp):.3f} +/- {np.nanstd(gp):.3f} | "
              f"untrained {np.nanmean(rn):.3f} +/- {np.nanstd(rn):.3f}")
        print(f"         trained - lexical   = {np.nanmean(d_lex):+.3f} +/- {np.nanstd(d_lex):.3f}; "
              f"trained > lexical in {int(np.sum(d_lex > 0))}/{n_seeds} seeds")
        print(f"         trained - untrained = {np.nanmean(d_rnd):+.3f} +/- {np.nanstd(d_rnd):.3f}; "
              f"trained > untrained in {int(np.sum(d_rnd > 0))}/{n_seeds} seeds")
        out["layers"][l] = dict(trained=gp.tolist(), random_init=rn.tolist(),
                                n_trained_gt_lexical=int(np.sum(d_lex > 0)),
                                n_trained_gt_random=int(np.sum(d_rnd > 0)))
    return out


def template_partial_summary(name, tp_store, lex_seed, templates, layers):
    """Robustness to prompt wording: partial(grid|euclid) for every template x grid seed."""
    trained, untrained = tp_store[name], tp_store[name + "_random_init"]
    lex = np.array(lex_seed)
    print(f"\n=== {name}: partial(grid|euclid) across {len(templates)} prompt templates "
          f"x {len(lex)} grid seeds ===")
    for i, t in enumerate(templates):
        print(f"  T{i + 1}: {t!r}")
    print(f"lexical baseline (template-independent): {np.nanmean(lex):.3f}")
    out = {}
    for l in layers:
        m_tr = [float(np.nanmean(trained[t][l])) for t in templates]
        m_un = [float(np.nanmean(untrained[t][l])) for t in templates]
        cells = np.array([np.array(trained[t][l]) > lex for t in templates])   # (T, seeds)
        n_t = int(sum(m > np.nanmean(lex) for m in m_tr))
        print(f"layer {l}: trained per-template mean: " + " ".join(f"{m:+.3f}" for m in m_tr))
        print(f"         untrained per-template mean: " + " ".join(f"{m:+.3f}" for m in m_un))
        print(f"         trained > lexical mean in {n_t}/{len(templates)} templates; "
              f"in {int(cells.sum())}/{cells.size} (template, seed) cells")
        out[l] = dict(trained_template_means=m_tr, untrained_template_means=m_un,
                      n_templates_trained_gt_lexical=n_t,
                      n_cells_trained_gt_lexical=int(cells.sum()), n_cells=int(cells.size))
    return out


def make_plots(out_dir, results, name, suffix, grid_rdm, euc_rdm, best_rdm, best_layer):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    trained, untrained = results[name], results[name + "_random_init"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, key, title in [(axes[0], "rho_grid", "RSA with simulated grid-cell code"),
                           (axes[1], "rho_euclid", "RSA with Euclidean distance")]:
        for label, rows, style in [(name, trained, "o-"), (name + "_random_init", untrained, "s--")]:
            ax.plot([r["layer"] for r in rows], [r[key] for r in rows], style, label=label)
        if key == "rho_grid":
            ax.plot([r["layer"] for r in trained], [r["rho_partial"] for r in trained], "^:",
                    label=f"{name} partial (grid | euclid)")
            ax.plot([r["layer"] for r in trained], [r["null95_grid"] for r in trained],
                    color="gray", linewidth=1, label="permutation 95th pct")
        ax.axhline(0, color="k", linewidth=0.5)
        ax.set_xlabel("layer (0 = embeddings)")
        ax.set_title(title)
        ax.legend(fontsize=7)
    axes[0].set_ylabel("Spearman rho")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"layerwise_rsa{suffix}.png"), dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    for ax, rdm, title in zip(axes, [grid_rdm, euc_rdm, best_rdm],
                              ["Simulated grid code", "Euclidean distance",
                               f"{name} layer {best_layer}"]):
        ax.imshow(rdm, cmap="viridis")
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"rdms{suffix}.png"), dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["gpt2"],
                    help="HuggingFace model names; tokenizer must give single-token digits")
    ap.add_argument("--grid-size", type=int, default=8)      # keep <= 10 (single-digit tokens)
    ap.add_argument("--pooling", choices=["last", "mean"], default="last")
    ap.add_argument("--n-perm", type=int, default=5000)
    ap.add_argument("--n-grid-seeds", type=int, default=10)
    ap.add_argument("--seed-layers", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--real-rates", default=None,
                    help=".npy of REAL grid-cell rate maps, shape (n_cells, H, W), NaN=unvisited")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    assert args.grid_size <= 10, "use grid_size <= 10 so all prompts have equal token counts"
    os.makedirs(args.out, exist_ok=True)

    import artificial_world as aw            # imported here so torch loads only when needed

    coords = grid_coords(args.grid_size)
    euc_rdm = euclidean_rdm(coords)
    grid_rdms = [correlation_rdm(make_grid_population(coords, seed=s))
                 for s in range(max(1, args.n_grid_seeds))]
    grid_rdm = grid_rdms[0]                                   # main analysis uses seed 0
    print(f"{len(coords)} positions, {len(coords)*(len(coords)-1)//2} RDM pairs")
    print(f"Sanity: simulated grid RDM vs Euclidean RDM, rho = "
          f"{spearman_rsa(grid_rdm, euc_rdm):.3f}  (how much 'grid' is just 'space')")

    results = {"by_model": {}}
    rdm_store, tp_store = {}, {}
    for mi, name in enumerate(args.models):
        by_model = {}
        for random_init in (False, True):
            label = name + ("_random_init" if random_init else "")
            tok, model, device = aw.load_model(name, random_init=random_init)
            acts = {t: aw.extract_activations(tok, model, device, coords, t, args.pooling)
                    for t in aw.ALL_TEMPLATES}
            n_layers = acts[aw.TEMPLATES[0]].shape[0]

            # Main analysis: RDMs averaged over the three primary templates.
            per_template = [[correlation_rdm(acts[t][l]) for l in range(n_layers)]
                            for t in aw.TEMPLATES]
            mean_rdms = [np.mean([pt[l] for pt in per_template], axis=0) for l in range(n_layers)]
            rows = analyse_model(label, mean_rdms, grid_rdm, euc_rdm, args.n_perm)
            for l, r in enumerate(rows):
                r["rho_grid_per_template"] = [spearman_rsa(pt[l], grid_rdm) for pt in per_template]
            results[label], rdm_store[label] = rows, mean_rdms
            print_table(rows)

            # Prompt-wording robustness: partial(grid|euclid) for ALL templates x grid seeds.
            layers = [l for l in args.seed_layers if l < n_layers]
            tp_store[label] = {t: {l: [partial_spearman(correlation_rdm(acts[t][l]), g, euc_rdm)
                                       for g in grid_rdms] for l in layers}
                               for t in aw.ALL_TEMPLATES}

            if not random_init:
                base = correlation_rdm(aw.digit_embedding_baseline(tok, model, coords))
                rdm_store[name + "_lexical"] = base
                b = mantel_spearman(base, grid_rdm, args.n_perm)
                be = mantel_spearman(base, euc_rdm, args.n_perm)
                pb = partial_mantel_spearman(base, grid_rdm, euc_rdm, args.n_perm)
                print(f"\nLexical baseline (digit embeddings only): rho(grid)={b['rho']:.3f} "
                      f"p={b['p']:.4f} | rho(euclid)={be['rho']:.3f} p={be['p']:.4f}")
                print(f"Lexical partial(grid|euclid): rho={pb['rho']:.3f} p={pb['p']:.4f}")
                by_model["lexical_baseline"] = dict(
                    rho_grid=b["rho"], p_grid=b["p"], rho_euclid=be["rho"], p_euclid=be["p"],
                    rho_partial=pb["rho"], p_partial=pb["p"])

        # Post-hoc best layer for the raw-grid seed check (labelled as post hoc).
        valid = [r for r in results[name] if not np.isnan(r["rho_grid"])]
        best = max(valid, key=lambda r: r["rho_grid"])["layer"]
        per_seed = [spearman_rsa(rdm_store[name][best], g) for g in grid_rdms]
        print(f"\n{name}: grid-population seed robustness at layer {best} (chosen post hoc): "
              f"rho = {np.mean(per_seed):.3f} +/- {np.std(per_seed):.3f} "
              f"(range {min(per_seed):.3f} to {max(per_seed):.3f})")
        by_model["seed_robustness"] = dict(layer=best, rhos=per_seed)

        layers = [l for l in args.seed_layers if l < len(rdm_store[name])]
        by_model["seed_partial"] = seed_partial_analysis(euc_rdm, rdm_store, grid_rdms, layers, name)
        by_model["template_partial"] = template_partial_summary(
            name, tp_store, by_model["seed_partial"]["lexical"], aw.ALL_TEMPLATES, layers)
        results["by_model"][name] = by_model
        if mi == 0:                                            # legacy keys for the first model
            results["lexical_baseline"] = by_model["lexical_baseline"]
            results["seed_robustness"] = by_model["seed_robustness"]
            results["seed_partial"] = by_model["seed_partial"]

        make_plots(args.out, results, name, "" if mi == 0 else f"_{name.replace('/', '_')}",
                   grid_rdm, euc_rdm, rdm_store[name][best], best)

    # Optional: REAL recorded grid-cell rate maps in place of the simulation.
    if args.real_rates:
        real_rdm = real_rate_maps_to_rdm(np.load(args.real_rates), args.grid_size)
        print(f"\nREAL DATA sanity: real RDM vs Euclidean RDM, rho = "
              f"{spearman_rsa(real_rdm, euc_rdm):.3f}")
        results["real_data"] = {}
        for name in args.models:
            rows_t = analyse_model(name, rdm_store[name], real_rdm, euc_rdm, args.n_perm)
            rows_u = analyse_model(name + "_random_init", rdm_store[name + "_random_init"],
                                   real_rdm, euc_rdm, args.n_perm)
            print_table(rows_t, title=f"REAL DATA: {name}")
            print_table(rows_u, title=f"REAL DATA: {name}_random_init")
            base = rdm_store[name + "_lexical"]
            lb = mantel_spearman(base, real_rdm, args.n_perm)
            lp = partial_mantel_spearman(base, real_rdm, euc_rdm, args.n_perm)
            print(f"REAL DATA lexical baseline: rho={lb['rho']:.3f} p={lb['p']:.4f} | "
                  f"partial(real|euclid)={lp['rho']:.3f} p={lp['p']:.4f}")
            results["real_data"][name] = dict(trained=rows_t, random_init=rows_u,
                                              lexical=dict(rho=lb["rho"], p=lb["p"],
                                                           rho_partial=lp["rho"], p_partial=lp["p"]))
        print("\nCaution: real-data results are exploratory. Arena-to-digit mapping is arbitrary and")
        print("real grid spacing relative to arena size differs from the simulation.")

    sig = lambda rows, k: [r["layer"] for r in rows if r[k] is not None
                           and not np.isnan(r[k]) and r[k] < 0.05]
    print("\nLayers significant after Bonferroni correction (p < 0.05), simulated grid:")
    for name in args.models:
        for label in (name, name + "_random_init"):
            print(f"  {label}: grid {sig(results[label], 'p_grid_bonf')} | "
                  f"euclid {sig(results[label], 'p_euclid_bonf')} | "
                  f"partial {sig(results[label], 'p_partial_bonf')}")
    print("Interpret with care: compare trained against untrained and the lexical baseline,")
    print("and check whether the grid effect survives controlling for Euclidean distance.")

    with open(os.path.join(args.out, "results.json"), "w") as fh:
        json.dump(results, fh, indent=2, default=float)
    print(f"\nSaved results.json and figures to {args.out}/")


if __name__ == "__main__":
    main()