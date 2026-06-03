"""
eval_image_crn.py -- residualized common-random-number (CRN) reference SELECTION
on a real classifier, followed by the insertion/deletion eval pipeline.

This is the real-classifier instantiation of the §3-§4 selection method in
"Cheap Reference Selection by Residualized Paired Contrasts" (the RePro paper),
wired in front of the existing eval_image.py faithfulness pipeline.

What changes vs eval_image.py
-----------------------------
eval_image.py selected `best_reference` via RefLIME's §6 SNR criterion (gamma),
which requires fitting a surrogate per reference -- the post-hoc, floored route
the paper argues against. Here we REPLACE that with the paper's cheap,
pre-explanation selector:

  * draw a single batch of shared masks z ~ mu (Bernoulli-1/2, the RefLIME
    MaskLibrary distribution) plus complements z' = 1 - z;
  * for each reference rho, the contrasted function is the REAL masked response
        g_rho(z) = f_c(Phi_rho(x, z))            [target-class probability]
    obtained by querying the classifier through rho on those masks;
  * split masks into an estimation half E and a contrast half C;
  * on E, estimate the dense linear block beta_hat_rho from the odd fold;
  * on C, residualize  r~_rho = g_rho - <beta_hat_rho, chi>  and form the
    per-sample energy proxy  q_rho = (e - e_bar)^2 + (o - o_bar)^2;
  * rank references by a Copeland tournament on sign(mean(q_a - q_b)); the
    reference with the LOWEST residual energy m_rho (most wins) is selected.

The estimator detail (split-sample, no in-sample beta_hat overfit) and the
energy proxy are taken directly from repro_residualized.py, lifted from the
synthetic planted-coefficient g to the real g_rho = f_c(Phi_rho(x, .)).

Then the SELECTED reference is fed into the unchanged ins/del faithfulness
pipeline (full method-attr x measure-baseline cross product), with the CRN
winner as the headline explanation.

Note: like eval_image.py, this module does NOT import torch at module load;
all torch work happens inside main().

Usage
-----
    python eval_image_crn.py --image cat.jpg --model resnet50 --grid 12 \
        --select-Q 5000 --device cuda --steps 50 --measure-repeat 4 \
        --out-dir out/

    # also run RefLIME's SNR selection alongside, for comparison only:
    python eval_image_crn.py --image cat.jpg --also-snr ...
"""
from __future__ import annotations
import argparse
import json
import os

import numpy as np


# --------------------------------------------------------------------------- #
#  CLI
# --------------------------------------------------------------------------- #
def build_argparser():
    p = argparse.ArgumentParser(
        description="Residualized-CRN reference selection + ins/del eval."
    )
    p.add_argument("--image", required=True, help="path to an RGB image")
    p.add_argument("--model", default="resnet50",
                   help="torchvision model name (resnet50, vit_b_16, ...)")
    p.add_argument("--device", default="cuda", help="cpu | cuda")
    p.add_argument("--grid", type=int, default=12, help="grid is (grid,grid)")
    p.add_argument("--target", type=int, default=None,
                   help="class index; default = model's top-1")
    p.add_argument("--references", default="black,gray,mean,blur,inpaint",
                   help="comma-separated subset of the reference family used "
                        "as selection candidates AND measure baselines")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=64)

    # ---- selection (residualized CRN) ----
    p.add_argument("--select-Q", type=int, default=5000,
                   help="shared mask draws per reference for the CRN contrast "
                        "(paper uses 5000)")
    p.add_argument("--select-repeat", type=int, default=1,
                   help="repeats per stochastic reference when querying g_rho "
                        "during selection (averaged); >=1. Deterministic "
                        "references ignore this.")

    # ---- explanation surrogate (RefLIME, for the selected ref's attr map) ----
    p.add_argument("--n-samples", type=int, default=2000)
    p.add_argument("--val-frac", type=float, default=0.3)
    p.add_argument("--c", type=float, default=1.26,
                   help="leakage constant (see synthetic_verify.py lemma1)")

    # ---- measurement (ins/del) ----
    p.add_argument("--steps", type=int, default=50,
                   help="ins/del curve resolution (number of reveal chunks)")
    p.add_argument("--measure-repeat", type=int, default=4,
                   help="repeats per stochastic measure baseline (averaged)")

    p.add_argument("--also-snr", action="store_true",
                   help="also compute RefLIME's SNR selection for comparison "
                        "(does not change which reference is used downstream)")
    p.add_argument("--eval-all-maps", action="store_true",
                   help="run ins/del for every reference's attr map, not just "
                        "the CRN-selected one (matches eval_image.py auditing)")
    p.add_argument("--out-dir", default="out")
    return p


# =========================================================================== #
#  PART A -- Residualized CRN reference selection (paper §3-§4, real model)
#
#  These mirror repro_residualized.py exactly, but operate on a precomputed
#  matrix G of real masked responses:  G[m, t] = g_{rho_m}(z^(t)),  with the
#  complement responses Gc[m, t] = g_{rho_m}(z'^(t)).  chi is built from the
#  same {0,1} mask matrix Z used to query the model, so the centered design
#  chi = 2*(Z - 0.5) matches the paper's Walsh basis.
# =========================================================================== #
def _chi(Z):
    """Centered Walsh design from {0,1} masks: chi_i = 2*(z_i - 1/2)."""
    return 2.0 * (Z - 0.5)


def _estimate_beta(Zhalf, g, gc):
    """Dense linear-coeff estimate via the odd fold:  beta_i = E[o * chi_i].

    g  = g_rho(z)   on the estimation masks
    gc = g_rho(z')  on their complements
    (repro_residualized.estimate_beta, with g/gc supplied instead of recomputed)
    """
    o = 0.5 * (g - gc)                      # odd fold (linear lives here)
    chi = _chi(Zhalf)
    return (chi * o[:, None]).mean(0)       # (d,)


def _residual_energy_terms(Zhalf, g, gc, beta_hat):
    """Per-sample interaction-energy proxy AFTER removing the linear part.

    Identical algebra to repro_residualized.residual_energy_terms, but the
    masked responses g / gc are passed in (real model queries) rather than
    evaluated from planted coefficients.
    """
    chi = _chi(Zhalf)
    y = g - chi @ beta_hat                  # residualized response
    chic = _chi(1.0 - Zhalf)                # complement design
    yc = gc - chic @ beta_hat
    e = 0.5 * (y + yc)                       # even fold of residual
    o = 0.5 * (y - yc)                       # odd fold of residual
    return (e - e.mean()) ** 2 + (o - o.mean()) ** 2


def _contrast_resid(Z, Ga, Gca, Gb, Gcb):
    """Split-sample residualized CRN contrast of m_a - m_b on common masks.

    Z          : (Q, d)  shared {0,1} masks
    Ga, Gca    : (Q,)     g_a(z), g_a(z')
    Gb, Gcb    : (Q,)     g_b(z), g_b(z')
    Returns mean(q_a - q_b) over the contrast half (sign gives the ranking).
    """
    Q = Z.shape[0]
    h = Q // 2
    Ze, Zc = Z[:h], Z[h:]                    # estimation / contrast halves
    ba = _estimate_beta(Ze, Ga[:h], Gca[:h])
    bb = _estimate_beta(Ze, Gb[:h], Gcb[:h])
    qa = _residual_energy_terms(Zc, Ga[h:], Gca[h:], ba)
    qb = _residual_energy_terms(Zc, Gb[h:], Gcb[h:], bb)
    return float((qa - qb).mean())


def _m_independent_resid(Z, G, Gc):
    """Residualized energy estimate for ONE reference (diagnostic / m_hat proxy).
    Split-sample, matching repro_residualized.m_independent_resid."""
    Q = Z.shape[0]
    h = Q // 2
    b = _estimate_beta(Z[:h], G[:h], Gc[:h])
    q = _residual_energy_terms(Z[h:], G[h:], Gc[h:], b)
    return float(q.mean())


def _copeland_pick(names, G, Gc, Z):
    """Copeland tournament: a reference wins a pair when its contrast is
    negative (smaller residual energy m). Returns (winner_name, wins, contrasts).

    G, Gc : dict name -> (Q,) real masked responses on shared masks Z.
    """
    n = len(names)
    wins = {nm: 0 for nm in names}
    contrasts = {}
    for i in range(n):
        for j in range(i + 1, n):
            a, b = names[i], names[j]
            d_ab = _contrast_resid(Z, G[a], Gc[a], G[b], Gc[b])
            contrasts[(a, b)] = d_ab
            if d_ab < 0:                      # a has smaller m
                wins[a] += 1
            else:
                wins[b] += 1
    winner = max(names, key=lambda nm: wins[nm])
    return winner, wins, contrasts


def _query_g(expl, x, rho, Z01, target, repeat):
    """Query the real masked response g_rho(z) = f_c(Phi_rho(x, z)) for a
    matrix of {0,1} cell masks Z01, reusing RefLIME's batched _query.

    For stochastic references, average `repeat` independent draws to tame
    sampling variance (the paper notes sigma_obs > 0 for inpaint-like rho).
    """
    import torch  # noqa: F401  (kept local; selection runs inside main())
    Zt = torch.tensor(Z01, dtype=torch.float32)
    is_stoch = bool(getattr(rho, "is_stochastic", False))
    reps = max(1, repeat) if is_stoch else 1
    acc = np.zeros(Z01.shape[0], dtype=np.float64)
    for _ in range(reps):
        acc += expl._query(x, rho, Zt, target)
    return acc / reps


def select_reference_crn(expl, x, references, target, Q, seed, repeat):
    """Run residualized-CRN selection over `references` on the real model.

    Returns dict with: winner, wins, m_hat (per-ref independent estimate,
    diagnostic), and the pairwise contrasts.
    """
    lib = expl._lib
    d = lib.n_cells
    # shared masks z ~ mu (same Bernoulli-1/2 product the MaskLibrary uses)
    rng = np.random.default_rng(seed)
    Z = (rng.random((Q, d)) > 0.5).astype(np.float64)
    Zc = 1.0 - Z

    names = list(references.keys())
    G, Gc = {}, {}
    for nm in names:
        rho = references[nm]
        G[nm] = _query_g(expl, x, rho, Z, target, repeat)
        Gc[nm] = _query_g(expl, x, rho, Zc, target, repeat)

    winner, wins, contrasts = _copeland_pick(names, G, Gc, Z)
    m_hat = {nm: _m_independent_resid(Z, G[nm], Gc[nm]) for nm in names}
    return {
        "winner": winner,
        "wins": wins,
        "m_hat": m_hat,
        "contrasts": {f"{a}|{b}": v for (a, b), v in contrasts.items()},
        "Q": Q,
    }


# =========================================================================== #
#  PART B -- Insertion / Deletion faithfulness (unchanged from eval_image.py)
# =========================================================================== #
def _cell_order_from_attr(attr, lib):
    """High-attribution-first ordering of cell ids from an (H,W) map."""
    ids = lib.cell_ids.detach().cpu().numpy()
    n_cells = lib.n_cells
    cell_scores = np.full(n_cells, -np.inf, dtype=np.float64)
    for c in range(n_cells):
        m = ids == c
        if m.any():
            cell_scores[c] = float(attr[m].mean())
    return np.argsort(-cell_scores)


def insertion_deletion_auc(model, x, attr, rho_b, target, lib, device,
                           steps=50, repeat=1):
    """Insertion & deletion AUC for one attribution map under one MEASURE
    baseline rho_b. Mean over `repeat` draws (folded to 1 for deterministic)."""
    import torch
    import torch.nn.functional as F

    n_cells = lib.n_cells
    order = _cell_order_from_attr(attr, lib)
    is_stoch = bool(getattr(rho_b, "is_stochastic", False))
    reps = max(1, repeat) if is_stoch else 1

    def one_curve(insertion: bool, seed_offset: int):
        keep_cells = np.zeros(n_cells) if insertion else np.ones(n_cells)
        chunk = max(1, n_cells // steps)
        ys = []
        keep_states = []
        s = 0
        while True:
            keep_states.append(keep_cells.copy())
            if s >= n_cells:
                break
            nxt = order[s:s + chunk]
            keep_cells[nxt] = 1.0 if insertion else 0.0
            s += chunk
        Z = torch.tensor(np.stack(keep_states), dtype=torch.float32)
        with torch.no_grad():
            for b in range(0, Z.shape[0], 64):
                zb = Z[b:b + 64]
                keep = lib.to_pixel_keep(zb)
                comp = rho_b(x, keep)
                p = F.softmax(model(comp.to(device)), dim=1)[:, target]
                ys.extend(p.detach().cpu().numpy().tolist())
        return float(np.trapezoid(ys) / len(ys))

    ins_vals, del_vals = [], []
    for r in range(reps):
        ins_vals.append(one_curve(True, r))
        del_vals.append(one_curve(False, r))
    return {
        "insertion_auc": float(np.mean(ins_vals)),
        "deletion_auc": float(np.mean(del_vals)),
    }


def _agg(per_baseline, key, exclude=None):
    vals = [v[key] for name, v in per_baseline.items() if name != exclude]
    if not vals:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    return {"mean": float(np.mean(vals)),
            "std": float(np.std(vals, ddof=0)),
            "n": len(vals)}


# =========================================================================== #
#  Main
# =========================================================================== #
def main():
    args = build_argparser().parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    import torchvision as tv
    from PIL import Image

    from lime import RefLIME, default_reference_family

    device = args.device

    # ---- model + image ----
    weights = tv.models.get_model_weights(args.model).DEFAULT
    model = tv.models.get_model(args.model, weights=weights).eval().to(device)
    prep = weights.transforms()
    img = Image.open(args.image).convert("RGB")
    x = prep(img).unsqueeze(0).to(device)

    # ---- reference family (candidates AND measure baselines) ----
    fam_all = default_reference_family()
    chosen = [r.strip() for r in args.references.split(",")]
    references = {k: fam_all[k] for k in chosen if k in fam_all}
    if not references:
        raise SystemExit(f"no valid references in {chosen!r}; "
                         f"available: {list(fam_all)}")

    # ---- build a RefLIME instance: gives us the MaskLibrary + _query, and the
    #      surrogate fit we use to produce the selected reference's attr map. ----
    expl = RefLIME(model, device=device, grid=(args.grid, args.grid),
                   n_samples=args.n_samples, val_frac=args.val_frac,
                   c=args.c, batch_size=args.batch_size, seed=args.seed)

    # RefLIME builds self._lib inside explain(); replicate that here so the
    # selection masks/cells match the eval side exactly.
    x = x.to(device)
    _, _, H, W = x.shape
    from lime import MaskLibrary
    expl._lib = MaskLibrary(H, W, (args.grid, args.grid),
                            device=device, seed=args.seed)
    lib = expl._lib

    target = args.target
    if target is None:
        import torch
        with torch.no_grad():
            target = int(model(x).argmax(dim=1).item())

    # ======================================================================= #
    #  STEP 1 -- residualized-CRN reference selection on the real model
    # ======================================================================= #
    print(f"\ntarget class index: {target}")
    print(f"\n[selection] residualized CRN over {list(references)} "
          f"at Q={args.select_Q} ...")
    sel = select_reference_crn(expl, x, references, target,
                               Q=args.select_Q, seed=args.seed,
                               repeat=args.select_repeat)
    best = sel["winner"]

    print(f"\n{'reference':>10} {'wins':>6} {'m_hat(resid)':>14}")
    for nm in references:
        flag = "  <== SELECTED" if nm == best else ""
        print(f"{nm:>10} {sel['wins'][nm]:>6d} {sel['m_hat'][nm]:>14.6f}{flag}")
    print(f"\n>>> CRN-selected reference (lowest residual energy m): {best}")

    # optional: RefLIME SNR selection, comparison only
    snr_pick = None
    if args.also_snr:
        print("\n[compare] computing RefLIME SNR selection (gamma) ...")
        res_snr = expl.explain(x, target=target, references=references)
        snr_pick = res_snr.best_reference
        print(f"  SNR-selected: {snr_pick}  |  CRN-selected: {best}  |  "
              f"agree: {snr_pick == best}")

    # ======================================================================= #
    #  STEP 2 -- produce attribution map(s) via the surrogate fit
    #            (only for the references we will measure)
    # ======================================================================= #
    maps_to_eval = list(references) if args.eval_all_maps else [best]
    attr_maps = {}
    print(f"\n[explain] fitting surrogate for: {maps_to_eval}")
    for nm in maps_to_eval:
        pr = expl._explain_one(x, nm, references[nm], target)
        attr_maps[nm] = pr.attr
        np.save(os.path.join(args.out_dir, f"attr_{nm}.npy"), pr.attr)

    # ======================================================================= #
    #  STEP 3 -- ins/del faithfulness: (method attr) x (measure baseline)
    # ======================================================================= #
    print(f"\n[measure] ins/del over {len(maps_to_eval)} attr map(s) "
          f"x {len(references)} measure baselines ...")
    eval_block = {}
    for m_name in maps_to_eval:
        attr = attr_maps[m_name]
        per_baseline = {}
        for b_name, rho_b in references.items():
            per_baseline[b_name] = insertion_deletion_auc(
                model, x, attr, rho_b, target, lib, device,
                steps=args.steps, repeat=args.measure_repeat)
        agg_all = {
            "insertion_auc": _agg(per_baseline, "insertion_auc"),
            "deletion_auc": _agg(per_baseline, "deletion_auc"),
        }
        agg_excl = {
            "insertion_auc": _agg(per_baseline, "insertion_auc", exclude=m_name),
            "deletion_auc": _agg(per_baseline, "deletion_auc", exclude=m_name),
        }
        eval_block[m_name] = {
            "per_baseline": per_baseline,
            "agg_all": agg_all,
            "agg_exclude_self": agg_excl,
        }
        tag = "  <== SELECTED" if m_name == best else ""
        print(f"\n[method ref: {m_name}]{tag}")
        print(f"  all baselines        : "
              f"ins {agg_all['insertion_auc']['mean']:.3f}"
              f" +/- {agg_all['insertion_auc']['std']:.3f} | "
              f"del {agg_all['deletion_auc']['mean']:.3f}"
              f" +/- {agg_all['deletion_auc']['std']:.3f}")
        print(f"  exclude-self ({m_name:>5}) : "
              f"ins {agg_excl['insertion_auc']['mean']:.3f}"
              f" +/- {agg_excl['insertion_auc']['std']:.3f} | "
              f"del {agg_excl['deletion_auc']['mean']:.3f}"
              f" +/- {agg_excl['deletion_auc']['std']:.3f}")

    # ---- headline: the CRN-selected explanation ----
    head_all = eval_block[best]["agg_all"]
    head_excl = eval_block[best]["agg_exclude_self"]
    print("\n" + "=" * 62)
    print(f">>> SELECTED explanation (residualized-CRN, lowest m): {best}")
    print(f"    ins/del, ALL baselines        : "
          f"ins {head_all['insertion_auc']['mean']:.3f}"
          f" +/- {head_all['insertion_auc']['std']:.3f} | "
          f"del {head_all['deletion_auc']['mean']:.3f}"
          f" +/- {head_all['deletion_auc']['std']:.3f}")
    print(f"    ins/del, EXCLUDING {best:>5} base : "
          f"ins {head_excl['insertion_auc']['mean']:.3f}"
          f" +/- {head_excl['insertion_auc']['std']:.3f} | "
          f"del {head_excl['deletion_auc']['mean']:.3f}"
          f" +/- {head_excl['deletion_auc']['std']:.3f}")
    print("=" * 62)

    # ---- dump ----
    summary = {
        "model": args.model,
        "target": target,
        "selection_method": "residualized_crn",
        "selected_reference": best,
        "snr_reference_for_compare": snr_pick,
        "references": list(references),
        "selection": sel,
        "eval": eval_block,
        "config": {
            "grid": args.grid, "select_Q": args.select_Q,
            "select_repeat": args.select_repeat,
            "n_samples": args.n_samples, "val_frac": args.val_frac,
            "c": args.c, "steps": args.steps,
            "measure_repeat": args.measure_repeat, "seed": args.seed,
            "eval_all_maps": args.eval_all_maps,
        },
    }
    with open(os.path.join(args.out_dir, "eval_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nsaved attributions + eval_summary.json to {args.out_dir}/")


if __name__ == "__main__":
    main()