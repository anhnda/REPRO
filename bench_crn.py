"""
benchmark_crn.py -- dataset loop for residualized-CRN reference selection +
ins/del faithfulness, over a folder of images (default benchmark_50/*.JPEG).

Why this exists
---------------
A single image cannot test whether low residual energy m_rho tracks ins/del
faithfulness. NOTE this is an EXPLORATORY hypothesis of our own, NOT a claim in
the RePro paper -- that paper ranks references by recoverable residual energy m
and explicitly does not link m to a reference-free faithfulness ground truth.
Per-image Spearman over 5 references is pinned to coarse values (+1.0, -0.3, ...)
and swings wildly image to image -- we have seen m_hat_vs_insertion go +1.0 on
one image and -0.3 on the next. If there is any signal, it lives in the
DISTRIBUTION of these quantities across many images, not in any one.

This runner reuses the EXACT per-image pipeline from eval_image_crn.py (imported,
not reimplemented), runs it on every image, and aggregates with statistics that
suit n > 5 images:

  * mean / std / 95% CI of each of the 4 Spearman correlations across images;
  * a SIGN TEST on each correlation vs our hypothesized sign
    (fraction of images on the predicted side, with a binomial two-sided p);
  * the DISTRIBUTION of the CRN-selected reference's faithfulness rank
    (insertion and deletion) -- the most noise-robust signal, since it does not
    depend on the coarse 5-point Spearman;
  * how often each reference is selected, and how often the selected reference
    also wins faithfulness.

Resumable: per-image results are written to <out>/per_image/<stem>.json and
skipped on re-run, so the (expensive) sweep can be done in chunks.

Cost note: each image runs selection (5 refs x 2*Q forward passes) PLUS
--eval-all-maps (5 surrogate fits + 5x5 ins/del curves). Heavy. Use --limit to
smoke-test on a few images first.

torch is imported inside main() only (same convention as eval_image_crn.py).

Usage
-----
    python benchmark_crn.py --glob "benchmark_50/*.JPEG" --device cuda \
        --select-Q 5000 --out-dir bench_out/
    python benchmark_crn.py --limit 3            # quick smoke test
    python benchmark_crn.py --resume             # skip already-done images
"""
from __future__ import annotations
import argparse
import glob
import json
import math
import os
import sys
import traceback

import numpy as np


# --------------------------------------------------------------------------- #
#  CLI
# --------------------------------------------------------------------------- #
def build_argparser():
    p = argparse.ArgumentParser(
        description="Dataset-loop residualized-CRN selection + ins/del eval."
    )
    p.add_argument("--glob", default="benchmark_50/*.JPEG",
                   help="glob for input images")
    p.add_argument("--model", default="resnet50")
    p.add_argument("--device", default="cuda", help="cpu | cuda")
    p.add_argument("--grid", type=int, default=12)
    p.add_argument("--references", default="black,white,mean,blur,inpaint")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=64)

    # selection
    p.add_argument("--select-Q", type=int, default=5000)
    p.add_argument("--select-repeat", type=int, default=1)

    # explanation surrogate
    p.add_argument("--n-samples", type=int, default=2000)
    p.add_argument("--val-frac", type=float, default=0.3)
    p.add_argument("--c", type=float, default=1.26)

    # measurement
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--measure-repeat", type=int, default=4)

    p.add_argument("--limit", type=int, default=None,
                   help="process at most this many images (smoke test)")
    p.add_argument("--resume", action="store_true",
                   help="skip images whose per_image/<stem>.json already exists")
    p.add_argument("--out-dir", default="bench_out")
    return p


# --------------------------------------------------------------------------- #
#  Per-image pipeline (reuses eval_image_crn building blocks verbatim)
# --------------------------------------------------------------------------- #
def run_one_image(image_path, model, prep, references, args, device):
    """Selection + per-reference attr maps + full cross-baseline ins/del +
    within-image agreement, for ONE image. Returns a JSON-able dict.

    Method is identical to eval_image_crn.py: we import its functions rather
    than copying logic, so the two stay in lockstep.
    """
    import torch
    from PIL import Image

    from lime import RefLIME, MaskLibrary
    from eval_image_crn import (
        select_reference_crn, insertion_deletion_auc, _agg, _spearman,
    )

    img = Image.open(image_path).convert("RGB")
    x = prep(img).unsqueeze(0).to(device)

    expl = RefLIME(model, device=device, grid=(args.grid, args.grid),
                   n_samples=args.n_samples, val_frac=args.val_frac,
                   c=args.c, batch_size=args.batch_size, seed=args.seed)
    _, _, H, W = x.shape
    expl._lib = MaskLibrary(H, W, (args.grid, args.grid),
                            device=device, seed=args.seed)
    lib = expl._lib

    with torch.no_grad():
        target = int(model(x).argmax(dim=1).item())

    # 1) selection
    sel = select_reference_crn(expl, x, references, target,
                               Q=args.select_Q, seed=args.seed,
                               repeat=args.select_repeat)
    best = sel["winner"]

    # 2) attr maps for ALL references (needed for the agreement Spearmans)
    ref_names = list(references.keys())
    attr_maps = {nm: expl._explain_one(x, nm, references[nm], target).attr
                 for nm in ref_names}

    # 3) ins/del cross product
    eval_block = {}
    for m_name in ref_names:
        attr = attr_maps[m_name]
        per_baseline = {
            b_name: insertion_deletion_auc(
                model, x, attr, rho_b, target, lib, device,
                steps=args.steps, repeat=args.measure_repeat)
            for b_name, rho_b in references.items()
        }
        eval_block[m_name] = {
            "agg_all": {
                "insertion_auc": _agg(per_baseline, "insertion_auc"),
                "deletion_auc": _agg(per_baseline, "deletion_auc"),
            },
            "agg_exclude_self": {
                "insertion_auc": _agg(per_baseline, "insertion_auc",
                                      exclude=m_name),
                "deletion_auc": _agg(per_baseline, "deletion_auc",
                                     exclude=m_name),
            },
        }

    # 4) within-image agreement (exclude-self AUCs)
    m_hat_v = np.array([sel["m_hat"][n] for n in ref_names])
    snr_v = np.array([sel["snr"][n] for n in ref_names])
    ins_v = np.array([eval_block[n]["agg_exclude_self"]["insertion_auc"]["mean"]
                      for n in ref_names])
    del_v = np.array([eval_block[n]["agg_exclude_self"]["deletion_auc"]["mean"]
                      for n in ref_names])

    bi = ref_names.index(best)
    ins_rank = int(np.sum(ins_v > ins_v[bi]) + 1)   # 1 = best (highest ins)
    del_rank = int(np.sum(del_v < del_v[bi]) + 1)   # 1 = best (lowest del)

    spearman = {
        "m_hat_vs_insertion": _spearman(m_hat_v, ins_v),   # conj < 0
        "m_hat_vs_deletion": _spearman(m_hat_v, del_v),    # conj > 0
        "snr_vs_insertion": _spearman(snr_v, ins_v),       # conj > 0
        "snr_vs_deletion": _spearman(snr_v, del_v),        # conj < 0
    }

    return {
        "image": os.path.basename(image_path),
        "target": target,
        "selected_reference": best,
        "ranked_by_m": sel["ranked"],
        "m_hat": sel["m_hat"],
        "selected_insertion_rank": ins_rank,
        "selected_deletion_rank": del_rank,
        "insertion_winner": ref_names[int(np.argmax(ins_v))],
        "deletion_winner": ref_names[int(np.argmin(del_v))],
        "n_references": len(ref_names),
        "spearman_exclude_self": spearman,
        "exclude_self_insertion": {n: float(v) for n, v in zip(ref_names, ins_v)},
        "exclude_self_deletion": {n: float(v) for n, v in zip(ref_names, del_v)},
    }


# --------------------------------------------------------------------------- #
#  Cross-image aggregation
# --------------------------------------------------------------------------- #
def _binom_two_sided_p(k, n, p=0.5):
    """Exact two-sided binomial p-value for k successes in n trials under p=0.5.
    Pure-python; no scipy dependency."""
    if n == 0:
        return float("nan")
    from math import comb
    def pmf(i):
        return comb(n, i) * (p ** i) * ((1 - p) ** (n - i))
    obs = pmf(k)
    # sum probabilities of outcomes at most as likely as observed
    total = 0.0
    for i in range(n + 1):
        if pmf(i) <= obs + 1e-12:
            total += pmf(i)
    return min(1.0, total)


def _summ(vals):
    a = np.asarray([v for v in vals if v is not None and not math.isnan(v)],
                   dtype=float)
    if a.size == 0:
        return {"n": 0}
    mean = float(a.mean())
    std = float(a.std(ddof=1)) if a.size > 1 else 0.0
    se = std / math.sqrt(a.size) if a.size > 1 else 0.0
    return {
        "n": int(a.size),
        "mean": mean,
        "std": std,
        "ci95_lo": mean - 1.96 * se,
        "ci95_hi": mean + 1.96 * se,
        "median": float(np.median(a)),
    }


# our hypothesized sign for each correlation: True means "we'd expect < 0"
# (exploratory; NOT a sign the RePro paper predicts)
_EXPECT_NEG = {
    "m_hat_vs_insertion": True,
    "m_hat_vs_deletion": False,
    "snr_vs_insertion": False,
    "snr_vs_deletion": True,
}


def aggregate(results):
    keys = ["m_hat_vs_insertion", "m_hat_vs_deletion",
            "snr_vs_insertion", "snr_vs_deletion"]
    spearman_summary = {}
    for k in keys:
        vals = [r["spearman_exclude_self"][k] for r in results]
        s = _summ(vals)
        # sign test: count images on the predicted side (strict), ignore exact 0
        expect_neg = _EXPECT_NEG[k]
        clean = [v for v in vals if v is not None and not math.isnan(v) and v != 0.0]
        n_eff = len(clean)
        on_side = sum((v < 0) == expect_neg for v in clean)
        s["sign_test"] = {
            "hypothesized_sign": "negative" if expect_neg else "positive",
            "n_nonzero": n_eff,
            "n_on_predicted_side": on_side,
            "frac_on_predicted_side": (on_side / n_eff) if n_eff else float("nan"),
            "binom_p_two_sided": _binom_two_sided_p(on_side, n_eff),
        }
        spearman_summary[k] = s

    # selected reference distribution
    sel_counts = {}
    for r in results:
        sel_counts[r["selected_reference"]] = \
            sel_counts.get(r["selected_reference"], 0) + 1

    # selected-reference faithfulness rank distribution (the robust signal)
    n_ref = results[0]["n_references"] if results else 0
    ins_ranks = [r["selected_insertion_rank"] for r in results]
    del_ranks = [r["selected_deletion_rank"] for r in results]

    def rank_hist(ranks):
        h = {i: 0 for i in range(1, n_ref + 1)}
        for rk in ranks:
            h[rk] = h.get(rk, 0) + 1
        return h

    sel_won_ins = sum(r["insertion_winner"] == r["selected_reference"]
                      for r in results)
    sel_won_del = sum(r["deletion_winner"] == r["selected_reference"]
                      for r in results)
    n = len(results)

    return {
        "n_images": n,
        "n_references": n_ref,
        "selected_reference_counts": sel_counts,
        "selected_insertion_rank": {
            "summary": _summ(ins_ranks),
            "histogram": rank_hist(ins_ranks),
            "frac_bottom2": (sum(rk >= n_ref - 1 for rk in ins_ranks) / n)
                            if n else float("nan"),
        },
        "selected_deletion_rank": {
            "summary": _summ(del_ranks),
            "histogram": rank_hist(del_ranks),
            "frac_bottom2": (sum(rk >= n_ref - 1 for rk in del_ranks) / n)
                            if n else float("nan"),
        },
        "selected_won_insertion_frac": (sel_won_ins / n) if n else float("nan"),
        "selected_won_deletion_frac": (sel_won_del / n) if n else float("nan"),
        "spearman_summary": spearman_summary,
    }


def print_summary(agg):
    n = agg["n_images"]
    nr = agg["n_references"]
    print("\n" + "=" * 70)
    print(f"BENCHMARK SUMMARY  ({n} images, {nr} references each)")
    print("=" * 70)

    print("\nselected reference (residualized-CRN, lowest m):")
    for nm, c in sorted(agg["selected_reference_counts"].items(),
                        key=lambda kv: -kv[1]):
        print(f"    {nm:>8} : {c:>3d}  ({c / n:.0%})")

    print("\nselected reference's FAITHFULNESS rank  (1 = best of "
          f"{nr}; robust to Spearman coarseness):")
    si = agg["selected_insertion_rank"]
    sd = agg["selected_deletion_rank"]
    print(f"  insertion: mean rank {si['summary']['mean']:.2f} "
          f"(median {si['summary']['median']:.0f}) | "
          f"in bottom-2 on {si['frac_bottom2']:.0%} of images")
    print(f"             histogram (rank:count) {si['histogram']}")
    print(f"  deletion : mean rank {sd['summary']['mean']:.2f} "
          f"(median {sd['summary']['median']:.0f}) | "
          f"in bottom-2 on {sd['frac_bottom2']:.0%} of images")
    print(f"             histogram (rank:count) {sd['histogram']}")
    print(f"  selected ref also WON insertion on "
          f"{agg['selected_won_insertion_frac']:.0%} of images; "
          f"won deletion on {agg['selected_won_deletion_frac']:.0%}")

    print("\nEXPLORATORY: m vs ins/del faithfulness -- Spearman across refs, "
          "pooled over images")
    print("  (our own hypothesis; the RePro paper makes NO such claim)")
    print(f"  {'correlation':>20} {'mean':>8} {'95% CI':>18} "
          f"{'hyp':>5} {'on-side':>9} {'binom p':>9}")
    for k, s in agg["spearman_summary"].items():
        if s.get("n", 0) == 0:
            continue
        st = s["sign_test"]
        ci = f"[{s['ci95_lo']:+.2f},{s['ci95_hi']:+.2f}]"
        pred = "<0" if _EXPECT_NEG[k] else ">0"
        print(f"  {k:>20} {s['mean']:>+8.3f} {ci:>18} {pred:>5} "
              f"{st['n_on_predicted_side']:>3d}/{st['n_nonzero']:<3d} "
              f"{st['binom_p_two_sided']:>9.3f}")

    print("\ninterpretation guide:")
    print("  * REMINDER: the m<->faithfulness link is OUR hypothesis, not a")
    print("    RePro claim. The paper only asserts robust ranking by m.")
    print("  * If our hypothesis held, each correlation's mean would sit on its")
    print("    hypothesized side with sign-test p < 0.05.")
    print("  * The faithfulness-rank histogram is the headline: if the")
    print("    CRN-selected (lowest-m) reference is reliably mid/bottom rank,")
    print("    then low-m selection does NOT buy ins/del faithfulness on this")
    print("    model -- a finding about real reference families, beyond the paper.")
    print("  * Spearman-over-5-points is coarse; trust the pooled sign test and")
    print("    the rank histogram over any single image's correlation.")
    print("=" * 70)


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #
def main():
    args = build_argparser().parse_args()
    per_dir = os.path.join(args.out_dir, "per_image")
    os.makedirs(per_dir, exist_ok=True)

    paths = sorted(glob.glob(args.glob))
    if not paths:
        raise SystemExit(f"no images matched glob {args.glob!r}")
    if args.limit:
        paths = paths[:args.limit]
    print(f"found {len(paths)} images for glob {args.glob!r}")

    # torch / model loaded once, reused across images
    import torch  # noqa: F401
    import torchvision as tv
    from lime import default_reference_family

    device = args.device
    weights = tv.models.get_model_weights(args.model).DEFAULT
    model = tv.models.get_model(args.model, weights=weights).eval().to(device)
    prep = weights.transforms()

    fam_all = default_reference_family()
    chosen = [r.strip() for r in args.references.split(",")]
    references = {k: fam_all[k] for k in chosen if k in fam_all}
    if not references:
        raise SystemExit(f"no valid references in {chosen!r}; "
                         f"available: {list(fam_all)}")

    results = []
    failed = []
    for i, path in enumerate(paths, 1):
        stem = os.path.splitext(os.path.basename(path))[0]
        out_json = os.path.join(per_dir, f"{stem}.json")
        if args.resume and os.path.exists(out_json):
            with open(out_json) as f:
                results.append(json.load(f))
            print(f"[{i}/{len(paths)}] {stem}: cached, skipped")
            continue
        print(f"[{i}/{len(paths)}] {stem}: running ...", flush=True)
        try:
            r = run_one_image(path, model, prep, references, args, device)
            with open(out_json, "w") as f:
                json.dump(r, f, indent=2)
            results.append(r)
            sp = r["spearman_exclude_self"]
            print(f"    target={r['target']} selected={r['selected_reference']} "
                  f"ins_rank={r['selected_insertion_rank']}/{r['n_references']} "
                  f"del_rank={r['selected_deletion_rank']}/{r['n_references']} "
                  f"m_vs_ins={sp['m_hat_vs_insertion']:+.2f} "
                  f"m_vs_del={sp['m_hat_vs_deletion']:+.2f}")
        except Exception as e:  # keep the sweep alive; log and continue
            failed.append((stem, repr(e)))
            print(f"    FAILED: {e}")
            traceback.print_exc(file=sys.stdout)

    if not results:
        raise SystemExit("no successful images; nothing to aggregate")

    agg = aggregate(results)
    agg["failed"] = failed
    agg["config"] = vars(args)
    with open(os.path.join(args.out_dir, "benchmark_summary.json"), "w") as f:
        json.dump(agg, f, indent=2)

    print_summary(agg)
    if failed:
        print(f"\n{len(failed)} image(s) failed: "
              f"{', '.join(s for s, _ in failed)}")
    print(f"\nper-image JSON in {per_dir}/  |  "
          f"aggregate in {os.path.join(args.out_dir, 'benchmark_summary.json')}")


if __name__ == "__main__":
    main()