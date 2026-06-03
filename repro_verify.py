"""
RePro synthetic verification — antithetic-pairs estimator, Torch + CUDA.

Why this form: the previous estimator m = Var(g) - s subtracts two large numbers
when the reference is good (m << s), causing catastrophic cancellation exactly in
the regime we want to select. Antithetic folding fixes the dominant part.

Antithetic fold. Draw z and its complement z' = 1 - z. In the centered Walsh basis
chi_S(z') = (-1)^{|S|} chi_S(z), so:
    even fold  e(z) = 1/2 ( g(z) + g(z') ) = beta_0 + sum_{|S| even >=2} beta_S chi_S
    odd  fold  o(z) = 1/2 ( g(z) - g(z') ) = sum_{|S| odd}            beta_S chi_S
=> Var(e) = sum_{|S| even >=2} beta_S^2        (NO linear term, NO big subtraction)
   Var(o) = sum_{|S| odd}      beta_S^2        (= linear energy s + odd interactions)
Interaction residual energy:
    m = [ Var(e) ]                      (even interactions, clean)
      + [ Var(o) - s ]                  (odd interactions; subtract linear)
where s = sum_i beta_i^2 estimated densely. The even block — the usual dominant part
for pairwise structure — is now estimated with NO cancellation. Only odd-order
interactions carry a residual subtraction, and that term is small when present.

Checks:
  (a) m_hat unbiased, low error even though m << s,
  (b) concentration 1/sqrt(Q), no d^2 blowup,
  (c) HARD selection: tightly-spaced m in {2.0, 2.3, 2.7, 3.2}.

Run:  python repro_verify_torch.py
"""

import torch

DEV = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float64
print(f"device = {DEV}")


def make_planted(d, lin_scale, quad_pairs, quad_scale, seed, target_m=None):
    r = torch.Generator(device=DEV).manual_seed(seed)
    beta_lin = torch.randn(d, generator=r, device=DEV, dtype=DTYPE) * lin_scale
    pairs = torch.randint(0, d, (quad_pairs, 2), generator=r, device=DEV)
    coefs = torch.randn(quad_pairs, generator=r, device=DEV, dtype=DTYPE) * quad_scale
    keep = pairs[:, 0] != pairs[:, 1]
    pairs, coefs = pairs[keep], coefs[keep]
    if target_m is not None:                       # rescale pairwise block to exact m
        cur = float((coefs ** 2).sum())
        coefs = coefs * (target_m / cur) ** 0.5
    return beta_lin, pairs, coefs


def g_eval(Z, beta_lin, pairs, coefs):
    chi = 2.0 * (Z - 0.5)
    out = chi @ beta_lin
    if pairs.numel():
        out = out + (chi[:, pairs[:, 0]] * chi[:, pairs[:, 1]]) @ coefs
    return out


def true_m(coefs):
    return float((coefs ** 2).sum())


def true_s(beta_lin):
    return float((beta_lin ** 2).sum())


def estimate_m(beta_lin, pairs, coefs, d, Q, seed):
    """Antithetic-pairs fit-free estimate of interaction residual energy m."""
    r = torch.Generator(device=DEV).manual_seed(seed)
    Z = torch.randint(0, 2, (Q, d), generator=r, device=DEV).to(DTYPE)
    Zc = 1.0 - Z
    y = g_eval(Z, beta_lin, pairs, coefs)
    yc = g_eval(Zc, beta_lin, pairs, coefs)
    e = 0.5 * (y + yc)                              # even fold
    o = 0.5 * (y - yc)                              # odd fold
    var_e = e.var(unbiased=True)                    # even interaction energy (clean)
    var_o = o.var(unbiased=True)                    # = s + odd interactions
    chi = 2.0 * (Z - 0.5)
    beta_hat = (chi * o[:, None]).mean(0)           # E[o * chi_i] = beta_i
    s_hat = (beta_hat ** 2).sum()
    return float(var_e + (var_o - s_hat))


if __name__ == "__main__":
    d = 40

    # ---- (a) unbiasedness with m << s ----
    beta_lin, pairs, coefs = make_planted(d, 1.0, 8, 0.8, seed=10)
    m, s = true_m(coefs), true_s(beta_lin)
    Q = 200_000
    mhat = estimate_m(beta_lin, pairs, coefs, d, Q, seed=1)
    print(f"\n[a] planted m={m:.4f}  s={s:.4f}  (m/s={m/s:.3f})   m_hat={mhat:.4f}   abs_err={abs(mhat-m):.4f}")

    # ---- (b) concentration ----
    print("\n[b] concentration of m_hat:")
    for Q in [200, 1000, 5000, 20000]:
        errs = torch.tensor([abs(estimate_m(beta_lin, pairs, coefs, d, Q, 100 + rep) - m)
                             for rep in range(200)])
        print(f"    Q={Q:6d}   mean|m_hat - m|={errs.mean():.4f}   std={errs.std():.4f}")

    # ---- (c) HARD selection: tightly-spaced m ----
    print("\n[c] HARD reference selection, m in {2.0, 2.3, 2.7, 3.2}:")
    refs = {}
    for k, tm in enumerate([2.0, 2.3, 2.7, 3.2]):
        bl, pr, co = make_planted(d, 1.0, 8, 0.8, seed=10 + k, target_m=tm)
        refs[f"rho_{k}"] = (bl, pr, co, true_m(co))
    true_best = min(refs, key=lambda r: refs[r][3])
    print("    true m:", {r: round(refs[r][3], 3) for r in refs}, " best:", true_best)
    for Q in [1000, 5000, 20000, 80000]:
        hits, trials = 0, 200
        for t in range(trials):
            scores = {r: estimate_m(bl, pr, co, d, Q, 1000 + t * 17 + i)
                      for i, (r, (bl, pr, co, _)) in enumerate(refs.items())}
            pick = min(scores, key=lambda rr: scores[rr])
            hits += (pick == true_best)
        print(f"    Q={Q:6d}   P(correct select)={hits/trials:.3f}")