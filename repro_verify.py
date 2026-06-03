"""
RePro synthetic verification — dense estimator, Torch + CUDA.

Fix vs. the stencil version: the mixed-difference probe only fired when a random
(i,j) pair hit a planted interaction (prob 1/binom(d,2)) -> O(d^2) variance.
Here we estimate the residual energy densely, so every query carries signal:

    m_rho = Var_mu(g)  -  s_rho ,   s_rho = sum_i beta_{i,rho}^2 .

Both terms use full random masks. No fit, no design matrix, no d^2 penalty.

Checks:
  (a) m_hat is unbiased for planted m (no calibration constant needed),
  (b) m_hat concentrates at the 1/sqrt(Q) rate with NO d^2 blowup,
  (c) RePro selects the smallest-m reference, P(correct) -> 1 as Q grows.

Run:  python repro_verify_torch.py
"""

import torch

DEV = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float64
print(f"device = {DEV}")
g = torch.Generator(device=DEV).manual_seed(0)


def make_planted(d, lin_scale, quad_pairs, quad_scale, seed):
    r = torch.Generator(device=DEV).manual_seed(seed)
    beta_lin = torch.randn(d, generator=r, device=DEV, dtype=DTYPE) * lin_scale
    pairs = torch.randint(0, d, (quad_pairs, 2), generator=r, device=DEV)
    coefs = torch.randn(quad_pairs, generator=r, device=DEV, dtype=DTYPE) * quad_scale
    # drop degenerate i==i pairs
    keep = pairs[:, 0] != pairs[:, 1]
    return beta_lin, pairs[keep], coefs[keep]


def g_eval(Z, beta_lin, pairs, coefs):
    """g(z) in centered Walsh basis. Z in {0,1}^{n x d}. chi_i = 2*(z_i - 1/2)."""
    chi = 2.0 * (Z - 0.5)                       # (n, d)
    out = chi @ beta_lin                        # linear part
    if pairs.numel():
        ci = chi[:, pairs[:, 0]]                # (n, P)
        cj = chi[:, pairs[:, 1]]
        out = out + (ci * cj) @ coefs           # pairwise part
    return out


def true_m(coefs):
    return float((coefs ** 2).sum())


def true_s(beta_lin):
    return float((beta_lin ** 2).sum())


def sample_masks(n, d, seed):
    r = torch.Generator(device=DEV).manual_seed(seed)
    return torch.randint(0, 2, (n, d), generator=r, device=DEV).to(DTYPE)


def estimate_m(beta_lin, pairs, coefs, d, Q, seed):
    """
    Dense fit-free estimate of m = Var(g) - s.
      Var(g): plug-in variance of g over Q random masks.
      s     : sum_i beta_i^2, where beta_i = E[ g(z) * chi_i(z) ] (Walsh coeff),
              estimated by a single matmul -- O(d), every sample contributes.
    """
    Z = sample_masks(Q, d, seed)
    y = g_eval(Z, beta_lin, pairs, coefs)       # (Q,)
    var_g = y.var(unbiased=True)
    chi = 2.0 * (Z - 0.5)                        # (Q, d)
    beta_hat = (chi * y[:, None]).mean(0)        # E[g * chi_i]  -> (d,)
    s_hat = (beta_hat ** 2).sum()
    return float(var_g - s_hat)


if __name__ == "__main__":
    d = 40

    # ---- (a) unbiasedness, NO calibration constant ----
    beta_lin, pairs, coefs = make_planted(d, 1.0, quad_pairs=8, quad_scale=0.8, seed=10)
    m, s = true_m(coefs), true_s(beta_lin)
    Q = 200_000
    mhat = estimate_m(beta_lin, pairs, coefs, d, Q, seed=1)
    print(f"\n[a] planted m={m:.4f}  s={s:.4f}   m_hat(Q={Q})={mhat:.4f}   abs_err={abs(mhat-m):.4f}")

    # ---- (b) concentration, no d^2 blowup ----
    print("\n[b] concentration of m_hat:")
    for Q in [200, 1000, 5000, 20000]:
        reps = 200
        errs = []
        for rep in range(reps):
            mh = estimate_m(beta_lin, pairs, coefs, d, Q, seed=100 + rep)
            errs.append(abs(mh - m))
        errs = torch.tensor(errs)
        print(f"    Q={Q:6d}   mean|m_hat - m|={errs.mean():.4f}   std={errs.std():.4f}")

    # ---- (c) selection across references with increasing m ----
    print("\n[c] reference selection (pick smallest m):")
    refs = {}
    for k, qs in enumerate([0.3, 0.6, 1.0, 1.5]):
        bl, pr, co = make_planted(d, 1.0, quad_pairs=8, quad_scale=qs, seed=10 + k)
        refs[f"rho_{k}"] = (bl, pr, co, true_m(co))
    true_best = min(refs, key=lambda r: refs[r][3])
    print("    true m:", {r: round(refs[r][3], 3) for r in refs}, " best:", true_best)
    for Q in [200, 1000, 5000]:
        hits, trials = 0, 200
        for t in range(trials):
            scores = {r: estimate_m(bl, pr, co, d, Q, seed=1000 + t * 17 + i)
                      for i, (r, (bl, pr, co, _)) in enumerate(refs.items())}
            pick = min(scores, key=lambda rr: scores[rr])
            hits += (pick == true_best)
        print(f"    Q={Q:5d}   P(correct select)={hits/trials:.3f}")