"""
RePro — RESIDUALIZED CRN paired contrasts (the fix).

The linear-decorrelation grid showed CRN's advantage was entirely a shared-linear-block
effect: it collapsed from 1e5x to 1x the moment beta_lin decorrelated (alpha_lin: 1 -> 0.8),
because the decorrelated linear part then dominated the per-sample contrast variance.

FIX: residualize. The linear block is the one part we can estimate well and cheaply
(dense, O(d), and LIME recovers it anyway). Estimate beta_hat_rho per reference, subtract
its linear contribution from g PER SAMPLE, and contrast the RESIDUALS:
    g_rho(z)  ->  rtil_rho(z) = g_rho(z) - <beta_hat_rho, chi(z)>
Then the only structure left to cancel/differ is the interaction part. The contrast
variance should now track alpha_int (which was flat & favorable), not alpha_lin.

Estimator detail: beta_hat from the odd fold is itself noisy, and a per-sample plug-in
subtraction using the SAME masks would bias the residual variance downward (overfitting
the linear coeffs to the same Q). We avoid this with a clean split:
   - estimate beta_hat on a HALF of the masks,
   - residualize & contrast on the OTHER half.
This keeps E[residual interaction energy] = m_rho (linear removed in expectation, no
in-sample overfit).

Outputs the same (alpha_lin, alpha_int) grid as before for direct comparison.

Run:  python repro_residualized.py
"""

import torch

DEV = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float64
print(f"device = {DEV}")

LIN_NORM = 7.7


def make_family(d, alpha_lin, alpha_int, seed, targets=(2.0, 2.3, 2.7, 3.2)):
    r = torch.Generator(device=DEV).manual_seed(seed)
    shared_lin = torch.randn(d, generator=r, device=DEV, dtype=DTYPE)
    shared_lin = shared_lin / shared_lin.norm()
    pairs = torch.randint(0, d, (8, 2), generator=r, device=DEV)
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]
    P = pairs.shape[0]
    base_dir = torch.randn(P, generator=r, device=DEV, dtype=DTYPE)
    base_dir = base_dir / base_dir.norm()
    fam = []
    for tm in targets:
        own_lin = torch.randn(d, generator=r, device=DEV, dtype=DTYPE)
        own_lin = own_lin / own_lin.norm()
        bl_dir = alpha_lin * shared_lin + (1 - alpha_lin) * own_lin
        bl_dir = bl_dir / bl_dir.norm()
        beta_lin = bl_dir * LIN_NORM
        own = torch.randn(P, generator=r, device=DEV, dtype=DTYPE)
        own = own / own.norm()
        d_k = alpha_int * base_dir + (1 - alpha_int) * own
        d_k = d_k / d_k.norm()
        co = d_k * (tm ** 0.5)
        fam.append((beta_lin, pairs, co, tm))
    return fam


def g_eval(Z, beta_lin, pairs, coefs):
    chi = 2.0 * (Z - 0.5)
    out = chi @ beta_lin
    if pairs.numel():
        out = out + (chi[:, pairs[:, 0]] * chi[:, pairs[:, 1]]) @ coefs
    return out


def masks(Q, d, seed):
    r = torch.Generator(device=DEV).manual_seed(seed)
    Z = torch.randint(0, 2, (Q, d), generator=r, device=DEV).to(DTYPE)
    return Z, 1.0 - Z


def estimate_beta(Z, Zc, bl, pr, co):
    """Dense linear-coeff estimate via odd fold: beta_i = E[o * chi_i]."""
    y = g_eval(Z, bl, pr, co); yc = g_eval(Zc, bl, pr, co)
    o = 0.5 * (y - yc)
    chi = 2.0 * (Z - 0.5)
    return (chi * o[:, None]).mean(0)            # (d,)


def residual_energy_terms(Z, Zc, bl, pr, co, beta_hat):
    """Per-sample interaction-energy proxy AFTER removing the linear part beta_hat."""
    chi = 2.0 * (Z - 0.5)
    lin = chi @ beta_hat                          # estimated linear contribution
    y = g_eval(Z, bl, pr, co) - lin
    chic = 2.0 * (Zc - 0.5)
    linc = chic @ beta_hat
    yc = g_eval(Zc, bl, pr, co) - linc
    e = 0.5 * (y + yc)                            # even fold of residual
    o = 0.5 * (y - yc)                            # odd fold of residual (linear ~removed)
    q = (e - e.mean())**2 + (o - o.mean())**2
    return q


def contrast_resid(a, b, d, Q, seed):
    """Split-sample residualized CRN contrast of m_a - m_b on common masks."""
    Z, Zc = masks(Q, d, seed)
    h = Q // 2
    Ze, Zce = Z[:h], Zc[:h]                       # estimation half
    Zc2, Zcc2 = Z[h:], Zc[h:]                     # contrast half (shared across a,b)
    ba = estimate_beta(Ze, Zce, *a[:3])
    bb = estimate_beta(Ze, Zce, *b[:3])
    qa = residual_energy_terms(Zc2, Zcc2, *a[:3], ba)
    qb = residual_energy_terms(Zc2, Zcc2, *b[:3], bb)
    return float((qa - qb).mean())


def m_independent_resid(bl, pr, co, d, Q, seed):
    Z, Zc = masks(Q, d, seed)
    h = Q // 2
    b = estimate_beta(Z[:h], Zc[:h], bl, pr, co)
    q = residual_energy_terms(Z[h:], Zc[h:], bl, pr, co, b)
    return float(q.mean())


def tournament_pick(fam, d, Q, seed):
    n = len(fam); wins = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if contrast_resid(fam[i], fam[j], d, Q, seed + i * 13 + j) < 0: wins[i] += 1
            else: wins[j] += 1
    return max(range(n), key=lambda k: wins[k])


if __name__ == "__main__":
    d = 40
    Q = 5000
    alpha_lins = [1.0, 0.8, 0.5, 0.2, 0.0]
    alpha_ints = [1.0, 0.5, 0.0]

    print(f"\nRESIDUALIZED grid (d={d}, Q={Q}, m={{2.0,2.3,2.7,3.2}}, best=rho_0)")
    print("Each cell: var-reduction (CRN vs independent) / P(select best)\n")
    header = "alpha_lin \\ alpha_int |" + "".join(f"{ai:>18.1f}" for ai in alpha_ints)
    print(header); print("-" * len(header))

    for al in alpha_lins:
        cells = []
        for ai in alpha_ints:
            hits, trials = 0, 150
            for t in range(trials):
                fam = make_family(d, al, ai, seed=10)
                hits += (tournament_pick(fam, d, Q, 300 + t * 29) == 0)
            psel = hits / trials
            fam = make_family(d, al, ai, seed=10)
            crn, ind = [], []
            for t in range(150):
                crn.append(contrast_resid(fam[0], fam[1], d, Q, 13000 + t))
                ind.append(m_independent_resid(*fam[0][:3], d, Q, 14000 + t)
                           - m_independent_resid(*fam[1][:3], d, Q, 15000 + t))
            crn = torch.tensor(crn); ind = torch.tensor(ind)
            vr = (ind.std() / crn.std()) ** 2 if crn.std() > 0 else float('inf')
            cells.append(f"{vr:>8.0f}x/{psel:.2f}")
        print(f"{al:>20.1f} |" + "".join(f"{c:>18}" for c in cells))

    print("\nCompare to the non-residualized grid: the alpha_lin=0 row is the test.")
    print("If residualization works, that row should NO LONGER collapse to 1x/0.57 --")
    print("the contrast variance should now follow alpha_int, not alpha_lin.")