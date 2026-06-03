"""
RePro — CRN paired-contrast reference selection, with the ALPHA-SWEEP viability test.

Established in prior runs:
  - Reading m_rho off each reference independently is floored by Var(g): useless at
    realistic (tight) gaps.
  - Estimating the DIFFERENCE m_a - m_b on COMMON random masks (CRN) cancels the shared
    high-variance backbone; sign of the difference -> ranking. Tournament (Copeland wins)
    selects the best reference.

Caveat that this script now tests directly: the huge variance reduction only holds when
references SHARE structure. Real reference operators reshape g in correlated-but-not-
collinear ways. We introduce a correlation knob alpha in [0,1]:
    alpha = 1 : references are scalar multiples of a shared interaction block (best case)
    alpha = 0 : each reference has independent structure (worst case, ~ independent est.)
We sweep alpha and report the variance-reduction factor and selection accuracy, to find
the viability frontier (where CRN stops beating independent estimation).

Run:  python repro_verify_torch.py
"""

import torch

DEV = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float64
print(f"device = {DEV}")


def make_family(d, alpha, seed, targets=(2.0, 2.3, 2.7, 3.2)):
    """
    Each reference = shared backbone blended with an idiosyncratic perturbation, by alpha.
      beta_lin_k = shared_lin (shared linear part kept common; CRN cancels it regardless)
      interaction block: directions = normalize( alpha*base_dir + (1-alpha)*own_dir_k )
                         then scaled so true m_k = targets[k].
    alpha=1 -> all refs collinear in interaction space (shared structure).
    alpha=0 -> each ref's interactions point in its own random direction.
    """
    r = torch.Generator(device=DEV).manual_seed(seed)
    shared_lin = torch.randn(d, generator=r, device=DEV, dtype=DTYPE)
    pairs = torch.randint(0, d, (8, 2), generator=r, device=DEV)
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]
    P = pairs.shape[0]
    base_dir = torch.randn(P, generator=r, device=DEV, dtype=DTYPE)
    base_dir = base_dir / base_dir.norm()
    fam = []
    for k, tm in enumerate(targets):
        own = torch.randn(P, generator=r, device=DEV, dtype=DTYPE)
        own = own / own.norm()
        d_k = alpha * base_dir + (1 - alpha) * own
        d_k = d_k / d_k.norm()
        co = d_k * (tm ** 0.5)              # ||co||^2 = tm  -> true m = tm
        fam.append((shared_lin, pairs, co, tm))
    return fam


def g_eval(Z, beta_lin, pairs, coefs):
    chi = 2.0 * (Z - 0.5)
    out = chi @ beta_lin
    if pairs.numel():
        out = out + (chi[:, pairs[:, 0]] * chi[:, pairs[:, 1]]) @ coefs
    return out


def folds(Z, Zc, bl, pr, co):
    y = g_eval(Z, bl, pr, co); yc = g_eval(Zc, bl, pr, co)
    return 0.5 * (y + yc), 0.5 * (y - yc)      # even, odd


def masks(Q, d, seed):
    r = torch.Generator(device=DEV).manual_seed(seed)
    Z = torch.randint(0, 2, (Q, d), generator=r, device=DEV).to(DTYPE)
    return Z, 1.0 - Z


def m_independent(bl, pr, co, d, Q, seed):
    Z, Zc = masks(Q, d, seed)
    e, o = folds(Z, Zc, bl, pr, co)
    chi = 2.0 * (Z - 0.5)
    s_hat = ((chi * o[:, None]).mean(0) ** 2).sum()
    return float(e.var(unbiased=True) + (o.var(unbiased=True) - s_hat))


def contrast(a, b, d, Q, seed):
    """CRN paired estimate of m_a - m_b. Returns scalar mean."""
    Z, Zc = masks(Q, d, seed)
    ea, oa = folds(Z, Zc, *a[:3])
    eb, ob = folds(Z, Zc, *b[:3])
    chi = 2.0 * (Z - 0.5)
    sa = ((chi * oa[:, None]).mean(0) ** 2).sum()
    sb = ((chi * ob[:, None]).mean(0) ** 2).sum()
    qa = (ea - ea.mean())**2 + (oa - oa.mean())**2
    qb = (eb - eb.mean())**2 + (ob - ob.mean())**2
    return float((qa - qb).mean() - (sa - sb))


def tournament_pick(fam, d, Q, seed):
    n = len(fam); wins = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            c = contrast(fam[i], fam[j], d, Q, seed + i * 13 + j)
            if c < 0: wins[i] += 1
            else:     wins[j] += 1
    return max(range(n), key=lambda k: wins[k])


if __name__ == "__main__":
    d = 40
    Q = 5000
    print(f"\nalpha sweep  (d={d}, Q={Q}, true m={{2.0,2.3,2.7,3.2}}, best=rho_0)")
    print(f"{'alpha':>6} | {'P(select rho_0)':>16} | {'var-reduction (rho0 vs rho1)':>28}")
    print("-" * 58)
    for alpha in [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]:
        # selection accuracy
        hits, trials = 0, 200
        for t in range(trials):
            fam = make_family(d, alpha, seed=10)        # family fixed; randomness in masks
            hits += (tournament_pick(fam, d, Q, 300 + t * 29) == 0)
        psel = hits / trials

        # variance reduction: CRN paired diff vs independent diff, rho_0 vs rho_1
        fam = make_family(d, alpha, seed=10)
        crn, ind = [], []
        for t in range(200):
            crn.append(contrast(fam[0], fam[1], d, Q, 13000 + t))
            ind.append(m_independent(*fam[0][:3], d, Q, 14000 + t)
                       - m_independent(*fam[1][:3], d, Q, 15000 + t))
        crn = torch.tensor(crn); ind = torch.tensor(ind)
        vr = (ind.std() / crn.std()) ** 2 if crn.std() > 0 else float('inf')
        print(f"{alpha:6.1f} | {psel:16.3f} | {vr:26.1f}x")

    print("\nViability frontier = alpha where var-reduction crosses ~2x and P(select)")
    print("falls toward chance. Real reference operators on one input must sit ABOVE")
    print("that alpha for CRN selection to beat just running the explanations.")