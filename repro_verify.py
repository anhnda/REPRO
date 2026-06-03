"""
RePro — LAST TRY: paired contrasts via common random numbers (CRN).

Diagnosis from prior runs: any statistic that reads m_rho off the full function g
has variance floored by Var(g), not by m. Estimating each m_rho independently and
then comparing fails at realistic (tight) gaps -> needed ~80k queries/ref for 0.81.

Fix that actually targets the floor: selection only needs the RANKING, i.e. the sign
of  D = m_rho - m_rho'  for reference pairs. Estimate that DIFFERENCE directly on the
SAME random masks (common random numbers). When two references of the same input share
structure, the shared high-variance part cancels in the per-sample difference, so
Var(D_hat) depends on how much rho and rho' DIFFER, not on Var(g). Rank by the resulting
tournament (Copeland: each ref's number of pairwise wins).

Per-sample contrast (antithetic even-fold interaction proxy, differenced across refs):
   For mask z (and complement), even-fold e_rho(z) = 1/2(g_rho(z)+g_rho(1-z)).
   Centered even-fold square  q_rho(z) = (e_rho(z) - mean_e_rho)^2  has  E[q_rho] = Var(e_rho).
   Paired contrast for refs a,b on the SAME z:  c_ab(z) = q_a(z) - q_b(z),
   E[c_ab] = Var(e_a) - Var(e_b)  -> sign gives the even-interaction-energy ordering.
   (Odd-order handled symmetrically with the odd fold; summed in.)
The variance of mean(c_ab) is driven by Var(q_a - q_b), which is small when g_a, g_b
share structure -- the regime of references on one fixed input.

Checks:
  (a) paired contrast recovers sign of (m_a - m_b) far cheaper than independent m_hat,
  (b) HARD tournament selection, m in {2.0,2.3,2.7,3.2}, shared-structure refs,
  (c) variance comparison: Var(paired contrast) vs Var(independent difference).

Run:  python repro_verify_torch.py
"""

import torch

DEV = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float64
print(f"device = {DEV}")


def make_family(d, n_refs, seed):
    """References on ONE input share a common linear+interaction backbone; each ref
    perturbs the interaction block (models how a reference operator reshapes g).
    Returns list of (beta_lin, pairs, coefs, true_m), with tightly-spaced m."""
    r = torch.Generator(device=DEV).manual_seed(seed)
    d_ = d
    base_lin = torch.randn(d_, generator=r, device=DEV, dtype=DTYPE)          # shared
    pairs = torch.randint(0, d_, (8, 2), generator=r, device=DEV)
    keep = pairs[:, 0] != pairs[:, 1]
    pairs = pairs[keep]
    base_co = torch.randn(pairs.shape[0], generator=r, device=DEV, dtype=DTYPE)  # shared shape
    targets = [2.0, 2.3, 2.7, 3.2][:n_refs]
    fam = []
    for tm in targets:
        co = base_co * (tm / float((base_co**2).sum()))**0.5   # same directions, scaled to m
        fam.append((base_lin, pairs, co, tm))
    return fam


def g_eval(Z, beta_lin, pairs, coefs):
    chi = 2.0 * (Z - 0.5)
    out = chi @ beta_lin
    if pairs.numel():
        out = out + (chi[:, pairs[:, 0]] * chi[:, pairs[:, 1]]) @ coefs
    return out


def even_fold_sq(Z, Zc, bl, pr, co):
    y = g_eval(Z, bl, pr, co); yc = g_eval(Zc, bl, pr, co)
    e = 0.5 * (y + yc)
    o = 0.5 * (y - yc)
    # per-sample interaction proxy: centered even square + (odd square - linear proj)
    # linear proj removed densely from odd fold:
    return e, o


def masks(Q, d, seed):
    r = torch.Generator(device=DEV).manual_seed(seed)
    Z = torch.randint(0, 2, (Q, d), generator=r, device=DEV).to(DTYPE)
    return Z, 1.0 - Z


def m_independent(bl, pr, co, d, Q, seed):
    Z, Zc = masks(Q, d, seed)
    e, o = even_fold_sq(Z, Zc, bl, pr, co)
    chi = 2.0 * (Z - 0.5)
    s_hat = ((chi * o[:, None]).mean(0) ** 2).sum()
    return float(e.var(unbiased=True) + (o.var(unbiased=True) - s_hat))


def contrast_sign(a, b, d, Q, seed):
    """E[c]=Var(e_a)-Var(e_b)+(odd terms); estimated on COMMON masks. Returns mean, std."""
    Z, Zc = masks(Q, d, seed)
    bl_a, pr_a, co_a, _ = a
    bl_b, pr_b, co_b, _ = b
    ea, oa = even_fold_sq(Z, Zc, bl_a, pr_a, co_a)
    eb, ob = even_fold_sq(Z, Zc, bl_b, pr_b, co_b)
    chi = 2.0 * (Z - 0.5)
    sa = ((chi * oa[:, None]).mean(0) ** 2).sum()
    sb = ((chi * ob[:, None]).mean(0) ** 2).sum()
    qa = (ea - ea.mean())**2 + ( (oa - oa.mean())**2 )   # interaction-energy proxy per sample
    qb = (eb - eb.mean())**2 + ( (ob - ob.mean())**2 )
    c = (qa - qb)                                        # paired, common random numbers
    # subtract the linear part of the odd contribution (constant offset, no extra variance):
    offset = (sa - sb)
    cmean = c.mean() - offset
    return float(cmean), float(c.std() / (Q ** 0.5))


if __name__ == "__main__":
    d = 40
    fam = make_family(d, 4, seed=10)
    print("    true m:", {f"rho_{k}": round(fam[k][3], 3) for k in range(4)}, " best: rho_0")

    # ---- (a) paired contrast sign accuracy vs independent, small Q ----
    print("\n[a] adjacent-pair sign accuracy (rho_0 vs rho_1, true diff = -0.3):")
    for Q in [1000, 5000, 20000]:
        ind_hits = crn_hits = 0
        trials = 300
        for t in range(trials):
            # independent: two separate m_hats, different masks
            ma = m_independent(*fam[0][:3], d, Q, 7000 + t)
            mb = m_independent(*fam[1][:3], d, Q, 9000 + t)
            ind_hits += (ma - mb) < 0
            # CRN paired contrast
            cm, _ = contrast_sign(fam[0], fam[1], d, Q, 5000 + t)
            crn_hits += cm < 0
        print(f"    Q={Q:6d}   P(sign right): independent={ind_hits/trials:.3f}   CRN-paired={crn_hits/trials:.3f}")

    # ---- (b) full tournament selection (Copeland wins) ----
    print("\n[b] tournament selection via paired contrasts:")
    for Q in [1000, 5000, 20000]:
        hits, trials = 0, 200
        for t in range(trials):
            wins = [0, 0, 0, 0]
            for i in range(4):
                for j in range(i + 1, 4):
                    cm, _ = contrast_sign(fam[i], fam[j], d, Q, 200 + t * 31 + i * 4 + j)
                    if cm < 0: wins[i] += 1
                    else:      wins[j] += 1
            pick = max(range(4), key=lambda k: wins[k])
            hits += (pick == 0)
        print(f"    Q={Q:6d}   P(correct select)={hits/trials:.3f}")

    # ---- (c) variance: paired contrast vs independent difference ----
    print("\n[c] std of the (rho_0 - rho_1) difference estimate at Q=5000:")
    Q = 5000
    crn_vals, ind_vals = [], []
    for t in range(300):
        cm, _ = contrast_sign(fam[0], fam[1], d, Q, 13000 + t)
        crn_vals.append(cm)
        ind_vals.append(m_independent(*fam[0][:3], d, Q, 14000 + t)
                        - m_independent(*fam[1][:3], d, Q, 15000 + t))
    crn_vals = torch.tensor(crn_vals); ind_vals = torch.tensor(ind_vals)
    print(f"    independent diff: mean={ind_vals.mean():.4f} std={ind_vals.std():.4f}")
    print(f"    CRN paired diff : mean={crn_vals.mean():.4f} std={crn_vals.std():.4f}")
    print(f"    variance reduction factor ~ {(ind_vals.std()/crn_vals.std())**2:.1f}x")