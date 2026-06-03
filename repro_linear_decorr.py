"""
RePro — LINEAR-DECORRELATION stress test (the untested case).

Prior sweep decorrelated only the interaction block while holding the linear block
beta_lin COMMON across references. CRN stayed >1000x because the shared linear backbone
(where the variance lives: s~59 vs m~2) cancels regardless. THIS is the part that flattered
the method.

Here we decorrelate the LINEAR block by its own knob alpha_lin, independently of the
interaction knob alpha_int:
    beta_lin_k = normalize( alpha_lin * shared_lin + (1-alpha_lin) * own_lin_k ) * lin_norm
    interaction dirs as before, controlled by alpha_int
alpha_lin = 1 : linear backbone fully shared (previous best case)
alpha_lin = 0 : each reference has an independent linear part (worst case)

We expect the CRN variance reduction to DROP as alpha_lin falls, because the dominant
variance source no longer cancels. The question is whether it stays above the ~2x
viability line (and whether selection survives) at realistic alpha_lin.

Outputs a 2-D grid: var-reduction and P(select best) over (alpha_lin, alpha_int).

Run:  python repro_linear_decorr.py
"""

import torch

DEV = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float64
print(f"device = {DEV}")

LIN_NORM = 7.7          # ||beta_lin|| so that s = ||.||^2 ~ 59, matching earlier runs


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
        # linear block: blend shared with own, by alpha_lin
        own_lin = torch.randn(d, generator=r, device=DEV, dtype=DTYPE)
        own_lin = own_lin / own_lin.norm()
        bl_dir = alpha_lin * shared_lin + (1 - alpha_lin) * own_lin
        bl_dir = bl_dir / bl_dir.norm()
        beta_lin = bl_dir * LIN_NORM
        # interaction block: blend shared with own, by alpha_int, scaled to true m = tm
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


def folds(Z, Zc, bl, pr, co):
    y = g_eval(Z, bl, pr, co); yc = g_eval(Zc, bl, pr, co)
    return 0.5 * (y + yc), 0.5 * (y - yc)


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
            if contrast(fam[i], fam[j], d, Q, seed + i * 13 + j) < 0: wins[i] += 1
            else: wins[j] += 1
    return max(range(n), key=lambda k: wins[k])


if __name__ == "__main__":
    d = 40
    Q = 5000
    alpha_lins = [1.0, 0.8, 0.5, 0.2, 0.0]
    alpha_ints = [1.0, 0.5, 0.0]

    print(f"\nLinear-decorrelation grid (d={d}, Q={Q}, m={{2.0,2.3,2.7,3.2}}, best=rho_0)")
    print("Each cell: var-reduction (CRN vs independent) / P(select best)\n")
    header = "alpha_lin \\ alpha_int |" + "".join(f"{ai:>18.1f}" for ai in alpha_ints)
    print(header); print("-" * len(header))

    for al in alpha_lins:
        cells = []
        for ai in alpha_ints:
            # selection accuracy
            hits, trials = 0, 150
            for t in range(trials):
                fam = make_family(d, al, ai, seed=10)
                hits += (tournament_pick(fam, d, Q, 300 + t * 29) == 0)
            psel = hits / trials
            # variance reduction rho_0 vs rho_1
            fam = make_family(d, al, ai, seed=10)
            crn, ind = [], []
            for t in range(150):
                crn.append(contrast(fam[0], fam[1], d, Q, 13000 + t))
                ind.append(m_independent(*fam[0][:3], d, Q, 14000 + t)
                           - m_independent(*fam[1][:3], d, Q, 15000 + t))
            crn = torch.tensor(crn); ind = torch.tensor(ind)
            vr = (ind.std() / crn.std()) ** 2 if crn.std() > 0 else float('inf')
            cells.append(f"{vr:>8.0f}x/{psel:.2f}")
        print(f"{al:>20.1f} |" + "".join(f"{c:>18}" for c in cells))

    print("\nKey row: alpha_lin=0 (linear block fully decorrelated) is the true worst case.")
    print("If var-reduction there stays >> 2x and P(select)~1, the method survives the")
    print("realistic regime where references differ in BOTH linear and interaction structure.")