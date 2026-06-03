"""
RePro synthetic verification (NumPy only, no Torch, no model).

Plants a function g(z) = sum_{|S|<=2} beta_S chi_S(z) with KNOWN residual energy m,
and checks:
  (a) the mixed-difference stencil estimates m unbiasedly (recovers calibration const),
  (b) m_hat concentrates as Q grows (Bernstein rate),
  (c) RePro selects the smallest-m reference at Q ~ log(M/delta)/gap^2.
"""

import numpy as np

rng = np.random.default_rng(0)


def make_planted(d, lin_scale, quad_pairs, quad_scale, seed):
    """Return (beta_lin[d], dict{(i,j):coef}) with planted linear + pairwise terms."""
    r = np.random.default_rng(seed)
    beta_lin = r.normal(0, lin_scale, size=d)
    quad = {}
    idx = r.choice(d, size=(quad_pairs, 2))
    for a, b in idx:
        if a == b:
            continue
        quad[(min(a, b), max(a, b))] = r.normal(0, quad_scale)
    return beta_lin, quad


def g_eval(Z, beta_lin, quad):
    """g(z) in centered Walsh basis. Z is {0,1}^{n x d}. chi_i = 2*zbar_i."""
    zbar = Z - 0.5
    chi = 2.0 * zbar                       # chi_{i}
    out = chi @ beta_lin
    for (i, j), c in quad.items():
        out = out + c * chi[:, i] * chi[:, j]
    return out


def true_m(quad):
    return sum(c * c for c in quad.values())


def true_s(beta_lin):
    return float(np.sum(beta_lin ** 2))


def stencil_estimate_m(beta_lin, quad, d, Q, seed):
    """Mixed second difference over random (i,j,z). Returns normalized m_hat."""
    r = np.random.default_rng(seed)
    vals = np.empty(Q)
    for q in range(Q):
        i, j = r.choice(d, size=2, replace=False)
        z = r.integers(0, 2, size=d).astype(float)
        def forced(zi, zj):
            zz = z.copy(); zz[i] = zi; zz[j] = zj
            return g_eval(zz[None, :], beta_lin, quad)[0]
        delta = 0.25 * (forced(1, 1) - forced(1, 0) - forced(0, 1) + forced(0, 0))
        vals[q] = delta * delta
    # calibration: E[Delta^2] = c_d * m ; for a single planted pair the stencil
    # returns exactly (c/4)^2 contribution per matching pair -> empirical const below.
    return vals


if __name__ == "__main__":
    d = 40

    # ---- (a) unbiasedness / calibration on a single planted pair ----
    beta_lin = np.zeros(d)
    quad = {(3, 7): 1.5}
    m = true_m(quad)
    Q = 40000
    vals = stencil_estimate_m(beta_lin, quad, d, Q, seed=1)
    ehat = vals.mean()
    c_d = ehat / m
    print(f"[a] planted m={m:.4f}  E[Delta^2]={ehat:.6f}  =>  c_d={c_d:.6f}  (expect const ~ 1/binom(d,2)*scale)")

    # ---- (b) concentration as Q grows ----
    print("\n[b] concentration of m_hat (using calibration c_d):")
    for Q in [200, 1000, 5000, 20000]:
        reps = 200
        errs = []
        for rep in range(reps):
            v = stencil_estimate_m(beta_lin, quad, d, Q, seed=100 + rep)
            mhat = v.mean() / c_d
            errs.append(abs(mhat - m))
        print(f"    Q={Q:6d}   mean|m_hat - m|={np.mean(errs):.4f}   std={np.std(errs):.4f}")

    # ---- (c) selection: 4 references with different planted m ----
    print("\n[c] reference selection (pick smallest m):")
    refs = {}
    for k, qs in enumerate([0.3, 0.6, 1.0, 1.5]):   # increasing residual energy
        bl, qd = make_planted(d, lin_scale=1.0, quad_pairs=8, quad_scale=qs, seed=10 + k)
        refs[f"rho_{k}"] = (bl, qd, true_m(qd))
    true_best = min(refs, key=lambda r: refs[r][2])
    print("    true m per ref:", {r: round(refs[r][2], 3) for r in refs})
    print("    true best:", true_best)

    for Q in [200, 1000, 5000]:
        hits = 0
        trials = 100
        for t in range(trials):
            scores = {}
            for r, (bl, qd, _) in refs.items():
                v = stencil_estimate_m(bl, qd, d, Q, seed=1000 + t * 10 + hash(r) % 7)
                scores[r] = v.mean()      # ranking is scale-free; pick smallest
            pick = min(scores, key=lambda rr: scores[rr])
            hits += (pick == true_best)
        print(f"    Q={Q:5d}   P(correct select)={hits/trials:.2f}")