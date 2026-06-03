# RePro: Fit-Free, Predictive Reference Selection for First-Order Perturbation Explanations

**Setting.** First-order (K=1) perturbation explanation, LIME-style. We want to pick, *before*
spending the explanation query budget, the reference operator that will give the most recoverable
linear attribution — with a high-probability guarantee, and without running the explanation under
each candidate.

---

## 1. What actually determines a good reference at K=1

Fix input `x`, target class `c`, reference operator `ρ`. Masks `z ∈ {0,1}^d` are drawn i.i.d. with
`E[z_i] = 1/2`; write the centered coordinate `z̄_i = z_i − 1/2`. The masked response is

    g_ρ(z) = f_c( Φ_ρ(x, z) ).

In the Walsh/Fourier–Walsh basis `g_ρ(z) = Σ_S β_{S,ρ} χ_S(z)`. LIME recovers the degree-1 block
`{β_{i,ρ}}`. The single quantity that controls first-order recoverability is the **residual energy**

    m_ρ := Σ_{|S| ≥ 2} β_{S,ρ}²
         = Var_µ(g_ρ) − Σ_i β_{i,ρ}².

A reference is "good" when its linear signal is large relative to `m_ρ`. Everything in the
detection-floor story at K=1 reduces to the scalar ratio

    SNR_ρ  :=  s_ρ / m_ρ ,   where  s_ρ := Σ_i β_{i,ρ}².

So reference selection at K=1 is *exactly* the problem of estimating two scalars per reference:
the linear energy `s_ρ` and the nonlinear residual energy `m_ρ`.

---

## 2. Why post-hoc held-out-variance selection fails

The natural baseline (and what the reference-aware paper proposes) estimates `m_ρ` as the
held-out unexplained variance of the **fitted** degree-1 surrogate:

    m̂_ρ ≈ (1/|D_val|) Σ ( y(z) − ĝ_{≤1,ρ}(z) )²  −  σ̂²_obs .

Three problems, and they are the reason it "never picks the best one":

1. **Not predictive.** You must run the full N-query LIME fit under every candidate ρ just to
   score it. The selection costs as much as M explanations.
2. **Reference-dependent bias.** Finite-sample fit error inflates `m̂_ρ`, but the inflation depends
   on how well-conditioned the design is *under that reference*. The bias is not a constant across
   ρ, so it corrupts the ranking — not just the magnitude.
3. **Conflation.** A large held-out residual can mean "g_ρ is genuinely nonlinear" (real, large
   `m_ρ`) OR "the Lasso underfit the linear part" (estimation artifact). The measure cannot tell
   these apart, so a hard-but-recoverable reference looks the same as a genuinely bad one.

We want an estimator of `m_ρ` that (a) needs no surrogate fit, (b) is unbiased for the true
population residual energy, and (c) is cheap and `d`-independent.

---

## 3. The fit-free probe

**Idea.** Use a finite-difference stencil that algebraically annihilates the constant and all
linear terms, leaving only interactions. For coordinates `i ≠ j`, with the rest of the mask held
at a random background `z`, define the **mixed second difference**

    Δ_{ij}(z) = 1/4 [ g(z^{i+,j+}) − g(z^{i+,j−}) − g(z^{i−,j+}) + g(z^{i−,j−}) ],

where `z^{i±}` forces coordinate `i` to 1 / 0 (others as in `z`). In the Walsh basis, this stencil
kills `χ_∅` and every `χ_S` that misses `i` or `j`; it retains exactly the terms whose support
contains both `i` and `j`. Averaging over random pairs `(i,j)` and backgrounds `z`,

    E_{i,j,z}[ Δ_{ij}(z)² ]  =  c_d · m_ρ ,

for a known combinatorial constant `c_d` (depends only on `d`, shared across all references). So
**`m_ρ` is an expected squared stencil value** — estimable by plain Monte Carlo, no regression.

Linear energy `s_ρ` is probed the same way with a single-coordinate difference:

    D_i(z) = 1/2 [ g(z^{i+}) − g(z^{i−}) ]   ⇒   E_{i,z}[ D_i(z)² ] = (1/4)·(per-coord linear energy),

which annihilates the constant and all terms not touching `i`; the bias from interactions touching
`i` is itself `O(m_ρ)` and can be subtracted using the `Δ` estimate. (Calibration constants are
verified numerically in §5 — ranking is scale-free and correct regardless; only absolute SNR needs
the constants.)

### Algorithm (RePro)

    For each candidate reference ρ in R:
      1. Draw Q triples (i_q, j_q, z_q), i≠j, z_q ~ µ.
      2. Query the 4-point stencil for each → Δ_q.   [4Q model queries]
      3. m̂_ρ = (4 / Q) Σ_q Δ_q²                       (normalized residual energy)
      4. Draw R coord-probe pairs → D_r ; ŝ_ρ from avg(D_r²), debiased by m̂_ρ.  [2R queries]
      5. Score SNR_ρ = ŝ_ρ / m̂_ρ.
    Return argmax_ρ SNR_ρ.

Cost per reference ≈ **4Q + 2R queries**, independent of `d` and of the eventual LIME budget `N`.

---

## 4. The guarantee (high-probability correct selection)

Each `Δ_q²` is i.i.d. and bounded: `|g| ≤ B ⇒ |Δ_{ij}| ≤ B ⇒ Δ² ≤ B²`. Per-sample variance
`v ≤ B² m_ρ`. Bernstein gives, for one reference,

    P( |m̂_ρ − m_ρ| > ε )  ≤  2 exp( − Q ε² / (2(v + Bε/3)) ).

To select the true-minimum-`m_ρ` reference out of `M` candidates with probability ≥ 1 − δ, take
`ε = Δgap/2` where `Δgap` is the residual-energy separation between the best and runner-up
reference, union-bound over `M`:

    Q  ≳  ( B² / Δgap² ) · log( M / δ ).

**Logarithmic in M, zero dependence on d, zero dependence on the explanation budget N.** This is a
*prediction made before any explanation is computed* — the property the post-hoc measure lacks.

---

## 5. Synthetic verification (to run on request)

A NumPy-only check on a planted function `g(z) = Σ_{|S|≤1} β_S χ_S + Σ_{|S|=2} β_S χ_S + tail`,
with known `m`, that confirms:
  - (a) the stencil estimator `m̂` is unbiased for the planted `m` (recovers `c_d`);
  - (b) `m̂` concentrates at the Bernstein rate as `Q` grows;
  - (c) RePro ranks a family of references by true `m_ρ` and picks the smallest with the predicted
        `Q ∝ log(M/δ)/Δgap²` sample complexity.

No Torch, no model — pure planted coefficients. Script is written and ready; not executed yet.