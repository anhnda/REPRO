"""
Reference-aware first-order (K=1) perturbation explanation.

This is the §4 instantiation of "A Reference-Aware Theory of Perturbation
Explanations". It is deliberately faithful to Theorem 1 rather than to the
historical LIME implementation:

  * masks are drawn i.i.d. from the product distribution mu with E[z_i] = 1/2
    (uniform Bernoulli-1/2), NO distance kernel reweighting;
  * the design uses the centered orthonormal basis  chi_i(z) = 2*(z_i - 1/2),
    so the fitted coefficients ARE the multilinear (Fourier-Walsh) beta_{i,rho};
  * the surrogate is fit by Lasso (signed-support recovery, the estimator the
    detection floor is proven for);
  * the higher-order residual energy m_{>1,rho} is estimated as held-out
    unexplained variance (the §6 proxy) -- never by fitting higher-order terms.

The reference operator rho is pluggable via a MaskLibrary, so the same code runs
black / white / mean / blur / inpaint references and supports the §6 selection
criterion.

Library use:
    expl = RefLIME(model, device="cuda")
    out  = expl.explain(x, references={"blur": blur_ref, "black": black_ref})
    # out.best_reference, out.per_reference[name].{attr, floor, m_hat, snr}

Author-agnostic: `model` is any callable mapping (B,C,H,W) -> (B,n_classes) logits.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from _core import centered_design, lasso_fit  # shared, sklearn-backed

# Standard ImageNet normalization stats (used to map pixel-space constants such
# as white=1.0 into the normalized space the operators actually run in).
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# --------------------------------------------------------------------------- #
#  Reference operators (rho).  Each maps (x, keep_mask) -> completed image.
#  keep_mask is (B,1,H,W) in {0,1}: 1 = keep original pixel, 0 = replace by rho.
#  Deterministic references give sigma_obs = 0 (Corollary 1 regime).
# --------------------------------------------------------------------------- #
ReferenceOp = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def black_reference() -> ReferenceOp:
    def rho(x, keep):
        return keep * x  # replace with 0
    rho.is_stochastic = False  # type: ignore[attr-defined]
    return rho


def constant_reference(pixel_value: float = 1.0,
                       mean: Sequence[float] = IMAGENET_MEAN,
                       std: Sequence[float] = IMAGENET_STD) -> ReferenceOp:
    """Fill masked region with a constant PIXEL-space value (0..1), mapped into
    normalized space per channel:  fill_c = (pixel_value - mean_c) / std_c.

    pixel_value=1.0 -> white;  0.0 -> black-in-pixel-space (NOT the same as the
    black_reference above, which fills normalized 0).  This is what makes
    'white' a genuinely distinct operator from black: after ImageNet norm the
    channel means are ~0, so a normalized-0 fill coincides with black, whereas a
    pixel-space constant maps to a nonzero, per-channel-distinct value.
    """
    m = torch.tensor(mean, dtype=torch.float32).view(1, -1, 1, 1)
    s = torch.tensor(std, dtype=torch.float32).view(1, -1, 1, 1)
    fill_norm = (pixel_value - m) / s          # (1,C,1,1) in normalized space

    def rho(x, keep):
        f = fill_norm.to(device=x.device, dtype=x.dtype)
        return keep * x + (1 - keep) * f
    rho.is_stochastic = False  # type: ignore[attr-defined]
    return rho


def white_reference(mean: Sequence[float] = IMAGENET_MEAN,
                    std: Sequence[float] = IMAGENET_STD) -> ReferenceOp:
    """White fill: pixel value 1.0 mapped through ImageNet normalization."""
    return constant_reference(1.0, mean, std)


def mean_reference() -> ReferenceOp:
    def rho(x, keep):
        fill = x.mean(dim=(2, 3), keepdim=True)  # per-image, per-channel mean
        return keep * x + (1 - keep) * fill
    rho.is_stochastic = False  # type: ignore[attr-defined]
    return rho


def _gaussian_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    radius = max(1, int(3 * sigma))
    k = torch.arange(-radius, radius + 1, device=x.device, dtype=x.dtype)
    g = torch.exp(-(k ** 2) / (2 * sigma * sigma))
    g = (g / g.sum()).view(1, 1, -1)
    C = x.shape[1]
    gh = g.view(1, 1, -1, 1).repeat(C, 1, 1, 1)
    gw = g.view(1, 1, 1, -1).repeat(C, 1, 1, 1)
    x = F.conv2d(x, gh, padding=(radius, 0), groups=C)
    x = F.conv2d(x, gw, padding=(0, radius), groups=C)
    return x


def blur_reference(sigma: float = 11.0) -> ReferenceOp:
    def rho(x, keep):
        b = _gaussian_blur(x, sigma)
        return keep * x + (1 - keep) * b
    rho.is_stochastic = False  # type: ignore[attr-defined]
    return rho


def noisy_inpaint_reference(sigma_blur: float = 11.0, noise: float = 0.15) -> ReferenceOp:
    """A cheap *stochastic* on-manifold-ish reference: blurred fill + Gaussian
    texture noise. Stands in for sampled inpainting/LM-infill. sigma_obs > 0,
    so it exercises the bias-variance tradeoff of Remark 2."""
    def rho(x, keep):
        b = _gaussian_blur(x, sigma_blur)            # (1,C,H,W)
        b = b.expand(keep.shape[0], -1, -1, -1)      # (B,C,H,W), no copy
        b = b + noise * torch.randn_like(b)          # independent noise per row
        return keep * x + (1 - keep) * b
    rho.is_stochastic = True  # type: ignore[attr-defined]
    return rho


def default_reference_family() -> Dict[str, ReferenceOp]:
    return {
        "black": black_reference(),
        "white": white_reference(),
        "mean": mean_reference(),
        "blur": blur_reference(11.0),
        "inpaint": noisy_inpaint_reference(11.0, 0.15),
    }


# --------------------------------------------------------------------------- #
#  Mask library: grid cells -> per-pixel keep masks, plus i.i.d. mu sampling.
# --------------------------------------------------------------------------- #
class MaskLibrary:
    """Interpretable units = grid cells. Draws i.i.d. masks z ~ mu (Bernoulli-1/2,
    product) and expands each cell-vector to a per-pixel keep mask."""

    def __init__(self, H: int, W: int, grid: Tuple[int, int] = (12, 12),
                 device: str = "cpu", seed: int = 0):
        self.H, self.W = H, W
        self.grid = grid
        self.n_cells = grid[0] * grid[1]
        self.device = device
        self._gen = torch.Generator(device="cpu").manual_seed(seed)
        gh, gw = grid
        ys = (torch.arange(H) * gh // H).clamp(max=gh - 1)
        xs = (torch.arange(W) * gw // W).clamp(max=gw - 1)
        self.cell_ids = (ys.view(-1, 1) * gw + xs.view(1, -1)).to(device)  # (H,W)

    def sample(self, n: int, include_all_on: bool = False) -> torch.Tensor:
        """Return Z in {0,1} of shape (n, n_cells), drawn i.i.d. from mu."""
        Z = (torch.rand(n, self.n_cells, generator=self._gen) > 0.5).float()
        if include_all_on:
            Z[0] = 1.0
        return Z

    def to_pixel_keep(self, Z: torch.Tensor) -> torch.Tensor:
        """(B,n_cells) {0,1} cell vectors -> (B,1,H,W) per-pixel keep mask."""
        Z = Z.to(self.device)
        keep = Z[:, self.cell_ids]          # (B,H,W)
        return keep.unsqueeze(1)            # (B,1,H,W)






# --------------------------------------------------------------------------- #
#  Result containers
# --------------------------------------------------------------------------- #
@dataclass
class PerReference:
    name: str
    attr: np.ndarray            # (H,W) painted coefficient map
    beta: np.ndarray            # (d,) first-order coefficients
    intercept: float
    m_hat: float                # held-out residual energy m_{>1,rho}
    sigma_obs: float            # query-noise scale (0 for deterministic rho)
    floor: float                # detection floor (sigma_obs + c sqrt(m_hat)) sqrt(log d / N)
    beta_min: float             # smallest *active* |coef| above floor
    snr: float                  # gamma = beta_min / (sigma_obs + c sqrt(m_hat))
    n_active: int


@dataclass
class RefLIMEResult:
    target: int
    per_reference: Dict[str, PerReference] = field(default_factory=dict)
    best_reference: Optional[str] = None


# --------------------------------------------------------------------------- #
#  The explainer
# --------------------------------------------------------------------------- #
class RefLIME:
    """Reference-aware K=1 perturbation explainer.

    model : callable (B,C,H,W) -> (B,n_classes) logits
    """

    def __init__(self, model, device: str = "cpu", grid=(12, 12),
                 n_samples: int = 2000, val_frac: float = 0.3,
                 c: float = 1.0, batch_size: int = 64, seed: int = 0,
                 sigma_repeat: int = 8):
        self.model = model
        self.device = device
        self.grid = grid
        self.n_samples = n_samples
        self.val_frac = val_frac
        self.c = c
        self.batch_size = batch_size
        self.seed = seed
        self.sigma_repeat = sigma_repeat  # repeats for sigma_obs estimation

    # ---- query the black box for a batch of masks under one reference -------
    @torch.no_grad()
    def _query(self, x, rho, Z, target):
        lib = self._lib
        out = np.zeros(Z.shape[0], dtype=np.float64)
        for s in range(0, Z.shape[0], self.batch_size):
            zb = Z[s:s + self.batch_size]
            keep = lib.to_pixel_keep(zb)               # (B,1,H,W)
            comp = rho(x, keep)                         # apply reference
            logits = self.model(comp.to(self.device))
            p = F.softmax(logits, dim=1)[:, target]
            out[s:s + zb.shape[0]] = p.detach().cpu().numpy()
        return out
    @torch.no_grad()
    def _query_logit(self, x, rho, Z, target):
        """Same as _query but returns target-class LOGIT (pre-softmax).
        Used only for the saturation diagnostic; the floor still uses
        probability-space sigma_obs to match the §3.1 observation model."""
        lib = self._lib
        out = np.zeros(Z.shape[0], dtype=np.float64)
        for s in range(0, Z.shape[0], self.batch_size):
            zb = Z[s:s + self.batch_size]
            keep = lib.to_pixel_keep(zb)
            comp = rho(x, keep)
            logits = self.model(comp.to(self.device))
            lg = logits[:, target]
            out[s:s + zb.shape[0]] = lg.detach().cpu().numpy()
        return out
    # ---- estimate sigma_obs for a stochastic reference ----------------------
    @torch.no_grad()
    def _estimate_sigma_obs(self, x, rho, target) -> float:
        if not getattr(rho, "is_stochastic", False):
            return 0.0
        lib = self._lib
        Zr = lib.sample(min(64, self.n_samples))        # a few fixed masks
        per_mask_std = []
        per_mask_std_logit = []
        for i in range(Zr.shape[0]):
            zi = Zr[i:i + 1].repeat(self.sigma_repeat, 1)
            vals = self._query(x, rho, zi, target)
            vals_lg = self._query_logit(x, rho, zi, target)
            per_mask_std.append(vals.std())
            per_mask_std_logit.append(vals_lg.std())
        sigma_prob = float(np.mean(per_mask_std))
        sigma_logit = float(np.mean(per_mask_std_logit))
        # If logit-std > 0 but prob-std ~ 0, the reference IS stochastic and
        # sigma_obs=0 is saturation (Corollary 1 regime), not a dead noise path.
        if sigma_logit > 1e-6 and sigma_prob < 1e-6:
            regime = "saturated (prob flat, logit varies)"
        elif sigma_logit < 1e-6:
            regime = "NO noise reaching output (check rho)"
        else:
            regime = "active"
        print(f"[sigma] prob_std={sigma_prob:.6f} logit_std={sigma_logit:.6f}"
              f"  -> {regime}")
        return sigma_prob
    # ---- explain under a single reference -----------------------------------
    def _explain_one(self, x, name, rho, target) -> PerReference:
        lib = self._lib
        d = lib.n_cells
        N = self.n_samples
        Z = lib.sample(N)
        y = self._query(x, rho, Z, target)

        n_val = int(self.val_frac * N)
        n_tr = N - n_val
        Ztr, Zval = Z[:n_tr].numpy(), Z[n_tr:].numpy()
        ytr, yval = y[:n_tr], y[n_tr:]

        Xtr = centered_design(Ztr)
        Xval = centered_design(Zval)

        sigma_obs = self._estimate_sigma_obs(x, rho, target)

        # We need m_hat to set lambda, but lambda needs m_hat -> one cheap pass:
        # fit with a pilot lambda, get residual, then set principled lambda.
        log_d = math.log(d + 1)
        pilot_lam = 0.5 * np.std(ytr) * math.sqrt(log_d / n_tr)
        beta_p, b0_p = lasso_fit(Xtr, ytr, pilot_lam)
        resid_val = yval - (Xval @ beta_p + b0_p)
        m_hat = max(float(np.mean(resid_val ** 2) - sigma_obs ** 2), 0.0)

        # principled lambda  ~ (sigma_obs + c sqrt(m_hat)) sqrt(log d / N)
        lam = (sigma_obs + self.c * math.sqrt(m_hat)) * math.sqrt(log_d / n_tr)
        lam = max(lam, 1e-6)
        beta, b0 = lasso_fit(Xtr, ytr, lam)

        floor = (sigma_obs + self.c * math.sqrt(m_hat)) * math.sqrt(log_d / N)
        active = np.abs(beta) > floor
        n_active = int(active.sum())
        beta_min = float(np.abs(beta[active]).min()) if n_active else 0.0
        denom = sigma_obs + self.c * math.sqrt(m_hat) + 1e-12
        snr = beta_min / denom

        # paint coefficients back to pixels
        coef_t = torch.tensor(beta, dtype=torch.float32, device=self.device)
        attr = coef_t[lib.cell_ids].cpu().numpy()

        return PerReference(name=name, attr=attr, beta=beta, intercept=b0,
                            m_hat=m_hat, sigma_obs=sigma_obs, floor=floor,
                            beta_min=beta_min, snr=snr, n_active=n_active)

    # ---- public API ----------------------------------------------------------
    @torch.no_grad()
    def explain(self, x: torch.Tensor, target: Optional[int] = None,
                references: Optional[Dict[str, ReferenceOp]] = None
                ) -> RefLIMEResult:
        x = x.to(self.device)
        assert x.dim() == 4 and x.shape[0] == 1, "pass a single image (1,C,H,W)"
        _, _, H, W = x.shape
        self._lib = MaskLibrary(H, W, self.grid, device=self.device, seed=self.seed)

        if target is None:
            target = int(self.model(x).argmax(dim=1).item())
        if references is None:
            references = default_reference_family()

        result = RefLIMEResult(target=target)
        for name, rho in references.items():
            result.per_reference[name] = self._explain_one(x, name, rho, target)

        # §6 selection: pick max explanation SNR (gamma)
        result.best_reference = max(
            result.per_reference.values(), key=lambda r: r.snr
        ).name
        return result