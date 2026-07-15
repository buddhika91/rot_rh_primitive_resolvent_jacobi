#!/usr/bin/env python3
"""
rot_rh_primitive_resolvent_gue_audit.py

GUE audit for the canonical primitive Xi-resolvent -> Stieltjes/Jacobi operator.

Pipeline
--------
    Xi
      -> Phi(u) = Xi(1/2 + i sqrt(u)) / Xi(1/2)
      -> R(u) = -Phi'(u)/Phi(u)
      -> Stieltjes moments r_n
      -> canonical Jacobi coefficients (alpha_n, beta_n)
      -> finite Jacobi matrix J_d
      -> gamma_j = 1/sqrt(lambda_j(J_d))
      -> unfolding and local spectral statistics.

No operator coefficients are fitted.

Statistics
----------
For each Jacobi depth:
- positive-node coverage
- zero reconstruction RMSE
- unfolded nearest-neighbour spacings
- KS distance to:
    * GUE Wigner surmise
    * GOE Wigner surmise
    * Poisson
- mean consecutive-spacing ratio
- KS distance for ratio statistics to:
    * GUE
    * GOE
    * Poisson
- number variance Sigma^2(L)
- Dyson-Mehta Delta_3(L)
- bulk and edge spacing summaries
- spectral rigidity score

Controls
--------
- signal
- signflip
- permuted
- gaussian
- phase_scramble

Important limitation
--------------------
The finite Jacobi operator is reconstructed from Xi itself. Therefore this
tests whether the canonical resolvent realization reproduces GUE-like local
statistics; it is not an independent proof of RH or an independent prediction
of the zeros.

Dependencies
------------
numpy, scipy, mpmath
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import mpmath as mp
import numpy as np

try:
    from scipy.linalg import eigh
    from scipy.stats import kstest
except ImportError as exc:
    raise SystemExit("Install dependencies with: pip install numpy scipy mpmath") from exc


EPS = 1e-300


# =============================================================================
# Utilities
# =============================================================================

def parse_int_list(text: str) -> List[int]:
    vals = [int(x.strip()) for x in text.split(",") if x.strip()]
    if not vals:
        raise argparse.ArgumentTypeError("Expected comma-separated integers")
    return vals


def save_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fields: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def trim_series(p: Sequence[mp.mpc], n: int) -> List[mp.mpc]:
    return list(p[:n]) + [mp.mpc(0)] * max(0, n - len(p))


def series_derivative(p: Sequence[mp.mpc]) -> List[mp.mpc]:
    if len(p) <= 1:
        return [mp.mpc(0)]
    return [(k + 1) * p[k + 1] for k in range(len(p) - 1)]


def series_inverse(p: Sequence[mp.mpc], n: int) -> List[mp.mpc]:
    p = trim_series(p, n)
    if abs(p[0]) == 0:
        raise ZeroDivisionError("Series inverse requires nonzero constant term.")

    q = [mp.mpc(0)] * n
    q[0] = 1 / p[0]

    for k in range(1, n):
        total = mp.mpc(0)
        for j in range(1, k + 1):
            total += p[j] * q[k - j]
        q[k] = -total / p[0]

    return q


def series_mul(
    a: Sequence[mp.mpc],
    b: Sequence[mp.mpc],
    n: int,
) -> List[mp.mpc]:
    a = trim_series(a, n)
    b = trim_series(b, n)
    out = [mp.mpc(0)] * n

    for k in range(n):
        total = mp.mpc(0)
        for j in range(k + 1):
            total += a[j] * b[k - j]
        out[k] = total

    return out


# =============================================================================
# Xi and primitive resolvent moments
# =============================================================================

def xi(s: mp.mpc) -> mp.mpc:
    return (
        mp.mpf("0.5")
        * s
        * (s - 1)
        * mp.power(mp.pi, -s / 2)
        * mp.gamma(s / 2)
        * mp.zeta(s)
    )


def xi_taylor_cauchy(
    max_order: int,
    radius: mp.mpf,
    samples: int,
    phase_scramble: bool = False,
    seed: int = 0,
) -> List[mp.mpc]:
    rng = random.Random(seed)
    s0 = mp.mpf("0.5")

    omegas: List[mp.mpc] = []
    values: List[mp.mpc] = []

    for j in range(samples):
        angle = 2 * mp.pi * j / samples
        omega = mp.e ** (1j * angle)
        value = xi(s0 + radius * omega)

        if phase_scramble:
            value *= mp.e ** (1j * rng.uniform(-math.pi, math.pi))

        omegas.append(omega)
        values.append(value)

    coeffs: List[mp.mpc] = []

    for n in range(max_order + 1):
        total = mp.mpc(0)
        for value, omega in zip(values, omegas):
            total += value * omega ** (-n)
        coeffs.append(total / samples / radius ** n)

    return coeffs


def phi_from_xi_coeffs(
    coeffs: Sequence[mp.mpc],
    order_u: int,
) -> List[mp.mpc]:
    a0 = coeffs[0]
    return [
        coeffs[2 * j] * ((-1) ** j) / a0
        for j in range(order_u + 1)
    ]


def primitive_resolvent(
    phi: Sequence[mp.mpc],
    n: int,
) -> List[mp.mpc]:
    dphi = series_derivative(phi)
    inv_phi = series_inverse(phi, n)
    return [-x for x in series_mul(dphi, inv_phi, n)]


# =============================================================================
# Canonical Stieltjes/Jacobi recursion
# =============================================================================

def poly_trim(p: Sequence[mp.mpf], n: int) -> List[mp.mpf]:
    return list(p[:n]) + [mp.mpf("0")] * max(0, n - len(p))


def poly_add(
    a: Sequence[mp.mpf],
    b: Sequence[mp.mpf],
    ca: mp.mpf = mp.mpf("1"),
    cb: mp.mpf = mp.mpf("1"),
) -> List[mp.mpf]:
    n = max(len(a), len(b))
    aa = poly_trim(a, n)
    bb = poly_trim(b, n)
    return [ca * aa[i] + cb * bb[i] for i in range(n)]


def poly_scale(p: Sequence[mp.mpf], c: mp.mpf) -> List[mp.mpf]:
    return [c * x for x in p]


def poly_x(p: Sequence[mp.mpf]) -> List[mp.mpf]:
    return [mp.mpf("0")] + list(p)


def moment_inner(
    p: Sequence[mp.mpf],
    q: Sequence[mp.mpf],
    moments: Sequence[mp.mpf],
) -> mp.mpf:
    total = mp.mpf("0")
    for i, pi in enumerate(p):
        for j, qj in enumerate(q):
            total += pi * qj * moments[i + j]
    return total


@dataclass
class JacobiResult:
    alpha: List[mp.mpf]
    beta: List[mp.mpf]
    orthogonality_defect: mp.mpf
    breakdown_index: int


def stieltjes_jacobi(
    moments: Sequence[mp.mpf],
    depth: int,
    tol: mp.mpf,
) -> JacobiResult:
    if len(moments) < 2 * depth + 2:
        raise ValueError("Need at least 2*depth+2 moments.")
    if moments[0] <= tol:
        raise ValueError("m0 is not positive.")

    p_prev = [mp.mpf("0")]
    p = [1 / mp.sqrt(moments[0])]
    polys = [p]

    alpha: List[mp.mpf] = []
    beta: List[mp.mpf] = []
    beta_prev = mp.mpf("0")
    breakdown = -1

    for n in range(depth):
        xp = poly_x(p)
        a = moment_inner(xp, p, moments)
        alpha.append(a)

        residual = poly_add(xp, p, mp.mpf("1"), -a)

        if n > 0:
            residual = poly_add(
                residual,
                p_prev,
                mp.mpf("1"),
                -beta_prev,
            )

        # Full reorthogonalization.
        for basis in polys:
            coeff = moment_inner(residual, basis, moments)
            residual = poly_add(
                residual,
                basis,
                mp.mpf("1"),
                -coeff,
            )

        if n == depth - 1:
            break

        norm2 = moment_inner(residual, residual, moments)

        if norm2 <= tol:
            breakdown = n
            break

        b = mp.sqrt(norm2)
        beta.append(b)

        p_prev = p
        p = poly_scale(residual, 1 / b)
        beta_prev = b
        polys.append(p)

    orth_defect = mp.mpf("0")

    for i, pi in enumerate(polys):
        for j, pj in enumerate(polys):
            target = mp.mpf("1") if i == j else mp.mpf("0")
            orth_defect = max(
                orth_defect,
                abs(moment_inner(pi, pj, moments) - target),
            )

    return JacobiResult(
        alpha=alpha,
        beta=beta,
        orthogonality_defect=orth_defect,
        breakdown_index=breakdown,
    )


def jacobi_matrix(result: JacobiResult) -> np.ndarray:
    alpha = np.array([float(x) for x in result.alpha], dtype=float)
    beta = np.array([float(x) for x in result.beta], dtype=float)

    matrix = np.diag(alpha)

    if len(beta):
        matrix += np.diag(beta, 1)
        matrix += np.diag(beta, -1)

    return 0.5 * (matrix + matrix.T)


def predict_gamma(result: JacobiResult) -> np.ndarray:
    nodes = np.real(
        eigh(
            jacobi_matrix(result),
            eigvals_only=True,
            check_finite=False,
            driver="evr",
        )
    )
    nodes = nodes[np.isfinite(nodes) & (nodes > 0)]

    # Largest x = 1/gamma^2 corresponds to smallest gamma.
    nodes = np.sort(nodes)[::-1]
    return 1.0 / np.sqrt(nodes)


# =============================================================================
# Controls
# =============================================================================

def apply_control(
    moments: Sequence[mp.mpf],
    control: str,
    seed: int,
) -> List[mp.mpf]:
    rng = random.Random(seed)
    values = [mp.mpf(x) for x in moments]

    if control == "signal":
        return values

    if control == "signflip":
        return [values[0]] + [
            x * (-1 if rng.random() < 0.5 else 1)
            for x in values[1:]
        ]

    if control == "permuted":
        tail = values[1:]
        rng.shuffle(tail)
        return [values[0]] + tail

    if control == "gaussian":
        logs = [
            float(mp.log10(max(abs(x), mp.mpf("1e-300"))))
            for x in values[1:]
        ]
        mu = float(np.mean(logs))
        sigma = max(float(np.std(logs)), 0.2)

        return [values[0]] + [
            mp.power(10, rng.gauss(mu, sigma))
            for _ in values[1:]
        ]

    raise ValueError(control)


# =============================================================================
# Unfolding
# =============================================================================

def rvm_count(t: np.ndarray) -> np.ndarray:
    t = np.asarray(t, dtype=float)
    x = np.maximum(t / (2.0 * np.pi), 1e-15)

    return (
        x * np.log(x)
        - x
        + 7.0 / 8.0
    )


def unfold_spectrum(
    gamma: np.ndarray,
    method: str,
    poly_degree: int,
) -> np.ndarray:
    gamma = np.sort(np.asarray(gamma, dtype=float))

    if method == "rvm":
        unfolded = rvm_count(gamma)

    elif method == "polynomial":
        indices = np.arange(1, len(gamma) + 1, dtype=float)
        degree = min(poly_degree, max(1, len(gamma) - 2))
        coeff = np.polyfit(gamma, indices, degree)
        unfolded = np.polyval(coeff, gamma)

    else:
        raise ValueError(method)

    # Force increasing order, then renormalize mean spacing.
    unfolded = np.maximum.accumulate(unfolded)
    spacings = np.diff(unfolded)

    if len(spacings) and np.mean(spacings) > 0:
        unfolded = unfolded / np.mean(spacings)

    return unfolded


# =============================================================================
# Theoretical spacing distributions
# =============================================================================

def cdf_poisson(s: np.ndarray) -> np.ndarray:
    s = np.maximum(np.asarray(s, dtype=float), 0.0)
    return 1.0 - np.exp(-s)


def cdf_goe(s: np.ndarray) -> np.ndarray:
    s = np.maximum(np.asarray(s, dtype=float), 0.0)
    return 1.0 - np.exp(-np.pi * s * s / 4.0)


def cdf_gue(s: np.ndarray) -> np.ndarray:
    """
    CDF of GUE Wigner surmise:
        P(s) = (32/pi^2) s^2 exp(-4 s^2/pi).
    """
    s = np.maximum(np.asarray(s, dtype=float), 0.0)
    a = 4.0 / np.pi

    # Integral of c*s^2*exp(-a*s^2), simplified.
    return (
        math.erf(1.0) * 0.0
        + np.vectorize(math.erf)(2.0 * s / math.sqrt(np.pi))
        - (4.0 / np.pi) * s * np.exp(-4.0 * s * s / np.pi)
    )


def spacing_ks(spacings: np.ndarray) -> Dict[str, float]:
    spacings = np.asarray(spacings, dtype=float)
    spacings = spacings[np.isfinite(spacings) & (spacings >= 0)]

    if len(spacings) < 2:
        return {
            "ks_gue": float("nan"),
            "ks_goe": float("nan"),
            "ks_poisson": float("nan"),
        }

    spacings = spacings / max(float(np.mean(spacings)), 1e-15)

    return {
        "ks_gue": float(kstest(spacings, cdf_gue).statistic),
        "ks_goe": float(kstest(spacings, cdf_goe).statistic),
        "ks_poisson": float(kstest(spacings, cdf_poisson).statistic),
    }


# =============================================================================
# Spacing-ratio distributions
# =============================================================================

def spacing_ratios(spacings: np.ndarray) -> np.ndarray:
    spacings = np.asarray(spacings, dtype=float)

    if len(spacings) < 2:
        return np.array([], dtype=float)

    left = spacings[:-1]
    right = spacings[1:]
    denom = np.maximum(left, right)

    mask = denom > 0
    return np.minimum(left[mask], right[mask]) / denom[mask]


def ratio_pdf(r: np.ndarray, beta: int) -> np.ndarray:
    """
    Atas et al. Wigner-like ratio distribution for r=min(s_n,s_{n+1})/
    max(s_n,s_{n+1}) on [0,1].

    Normalization is computed numerically.
    """
    r = np.asarray(r, dtype=float)
    raw = (r + r * r) ** beta / (1.0 + r + r * r) ** (1.0 + 1.5 * beta)

    grid = np.linspace(0.0, 1.0, 20001)
    raw_grid = (
        (grid + grid * grid) ** beta
        / (1.0 + grid + grid * grid) ** (1.0 + 1.5 * beta)
    )
    norm = np.trapezoid(raw_grid, grid)

    return raw / max(norm, 1e-15)


def ratio_cdf_factory(beta: int):
    grid = np.linspace(0.0, 1.0, 50001)
    pdf = ratio_pdf(grid, beta)
    dx = grid[1] - grid[0]
    cdf = np.cumsum(pdf) * dx
    cdf /= max(cdf[-1], 1e-15)

    def cdf_fn(x):
        x = np.asarray(x, dtype=float)
        return np.interp(x, grid, cdf, left=0.0, right=1.0)

    return cdf_fn


RATIO_CDF_GOE = ratio_cdf_factory(1)
RATIO_CDF_GUE = ratio_cdf_factory(2)


def ratio_cdf_poisson(r: np.ndarray) -> np.ndarray:
    r = np.clip(np.asarray(r, dtype=float), 0.0, 1.0)
    return 2.0 * r / (1.0 + r)


def ratio_statistics(ratios: np.ndarray) -> Dict[str, float]:
    ratios = np.asarray(ratios, dtype=float)
    ratios = ratios[np.isfinite(ratios) & (ratios >= 0) & (ratios <= 1)]

    if len(ratios) < 2:
        return {
            "ratio_count": int(len(ratios)),
            "mean_ratio": float("nan"),
            "ratio_ks_gue": float("nan"),
            "ratio_ks_goe": float("nan"),
            "ratio_ks_poisson": float("nan"),
        }

    return {
        "ratio_count": int(len(ratios)),
        "mean_ratio": float(np.mean(ratios)),
        "ratio_ks_gue": float(kstest(ratios, RATIO_CDF_GUE).statistic),
        "ratio_ks_goe": float(kstest(ratios, RATIO_CDF_GOE).statistic),
        "ratio_ks_poisson": float(
            kstest(ratios, ratio_cdf_poisson).statistic
        ),
    }


# =============================================================================
# Long-range statistics
# =============================================================================

def number_variance(
    unfolded: np.ndarray,
    L_values: Sequence[float],
    windows: int,
) -> List[Dict[str, float]]:
    unfolded = np.sort(np.asarray(unfolded, dtype=float))
    rows: List[Dict[str, float]] = []

    if len(unfolded) < 4:
        return rows

    lo = float(unfolded[0])
    hi = float(unfolded[-1])

    for L in L_values:
        if hi - lo <= L:
            continue

        starts = np.linspace(lo, hi - L, windows)
        counts = np.array([
            np.sum((unfolded >= start) & (unfolded < start + L))
            for start in starts
        ], dtype=float)

        rows.append({
            "L": float(L),
            "number_variance": float(np.var(counts)),
            "mean_count": float(np.mean(counts)),
            "windows": int(len(starts)),
        })

    return rows


def delta3_statistic(
    unfolded: np.ndarray,
    L_values: Sequence[float],
    windows: int,
    samples_per_window: int = 128,
) -> List[Dict[str, float]]:
    """
    Numerical Dyson-Mehta Delta_3:
    mean least-squares deviation of staircase N(x) from its best linear fit.
    """
    unfolded = np.sort(np.asarray(unfolded, dtype=float))
    rows: List[Dict[str, float]] = []

    if len(unfolded) < 4:
        return rows

    lo = float(unfolded[0])
    hi = float(unfolded[-1])

    for L in L_values:
        if hi - lo <= L:
            continue

        starts = np.linspace(lo, hi - L, windows)
        values = []

        for start in starts:
            x = np.linspace(start, start + L, samples_per_window)
            staircase = np.searchsorted(unfolded, x, side="right").astype(float)

            A = np.column_stack([x, np.ones_like(x)])
            coeff, *_ = np.linalg.lstsq(A, staircase, rcond=None)
            fit = A @ coeff

            values.append(float(np.mean((staircase - fit) ** 2)))

        rows.append({
            "L": float(L),
            "delta3": float(np.mean(values)),
            "windows": int(len(values)),
        })

    return rows


def theoretical_long_range(L: float) -> Dict[str, float]:
    """
    Leading asymptotic comparisons. These are descriptive guides, not exact
    finite-size formulas.
    """
    L = max(float(L), 1e-12)

    return {
        "poisson_number_variance": L,
        "gue_number_variance_asymptotic": (
            math.log(2.0 * math.pi * L) + 0.5772156649015329 + 1.0
        ) / (math.pi ** 2),
        "poisson_delta3": L / 15.0,
        "gue_delta3_asymptotic": (
            math.log(2.0 * math.pi * L)
            - 1.25
        ) / (2.0 * math.pi ** 2),
    }


# =============================================================================
# Zero reconstruction metrics
# =============================================================================

def zero_reconstruction_metrics(
    predicted: np.ndarray,
    target: np.ndarray,
) -> Dict[str, float]:
    count = min(len(predicted), len(target))

    if count == 0:
        return {
            "zero_count": 0,
            "zero_rmse": float("nan"),
            "zero_relative_rmse": float("nan"),
            "zero_max_abs_error": float("nan"),
        }

    p = predicted[:count]
    t = target[:count]
    error = p - t

    return {
        "zero_count": int(count),
        "zero_rmse": float(np.sqrt(np.mean(error ** 2))),
        "zero_relative_rmse": float(
            np.sqrt(np.mean((error / t) ** 2))
        ),
        "zero_max_abs_error": float(np.max(np.abs(error))),
    }


# =============================================================================
# Main
# =============================================================================

def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()

    p.add_argument("--dps", type=int, default=180)
    p.add_argument("--radius", type=float, default=4.0)
    p.add_argument("--samples", type=int, default=4096)

    p.add_argument(
        "--depths",
        type=parse_int_list,
        default=[16, 20, 24, 28, 32],
    )
    p.add_argument(
        "--zero-reference-count",
        type=int,
        default=32,
    )

    p.add_argument(
        "--unfold",
        choices=["rvm", "polynomial"],
        default="rvm",
    )
    p.add_argument("--poly-degree", type=int, default=5)

    p.add_argument("--trim-fraction", type=float, default=0.10)
    p.add_argument("--bulk-fraction", type=float, default=0.60)

    p.add_argument(
        "--L-values",
        default="1,2,3,4,5",
    )
    p.add_argument("--windows", type=int, default=64)

    p.add_argument(
        "--controls",
        default="signal,signflip,permuted,gaussian",
    )
    p.add_argument("--seed", type=int, default=20260714)

    p.add_argument(
        "--out-prefix",
        default="rot_rh_primitive_resolvent_gue",
    )

    return p


def main() -> int:
    args = parser().parse_args()

    depths = sorted(set(args.depths))
    controls = [x.strip() for x in args.controls.split(",") if x.strip()]
    L_values = [
        float(x.strip())
        for x in args.L_values.split(",")
        if x.strip()
    ]

    if "signal" not in controls:
        raise SystemExit("--controls must include signal")

    max_depth = max(depths)
    moment_count = 2 * max_depth + 8
    phi_order = moment_count + 2
    xi_order = 2 * phi_order + 2

    prefix = Path(args.out_prefix).expanduser().resolve()
    prefix.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 124)
    print("ROT-RH / PRIMITIVE XI-RESOLVENT GUE AUDIT")
    print("=" * 124)
    print(f"dps                  : {args.dps}")
    print(f"radius               : {args.radius}")
    print(f"samples              : {args.samples}")
    print(f"depths               : {depths}")
    print(f"moment count         : {moment_count}")
    print(f"unfolding            : {args.unfold}")
    print(f"controls             : {controls}")
    print("=" * 124)

    t0 = time.time()
    mp.mp.dps = args.dps
    tol = mp.power(10, -(args.dps // 2))

    print("[1/7] Computing primitive Xi-resolvent moments...")
    xi_coeffs = xi_taylor_cauchy(
        max_order=xi_order,
        radius=mp.mpf(str(args.radius)),
        samples=args.samples,
        phase_scramble=False,
        seed=args.seed,
    )
    phi = phi_from_xi_coeffs(xi_coeffs, phi_order)
    signal_moments = [
        mp.re(x)
        for x in primitive_resolvent(phi, moment_count)
    ]

    print("[2/7] Computing reference zeta zeros...")
    zero_reference = np.array(
        [
            float(mp.im(mp.zetazero(k)))
            for k in range(1, args.zero_reference_count + 1)
        ],
        dtype=float,
    )

    phase_scrambled_moments = None

    if "phase_scramble" in controls:
        print("[3/7] Computing phase-scrambled Xi control moments...")
        xi_scrambled = xi_taylor_cauchy(
            max_order=xi_order,
            radius=mp.mpf(str(args.radius)),
            samples=args.samples,
            phase_scramble=True,
            seed=args.seed + 999,
        )
        phi_scrambled = phi_from_xi_coeffs(
            xi_scrambled,
            phi_order,
        )
        phase_scrambled_moments = [
            mp.re(x)
            for x in primitive_resolvent(
                phi_scrambled,
                moment_count,
            )
        ]
    else:
        print("[3/7] Phase-scrambled control skipped.")

    print("[4/7] Building canonical Jacobi operators...")
    summary_rows = []
    spacing_rows = []
    ratio_rows = []
    long_range_rows = []
    eigenvalue_rows = []
    coefficient_rows = []
    prefix_rows = []

    signal_results: Dict[int, JacobiResult] = {}

    for ci, control in enumerate(controls):
        if control == "phase_scramble":
            if phase_scrambled_moments is None:
                continue
            moments = phase_scrambled_moments
        else:
            moments = apply_control(
                signal_moments,
                control,
                args.seed + 1000 * ci,
            )

        for depth in depths:
            try:
                result = stieltjes_jacobi(
                    moments,
                    depth,
                    tol,
                )

                if control == "signal":
                    signal_results[depth] = result

                gamma = predict_gamma(result)
                gamma = np.sort(gamma[np.isfinite(gamma)])

                # Remove unstable extreme ends before local-statistics audit.
                trim = int(math.floor(args.trim_fraction * len(gamma)))
                if 2 * trim >= len(gamma) - 2:
                    trim = 0

                gamma_trim = (
                    gamma[trim:len(gamma) - trim]
                    if trim > 0
                    else gamma
                )

                unfolded = unfold_spectrum(
                    gamma_trim,
                    args.unfold,
                    args.poly_degree,
                )

                spacings = np.diff(unfolded)

                if len(spacings) and np.mean(spacings) > 0:
                    spacings = spacings / np.mean(spacings)

                ks = spacing_ks(spacings)
                ratios = spacing_ratios(spacings)
                ratio_stats = ratio_statistics(ratios)

                reconstruction = zero_reconstruction_metrics(
                    gamma,
                    zero_reference,
                )

                # Bulk/edge split.
                bulk_count = int(
                    math.floor(args.bulk_fraction * len(spacings))
                )
                edge_each = max(
                    0,
                    (len(spacings) - bulk_count) // 2,
                )

                if bulk_count > 0:
                    start = edge_each
                    bulk = spacings[start:start + bulk_count]
                else:
                    bulk = np.array([], dtype=float)

                edge = (
                    np.concatenate([
                        spacings[:edge_each],
                        spacings[-edge_each:],
                    ])
                    if edge_each > 0
                    else np.array([], dtype=float)
                )

                best_family = min(
                    {
                        "GUE": ks["ks_gue"],
                        "GOE": ks["ks_goe"],
                        "Poisson": ks["ks_poisson"],
                    },
                    key=lambda name: (
                        float("inf")
                        if not np.isfinite(
                            {
                                "GUE": ks["ks_gue"],
                                "GOE": ks["ks_goe"],
                                "Poisson": ks["ks_poisson"],
                            }[name]
                        )
                        else {
                            "GUE": ks["ks_gue"],
                            "GOE": ks["ks_goe"],
                            "Poisson": ks["ks_poisson"],
                        }[name]
                    ),
                )

                summary_rows.append({
                    "control": control,
                    "depth": depth,
                    "jacobi_size": len(result.alpha),
                    "positive_nodes": len(gamma),
                    "trimmed_nodes": len(gamma_trim),
                    "spacing_count": len(spacings),
                    **reconstruction,
                    **ks,
                    **ratio_stats,
                    "mean_spacing": (
                        float(np.mean(spacings))
                        if len(spacings)
                        else float("nan")
                    ),
                    "spacing_std": (
                        float(np.std(spacings))
                        if len(spacings)
                        else float("nan")
                    ),
                    "bulk_mean_spacing": (
                        float(np.mean(bulk))
                        if len(bulk)
                        else float("nan")
                    ),
                    "bulk_spacing_std": (
                        float(np.std(bulk))
                        if len(bulk)
                        else float("nan")
                    ),
                    "edge_mean_spacing": (
                        float(np.mean(edge))
                        if len(edge)
                        else float("nan")
                    ),
                    "edge_spacing_std": (
                        float(np.std(edge))
                        if len(edge)
                        else float("nan")
                    ),
                    "best_spacing_family": best_family,
                    "orthogonality_defect": float(
                        result.orthogonality_defect
                    ),
                    "breakdown_index": result.breakdown_index,
                    "min_beta": (
                        float(min(result.beta))
                        if result.beta
                        else float("nan")
                    ),
                })

                for i, value in enumerate(gamma):
                    eigenvalue_rows.append({
                        "control": control,
                        "depth": depth,
                        "index": i + 1,
                        "gamma": float(value),
                        "unfolded": (
                            float(unfolded[i - trim])
                            if trim <= i < len(gamma) - trim
                            else float("nan")
                        ),
                    })

                for i, value in enumerate(spacings):
                    spacing_rows.append({
                        "control": control,
                        "depth": depth,
                        "index": i + 1,
                        "spacing": float(value),
                    })

                for i, value in enumerate(ratios):
                    ratio_rows.append({
                        "control": control,
                        "depth": depth,
                        "index": i + 1,
                        "ratio": float(value),
                    })

                for row in number_variance(
                    unfolded,
                    L_values,
                    args.windows,
                ):
                    theory = theoretical_long_range(row["L"])
                    long_range_rows.append({
                        "control": control,
                        "depth": depth,
                        "statistic": "number_variance",
                        **row,
                        **theory,
                    })

                for row in delta3_statistic(
                    unfolded,
                    L_values,
                    args.windows,
                ):
                    theory = theoretical_long_range(row["L"])
                    long_range_rows.append({
                        "control": control,
                        "depth": depth,
                        "statistic": "delta3",
                        **row,
                        **theory,
                    })

                for i, value in enumerate(result.alpha):
                    coefficient_rows.append({
                        "control": control,
                        "depth": depth,
                        "kind": "alpha",
                        "index": i,
                        "value": mp.nstr(value, 50),
                    })

                for i, value in enumerate(result.beta, start=1):
                    coefficient_rows.append({
                        "control": control,
                        "depth": depth,
                        "kind": "beta",
                        "index": i,
                        "value": mp.nstr(value, 50),
                    })

            except Exception as exc:
                summary_rows.append({
                    "control": control,
                    "depth": depth,
                    "jacobi_size": 0,
                    "positive_nodes": 0,
                    "trimmed_nodes": 0,
                    "spacing_count": 0,
                    "error": str(exc),
                })

    print("[5/7] Auditing coefficient-prefix stability...")
    for d1, d2 in zip(depths[:-1], depths[1:]):
        if d1 not in signal_results or d2 not in signal_results:
            continue

        r1 = signal_results[d1]
        r2 = signal_results[d2]

        na = min(len(r1.alpha), len(r2.alpha))
        nb = min(len(r1.beta), len(r2.beta))

        da = [
            abs(r1.alpha[i] - r2.alpha[i])
            for i in range(na)
        ]
        db = [
            abs(r1.beta[i] - r2.beta[i])
            for i in range(nb)
        ]

        prefix_rows.append({
            "depth_small": d1,
            "depth_large": d2,
            "alpha_max_abs": (
                float(max(da))
                if da
                else 0.0
            ),
            "beta_max_abs": (
                float(max(db))
                if db
                else 0.0
            ),
            "alpha_rmse": (
                float(
                    mp.sqrt(
                        sum(x * x for x in da) / len(da)
                    )
                )
                if da
                else 0.0
            ),
            "beta_rmse": (
                float(
                    mp.sqrt(
                        sum(x * x for x in db) / len(db)
                    )
                )
                if db
                else 0.0
            ),
        })

    print("[6/7] Ranking GUE evidence...")
    signal_rows = [
        row
        for row in summary_rows
        if row.get("control") == "signal"
        and np.isfinite(float(row.get("ks_gue", float("nan"))))
    ]

    best_signal = (
        min(signal_rows, key=lambda row: float(row["ks_gue"]))
        if signal_rows
        else None
    )

    null_rows = [
        row
        for row in summary_rows
        if row.get("control") != "signal"
        and np.isfinite(float(row.get("ks_gue", float("nan"))))
    ]

    best_null = (
        min(null_rows, key=lambda row: float(row["ks_gue"]))
        if null_rows
        else None
    )

    null_ratio = (
        float(best_null["ks_gue"])
        / max(float(best_signal["ks_gue"]), 1e-15)
        if best_signal and best_null
        else float("nan")
    )

    print("[7/7] Writing outputs...")
    report = {
        "scientific_status": (
            "finite GUE audit of the primitive Xi-resolvent "
            "Stieltjes/Jacobi reconstruction; not an RH proof"
        ),
        "architecture": {
            "operator": "canonical Jacobi matrix from primitive Xi resolvent",
            "coefficient_fitting": False,
            "spectral_variable": "gamma=1/sqrt(lambda)",
            "unfolding": args.unfold,
        },
        "args": vars(args),
        "best_signal_gue": best_signal,
        "best_null_gue": best_null,
        "best_null_to_signal_gue_ks_ratio": null_ratio,
        "runtime_seconds": time.time() - t0,
    }

    save_csv(
        Path(str(prefix) + "_summary.csv"),
        summary_rows,
    )
    save_csv(
        Path(str(prefix) + "_spacings.csv"),
        spacing_rows,
    )
    save_csv(
        Path(str(prefix) + "_ratios.csv"),
        ratio_rows,
    )
    save_csv(
        Path(str(prefix) + "_long_range.csv"),
        long_range_rows,
    )
    save_csv(
        Path(str(prefix) + "_eigenvalues.csv"),
        eigenvalue_rows,
    )
    save_csv(
        Path(str(prefix) + "_jacobi_coefficients.csv"),
        coefficient_rows,
    )
    save_csv(
        Path(str(prefix) + "_prefix_stability.csv"),
        prefix_rows,
    )

    Path(str(prefix) + "_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print()
    print("=" * 124)
    print("FINAL PRIMITIVE RESOLVENT GUE RESULT")
    print("=" * 124)

    if best_signal:
        print(f"best signal depth         : {best_signal['depth']}")
        print(f"signal spacing count      : {best_signal['spacing_count']}")
        print(f"signal KS GUE             : {best_signal['ks_gue']:.8f}")
        print(f"signal KS GOE             : {best_signal['ks_goe']:.8f}")
        print(f"signal KS Poisson         : {best_signal['ks_poisson']:.8f}")
        print(f"signal mean ratio         : {best_signal['mean_ratio']:.8f}")
        print(f"signal ratio KS GUE       : {best_signal['ratio_ks_gue']:.8f}")
        print(f"signal zero relRMSE       : {best_signal['zero_relative_rmse']:.8e}")
        print(f"best spacing family       : {best_signal['best_spacing_family']}")

    if best_null:
        print(f"best null                 : {best_null['control']}")
        print(f"best null depth           : {best_null['depth']}")
        print(f"best null KS GUE          : {best_null['ks_gue']:.8f}")
        print(f"best-null/signal KS ratio : {null_ratio:.6f}")

    # With tiny spectra, KS thresholds must be interpreted cautiously.
    promising = (
        best_signal is not None
        and int(best_signal["spacing_count"]) >= 20
        and float(best_signal["ks_gue"])
            < float(best_signal["ks_goe"])
        and float(best_signal["ks_gue"])
            < float(best_signal["ks_poisson"])
        and (
            not np.isfinite(null_ratio)
            or null_ratio > 1.10
        )
    )

    print(
        "PROMISING FINITE GUE SIGNAL"
        if promising
        else "NO CONVINCING GUE SIGNAL AT CURRENT DEPTH"
    )
    print(f"runtime_seconds           : {time.time() - t0:.2f}")
    print(f"outputs                   : {prefix}_*.csv/json")
    print("=" * 124)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
