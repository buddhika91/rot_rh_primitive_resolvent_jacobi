#!/usr/bin/env python3
"""
rot_rh_resolvent_preregistered_gue_validation_fast_v2.py

Preregistered deeper-depth GUE validation for the canonical primitive
Xi-resolvent Stieltjes/Jacobi operator.

Purpose
-------
The previous frontier scan identified a promising fixed rule:

    relative depth tolerance = 1e-6
    absolute depth tolerance = 1e-6
    trim fraction            = 0
    unfolding methods        = rvm, polynomial, local

This script freezes that rule before evaluating deeper Jacobi truncations.

No tolerance scan.
No GUE optimization.
No zero correction.
No coefficient fitting.

Design
------
Calibration depths:
    used only to verify that the fixed rule reproduces the earlier behavior.

Evaluation depths:
    deeper holdout depths used for the actual preregistered test.

For every evaluation depth d, the spectrum is compared only with the previous
available depth d_prev. The longest consecutive block satisfying the frozen
stability rule is selected. The block is then scored under all requested
unfolding methods.

Primary preregistered endpoint
------------------------------
At the deepest evaluation depth:

1. at least --minimum-spacings stable consecutive spacings;
2. GUE KS < GOE KS;
3. GUE KS < Poisson KS;
4. GUE wins under at least --minimum-winning-methods unfolding methods;
5. mean ratio in [--ratio-min, --ratio-max];
6. bootstrap GUE-win fraction >= --minimum-bootstrap-win;
7. zero relative RMSE <= --maximum-zero-relative-rmse.

The operator remains the canonical resolvent-derived Jacobi operator.

This is a finite numerical validation, not an RH proof.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
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


# =============================================================================
# Parsing / files
# =============================================================================

def parse_int_list(text: str) -> List[int]:
    values = [int(x.strip()) for x in text.split(",") if x.strip()]
    if not values:
        raise argparse.ArgumentTypeError("Expected comma-separated integers")
    return values


def save_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        path.write_text("", encoding="utf-8")
        return

    columns: List[str] = []
    seen = set()

    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                columns.append(key)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


# =============================================================================
# Series tools
# =============================================================================

def trim_series(values: Sequence[mp.mpc], length: int) -> List[mp.mpc]:
    return list(values[:length]) + [mp.mpc(0)] * max(0, length - len(values))


def series_derivative(values: Sequence[mp.mpc]) -> List[mp.mpc]:
    if len(values) <= 1:
        return [mp.mpc(0)]
    return [(k + 1) * values[k + 1] for k in range(len(values) - 1)]


def series_inverse(values: Sequence[mp.mpc], length: int) -> List[mp.mpc]:
    values = trim_series(values, length)

    if abs(values[0]) == 0:
        raise ZeroDivisionError("Series inverse requires a nonzero constant.")

    inverse = [mp.mpc(0)] * length
    inverse[0] = 1 / values[0]

    for k in range(1, length):
        total = mp.mpc(0)

        for j in range(1, k + 1):
            total += values[j] * inverse[k - j]

        inverse[k] = -total / values[0]

    return inverse


def series_multiply(
    left: Sequence[mp.mpc],
    right: Sequence[mp.mpc],
    length: int,
) -> List[mp.mpc]:
    left = trim_series(left, length)
    right = trim_series(right, length)
    result = [mp.mpc(0)] * length

    for k in range(length):
        total = mp.mpc(0)

        for j in range(k + 1):
            total += left[j] * right[k - j]

        result[k] = total

    return result


# =============================================================================
# Xi / primitive resolvent
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
    maximum_order: int,
    radius: mp.mpf,
    samples: int,
) -> List[mp.mpc]:
    """
    Symmetry-reduced Cauchy extraction.

    Xi(1/2+z)=Xi(1/2-z) and Xi(conj(s))=conj(Xi(s)). Therefore only
    one quarter of the contour needs an expensive Xi evaluation. Odd Taylor
    coefficients vanish, and even coefficients are recovered from conjugate
    pairs. For samples=8192 this uses 2049 Xi evaluations instead of 8192.
    """
    if samples < 8 or samples % 4 != 0:
        raise ValueError("--samples must be divisible by 4 and at least 8")

    center = mp.mpf("0.5")
    quarter = samples // 4
    half = samples // 2

    roots: List[mp.mpc] = []
    values: List[mp.mpc] = []

    for j in range(quarter + 1):
        angle = 2 * mp.pi * j / samples
        root = mp.e ** (1j * angle)
        roots.append(root)
        values.append(xi(center + radius * root))

        if j and (j % max(1, quarter // 8) == 0 or j == quarter):
            print(f"      Xi contour {j:5d}/{quarter} ({100*j/quarter:5.1f}%)", flush=True)

    coefficients: List[mp.mpc] = [mp.mpc(0)] * (maximum_order + 1)
    max_k = maximum_order // 2

    # phase[j] = root_j^(-2k), updated recursively to avoid repeated powers.
    phases = [mp.mpc(1)] * (quarter + 1)
    phase_steps = [root ** (-2) for root in roots]
    radius_power = mp.mpf(1)
    radius_step = radius * radius

    for k in range(max_k + 1):
        total = mp.re(values[0] * phases[0])
        total += mp.re(values[quarter] * phases[quarter])

        for j in range(1, quarter):
            total += 2 * mp.re(values[j] * phases[j])

        coefficients[2 * k] = mp.mpc(total / half / radius_power)

        for j in range(quarter + 1):
            phases[j] *= phase_steps[j]
        radius_power *= radius_step

    return coefficients


def phi_from_xi_coefficients(
    coefficients: Sequence[mp.mpc],
    order_u: int,
) -> List[mp.mpc]:
    constant = coefficients[0]

    return [
        coefficients[2 * j] * ((-1) ** j) / constant
        for j in range(order_u + 1)
    ]


def primitive_resolvent_moments(
    phi: Sequence[mp.mpc],
    count: int,
) -> List[mp.mpf]:
    derivative = series_derivative(phi)
    inverse = series_inverse(phi, count)
    product = series_multiply(derivative, inverse, count)

    return [mp.re(-value) for value in product]


# =============================================================================
# Canonical Stieltjes recursion
# =============================================================================

def polynomial_trim(values: Sequence[mp.mpf], length: int) -> List[mp.mpf]:
    return list(values[:length]) + [mp.mpf("0")] * max(0, length - len(values))


def polynomial_add(
    left: Sequence[mp.mpf],
    right: Sequence[mp.mpf],
    left_scale: mp.mpf = mp.mpf("1"),
    right_scale: mp.mpf = mp.mpf("1"),
) -> List[mp.mpf]:
    length = max(len(left), len(right))
    left = polynomial_trim(left, length)
    right = polynomial_trim(right, length)

    return [
        left_scale * left[i] + right_scale * right[i]
        for i in range(length)
    ]


def polynomial_scale(values: Sequence[mp.mpf], scale: mp.mpf) -> List[mp.mpf]:
    return [scale * value for value in values]


def multiply_by_x(values: Sequence[mp.mpf]) -> List[mp.mpf]:
    return [mp.mpf("0")] + list(values)


def moment_inner_product(
    left: Sequence[mp.mpf],
    right: Sequence[mp.mpf],
    moments: Sequence[mp.mpf],
) -> mp.mpf:
    total = mp.mpf("0")

    for i, left_value in enumerate(left):
        for j, right_value in enumerate(right):
            total += left_value * right_value * moments[i + j]

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
    tolerance: mp.mpf,
) -> JacobiResult:
    if len(moments) < 2 * depth + 2:
        raise ValueError("At least 2*depth+2 moments are required.")

    if moments[0] <= tolerance:
        raise ValueError("The zeroth moment must be positive.")

    previous = [mp.mpf("0")]
    current = [1 / mp.sqrt(moments[0])]
    basis = [current]

    alpha: List[mp.mpf] = []
    beta: List[mp.mpf] = []
    previous_beta = mp.mpf("0")
    breakdown_index = -1

    for n in range(depth):
        x_current = multiply_by_x(current)
        diagonal = moment_inner_product(x_current, current, moments)
        alpha.append(diagonal)

        residual = polynomial_add(
            x_current,
            current,
            mp.mpf("1"),
            -diagonal,
        )

        if n > 0:
            residual = polynomial_add(
                residual,
                previous,
                mp.mpf("1"),
                -previous_beta,
            )

        for basis_polynomial in basis:
            projection = moment_inner_product(
                residual,
                basis_polynomial,
                moments,
            )

            residual = polynomial_add(
                residual,
                basis_polynomial,
                mp.mpf("1"),
                -projection,
            )

        if n == depth - 1:
            break

        norm_squared = moment_inner_product(
            residual,
            residual,
            moments,
        )

        if norm_squared <= tolerance:
            breakdown_index = n
            break

        off_diagonal = mp.sqrt(norm_squared)
        beta.append(off_diagonal)

        previous = current
        current = polynomial_scale(residual, 1 / off_diagonal)
        previous_beta = off_diagonal
        basis.append(current)

    defect = mp.mpf("0")

    for i, left in enumerate(basis):
        for j, right in enumerate(basis):
            target = mp.mpf("1") if i == j else mp.mpf("0")
            defect = max(
                defect,
                abs(moment_inner_product(left, right, moments) - target),
            )

    return JacobiResult(alpha, beta, defect, breakdown_index)


def jacobi_matrix(result: JacobiResult) -> np.ndarray:
    alpha = np.array([float(value) for value in result.alpha], dtype=float)
    beta = np.array([float(value) for value in result.beta], dtype=float)

    matrix = np.diag(alpha)

    if len(beta):
        matrix += np.diag(beta, 1)
        matrix += np.diag(beta, -1)

    return 0.5 * (matrix + matrix.T)


def reconstructed_ordinates(result: JacobiResult) -> np.ndarray:
    nodes = np.real(
        eigh(
            jacobi_matrix(result),
            eigvals_only=True,
            check_finite=False,
            driver="evr",
        )
    )
    nodes = np.sort(nodes[np.isfinite(nodes) & (nodes > 0)])[::-1]

    return 1.0 / np.sqrt(nodes)


# =============================================================================
# Frozen stable-block rule
# =============================================================================

def longest_true_block(mask: np.ndarray) -> Tuple[int, int]:
    best_start = 0
    best_end = 0
    current_start = None

    for index, value in enumerate(mask):
        if value and current_start is None:
            current_start = index

        closing = (not value) or (index == len(mask) - 1)

        if closing and current_start is not None:
            current_end = (
                index + 1
                if value and index == len(mask) - 1
                else index
            )

            if current_end - current_start > best_end - best_start:
                best_start = current_start
                best_end = current_end

            current_start = None

    return best_start, best_end


def frozen_stable_block(
    current: np.ndarray,
    previous: np.ndarray,
    relative_tolerance: float,
    absolute_tolerance: float,
    trim_fraction: float,
) -> Dict[str, object]:
    count = min(len(current), len(previous))

    relative = np.full(len(current), np.inf, dtype=float)
    absolute = np.full(len(current), np.inf, dtype=float)

    if count:
        absolute[:count] = np.abs(current[:count] - previous[:count])
        relative[:count] = absolute[:count] / np.maximum(
            np.abs(current[:count]),
            1e-300,
        )

    stable_mask = (
        (relative <= relative_tolerance)
        | (absolute <= absolute_tolerance)
    )

    start, end = longest_true_block(stable_mask)
    stable_count = end - start

    trim = int(math.floor(trim_fraction * stable_count))

    if stable_count - 2 * trim < 4:
        trim = 0

    bulk_start = start + trim
    bulk_end = end - trim

    return {
        "stable_mask": stable_mask,
        "relative_difference": relative,
        "absolute_difference": absolute,
        "stable_start": start,
        "stable_end": end,
        "stable_count": stable_count,
        "bulk_start": bulk_start,
        "bulk_end": bulk_end,
        "block": current[bulk_start:bulk_end],
    }


# =============================================================================
# Unfolding / distributions
# =============================================================================

def rvm_count(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    scaled = np.maximum(values / (2.0 * np.pi), 1e-15)

    return (
        scaled * np.log(scaled)
        - scaled
        + 7.0 / 8.0
    )


def unfold(
    gamma: np.ndarray,
    method: str,
    polynomial_degree: int,
) -> np.ndarray:
    gamma = np.sort(np.asarray(gamma, dtype=float))

    if method == "rvm":
        unfolded = rvm_count(gamma)

    elif method == "polynomial":
        indices = np.arange(1, len(gamma) + 1, dtype=float)
        degree = min(polynomial_degree, max(1, len(gamma) - 2))
        coefficients = np.polyfit(gamma, indices, degree)
        unfolded = np.polyval(coefficients, gamma)

    elif method == "local":
        spacings = np.diff(gamma)

        if len(spacings) < 3:
            return np.arange(len(gamma), dtype=float)

        window = max(3, min(9, (len(spacings) // 3) * 2 + 1))
        kernel = np.ones(window, dtype=float) / window
        padded = np.pad(
            spacings,
            (window // 2, window // 2),
            mode="edge",
        )
        local_mean = np.convolve(padded, kernel, mode="valid")[:len(spacings)]
        normalized = spacings / np.maximum(local_mean, 1e-15)
        unfolded = np.concatenate([[0.0], np.cumsum(normalized)])

    else:
        raise ValueError(method)

    unfolded = np.maximum.accumulate(unfolded)

    if len(unfolded) > 1:
        mean_spacing = float(np.mean(np.diff(unfolded)))

        if mean_spacing > 0:
            unfolded /= mean_spacing

    return unfolded


def gue_cdf(values: np.ndarray) -> np.ndarray:
    values = np.maximum(np.asarray(values, dtype=float), 0.0)

    return (
        np.vectorize(math.erf)(2.0 * values / math.sqrt(np.pi))
        - (4.0 / np.pi)
        * values
        * np.exp(-4.0 * values * values / np.pi)
    )


def goe_cdf(values: np.ndarray) -> np.ndarray:
    values = np.maximum(np.asarray(values, dtype=float), 0.0)
    return 1.0 - np.exp(-np.pi * values * values / 4.0)


def poisson_cdf(values: np.ndarray) -> np.ndarray:
    values = np.maximum(np.asarray(values, dtype=float), 0.0)
    return 1.0 - np.exp(-values)


def ratio_pdf(values: np.ndarray, beta: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)

    raw = (
        (values + values * values) ** beta
        / (1.0 + values + values * values) ** (1.0 + 1.5 * beta)
    )

    grid = np.linspace(0.0, 1.0, 40001)
    grid_raw = (
        (grid + grid * grid) ** beta
        / (1.0 + grid + grid * grid) ** (1.0 + 1.5 * beta)
    )
    normalization = np.trapezoid(grid_raw, grid)

    return raw / max(normalization, 1e-15)


def ratio_cdf_factory(beta: int):
    grid = np.linspace(0.0, 1.0, 50001)
    pdf = ratio_pdf(grid, beta)
    step = grid[1] - grid[0]
    cdf = np.cumsum(pdf) * step
    cdf /= cdf[-1]

    def cdf_function(values):
        return np.interp(values, grid, cdf, left=0.0, right=1.0)

    return cdf_function


RATIO_GUE_CDF = ratio_cdf_factory(2)
RATIO_GOE_CDF = ratio_cdf_factory(1)


def ratio_poisson_cdf(values: np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(values, dtype=float), 0.0, 1.0)
    return 2.0 * values / (1.0 + values)


def spacing_ratios(spacings: np.ndarray) -> np.ndarray:
    spacings = np.asarray(spacings, dtype=float)

    if len(spacings) < 2:
        return np.array([], dtype=float)

    left = spacings[:-1]
    right = spacings[1:]
    denominator = np.maximum(left, right)
    mask = denominator > 0

    return np.minimum(left[mask], right[mask]) / denominator[mask]


def spectral_statistics(
    gamma: np.ndarray,
    method: str,
    polynomial_degree: int,
) -> Dict[str, object]:
    gamma = np.sort(np.asarray(gamma, dtype=float))

    if len(gamma) < 4:
        return {
            "node_count": len(gamma),
            "spacing_count": max(0, len(gamma) - 1),
            "ks_gue": float("nan"),
            "ks_goe": float("nan"),
            "ks_poisson": float("nan"),
            "mean_ratio": float("nan"),
            "ratio_ks_gue": float("nan"),
            "ratio_ks_goe": float("nan"),
            "ratio_ks_poisson": float("nan"),
            "spacings": np.array([], dtype=float),
        }

    unfolded = unfold(gamma, method, polynomial_degree)
    spacings = np.diff(unfolded)
    spacings = spacings[np.isfinite(spacings) & (spacings >= 0)]

    if len(spacings):
        spacings /= max(float(np.mean(spacings)), 1e-15)

    ratios = spacing_ratios(spacings)

    return {
        "node_count": len(gamma),
        "spacing_count": len(spacings),
        "ks_gue": (
            float(kstest(spacings, gue_cdf).statistic)
            if len(spacings) >= 2
            else float("nan")
        ),
        "ks_goe": (
            float(kstest(spacings, goe_cdf).statistic)
            if len(spacings) >= 2
            else float("nan")
        ),
        "ks_poisson": (
            float(kstest(spacings, poisson_cdf).statistic)
            if len(spacings) >= 2
            else float("nan")
        ),
        "mean_ratio": (
            float(np.mean(ratios))
            if len(ratios)
            else float("nan")
        ),
        "ratio_ks_gue": (
            float(kstest(ratios, RATIO_GUE_CDF).statistic)
            if len(ratios) >= 2
            else float("nan")
        ),
        "ratio_ks_goe": (
            float(kstest(ratios, RATIO_GOE_CDF).statistic)
            if len(ratios) >= 2
            else float("nan")
        ),
        "ratio_ks_poisson": (
            float(kstest(ratios, ratio_poisson_cdf).statistic)
            if len(ratios) >= 2
            else float("nan")
        ),
        "spacings": spacings,
    }


# =============================================================================
# Bootstrap / zero metrics
# =============================================================================

def bootstrap_gue_win_fraction(
    spacings: np.ndarray,
    trials: int,
    seed: int,
) -> float:
    spacings = np.asarray(spacings, dtype=float)

    if len(spacings) < 4 or trials <= 0:
        return float("nan")

    rng = np.random.default_rng(seed)
    wins = 0

    for _ in range(trials):
        sample = rng.choice(spacings, size=len(spacings), replace=True)
        sample /= max(float(np.mean(sample)), 1e-15)

        ks_gue = float(kstest(sample, gue_cdf).statistic)
        ks_goe = float(kstest(sample, goe_cdf).statistic)
        ks_poisson = float(kstest(sample, poisson_cdf).statistic)

        if ks_gue < ks_goe and ks_gue < ks_poisson:
            wins += 1

    return wins / trials


def zero_metrics(
    gamma: np.ndarray,
    reference: np.ndarray,
) -> Dict[str, float]:
    count = min(len(gamma), len(reference))

    if count == 0:
        return {
            "zero_count": 0,
            "zero_rmse": float("nan"),
            "zero_relative_rmse": float("nan"),
            "zero_max_abs_error": float("nan"),
        }

    error = gamma[:count] - reference[:count]

    return {
        "zero_count": count,
        "zero_rmse": float(np.sqrt(np.mean(error * error))),
        "zero_relative_rmse": float(
            np.sqrt(np.mean((error / reference[:count]) ** 2))
        ),
        "zero_max_abs_error": float(np.max(np.abs(error))),
    }


# =============================================================================
# Cache helpers
# =============================================================================

def cache_paths(cache_prefix: Path) -> Dict[str, Path]:
    return {
        "moments": Path(str(cache_prefix) + "_moments.pkl"),
        "spectra": Path(str(cache_prefix) + "_spectra.pkl"),
        "meta": Path(str(cache_prefix) + "_meta.json"),
    }


def load_cache(
    cache_prefix: Path,
    required_depths: Sequence[int],
    args: argparse.Namespace,
) -> Tuple[List[mp.mpf] | None, Dict[int, np.ndarray] | None]:
    paths = cache_paths(cache_prefix)

    if not paths["meta"].exists():
        return None, None

    try:
        meta = json.loads(paths["meta"].read_text(encoding="utf-8"))
    except Exception:
        return None, None

    compatible = (
        meta.get("dps") == args.dps
        and float(meta.get("radius")) == float(args.radius)
        and meta.get("samples") == args.samples
        and max(meta.get("depths", [0])) >= max(required_depths)
        and meta.get("algorithm") == "quarter_contour_single_stieltjes_v2"
    )

    if not compatible:
        return None, None

    moments = None
    spectra = None

    if paths["moments"].exists():
        with paths["moments"].open("rb") as handle:
            moments = pickle.load(handle)

    if paths["spectra"].exists():
        with paths["spectra"].open("rb") as handle:
            spectra = pickle.load(handle)

    return moments, spectra


def save_moments_checkpoint(
    cache_prefix: Path,
    moments: Sequence[mp.mpf],
    required_depths: Sequence[int],
    args: argparse.Namespace,
) -> None:
    """Save the expensive resolvent moments immediately after stage 1."""
    paths = cache_paths(cache_prefix)
    cache_prefix.parent.mkdir(parents=True, exist_ok=True)

    with paths["moments"].open("wb") as handle:
        pickle.dump(list(moments), handle)

    paths["meta"].write_text(
        json.dumps(
            {
                "dps": args.dps,
                "radius": args.radius,
                "samples": args.samples,
                "depths": sorted(required_depths),
                "algorithm": "quarter_contour_single_stieltjes_v2",
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def save_cache(
    cache_prefix: Path,
    moments: Sequence[mp.mpf],
    spectra: Dict[int, np.ndarray],
    args: argparse.Namespace,
) -> None:
    paths = cache_paths(cache_prefix)
    cache_prefix.parent.mkdir(parents=True, exist_ok=True)

    with paths["moments"].open("wb") as handle:
        pickle.dump(list(moments), handle)

    with paths["spectra"].open("wb") as handle:
        pickle.dump(spectra, handle)

    paths["meta"].write_text(
        json.dumps(
            {
                "dps": args.dps,
                "radius": args.radius,
                "samples": args.samples,
                "depths": sorted(spectra),
                "algorithm": "quarter_contour_single_stieltjes_v2",
            },
            indent=2,
        ),
        encoding="utf-8",
    )


# =============================================================================
# Main
# =============================================================================

def parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument("--dps", type=int, default=400)
    parser.add_argument("--radius", type=float, default=4.0)
    parser.add_argument("--samples", type=int, default=8192)

    parser.add_argument(
        "--calibration-depths",
        type=parse_int_list,
        default=[48, 52],
    )
    parser.add_argument(
        "--evaluation-depths",
        type=parse_int_list,
        default=[56, 60, 64],
    )
    parser.add_argument("--zero-reference-count", type=int, default=64)

    # Frozen rule.
    parser.add_argument(
        "--stable-relative-tolerance",
        type=float,
        default=1e-6,
    )
    parser.add_argument(
        "--stable-absolute-tolerance",
        type=float,
        default=1e-6,
    )
    parser.add_argument("--trim-fraction", type=float, default=0.0)

    parser.add_argument(
        "--unfold-methods",
        default="rvm,polynomial,local",
    )
    parser.add_argument("--poly-degree", type=int, default=5)

    # Preregistered endpoints.
    parser.add_argument("--minimum-spacings", type=int, default=28)
    parser.add_argument("--minimum-winning-methods", type=int, default=2)
    parser.add_argument("--ratio-min", type=float, default=0.55)
    parser.add_argument("--ratio-max", type=float, default=0.65)
    parser.add_argument("--minimum-bootstrap-win", type=float, default=0.80)
    parser.add_argument(
        "--maximum-zero-relative-rmse",
        type=float,
        default=1e-8,
    )
    parser.add_argument("--bootstrap-trials", type=int, default=1000)

    parser.add_argument("--seed", type=int, default=20260714)

    parser.add_argument(
        "--cache-prefix",
        default="rot_rh_resolvent_preregistered_fast_v2_cache",
    )
    parser.add_argument("--no-cache", action="store_true")

    parser.add_argument(
        "--out-prefix",
        default="rot_rh_resolvent_preregistered_gue_fast_v2",
    )

    return parser


def main() -> int:
    args = parser().parse_args()

    calibration_depths = sorted(set(args.calibration_depths))
    evaluation_depths = sorted(set(args.evaluation_depths))
    all_depths = sorted(set(calibration_depths + evaluation_depths))

    if not evaluation_depths:
        raise SystemExit("At least one evaluation depth is required.")

    methods = [
        value.strip()
        for value in args.unfold_methods.split(",")
        if value.strip()
    ]

    valid_methods = {"rvm", "polynomial", "local"}

    if set(methods) - valid_methods:
        raise SystemExit("Unknown unfolding method.")

    maximum_depth = max(all_depths)
    moment_count = 2 * maximum_depth + 8
    phi_order = moment_count + 2
    xi_order = 2 * phi_order + 2

    prefix = Path(args.out_prefix).expanduser().resolve()
    prefix.parent.mkdir(parents=True, exist_ok=True)

    cache_prefix = Path(args.cache_prefix).expanduser().resolve()

    print("=" * 128)
    print("ROT-RH / FAST PREREGISTERED DEEP GUE VALIDATION")
    print("=" * 128)
    print(f"dps / radius / samples       : {args.dps} / {args.radius} / {args.samples}")
    print(f"calibration depths           : {calibration_depths}")
    print(f"evaluation depths            : {evaluation_depths}")
    print(f"frozen rel/abs tolerance     : {args.stable_relative_tolerance} / {args.stable_absolute_tolerance}")
    print(f"frozen trim fraction         : {args.trim_fraction}")
    print(f"unfolding methods            : {methods}")
    print(f"minimum spacings             : {args.minimum_spacings}")
    print(f"minimum winning methods      : {args.minimum_winning_methods}")
    print(f"minimum bootstrap GUE win    : {args.minimum_bootstrap_win}")
    print(f"maximum zero relative RMSE   : {args.maximum_zero_relative_rmse}")
    print("=" * 128)

    start_time = time.time()
    mp.mp.dps = args.dps
    stieltjes_tolerance = mp.power(10, -(args.dps // 2))

    moments = None
    spectra = None

    if not args.no_cache:
        moments, spectra = load_cache(cache_prefix, all_depths, args)

        if moments is not None and spectra is not None:
            print("[cache] Loaded compatible moments and spectra.")

    if moments is None:
        print("[1/6] Computing primitive Xi-resolvent moments...")
        xi_coefficients = xi_taylor_cauchy(
            xi_order,
            mp.mpf(str(args.radius)),
            args.samples,
        )
        phi = phi_from_xi_coefficients(xi_coefficients, phi_order)
        moments = primitive_resolvent_moments(phi, moment_count)

        if not args.no_cache:
            save_moments_checkpoint(cache_prefix, moments, all_depths, args)
            print("[cache] Saved resolvent-moment checkpoint.")

    if spectra is None:
        print("[2/6] Building calibration and holdout Jacobi spectra...")
        spectra = {}
        jacobi_rows = []

        # One maximum-depth Stieltjes run generates the entire canonical
        # coefficient prefix. Shallower matrices are exact truncations.
        maximum_result = stieltjes_jacobi(
            moments,
            maximum_depth,
            stieltjes_tolerance,
        )

        for depth in all_depths:
            truncated = JacobiResult(
                alpha=maximum_result.alpha[:depth],
                beta=maximum_result.beta[:max(0, depth - 1)],
                orthogonality_defect=maximum_result.orthogonality_defect,
                breakdown_index=(
                    maximum_result.breakdown_index
                    if 0 <= maximum_result.breakdown_index < depth
                    else -1
                ),
            )
            spectra[depth] = reconstructed_ordinates(truncated)

            jacobi_rows.append({
                "depth": depth,
                "node_count": len(spectra[depth]),
                "orthogonality_defect": float(maximum_result.orthogonality_defect),
                "breakdown_index": truncated.breakdown_index,
                "minimum_beta": (
                    float(min(truncated.beta))
                    if truncated.beta
                    else float("nan")
                ),
            })

            print(
                f"  depth={depth:3d} nodes={len(spectra[depth]):3d} "
                f"orth(max)={float(maximum_result.orthogonality_defect):.3e}"
            )

        save_csv(Path(str(prefix) + "_jacobi_quality.csv"), jacobi_rows)

        if not args.no_cache:
            save_cache(cache_prefix, moments, spectra, args)
            print("[cache] Saved moments and spectra.")

    print("[3/6] Computing reference zeta zeros...")
    reference_zeros = np.array(
        [
            float(mp.im(mp.zetazero(index)))
            for index in range(1, args.zero_reference_count + 1)
        ],
        dtype=float,
    )

    print("[4/6] Applying the frozen stability rule...")
    depth_rows = []
    node_rows = []
    method_rows = []
    spacing_rows = []

    for position, depth in enumerate(all_depths):
        if position == 0:
            continue

        previous_depth = all_depths[position - 1]
        current = spectra[depth]
        previous = spectra[previous_depth]

        selected = frozen_stable_block(
            current=current,
            previous=previous,
            relative_tolerance=args.stable_relative_tolerance,
            absolute_tolerance=args.stable_absolute_tolerance,
            trim_fraction=args.trim_fraction,
        )

        block = np.asarray(selected["block"], dtype=float)
        target_slice = reference_zeros[
            int(selected["bulk_start"]):
            int(selected["bulk_start"]) + len(block)
        ]
        zero_result = zero_metrics(block, target_slice)

        split = (
            "evaluation"
            if depth in evaluation_depths
            else "calibration"
        )

        depth_rows.append({
            "split": split,
            "depth": depth,
            "previous_depth": previous_depth,
            "total_nodes": len(current),
            "stable_start_index": int(selected["stable_start"]) + 1,
            "stable_end_index": int(selected["stable_end"]),
            "stable_count": int(selected["stable_count"]),
            "bulk_start_index": int(selected["bulk_start"]) + 1,
            "bulk_end_index": int(selected["bulk_end"]),
            "bulk_count": len(block),
            **zero_result,
        })

        for index, value in enumerate(current):
            node_rows.append({
                "split": split,
                "depth": depth,
                "previous_depth": previous_depth,
                "index": index + 1,
                "gamma": float(value),
                "is_stable": bool(selected["stable_mask"][index]),
                "is_bulk": bool(
                    selected["bulk_start"]
                    <= index
                    < selected["bulk_end"]
                ),
                "relative_depth_difference": float(
                    selected["relative_difference"][index]
                ),
                "absolute_depth_difference": float(
                    selected["absolute_difference"][index]
                ),
                "target_zero": (
                    float(reference_zeros[index])
                    if index < len(reference_zeros)
                    else float("nan")
                ),
            })

        for method_index, method in enumerate(methods):
            stats = spectral_statistics(
                block,
                method,
                args.poly_degree,
            )
            bootstrap = bootstrap_gue_win_fraction(
                stats["spacings"],
                args.bootstrap_trials,
                args.seed + 10000 * depth + method_index,
            )

            gue_wins = (
                np.isfinite(stats["ks_gue"])
                and stats["ks_gue"] < stats["ks_goe"]
                and stats["ks_gue"] < stats["ks_poisson"]
            )

            ratio_pass = (
                np.isfinite(stats["mean_ratio"])
                and args.ratio_min
                <= stats["mean_ratio"]
                <= args.ratio_max
            )

            method_rows.append({
                "split": split,
                "depth": depth,
                "previous_depth": previous_depth,
                "unfolding": method,
                "stable_count": int(selected["stable_count"]),
                "bulk_count": len(block),
                "spacing_count": int(stats["spacing_count"]),
                "ks_gue": stats["ks_gue"],
                "ks_goe": stats["ks_goe"],
                "ks_poisson": stats["ks_poisson"],
                "gue_wins": bool(gue_wins),
                "mean_ratio": stats["mean_ratio"],
                "ratio_pass": bool(ratio_pass),
                "ratio_ks_gue": stats["ratio_ks_gue"],
                "ratio_ks_goe": stats["ratio_ks_goe"],
                "ratio_ks_poisson": stats["ratio_ks_poisson"],
                "bootstrap_gue_win_fraction": bootstrap,
                **zero_result,
            })

            for index, spacing in enumerate(stats["spacings"]):
                spacing_rows.append({
                    "split": split,
                    "depth": depth,
                    "unfolding": method,
                    "index": index + 1,
                    "spacing": float(spacing),
                })

    print("[5/6] Evaluating preregistered endpoints...")
    evaluation_summary_rows = []

    for depth in evaluation_depths:
        rows = [
            row
            for row in method_rows
            if row["depth"] == depth
            and row["split"] == "evaluation"
        ]

        if not rows:
            evaluation_summary_rows.append({
                "depth": depth,
                "status": "NO_RESULT",
            })
            continue

        winning_methods = sum(bool(row["gue_wins"]) for row in rows)
        ratio_passing_methods = sum(bool(row["ratio_pass"]) for row in rows)
        minimum_spacings = min(int(row["spacing_count"]) for row in rows)
        finite_bootstrap_values = [
            float(row["bootstrap_gue_win_fraction"])
            for row in rows
            if np.isfinite(row["bootstrap_gue_win_fraction"])
        ]
        minimum_bootstrap = (
            min(finite_bootstrap_values)
            if finite_bootstrap_values
            else 0.0
        )
        maximum_zero_relative_rmse = max(
            float(row["zero_relative_rmse"])
            for row in rows
            if np.isfinite(row["zero_relative_rmse"])
        )

        pass_spacings = minimum_spacings >= args.minimum_spacings
        pass_winning_methods = (
            winning_methods >= args.minimum_winning_methods
        )
        pass_ratio = ratio_passing_methods >= args.minimum_winning_methods
        pass_bootstrap = (
            minimum_bootstrap >= args.minimum_bootstrap_win
        )
        pass_zero = (
            maximum_zero_relative_rmse
            <= args.maximum_zero_relative_rmse
        )

        overall_pass = (
            pass_spacings
            and pass_winning_methods
            and pass_ratio
            and pass_bootstrap
            and pass_zero
        )

        evaluation_summary_rows.append({
            "depth": depth,
            "status": "PASS" if overall_pass else "FAIL",
            "methods": len(rows),
            "gue_winning_methods": winning_methods,
            "ratio_passing_methods": ratio_passing_methods,
            "minimum_spacing_count": minimum_spacings,
            "minimum_bootstrap_gue_win_fraction": minimum_bootstrap,
            "maximum_zero_relative_rmse": maximum_zero_relative_rmse,
            "pass_spacings": pass_spacings,
            "pass_winning_methods": pass_winning_methods,
            "pass_ratio": pass_ratio,
            "pass_bootstrap": pass_bootstrap,
            "pass_zero_accuracy": pass_zero,
        })

    deepest_depth = max(evaluation_depths)
    deepest_summary = next(
        (
            row
            for row in evaluation_summary_rows
            if row["depth"] == deepest_depth
        ),
        None,
    )

    print("[6/6] Writing outputs...")
    report = {
        "scientific_status": (
            "preregistered finite deeper-depth GUE validation of the "
            "unchanged canonical resolvent Jacobi operator; not an RH proof"
        ),
        "frozen_rule": {
            "relative_tolerance": args.stable_relative_tolerance,
            "absolute_tolerance": args.stable_absolute_tolerance,
            "trim_fraction": args.trim_fraction,
            "unfolding_methods": methods,
        },
        "primary_endpoint": {
            "depth": deepest_depth,
            "minimum_spacings": args.minimum_spacings,
            "minimum_winning_methods": args.minimum_winning_methods,
            "ratio_interval": [args.ratio_min, args.ratio_max],
            "minimum_bootstrap_win": args.minimum_bootstrap_win,
            "maximum_zero_relative_rmse": args.maximum_zero_relative_rmse,
        },
        "deepest_evaluation_result": deepest_summary,
        "all_evaluation_results": evaluation_summary_rows,
        "runtime_seconds": time.time() - start_time,
    }

    save_csv(Path(str(prefix) + "_depth_blocks.csv"), depth_rows)
    save_csv(Path(str(prefix) + "_nodes.csv"), node_rows)
    save_csv(Path(str(prefix) + "_method_statistics.csv"), method_rows)
    save_csv(Path(str(prefix) + "_spacings.csv"), spacing_rows)
    save_csv(
        Path(str(prefix) + "_evaluation_summary.csv"),
        evaluation_summary_rows,
    )

    Path(str(prefix) + "_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print()
    print("=" * 128)
    print("FINAL PREREGISTERED DEEP GUE VALIDATION")
    print("=" * 128)

    for row in evaluation_summary_rows:
        if row.get("status") == "NO_RESULT":
            print(f"depth={row['depth']:3d} status=NO_RESULT")
            continue

        print(
            f"depth={row['depth']:3d} "
            f"status={row['status']:4s} "
            f"spacings={row['minimum_spacing_count']:3d} "
            f"GUE_methods={row['gue_winning_methods']}/{row['methods']} "
            f"ratio_methods={row['ratio_passing_methods']}/{row['methods']} "
            f"bootstrap_min={row['minimum_bootstrap_gue_win_fraction']:.3f} "
            f"zero_rel={row['maximum_zero_relative_rmse']:.3e}"
        )

    print("-" * 128)

    if deepest_summary and deepest_summary.get("status") == "PASS":
        print("PREREGISTERED PRIMARY ENDPOINT PASSED")
    else:
        print("PREREGISTERED PRIMARY ENDPOINT DID NOT PASS")

    print(f"runtime_seconds: {time.time() - start_time:.2f}")
    print(f"outputs: {prefix}_*.csv/json")
    print("=" * 128)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
