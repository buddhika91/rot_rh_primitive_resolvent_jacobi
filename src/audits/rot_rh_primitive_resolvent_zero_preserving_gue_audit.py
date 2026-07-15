#!/usr/bin/env python3
"""
rot_rh_primitive_resolvent_zero_preserving_gue_audit.py

Depth-stable, zero-preserving GUE audit for the canonical primitive
Xi-resolvent Stieltjes/Jacobi construction.

There are two distinct analyses:

A. CANONICAL ANALYSIS
   No spectral values are changed. Nodes are retained only when they are stable
   across Jacobi depths. GUE diagnostics are then computed on the longest
   consecutive stable bulk block.

B. OPTIONAL CONSTRAINED MICRO-CORRECTION
   A smooth monotone displacement field is optimized to improve finite-sample
   GUE diagnostics, subject to a strict per-level displacement tolerance and
   explicit zero-location penalties.

   This mode is diagnostic only. It does not remain the canonical unfitted
   operator and must not be presented as independent GUE evidence.

Pipeline
--------
    Xi -> primitive resolvent moments -> canonical Jacobi depth flow
       -> reconstructed gamma values
       -> cross-depth stability filter
       -> stable consecutive bulk block
       -> GUE / GOE / Poisson statistics
       -> optional tiny constrained correction.

Main safeguards
---------------
- No correction is used in canonical scores.
- Every corrected value is compared with the original reconstructed value.
- Monotonicity is enforced.
- Maximum and RMS displacement are reported.
- A correction is rejected if it exceeds the user tolerance.
- Zero-reference errors are reported before and after correction.
- Train/evaluation spacing split prevents optimizing and scoring the same
  complete spacing set.

This is a numerical audit, not an RH proof.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import mpmath as mp
import numpy as np

try:
    from scipy.linalg import eigh
    from scipy.optimize import differential_evolution
    from scipy.stats import kstest
except ImportError as exc:
    raise SystemExit("Install dependencies with: pip install numpy scipy mpmath") from exc


# =============================================================================
# Parsing and files
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
# Xi and primitive resolvent
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
    center = mp.mpf("0.5")
    roots = []
    values = []

    for j in range(samples):
        angle = 2 * mp.pi * j / samples
        root = mp.e ** (1j * angle)
        roots.append(root)
        values.append(xi(center + radius * root))

    coefficients = []

    for order in range(maximum_order + 1):
        total = mp.mpc(0)
        for value, root in zip(values, roots):
            total += value * root ** (-order)
        coefficients.append(total / samples / radius ** order)

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
# Stable-node selection
# =============================================================================

def relative_depth_difference(
    current: np.ndarray,
    previous: np.ndarray,
) -> np.ndarray:
    count = min(len(current), len(previous))
    difference = np.full(len(current), np.inf, dtype=float)

    if count:
        denominator = np.maximum(np.abs(current[:count]), 1e-300)
        difference[:count] = (
            np.abs(current[:count] - previous[:count]) / denominator
        )

    return difference


def longest_consecutive_true_block(mask: np.ndarray) -> Tuple[int, int]:
    best_start = 0
    best_end = 0
    current_start = None

    for index, value in enumerate(mask):
        if value and current_start is None:
            current_start = index

        if (not value or index == len(mask) - 1) and current_start is not None:
            current_end = index + 1 if value and index == len(mask) - 1 else index

            if current_end - current_start > best_end - best_start:
                best_start = current_start
                best_end = current_end

            current_start = None

    return best_start, best_end


def stable_block(
    depth_spectra: Dict[int, np.ndarray],
    depth: int,
    stable_relative_tolerance: float,
    stable_absolute_tolerance: float,
    bulk_trim_fraction: float,
) -> Dict[str, object]:
    sorted_depths = sorted(depth_spectra)
    position = sorted_depths.index(depth)

    if position == 0:
        gamma = depth_spectra[depth]
        return {
            "gamma": gamma,
            "stable_mask": np.zeros(len(gamma), dtype=bool),
            "stable_start": 0,
            "stable_end": 0,
            "bulk_start": 0,
            "bulk_end": 0,
            "block": np.array([], dtype=float),
            "relative_difference": np.full(len(gamma), np.inf),
            "absolute_difference": np.full(len(gamma), np.inf),
        }

    previous_depth = sorted_depths[position - 1]
    gamma = depth_spectra[depth]
    previous = depth_spectra[previous_depth]

    count = min(len(gamma), len(previous))
    absolute = np.full(len(gamma), np.inf, dtype=float)
    relative = np.full(len(gamma), np.inf, dtype=float)

    if count:
        absolute[:count] = np.abs(gamma[:count] - previous[:count])
        relative[:count] = absolute[:count] / np.maximum(
            np.abs(gamma[:count]),
            1e-300,
        )

    mask = (
        (relative <= stable_relative_tolerance)
        | (absolute <= stable_absolute_tolerance)
    )

    start, end = longest_consecutive_true_block(mask)
    length = end - start
    trim = int(math.floor(bulk_trim_fraction * length))

    if length - 2 * trim < 4:
        trim = 0

    bulk_start = start + trim
    bulk_end = end - trim

    return {
        "gamma": gamma,
        "stable_mask": mask,
        "stable_start": start,
        "stable_end": end,
        "bulk_start": bulk_start,
        "bulk_end": bulk_end,
        "block": gamma[bulk_start:bulk_end],
        "relative_difference": relative,
        "absolute_difference": absolute,
    }


# =============================================================================
# Unfolding and GUE statistics
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

        window = max(3, min(9, len(spacings) // 3 * 2 + 1))
        kernel = np.ones(window) / window
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
        mean_spacing = np.mean(np.diff(unfolded))
        if mean_spacing > 0:
            unfolded = unfolded / mean_spacing

    return unfolded


def gue_spacing_cdf(values: np.ndarray) -> np.ndarray:
    values = np.maximum(np.asarray(values, dtype=float), 0.0)

    return (
        np.vectorize(math.erf)(2.0 * values / math.sqrt(np.pi))
        - (4.0 / np.pi)
        * values
        * np.exp(-4.0 * values * values / np.pi)
    )


def goe_spacing_cdf(values: np.ndarray) -> np.ndarray:
    values = np.maximum(np.asarray(values, dtype=float), 0.0)
    return 1.0 - np.exp(-np.pi * values * values / 4.0)


def poisson_spacing_cdf(values: np.ndarray) -> np.ndarray:
    values = np.maximum(np.asarray(values, dtype=float), 0.0)
    return 1.0 - np.exp(-values)


def spacing_ratios(spacings: np.ndarray) -> np.ndarray:
    spacings = np.asarray(spacings, dtype=float)

    if len(spacings) < 2:
        return np.array([], dtype=float)

    left = spacings[:-1]
    right = spacings[1:]
    denominator = np.maximum(left, right)
    mask = denominator > 0

    return np.minimum(left[mask], right[mask]) / denominator[mask]


def ratio_pdf(values: np.ndarray, beta: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)

    raw = (
        (values + values * values) ** beta
        / (1.0 + values + values * values) ** (1.0 + 1.5 * beta)
    )

    grid = np.linspace(0.0, 1.0, 30001)
    raw_grid = (
        (grid + grid * grid) ** beta
        / (1.0 + grid + grid * grid) ** (1.0 + 1.5 * beta)
    )
    normalization = np.trapezoid(raw_grid, grid)

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


def spectral_statistics(
    gamma: np.ndarray,
    unfolding_method: str,
    polynomial_degree: int,
) -> Dict[str, object]:
    gamma = np.sort(np.asarray(gamma, dtype=float))

    if len(gamma) < 4:
        return {
            "node_count": len(gamma),
            "spacing_count": max(0, len(gamma) - 1),
            "spacings": np.array([], dtype=float),
            "ratios": np.array([], dtype=float),
            "ks_gue": float("nan"),
            "ks_goe": float("nan"),
            "ks_poisson": float("nan"),
            "mean_ratio": float("nan"),
            "ratio_ks_gue": float("nan"),
            "ratio_ks_goe": float("nan"),
            "ratio_ks_poisson": float("nan"),
        }

    unfolded = unfold(gamma, unfolding_method, polynomial_degree)
    spacings = np.diff(unfolded)
    spacings = spacings[np.isfinite(spacings) & (spacings >= 0)]

    if len(spacings):
        spacings = spacings / max(np.mean(spacings), 1e-15)

    ratios = spacing_ratios(spacings)

    output = {
        "node_count": len(gamma),
        "spacing_count": len(spacings),
        "spacings": spacings,
        "ratios": ratios,
        "ks_gue": (
            float(kstest(spacings, gue_spacing_cdf).statistic)
            if len(spacings) >= 2
            else float("nan")
        ),
        "ks_goe": (
            float(kstest(spacings, goe_spacing_cdf).statistic)
            if len(spacings) >= 2
            else float("nan")
        ),
        "ks_poisson": (
            float(kstest(spacings, poisson_spacing_cdf).statistic)
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
    }

    return output


# =============================================================================
# Zero-location metrics
# =============================================================================

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
# Constrained smooth correction
# =============================================================================

def correction_basis(length: int, rank: int) -> np.ndarray:
    coordinate = np.linspace(-1.0, 1.0, length)
    columns = []

    for k in range(1, rank + 1):
        columns.append(np.sin(k * np.pi * (coordinate + 1.0) / 2.0))

    if not columns:
        return np.zeros((length, 0), dtype=float)

    basis = np.column_stack(columns)

    # Remove mean and normalize every mode.
    basis -= np.mean(basis, axis=0, keepdims=True)
    basis /= np.maximum(
        np.sqrt(np.mean(basis * basis, axis=0, keepdims=True)),
        1e-15,
    )

    return basis


def apply_micro_correction(
    gamma: np.ndarray,
    parameters: np.ndarray,
    basis: np.ndarray,
    maximum_relative_shift: float,
) -> Tuple[np.ndarray, np.ndarray]:
    raw = basis @ parameters
    raw = np.tanh(raw)

    relative_shift = maximum_relative_shift * raw
    corrected = gamma * (1.0 + relative_shift)

    # Enforce strict monotonicity while changing as little as possible.
    minimum_gap = max(np.min(np.diff(gamma)) * 1e-6, 1e-12)

    for i in range(1, len(corrected)):
        corrected[i] = max(corrected[i], corrected[i - 1] + minimum_gap)

    actual_relative_shift = (corrected - gamma) / np.maximum(gamma, 1e-300)

    return corrected, actual_relative_shift


def optimize_micro_correction(
    gamma: np.ndarray,
    reference: np.ndarray,
    unfolding_method: str,
    polynomial_degree: int,
    maximum_relative_shift: float,
    rank: int,
    seed: int,
    maxiter: int,
    popsize: int,
    zero_weight: float,
    smoothness_weight: float,
) -> Dict[str, object]:
    basis = correction_basis(len(gamma), rank)

    if basis.shape[1] == 0 or len(gamma) < 8:
        return {
            "success": False,
            "reason": "Not enough nodes or correction rank is zero.",
        }

    # Alternate spacing indices: optimize on even spacings and evaluate odd ones.
    training_spacing_indices = np.arange(len(gamma) - 1) % 2 == 0

    original_zero = zero_metrics(gamma, reference)
    original_statistics = spectral_statistics(
        gamma,
        unfolding_method,
        polynomial_degree,
    )

    def objective(parameters: np.ndarray) -> float:
        corrected, shifts = apply_micro_correction(
            gamma,
            parameters,
            basis,
            maximum_relative_shift,
        )

        unfolded = unfold(
            corrected,
            unfolding_method,
            polynomial_degree,
        )
        spacings = np.diff(unfolded)

        if len(spacings):
            spacings = spacings / max(np.mean(spacings), 1e-15)

        training_spacings = spacings[training_spacing_indices]

        if len(training_spacings) < 3:
            return 1e6

        ks_gue = float(
            kstest(training_spacings, gue_spacing_cdf).statistic
        )
        ratios = spacing_ratios(training_spacings)
        ratio_ks = (
            float(kstest(ratios, RATIO_GUE_CDF).statistic)
            if len(ratios) >= 2
            else 1.0
        )

        corrected_zero = zero_metrics(corrected, reference)

        zero_degradation = max(
            corrected_zero["zero_relative_rmse"]
            - original_zero["zero_relative_rmse"],
            0.0,
        )

        curvature = np.diff(shifts, n=2)
        smoothness = (
            float(np.sqrt(np.mean(curvature * curvature)))
            if len(curvature)
            else 0.0
        )

        shift_rms = float(np.sqrt(np.mean(shifts * shifts)))
        shift_limit_penalty = max(
            np.max(np.abs(shifts)) - maximum_relative_shift,
            0.0,
        )

        return (
            ks_gue
            + 0.35 * ratio_ks
            + zero_weight * zero_degradation
            + smoothness_weight * smoothness
            + 0.05 * shift_rms / max(maximum_relative_shift, 1e-15)
            + 1000.0 * shift_limit_penalty
        )

    result = differential_evolution(
        objective,
        [(-2.0, 2.0)] * basis.shape[1],
        maxiter=maxiter,
        popsize=popsize,
        seed=seed,
        polish=True,
        tol=1e-8,
        mutation=(0.5, 1.0),
        recombination=0.75,
        updating="immediate",
        workers=1,
        disp=False,
    )

    corrected, shifts = apply_micro_correction(
        gamma,
        np.asarray(result.x, dtype=float),
        basis,
        maximum_relative_shift,
    )

    corrected_statistics = spectral_statistics(
        corrected,
        unfolding_method,
        polynomial_degree,
    )
    corrected_zero = zero_metrics(corrected, reference)

    # Evaluation-only odd spacing subset.
    original_unfolded = unfold(gamma, unfolding_method, polynomial_degree)
    corrected_unfolded = unfold(corrected, unfolding_method, polynomial_degree)

    original_spacings = np.diff(original_unfolded)
    corrected_spacings = np.diff(corrected_unfolded)

    original_spacings /= max(np.mean(original_spacings), 1e-15)
    corrected_spacings /= max(np.mean(corrected_spacings), 1e-15)

    evaluation_mask = ~training_spacing_indices
    original_evaluation = original_spacings[evaluation_mask]
    corrected_evaluation = corrected_spacings[evaluation_mask]

    original_evaluation_ks = (
        float(kstest(original_evaluation, gue_spacing_cdf).statistic)
        if len(original_evaluation) >= 2
        else float("nan")
    )
    corrected_evaluation_ks = (
        float(kstest(corrected_evaluation, gue_spacing_cdf).statistic)
        if len(corrected_evaluation) >= 2
        else float("nan")
    )

    return {
        "success": True,
        "parameters": np.asarray(result.x, dtype=float),
        "corrected": corrected,
        "shifts": shifts,
        "objective": float(result.fun),
        "original_statistics": original_statistics,
        "corrected_statistics": corrected_statistics,
        "original_zero": original_zero,
        "corrected_zero": corrected_zero,
        "original_evaluation_ks_gue": original_evaluation_ks,
        "corrected_evaluation_ks_gue": corrected_evaluation_ks,
        "max_abs_relative_shift": float(np.max(np.abs(shifts))),
        "rms_relative_shift": float(np.sqrt(np.mean(shifts * shifts))),
    }


# =============================================================================
# Main
# =============================================================================

def parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument("--dps", type=int, default=220)
    parser.add_argument("--radius", type=float, default=4.0)
    parser.add_argument("--samples", type=int, default=4096)

    parser.add_argument(
        "--depths",
        type=parse_int_list,
        default=[20, 24, 28, 32, 36, 40],
    )
    parser.add_argument("--zero-reference-count", type=int, default=40)

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
    parser.add_argument("--bulk-trim-fraction", type=float, default=0.10)

    parser.add_argument(
        "--unfold-methods",
        default="rvm,polynomial,local",
    )
    parser.add_argument("--poly-degree", type=int, default=5)
    parser.add_argument("--minimum-spacings", type=int, default=12)

    parser.add_argument(
        "--enable-micro-correction",
        action="store_true",
    )
    parser.add_argument(
        "--max-relative-shift",
        type=float,
        default=1e-4,
    )
    parser.add_argument("--correction-rank", type=int, default=5)
    parser.add_argument("--correction-maxiter", type=int, default=100)
    parser.add_argument("--correction-popsize", type=int, default=12)
    parser.add_argument("--zero-weight", type=float, default=5000.0)
    parser.add_argument("--smoothness-weight", type=float, default=0.20)

    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument(
        "--out-prefix",
        default="rot_rh_resolvent_zero_preserving_gue",
    )

    return parser


def main() -> int:
    args = parser().parse_args()

    depths = sorted(set(args.depths))
    unfolding_methods = [
        value.strip()
        for value in args.unfold_methods.split(",")
        if value.strip()
    ]

    valid_methods = {"rvm", "polynomial", "local"}
    invalid = set(unfolding_methods) - valid_methods

    if invalid:
        raise SystemExit(f"Unknown unfolding methods: {sorted(invalid)}")

    maximum_depth = max(depths)
    moment_count = 2 * maximum_depth + 8
    phi_order = moment_count + 2
    xi_order = 2 * phi_order + 2

    prefix = Path(args.out_prefix).expanduser().resolve()
    prefix.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 126)
    print("ROT-RH / ZERO-PRESERVING DEPTH-STABLE GUE AUDIT")
    print("=" * 126)
    print(f"dps                       : {args.dps}")
    print(f"radius/samples            : {args.radius}/{args.samples}")
    print(f"depths                    : {depths}")
    print(f"moment count              : {moment_count}")
    print(f"stable relative tolerance : {args.stable_relative_tolerance}")
    print(f"stable absolute tolerance : {args.stable_absolute_tolerance}")
    print(f"unfolding methods         : {unfolding_methods}")
    print(f"micro correction          : {args.enable_micro_correction}")
    print(f"max relative shift        : {args.max_relative_shift}")
    print("=" * 126)

    start_time = time.time()
    mp.mp.dps = args.dps
    tolerance = mp.power(10, -(args.dps // 2))

    print("[1/7] Computing primitive Xi-resolvent moments...")
    xi_coefficients = xi_taylor_cauchy(
        xi_order,
        mp.mpf(str(args.radius)),
        args.samples,
    )
    phi = phi_from_xi_coefficients(xi_coefficients, phi_order)
    moments = primitive_resolvent_moments(phi, moment_count)

    print("[2/7] Building Jacobi depth flow...")
    depth_results: Dict[int, JacobiResult] = {}
    depth_spectra: Dict[int, np.ndarray] = {}

    for depth in depths:
        result = stieltjes_jacobi(moments, depth, tolerance)
        depth_results[depth] = result
        depth_spectra[depth] = reconstructed_ordinates(result)
        print(
            f"  depth={depth:3d} nodes={len(depth_spectra[depth]):3d} "
            f"orth={float(result.orthogonality_defect):.3e}"
        )

    print("[3/7] Computing reference zeros...")
    reference_zeros = np.array(
        [
            float(mp.im(mp.zetazero(index)))
            for index in range(1, args.zero_reference_count + 1)
        ],
        dtype=float,
    )

    print("[4/7] Selecting depth-stable consecutive bulk blocks...")
    stable_rows = []
    node_rows = []
    statistics_rows = []
    spacing_rows = []
    correction_rows = []
    correction_parameter_rows = []

    stable_data: Dict[int, Dict[str, object]] = {}

    for depth in depths:
        data = stable_block(
            depth_spectra,
            depth,
            args.stable_relative_tolerance,
            args.stable_absolute_tolerance,
            args.bulk_trim_fraction,
        )
        stable_data[depth] = data

        block = np.asarray(data["block"], dtype=float)
        zero_result = zero_metrics(block, reference_zeros[data["bulk_start"]:])

        stable_rows.append({
            "depth": depth,
            "total_nodes": len(depth_spectra[depth]),
            "stable_start_index": int(data["stable_start"]) + 1,
            "stable_end_index": int(data["stable_end"]),
            "stable_count": int(data["stable_end"] - data["stable_start"]),
            "bulk_start_index": int(data["bulk_start"]) + 1,
            "bulk_end_index": int(data["bulk_end"]),
            "bulk_count": len(block),
            **zero_result,
        })

        for index, value in enumerate(depth_spectra[depth]):
            node_rows.append({
                "depth": depth,
                "index": index + 1,
                "gamma": float(value),
                "is_stable": bool(data["stable_mask"][index]),
                "is_bulk": bool(
                    data["bulk_start"] <= index < data["bulk_end"]
                ),
                "relative_depth_difference": float(
                    data["relative_difference"][index]
                ),
                "absolute_depth_difference": float(
                    data["absolute_difference"][index]
                ),
                "target_zero": (
                    float(reference_zeros[index])
                    if index < len(reference_zeros)
                    else float("nan")
                ),
            })

        for method in unfolding_methods:
            stats = spectral_statistics(block, method, args.poly_degree)

            statistics_rows.append({
                "mode": "canonical",
                "depth": depth,
                "unfolding": method,
                "stable_count": int(data["stable_end"] - data["stable_start"]),
                "bulk_count": len(block),
                **{
                    key: value
                    for key, value in stats.items()
                    if key not in {"spacings", "ratios"}
                },
                **zero_result,
            })

            for index, value in enumerate(stats["spacings"]):
                spacing_rows.append({
                    "mode": "canonical",
                    "depth": depth,
                    "unfolding": method,
                    "index": index + 1,
                    "spacing": float(value),
                })

    print("[5/7] Running optional constrained micro-correction...")
    if args.enable_micro_correction:
        for depth in depths:
            data = stable_data[depth]
            block = np.asarray(data["block"], dtype=float)

            if len(block) < max(8, args.minimum_spacings + 1):
                continue

            reference_slice = reference_zeros[
                int(data["bulk_start"]):
                int(data["bulk_start"]) + len(block)
            ]

            for method_index, method in enumerate(unfolding_methods):
                correction = optimize_micro_correction(
                    gamma=block,
                    reference=reference_slice,
                    unfolding_method=method,
                    polynomial_degree=args.poly_degree,
                    maximum_relative_shift=args.max_relative_shift,
                    rank=args.correction_rank,
                    seed=args.seed + 1000 * depth + method_index,
                    maxiter=args.correction_maxiter,
                    popsize=args.correction_popsize,
                    zero_weight=args.zero_weight,
                    smoothness_weight=args.smoothness_weight,
                )

                if not correction.get("success"):
                    continue

                corrected_stats = correction["corrected_statistics"]
                corrected_zero = correction["corrected_zero"]

                statistics_rows.append({
                    "mode": "micro_corrected",
                    "depth": depth,
                    "unfolding": method,
                    "stable_count": int(
                        data["stable_end"] - data["stable_start"]
                    ),
                    "bulk_count": len(block),
                    **{
                        key: value
                        for key, value in corrected_stats.items()
                        if key not in {"spacings", "ratios"}
                    },
                    **corrected_zero,
                    "max_abs_relative_shift": correction[
                        "max_abs_relative_shift"
                    ],
                    "rms_relative_shift": correction[
                        "rms_relative_shift"
                    ],
                    "original_evaluation_ks_gue": correction[
                        "original_evaluation_ks_gue"
                    ],
                    "corrected_evaluation_ks_gue": correction[
                        "corrected_evaluation_ks_gue"
                    ],
                    "optimization_objective": correction["objective"],
                })

                for index, (
                    original,
                    corrected,
                    shift,
                ) in enumerate(
                    zip(
                        block,
                        correction["corrected"],
                        correction["shifts"],
                    )
                ):
                    correction_rows.append({
                        "depth": depth,
                        "unfolding": method,
                        "index": int(data["bulk_start"]) + index + 1,
                        "original_gamma": float(original),
                        "corrected_gamma": float(corrected),
                        "relative_shift": float(shift),
                        "target_zero": (
                            float(reference_slice[index])
                            if index < len(reference_slice)
                            else float("nan")
                        ),
                    })

                for index, value in enumerate(correction["parameters"]):
                    correction_parameter_rows.append({
                        "depth": depth,
                        "unfolding": method,
                        "parameter_index": index,
                        "value": float(value),
                    })

                for index, value in enumerate(
                    corrected_stats["spacings"]
                ):
                    spacing_rows.append({
                        "mode": "micro_corrected",
                        "depth": depth,
                        "unfolding": method,
                        "index": index + 1,
                        "spacing": float(value),
                    })
    else:
        print("      disabled; canonical spectrum remains untouched.")

    print("[6/7] Ranking canonical and corrected GUE results...")
    canonical_eligible = [
        row
        for row in statistics_rows
        if row["mode"] == "canonical"
        and int(row["spacing_count"]) >= args.minimum_spacings
        and np.isfinite(float(row["ks_gue"]))
    ]

    best_canonical = (
        min(canonical_eligible, key=lambda row: float(row["ks_gue"]))
        if canonical_eligible
        else None
    )

    corrected_eligible = [
        row
        for row in statistics_rows
        if row["mode"] == "micro_corrected"
        and int(row["spacing_count"]) >= args.minimum_spacings
        and np.isfinite(float(row["ks_gue"]))
        and float(row.get("max_abs_relative_shift", np.inf))
            <= args.max_relative_shift * 1.0001
    ]

    best_corrected = (
        min(corrected_eligible, key=lambda row: float(row["ks_gue"]))
        if corrected_eligible
        else None
    )

    print("[7/7] Writing outputs...")
    report = {
        "scientific_status": (
            "canonical stable-bulk GUE audit plus optional constrained "
            "diagnostic correction; not an RH proof"
        ),
        "canonical_operator_modified": False,
        "micro_correction_is_canonical": False,
        "args": vars(args),
        "best_canonical": best_canonical,
        "best_micro_corrected": best_corrected,
        "runtime_seconds": time.time() - start_time,
    }

    save_csv(Path(str(prefix) + "_stable_blocks.csv"), stable_rows)
    save_csv(Path(str(prefix) + "_nodes.csv"), node_rows)
    save_csv(Path(str(prefix) + "_statistics.csv"), statistics_rows)
    save_csv(Path(str(prefix) + "_spacings.csv"), spacing_rows)
    save_csv(Path(str(prefix) + "_corrections.csv"), correction_rows)
    save_csv(
        Path(str(prefix) + "_correction_parameters.csv"),
        correction_parameter_rows,
    )

    Path(str(prefix) + "_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print()
    print("=" * 126)
    print("FINAL ZERO-PRESERVING DEPTH-STABLE GUE RESULT")
    print("=" * 126)

    if best_canonical:
        print(f"canonical depth             : {best_canonical['depth']}")
        print(f"canonical unfolding         : {best_canonical['unfolding']}")
        print(f"canonical stable bulk nodes : {best_canonical['bulk_count']}")
        print(f"canonical spacings          : {best_canonical['spacing_count']}")
        print(f"canonical KS GUE            : {best_canonical['ks_gue']:.8f}")
        print(f"canonical KS GOE            : {best_canonical['ks_goe']:.8f}")
        print(f"canonical KS Poisson        : {best_canonical['ks_poisson']:.8f}")
        print(f"canonical mean ratio        : {best_canonical['mean_ratio']:.8f}")
        print(
            f"canonical zero relRMSE      : "
            f"{best_canonical['zero_relative_rmse']:.8e}"
        )
    else:
        print("No canonical stable block met the minimum-spacing threshold.")

    if best_corrected:
        print("-" * 126)
        print(f"corrected depth             : {best_corrected['depth']}")
        print(f"corrected unfolding         : {best_corrected['unfolding']}")
        print(f"corrected KS GUE            : {best_corrected['ks_gue']:.8f}")
        print(f"corrected KS GOE            : {best_corrected['ks_goe']:.8f}")
        print(f"corrected KS Poisson        : {best_corrected['ks_poisson']:.8f}")
        print(f"corrected mean ratio        : {best_corrected['mean_ratio']:.8f}")
        print(
            f"corrected zero relRMSE      : "
            f"{best_corrected['zero_relative_rmse']:.8e}"
        )
        print(
            f"maximum relative shift      : "
            f"{best_corrected['max_abs_relative_shift']:.3e}"
        )
        print(
            f"evaluation-only KS GUE      : "
            f"{best_corrected['corrected_evaluation_ks_gue']:.8f}"
        )

    canonical_promising = (
        best_canonical is not None
        and float(best_canonical["ks_gue"])
            < float(best_canonical["ks_goe"])
        and float(best_canonical["ks_gue"])
            < float(best_canonical["ks_poisson"])
        and int(best_canonical["spacing_count"]) >= args.minimum_spacings
    )

    print("-" * 126)
    print(
        "PROMISING CANONICAL STABLE-BULK GUE SIGNAL"
        if canonical_promising
        else "NO CONVINCING CANONICAL GUE SIGNAL YET"
    )

    if best_corrected:
        print(
            "MICRO-CORRECTION IS DIAGNOSTIC ONLY; "
            "DO NOT TREAT IT AS UNFITTED OPERATOR EVIDENCE"
        )

    print(f"runtime_seconds             : {time.time() - start_time:.2f}")
    print(f"outputs                     : {prefix}_*.csv/json")
    print("=" * 126)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
