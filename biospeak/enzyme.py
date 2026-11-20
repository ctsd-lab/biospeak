"""Formal specifications for enzymatic output flux."""

import math

import numpy as np
from pacti.contracts import PolyhedralIoContract


def binding_complex(first, second, K):
    """Physical root of K C = (first - C)(second - C)."""
    first, second = np.broadcast_arrays(
        np.asarray(first, dtype=float), np.asarray(second, dtype=float)
    )
    if K <= 0 or np.any(first < 0) or np.any(second < 0):
        raise ValueError("K must be positive and concentrations nonnegative.")
    total = first + second + K
    discriminant = (first - second) ** 2 + 2 * K * (first + second) + K**2
    return 2 * first * second / (total + np.sqrt(discriminant))


def log_sensitivity(enzyme, substrate, K):
    """Logarithmic sensitivities with respect to enzyme and substrate totals."""
    complex_level = binding_complex(enzyme, substrate, K)
    free_enzyme = np.asarray(enzyme) - complex_level
    free_substrate = np.asarray(substrate) - complex_level
    denominator = K + free_enzyme + free_substrate
    return (K + free_substrate) / denominator, (K + free_enzyme) / denominator


def enzyme_contract(K, kcat, regime, margins=(math.log(2), math.log(2))):
    """Construct the flux contract for dominance by K, substrate, or enzyme.

    Inputs are z_u = log(E_tot), z_x = log(S_tot); output is z_y = log(dP/dt).
    The margins compare the dominant quantity to the other two quantities.
    """
    if K <= 0 or kcat <= 0 or any(delta <= 0 for delta in margins):
        raise ValueError("K, kcat, and dominance margins must be positive.")
    first, second = margins
    log_K, log_kcat = math.log(K), math.log(kcat)
    if regime == "K":
        assumptions = [
            {"coefficients": {"z_u": 1}, "constant": log_K - first},
            {"coefficients": {"z_x": 1}, "constant": log_K - second},
        ]
        slope = {"z_u": 1, "z_x": 1}
        offset = log_kcat - log_K
        eta = math.log1p(math.exp(-first) + math.exp(-second))
    elif regime in ("substrate", "enzyme"):
        dominant, other = ("z_x", "z_u") if regime == "substrate" else ("z_u", "z_x")
        assumptions = [
            {"coefficients": {other: 1, dominant: -1}, "constant": -first},
            {"coefficients": {dominant: -1}, "constant": -log_K - second},
        ]
        slope = {other: 1}
        offset = log_kcat
        eta = -math.log((1 - math.exp(-first)) / (1 + math.exp(-second)))
    else:
        raise ValueError("regime must be 'K', 'substrate', or 'enzyme'.")
    return PolyhedralIoContract.from_dict(
        {
            "input_vars": ["z_u", "z_x"],
            "output_vars": ["z_y"],
            "assumptions": assumptions,
            "guarantees": [
                {
                    "coefficients": {"z_y": 1, **{k: -v for k, v in slope.items()}},
                    "constant": offset,
                },
                {"coefficients": {"z_y": -1, **slope}, "constant": eta - offset},
            ],
        }
    )
