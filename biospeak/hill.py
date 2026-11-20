"""Equilibrium contracts for proportional activation and repression."""

import math

import numpy as np
from pacti.contracts import PolyhedralIoContract

from .contracts import interval, row


def coefficient_bounds(bounds, protein_only=False):
    if protein_only:
        return bounds["k"][0] / bounds["delta"][1], bounds["k"][1] / bounds["delta"][0]
    return (
        bounds["k"][0]
        * bounds["ktl"][0]
        / (bounds["delta_m"][1] * bounds["delta_p"][1]),
        bounds["k"][1]
        * bounds["ktl"][1]
        / (bounds["delta_m"][0] * bounds["delta_p"][0]),
    )


def log_hill(z, K, n, positive):
    qlog = n * (np.asarray(z) - math.log(K))
    return (qlog if positive else 0.0) - np.logaddexp(0.0, qlog)


def log_hill_slope(z, K, n, positive):
    q = math.exp(n * (z - math.log(K)))
    return n * ((1.0 if positive else 0.0) - q / (1 + q))


def hill_contracts(
    n,
    positive,
    K_bounds,
    coefficients,
    kind="endpoint",
    input_name="z_s",
    output_name="z_p",
    dna_name="z_d",
    edges=(1.0, 5.0, 20.0, 100.0),
    dna_bounds=(0.5, 2.0),
):
    """DNA is an input, with coefficient one in every guarantee."""
    cmin, cmax = coefficients
    Kmin, Kmax = K_bounds
    lower_K, upper_K = (Kmax, Kmin) if positive else (Kmin, Kmax)
    entries = []
    for name, slo, shi in zip(("Low", "Transition", "High"), edges[:-1], edges[1:]):
        a, b = math.log(slo), math.log(shi)
        if kind == "endpoint":
            f0, f1 = [float(log_hill(z, lower_K, n, positive)) for z in (a, b)]
            slope = (f1 - f0) / (b - a)
            lower = [(slope, math.log(cmin) + f0 - slope * a)]
            upper = []
            for z in (a, b):
                derivative = log_hill_slope(z, upper_K, n, positive)
                upper.append(
                    (
                        derivative,
                        math.log(cmax)
                        + float(log_hill(z, upper_K, n, positive))
                        - derivative * z,
                    )
                )
        elif kind in ("reference", "both_references"):
            references = (
                (0, 1) if kind == "both_references" else ((0,) if positive else (1,))
            )
            lower, upper = [], []
            qmin, qmax = (slo / Kmax) ** n, (shi / Kmin) ** n
            for reference in references:
                L, U = (
                    (1 + qmin, 1 + qmax)
                    if reference == 0
                    else (1 + 1 / qmax, 1 + 1 / qmin)
                )
                slope = n * (int(positive) - reference)
                offsets = [-slope * math.log(k) for k in K_bounds]
                lower.append((slope, math.log(cmin) + min(offsets) - math.log(U)))
                upper.append((slope, math.log(cmax) + max(offsets) - math.log(L)))
        elif kind == "interval":
            lower = [
                (
                    0.0,
                    math.log(cmin)
                    + float(log_hill(a if positive else b, lower_K, n, positive)),
                )
            ]
            upper = [
                (
                    0.0,
                    math.log(cmax)
                    + float(log_hill(b if positive else a, upper_K, n, positive)),
                )
            ]
        else:
            raise ValueError(kind)
        guarantees = [
            row({input_name: s, dna_name: 1.0, output_name: -1.0}, -c) for s, c in lower
        ]
        guarantees += [
            row({input_name: -s, dna_name: -1.0, output_name: 1.0}, c) for s, c in upper
        ]
        entries.append(
            dict(
                region=name,
                input_interval=[slo, shi],
                n=n,
                positive=positive,
                kind=kind,
                inputs=[input_name, dna_name],
                outputs=[output_name],
                assumptions=interval(input_name, math.log(slo), math.log(shi))
                + interval(dna_name, *np.log(dna_bounds)),
                guarantees=guarantees,
                lower=lower,
                upper=upper,
            )
        )
    return entries


def hill_limits(entry, s, dna=1.0):
    z = np.log(np.asarray(s))
    lo = np.maximum.reduce([slope * z + constant for slope, constant in entry["lower"]])
    hi = np.minimum.reduce([slope * z + constant for slope, constant in entry["upper"]])
    return dna * np.exp(lo), dna * np.exp(hi)


def to_contract(entry, extra=()):
    """Construct the Pacti contract for one input interval."""
    return PolyhedralIoContract.from_dict(
        dict(
            input_vars=entry["inputs"],
            output_vars=entry["outputs"],
            assumptions=entry["assumptions"] + list(extra),
            guarantees=entry["guarantees"],
        ),
        simplify=False,
    )
