"""Polyhedral contracts for the characterized Cello NOR and AND gates."""

import csv
import json
import math
from itertools import combinations, product
from pathlib import Path

import numpy as np
from pacti.contracts import PolyhedralIoContract

TOL = 1e-9
MODES = ("low", "transition", "high")


def region_bounds(gate, modes):
    if gate["type"] == "NOR":
        thresholds = [(gate["IL"], gate["IH"])]
    else:
        thresholds = [(gate["IL1"], gate["IH1"]), (gate["IL2"], gate["IH2"])]
    return [
        (0.0, lo) if mode == "low" else (hi, math.inf) if mode == "high" else (lo, hi)
        for mode, (lo, hi) in zip(modes, thresholds)
    ]


def terms(gate):
    if gate["type"] == "NOR":
        return [(gate["ymax"], (0,)), (gate["ymin"], (1,))], [(1.0, (0,)), (1.0, (1,))]
    exponents = [(0, 0), (1, 0), (0, 1), (1, 1)]
    return [(gate["ymin"] if a != (1, 1) else gate["ymax"], a) for a in exponents], [
        (1.0, a) for a in exponents
    ]


def sum_bounds(monomials, reference, q_bounds):
    c0, a0 = reference
    lower_sum = upper_sum = 0.0
    for coefficient, exponent in monomials:
        lower = upper = coefficient / c0
        for a, b, (lo, hi) in zip(exponent, a0, q_bounds):
            difference = a - b
            if difference == 1:
                lower *= lo
                upper *= hi
            elif difference == -1:
                lower *= 0.0 if math.isinf(hi) else 1 / hi
                upper *= 1 / lo
            else:
                assert difference == 0
        assert math.isfinite(upper)
        lower_sum += lower
        upper_sum += upper
    assert lower_sum >= 1.0 - 1e-12
    return lower_sum, upper_sum


def row(coefficients, constant):
    return {"coefficients": coefficients, "constant": float(constant)}


def construct_monomial_ratio(gate, modes):
    bounds = region_bounds(gate, modes)
    thresholds = [gate["K"]] if gate["type"] == "NOR" else [gate["K1"], gate["K2"]]
    input_names = ["zu"] if gate["type"] == "NOR" else ["zu1", "zu2"]
    q_bounds = [
        ((lo / k) ** gate["n"], (hi / k) ** gate["n"])
        for (lo, hi), k in zip(bounds, thresholds)
    ]
    numerator, denominator = terms(gate)
    if gate["type"] == "NOR":
        exponent_p = (int(modes[0] == "high"),)
        exponent_q = (int(modes[0] != "low"),)
    else:
        exponent_p = tuple(int(mode != "low") for mode in modes)
        exponent_q = tuple(int(mode == "high") for mode in modes)
    reference_p = next(t for t in numerator if t[1] == exponent_p)
    reference_q = next(t for t in denominator if t[1] == exponent_q)
    lp, up = sum_bounds(numerator, reference_p, q_bounds)
    lq, uq = sum_bounds(denominator, reference_q, q_bounds)
    slopes = [(a - b) * gate["n"] for a, b in zip(exponent_p, exponent_q)]
    center = math.log(reference_p[0] / reference_q[0]) - sum(
        s * math.log(k) for s, k in zip(slopes, thresholds)
    )
    lower_constant = center + math.log(lp / uq)
    upper_constant = center + math.log(up / lq)
    assumptions = []
    for name, (lo, hi) in zip(input_names, bounds):
        if lo > 0:
            assumptions.append(row({name: -1.0}, -math.log(lo)))
        if math.isfinite(hi):
            assumptions.append(row({name: 1.0}, math.log(hi)))
    guarantees = [
        row(
            {"zy": -1.0, **{name: s for name, s in zip(input_names, slopes) if s}},
            -lower_constant,
        ),
        row(
            {"zy": 1.0, **{name: -s for name, s in zip(input_names, slopes) if s}},
            upper_constant,
        ),
        row({"zy": -1.0}, -math.log(gate["ymin"])),
        row({"zy": 1.0}, math.log(gate["ymax"])),
    ]
    data = dict(
        input_vars=input_names,
        output_vars=["zy"],
        assumptions=assumptions,
        guarantees=guarantees,
    )
    contract = PolyhedralIoContract.from_dict(data)
    assert not (contract.a | contract.g).is_empty()
    return (
        contract,
        data,
        dict(
            L_P=lp,
            U_P=up,
            L_Q=lq,
            U_Q=uq,
            log_slopes=slopes,
            lower_constant=lower_constant,
            upper_constant=upper_constant,
        ),
    )


def log_sum_data(gate, monomials, reference, points):
    """Logarithm and gradient after a common reference monomial is removed."""
    thresholds = [gate["K"]] if gate["type"] == "NOR" else [gate["K1"], gate["K2"]]
    exponents = np.array([gate["n"] * (np.array(a) - reference) for _, a in monomials])
    offsets = np.array([math.log(c) for c, _ in monomials]) - exponents @ np.log(
        thresholds
    )
    arguments = np.asarray(points) @ exponents.T + offsets
    largest = arguments.max(axis=-1, keepdims=True)
    weights = np.exp(arguments - largest)
    total = weights.sum(axis=-1, keepdims=True)
    return (largest + np.log(total)).ravel(), weights / total @ exponents


def convex_sum_bounds(gate, monomials, modes, bounds):
    """Global tangents and affine upper bounds on each given input region."""
    dimension = len(bounds)
    reference = np.full(dimension, int(modes[0] == "high"))
    if modes[0] == "transition":
        vertices = np.log(np.array(list(product(*bounds))))
    else:
        # After normalization both sums increase toward the finite corner.
        vertices = np.log([[hi if modes[0] == "low" else lo for lo, hi in bounds]])
    values, gradients = log_sum_data(gate, monomials, reference, vertices)
    tangents = [
        dict(slopes=g.tolist(), constant=float(value - g @ vertex))
        for vertex, value, g in zip(vertices, values, gradients)
    ]
    if modes[0] != "transition":
        upper = [dict(slopes=[0.0] * dimension, constant=float(values[0]))]
    else:
        upper = []
        augmented = np.column_stack([vertices, np.ones(len(vertices))])
        for indices in combinations(range(len(vertices)), dimension + 1):
            try:
                plane = np.linalg.solve(augmented[list(indices)], values[list(indices)])
            except np.linalg.LinAlgError:
                continue
            gap = float(np.max(values - augmented @ plane))
            if gap > TOL:
                continue
            plane[-1] += max(0.0, gap)
            if not any(
                np.allclose(plane, candidate, rtol=0.0, atol=1e-11)
                for candidate in upper
            ):
                upper.append(plane)
        assert upper
        upper = [dict(slopes=p[:-1].tolist(), constant=float(p[-1])) for p in upper]
    return dict(lower=tangents, upper=upper, normalization_exponent=reference.tolist())


def construct(gate, modes):
    """Refine the monomial ratio contract using valid log sum bounds."""
    _, baseline, constants = construct_monomial_ratio(gate, modes)
    bounds = region_bounds(gate, modes)
    numerator, denominator = terms(gate)
    a = convex_sum_bounds(gate, numerator, modes, bounds)
    b = convex_sum_bounds(gate, denominator, modes, bounds)
    data = {**baseline, "guarantees": list(baseline["guarantees"])}
    for lower_a in a["lower"]:
        for upper_b in b["upper"]:
            slopes = np.array(lower_a["slopes"]) - upper_b["slopes"]
            constant = lower_a["constant"] - upper_b["constant"]
            data["guarantees"].append(
                row({"zy": -1.0, **dict(zip(data["input_vars"], slopes))}, -constant)
            )
    for upper_a in a["upper"]:
        for lower_b in b["lower"]:
            slopes = np.array(upper_a["slopes"]) - lower_b["slopes"]
            constant = upper_a["constant"] - lower_b["constant"]
            data["guarantees"].append(
                row({"zy": 1.0, **dict(zip(data["input_vars"], -slopes))}, constant)
            )
    contract = PolyhedralIoContract.from_dict(data, simplify=False)
    assert not (contract.a | contract.g).is_empty()
    constants.update(
        monomial_ratio_contract=baseline, numerator_bounds=a, denominator_bounds=b
    )
    return contract, data, constants


def response(gate, inputs):
    if gate["type"] == "NOR":
        q = (inputs[0] / gate["K"]) ** gate["n"]
        return gate["ymin"] + (gate["ymax"] - gate["ymin"]) / (1 + q)
    q1 = (inputs[0] / gate["K1"]) ** gate["n"]
    q2 = (inputs[1] / gate["K2"]) ** gate["n"]
    return gate["ymin"] + (gate["ymax"] - gate["ymin"]) * q1 / (1 + q1) * q2 / (1 + q2)


def load_gates():
    """Read the 20 NOR and two AND gate characterizations."""
    directory = Path(__file__).parent / "data" / "cello"
    with (directory / "gate_parameters.csv").open(
        encoding="utf-8", newline=""
    ) as stream:
        gates = list(csv.DictReader(stream))
    for gate in gates:
        for key in ["ymin", "ymax", "K", "n", "IL", "IH"]:
            gate[key] = float(gate[key])
    return gates + json.loads(
        (directory / "and_gate_parameters.json").read_text(encoding="utf-8")
    )


def build_library(gates=None):
    """Construct low, transition, and high contracts for each gate."""
    if gates is None:
        gates = load_gates()
    return {
        gate["name"]: {
            mode: construct(gate, (mode,) * (1 if gate["type"] == "NOR" else 2))[0]
            for mode in MODES
        }
        for gate in gates
    }
