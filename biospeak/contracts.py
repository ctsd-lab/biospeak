"""Polyhedral contracts from positive monomial sums and equilibrium balances."""

import math

import numpy as np
import sympy as sp
from pacti.contracts import PolyhedralIoContract
from pacti.iocontract import Var
from pacti.terms.polyhedra.polyhedra import PolyhedralTerm, PolyhedralTermList

from ._polyhedra import check_interface_assumption


def refines(candidate, specification):
    """Check assumption coverage and output guarantees by linear programming."""
    if not candidate.shares_io_with(specification):
        raise ValueError("The contracts must have the same input and output ports.")
    if specification.a.is_empty() or (specification.a | candidate.g).is_empty():
        return False
    return all(
        check_interface_assumption(context, term)["established"]
        for context, terms in (
            (specification.a, candidate.a),
            (specification.a | candidate.g, specification.g),
        )
        for term in terms.terms
    )


def row(coefficients, constant, origin="", meaning="", symbolic=""):
    coefficients = {str(k): float(v) for k, v in coefficients.items() if v != 0}
    if not math.isfinite(float(constant)):
        raise ValueError("A finite bound is required.")
    return dict(
        coefficients=coefficients,
        constant=float(constant),
        origin=origin,
        meaning=meaning,
        symbolic=str(symbolic),
    )


def interval(var, lo, hi, origin=""):
    if lo > hi:
        raise ValueError("Empty interval.")
    return [
        row({var: -1}, -lo, origin, "lower bound"),
        row({var: 1}, hi, origin, "upper bound"),
    ]


def positive_terms(expression, variables):
    terms = sp.Poly(sp.expand(expression), *variables).terms()
    result = []
    for exponents, coefficient in terms:
        c = float(coefficient)
        if c <= 0:
            raise ValueError("Positive polynomial coefficients are required.")
        result.append((np.array(exponents, dtype=float), c))
    return result


def monomial_sup(exponents, coefficient, domains):
    value = coefficient
    for exponent, (lo, hi) in zip(exponents, domains):
        if exponent == 0:
            continue
        edge = hi if exponent > 0 else lo
        if edge == 0 and exponent < 0:
            return math.inf
        value *= edge**exponent
    return float(value)


def sum_enclosure(expression, variables, domains):
    """Select a monomial anchor using finite ratio bounds over the declared region."""
    terms = positive_terms(expression, variables)
    candidates = []
    for anchor, (a, c) in enumerate(terms):
        ratios = [
            monomial_sup(b - a, d / c, domains)
            for j, (b, d) in enumerate(terms)
            if j != anchor
        ]
        if all(math.isfinite(v) for v in ratios):
            candidates.append((sum(ratios), anchor, ratios))
    if not candidates:
        raise ValueError("This region provides no finite monomial enclosure.")
    rho, idx, ratios = min(candidates, key=lambda item: item[0])
    exponent, coefficient = terms[idx]
    return dict(
        exponents=exponent,
        log_constant=math.log(coefficient),
        width=math.log1p(rho),
        expression=str(expression),
        anchor=str(
            coefficient * sp.prod(v ** int(e) for v, e in zip(variables, exponent))
        ),
        ratio_bounds=ratios,
    )


def quotient_enclosure(expression, variables, domains):
    """Bound a positive rational expression by affine functions of log inputs."""
    numerator, denominator = sp.fraction(sp.factor(expression))
    slope = np.zeros(len(variables))
    lo = 0.0
    hi = 0.0
    trace = []
    for side, part in [("numerator", numerator), ("denominator", denominator)]:
        constant, factors = sp.factor_list(part, *variables)
        value = float(constant)
        if value <= 0:
            raise ValueError("Positive numerator and denominator are required.")
        sign = 1 if side == "numerator" else -1
        lo += sign * math.log(value)
        hi += sign * math.log(value)
        for factor, multiplicity in factors:
            bound = sum_enclosure(factor, variables, domains)
            slope += sign * multiplicity * bound["exponents"]
            base = multiplicity * bound["log_constant"]
            width = multiplicity * bound["width"]
            if sign == 1:
                lo += base
                hi += base + width
            else:
                lo -= base + width
                hi -= base
            trace.append(
                dict(
                    side=side,
                    multiplicity=int(multiplicity),
                    **{k: v for k, v in bound.items() if k != "exponents"},
                )
            )
    return dict(
        slope=slope,
        lower_offset=lo,
        upper_offset=hi,
        trace=trace,
        expression=str(expression),
    )


def affine_rows(enclosure, output, ports, origin):
    # ports: q coordinate -> (log interface variable, elasticity).
    upper = {output: 1.0}
    lower = {output: -1.0}
    for slope, (var, elasticity) in zip(enclosure["slope"], ports):
        upper[var] = upper.get(var, 0) - slope * elasticity
        lower[var] = lower.get(var, 0) + slope * elasticity
    return [
        row(
            upper,
            enclosure["upper_offset"],
            origin,
            "positive sum upper enclosure",
            enclosure["expression"],
        ),
        row(
            lower,
            -enclosure["lower_offset"],
            origin,
            "positive sum lower enclosure",
            enclosure["expression"],
        ),
    ]


def normalized_dict(contract):
    """Canonical halfspaces in already aligned, dimensionless coordinates."""
    out = {k: list(contract[k]) for k in ["input_vars", "output_vars"]}
    for name in ["assumptions", "guarantees"]:
        out[name] = []
        for t in contract[name]:
            b = float(t["constant"])
            coeff = {str(k): float(v) for k, v in t["coefficients"].items()}
            scale = (
                (abs(b) if b else max([abs(v) for v in coeff.values()] + [1.0]))
                if coeff
                else 1.0
            )
            out[name].append(
                dict(
                    coefficients={k: v / scale for k, v in coeff.items()},
                    constant=b / scale,
                )
            )
    return out


def prepare_contract(data):
    """Construct a contract without rounding its machine coefficients."""
    if isinstance(data, PolyhedralIoContract):
        data = data.to_machine_dict()
    return PolyhedralIoContract.from_dict(normalized_dict(data), simplify=False)


def prepare_terms(rows):
    data = normalized_dict(
        dict(input_vars=[], output_vars=[], assumptions=[], guarantees=rows)
    )
    return PolyhedralTermList(
        [
            PolyhedralTerm(
                {Var(k): v for k, v in r["coefficients"].items()}, r["constant"]
            )
            for r in data["guarantees"]
        ]
    )


def finite_bounds(contract, variable):
    """Ask Pacti for consistency and both extrema of the named variable."""
    if (contract.a | contract.g).is_empty():
        raise ValueError("The composed assumptions and guarantees are inconsistent.")
    lower, upper = contract.get_variable_bounds(variable)
    if lower is None or upper is None:
        raise ValueError("Pacti returned an unbounded output interval.")
    if not np.isfinite([lower, upper]).all():
        raise ValueError("Pacti returned a nonfinite output interval.")
    return float(lower), float(upper)


def compose_checked(components, external_rows):
    """Discharge assumptions before admitting each component guarantee."""
    environment = prepare_terms(external_rows)
    if environment.is_empty():
        return None, [], "The requested external region is empty."
    context = environment.copy()
    composed = None
    obligations = []
    for name, definition in components.items():
        component = prepare_contract(definition)
        if context.is_empty():
            return (
                None,
                obligations,
                "The accumulated interface relation is inconsistent.",
            )
        for index, assumption in enumerate(component.a.terms):
            source = definition["assumptions"][index]
            check = check_interface_assumption(
                context,
                assumption,
                derived_boundary=bool(source.get("derived_boundary", False)),
            )
            obligations.append(
                dict(
                    component=name,
                    meaning=source.get("meaning", "assumption"),
                    **check,
                    origin=source.get("origin", ""),
                    symbolic=source.get("symbolic", ""),
                    coefficients={
                        str(k): float(v) for k, v in assumption.variables.items()
                    },
                    constant=float(assumption.constant),
                    source_coefficients=source["coefficients"],
                    source_constant=source["constant"],
                    boundary_source=source.get("boundary_source", ""),
                )
            )
        failed = [r for r in obligations if not r["established"]]
        if failed:
            item = failed[0]
            residual = item["residual"]
            size = "unavailable" if residual is None else format(residual, ".2g")
            return (
                None,
                obligations,
                (
                    item["component"]
                    + ": "
                    + item["meaning"]
                    + " unestablished; residual="
                    + size
                    + "; source="
                    + item["origin"]
                ),
            )
        context = context | component.g
        if context.is_empty():
            return (
                None,
                obligations,
                "The admitted component guarantees are inconsistent.",
            )
        try:
            composed = (
                component
                if composed is None
                else composed.compose(component, simplify=False)
            )
            # to_machine_dict preserves coefficients; to_dict is for display strings.
            composed = prepare_contract(composed.to_machine_dict())
        except ValueError as exc:
            return None, obligations, "Pacti composition was unresolved: " + str(exc)
    if composed is None:
        return None, obligations, "No component contracts were supplied."
    # External coverage uses the same normalized residual tolerance.
    for assumption in composed.a.terms:
        check = check_interface_assumption(
            environment, assumption, derived_boundary=False
        )
        obligations.append(
            dict(
                component="composed_domain",
                meaning="external coverage",
                origin="specified external region",
                symbolic="",
                coefficients={
                    str(k): float(v) for k, v in assumption.variables.items()
                },
                constant=float(assumption.constant),
                **check,
            )
        )
        if not check["established"]:
            return (
                None,
                obligations,
                "The composed assumptions do not cover the requested region.",
            )
    if (composed.a | composed.g).is_empty():
        return None, obligations, "The composed relation is inconsistent."
    return composed, obligations, ""
