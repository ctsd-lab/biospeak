"""Pacti implication checks with one normalized residual tolerance."""

from __future__ import annotations

import math
from typing import Any

import sympy as sp
from pacti.terms.polyhedra.polyhedra import PolyhedralTermList

RESIDUAL_TOLERANCE = 1e-9


def numerical_zero(value: float, tolerance: float = RESIDUAL_TOLERANCE) -> float:
    """Apply the declared tolerance to a comparison residual, never model data."""
    if not math.isfinite(value):
        raise ValueError("A comparison residual must be finite.")
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("The residual tolerance must be finite and nonnegative.")
    return 0.0 if abs(value) <= tolerance else value


def constraint_scale(term: Any) -> float:
    """Use the same positive row scale as the contract preparation step."""
    return abs(float(term.constant)) or max(
        [abs(float(value)) for value in term.variables.values()] + [1.0]
    )


def _validate_term(term: Any) -> None:
    values = [float(term.constant), *(float(v) for v in term.variables.values())]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Interface constraints must have finite coefficients.")


def _rational(value: Any) -> sp.Rational:
    # Preserve the exact value of the stored binary float.
    return sp.Rational(float(value))


def _explicit_row_certificate(context: Any, assumption: Any) -> int | None:
    """Find one stored halfspace that implies the requested halfspace exactly.

    For rows a*x <= b and c*x <= d, a positive scalar lambda with
    c = lambda*a and lambda*b <= d is a sufficient algebraic certificate.
    This also handles an identical row already present in the context. No
    optimization, tolerance, or change of coefficients is used in this check.
    """
    target = {
        key: _rational(value)
        for key, value in assumption.variables.items()
        if float(value) != 0.0
    }
    rhs = _rational(assumption.constant)
    if not target:
        return -1 if rhs >= 0 else None
    for index, term in enumerate(context.terms):
        source = {
            key: _rational(value)
            for key, value in term.variables.items()
            if float(value) != 0.0
        }
        if set(source) != set(target):
            continue
        pivot = next(iter(target))
        factor = target[pivot] / source[pivot]
        if factor <= 0:
            continue
        if any(target[key] != factor * source[key] for key in target):
            continue
        if factor * _rational(term.constant) <= rhs:
            return index
    return None


def check_interface_assumption(
    context: Any,
    assumption: Any,
    *,
    derived_boundary: bool = False,
    tolerance: float = RESIDUAL_TOLERANCE,
) -> dict[str, Any]:
    """Check max(a*z-b) after positive row normalization.

    All interface and performance checks use the same absolute tolerance.
    The original residual, Pacti comparison, and constraint coefficients are
    retained. ``derived_boundary`` records provenance only.
    """
    if not isinstance(derived_boundary, bool):
        raise TypeError("derived_boundary must be a Boolean provenance flag.")
    numerical_zero(0.0, tolerance)
    _validate_term(assumption)
    for term in context.terms:
        _validate_term(term)
    if context.is_empty():
        raise ValueError("An empty interface context cannot establish an assumption.")

    strict = False
    errors: list[str] = []
    try:
        strict = bool(context.refines(PolyhedralTermList([assumption.copy()])))
    except (ValueError, RuntimeError, ArithmeticError) as exc:
        errors.append("refinement: " + str(exc))

    direct_index = _explicit_row_certificate(context, assumption)
    exact_row = direct_index is not None
    maximum = None
    residual = None
    scale = constraint_scale(assumption)
    normalized_residual = None
    comparison_residual = None
    try:
        value = context.optimize(assumption.variables, maximize=True)
        if value is not None and math.isfinite(float(value)):
            maximum = float(value)
            difference = maximum - float(assumption.constant)
            if math.isfinite(difference):
                residual = difference
                normalized_residual = difference / scale
                comparison_residual = numerical_zero(normalized_residual, tolerance)
            else:
                errors.append("optimization: nonfinite residual")
        else:
            errors.append("optimization: no finite maximum")
    except (ValueError, RuntimeError, ArithmeticError) as exc:
        errors.append("optimization: " + str(exc))

    within_tolerance = bool(
        normalized_residual is not None and abs(normalized_residual) <= tolerance
    )
    numerical_established = bool(
        not errors and comparison_residual is not None and comparison_residual <= 0
    )
    established = bool(exact_row or numerical_established)
    basis = (
        "explicit_row_certificate"
        if exact_row
        else "numerical_tolerance"
        if numerical_established and within_tolerance
        else "pacti_refinement"
        if numerical_established and strict
        else "pacti_optimization"
        if numerical_established
        else "unresolved"
    )
    return dict(
        established=established,
        residual=residual,
        strict_refinement=strict,
        normalized_residual=normalized_residual,
        comparison_residual=comparison_residual,
        absolute_tolerance=tolerance,
        normalization_scale=scale,
        within_tolerance=within_tolerance,
        roundoff_allowance=tolerance * scale,
        roundoff_boundary=bool(established and not strict and within_tolerance),
        explicit_row_certificate=exact_row,
        source_row_index=direct_index,
        derived_boundary=derived_boundary,
        decision_basis=basis,
        maximum=maximum,
        solver_error="; ".join(errors),
    )
