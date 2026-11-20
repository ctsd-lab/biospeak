"""Component contracts for RBS library screening."""

import csv
import math
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import sympy as sp

from . import contracts as cc
from ._polyhedra import (
    RESIDUAL_TOLERANCE,
    check_interface_assumption,
    constraint_scale,
    numerical_zero,
)
from .contracts import compose_checked, finite_bounds, prepare_contract


@dataclass(frozen=True)
class Promoter:
    id: str
    k: float
    K: float  # mM
    n: float
    alpha: float
    rbs: str
    promoter: str


@dataclass(frozen=True)
class Gate:
    KR: float
    KS: float
    nR: float
    nS: float


@dataclass(frozen=True)
class Repressor:
    k: float
    K: float
    n: float
    alpha: float
    rbs: str


@dataclass(frozen=True)
class Mode:
    name: str
    I1: tuple
    I2: tuple
    states: tuple
    direction: str
    threshold: float


def read_csv(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def load_calibration(root=None):
    if root is None:
        root = Path(__file__).parent / "data" / "rbs"
    promoters = {}
    for r in read_csv(root / "promoter_fits.csv"):
        if r["fit_usable"].lower() != "true":
            continue
        # The source table is in mM. Stored molar values must describe the same input.
        if not math.isclose(float(r["K_M"]) * 1e3, float(r["K_mM"]), rel_tol=1e-12):
            raise ValueError("The promoter threshold units do not agree.")
        promoters[r["module_id"]] = Promoter(
            r["module_id"],
            float(r["k_au"]),
            float(r["K_mM"]),
            float(r["n"]),
            float(r["alpha"]),
            r["rbs"],
            r["promoter"],
        )
    r = read_csv(root / "and_fit.csv")[0]
    gate = Gate(
        float(r["KR_proxy_au"]), float(r["KS_proxy_au"]), float(r["nR"]), float(r["nS"])
    )
    repressors = {
        r["ci_rbs"]: Repressor(
            float(r["k_au"]),
            float(r["K_proxy_au"]),
            float(r["n"]),
            float(r["alpha"]),
            r["ci_rbs"],
        )
        for r in read_csv(root / "not_fits.csv")
        if r["fit_usable"].lower() == "true"
    }
    return promoters, gate, repressors


def modes():
    # The mixed modes and OFF are the specification. LL follows by monotonicity.
    return [
        Mode("OFF", (0.4, 1.6), (0.33, 1.3), ("high", "high"), "upper", 0.2),
        Mode("ON1", (0, 0.0016), (0.33, 1.3), ("low", "high"), "lower", 0.8),
        Mode("ON2", (0.4, 1.6), (0, 0.0052), ("high", "low"), "lower", 0.8),
    ]


class RBSLibrary:
    """Construct and compose calibrated expression, AND, and repression contracts."""

    def __init__(self, derived, gate, not_gate):
        self.derived = derived
        self.gate = gate
        self.not_gate = not_gate
        self._upstream_cache = {}
        alpha = next(
            s
            for s in derived["activation"].normalized.free_symbols
            if str(s) == "alpha"
        )
        self.promoter_fun = sp.lambdify(
            (derived["activation"].activity_symbols[0], alpha),
            derived["activation"].normalized,
            "numpy",
        )
        self.and_fun = sp.lambdify(
            derived["and_occupancy"].activity_symbols,
            derived["and_occupancy"].normalized,
            "numpy",
        )
        self.not_fun = sp.lambdify(
            derived["not_occupancy"].activity_symbols,
            derived["not_occupancy"].regulated,
            "numpy",
        )
        # Invert the derived equations symbolically before using any numeric calibration.
        t = sp.Symbol("t", positive=True)
        qC = derived["not_occupancy"].activity_symbols[0]
        self.repression_inverse = sp.solve(
            sp.Eq(derived["not_occupancy"].regulated, t), qC
        )[0]
        self.inv_repression = sp.lambdify(t, self.repression_inverse, "numpy")

    def expression(self, p, inducer):
        q = (np.asarray(inducer, dtype=float) / p.K) ** p.n
        return p.k * self.promoter_fun(q, p.alpha)

    def activities(self, R, S):
        g = self.gate
        return (np.asarray(R) / g.KR) ** g.nR, (np.asarray(S) / g.KS) ** g.nS

    def activity(self, pR, pS, I1, I2):
        return self.and_fun(
            *self.activities(self.expression(pR, I1), self.expression(pS, I2))
        )

    def fluorescence(self, pR, pS, I1, I2, gain):
        cbar = (gain / self.not_gate.K) * self.activity(pR, pS, I1, I2)
        return self.not_gate.alpha + self.not_fun(cbar**self.not_gate.n)

    def required_gain(self, pR, pS, selected_modes):
        lows = []
        highs = []
        for mode in selected_modes:
            up = self.upstream(pR, pS, mode)
            if up["status"] != "established":
                return dict(
                    status="unknown", reason="an upstream region is unestablished"
                )
            alo, ahi = up["activity_interval"]
            ng = self.not_gate
            residual = mode.threshold - ng.alpha
            if not 0 < residual < 1:
                raise ValueError(
                    "The requested fluorescence lies outside the invertible response range."
                )
            C = ng.K * float(self.inv_repression(residual)) ** (1 / ng.n)
            if mode.direction == "upper":
                lows.append(C / alo)
            else:
                highs.append(C / ahi)
        lo, hi = max(lows), min(highs)
        return dict(
            status="feasible" if lo <= hi else "unknown",
            lower=lo,
            upper=hi,
            selected=math.sqrt(lo * hi) if lo <= hi else None,
            meaning="sufficient expression gain requirement in the declared repressor proxy units",
        )

    def upstream(self, pR, pS, mode):
        key = (pR, pS, mode)
        if key in self._upstream_cache:
            return deepcopy(self._upstream_cache[key])
        components = {}
        for index, p, domain, scale in [
            (1, pR, mode.I1, self.gate.KR),
            (2, pS, mode.I2, self.gate.KS),
        ]:
            variable = "zR" if index == 1 else "zS"
            levels = np.asarray(self.expression(p, np.array(domain)), float) / scale
            if np.any(levels <= 0) or not np.isfinite(levels).all():
                return dict(
                    status="unknown",
                    contracts=components,
                    coverage=[],
                    reason="The calibrated log interface is outside its positive domain.",
                )
            guarantees = cc.interval(
                variable,
                *np.log(levels),
                "symbolic expression equilibrium and its monotonicity on the inducer interval",
            )
            for term in guarantees:
                term.update(
                    symbolic=str(self.derived["activation"].normalized),
                    calibration_record=p.id,
                )
            components["expression_" + str(index)] = dict(
                input_vars=["u" + str(index)],
                output_vars=[variable],
                assumptions=cc.interval(
                    "u" + str(index), *domain, "specified external inducer interval"
                ),
                guarantees=guarantees,
            )
        external = (
            components["expression_1"]["assumptions"]
            + components["expression_2"]["assumptions"]
        )
        prefix, checks, reason = compose_checked(components, external)
        if prefix is None:
            return dict(
                status="unknown", contracts=components, coverage=checks, reason=reason
            )
        domains, assumptions, activities = [], [], []
        for branch, exponent, p in [("R", self.gate.nR, pR), ("S", self.gate.nS, pS)]:
            variable = "z" + branch
            zlo, zhi = finite_bounds(prefix, variable)
            qlo, qhi = math.exp(exponent * zlo), math.exp(exponent * zhi)
            domains.append((qlo, qhi))
            new_rows = cc.interval(
                variable,
                zlo,
                zhi,
                "image certified by the upstream expression contract",
            )
            for term in new_rows:
                term.update(
                    symbolic="q_" + branch + "=exp(n_" + branch + "*z_" + branch + ")",
                    calibration_record=p.id,
                    derived_boundary=True,
                    boundary_source="Pacti bound on "
                    + variable
                    + " in the expression composition",
                )
            assumptions += new_rows
            activities.append(
                dict(
                    branch=branch,
                    q_min=qlo,
                    q_max=qhi,
                    log_min=zlo,
                    log_max=zhi,
                    calibration=p.id,
                )
            )
        module = self.derived["and_occupancy"]
        enclosure = cc.quotient_enclosure(
            module.normalized, module.activity_symbols, domains
        )
        guarantees = cc.affine_rows(
            enclosure,
            "za",
            [("zR", self.gate.nR), ("zS", self.gate.nS)],
            "symbolic AND occupancy equilibrium, conserved promoter, and bounded positive sums",
        )
        guarantees.append(
            cc.row(
                {"za": 1},
                0.0,
                "promoter conservation and positive states",
                "active promoter fraction at most one",
            )
        )
        components["and"] = dict(
            input_vars=["zR", "zS"],
            output_vars=["za"],
            assumptions=assumptions,
            guarantees=guarantees,
        )
        composed, checks, reason = compose_checked(components, external)
        result = dict(
            status="unknown",
            contracts=components,
            coverage=checks,
            reason=reason,
            trace=enclosure["trace"],
            calibrated_activities=activities,
        )
        if composed is not None:
            try:
                lo, hi = finite_bounds(composed, "za")
            except ValueError as error:
                result["reason"] = str(error)
            else:
                result.update(
                    status="established",
                    activity_interval=(math.exp(lo), math.exp(hi)),
                    composed=composed.to_machine_dict(),
                )
        self._upstream_cache[key] = deepcopy(result)
        return result

    def construct(self, pR, pS, mode, gain):
        if not math.isfinite(gain) or gain <= 0:
            raise ValueError("The repressor gain must be positive and finite.")
        upstream = self.upstream(pR, pS, mode)
        if upstream["status"] != "established":
            return upstream
        components = deepcopy(upstream["contracts"])
        alo, ahi = upstream["activity_interval"]
        repressor = self.not_gate
        log_gain = math.log(gain / repressor.K)
        components["repressor_expression"] = dict(
            input_vars=["za"],
            output_vars=["zC"],
            assumptions=[],
            guarantees=[
                cc.row(
                    {"zC": 1, "za": -1},
                    log_gain,
                    "transcript and protein equilibrium balances",
                    "expression equality",
                ),
                cc.row(
                    {"zC": -1, "za": 1},
                    -log_gain,
                    "transcript and protein equilibrium balances",
                    "expression equality",
                ),
            ],
        )
        # Query the same Pacti composition for the downstream input domain.
        external = (
            components["expression_1"]["assumptions"]
            + components["expression_2"]["assumptions"]
        )
        prefix, checks, reason = compose_checked(components, external)
        if prefix is None:
            return dict(
                status="unknown", contracts=components, coverage=checks, reason=reason
            )
        clo, chi = finite_bounds(prefix, "zC")
        qlo, qhi = math.exp(repressor.n * clo), math.exp(repressor.n * chi)
        module = self.derived["not_occupancy"]
        enclosure = cc.quotient_enclosure(
            module.regulated, module.activity_symbols, [(qlo, qhi)]
        )
        guarantees = cc.affine_rows(
            enclosure,
            "zw",
            [("zC", repressor.n)],
            "imported repression balance, promoter conservation, and positive sum enclosure",
        )
        # Component endpoint bounds follow from the symbolically established order.
        wlo, whi = float(self.not_fun(qhi)), float(self.not_fun(qlo))
        guarantees += cc.interval(
            "zw",
            math.log(wlo),
            math.log(whi),
            "monotone repression equilibrium on the complete certified input interval",
        )
        repression_assumptions = cc.interval(
            "zC", clo, chi, "image of the composed upstream contract"
        )
        for term in repression_assumptions:
            term.update(
                derived_boundary=True,
                boundary_source="Pacti bound on zC in the preceding composed contract",
            )
        components["repression"] = dict(
            input_vars=["zC"],
            output_vars=["zw"],
            assumptions=repression_assumptions,
            guarantees=guarantees,
        )
        if not 0 < mode.threshold - repressor.alpha < 1:
            raise ValueError(
                "The output target must be within the calibrated repression range."
            )
        sign = 1 if mode.direction == "upper" else -1
        target = math.log(mode.threshold - repressor.alpha)
        specification = dict(
            input_vars=["u1", "u2"],
            output_vars=["zw"],
            assumptions=external,
            guarantees=[
                cc.row(
                    {"zw": sign},
                    sign * target,
                    "prescribed normalized fluorescence",
                    "output requirement",
                )
            ],
        )
        composed, checks, reason = compose_checked(components, external)
        result = dict(
            status="unknown",
            contracts=components,
            specification=specification,
            coverage=checks,
            reason=reason,
            activity_interval=upstream["activity_interval"],
            trace=upstream["trace"] + enclosure["trace"],
        )
        if composed is None:
            return result
        spec = prepare_contract(specification)
        assumption_checks = [
            check_interface_assumption(spec.a, term) for term in composed.a.terms
        ]
        assumption_coverage = all(check["established"] for check in assumption_checks)
        consistent = not (composed.a | composed.g).is_empty()
        raw_refines = bool(composed.refines(spec))
        context = spec.a | composed.g
        guarantee_checks = [
            check_interface_assumption(context, term) for term in spec.g.terms
        ]
        refined = bool(
            assumption_coverage
            and consistent
            and all(check["established"] for check in guarantee_checks)
        )
        lower, upper = finite_bounds(composed, "zw")
        flo, fhi = repressor.alpha + math.exp(lower), repressor.alpha + math.exp(upper)
        violation_residuals = []
        for term in spec.g.terms:
            minimum = context.optimize(term.variables, maximize=False)
            residual = (
                None
                if minimum is None
                else (float(minimum) - float(term.constant)) / constraint_scale(term)
            )
            violation_residuals.append(residual)
        contradiction = bool(
            assumption_coverage
            and consistent
            and any(
                residual is not None
                and math.isfinite(residual)
                and numerical_zero(residual) > 0
                for residual in violation_residuals
            )
        )
        decision = (
            "certified" if refined else "violated" if contradiction else "unknown"
        )
        if decision == "unknown" and not reason:
            reason = "The composed output enclosure does not establish the required threshold."
        result["reason"] = reason
        result.update(
            status=decision,
            fluorescence_interval=(flo, fhi),
            pacti=dict(
                executed=True,
                status=decision,
                refines=refined,
                assumption_coverage=assumption_coverage,
                nonempty=consistent,
                raw_refines=raw_refines,
                absolute_tolerance=RESIDUAL_TOLERANCE,
                assumption_checks=assumption_checks,
                guarantee_checks=guarantee_checks,
                violation_residuals=violation_residuals,
                log_output_bounds=[lower, upper],
                composed=composed.to_machine_dict(),
                specification=spec.to_machine_dict(),
            ),
        )
        return result


def verify_library(constructor, promoters, selected_modes, gain):
    """Screen each calibrated RBS assignment against all required modes."""
    rows, certificates = [], {}
    first = [
        p for p in promoters.values() if p.promoter == "Plac" and p.id.endswith("30C")
    ]
    second = [
        p for p in promoters.values() if p.promoter == "PBAD" and p.id.endswith("30C")
    ]
    for pR in first:
        for pS in second:
            results = {
                m.name: constructor.construct(pR, pS, m, gain) for m in selected_modes
            }
            certificates[pR.id + "__" + pS.id] = results
            statuses = [result["status"] for result in results.values()]
            decision = (
                "certified"
                if all(s == "certified" for s in statuses)
                else "violated"
                if "violated" in statuses
                else "unknown"
            )
            rows.append(dict(R_RBS=pR.rbs, S_RBS=pS.rbs, contract=decision))
    return rows, certificates
