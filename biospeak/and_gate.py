"""Regulation of output flux in an AND logic gate."""

from dataclasses import dataclass, field

import numpy as np

from .contracts import compose_checked, interval, prepare_contract, row
from .enzyme import binding_complex


@dataclass
class ANDGate:
    """Two expression modules and a complex that drives reporter production."""

    name: str
    gain1: float
    gain2: float
    K: float
    kout: float
    mechanism: str = "binding"
    parameters: dict = field(default_factory=dict)

    def flux(self, xi1, xi2):
        """Reporter production flux in nM/min for promoter activities in RPU."""
        first, second = np.broadcast_arrays(
            self.gain1 * np.asarray(xi1, dtype=float),
            self.gain2 * np.asarray(xi2, dtype=float),
        )
        if self.mechanism == "SipC competition":
            p = self.parameters
            px, py = p["gamma_x"] * first, p["gamma_y"] * second
            lo, hi = np.zeros_like(second), second.copy()
            for _ in range(70):
                y = (lo + hi) / 2
                x = px / (p["gamma_x"] + p["gamma_xy"] * y / self.K)
                z = p["pZ"] / (
                    p["gamma_z"] + (p["gamma_yz"] + p["secretion"]) * y / p["KYZ"]
                )
                residual = p["gamma_y"] * y + p["gamma_xy"] * x * y / self.K
                residual += p["gamma_yz"] * y * z / p["KYZ"] - py
                hi = np.where(residual > 0, y, hi)
                lo = np.where(residual <= 0, y, lo)
            y = (lo + hi) / 2
            x = px / (p["gamma_x"] + p["gamma_xy"] * y / self.K)
            return self.kout * x * y / self.K
        complex_level = binding_complex(first, second, self.K)
        if self.mechanism == "HrpV inhibition":
            complex_level = np.maximum(complex_level - self.parameters["V_tot"], 0)
        elif self.mechanism != "binding":
            raise ValueError("Unknown AND mechanism: " + self.mechanism)
        return self.kout * complex_level

    def component_contracts(self, xi1, xi2):
        """Expression equalities and reporter flux bounds on the input region."""
        components = {}
        for index, gain, domain in [(1, self.gain1, xi1), (2, self.gain2, xi2)]:
            source, target = f"z_xi{index}", f"z_T{index}"
            components[f"expression_{index}"] = {
                "input_vars": [source],
                "output_vars": [target],
                "assumptions": interval(source, *np.log(domain)),
                "guarantees": [
                    row({target: 1, source: -1}, np.log(gain)),
                    row({target: -1, source: 1}, -np.log(gain)),
                ],
            }
        lower = float(self.flux(xi1[0], xi2[0]))
        upper = float(self.flux(xi1[1], xi2[1]))
        components["reporter"] = {
            "input_vars": ["z_T1", "z_T2"],
            "output_vars": ["y"],
            "assumptions": interval("z_T1", *np.log(self.gain1 * np.array(xi1)))
            + interval("z_T2", *np.log(self.gain2 * np.array(xi2))),
            "guarantees": interval("y", lower, upper),
        }
        return components

    def verify(self, xi1, xi2, direction, threshold):
        """Compose the component contracts and check a required output flux."""
        components = self.component_contracts(xi1, xi2)
        assumptions = (
            components["expression_1"]["assumptions"]
            + components["expression_2"]["assumptions"]
        )
        sign = -1 if direction == "lower" else 1
        if direction not in ("lower", "upper"):
            raise ValueError("direction must be 'lower' or 'upper'.")
        specification = prepare_contract(
            {
                "input_vars": ["z_xi1", "z_xi2"],
                "output_vars": ["y"],
                "assumptions": assumptions,
                "guarantees": [row({"y": sign}, sign * threshold)],
            }
        )
        composed, checks, reason = compose_checked(components, assumptions)
        if composed is None:
            return dict(
                refines=False, composed=None, specification=specification, reason=reason
            )
        from ._polyhedra import check_interface_assumption

        refines = all(
            check_interface_assumption(composed.g | specification.a, term)[
                "established"
            ]
            for term in specification.g.terms
        )
        return dict(
            refines=refines,
            composed=composed,
            specification=specification,
            flux_interval=composed.get_variable_bounds("y"),
        )


def gate_designs():
    """The binding circuits, HrpV inhibition, weaker association, and SipC competition."""
    return [
        ANDGate("HrpR/HrpS", 4000, 800, 250, 1.2),
        ANDGate("Split T7", 500, 700, 1200, 10),
        ANDGate("InvF/SicA", 800, 4000, 250, 1),
        ANDGate(
            "HrpR/HrpS with HrpV inhibition",
            4000,
            800,
            250,
            1.2,
            "HrpV inhibition",
            {"V_tot": 120},
        ),
        ANDGate("Split T7 with weaker fragment association", 500, 700, 3000, 10),
        ANDGate(
            "InvF/SicA with SipC competition",
            800,
            4000,
            250,
            1,
            "SipC competition",
            {
                "gamma_x": 0.022,
                "gamma_xy": 0.022,
                "gamma_y": 0.046,
                "gamma_z": 0.139,
                "gamma_yz": 0.022,
                "secretion": 0.1,
                "KYZ": 28,
                "pZ": 90 * 1e24 / 6.02214076e23,
            },
        ),
    ]
