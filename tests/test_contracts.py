"""Equilibrium enclosures, refinement, and the circuit decisions in the paper."""

from itertools import product

import numpy as np
import pytest
import sympy as sp
from pacti.iocontract import Var

from biospeak import (
    RBSLibrary,
    binding_complex,
    derive_modules,
    enzyme_contract,
    gate_designs,
    hill_contracts,
    hill_limits,
    load_calibration,
    modes,
    refines,
    to_contract,
)
from biospeak.cello import construct, load_gates, response
from biospeak.contracts import quotient_enclosure


def test_equilibrium_laws_satisfy_reaction_balances():
    modules = derive_modules()
    for module in modules.values():
        for balance in list(module.balances.values()) + list(module.pools.values()):
            assert sp.cancel(balance.subs(module.solution, simultaneous=True)) == 0
    qR, qS = modules["and_occupancy"].activity_symbols
    assert (
        sp.cancel(modules["and_occupancy"].normalized - qR * qS / ((1 + qR) * (1 + qS)))
        == 0
    )


@pytest.mark.parametrize("positive", [True, False])
def test_hill_refinement_encloses_parameter_extrema_and_shares_boundaries(positive):
    coefficients = (0.9 / 0.11, 1.1 / 0.09)
    refined = hill_contracts(2, positive, (8, 12), coefficients)
    reference = hill_contracts(2, positive, (8, 12), coefficients, kind="reference")
    for entry, other in zip(refined, reference):
        assert refines(to_contract(entry), to_contract(other))
        inputs = np.geomspace(*entry["input_interval"], 51)
        lower, upper = hill_limits(entry, inputs, 2)
        for k, K, delta in product((0.9, 1.1), (8, 12), (0.09, 0.11)):
            q = (inputs / K) ** 2
            output = 2 * k / delta * (q if positive else 1) / (1 + q)
            assert np.all(output >= lower - 1e-10)
            assert np.all(output <= upper + 1e-10)
    for first, second in zip(refined[:-1], refined[1:]):
        boundary = first["input_interval"][1]
        assert np.allclose(hill_limits(first, boundary), hill_limits(second, boundary))


@pytest.mark.parametrize("regime", ["K", "substrate", "enzyme"])
def test_enzymatic_contract_encloses_physical_binding_equilibrium(regime):
    contract = enzyme_contract(20, 3, regime)
    admitted = 0
    for enzyme, substrate in product(np.geomspace(0.01, 1e4, 23), repeat=2):
        complex_level = binding_complex(enzyme, substrate, 20)
        assert np.isclose(
            20 * complex_level, (enzyme - complex_level) * (substrate - complex_level)
        )
        values = {
            Var("z_u"): np.log(enzyme),
            Var("z_x"): np.log(substrate),
            Var("z_y"): np.log(3 * complex_level),
        }
        if contract.a.contains_behavior(values):
            admitted += 1
            assert contract.g.contains_behavior(values)
    assert admitted > 0


def test_all_cello_regions_enclose_the_characterized_response():
    for gate in load_gates():
        for mode in ["low", "transition", "high"]:
            dimension = 1 if gate["type"] == "NOR" else 2
            contract, data, details = construct(gate, (mode,) * dimension)
            assert not (contract.a | contract.g).is_empty()
            assert refines(
                contract,
                type(contract).from_dict(
                    details["monomial_ratio_contract"], simplify=False
                ),
            )
            thresholds = (
                [(gate["IL"], gate["IH"])]
                if dimension == 1
                else [(gate["IL1"], gate["IH1"]), (gate["IL2"], gate["IH2"])]
            )
            grids = [
                np.geomspace(lo / 100, lo, 7)
                if mode == "low"
                else np.geomspace(hi, hi * 100, 7)
                if mode == "high"
                else np.geomspace(lo, hi, 7)
                for lo, hi in thresholds
            ]
            for values in product(*grids):
                coordinates = dict(zip(data["input_vars"], np.log(values)))
                coordinates["zy"] = np.log(response(gate, values))
                for term in data["guarantees"]:
                    residual = (
                        sum(c * coordinates[k] for k, c in term["coefficients"].items())
                        - term["constant"]
                    )
                    assert residual <= 1e-8


def test_and_mechanism_decisions():
    conditions = [
        ((0.35, 0.65), (0.35, 0.65), "lower", 120),
        ((0.0055, 0.07), (0.0055, 0.65), "upper", 60),
        ((0.0055, 0.65), (0.0055, 0.07), "upper", 60),
    ]
    expected = [
        [True, False, False],
        [True, False, False],
        [True, True, False],
        [True, True, True],
        [True, True, True],
        [True, True, True],
    ]
    for design, decisions in zip(gate_designs(), expected):
        for condition, decision in zip(conditions, decisions):
            result = design.verify(*condition)
            assert result["refines"] == decision
            assert result["composed"] is not None


def test_reference_rbs_gain_and_output_enclosures():
    modules = derive_modules()
    promoters, gate, repressors = load_calibration()
    constructor = RBSLibrary(modules, gate, repressors["rbs34"])
    first, second = promoters["Plac_rbsH_30C"], promoters["PBAD_rbs33_30C"]
    gain = 17944.802603618242
    requirement = constructor.required_gain(first, second, modes())
    assert requirement["lower"] < gain < requirement["upper"]
    for mode in modes():
        result = constructor.construct(first, second, mode, gain)
        assert result["status"] == "certified"
        lower, upper = result["fluorescence_interval"]
        outputs = constructor.fluorescence(
            first, second, np.array(mode.I1), np.array(mode.I2), gain
        )
        assert outputs.min() >= lower - 1e-10
        assert outputs.max() <= upper + 1e-10


def test_positive_quotient_enclosure():
    x, u = sp.symbols("x u", positive=True)
    expression = x * u / ((1 + x) * (1 + u))
    bound = quotient_enclosure(expression, (x, u), [(0.1, 2), (0.2, 3)])
    first, second = np.meshgrid(np.geomspace(0.1, 2, 30), np.geomspace(0.2, 3, 30))
    center = bound["slope"][0] * np.log(first) + bound["slope"][1] * np.log(second)
    exact = np.log(sp.lambdify((x, u), expression, "numpy")(first, second))
    assert np.all(exact >= center + bound["lower_offset"] - 1e-12)
    assert np.all(exact <= center + bound["upper_offset"] + 1e-12)
