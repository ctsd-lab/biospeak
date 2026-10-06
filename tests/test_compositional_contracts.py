"""Regression checks for the existing fixed-parameter AND-gate case study."""

import math

import numpy as np
import pytest
from pacti.iocontract import Var

import compositional_regime_contracts as model


CANDIDATES = model.make_candidate_designs()
CONDITIONS = model.make_top_level_conditions()
INITIAL_OUTCOMES = {
    "HrpR/HrpS": (True, False, False),
    "split T7 RNAP": (True, False, False),
    "InvF/SicA": (True, True, False),
}
KNOWN_BOUNDARY_FAILURE = pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason=("Pacti 0.1.0 composition shifts equal log boundaries by roundoff, "
            "then strict containment rejects assumption coverage; no padding applied"),
)


def test_fixed_specification_and_redesign_choices():
    assert [(c.name, c.xi1_bounds, c.xi2_bounds, c.guarantee_kind,
             c.flux_threshold) for c in CONDITIONS] == [
        ("ON", (0.35, 0.65), (0.35, 0.65), "lower", 120.0),
        ("OFF-1", (0.0055, 0.07), (0.35, 0.65), "upper", 60.0),
        ("OFF-2", (0.35, 0.65), (0.0055, 0.07), "upper", 60.0),
    ]
    assert model.REDESIGN_PARAMETERS == {
        "HrpR/HrpS": {"hrpv_nM": 120.0},
        "split T7 RNAP": {"affinity_nM": 3000.0},
        "InvF/SicA": {
            "sink_affinity_nM": 28.0,
            "gamma_x_per_min": 0.022,
            "gamma_xy_per_min": 0.022,
            "gamma_y_per_min": 0.046,
            "gamma_z_per_min": 0.139,
            "gamma_yz_per_min": 0.022,
            "secretion_rate_per_min": 0.1,
            "sipc_proteins_per_transcript": 90.0,
            "sipc_transcription_rate_per_min": 1.0,
        },
    }


def test_sensor_assignments_preserve_the_rpu_interface():
    for first, second in zip(CANDIDATES[::2], CANDIDATES[1::2]):
        assert first.sensor_1 == second.sensor_2
        assert first.sensor_2 == second.sensor_1
        for flux_fn in (model.initial_flux_bounds, model.redesigned_flux_bounds):
            np.testing.assert_array_equal(flux_fn(first, 0.35, 0.65),
                                          flux_fn(second, 0.35, 0.65))


@KNOWN_BOUNDARY_FAILURE
@pytest.mark.parametrize("candidate", CANDIDATES, ids=lambda c: c.name)
def test_initial_refinement_keeps_failed_modes(candidate):
    results = [model.verify_condition(candidate, c) for c in CONDITIONS]
    assert all(r["assumptions"] for r in results)
    assert tuple(bool(r["refines"]) for r in results) == INITIAL_OUTCOMES[candidate.core.name]
    assert not all(r["refines"] for r in results)


@KNOWN_BOUNDARY_FAILURE
@pytest.mark.parametrize("candidate", CANDIDATES, ids=lambda c: c.name)
def test_selected_redesign_refines_all_three_modes(candidate):
    for condition in CONDITIONS:
        result = model.verify_redesigned_condition(candidate, condition)
        assert result["assumptions"]
        assert result["guarantees"]
        assert result["refines"]


@pytest.mark.parametrize("candidate", CANDIDATES, ids=lambda c: c.name)
def test_conditional_guarantees_are_separate_from_failed_coverage(candidate):
    initial = [model.verify_condition(candidate, c) for c in CONDITIONS]
    redesigned = [model.verify_redesigned_condition(candidate, c) for c in CONDITIONS]
    assert tuple(bool(r["guarantees"]) for r in initial) == INITIAL_OUTCOMES[candidate.core.name]
    assert all(r["guarantees"] for r in redesigned)
    # These guarantee checks alone do not establish overall refinement.


@pytest.mark.parametrize("candidate", CANDIDATES[::2], ids=lambda c: c.core.name)
@pytest.mark.parametrize("condition", CONDITIONS, ids=lambda c: c.name)
def test_fixed_parameter_envelopes_and_sampled_corner_bounds(candidate, condition):
    x, y = np.meshgrid(np.linspace(*condition.xi1_bounds, 5),
                       np.linspace(*condition.xi2_bounds, 5))
    for flux_fn, bounds_fn in [
        (model.initial_flux_bounds, model.condition_flux_interval),
        (model.redesigned_flux_bounds, model.redesigned_condition_flux_interval),
    ]:
        lower, upper = flux_fn(candidate, x, y)
        np.testing.assert_array_equal(lower, upper)
        lo, hi = bounds_fn(candidate, condition.xi1_bounds, condition.xi2_bounds)
        assert np.all(lower >= lo - 1e-10)
        assert np.all(upper <= hi + 1e-10)


@KNOWN_BOUNDARY_FAILURE
def test_closed_core_assumptions_include_exact_expression_endpoints():
    candidate = CANDIDATES[0]
    for condition in CONDITIONS:
        core = model.create_finite_box_core_contract(
            candidate, condition.xi1_bounds, condition.xi2_bounds)
        for x in condition.xi1_bounds:
            for y in condition.xi2_bounds:
                assert core.a.contains_behavior({
                    Var("z_input1"): math.log(candidate.core.gain_1_nM_per_rpu * x),
                    Var("z_input2"): math.log(candidate.core.gain_2_nM_per_rpu * y),
                })
        assert model.verify_condition(candidate, condition)["assumptions"]


def test_hrpv_zero_flux_is_physical_zero_with_only_a_log_upper_floor():
    candidate = CANDIDATES[0]
    condition = CONDITIONS[2]
    lower, upper = model.redesigned_flux_bounds(candidate, 0.65, 0.07)
    assert lower == upper == 0.0
    core = model.create_redesigned_finite_box_core_contract(
        candidate, condition.xi1_bounds, condition.xi2_bounds)
    assert core.g.contains_behavior({Var("z_flux"): math.log(model.ZERO_FLUX_UPPER_EPS)})
    assert core.g.contains_behavior({Var("z_flux"): math.log(model.ZERO_FLUX_UPPER_EPS) - 100.0})
    assert not core.g.contains_behavior({Var("z_flux"): math.log(model.ZERO_FLUX_UPPER_EPS) + 1.0})


@pytest.mark.parametrize("core_name,key,low,high", [
    ("HrpR/HrpS", "hrpv_nM", 114.410205, 130.680587),
    ("split T7 RNAP", "affinity_nM", 2286.166667, 3164.916667),
])
def test_conditional_guarantee_window_has_failing_neighbors(monkeypatch, core_name, key, low, high):
    candidate = next(c for c in CANDIDATES if c.core.name == core_name)
    parameters = model.REDESIGN_PARAMETERS[core_name]
    # Stay away from rounded-boundary/LP tolerances, while checking both sides.
    for value, expected in [(low - 0.1, False), (low + 0.1, True),
                            (high - 0.1, True), (high + 0.1, False)]:
        monkeypatch.setitem(parameters, key, value)
        results = [model.verify_redesigned_condition(candidate, c) for c in CONDITIONS]
        # Check guarantee implications only. Coverage remains the strict xfail above.
        assert all(r["guarantees"] for r in results) is expected
