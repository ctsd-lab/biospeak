"""Contracts and fixed-parameter calculations for the compositional AND gate.

The external interface uses promoter activities ``xi_1`` and ``xi_2`` in
relative promoter units (RPU). Two lumped expression modules produce the
binding partners of a catalytic complex. The complex drives one output flux.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

import numpy as np
from pacti.contracts import PolyhedralIoContract


Regime = Literal["input 1 excess", "input 2 excess", "affinity-dominated"]


@dataclass(frozen=True)
class Sensor:
    """A Marionette sensor characterized in Meyer et al. (2019)."""

    name: str
    promoter: str
    regulator: str
    start_uM: float
    half_max_uM: float
    hill_n: float
    leak_rpu: float
    max_rpu: float
    linear_max_rpu: float


@dataclass(frozen=True)
class Core:
    """Effective parameters for one two-component transcriptional AND core."""

    name: str
    component_1: str
    component_2: str
    output_promoter: str
    affinity_nM: float
    output_rate_per_min: float
    gain_1_nM_per_rpu: float
    gain_2_nM_per_rpu: float
    nominal_regime: Regime
    source: str
    note: str


@dataclass(frozen=True)
class Candidate:
    """One ordered sensor assignment and integration core."""

    name: str
    core: Core
    sensor_1: Sensor
    sensor_2: Sensor


@dataclass(frozen=True)
class Condition:
    """One part of the continuous AND specification."""

    name: str
    xi1_bounds: tuple[float, float]
    xi2_bounds: tuple[float, float]
    guarantee_kind: Literal["lower", "upper"]
    flux_threshold: float


RedesignKind = Literal["hrpv_threshold", "synzip_affinity", "sipc_titration"]


@dataclass(frozen=True)
class MechanismRedesign:
    """One literature-grounded mechanism change applied to an initial core."""

    core_name: str
    label: str
    mechanism: RedesignKind
    source: str
    note: str


# Values are transcribed from the sensor table used by the Pacti Marionette
# case study, which in turn cites Meyer et al., Nature Chemical Biology (2019).
MARIONETTE_SENSORS = {
    "aTc": Sensor(
        name="aTc",
        promoter="PTet*",
        regulator="TetR",
        start_uM=0.0018,
        half_max_uM=0.013,
        hill_n=3.8,
        leak_rpu=0.0049,
        max_rpu=2.4,
        linear_max_rpu=1.0,
    ),
    "Cuma": Sensor(
        name="Cuma",
        promoter="PCymRC",
        regulator="CymRAM",
        start_uM=0.3,
        half_max_uM=8.9,
        hill_n=2.4,
        leak_rpu=0.0043,
        max_rpu=3.7,
        linear_max_rpu=2.0,
    ),
}


# The architectures and component identities are literature-backed. The
# effective affinity, expression gains, and output rates are provisional
# nominal values in a common coarse-grained unit system. They are deliberately
# exposed here, rather than presented as measurements from those papers.
CORES = (
    Core(
        name="HrpR/HrpS",
        component_1="HrpR",
        component_2="HrpS",
        output_promoter="PhrpL",
        affinity_nM=250.0,
        output_rate_per_min=1.2,
        gain_1_nM_per_rpu=4000.0,
        gain_2_nM_per_rpu=800.0,
        nominal_regime="input 1 excess",
        source="Wang et al., Nature Communications 2011, doi:10.1038/ncomms1516",
        note="HrpR and HrpS form the co-activating complex for the sigma-54-dependent PhrpL promoter.",
    ),
    Core(
        name="split T7 RNAP",
        component_1="T7 RNAP N fragment",
        component_2="T7 RNAP C fragment",
        output_promoter="PT7",
        affinity_nM=1200.0,
        output_rate_per_min=10.0,
        gain_1_nM_per_rpu=500.0,
        gain_2_nM_per_rpu=700.0,
        nominal_regime="affinity-dominated",
        source="Shis and Bennett, PNAS 2013, doi:10.1073/pnas.1220157110",
        note="The two expressed fragments associate to restore T7 transcriptional activity.",
    ),
    Core(
        name="InvF/SicA",
        component_1="InvF",
        component_2="SicA",
        output_promoter="PsicA",
        affinity_nM=250.0,
        output_rate_per_min=1.0,
        gain_1_nM_per_rpu=800.0,
        gain_2_nM_per_rpu=4000.0,
        nominal_regime="input 2 excess",
        source="Moon et al., Nature 2012, doi:10.1038/nature11516",
        note="The transcription factor and chaperone form the active complex that drives the output promoter.",
    ),
)



DEFAULT_ON_BOX = ((0.35, 0.65), (0.35, 0.65))
DEFAULT_OFF_INTERVAL = (0.0055, 0.07)
DEFAULT_ON_THRESHOLD_NM_PER_MIN = 120.0
DEFAULT_OFF_THRESHOLD_NM_PER_MIN = 60.0

# Mechanism-specific redesign parameters.
#
# Both initial and redesigned candidates are evaluated at fixed selected
# parameter values. The contract assumptions still cover the complete ON/OFF
# input regions; the parameter values define the designs being compared.
#
# HrpR/HrpS: HrpV is treated as a tunable regulatory level. For the fixed
# ON/OFF specification and fixed redesigned parameters, the feasible HrpV
# interval is approximately 114.4--130.7 nM; use 120 nM.
#
# split T7: use the effective fragment-association parameter as the redesign
# variable. Contract satisfaction gives an effective K interval of approximately
# 2.286--3.165 uM; use K = 3.0 uM to provide additional OFF-state margin while
# retaining the ON guarantee.
#
# InvF/SicA: replace the static SipC pool by a reduced steady-state turnover
# model using the published Temme et al. rates: gamma_x = gamma_xy = 0.022 /min,
# gamma_y = 0.046 /min, gamma_z = 0.139 /min, gamma_yz = 0.022 /min,
# k_z = 0.1 /min, K_yz = 28 nM, alpha_z = 90 proteins/transcript, and
# k0 = 1 transcript/min.
MOLECULE_TO_NM_AT_1_FL = 1.0e24 / 6.02214076e23


REDESIGNS = {
    "HrpR/HrpS": MechanismRedesign(
        core_name="HrpR/HrpS",
        label="tuned HrpV inhibition",
        mechanism="hrpv_threshold",
        source=(
            "Jovanovic et al., Nature Communications 2011, "
            "doi:10.1038/ncomms1177; Wang et al., Nucleic Acids Research 2014, "
            "doi:10.1093/nar/gku593"
        ),
        note=(
            "HrpV is used as a tunable control input; the selected effective "
            "level is 0.12 uM, within the 0.1--0.8 uM inhibition assay range."
        ),
    ),
    "split T7 RNAP": MechanismRedesign(
        core_name="split T7 RNAP",
        label="weakened zipper recruitment",
        mechanism="synzip_affinity",
        source=(
            "Segall-Shapiro et al., Molecular Systems Biology 2014, "
            "doi:10.15252/msb.20145299; Thompson et al., ACS Synthetic "
            "Biology 2012, doi:10.1021/sb200015u; Anderson et al., "
            "ACS Omega 2018, doi:10.1021/acsomega.8b00156"
        ),
        note=(
            "The effective association parameter is set to 3.0 uM, inside the "
            "contract-feasible interval for the fixed ON/OFF requirement."
        ),
    ),
    "InvF/SicA": MechanismRedesign(
        core_name="InvF/SicA",
        label="SipC turnover and secretion",
        mechanism="sipc_titration",
        source=(
            "Temme et al., Journal of Molecular Biology 2008, "
            "doi:10.1016/j.jmb.2007.12.044"
        ),
        note=(
            "The static SipC pool is replaced by a steady-state turnover model "
            "using the published SicA/SipC degradation and SipC secretion rates."
        ),
    ),
}

REDESIGN_PARAMETERS = {
    "HrpR/HrpS": {
        "hrpv_nM": 120.0,
    },
    "split T7 RNAP": {
        "affinity_nM": 3000.0,
    },
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

# Numerical representation floor used only when the physical model gives
# exactly zero output but the contract must be represented in z_flux=log(y).
# This is not a biological leak term or physical model parameter: the original
# model value remains y=0, while the log-coordinate contract uses
# y <= ZERO_FLUX_UPPER_EPS to obtain a finite logarithmic constant.
ZERO_FLUX_UPPER_EPS = 1.0e-12


def make_candidate_designs() -> list[Candidate]:
    """Return three cores crossed with both ordered aTc/Cuma assignments."""

    orders = (("aTc", "Cuma"), ("Cuma", "aTc"))
    candidates: list[Candidate] = []
    index = 1
    for core in CORES:
        for sensor_1, sensor_2 in orders:
            candidates.append(
                Candidate(
                    name=f"D{index}",
                    core=core,
                    sensor_1=MARIONETTE_SENSORS[sensor_1],
                    sensor_2=MARIONETTE_SENSORS[sensor_2],
                )
            )
            index += 1
    return candidates


def sensor_activity(sensor: Sensor, inducer_uM: np.ndarray | float) -> np.ndarray:
    """Evaluate the literature Hill fit from inducer concentration to RPU."""

    inducer = np.asarray(inducer_uM, dtype=float)
    powered = np.power(inducer, sensor.hill_n)
    half_powered = sensor.half_max_uM**sensor.hill_n
    return sensor.leak_rpu + (sensor.max_rpu - sensor.leak_rpu) * powered / (
        half_powered + powered
    )


def complex_concentration(
    input_1_nM: np.ndarray | float,
    input_2_nM: np.ndarray | float,
    affinity_nM: float,
) -> np.ndarray:
    """Return the physical root of K*C=(input_1-C)*(input_2-C)."""

    input_1 = np.asarray(input_1_nM, dtype=float)
    input_2 = np.asarray(input_2_nM, dtype=float)
    total = input_1 + input_2 + affinity_nM
    discriminant = np.maximum(total * total - 4.0 * input_1 * input_2, 0.0)
    return 0.5 * (total - np.sqrt(discriminant))


def nominal_flux(
    candidate: Candidate,
    xi_1_rpu: np.ndarray | float,
    xi_2_rpu: np.ndarray | float,
) -> np.ndarray:
    """Evaluate the nominal complex-mediated output flux in nM/min."""

    core = candidate.core
    input_1 = core.gain_1_nM_per_rpu * np.asarray(xi_1_rpu, dtype=float)
    input_2 = core.gain_2_nM_per_rpu * np.asarray(xi_2_rpu, dtype=float)
    return core.output_rate_per_min * complex_concentration(
        input_1, input_2, core.affinity_nM
    )


def initial_flux_bounds(
    candidate: Candidate,
    xi_1_rpu: np.ndarray | float,
    xi_2_rpu: np.ndarray | float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the fixed-parameter flux envelope.

    The initial design parameters are fixed. The two returned arrays coincide;
    the pair interface is retained because the plotting and contract code use a
    common lower/upper-envelope representation.
    """

    flux = nominal_flux(candidate, xi_1_rpu, xi_2_rpu)
    return flux, flux


def _invf_sica_turnover_flux(
    candidate: Candidate,
    xi_1_rpu: np.ndarray | float,
    xi_2_rpu: np.ndarray | float,
) -> np.ndarray:
    """Reduced steady-state InvF/SicA/SipC model using Temme et al. rates.

    The two promoter inputs retain the existing effective abundance gains.
    These abundances are converted to production fluxes using the published
    InvF/SicA degradation rates. SipC production uses alpha_z=90 proteins per
    transcript and k0=1 transcript/min from Temme et al. Fast SicA:SipC and
    InvF:SicA binding are combined with the steady-state degradation/secretion
    balances. A vectorized bisection solves for free SicA.
    """

    core = candidate.core
    params = REDESIGN_PARAMETERS[core.name]

    xi_1 = np.asarray(xi_1_rpu, dtype=float)
    xi_2 = np.asarray(xi_2_rpu, dtype=float)
    xi_1, xi_2 = np.broadcast_arrays(xi_1, xi_2)

    gamma_x = params["gamma_x_per_min"]
    gamma_xy = params["gamma_xy_per_min"]
    gamma_y = params["gamma_y_per_min"]
    gamma_z = params["gamma_z_per_min"]
    gamma_yz = params["gamma_yz_per_min"]
    secretion = params["secretion_rate_per_min"]
    k_yz = params["sink_affinity_nM"]

    # Existing expression gains define the no-interaction steady-state scales.
    x_reference = core.gain_1_nM_per_rpu * xi_1
    y_reference = core.gain_2_nM_per_rpu * xi_2
    production_x = gamma_x * x_reference
    production_y = gamma_y * y_reference

    # Temme et al.: alpha_z=90 proteins/transcript and k0=1 transcript/min.
    production_z = (
        params["sipc_proteins_per_transcript"]
        * params["sipc_transcription_rate_per_min"]
        * MOLECULE_TO_NM_AT_1_FL
    )

    # For a trial free SicA level y, the X and Z steady-state equations are
    # explicit because C_XY=x*y/K_XY and C_YZ=y*z/K_YZ.
    y_low = np.zeros_like(x_reference, dtype=float)
    y_high = y_reference.copy()

    for _ in range(70):
        y_free = 0.5 * (y_low + y_high)
        x_free = production_x / (
            gamma_x + gamma_xy * y_free / core.affinity_nM
        )
        c_xy = x_free * y_free / core.affinity_nM

        z_free = production_z / (
            gamma_z + (gamma_yz + secretion) * y_free / k_yz
        )
        c_yz = y_free * z_free / k_yz

        residual = (
            gamma_y * y_free
            + gamma_xy * c_xy
            + gamma_yz * c_yz
            - production_y
        )
        y_high = np.where(residual > 0.0, y_free, y_high)
        y_low = np.where(residual <= 0.0, y_free, y_low)

    y_free = 0.5 * (y_low + y_high)
    x_free = production_x / (
        gamma_x + gamma_xy * y_free / core.affinity_nM
    )
    c_xy = x_free * y_free / core.affinity_nM
    return core.output_rate_per_min * c_xy


def redesigned_flux_bounds(
    candidate: Candidate,
    xi_1_rpu: np.ndarray | float,
    xi_2_rpu: np.ndarray | float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the fixed-parameter flux envelope for a redesigned architecture.

    The selected redesign parameters are fixed, while the input variables range
    over the complete contract region. The two returned arrays therefore
    coincide.
    """

    core = candidate.core
    xi_1 = np.asarray(xi_1_rpu, dtype=float)
    xi_2 = np.asarray(xi_2_rpu, dtype=float)
    redesign = REDESIGNS[core.name]
    params = REDESIGN_PARAMETERS[core.name]

    if redesign.mechanism == "hrpv_threshold":
        complex_value = complex_concentration(
            core.gain_1_nM_per_rpu * xi_1,
            core.gain_2_nM_per_rpu * xi_2,
            core.affinity_nM,
        )
        active_complex = np.maximum(
            complex_value - params["hrpv_nM"], 0.0
        )
        flux = core.output_rate_per_min * active_complex
        return flux, flux

    if redesign.mechanism == "synzip_affinity":
        complex_value = complex_concentration(
            core.gain_1_nM_per_rpu * xi_1,
            core.gain_2_nM_per_rpu * xi_2,
            params["affinity_nM"],
        )
        flux = core.output_rate_per_min * complex_value
        return flux, flux

    if redesign.mechanism == "sipc_titration":
        flux = _invf_sica_turnover_flux(candidate, xi_1, xi_2)
        return flux, flux

    raise ValueError(f"Unknown redesign mechanism: {redesign.mechanism}")


def redesigned_condition_flux_interval(
    candidate: Candidate,
    xi1_bounds: tuple[float, float],
    xi2_bounds: tuple[float, float],
) -> tuple[float, float]:
    """Return redesigned flux extrema over one rectangular mode."""

    lower, _ = redesigned_flux_bounds(
        candidate, xi1_bounds[0], xi2_bounds[0]
    )
    _, upper = redesigned_flux_bounds(
        candidate, xi1_bounds[1], xi2_bounds[1]
    )
    return float(lower), float(upper)


def condition_flux_interval(
    candidate: Candidate,
    xi1_bounds: tuple[float, float],
    xi2_bounds: tuple[float, float],
) -> tuple[float, float]:
    """Return fixed-parameter flux extrema over one rectangular mode."""

    lower, _ = initial_flux_bounds(
        candidate, xi1_bounds[0], xi2_bounds[0]
    )
    _, upper = initial_flux_bounds(
        candidate, xi1_bounds[1], xi2_bounds[1]
    )
    return float(lower), float(upper)


def infer_regime(candidate: Candidate, xi_1_rpu: float, xi_2_rpu: float) -> Regime:
    """Label the largest structural term at a nominal operating point."""

    core = candidate.core
    input_1 = core.gain_1_nM_per_rpu * xi_1_rpu
    input_2 = core.gain_2_nM_per_rpu * xi_2_rpu
    complex_value = float(complex_concentration(input_1, input_2, core.affinity_nM))
    terms = {
        "input 1 excess": input_1 - complex_value,
        "input 2 excess": input_2 - complex_value,
        "affinity-dominated": core.affinity_nM,
    }
    return max(terms, key=terms.get)  # type: ignore[return-value]


def create_expression_contract(
    branch: int,
    sensor: Sensor,
    gain_nM_per_rpu: float,
) -> PolyhedralIoContract:
    """Create the fixed-gain promoter-activity to abundance contract."""

    if branch not in (1, 2):
        raise ValueError("branch must be 1 or 2")
    z_xi = f"z_xi{branch}"
    z_input = f"z_input{branch}"
    gain_low = gain_nM_per_rpu
    gain_high = gain_nM_per_rpu
    return PolyhedralIoContract.from_dict(
        {
            "input_vars": [z_xi],
            "output_vars": [z_input],
            "assumptions": [
                {
                    "coefficients": {z_xi: -1.0},
                    "constant": -math.log(sensor.leak_rpu),
                },
                {
                    "coefficients": {z_xi: 1.0},
                    "constant": math.log(sensor.linear_max_rpu),
                },
            ],
            "guarantees": [
                {
                    "coefficients": {z_xi: 1.0, z_input: -1.0},
                    "constant": -math.log(gain_low),
                },
                {
                    "coefficients": {z_input: 1.0, z_xi: -1.0},
                    "constant": math.log(gain_high),
                },
            ],
        }
    )


def create_finite_box_core_contract(
    candidate: Candidate,
    xi1_bounds: tuple[float, float],
    xi2_bounds: tuple[float, float],
) -> PolyhedralIoContract:
    """Create a globally valid fixed-parameter QSS contract on one finite box.

    The lower bound follows from C >= input_1*input_2/(K+input_1+input_2).
    The three upper bounds use C <= input_1, C <= input_2, and
    C <= input_1*input_2/K.
    """

    core = candidate.core
    input_1_low = core.gain_1_nM_per_rpu * xi1_bounds[0]
    input_1_high = core.gain_1_nM_per_rpu * xi1_bounds[1]
    input_2_low = core.gain_2_nM_per_rpu * xi2_bounds[0]
    input_2_high = core.gain_2_nM_per_rpu * xi2_bounds[1]
    affinity_low = core.affinity_nM
    affinity_high = core.affinity_nM
    rate_low = core.output_rate_per_min
    rate_high = core.output_rate_per_min
    denominator_high = affinity_high + input_1_high + input_2_high
    endpoint_lower = rate_low * float(
        complex_concentration(input_1_low, input_2_low, affinity_high)
    )
    endpoint_upper = rate_high * float(
        complex_concentration(input_1_high, input_2_high, affinity_low)
    )

    # Assumption boundaries are closed polyhedral inequalities. Keep the exact
    # shared boundaries so upstream gain guarantees and core assumptions meet
    # at equality without introducing a modeling tolerance.

    return PolyhedralIoContract.from_dict(
        {
            "input_vars": ["z_input1", "z_input2"],
            "output_vars": ["z_flux"],
            "assumptions": [
                {
                    "coefficients": {"z_input1": -1.0},
                    "constant": -math.log(input_1_low),
                },
                {
                    "coefficients": {"z_input1": 1.0},
                    "constant": math.log(input_1_high),
                },
                {
                    "coefficients": {"z_input2": -1.0},
                    "constant": -math.log(input_2_low),
                },
                {
                    "coefficients": {"z_input2": 1.0},
                    "constant": math.log(input_2_high),
                },
            ],
            "guarantees": [
                {
                    "coefficients": {"z_flux": -1.0},
                    "constant": -math.log(endpoint_lower),
                },
                {
                    "coefficients": {"z_flux": 1.0},
                    "constant": math.log(endpoint_upper),
                },
                {
                    "coefficients": {
                        "z_input1": 1.0,
                        "z_input2": 1.0,
                        "z_flux": -1.0,
                    },
                    "constant": math.log(denominator_high / rate_low),
                },
                {
                    "coefficients": {"z_flux": 1.0, "z_input1": -1.0},
                    "constant": math.log(rate_high),
                },
                {
                    "coefficients": {"z_flux": 1.0, "z_input2": -1.0},
                    "constant": math.log(rate_high),
                },
                {
                    "coefficients": {
                        "z_flux": 1.0,
                        "z_input1": -1.0,
                        "z_input2": -1.0,
                    },
                    "constant": math.log(rate_high / affinity_low),
                },
            ],
        }
    )


def create_redesigned_finite_box_core_contract(
    candidate: Candidate,
    xi1_bounds: tuple[float, float],
    xi2_bounds: tuple[float, float],
) -> PolyhedralIoContract:
    """Create a sound polyhedral outer contract for one redesigned mode.

    The finite input assumptions are the same expression-module abundance
    bounds used for the initial design. The mechanism-specific nonlinear model
    is enclosed by monotonic endpoint output bounds over that finite region.
    The resulting guarantee is polyhedral in log coordinates. If a hard HrpV
    threshold permits zero output, no positive lower guarantee is asserted.
    """

    core = candidate.core
    input_1_low = core.gain_1_nM_per_rpu * xi1_bounds[0]
    input_1_high = core.gain_1_nM_per_rpu * xi1_bounds[1]
    input_2_low = core.gain_2_nM_per_rpu * xi2_bounds[0]
    input_2_high = core.gain_2_nM_per_rpu * xi2_bounds[1]
    endpoint_lower, endpoint_upper = redesigned_condition_flux_interval(
        candidate, xi1_bounds, xi2_bounds
    )
    guarantees: list[dict[str, object]] = []
    if endpoint_lower > 0.0:
        guarantees.append(
            {
                "coefficients": {"z_flux": -1.0},
                "constant": -math.log(endpoint_lower),
            }
        )
    guarantees.append(
        {
            "coefficients": {"z_flux": 1.0},
            "constant": math.log(max(endpoint_upper, ZERO_FLUX_UPPER_EPS)),
        }
    )

    return PolyhedralIoContract.from_dict(
        {
            "input_vars": ["z_input1", "z_input2"],
            "output_vars": ["z_flux"],
            "assumptions": [
                {
                    "coefficients": {"z_input1": -1.0},
                    "constant": -math.log(input_1_low),
                },
                {
                    "coefficients": {"z_input1": 1.0},
                    "constant": math.log(input_1_high),
                },
                {
                    "coefficients": {"z_input2": -1.0},
                    "constant": -math.log(input_2_low),
                },
                {
                    "coefficients": {"z_input2": 1.0},
                    "constant": math.log(input_2_high),
                },
            ],
            "guarantees": guarantees,
        }
    )


def compose_redesigned_candidate(
    candidate: Candidate,
    xi1_bounds: tuple[float, float],
    xi2_bounds: tuple[float, float],
) -> tuple[
    PolyhedralIoContract,
    PolyhedralIoContract,
    PolyhedralIoContract,
    PolyhedralIoContract,
]:
    """Compose expression modules with the mechanism-specific redesigned core."""

    core = candidate.core
    c1 = create_expression_contract(
        1, candidate.sensor_1, core.gain_1_nM_per_rpu
    )
    c2 = create_expression_contract(
        2, candidate.sensor_2, core.gain_2_nM_per_rpu
    )
    c3 = create_redesigned_finite_box_core_contract(
        candidate, xi1_bounds, xi2_bounds
    )
    c12 = c1.compose(c2, vars_to_keep=["z_input1", "z_input2"])
    cfull = c12.compose(c3, vars_to_keep=["z_flux"])
    return c1, c2, c3, cfull


def compose_candidate(
    candidate: Candidate,
    xi1_bounds: tuple[float, float],
    xi2_bounds: tuple[float, float],
) -> tuple[
    PolyhedralIoContract,
    PolyhedralIoContract,
    PolyhedralIoContract,
    PolyhedralIoContract,
]:
    """Compose two lumped expression contracts with the catalytic core."""

    core = candidate.core
    c1 = create_expression_contract(
        1, candidate.sensor_1, core.gain_1_nM_per_rpu
    )
    c2 = create_expression_contract(
        2, candidate.sensor_2, core.gain_2_nM_per_rpu
    )
    c3 = create_finite_box_core_contract(
        candidate, xi1_bounds, xi2_bounds
    )
    c12 = c1.compose(c2, vars_to_keep=["z_input1", "z_input2"])
    cfull = c12.compose(c3, vars_to_keep=["z_flux"])
    return c1, c2, c3, cfull


def create_specification(condition: Condition) -> PolyhedralIoContract:
    """Create one polyhedral ON or OFF top-level specification."""

    if condition.guarantee_kind == "lower":
        guarantees = [
            {
                "coefficients": {"z_flux": -1.0},
                "constant": -math.log(condition.flux_threshold),
            }
        ]
    else:
        guarantees = [
            {
                "coefficients": {"z_flux": 1.0},
                "constant": math.log(condition.flux_threshold),
            }
        ]

    return PolyhedralIoContract.from_dict(
        {
            "input_vars": ["z_xi1", "z_xi2"],
            "output_vars": ["z_flux"],
            "assumptions": [
                {
                    "coefficients": {"z_xi1": -1.0},
                    "constant": -math.log(condition.xi1_bounds[0]),
                },
                {
                    "coefficients": {"z_xi1": 1.0},
                    "constant": math.log(condition.xi1_bounds[1]),
                },
                {
                    "coefficients": {"z_xi2": -1.0},
                    "constant": -math.log(condition.xi2_bounds[0]),
                },
                {
                    "coefficients": {"z_xi2": 1.0},
                    "constant": math.log(condition.xi2_bounds[1]),
                },
            ],
            "guarantees": guarantees,
        }
    )


def verify_condition(
    candidate: Candidate,
    condition: Condition,
) -> dict[str, object]:
    """Compose and check assumption coverage and conditional guarantees."""

    _, _, _, composed = compose_candidate(
        candidate, condition.xi1_bounds, condition.xi2_bounds
    )
    specification = create_specification(condition)
    assumptions_ok = specification.a <= composed.a
    guarantees_ok = (composed.g | specification.a) <= (
        specification.g | specification.a
    )
    return {
        "condition": condition.name,
        "assumptions": assumptions_ok,
        "guarantees": guarantees_ok,
        "refines": assumptions_ok and guarantees_ok,
        "composed": composed,
        "specification": specification,
    }


def verify_redesigned_condition(
    candidate: Candidate,
    condition: Condition,
) -> dict[str, object]:
    """Compose the redesigned core and check one fixed top-level mode."""

    _, _, _, composed = compose_redesigned_candidate(
        candidate, condition.xi1_bounds, condition.xi2_bounds
    )
    specification = create_specification(condition)
    assumptions_ok = specification.a <= composed.a
    guarantees_ok = (composed.g | specification.a) <= (
        specification.g | specification.a
    )
    return {
        "condition": condition.name,
        "assumptions": assumptions_ok,
        "guarantees": guarantees_ok,
        "refines": assumptions_ok and guarantees_ok,
        "composed": composed,
        "specification": specification,
    }


def make_top_level_conditions(
    on_box: tuple[tuple[float, float], tuple[float, float]] = DEFAULT_ON_BOX,
    off_interval: tuple[float, float] = DEFAULT_OFF_INTERVAL,
    on_threshold: float = DEFAULT_ON_THRESHOLD_NM_PER_MIN,
    off_threshold: float = DEFAULT_OFF_THRESHOLD_NM_PER_MIN,
) -> list[Condition]:
    """Return the fixed biologically motivated AND-gate design brief."""

    return [
        Condition("ON", on_box[0], on_box[1], "lower", on_threshold),
        Condition("OFF-1", off_interval, on_box[1], "upper", off_threshold),
        Condition("OFF-2", on_box[0], off_interval, "upper", off_threshold),
    ]


def summarize_candidate_envelopes(
    candidates: list[Candidate],
    conditions: list[Condition] | None = None,
) -> dict[str, object]:
    """Summarize worst-case candidate flux scales under the fixed design brief."""

    if conditions is None:
        conditions = make_top_level_conditions()
    on_condition = next(condition for condition in conditions if condition.name == "ON")
    off1_condition = next(
        condition for condition in conditions if condition.name == "OFF-1"
    )
    off2_condition = next(
        condition for condition in conditions if condition.name == "OFF-2"
    )

    representatives: dict[str, Candidate] = {}
    for candidate in candidates:
        representatives.setdefault(candidate.core.name, candidate)

    on_floors: dict[str, float] = {}
    off_ceilings: dict[str, float] = {}
    for core_name, candidate in representatives.items():
        on_floors[core_name] = condition_flux_interval(
            candidate, on_condition.xi1_bounds, on_condition.xi2_bounds
        )[0]
        off_1 = condition_flux_interval(
            candidate, off1_condition.xi1_bounds, off1_condition.xi2_bounds
        )[1]
        off_2 = condition_flux_interval(
            candidate, off2_condition.xi1_bounds, off2_condition.xi2_bounds
        )[1]
        off_ceilings[core_name] = max(off_1, off_2)

    return {
        "on_floors": on_floors,
        "off_ceilings": off_ceilings,
        "on_threshold": on_condition.flux_threshold,
        "off_threshold": off1_condition.flux_threshold,
    }


def summarize_redesigned_envelopes(
    candidates: list[Candidate],
    conditions: list[Condition] | None = None,
) -> dict[str, object]:
    """Summarize mechanism-specific redesign envelopes under the fixed design brief."""

    if conditions is None:
        conditions = make_top_level_conditions()
    on_condition = next(condition for condition in conditions if condition.name == "ON")
    off1_condition = next(
        condition for condition in conditions if condition.name == "OFF-1"
    )
    off2_condition = next(
        condition for condition in conditions if condition.name == "OFF-2"
    )

    representatives: dict[str, Candidate] = {}
    for candidate in candidates:
        representatives.setdefault(candidate.core.name, candidate)

    on_floors: dict[str, float] = {}
    off_ceilings: dict[str, float] = {}
    for core_name, candidate in representatives.items():
        on_floors[core_name] = redesigned_condition_flux_interval(
            candidate, on_condition.xi1_bounds, on_condition.xi2_bounds
        )[0]
        off_1 = redesigned_condition_flux_interval(
            candidate, off1_condition.xi1_bounds, off1_condition.xi2_bounds
        )[1]
        off_2 = redesigned_condition_flux_interval(
            candidate, off2_condition.xi1_bounds, off2_condition.xi2_bounds
        )[1]
        off_ceilings[core_name] = max(off_1, off_2)

    return {
        "on_floors": on_floors,
        "off_ceilings": off_ceilings,
        "on_threshold": on_condition.flux_threshold,
        "off_threshold": off1_condition.flux_threshold,
    }
