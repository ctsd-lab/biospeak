"""Equilibrium laws from reaction balances and conserved promoter pools."""

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import sympy as sp
from sympy.solvers.solveset import NonlinearError

SB = "http://www.sbml.org/sbml/level3/version2/core"
MA = "http://www.w3.org/1998/Math/MathML"


def parse_reference_sbml(path):
    """Independent parser, restricted to the elementary template MathML above."""
    root = ET.parse(path).getroot()
    m = root.find(f"{{{SB}}}model")
    syms = {
        s.attrib["id"]: sp.Symbol(s.attrib["id"], positive=True)
        for s in m.find(f"{{{SB}}}listOfSpecies")
    }
    par_nodes = m.find(f"{{{SB}}}listOfParameters")
    params = {
        p.attrib["id"]: sp.Symbol(p.attrib["id"], positive=True) for p in par_nodes
    }
    vals = {params[p.attrib["id"]]: sp.Rational(p.attrib["value"]) for p in par_nodes}

    def parse(e):
        tag = e.tag.split("}")[-1]
        if tag == "ci":
            return {**syms, **params}[e.text.strip()]
        if tag == "cn":
            return sp.Rational(e.text)
        op = e[0].tag.split("}")[-1]
        args = [parse(c) for c in e[1:]]
        if op == "times":
            return sp.Mul(*args)
        if op == "plus":
            return sp.Add(*args)
        if op == "power":
            return args[0] ** args[1]
        raise ValueError(op)

    species = list(syms)
    rxs = list(m.find(f"{{{SB}}}listOfReactions"))
    N = sp.zeros(len(species), len(rxs))
    rates = []
    for j, r in enumerate(rxs):
        for group, sign in [("listOfReactants", -1), ("listOfProducts", 1)]:
            side = r.find(f"{{{SB}}}{group}")
            if side is not None:
                for item in side:
                    N[species.index(item.attrib["species"]), j] += sign * sp.Rational(
                        item.attrib["stoichiometry"]
                    )
        rates.append(parse(r.find(f"{{{SB}}}kineticLaw").find(f"{{{MA}}}math")[0]))
    boundary = [
        s.attrib["id"]
        for s in m.find(f"{{{SB}}}listOfSpecies")
        if s.attrib["boundaryCondition"] == "true"
    ]
    f = N * sp.Matrix(rates)
    for s in boundary:
        f[species.index(s)] = 0
    return dict(
        species=syms,
        parameters=params,
        parameter_values=vals,
        N=N,
        rates=sp.Matrix(rates),
        f=f,
        boundary=boundary,
    )


@dataclass
class DerivedModule:
    name: str
    balances: dict
    pools: dict
    solution: dict
    output: sp.Expr
    normalized: sp.Expr
    regulated: sp.Expr
    activity_symbols: tuple
    proof: dict
    source: str


def solve_equilibrium(balances, internal, pools):
    """Solve equilibrium balances linear in internal variables, with conserved pools."""
    equations = [balances[x] for x in internal] + list(pools.values())
    try:
        matrix, rhs = sp.linear_eq_to_matrix(equations, internal)
    except NonlinearError as exc:
        raise ValueError(
            "The equilibrium equations must be linear in internal variables."
        ) from exc
    pivot_rows = matrix.T.rref()[1][: len(internal)]
    if len(pivot_rows) != len(internal):
        raise ValueError("The equilibrium coefficient matrix has deficient rank.")
    witness = sp.factor(matrix[list(pivot_rows), :].det())
    if not (positive_rational(witness) or positive_rational(-witness)):
        raise ValueError(
            "Nonsingularity throughout the positive parameter domain is unresolved."
        )
    roots = sp.linsolve((matrix, rhs), internal)
    if roots is sp.EmptySet or len(roots) != 1:
        raise ValueError("A unique equilibrium was not established.")
    root = tuple(roots)[0]
    solution = {x: sp.factor(value) for x, value in zip(internal, root)}
    if any(value.free_symbols.intersection(internal) for value in solution.values()):
        raise ValueError(
            "The proposed pools leave an unresolved equilibrium direction."
        )
    residuals = [sp.cancel(e.subs(solution, simultaneous=True)) for e in equations]
    if any(r != 0 for r in residuals):
        raise ValueError("The equilibrium does not satisfy all balances and pools.")
    return solution, dict(
        unique_in_class=True,
        balance_identities=len(residuals),
        all_residuals_zero=True,
        rank_witness=str(witness),
    )


def positive_rational(expr):
    """Sufficient positivity certificate for a rational expression on positive inputs."""
    numerator, denominator = sp.fraction(sp.cancel(expr))
    symbols = sorted(expr.free_symbols, key=str)
    for part in [numerator, denominator]:
        if not symbols:
            if part <= 0:
                return False
        else:
            coeffs = sp.Poly(sp.expand(part), *symbols).coeffs()
            if (
                not coeffs
                or all(c == 0 for c in coeffs)
                or any(c.is_nonnegative is not True for c in coeffs)
            ):
                return False
    return True


def derive_modules(root=None):
    if root is None:
        root = Path(__file__).parent / "data"
    q, qR, qS, qC = sp.symbols("q q_R q_S q_C", positive=True)
    alpha, T, kt, kl, dm, dp = sp.symbols("alpha T k_t k_l d_m d_p", positive=True)
    kb, ku, kbR, kuR, kbS, kuS, kbC, kuC = sp.symbols(
        "k_b k_u k_bR k_uR k_bS k_uS k_bC k_uC", positive=True
    )
    definitions = {
        "activation": dict(
            inputs=("A",),
            dna=("D0", "D1"),
            output="P",
            activity=(q,),
            mapping={
                "kb_A": kb,
                "ku_A": ku,
                "tx_D0": kt * alpha,
                "tx_D1": kt * (alpha + 1),
            },
            ports={"A": ku * q / kb},
        ),
        "and_occupancy": dict(
            inputs=("R", "S"),
            dna=("D00", "D10", "D01", "D11"),
            output="C",
            activity=(qR, qS),
            mapping={"kb_R": kbR, "ku_R": kuR, "kb_S": kbS, "ku_S": kuS, "tx_D11": kt},
            ports={"R": kuR * qR / kbR, "S": kuS * qS / kbS},
        ),
        "not_occupancy": dict(
            inputs=("C",),
            dna=("N0", "N1"),
            output="G",
            activity=(qC,),
            mapping={
                "kb_C": kbC,
                "ku_C": kuC,
                "tx_N0": kt * (1 + alpha),
                "tx_N1": kt * alpha,
            },
            ports={"C": kuC * qC / kbC},
        ),
    }
    modules = {}
    for name, definition in definitions.items():
        loaded = parse_reference_sbml(root / "models" / (name + ".xml"))
        x = loaded["species"]
        replacements = {
            sp.Symbol(k, positive=True): v for k, v in definition["mapping"].items()
        }
        replacements.update(
            {
                sp.Symbol("ktl", positive=True): kl,
                sp.Symbol("dm", positive=True): dm,
                sp.Symbol("dp", positive=True): dp,
            }
        )
        replacements.update({x[k]: v for k, v in definition["ports"].items()})
        balances = {
            x[s]: sp.factor(f.subs(replacements)) for s, f in zip(x, loaded["f"])
        }
        pool = sum(x[d] for d in definition["dna"]) - T
        if sp.simplify(sum(balances[x[d]] for d in definition["dna"])) != 0:
            raise ValueError("Promoter conservation is not implied by the reactions.")
        internal = [x[s] for s in x if s not in definition["inputs"]]
        solution, proof = solve_equilibrium(balances, internal, {"T": pool})
        output = solution[x[definition["output"]]]
        normalized = sp.factor(output * dm * dp / (T * kt * kl))
        regulated = (
            sp.factor(normalized - alpha) if name != "and_occupancy" else normalized
        )
        proof["positive_equilibrium"] = all(
            positive_rational(v) for v in solution.values()
        )
        proof["pool_conserved"] = True
        proof["rate_identifications"] = {
            str(k): str(v) for k, v in replacements.items()
        }
        proof["monotonicity"] = {}
        for z in definition["activity"]:
            derivative = sp.factor(sp.diff(normalized, z))
            sign = (
                1
                if positive_rational(derivative)
                else -1
                if positive_rational(-derivative)
                else 0
            )
            proof["monotonicity"][str(z)] = dict(sign=sign, derivative=str(derivative))
        if not proof["positive_equilibrium"]:
            raise ValueError("Positivity could not be established.")
        modules[name] = DerivedModule(
            name,
            balances,
            {"T": pool},
            solution,
            output,
            normalized,
            regulated,
            definition["activity"],
            proof,
            "bundled SBML reaction snapshot",
        )
    return modules
