"""Isolated inherited defect measurements and facet/height conventions.

Only the pure collector selects required constraints. Code-only diagnostics
are not physical evidence; they are excluded or explicitly marked unknown.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
import math
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple
import numpy as np
from pymatgen.core import Composition, Element, Structure
from .paper_serialization import optional_bool

@dataclass
class ConstraintCheck:
    """Single hard-constraint diagnostic."""

    name: str
    satisfied: bool
    value: Any = None
    threshold: Any = None
    message: str = ""
    group: str = "generic"
    hard: bool = True
    score: Optional[float] = None

    def __post_init__(self):
        self.satisfied = optional_bool(self.satisfied)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def normalized_score(self) -> float:
        """Return a clipped partial score for CSR aggregation."""
        if self.score is None:
            return 1.0 if self.satisfied else 0.0
        try:
            value = float(self.score)
        except (TypeError, ValueError):
            return 1.0 if self.satisfied else 0.0
        return max(0.0, min(1.0, value))


def _defect_constraints(
    structure: Structure,
    task: Dict[str, Any],
    result_metadata: Optional[Dict[str, Any]] = None,
) -> List[ConstraintCheck]:
    checks: List[ConstraintCheck] = []
    task_input = task.get("input", task)
    validation = task.get("validation") or {}
    expected = task.get("expected_properties", {})
    code = _generated_code(result_metadata)
    composition = _element_composition(structure)
    elements = set(composition)
    defect_type = str(task_input.get("defect_type", "")).lower()

    host_material = task_input.get("host_material")
    if host_material:
        host_elements = _elements_from_formula_label(host_material)
        checks.append(_elements_present_check("host_elements", elements, host_elements, group="identity"))

    composition_contains = expected.get("composition_contains", [])
    if composition_contains:
        checks.append(_elements_present_check("composition_contains", elements, composition_contains, group="operation"))

    original_elements = expected.get("original_elements", [])
    if original_elements:
        checks.append(_elements_present_check("original_elements", elements, original_elements, group="identity"))

    min_atoms = expected.get("min_atoms")
    if min_atoms is not None:
        checks.append(
            ConstraintCheck(
                name="min_atoms",
                satisfied=len(structure) >= int(min_atoms),
                value=len(structure),
                threshold=int(min_atoms),
                group="size",
            )
        )

    dopant = task_input.get("dopant") or task_input.get("substitute_element")
    if dopant and (task_input.get("dopant_count") is not None or expected.get(f"{dopant}_count") is not None):
        requested = task_input.get("dopant_count", expected.get(f"{dopant}_count"))
        checks.append(_element_exact_count_check(structure, str(dopant), float(requested), "operation"))

    defect_count = expected.get("defect_count")
    if dopant and defect_count is not None:
        actual = float(composition.get(dopant, 0.0))
        checks.append(
            ConstraintCheck(
                name="defect_count",
                satisfied=abs(actual - float(defect_count)) <= 1e-6,
                value=actual,
                threshold=float(defect_count),
                group="operation",
            )
        )

    for key, value in expected.items():
        if value is True and key.startswith("contains_"):
            element = key.removeprefix("contains_")
            if _is_element_symbol(element):
                checks.append(_elements_present_check(key, elements, [element], group="operation"))

        count_match = re.fullmatch(r"([A-Z][a-z]?)_count", str(key))
        if count_match and isinstance(value, (int, float)):
            element = count_match.group(1)
            if element != dopant:
                checks.append(_element_exact_count_check(structure, element, float(value), "operation"))

    dopant_site = task_input.get("dopant_site")
    if dopant and dopant_site:
        checks.append(
            ConstraintCheck(
                name="substitution_species",
                satisfied=dopant in elements and dopant_site in elements,
                value=sorted(elements),
                threshold={"dopant": dopant, "site": dopant_site},
                group="operation",
            )
        )

    reference_counts = _expected_pristine_counts(task_input, expected)
    if reference_counts:
        checks.extend(_reference_composition_checks(structure, task_input, expected, reference_counts))

    if expected.get("charge_neutral") or validation.get("check_charge_balance"):
        checks.append(_charge_neutrality_check(structure))

    if expected.get("stoichiometry_preserved"):
        checks.append(_stoichiometry_preserved_check(structure, host_material))

    if expected.get("is_layered") or expected.get("layer_count") is not None:
        checks.append(
            _layered_structure_check(
                structure=structure,
                expected_layers=expected.get("layer_count"),
                group="geometry",
            )
        )

    vacancy_requested = (
        "vacancy" in defect_type
        or "schottky" in defect_type
        or "frenkel" in defect_type
        or expected.get("vacancy_count") is not None
        or any(str(key).endswith("_vacancy_count") for key in expected)
    )
    if vacancy_requested:
        checks.append(_defect_code_operation_check(
            code,
            task_input,
            expected,
            operation="vacancy",
            hard=False,
        ))

    if "antisite" in defect_type or expected.get("antisite_count") is not None:
        checks.append(_defect_code_operation_check(
            code,
            task_input,
            expected,
            operation="antisite",
            hard=False,
        ))

    if "interstitial" in defect_type or "frenkel" in defect_type:
        checks.append(_defect_code_operation_check(
            code,
            task_input,
            expected,
            operation="interstitial",
            hard=False,
        ))

    if expected.get("dumbbell_configuration") or "dumbbell" in defect_type:
        checks.append(_dumbbell_geometry_check(structure, task_input, validation))
        if task_input.get("dumbbell_direction"):
            checks.append(
                _code_vector_check(
                name="dumbbell_direction",
                code=code,
                vector=task_input["dumbbell_direction"],
                group="geometry",
                keywords=["dumbbell", "direction"],
                hard=False,
            )
        )

    if expected.get("linear_NVN_arrangement") or defect_type == "nvn_center":
        checks.append(_linear_dopant_arrangement_check(structure, task_input))

    if expected.get("Ge_adjacent_to_vacancy") or "split_vacancy" in defect_type:
        checks.append(_dopant_undercoordination_check(structure, task_input))

    interstitial_site = task_input.get("interstitial_site")
    if interstitial_site and expected.get("interstitial_at_octahedral"):
        checks.append(
            _code_keyword_check(
                name="interstitial_site",
                code=code,
                required=[str(interstitial_site)],
                group="geometry",
                message="Code-derived proxy for requested interstitial site.",
                hard=False,
            )
        )

    if expected.get("min_vacancy_distance") is not None:
        checks.append(
            _code_numeric_threshold_check(
                name="min_vacancy_distance",
                code=code,
                value=float(expected["min_vacancy_distance"]),
                group="geometry",
                keywords=["distance", "vac"],
                hard=False,
            )
        )

    return checks


def _count_z_layers_for_elements(structure: Structure, elements: Iterable[str]) -> int:
    selected = {
        str(element)
        for element in elements
        if element
    }
    z_coords = sorted(float(site.coords[2]) for site in structure if str(site.specie) in selected)
    if not z_coords:
        return 0

    layers = 1
    previous = z_coords[0]
    for z_coord in z_coords[1:]:
        if z_coord - previous > 0.75:
            layers += 1
        previous = z_coord
    return layers


def _generated_code(result_metadata: Optional[Dict[str, Any]]) -> str:
    if not result_metadata:
        return ""
    value = result_metadata.get("generated_code", "")
    return value if isinstance(value, str) else str(value)


def _element_composition(structure: Structure) -> Dict[str, float]:
    """Return composition keyed by plain element symbols, not oxidation-state labels."""
    counts: Dict[str, float] = {}
    for site in structure:
        specie = site.specie
        symbol = getattr(specie, "symbol", str(specie))
        counts[str(symbol)] = counts.get(str(symbol), 0.0) + 1.0
    return counts


def _is_element_symbol(value: str) -> bool:
    try:
        Element(str(value))
        return True
    except Exception:
        return False


def _element_exact_count_check(
    structure: Structure,
    element: str,
    expected_count: float,
    group: str,
) -> ConstraintCheck:
    composition = _element_composition(structure)
    actual = float(composition.get(element, 0.0))
    satisfied = abs(actual - expected_count) <= 1e-6
    score = max(0.0, 1.0 - abs(actual - expected_count) / max(expected_count, 1.0))
    return ConstraintCheck(
        name=f"{element}_count",
        satisfied=satisfied,
        value=actual,
        threshold=expected_count,
        group=group,
        score=score,
    )


def _expected_pristine_counts(
    task_input: Dict[str, Any],
    expected: Dict[str, Any],
) -> Dict[str, float]:
    original = expected.get("original_composition")
    if not isinstance(original, dict):
        return {}
    multiplier = _supercell_multiplier(task_input.get("supercell"))
    return {str(element): float(count) * multiplier for element, count in original.items()}


def _supercell_multiplier(supercell: Any) -> int:
    if not isinstance(supercell, (list, tuple)) or not supercell:
        return 1
    multiplier = 1
    for value in supercell:
        try:
            multiplier *= max(int(value), 1)
        except Exception:
            return 1
    return multiplier


def _reference_composition_checks(
    structure: Structure,
    task_input: Dict[str, Any],
    expected: Dict[str, Any],
    reference_counts: Dict[str, float],
) -> List[ConstraintCheck]:
    checks: List[ConstraintCheck] = []
    composition = _element_composition(structure)

    for key, value in expected.items():
        less_match = re.fullmatch(r"final_has_less_([A-Z][a-z]?)", str(key))
        if less_match and value is True:
            element = less_match.group(1)
            actual = float(composition.get(element, 0.0))
            reference = float(reference_counts.get(element, 0.0))
            checks.append(
                ConstraintCheck(
                    name=key,
                    satisfied=actual < reference,
                    value=actual,
                    threshold={"<": reference},
                    group="operation",
                    score=1.0 if actual < reference else 0.0,
                )
            )

        unchanged_match = re.fullmatch(r"([A-Z][a-z]?)_count_unchanged", str(key))
        if unchanged_match and value is True:
            element = unchanged_match.group(1)
            actual = float(composition.get(element, 0.0))
            reference = float(reference_counts.get(element, 0.0))
            checks.append(
                ConstraintCheck(
                    name=key,
                    satisfied=abs(actual - reference) <= 1e-6,
                    value=actual,
                    threshold=reference,
                    group="operation",
                    score=max(0.0, 1.0 - abs(actual - reference) / max(reference, 1.0)),
                )
            )

        net_match = re.fullmatch(r"net_([A-Z][a-z]?)_change", str(key))
        if net_match and isinstance(value, (int, float)):
            element = net_match.group(1)
            actual_change = float(composition.get(element, 0.0)) - float(reference_counts.get(element, 0.0))
            checks.append(
                ConstraintCheck(
                    name=key,
                    satisfied=abs(actual_change - float(value)) <= 1e-6,
                    value=actual_change,
                    threshold=float(value),
                    group="operation",
                    score=max(0.0, 1.0 - abs(actual_change - float(value)) / max(abs(float(value)), 1.0)),
                )
            )

    vacancy_element = task_input.get("vacancy_element") or task_input.get("defect_element")
    defect_count = expected.get("defect_count")
    if vacancy_element and defect_count is not None:
        reference = float(reference_counts.get(str(vacancy_element), 0.0))
        actual = float(composition.get(str(vacancy_element), 0.0))
        deficit = reference - actual
        checks.append(
            ConstraintCheck(
                name="vacancy_count",
                satisfied=abs(deficit - float(defect_count)) <= 1e-6,
                value=deficit,
                threshold=float(defect_count),
                group="operation",
                score=max(0.0, 1.0 - abs(deficit - float(defect_count)) / max(float(defect_count), 1.0)),
            )
        )

    return checks


def _charge_neutrality_check(structure: Structure) -> ConstraintCheck:
    composition = _element_composition(structure)
    states_by_element: List[Tuple[str, Tuple[int, ...]]] = []
    for element in composition:
        try:
            states = tuple(int(state) for state in Element(element).common_oxidation_states[:4])
            if not states:
                states = tuple(int(state) for state in Element(element).oxidation_states[:4])
        except Exception:
            states = ()
        if not states:
            return ConstraintCheck(
                name="charge_neutral",
                satisfied=False,
                value={"element": element, "reason": "no oxidation states"},
                threshold=0.0,
                group="chemistry",
                score=0.0,
            )
        states_by_element.append((element, states))

    best_charge = math.inf
    # These benchmarks have small element sets; cap state choices to avoid
    # turning charge balance into a combinatorial search for unusual formulas.
    choices = [states[:4] for _, states in states_by_element[:6]]
    for state_tuple in _cartesian_product_limited(choices, max_combinations=4096):
        charge = 0.0
        for (element, _), ox_state in zip(states_by_element, state_tuple):
            charge += composition[element] * ox_state
        best_charge = min(best_charge, abs(charge))
        if best_charge <= 1e-6:
            break

    total_atoms = sum(composition.values()) or 1.0
    score = max(0.0, 1.0 - best_charge / total_atoms)
    return ConstraintCheck(
        name="charge_neutral",
        satisfied=best_charge <= 1e-6,
        value=best_charge if math.isfinite(best_charge) else None,
        threshold=0.0,
        group="chemistry",
        score=score,
    )


def _cartesian_product_limited(values: List[Tuple[int, ...]], max_combinations: int) -> Iterable[Tuple[int, ...]]:
    if not values:
        return []

    def _walk(index: int, prefix: Tuple[int, ...], yielded: List[int]) -> Iterable[Tuple[int, ...]]:
        if yielded[0] >= max_combinations:
            return
        if index == len(values):
            yielded[0] += 1
            yield prefix
            return
        for value in values[index]:
            yield from _walk(index + 1, prefix + (value,), yielded)

    return _walk(0, tuple(), [0])


def _stoichiometry_preserved_check(structure: Structure, host_material: Optional[str]) -> ConstraintCheck:
    if not host_material:
        return ConstraintCheck(
            name="stoichiometry_preserved",
            satisfied=False,
            value=None,
            threshold="host formula",
            group="chemistry",
            score=0.0,
        )

    actual = _normalized_element_ratio(_element_composition(structure))
    expected = _normalized_element_ratio(Composition(str(host_material)).as_dict())
    elements = set(actual) | set(expected)
    deviations = [abs(actual.get(element, 0.0) - expected.get(element, 0.0)) for element in elements]
    max_deviation = max(deviations) if deviations else 1.0
    return ConstraintCheck(
        name="stoichiometry_preserved",
        satisfied=max_deviation <= 0.05,
        value=actual,
        threshold=expected,
        group="chemistry",
        score=max(0.0, 1.0 - max_deviation / 0.2),
    )


def _normalized_element_ratio(counts: Dict[str, Any]) -> Dict[str, float]:
    numeric = {str(element): float(count) for element, count in counts.items() if float(count) > 0.0}
    total = sum(numeric.values()) or 1.0
    return {element: round(count / total, 4) for element, count in numeric.items()}


def _layered_structure_check(
    structure: Structure,
    expected_layers: Optional[Any],
    group: str,
) -> ConstraintCheck:
    elements = set(_element_composition(structure))
    layer_count = _count_z_layers_for_elements(structure, elements)
    threshold = int(expected_layers) if expected_layers is not None else 2
    score = min(layer_count / max(float(threshold), 1.0), 1.0)
    return ConstraintCheck(
        name="layered_structure",
        satisfied=layer_count >= threshold,
        value=layer_count,
        threshold={">=": threshold},
        group=group,
        score=score,
    )


def _defect_code_operation_check(
    code: str,
    task_input: Dict[str, Any],
    expected: Dict[str, Any],
    operation: str,
    hard: bool = True,
) -> ConstraintCheck:
    lower = code.lower()
    if operation == "vacancy":
        op_terms = ["remove_sites", "remove_site", ".remove(", " pop(", "del "]
        elements = list(task_input.get("vacancy_types") or expected.get("vacancy_types") or [])
        for field in ("vacancy_element", "defect_element"):
            if task_input.get(field):
                elements.append(str(task_input[field]))
        required_terms = [str(element).lower() for element in elements if element]
    elif operation == "interstitial":
        op_terms = ["append", "insert", "interstitial"]
        element = task_input.get("interstitial_element") or task_input.get("defect_element")
        required_terms = [str(element).lower()] if element else []
    else:
        op_terms = ["replace", "substitut", "swap", "exchange"]
        atoms = task_input.get("swap_atoms") or []
        required_terms = [str(atom).lower() for atom in atoms]

    op_ok = bool(code) and any(term in lower for term in op_terms)
    terms_ok = all(term in lower for term in required_terms)
    satisfied = op_ok and terms_ok
    return ConstraintCheck(
        name=f"{operation}_operation",
        satisfied=satisfied,
        value={"operation_terms": op_ok, "required_terms": required_terms},
        threshold={"operation": operation, "code_derived": True},
        message="Code-derived proxy for a defect operation that is not uniquely recoverable from the final structure.",
        group="operation",
        hard=hard,
        score=1.0 if satisfied else (0.5 if op_ok else 0.0),
    )


def _dumbbell_geometry_check(
    structure: Structure,
    task_input: Dict[str, Any],
    validation: Dict[str, Any],
) -> ConstraintCheck:
    interstitial = task_input.get("interstitial_element")
    host_elements = set(_elements_from_formula_label(task_input.get("host_material", "")))
    if not interstitial or not host_elements:
        return ConstraintCheck(
            name="dumbbell_geometry",
            satisfied=False,
            value=None,
            threshold="interstitial-host distance",
            group="geometry",
            score=0.0,
        )

    interstitial_indices = [idx for idx, site in enumerate(structure) if str(site.specie.symbol) == interstitial]
    host_indices = [idx for idx, site in enumerate(structure) if str(site.specie.symbol) in host_elements]
    if not interstitial_indices or not host_indices:
        return ConstraintCheck(
            name="dumbbell_geometry",
            satisfied=False,
            value={"interstitial_count": len(interstitial_indices), "host_count": len(host_indices)},
            threshold="interstitial-host distance",
            group="geometry",
            score=0.0,
        )

    min_distance = min(float(structure.get_distance(i, j)) for i in interstitial_indices for j in host_indices)
    lower = float(validation.get("min_Fe_C_distance", validation.get("min_dumbbell_distance", 1.2)))
    upper = float(validation.get("max_Fe_C_distance", validation.get("max_dumbbell_distance", 2.8)))
    satisfied = lower <= min_distance <= upper
    if satisfied:
        score = 1.0
    elif min_distance < lower:
        score = min_distance / max(lower, 1e-6)
    else:
        score = max(0.0, 1.0 - (min_distance - upper) / max(upper, 1e-6))
    return ConstraintCheck(
        name="dumbbell_geometry",
        satisfied=satisfied,
        value=min_distance,
        threshold={"range": [lower, upper]},
        group="geometry",
        score=score,
    )


def _linear_dopant_arrangement_check(structure: Structure, task_input: Dict[str, Any]) -> ConstraintCheck:
    dopant = task_input.get("dopant")
    alignment = task_input.get("alignment")
    dopant_indices = [idx for idx, site in enumerate(structure) if getattr(site.specie, "symbol", str(site.specie)) == dopant]
    expected_count = int(task_input.get("dopant_count", 2))
    if len(dopant_indices) != expected_count:
        return ConstraintCheck(
            name="linear_dopant_arrangement",
            satisfied=False,
            value={"dopant_count": len(dopant_indices)},
            threshold={"dopant_count": expected_count},
            group="geometry",
            score=max(0.0, 1.0 - abs(len(dopant_indices) - expected_count) / max(expected_count, 1)),
        )
    if not alignment or len(dopant_indices) != 2:
        return ConstraintCheck(
            name="linear_dopant_arrangement",
            satisfied=True,
            value={"dopant_count": len(dopant_indices)},
            threshold={"dopant_count": expected_count},
            group="geometry",
            score=1.0,
        )

    vector = np.asarray(structure[dopant_indices[1]].coords - structure[dopant_indices[0]].coords, dtype=float)
    target = np.asarray(alignment, dtype=float)
    if np.linalg.norm(vector) <= 1e-8 or np.linalg.norm(target) <= 1e-8:
        cosine = 0.0
    else:
        cosine = float(abs(np.dot(vector, target) / (np.linalg.norm(vector) * np.linalg.norm(target))))
    return ConstraintCheck(
        name="linear_dopant_arrangement",
        satisfied=cosine >= 0.9,
        value={"cosine_to_alignment": cosine, "dopant_count": len(dopant_indices)},
        threshold={">=": 0.9, "alignment": alignment},
        group="geometry",
        score=max(0.0, min(cosine, 1.0)),
    )


def _dopant_undercoordination_check(structure: Structure, task_input: Dict[str, Any]) -> ConstraintCheck:
    dopant = task_input.get("dopant")
    host_elements = set(_elements_from_formula_label(task_input.get("host_material", ""))) - {str(dopant)}
    dopant_indices = [idx for idx, site in enumerate(structure) if getattr(site.specie, "symbol", str(site.specie)) == dopant]
    host_indices = [idx for idx, site in enumerate(structure) if getattr(site.specie, "symbol", str(site.specie)) in host_elements]
    if not dopant_indices or not host_indices:
        return ConstraintCheck(
            name="dopant_vacancy_adjacency",
            satisfied=False,
            value={"dopant_count": len(dopant_indices), "host_count": len(host_indices)},
            threshold="under-coordinated dopant neighbor shell",
            group="geometry",
            score=0.0,
        )

    neighbor_counts = []
    for dopant_idx in dopant_indices:
        count = sum(1 for host_idx in host_indices if structure.get_distance(dopant_idx, host_idx) <= 2.4)
        neighbor_counts.append(count)
    min_neighbors = min(neighbor_counts)
    satisfied = min_neighbors <= 3
    return ConstraintCheck(
        name="dopant_vacancy_adjacency",
        satisfied=satisfied,
        value={"neighbor_counts": neighbor_counts},
        threshold={"<=": 3, "cutoff": 2.4},
        message="Proxy for a split-vacancy/adjacent-vacancy center.",
        group="geometry",
        score=1.0 if satisfied else max(0.0, 1.0 - (min_neighbors - 3) / 2.0),
    )


def _adsorbate_height_target(adsorbate: Optional[str]) -> Tuple[float, Tuple[float, float]]:
    targets = {
        "H": (1.5, (1.25, 1.75)),
        "O": (1.2, (0.95, 1.45)),
        "CO": (1.9, (1.55, 2.25)),
        "OH": (1.5, (1.20, 1.90)),
        "NO": (1.9, (1.55, 2.25)),
        "H2O": (2.0, (1.50, 2.60)),
    }
    return targets.get(str(adsorbate), (2.0, (1.0, 3.0)))


def _adsorption_surface_facet_check(
    structure: Structure,
    metal: Optional[str],
    facet: str,
    code: str,
) -> ConstraintCheck:
    """Broad facet check for common adsorption slabs.

    For fcc (111)/(100)-style benchmark surfaces, the in-plane lattice angle is
    a useful representation-invariant proxy. We keep generated-code evidence as
    a fallback because arbitrary CIFs do not always preserve enough metadata to
    recover a Miller index unambiguously.
    """
    numbers = _facet_numbers(facet)
    gamma = float(structure.lattice.gamma)
    geometry_score: Optional[float] = None
    geometry_label = "unsupported"

    if numbers in ([1, 1, 1], [-1, -1, -1]):
        deviation = min(abs(gamma - 60.0), abs(gamma - 120.0))
        geometry_score = max(0.0, 1.0 - deviation / 30.0)
        geometry_label = "triangular_in_plane_lattice"
    elif numbers in ([1, 0, 0], [0, 1, 0], [0, 0, 1]):
        deviation = abs(gamma - 90.0)
        geometry_score = max(0.0, 1.0 - deviation / 30.0)
        geometry_label = "square_in_plane_lattice"

    code_check = _facet_code_check(
        "surface_facet_provenance",
        code,
        facet,
        group="surface",
        hard=False,
    )
    code_score = code_check.normalized_score()
    combined_score = max(geometry_score if geometry_score is not None else 0.0, code_score)
    satisfied = combined_score >= 0.65

    return ConstraintCheck(
        name="surface_facet",
        satisfied=satisfied,
        value={
            "facet": facet,
            "metal": metal,
            "gamma": gamma,
            "geometry": geometry_label,
            "geometry_score": geometry_score,
            "code_score": code_score,
        },
        threshold={"score": ">=0.65", "facet": facet},
        message="Facet check uses broad in-plane geometry with code provenance as fallback.",
        group="surface",
        score=combined_score,
    )


def _facet_code_check(
    name: str,
    code: str,
    facet: str,
    group: str,
    hard: bool = True,
) -> ConstraintCheck:
    variants = _facet_variants(facet)
    compact_code = re.sub(r"\s+", "", code.lower())
    lower = code.lower()
    facet_ok = any(variant in compact_code or variant in lower for variant in variants)
    construction_ok = any(term in lower for term in ["slabgenerator", "miller_index", "gb", "plane", "facet"])
    satisfied = bool(code) and facet_ok and construction_ok
    return ConstraintCheck(
        name=name,
        satisfied=satisfied,
        value={"facet": facet, "code_has_facet": facet_ok, "code_has_construction_term": construction_ok},
        threshold={"facet": facet, "code_derived": True},
        message="Code-derived proxy for the requested Miller facet.",
        group=group,
        hard=hard,
        score=1.0 if satisfied else (0.5 if facet_ok else 0.0),
    )


def _facet_variants(facet: str) -> List[str]:
    text = str(facet).strip().lower()
    numbers = _facet_numbers(text)
    variants = {text, text.replace(" ", ""), text.strip("()")}
    if numbers:
        joined = ",".join(str(value) for value in numbers)
        spaced = ", ".join(str(value) for value in numbers)
        variants.update({
            f"({joined})",
            f"({spaced})",
            f"[{joined}]",
            f"[{spaced}]",
            "".join(str(value) for value in numbers),
        })
    return sorted(variants)


def _facet_numbers(facet: str) -> List[int]:
    text = str(facet).strip().lower()
    numbers = [int(value) for value in re.findall(r"-?\d", text)]
    if len(numbers) == 4 and numbers[:3] == [0, 0, 0]:
        return [0, 0, numbers[3]]
    elif len(numbers) == 4:
        return [numbers[0], numbers[1], numbers[3]]
    return numbers


def _code_vector_check(
    name: str,
    code: str,
    vector: Iterable[Any],
    group: str,
    keywords: Optional[List[str]] = None,
    hard: bool = True,
) -> ConstraintCheck:
    values = [int(value) for value in vector]
    compact_code = re.sub(r"\s+", "", code.lower())
    lower = code.lower()
    joined = ",".join(str(value) for value in values)
    spaced = ", ".join(str(value) for value in values)
    variants = [f"[{joined}]", f"({joined})", f"[{spaced}]", f"({spaced})", joined, spaced]
    vector_ok = any(variant.replace(" ", "") in compact_code or variant in lower for variant in variants)
    keyword_ok = True if not keywords else any(keyword.lower() in lower for keyword in keywords)
    satisfied = bool(code) and vector_ok and keyword_ok
    return ConstraintCheck(
        name=name,
        satisfied=satisfied,
        value={"vector": values, "code_has_vector": vector_ok, "code_has_keyword": keyword_ok},
        threshold={"vector": values, "code_derived": True},
        message="Code-derived proxy for a vector-valued prompt constraint.",
        group=group,
        hard=hard,
        score=1.0 if satisfied else (0.5 if vector_ok else 0.0),
    )


def _code_keyword_check(
    name: str,
    code: str,
    required: Iterable[str],
    group: str,
    message: str,
    hard: bool = True,
) -> ConstraintCheck:
    lower = code.lower()
    required_terms = [str(term).lower() for term in required]
    present = [term for term in required_terms if term in lower]
    satisfied = bool(code) and len(present) == len(required_terms)
    return ConstraintCheck(
        name=name,
        satisfied=satisfied,
        value=present,
        threshold=required_terms,
        message=message,
        group=group,
        hard=hard,
        score=len(present) / len(required_terms) if required_terms else 1.0,
    )


def _code_numeric_threshold_check(
    name: str,
    code: str,
    value: float,
    group: str,
    keywords: Optional[List[str]] = None,
    tolerance: float = 1e-6,
    require_comparison: bool = False,
    hard: bool = True,
) -> ConstraintCheck:
    lower = code.lower()
    numbers = [float(match) for match in re.findall(r"(?<![A-Za-z])-?\d+(?:\.\d+)?", lower)]
    numeric_ok = any(abs(number - value) <= tolerance for number in numbers)
    keyword_ok = True if not keywords else any(keyword.lower() in lower for keyword in keywords)
    comparison_ok = True
    if require_comparison:
        comparison_ok = bool(re.search(r"(<=|<)\s*[-+]?\d*\.?\d+", lower) or "max_" in lower)
    satisfied = bool(code) and numeric_ok and keyword_ok and comparison_ok
    return ConstraintCheck(
        name=name,
        satisfied=satisfied,
        value={"target": value, "code_has_value": numeric_ok, "code_has_keyword": keyword_ok},
        threshold={"value": value, "tolerance": tolerance, "code_derived": True},
        message="Code-derived proxy for a numeric prompt constraint.",
        group=group,
        hard=hard,
        score=1.0 if satisfied else (0.5 if numeric_ok else 0.0),
    )


def _elements_present_check(
    name: str,
    present_elements: Iterable[str],
    expected_elements: Iterable[str],
    group: str = "identity",
    hard: bool = True,
) -> ConstraintCheck:
    present = {str(element) for element in present_elements}
    expected = {str(element) for element in expected_elements if element}
    missing = sorted(expected - present)
    score = 1.0 - (len(missing) / len(expected)) if expected else 1.0
    return ConstraintCheck(
        name=name,
        satisfied=not missing,
        value=sorted(present),
        threshold=sorted(expected),
        message="" if not missing else f"Missing elements: {missing}",
        group=group,
        hard=hard,
        score=score,
    )


def _elements_from_formula_label(label: str) -> List[str]:
    formula = str(label).split("_", 1)[0]
    formula = re.sub(r"\([^)]*\)", "", formula)
    formula = formula.replace("-", "")
    try:
        return [str(element) for element in Composition(formula).elements]
    except Exception:
        return re.findall(r"[A-Z][a-z]?", formula)


