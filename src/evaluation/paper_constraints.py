"""Pure prompt-driven PDF Table 5 constraints; no generation or MLIP calls.

The required list depends on the task, never on which source checks succeeded.
The same rows are consumed by the semantic critic and benchmark CSR/TSR.
Unknown evidence remains in the denominator. Geometry proxies are explicitly
labelled; construction-code keywords alone never certify a physical result.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import math
import re
import numpy as np

from . import _paper_proxies as proxies
from .validity import structural_validity_checks
from .paper_serialization import optional_bool, json_mapping


ALIASES = {"graphene": "C", "black_phosphorus": "P", "hBN": "BN"}
CODE_PROXIES = {"gb_plane", "rotation_axis", "rotation_angle"}
GEOMETRY_PROXIES = {
    "surface_facet", "surface_slab", "adsorption_site", "adsorbate_above_surface",
    "bicrystal_structure", "csl_relation", "gb_structure_type", "charge_neutral",
    "layered_structure", "dumbbell_geometry", "linear_dopant_arrangement",
    "dopant_vacancy_adjacency", "interface_composition", "min_thickness",
}


def _formula(label):
    from pymatgen.core import Composition, Element
    label = str(label or "")
    if not label:
        return None
    label = ALIASES.get(label, label.split("_")[0])
    try:
        composition = Composition(label).get_el_amt_dict()
        return composition if all(Element.is_valid_symbol(e) for e in composition) else None
    except (ValueError, TypeError, KeyError):
        return None


def _composition(structure):
    if structure is None:
        return None
    counts = Counter()
    for site in structure:
        for specie, occupancy in site.species.items():
            counts[specie.symbol] += float(occupancy)
    return dict(counts)


def _clipped(value):
    try:
        value = float(value)
        return min(1.0, max(0.0, value)) if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


class _Checks:
    def __init__(self, structure, source):
        self.structure = structure
        self.source = {}
        self.conflicts = set()
        self.rows = {}
        for check in source or []:
            check = deepcopy(check)
            check["satisfied"] = optional_bool(check.get("satisfied"))
            check = json_mapping(check)
            name = check.get("name")
            if name in self.source and self.source[name] != check:
                self.conflicts.add(name)
            self.source[name] = check

    def add(self, name, requirement=None, *, promote=False, unknown=None):
        source = self.source.get(name)
        kind = ("measured_cubic_orientation" if name in CODE_PROXIES and source and isinstance(source.get("value"), dict) and "left" in source["value"] else
                "inherited_code_proxy" if name in CODE_PROXIES else
                "inherited_geometry_proxy" if name in GEOMETRY_PROXIES else
                "inherited_protocol_check")
        row = dict(name=name, hard=True, satisfied=None, score=None,
                   requirement=requirement,
                   group=(source or {}).get("group", "task"),
                   value=(source or {}).get("value"),
                   threshold=(source or {}).get("threshold", requirement),
                   evidence={"kind": kind, "source_check": deepcopy(source)})
        reason = unknown
        if self.structure is None:
            reason = "No structure is available for the required check."
        elif name in self.conflicts:
            reason = "Conflicting source checks with the same canonical name."
        elif not source:
            reason = reason or "Required check is absent from the saved source evidence."
        elif not source.get("hard", True) and not promote:
            reason = reason or "Only a soft diagnostic is available for this requirement."
        elif isinstance(source.get("value"), dict) and source["value"].get("skipped"):
            reason = reason or "The source evaluator skipped this required measurement."
        elif "skipped" in str(source.get("message", "")).lower():
            reason = reason or "The source evaluator skipped this required measurement."
        if not reason and source:
            satisfied = optional_bool(source.get("satisfied"))
            if satisfied is True:
                row.update(satisfied=True, score=1.0)
            elif satisfied is False:
                raw = source.get("score")
                score = 0.0 if raw is None else _clipped(raw)
                if score is None:
                    row.update(satisfied=False, score=0.0)
                    reason = "Known failed check; invalid saved partial score contributes zero."
                    row["evidence"]["source_check"]["score"] = None
                    row["evidence"]["invalid_source_score"] = repr(raw)
                else:
                    row.update(satisfied=False, score=score)
            else:
                reason = "The source check has no resolved outcome."
        row["reason"] = reason or "Inherited check; passing scores normalized to one."
        row["evidence_available"] = row["satisfied"] is not None
        self.rows[name] = row
        return row

    def measured(self, name, satisfied, value, requirement, *, score=None, reason=""):
        satisfied = optional_bool(satisfied)
        row = self.add(name, requirement)
        row["evidence"] = {"kind": "saved_structure_measurement", "value": value,
                           "source_check": deepcopy(self.source.get(name))}
        if self.structure is not None:
            row.update(satisfied=satisfied,
                       score=None if satisfied is None else 1.0 if satisfied else (0.0 if score is None else _clipped(score)),
                       reason=reason or "Measured from the saved structure.",
                       evidence_available=satisfied is not None, value=value, threshold=requirement)
        return row


def _presence(checks, name, counts, elements, requirement=None):
    if not elements:
        return checks.add(name, requirement, unknown="Material formula is not unambiguously parsed.")
    present = sum((counts or {}).get(e, 0) > 0 for e in elements)
    return checks.measured(name, present == len(elements), counts,
                           requirement or {"elements": sorted(elements)},
                           score=present / len(elements),
                           reason="Element-presence evidence; inherited fraction-present partial score.")


def _defect(checks, task, inp, expected):
    counts = _composition(checks.structure)
    host = _formula(inp.get("host_material"))
    if inp.get("host_material"):
        _presence(checks, "host_elements", counts, host, inp["host_material"])
    original = expected.get("original_elements") or []
    if original and set(original) != set(host or {}):
        _presence(checks, "original_elements", counts, original)
    introduced = set(expected.get("composition_contains") or [])
    introduced.update(k[9:] for k, v in expected.items()
                      if v is True and re.fullmatch(r"contains_[A-Z][a-z]?", k))
    if introduced:
        _presence(checks, "composition_contains", counts, introduced)
    minimum = expected.get("min_atoms", inp.get("min_atoms", (inp.get("constraints") or {}).get("min_atoms")))
    if minimum is not None:
        checks.add("min_atoms", minimum)
    dopant = inp.get("dopant") or inp.get("substitute_element")
    exact_counts = {k[:-6]: v for k, v in expected.items()
                    if re.fullmatch(r"[A-Z][a-z]?_count", k) and isinstance(v, (int, float))}
    if dopant:
        requested = inp.get("dopant_count", expected.get(f"{dopant}_count", expected.get("defect_count")))
        if requested is not None:
            exact_counts[dopant] = requested
    for element, target in sorted(exact_counts.items()):
        actual = (counts or {}).get(element, 0.0)
        passed = abs(actual - float(target)) <= 1e-6
        score = max(0.0, 1.0 - abs(actual - float(target)) / max(float(target), 1.0))
        checks.measured(f"{element}_count", passed, actual, target, score=score,
                        reason="Explicit observable count; inherited 1e-6 equality tolerance and count decay.")
    if dopant and inp.get("dopant_site"):
        checks.add("substitution_species", {"dopant": dopant, "site": inp["dopant_site"]},
                   unknown="Element presence cannot establish substitution at the requested host site without a reference assignment.")
    for key, value in expected.items():
        if (re.fullmatch(r"final_has_less_[A-Z][a-z]?", key) or
                re.fullmatch(r"[A-Z][a-z]?_count_unchanged", key) or
                re.fullmatch(r"net_[A-Z][a-z]?_change", key)):
            checks.add(key, value)
    vacancy = inp.get("vacancy_element") or inp.get("defect_element")
    if vacancy and expected.get("defect_count") is not None:
        checks.add("vacancy_count", expected["defect_count"])
    # An explicit missing-atom count cannot disappear merely because the saved
    # CIF lacks a pristine reference. Do not promote the old code-operation
    # diagnostic as proof that the requested vacancies/interstitials exist.
    element_vacancies = {k: v for k, v in expected.items()
                         if re.fullmatch(r"[A-Z][a-z]?_vacancy_count", k)}
    total_vacancies = expected.get("vacancy_count", inp.get("vacancy_count"))
    if total_vacancies is not None and not (
            element_vacancies and sum(element_vacancies.values()) == total_vacancies):
        checks.add("vacancy_count", total_vacancies)
    for name, value in expected.items():
        if (re.fullmatch(r"[A-Z][a-z]?_(vacancy|interstitial)_count", name)
                or re.fullmatch(r"extra_[A-Z][a-z]?", name)
                or name in {"interstitial_count", "antisite_count"}):
            checks.add(name, value)
    if expected.get("extra_atoms") is not None and "interstitial_count" not in expected:
        checks.add("extra_atoms", expected["extra_atoms"])
    if inp.get("interstitial_site"):
        checks.add("interstitial_site", inp["interstitial_site"],
                   unknown="No validated site assignment relative to the pristine host; source code keywords are not proof.")
    if expected.get("charge_neutral") or task.get("validation", {}).get("check_charge_balance"):
        checks.add("charge_neutral", True)
    if expected.get("stoichiometry_preserved"):
        checks.add("stoichiometry_preserved", inp.get("host_material"))
    if expected.get("is_layered") or expected.get("layer_count") is not None:
        checks.add("layered_structure", expected.get("layer_count", 2))
    dtype = str(inp.get("defect_type", "")).lower()
    if expected.get("dumbbell_configuration") or "dumbbell" in dtype:
        checks.add("dumbbell_geometry", True)
    if expected.get("linear_NVN_arrangement") or dtype == "nvn_center":
        checks.add("linear_dopant_arrangement", True)
    if expected.get("Ge_adjacent_to_vacancy") or "split_vacancy" in dtype:
        checks.add("dopant_vacancy_adjacency", True)
    # Explicit spatial requirements must not silently disappear when the
    # generated structure has no pristine-site correspondence.
    geometry_keys = {"min_vacancy_distance", "vacancies_are_neighbors", "all_vacancies_neighbors",
                     "triangular_arrangement", "tetrahedral_arrangement", "vacancy_in_top_layer",
                     "vacancies_along_111", "dumbbell_axis", "chain_direction", "displaced_chain"}
    for name in sorted(geometry_keys & set(expected)):
        checks.add(name, expected[name], unknown="No independently validated pristine-site/spatial assignment for this requested geometry.")


def _adsorption(checks, task, inp):
    data = {**task, **inp}
    metal = data.get("metal") or data.get("catalyst_metal")
    plural = data.get("adsorbates") or []
    molecules = list(plural) or ([data["adsorbate"]] if data.get("adsorbate") else [])
    counts = _composition(checks.structure)
    _presence(checks, "catalyst_metal", counts, [metal] if metal else [], metal)
    target = Counter()
    formula_error = not molecules
    for molecule in molecules:
        composition = _formula(molecule)
        formula_error |= not bool(composition)
        if composition:
            target.update(composition)
    ambiguity = ("Missing/unparseable adsorbate specification." if formula_error else
                 "Adsorbate and substrate species overlap; no atom assignment is available."
                 if metal in target else None)
    actual = {e: n for e, n in (counts or {}).items() if e != metal}
    if ambiguity:
        checks.add("adsorbate_elements", list(molecules), unknown=ambiguity)
        checks.add("adsorbate_count", list(molecules), unknown=ambiguity)
    else:
        _presence(checks, "adsorbate_elements", actual, target)
        coverage = data.get("coverage")
        strict = bool(plural) and coverage is None
        bad_count = False
        if not plural and coverage is None:
            multiple = data.get("adsorbate_count", 1)
            if isinstance(multiple, (int, float)) and multiple >= 1 and float(multiple).is_integer():
                target = Counter({e: n * int(multiple) for e, n in target.items()})
                strict = True
            else:
                bad_count = True
        if bad_count:
            checks.add("adsorbate_count", dict(target), unknown="Invalid explicitly requested adsorbate cardinality.")
        elif strict:
            passed = set(actual) == set(target) and all(abs(actual[e] - target[e]) < 1e-6 for e in target)
            checks.measured("adsorbate_count", passed, actual, {"exact_composition": dict(target)},
                            reason="Combined adsorbate stoichiometry, not a total slab atom-count constraint.")
        else:
            ratios = [actual.get(e, 0) / n for e, n in target.items()]
            ratio = ratios[0]
            passed = (set(actual) == set(target) and ratio >= 1 and abs(ratio - round(ratio)) < 1e-6
                      and all(abs(x - ratio) < 1e-6 for x in ratios))
            checks.measured("adsorbate_count", passed, actual, {"integer_multiple_of": dict(target)},
                            reason="Formula stoichiometry only; no unstated single-molecule cardinality.")
    layers = data.get("min_slab_layers", data.get("min_layers", 4))
    slab = checks.add("surface_slab", {"min_layers": layers, "default_source": "fixed implementation convention: 4"})
    if layers != 4 and slab["satisfied"] is not None:
        metrics = (checks.source.get("surface_slab") or {}).get("value") or {}
        observed = metrics.get("num_layers") if isinstance(metrics, dict) else None
        if observed is None:
            checks.add("surface_slab", layers, unknown="Requested layer count differs from source default; measured count missing.")
        else:
            checks.measured("surface_slab", observed >= layers, observed, layers,
                            score=min(observed / max(float(layers), 1.0), 1.0))
    facet = data.get("facet") or data.get("miller_index")
    if facet:
        checks.add("surface_facet", facet)
    if plural:
        preferred = data.get("preferred_sites") or []
        for index, formula in enumerate(molecules):
            checks.add(f"adsorbate_{index}_height", formula)
            if index < len(preferred):
                checks.add(f"adsorbate_{index}_site", preferred[index])
    else:
        checks.add("adsorbate_above_surface", data.get("adsorbate"))
        site = data.get("site") or data.get("adsorption_site")
        stepped = "stepped" in str(data.get("task_type", "")) or str(site).lower() == "step"
        if site or stepped:
            checks.add("adsorption_site", site or "step edge versus terrace",
                       unknown="Stepped-surface site classification is unsupported." if stepped else None)
    if data.get("coverage") is not None:
        checks.add("coverage", data["coverage"])
    description = str(data.get("description", "")).lower()
    if re.search(r"\b(orientation|binding geometry|binding mode)\b", description):
        checks.add("adsorbate_binding_geometry", "requested molecular binding geometry/orientation",
                   unknown="Element counts cannot certify the requested molecular connectivity and binding orientation.")
    elif re.search(r"\b(intact molecule|molecular identity|connectivity|isomer)\b", description):
        checks.add("adsorbate_molecular_identity", "requested molecular identity/connectivity",
                   unknown="Formula stoichiometry does not establish molecular connectivity.")


def _grain_boundary(checks, inp, expected):
    requested = {
        "gb_material": inp.get("material"), "gb_structure_type": expected.get("structure_type"),
        "min_thickness": inp.get("constraints", {}).get("min_thickness"),
        "gb_plane": inp.get("gb_plane"), "rotation_axis": inp.get("rotation_axis"),
        "rotation_angle": expected.get("rotation_angle"),
    }
    for name, value in requested.items():
        if value is not None and value != "":
            checks.add(name, value)
    checks.add("bicrystal_structure", True)
    if inp.get("sigma") is not None:
        checks.add("csl_relation", {k: inp.get(k) for k in ("sigma", "gb_plane", "gb_type")})
    if inp.get("gb_type"):
        checks.add("boundary_type", inp["gb_type"])


def _interface(checks, task, inp, expected):
    film_label, substrate_label = inp.get("film_material"), inp.get("substrate_material")
    film, substrate = _formula(film_label), _formula(substrate_label)
    counts = _composition(checks.structure)
    if film_label:
        _presence(checks, "film_material", counts, film, film_label)
    if substrate_label:
        _presence(checks, "substrate_material", counts, substrate, substrate_label)
    itype = expected.get("interface_type") or task.get("category")
    if itype:
        checks.add("interface_composition", itype)
    limits = inp.get("constraints", {})
    if limits.get("min_vacuum") is not None:
        checks.add("min_vacuum", limits["min_vacuum"])
    overlap = bool(set(film or {}) & set(substrate or {}))
    ambiguity = ("Unparsed film/substrate formula; atom assignment is unavailable." if not film or not substrate else
                 "Film and substrate share elements; element-only atom assignment is ambiguous." if overlap else None)
    if limits.get("min_slab_layers") is not None:
        layer = checks.add("min_slab_layers", limits["min_slab_layers"], promote=True, unknown=ambiguity)
        value = (checks.source.get("min_slab_layers") or {}).get("value")
        if not ambiguity and isinstance(value, dict) and {"film", "substrate"} <= value.keys():
            minimum = min(value["film"], value["substrate"])
            target = float(limits["min_slab_layers"])
            if layer["satisfied"] is not None:
                checks.measured("min_slab_layers", minimum >= target, value, target,
                                score=min(minimum / max(target, 1.0), 1.0),
                                reason="Both materials count, including a missing side with zero layers; inherited layer measurement.")
    distance_range = expected.get("interface_distance_range")
    if distance_range is not None:
        checks.add("interface_distance", distance_range, promote=True, unknown=ambiguity)
    pair = re.findall(r"[A-Z][a-z]?", str(expected.get("bonding_type", "")))
    if len(pair) >= 2:
        a, b = pair[:2]
        cross = ((a in (film or {}) and b in (substrate or {})) or
                 (b in (film or {}) and a in (substrate or {})))
        reason = ambiguity or (None if cross else "Requested pair is not assignable across the film/substrate interface.")
        row = checks.add("bonding_type", {"pair": [a, b], "range": distance_range}, promote=True, unknown=reason)
        row["evidence"]["cross_interface_assignment"] = bool(cross and not ambiguity)
        if cross and not ambiguity:
            row["evidence"]["assignment_basis"] = "Disjoint material element sets; pair elements belong to opposite sides."


def select_constraints(structure, task, family, source_checks):
    """Return JSON-ready hard checks with ``satisfied``/``score`` or unknowns.

    Caller applies structural validity and aggregates; this adapter never runs
    generation, loads a model, or chooses thresholds from observed pass rates.
    """
    checks = _Checks(structure, source_checks)
    inp = task.get("input", task) or {}
    expected = task.get("expected_properties", {}) or {}
    if family == "defect":
        _defect(checks, task, inp, expected)
    elif family == "adsorption":
        _adsorption(checks, task, inp)
    elif family == "grain_boundary":
        _grain_boundary(checks, inp, expected)
    elif family == "interface":
        _interface(checks, task, inp, expected)
    else:
        checks.add("task_specification", family, unknown=f"Unsupported task family: {family}")
    if not checks.rows:
        checks.add("task_specification", family, unknown="No supported task requirements were parsed.")
    return [json_mapping(row) for row in checks.rows.values()]


def infer_task_type(task):
    """Normalize supported family names; retain subtype fields in the task."""
    inp = task.get("input", task) or {}
    kind = str(task.get("task_type", inp.get("task_type", ""))).lower()
    if "adsorp" in kind or "metal" in inp and ("adsorbate" in inp or "adsorbates" in inp):
        return "adsorption"
    if "grain" in kind or "gb_type" in inp or "sigma" in inp:
        return "grain_boundary"
    if "interface" in kind or "film_material" in inp:
        return "interface"
    if "defect" in kind or "defect_type" in inp or "host_material" in inp:
        return "defect"
    return kind or "unknown"


def _measured_row(name, passed, value, requirement, score=None, reason=""):
    passed = optional_bool(passed)
    return dict(name=name, satisfied=passed, value=value, threshold=requirement,
                score=1.0 if passed is True else score, hard=True,
                message=reason, group="geometry")


def _surface_sources(structure, task):
    from .paper_surface_geometry import normal_coordinates, layer_groups, adsorption_site, range_score, surface_facet
    data = {**task, **(task.get("input") or {})}
    metal = data.get("metal") or data.get("catalyst_metal")
    substrate = [i for i, site in enumerate(structure) if site.species.get_el_amt_dict() == {metal: 1.0}]
    ads = [i for i in range(len(structure)) if i not in substrate]
    if not substrate:
        return []
    # Wrap fractional coordinates before local image geometry.
    structure = structure.copy()
    structure.translate_sites(range(len(structure)), [0, 0, 0], frac_coords=True, to_unit_cell=True)
    heights, _, _, _ = normal_coordinates(structure, substrate)
    layers = layer_groups(heights, substrate)
    requested_layers = int(data.get("min_slab_layers", data.get("min_layers", 4)))
    rows = [_measured_row("surface_slab", len(layers) >= requested_layers,
                         {"num_layers": len(layers)}, {"minimum": requested_layers, "layer_tolerance": 0.5},
                         min(1., len(layers)/max(requested_layers, 1)))]
    facet = data.get("facet") or data.get("miller_index")
    if facet:
        passed,score,evidence=surface_facet(structure,substrate,metal,facet)
        rows.append(_measured_row("surface_facet",passed,evidence,facet,score,evidence.get("reason","")))
    if data.get("adsorbates"):
        rows.extend(_coadsorption_sources(structure, data, substrate, ads, heights))
    else:
        adsorbate = data.get("adsorbate")
        target, bounds = proxies._adsorbate_height_target(adsorbate)
        bounds = data.get("adsorption_height_range", data.get("height_range", bounds))
        if ads:
            height = float(min(heights[i] for i in ads) - max(heights[i] for i in substrate))
            passed, score = range_score(height, float(bounds[0]), float(bounds[1]))
            rows.append(_measured_row("adsorbate_above_surface", passed, {"height": height},
                                      {"range": list(bounds), "target": target}, score,
                                      "Periodic normal height of the lowest adsorbate atom above the substrate"))
        else:
            rows.append(_measured_row("adsorbate_above_surface", False, None, bounds, 0.))
        site = data.get("site") or data.get("adsorption_site")
        if site:
            passed, score, evidence = adsorption_site(structure, substrate, ads, site)
            rows.append(_measured_row("adsorption_site", passed, evidence, site, score, evidence.get("reason", "")))
    if data.get("coverage") is not None:
        formula = _formula(data.get("adsorbate"))
        counts = _composition(structure)
        if formula and metal not in formula and layers:
            ratios = [counts.get(e, 0) / n for e, n in formula.items()]
            if max(ratios)-min(ratios) < 1e-6 and abs(ratios[0]-round(ratios[0])) < 1e-6:
                actual = ratios[0] / len(layers[-1])
                target = float(data["coverage"])
                passed = abs(actual-target) <= 1e-6
                rows.append(_measured_row("coverage", passed, actual,
                    {"target": target, "tolerance": 1e-6, "convention": "molecules / resolved top-layer substrate atoms"},
                    max(0., 1.-abs(actual-target)/max(target,1e-6))))
    return rows


def _coadsorption_sources(structure, data, substrate, ads, heights):
    """Assign disconnected adsorbates by formula and fixed covalent radii.

    Equivalent repeated formulas are assigned as an unordered set; a minimum
    cost bijection respects all requested height/site conditions jointly.
    A merged/unparseable molecule graph supplies no fabricated assignment.
    """
    from scipy.optimize import linear_sum_assignment
    from .paper_surface_geometry import adsorption_site, range_score
    radii={"H":.31,"C":.76,"N":.71,"O":.66,"P":1.07,"S":1.05}
    formulas=[_formula(f) for f in data["adsorbates"]]
    if any(not f for f in formulas):return []
    parent={i:i for i in ads}
    def root(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]];i=parent[i]
        return i
    for pos,i in enumerate(ads):
        a=structure[i].specie.symbol
        if a not in radii:return []
        for j in ads[pos+1:]:
            b=structure[j].specie.symbol
            if b not in radii:return []
            if structure.get_distance(i,j)<=1.25*(radii[a]+radii[b]):
                parent[root(j)]=root(i)
    components={}
    for i in ads:components.setdefault(root(i),[]).append(i)
    groups=list(components.values())
    if len(groups)!=len(formulas):return []
    species=[dict(Counter(structure[i].specie.symbol for i in group)) for group in groups]
    preferred=data.get("preferred_sites") or []
    cost=np.full((len(formulas),len(groups)),1e6);evidence={}
    for i,formula in enumerate(formulas):
        for j,group in enumerate(groups):
            if species[j]!=formula:continue
            _,bounds=proxies._adsorbate_height_target(data["adsorbates"][i])
            height=float(min(heights[k] for k in group)-max(heights[k] for k in substrate))
            hp,hs=range_score(height,*bounds)
            row=[_measured_row(f"adsorbate_{i}_height",hp,{"height":height,"atom_indices":group,"assignment":"covalent-radius connected component"},
                               {"range":bounds,"bond_radius_multiplier":1.25},hs)]
            penalties=int(not hp)*10+1-hs
            if i<len(preferred):
                sp,ss,detail=adsorption_site(structure,substrate,group,preferred[i])
                row.append(_measured_row(f"adsorbate_{i}_site",sp,{**detail,"atom_indices":group},preferred[i],ss))
                penalties+=(10 if sp is not True else 0)+(1-(ss or 0))
            cost[i,j]=penalties;evidence[i,j]=row
    ri,ci=linear_sum_assignment(cost)
    if any(cost[i,j]>=1e6 for i,j in zip(ri,ci)):return []
    return [row for i,j in zip(ri,ci) for row in evidence[i,j]]


def _interface_sources(structure, task):
    from .paper_surface_geometry import normal_coordinates, layer_groups, min_cross_distance, range_score
    inp = task.get("input", task)
    expected = task.get("expected_properties") or {}
    limits = inp.get("constraints") or {}
    film, substrate = _formula(inp.get("film_material")), _formula(inp.get("substrate_material"))
    rows = []
    all_indices = list(range(len(structure)))
    heights, _, _, vacuum = normal_coordinates(structure, all_indices)
    if limits.get("min_vacuum") is not None:
        target = float(limits["min_vacuum"])
        rows.append(_measured_row("min_vacuum", vacuum + 0.1 >= target, vacuum,
                                  {"minimum": target, "tolerance": 0.1}, min(1., (vacuum+0.1)/max(target,1e-6))))
    if not film or not substrate or set(film) & set(substrate):
        return rows
    left = [i for i,s in enumerate(structure) if set(s.species.get_el_amt_dict()) <= set(film)]
    right = [i for i,s in enumerate(structure) if set(s.species.get_el_amt_dict()) <= set(substrate)]
    counts = _composition(structure)
    intended = set(film) | set(substrate)
    present = set(counts)
    rows.append(_measured_row("interface_composition", present == intended, sorted(present), sorted(intended),
                              len(present & intended)/max(len(present | intended), 1)))
    if limits.get("min_slab_layers") is not None:
        value = {"film": len(layer_groups(heights, left)), "substrate": len(layer_groups(heights, right))}
        target = int(limits["min_slab_layers"])
        minimum = min(value.values())
        rows.append(_measured_row("min_slab_layers", minimum >= target, value, target, min(1.,minimum/max(target,1))))
    distance_range = expected.get("interface_distance_range")
    if distance_range is not None:
        distance = min_cross_distance(structure, left, right)
        passed, score = (False, 0.) if distance is None else range_score(distance, *map(float, distance_range), tolerance=0.75)
        rows.append(_measured_row("interface_distance", passed, distance,
                                  {"range": distance_range, "tolerance": 0.75}, score))
    pair = re.findall(r"[A-Z][a-z]?", str(expected.get("bonding_type", "")))
    if len(pair) == 2:
        a,b = pair
        if a in substrate and b in film:
            a,b=b,a
        if a in film and b in substrate:
            li=[i for i in left if a in structure[i].species.get_el_amt_dict()]
            ri=[i for i in right if b in structure[i].species.get_el_amt_dict()]
            distance=min_cross_distance(structure, li, ri)
            bounds = distance_range or [1.5, 3.5]
            tolerance=0.75 if distance_range else 0.
            passed,score=(False,0.) if distance is None else range_score(distance,*map(float,bounds),tolerance=tolerance)
            rows.append(_measured_row("bonding_type",passed,{"pair":pair,"min_distance":distance,"cross_interface":True},
                                      {"range":bounds,"tolerance":tolerance},score))
    return rows


def collect_task_constraints(structure, task, task_type=None, result_metadata=None):
    """Collect the fixed required set without importing critic or protocol.

    Outcomes are True/False/None. Unknowns are retained with evidence and are
    scored conservatively by both consumers; no source diagnostic is silently
    treated as a satisfied hard constraint.
    """
    family = task_type or infer_task_type(task)
    if family not in {"defect", "adsorption", "grain_boundary", "interface"}:
        family = infer_task_type(task)
    usable = structure
    validity = structural_validity_checks(structure, family)
    if not all(optional_bool(row["satisfied"]) is True for row in validity):
        usable = None
    source, source_error = [], None
    if usable is not None:
        try:
            if family == "defect":
                task=deepcopy(task)
                inp=task.get("input",task)
                minimum=(task.get("expected_properties") or {}).get("min_atoms",inp.get("min_atoms",(inp.get("constraints") or {}).get("min_atoms")))
                if minimum is not None:
                    task.setdefault("expected_properties",{})["min_atoms"]=minimum
                source = [row.to_dict() for row in proxies._defect_constraints(usable, task, result_metadata)]
            elif family == "adsorption":
                source = _surface_sources(usable, task)
            elif family == "interface":
                source = _interface_sources(usable, task)
            elif family == "grain_boundary":
                from .paper_grain_geometry import grain_boundary_constraints
                source = grain_boundary_constraints(usable, task)
        except Exception as exc:
            source_error = f"{type(exc).__name__}: {exc}"
    rows = select_constraints(usable, task, family, source)
    for row in rows:
        row["satisfied"] = optional_bool(row["satisfied"])
        row["evidence_available"] = row["satisfied"] is not None
        row["status"] = "unknown" if row["satisfied"] is None else "pass" if row["satisfied"] else "fail"
        if source_error:
            row["evidence"]["measurement_error"] = source_error
        if row["satisfied"] is True:
            row["score"] = 1.0
        elif row["satisfied"] is False:
            row["score"] = _clipped(row.get("score")) or 0.0
        # Fixed, dimensionless convention for semantic Eq. 29. Physical
        # thresholds remain in threshold/evidence, never fitted to outcomes.
        row["violation"] = None if row["satisfied"] is None else 1.0 - row["score"]
        row["tolerance"] = 0.0
        row["alpha"] = 1.0
    return [json_mapping(row) for row in rows]
