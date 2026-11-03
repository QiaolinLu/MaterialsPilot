"""Deterministic physical primitives for the 7901 paper implementation.

Parameters absent from the paper are frozen implementation choices, documented
in docs/implementation.md. No parameter is estimated from benchmark outcomes.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any

import numpy as np
from pymatgen.core import Composition, Element, Lattice, Structure

# A conservative severe-overlap predicate, distinct from post-hoc validity.
# Explicit per-family entries make the task-conditioned tolerance inspectable.
TASK_OVERLAP_GAMMA = {
    "defect": 0.55, "adsorption": 0.55,
    "grain_boundary": 0.55, "interface": 0.55, "generic": 0.55,
}
BVS_LAMBDA = 1.0
BVS_NEIGHBOR_RADIUS = 4.0
ENERGY_LAMBDA_PER_EV = 1.0
ENERGY_MARGIN_EV_PER_ATOM = 0.0
SEMANTIC_ALPHA = 1.0


def task_record(description: str, task_input: dict | None) -> dict:
    """Accept both full dataset records and the runner's flattened inputs."""
    raw = dict(task_input or {})
    if isinstance(raw.get("input"), dict):
        task = raw
    else:
        task = dict(raw)
        task["input"] = dict(raw)
    task.setdefault("description", description)
    task.setdefault("task_type", infer_task_type(task))
    return task


def infer_task_type(task: dict) -> str:
    fields = task.get("input", task)
    explicit = str(task.get("task_type", fields.get("task_type", ""))).lower()
    aliases = {"heterostructure": "interface", "surface_adsorption": "adsorption",
               "grain-boundary": "grain_boundary", "jarvis_defect": "defect"}
    explicit = aliases.get(explicit, explicit)
    if explicit in TASK_OVERLAP_GAMMA:
        return explicit
    text = str(task.get("description", "")).lower()
    if fields.get("film_material") or "interface" in text or "heterostructure" in text:
        return "interface"
    if fields.get("sigma") or "grain boundary" in text or "grain-boundary" in text:
        return "grain_boundary"
    if fields.get("adsorbate") or fields.get("adsorbates") or "adsorp" in text or "adsorb" in text:
        return "adsorption"
    if fields.get("defect_type") or fields.get("host_material") or "defect" in text:
        return "defect"
    return "generic"


def covalent_radius(specie: Any) -> float:
    # Pymatgen packages the Cordero et al. (2008) covalent-radius table.
    from pymatgen.analysis.molecule_structure_comparator import CovalentRadius
    symbol = getattr(specie, "symbol", str(specie))
    value = float(CovalentRadius.radius[symbol])
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"No positive covalent radius for {symbol}")
    return value


def composition_signature(structure_or_composition: Any) -> dict[str, float]:
    composition = (structure_or_composition.composition
                   if isinstance(structure_or_composition, Structure)
                   else Composition(structure_or_composition))
    return {str(k): float(v) for k, v in
            composition.element_composition.fractional_composition.get_el_amt_dict().items()}


def same_composition(a: Any, b: Any, tolerance: float = 1e-8) -> bool:
    left, right = composition_signature(a), composition_signature(b)
    return left.keys() == right.keys() and all(abs(left[k] - right[k]) <= tolerance for k in left)


def adsorbate_formulas(task: dict) -> list[str]:
    fields = task.get("input", task)
    values = fields.get("adsorbates") or []
    if not isinstance(values, list):
        values = [values]
    if not values and fields.get("adsorbate"):
        values = [fields["adsorbate"]]
    formulas = []
    for value in values:
        if isinstance(value, dict):
            value = value.get("formula", value.get("adsorbate", value.get("species", "")))
        if value:
            formulas.append(str(value))
    return formulas


def molecular_template(formula: str):
    """Retrieve an independent ASE molecular geometry; never infer bonds from a clash."""
    from ase import Atoms
    from ase.build import molecule
    composition = Composition(formula)
    if composition.num_atoms == 1 and len(composition.elements) == 1:
        return Atoms([composition.elements[0].symbol], positions=[[0, 0, 0]])
    return molecule(formula)


def bonded_intramolecular_pairs(structure: Structure, task: dict) -> tuple[set[tuple[int, int]], dict]:
    """Recognize isolated adsorbate graphs against independent molecular templates.

    Exclusion requires the requested molecular formula, unambiguous non-substrate
    species, graph isomorphism, and each bond within 75--125% of its template
    length. Arbitrary close pairs and compressed bonds are never exempted.
    """
    import networkx as nx
    fields = task.get("input", task)
    formulas = adsorbate_formulas(task)
    excluded: set[tuple[int, int]] = set()
    diagnostics: dict = {"method": "requested_molecule_template_graph", "recognized_molecules": [], "unavailable": []}
    if not formulas:
        return excluded, diagnostics
    host_label = fields.get("metal", fields.get("catalyst_metal", fields.get("surface_metal", "")))
    try:
        host_elements = set(Composition(host_label).get_el_amt_dict()) if host_label else set()
    except Exception:
        host_elements = set()
    if host_label and not host_elements:
        diagnostics["unavailable"].append("Cannot identify substrate species")
        return excluded, diagnostics
    for formula in dict.fromkeys(formulas):
        try:
            template = molecular_template(formula)
            symbols = template.get_chemical_symbols()
            if len(symbols) < 2:
                continue
            if set(symbols) & host_elements:
                diagnostics["unavailable"].append(f"{formula}: adsorbate/substrate species overlap")
                continue
            reference = nx.Graph()
            for i, symbol in enumerate(symbols):
                reference.add_node(i, species=symbol)
            for i in range(len(symbols)):
                for j in range(i + 1, len(symbols)):
                    distance = float(template.get_distance(i, j))
                    if distance <= 1.25 * (covalent_radius(symbols[i]) + covalent_radius(symbols[j])):
                        reference.add_edge(i, j, length=distance)
            indices = [i for i, site in enumerate(structure) if site.specie.symbol in set(symbols)]
            observed = nx.Graph()
            for i in indices:
                observed.add_node(i, species=structure[i].specie.symbol)
            for pos, i in enumerate(indices):
                for j in indices[pos + 1:]:
                    distance = float(structure.get_distance(i, j))
                    radii = covalent_radius(structure[i].specie) + covalent_radius(structure[j].specie)
                    if distance <= 1.25 * radii:
                        observed.add_edge(i, j, length=distance)
            for component in nx.connected_components(observed):
                graph = observed.subgraph(component)
                if Counter(graph.nodes[i]["species"] for i in graph) != Counter(symbols):
                    continue
                matcher = nx.algorithms.isomorphism.GraphMatcher(
                    graph, reference,
                    node_match=lambda a, b: a["species"] == b["species"],
                    edge_match=lambda a, b: 0.75 * b["length"] <= a["length"] <= 1.25 * b["length"],
                )
                if matcher.is_isomorphic():
                    pairs = {tuple(sorted((i, j))) for i, j in graph.edges}
                    excluded.update(pairs)
                    diagnostics["recognized_molecules"].append({"formula": formula, "indices": sorted(component)})
        except Exception as exc:
            diagnostics["unavailable"].append(f"{formula}: {type(exc).__name__}: {exc}")
    diagnostics["excluded_bond_count"] = len(excluded)
    return excluded, diagnostics


def geometry_diagnostics(structure: Structure, task: dict, gamma: float | None = None) -> dict:
    if not len(structure) or not np.isfinite(structure.cart_coords).all() or not np.isfinite(structure.lattice.matrix).all():
        raise ValueError("Structure must have nonempty atoms and finite coordinates/lattice")
    if structure.volume <= 0 or not structure.is_ordered:
        raise ValueError("A positive cell and ordered atomic species are required")
    family = infer_task_type(task)
    gamma = TASK_OVERLAP_GAMMA[family] if gamma is None else float(gamma)
    if not math.isfinite(gamma) or gamma <= 0:
        raise ValueError("Overlap gamma must be positive")
    radii = [covalent_radius(site.specie) for site in structure]
    excluded, molecular = bonded_intramolecular_pairs(structure, task)
    cutoff = 2 * max(radii) * max(1.25, gamma) + 1e-6
    centers, neighbors, offsets, distances = structure.get_neighbor_list(r=cutoff, exclude_self=True)
    offending = []
    seen = set()
    minimum = math.inf
    for i_raw, j_raw, offset, distance_raw in zip(centers, neighbors, offsets, distances):
        i, j = int(i_raw), int(j_raw)
        distance = float(distance_raw)
        # Keep coincident distinct sites and nonzero periodic self-images.
        if i == j and np.allclose(offset, 0):
            continue
        key = tuple(sorted((i, j)))
        minimum = min(minimum, distance)
        threshold = gamma * (radii[i] + radii[j])
        mic_distance = float(structure.get_distance(i, j)) if i != j else distance
        if key in excluded and abs(distance - mic_distance) <= 1e-7:
            continue
        if distance < threshold and key not in seen:
            seen.add(key)
            offending.append({"i": i, "j": j, "species": f"{structure[i].specie.symbol}-{structure[j].specie.symbol}",
                              "distance": distance, "threshold": threshold, "image": np.asarray(offset).tolist()})
    return {"indicator": 0.0 if offending else 1.0, "gamma": gamma, "task_type": family,
            "radius_source": "pymatgen Cordero covalent radii", "method": "periodic_neighbor_list_MIC",
            "min_distance": None if math.isinf(minimum) else minimum,
            "num_overlap_pairs": len(offending), "overlap_pairs": offending[:20],
            "molecular_bond_exclusions": molecular}


def build_reference_structure(structure: Structure, task: dict) -> tuple[Structure | None, dict]:
    """Return an independently constructed same-composition reference, or unavailable.

    Never use the candidate geometry itself as its own energetic reference.
    An explicit reference has precedence: invalid or mismatched supplied evidence
    is reported rather than silently replaced by a different construction.
    """
    from pymatgen.io.ase import AseAtomsAdaptor
    fields = task.get("input", task)
    reference_spec = fields.get("thermodynamic_reference", task.get("thermodynamic_reference", {})) or {}
    explicit = reference_spec.get("structure", fields.get("reference_structure", fields.get("scaffold_structure",
                    task.get("reference_structure", task.get("scaffold_structure")))))
    if explicit is not None:
        try:
            reference = explicit.copy() if isinstance(explicit, Structure) else Structure.from_dict(explicit)
            metadata = {"source": reference_spec.get("provenance", "task.reference_structure"), "method": "explicit_scaffold"}
            return _validated_reference(reference, structure, task, metadata)
        except Exception as exc:
            return None, {"status": "unavailable", "reason": f"Invalid explicit reference: {exc}"}
    family = infer_task_type(task)
    if family == "adsorption":
        required = ("surface_lattice", "surface_coords", "surface_species")
        if all(k in fields for k in required):
            try:
                reference = Structure(fields["surface_lattice"], fields["surface_species"], fields["surface_coords"], coords_are_cartesian=True)
                formulas = adsorbate_formulas(task)
                if not formulas:
                    raise ValueError("No requested adsorbate formula")
                normal = np.cross(reference.lattice.matrix[0], reference.lattice.matrix[1])
                normal /= np.linalg.norm(normal)
                height_projection = reference.cart_coords @ normal
                top_index = int(np.argmax(height_projection))
                anchor = np.asarray(reference[top_index].coords)
                host_radius = covalent_radius(reference[top_index].specie)
                for molecule_index, formula in enumerate(formulas):
                    molecule = molecular_template(formula)
                    coords = molecule.get_positions().copy()
                    lowest = int(np.argmin(coords @ normal))
                    separation = host_radius + covalent_radius(molecule.get_chemical_symbols()[lowest])
                    translation = anchor + separation * normal - coords[lowest]
                    translation += (molecule_index / max(1, len(formulas))) * reference.lattice.matrix[0]
                    for symbol, position in zip(molecule.get_chemical_symbols(), coords + translation):
                        reference.append(symbol, position, coords_are_cartesian=True)
                metadata = {"source": "task.surface_lattice/surface_coords/surface_species + ASE molecule geometry",
                            "method": "independent_adsorption_initial_template", "placement": "top atom; covalent-radius-sum height",
                            "formulas": formulas, "composition_policy": "exact elemental fractions"}
                return _validated_reference(reference, structure, task, metadata)
            except Exception as exc:
                return None, {"status": "unavailable", "reason": f"Adsorption scaffold unavailable: {exc}"}
    material = fields.get("host_material", fields.get("material"))
    if not material:
        return None, {"status": "unavailable", "reason": "No explicit same-composition scaffold or supported task template"}
    try:
        from ase.build import bulk
        from ase.data import atomic_numbers, reference_states
        composition = Composition(material)
        if not same_composition(structure, composition):
            return None, {"status": "unavailable", "reason": "Task material scaffold composition differs from candidate"}
        crystal = fields.get("structure_type", fields.get("crystal_type"))
        expected = task.get("expected_properties", fields.get("expected_properties", {}))
        crystal = crystal or expected.get("structure_type")
        parameters = {}
        lattice_constant = fields.get("lattice_constant")
        if isinstance(lattice_constant, (float, int)):
            parameters["a"] = float(lattice_constant)
        if len(composition.elements) == 1:
            symbol = composition.elements[0].symbol
            reference_data = reference_states[atomic_numbers[symbol]]
            if reference_data is None:
                raise ValueError("ASE has no elemental bulk reference")
            requested_crystal = str(crystal).lower() if crystal else reference_data["symmetry"]
            if requested_crystal not in {"fcc", "bcc", "hcp", "diamond", "sc", "bct", "orthorhombic"}:
                raise ValueError(f"Unsupported bulk prototype {requested_crystal}")
            if requested_crystal != reference_data["symmetry"] and "a" not in parameters:
                raise ValueError("Non-reference crystal prototype needs a task lattice constant")
            atoms = bulk(symbol, crystalstructure=requested_crystal, **parameters)
            provenance = "ASE elemental reference_states lattice data"
        else:
            if crystal not in {"rocksalt", "zincblende", "wurtzite", "cesiumchloride", "fluorite"} or "a" not in parameters:
                raise ValueError("Compound reference needs supported task crystal prototype and lattice constant")
            atoms = bulk(material, crystalstructure=crystal, **parameters)
            provenance = "Task material/crystal prototype/lattice constant; ASE bulk constructor"
        reference = AseAtomsAdaptor.get_structure(atoms)
        return _validated_reference(reference, structure, task,
                                    {"source": provenance, "method": "independent_bulk_template", "material": material,
                                     "prototype": crystal or reference_data["symmetry"], "parameters": parameters})
    except Exception as exc:
        # Connect the already-existing retrieval corpus only when the previous
        # independent task/ASE path cannot construct a reference. Never replace
        # a working reference, compare energies to choose one, or alter a cell.
        reference, metadata = _knowledge_bulk_reference(structure, task)
        metadata["previous_reference_unavailable"] = f"Bulk scaffold unavailable: {exc}"
        return reference, metadata


def _knowledge_bulk_reference(candidate: Structure, task: dict):
    """Reuse the RAG's fully specified KB prototype as an independent scaffold.

    Compound cells without basis coordinates remain unsupported exactly as in
    RAG; only a named material with a compatible explicitly requested phase is
    eligible. Adsorbates and assembled interfaces cannot use one bulk component.
    """
    family = infer_task_type(task)
    if family not in {"defect", "grain_boundary"}:
        return None, {"status": "unavailable", "reason": "KB bulk reference is restricted to bulk defect/GB tasks"}
    fields = task.get("input", task)
    material = fields.get("host_material", fields.get("material"))
    expected = task.get("expected_properties", fields.get("expected_properties", {})) or {}
    requested_phase = fields.get("structure_type", fields.get("crystal_type", expected.get("structure_type")))
    if not material or not requested_phase:
        return None, {"status": "unavailable", "reason": "KB reference needs an explicit task material and crystal phase"}
    root = Path(__file__).resolve().parents[2]
    kb_relative = Path("data/knowledge/L2_structures/bulk/crystal_structures.json")
    path = root / kb_relative
    try:
        raw = path.read_bytes()
        data = json.loads(raw)
        entry = data.get(material)
        if not isinstance(entry, dict):
            return None, {"status": "unavailable", "reason": "No exact named material entry in the frozen KB"}
        stored_phase = entry.get("structure_type", entry.get("structure", ""))
        normalize_phase = lambda name: re.sub(r"[\s_\-]+", "", str(name).lower())
        if normalize_phase(requested_phase) != normalize_phase(stored_phase):
            return None, {"status": "unavailable", "reason": "Task crystal phase conflicts with the KB prototype",
                          "requested_phase": requested_phase, "kb_phase": stored_phase}
        # This function creates only the same prototypes available to the RAG;
        # importing it does not load or invoke either learned retrieval encoder.
        from src.rag import paper_retrieval
        reference = paper_retrieval.make_prototype(material, entry)
        if reference is None:
            return None, {"status": "unavailable", "reason": "KB entry lacks a fully specified constructible prototype"}
        metadata = {
            "source": "Existing frozen RAG knowledge-base crystal prototype",
            "method": "retrieved_kb_bulk_template",
            "template_status": "unrelaxed construction template; no claimed DFT relaxation or energy label",
            "material": material, "prototype": stored_phase,
            "kb_path": kb_relative.as_posix(), "kb_key": material,
            "kb_file_sha256": hashlib.sha256(raw).hexdigest(),
            "kb_entry_sha256": hashlib.sha256(json.dumps(entry, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
            "kb_declared_source": data.get("_metadata", {}).get("source"),
            "kb_declared_citation": data.get("_metadata", {}).get("citation"),
            "upstream_limitation": "The existing KB does not supply an individual MP/ICSD identifier for every entry",
            "constructor": "src.rag.paper_retrieval.make_prototype",
            "constructor_sha256": hashlib.sha256(Path(paper_retrieval.__file__).read_bytes()).hexdigest(),
            "selection_policy": "exact task material and phase; previous reference unavailable; no energy-based selection",
        }
        return _validated_reference(reference, candidate, task, metadata)
    except Exception as exc:
        return None, {"status": "unavailable", "reason": f"Frozen KB reference unavailable: {type(exc).__name__}: {exc}"}


def _validated_reference(reference: Structure, candidate: Structure, task: dict, metadata: dict):
    if not same_composition(reference, candidate):
        return None, {**metadata, "status": "unavailable", "reason": "Reference composition does not match candidate",
                      "reference_composition": composition_signature(reference), "candidate_composition": composition_signature(candidate)}
    diagnostics = geometry_diagnostics(reference, task)
    if diagnostics["indicator"] != 1:
        return None, {**metadata, "status": "unavailable", "reason": "Reference has severe geometric overlaps"}
    serialized = json.dumps(reference.as_dict(), sort_keys=True, default=str)
    return reference, {**metadata, "status": "available", "atom_count": len(reference),
                       "composition": composition_signature(reference),
                       "structure_sha256": hashlib.sha256(serialized.encode()).hexdigest()}


def semantic_components(checks: list[dict]) -> tuple[float, list[dict]]:
    """Eq.37--38: exponential satisfaction, then an equal-weight geometric mean.

    When a checker has no physical residual it exposes a bounded diagnostic
    score; its normalized deficit 1-score is the dimensionless violation. This
    is a frozen proxy, not an assertion that PDF specifies every tolerance.
    """
    components = []
    for check in checks:
        if not check.get("hard", True):
            continue
        known = check.get("satisfied") is not None and check.get("status") != "unknown"
        if not known:
            rho, violation, status = 0.0, None, "unknown"
        else:
            alpha = float(check.get("alpha", SEMANTIC_ALPHA))
            tolerance = float(check.get("tolerance", 0.0))
            if check.get("satisfied"):
                violation = 0.0
            elif check.get("violation") is not None:
                violation = float(check["violation"])
            else:
                score = check.get("score")
                score = float(score) if score is not None else 0.0
                violation = (1.0 - max(0.0, min(1.0, score))) if math.isfinite(score) else math.nan
            if not all(math.isfinite(x) for x in (alpha, tolerance, violation)) or alpha <= 0 or tolerance < 0 or violation < 0:
                rho, violation, status = 0.0, None, "unknown"
            else:
                rho, status = math.exp(-alpha * max(0.0, violation - tolerance)), "measured"
        components.append({"name": check.get("name"), "satisfied": check.get("satisfied"),
                           "rho": rho, "violation": violation, "status": status, "evidence": check})
    if not components or any(c["rho"] <= 0 for c in components):
        return 0.0, components
    return math.exp(sum(math.log(c["rho"]) for c in components) / len(components)), components
