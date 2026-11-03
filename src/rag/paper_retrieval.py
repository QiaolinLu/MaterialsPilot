"""Auditable implementation of 7901 Appendix B retrieval equations.

Learned channels use frozen official pretrained assets. Lexical rules are a
separate repair supplement, or a fallback only when all three scientific
channels have no results; they never enter RRF. See docs/implementation.md.
"""
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from copy import deepcopy
import hashlib
import json
import math
import re
import unicodedata
import numpy as np
from pymatgen.core import Composition, Element, Lattice, Structure
from .paper_encoders import ASSETS, cosine, get_encoders

PROPERTY_ALIASES = {
    "formation_energy_per_atom": "Eform", "formation_energy": "Eform", "eform": "Eform",
    "band_gap": "Egap", "bandgap": "Egap", "egap": "Egap",
    "shear_modulus": "Gshear", "gshear": "Gshear",
    "bulk_modulus": "Kbulk", "kbulk": "Kbulk",
}
STOPWORDS = set("a an the and or of for to in on at by from with as is are be create build generate structure atom atoms material task".split())
FORMULA_KEYS = {"material", "host_material", "host", "metal", "dopant", "element", "elements", "composition",
                "film_material", "substrate_material", "adsorbate", "adsorbates", "surface_species", "formula",
                "original_elements", "composition_contains", "dopant_site", "removed_element",
                "surface_metal", "pristine_formula", "adsorbate_formula", "defect_element", "substitute_element",
                "interstitial_element", "vacancy_element", "vacancy_elements", "antisite_species", "swap_atoms"}
STRUCTURE_MATERIAL_KEYS = {"material", "host_material", "host", "metal", "film_material", "substrate_material",
                           "surface_metal", "pristine_formula"}
GEOMETRY_KEYS = {"facet", "miller_index", "structure_type", "site", "supercell", "sigma", "gb_plane", "gb_type",
                 "rotation_axis", "rotation_angle", "film_facet", "substrate_facet", "vacuum", "coverage", "lattice"}
TASK_WORDS = {
    "adsorption": ("adsorption", "adsorbate", "adsorb", "coadsorption"),
    "grain_boundary": ("grain boundary", "grain_boundary", "csl", "twin boundary"),
    "interface": ("interface", "heterostructure", "heterojunction"),
    "defect": ("defect", "vacancy", "substitution", "interstitial", "dopant"),
}
# Material identity aliases, independent of task IDs. The graphene/hBN/P aliases
# match the existing evaluation schema; carbon allotropes retain their original
# names for structure lookup rather than being relabelled as a bulk prototype.
MATERIAL_FORMULAS = {"graphene": "C", "graphite": "C", "diamond": "C",
                     "hbn": "BN", "h-bn": "BN", "black_phosphorus": "P", "black phosphorus": "P"}
# The existing interface schema identifies YSZ's elements, but does not specify
# the yttria fraction. Element identity must never become an invented formula.
MATERIAL_ELEMENTS_ONLY = {"ysz": ["O", "Y", "Zr"]}


def tokens(text):
    return set(re.findall(r"[a-z0-9]+", unicodedata.normalize("NFKC", str(text)).lower().replace("_", " "))) - STOPWORDS


def formula(value):
    if not isinstance(value, str):
        return None
    value = unicodedata.normalize("NFKC", value).split("_")[0].strip()
    if not re.fullmatch(r"(?:[A-Z][a-z]?\d*(?:\.\d+)?|[()])+", value):
        return None
    try:
        comp = Composition(value, strict=True)
        if comp and all(element.symbol in Element.__members__ for element in comp.elements):
            return comp.as_dict()
    except (ValueError, TypeError):
        pass
    return None


@dataclass
class TaskQuery:
    text: str
    task_type: str | None
    elements: list[str] = field(default_factory=list)
    compositions: list[dict] = field(default_factory=list)
    materials: list[str] = field(default_factory=list)
    structure_materials: list[str] = field(default_factory=list)
    geometry: dict = field(default_factory=dict)
    properties: dict = field(default_factory=dict)
    constraints: dict = field(default_factory=dict)
    unresolved_compositions: list[dict] = field(default_factory=list)

    def as_dict(self):
        return asdict(self)


def parse_task_query(text: str, task_input: dict | None = None, task_type: str | None = None) -> TaskQuery:
    """Deterministic Eq13 parser: task fields are authoritative, not examples.

    This is an inspectable parser, not an unrecorded LLM query. Element presence
    alone does not imply a molecular count or a target thermodynamic property.
    Once structured materials exist, formatted prompt/repair text cannot add
    unrelated elements. Original task descriptions fill missing field groups;
    free query text is a fallback only when no original description is supplied.
    """
    data = task_input or {}
    original = data.get("description") or (data.get("_full_task") or {}).get("description")
    normalized = unicodedata.normalize("NFKC", original if isinstance(original, str) and original.strip() else text)
    if not task_type:
        explicit_type = data.get("task_type") or (data.get("_full_task") or {}).get("task_type")
        if isinstance(explicit_type, str):
            if explicit_type in TASK_WORDS:
                task_type = explicit_type
            elif explicit_type in {"co_adsorption", "stepped_adsorption", "complex_adsorption", "coverage_adsorption"}:
                task_type = "adsorption"
    if not task_type:
        for candidate, words in TASK_WORDS.items():
            if any(word in normalized.lower() for word in words):
                task_type = candidate
                break
    query = TaskQuery(text=text, task_type=task_type)
    known_elements = set()

    def add_material(value, structure_role=False):
        if isinstance(value, dict):
            if value and all(
                key in Element.__members__ and isinstance(amount, (int, float))
                and not isinstance(amount, bool) and math.isfinite(amount) and amount >= 0
                for key, amount in value.items()
            ) and sum(value.values()) > 0:
                composition = Composition(value)
                name = composition.reduced_formula
                if name not in query.materials:
                    query.materials.append(name)
                    query.compositions.append(composition.as_dict())
                if structure_role and name not in query.structure_materials:
                    query.structure_materials.append(name)
                return
            for subkey in ("formula", "name", "species", "material"):
                if subkey in value:
                    add_material(value[subkey], structure_role)
        elif isinstance(value, list):
            for item in value:
                add_material(item, structure_role)
        elif isinstance(value, str):
            # Element names are legitimate structured material fields as well.
            value = next((element.symbol for element in Element
                          if value.casefold().strip() == element.long_name.casefold()), value)
            alias = value.casefold().strip()
            if alias in MATERIAL_ELEMENTS_ONLY:
                known_elements.update(MATERIAL_ELEMENTS_ONLY[alias])
                if value not in query.materials:
                    query.materials.append(value)
                    query.unresolved_compositions.append({
                        "material": value, "known_elements": MATERIAL_ELEMENTS_ONLY[alias],
                        "reason": "Stabilizer fraction is unspecified; element identity only, no stoichiometry assigned.",
                    })
                if structure_role and value not in query.structure_materials:
                    query.structure_materials.append(value)
                return
            parsed = formula(MATERIAL_FORMULAS.get(alias, value))
            if parsed and value not in query.materials:
                query.materials.append(value)
                query.compositions.append(parsed)
            if parsed and structure_role and value not in query.structure_materials:
                query.structure_materials.append(value)

    def walk(mapping, path=""):
        for key, value in mapping.items():
            name = str(key).lower()
            keypath = f"{path}.{key}".strip(".")
            if name in FORMULA_KEYS:
                add_material(value, name in STRUCTURE_MATERIAL_KEYS)
            if name in GEOMETRY_KEYS:
                query.geometry[keypath] = value
            if name in {"constraints", "validation", "expected_properties"} and isinstance(value, dict):
                query.constraints[keypath] = value
            if name in PROPERTY_ALIASES and isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value):
                query.properties[PROPERTY_ALIASES[name]] = float(value)
            if isinstance(value, dict):
                walk(value, keypath)
    walk(data)
    if not query.compositions and not known_elements:
        for alias in (*MATERIAL_FORMULAS, *MATERIAL_ELEMENTS_ONLY):
            if re.search(r"\b" + re.escape(alias) + r"\b", normalized, re.I):
                add_material(alias)
        for match in re.finditer(r"(?<![A-Za-z])(?:[A-Z][a-z]?\d*)+(?![A-Za-z])", normalized):
            candidate = match.group()
            # Single-letter English pronouns and common method acronyms are not formulas.
            if candidate not in {"I", "A", "RAG", "API", "CGCNN", "DFT", "MLIP", "CSL", "FCC", "BCC", "HCP"}:
                add_material(candidate)
        for element in Element:
            if re.search(r"\b" + re.escape(element.long_name.lower()) + r"\b", normalized.lower()):
                add_material(element.symbol)
    if not any(key.rsplit(".", 1)[-1] == "structure_type" for key in query.geometry):
        for name in ("fcc", "bcc", "hcp", "diamond", "zincblende", "rocksalt", "rutile", "corundum"):
            if re.search(r"\b" + name + r"\b", normalized.lower()):
                query.geometry["structure_type"] = name
                break
    for alias, name in PROPERTY_ALIASES.items():
        if name in query.properties:
            continue
        match = re.search(r"\b" + alias.replace("_", r"[ _-]") + r"\s*(?:=|of|:)?\s*(-?\d+(?:\.\d+)?)", normalized, re.I)
        if match:
            query.properties[name] = float(match.group(1))
    query.elements = sorted(known_elements | {element for comp in query.compositions for element in comp})
    return query


def reciprocal_rank_fusion(rankings: dict[str, list[dict]], k: float = 60.0, weights=None):
    """Eq2/B.8 with one-based ranks, deterministic ties and no score rescaling."""
    if k <= 0:
        raise ValueError("RRF k must be positive")
    fused = {}
    for channel, ranking in rankings.items():
        seen = set()
        for rank, record in enumerate(ranking, 1):
            identifier = record["id"]
            if identifier in seen:
                raise ValueError(f"Duplicate document in {channel}: {identifier}")
            seen.add(identifier)
            target = fused.setdefault(identifier, {**record, "score": 0.0, "channel_ranks": {}, "channel_scores": {}})
            target["score"] += (weights or {}).get(channel, 1.0) / (k + rank)
            target["channel_ranks"][channel] = rank
            target["channel_scores"][channel] = record["score"]
    return sorted(fused.values(), key=lambda row: (-row["score"], row["id"]))


class PropertyIndex:
    """B.7 Z-score + weighted RBF, using observed finite dimensions only.

    Each corpus row must contain provenance. Missing dimensions are never zero
    imputed; no target means no ranking. Statistics use the frozen corpus, never
    the evaluation tasks. Current bundled corpus has legacy band_gap only.
    """
    def __init__(self, records, sigma=1.0, weights=None):
        if sigma <= 0:
            raise ValueError("RBF sigma must be positive")
        self.records, self.sigma = list(records), sigma
        self.weights = weights or {name: 1.0 for name in ("Eform", "Egap", "Gshear", "Kbulk")}
        self.statistics = {}
        for name in self.weights:
            values = [row["properties"][name] for row in self.records if name in row["properties"] and math.isfinite(row["properties"][name])]
            if len(values) >= 2 and float(np.std(values)) > 0:
                self.statistics[name] = {"mean": float(np.mean(values)), "std": float(np.std(values)), "n": len(values)}

    def rank(self, target):
        ranked = []
        for row in self.records:
            common = sorted(set(target) & set(row["properties"]) & set(self.statistics))
            if not common:
                continue
            distance = sum(self.weights[name] * ((float(target[name]) - row["properties"][name]) / self.statistics[name]["std"]) ** 2 for name in common)
            ranked.append({**row, "score": math.exp(-distance / (2 * self.sigma ** 2)), "property_dimensions": common})
        return sorted(ranked, key=lambda row: (-row["score"], row["id"]))


def make_prototype(material, info):
    """Instantiate only fully specified existing cells or explicit standard bases.

    No relaxed/DFT status is assigned. Missing internal coordinates remain missing
    (e.g. rutile/corundum); no fabricated coordinates or random structures.
    """
    comp = formula(material)
    if not comp:
        return None
    kind = str(info.get("structure_type", info.get("structure", ""))).lower()
    a = info.get("lattice_constant", info.get("lattice_constant_a"))
    c = info.get("lattice_constant_c")
    if not isinstance(a, (float, int)):
        return None
    if kind == "hcp" and c:
        lattice = Lattice.hexagonal(a, c)
    elif kind in {"fcc", "bcc", "diamond", "zincblende", "rocksalt", "fluorite", "perovskite"}:
        lattice = Lattice.cubic(a)
    else:
        return None
    basis = info.get("basis_positions")
    if isinstance(basis, dict):
        species, positions = [], []
        for element, points in basis.items():
            species.extend([element] * len(points)); positions.extend(points)
    elif basis and len(comp) == 1:
        positions, species = basis, [next(iter(comp))] * len(basis)
    elif len(comp) == 1 and kind in {"fcc", "bcc", "diamond", "hcp"}:
        bases = {
            "fcc": [[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0]],
            "bcc": [[0, 0, 0], [.5, .5, .5]],
            "hcp": [[0, 0, 0], [1/3, 2/3, .5]],
        }
        bases["diamond"] = bases["fcc"] + (np.asarray(bases["fcc"]) + .25).tolist()
        positions = bases[kind]; species = [next(iter(comp))] * len(positions)
    else:
        return None
    result = Structure(lattice, species, positions)
    return result if result.composition.reduced_composition == Composition(comp).reduced_composition else None


def knowledge_documents(store, categories):
    """Stable entry IDs and bounded leaf chunks, with original category provenance."""
    records = []
    for category in sorted(categories):
        data = store.get_data(category)
        source_hash = hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()
        excluded = data.get("_metadata", {}).get("deprecated_retrieval_paths", [])
        def visit(path, value, inherited=None):
            dotted = ".".join(path)
            if any(dotted == prefix or dotted.startswith(prefix + ".") for prefix in excluded):
                return
            serialized = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
            if isinstance(value, dict) and len(serialized) > 1800 and not value.get("summary"):
                for key, child in value.items():
                    if not str(key).startswith("_"):
                        visit([*path, str(key)], child, inherited)
                return
            if isinstance(value, list) and len(serialized) > 1800:
                for index, child in enumerate(value):
                    visit([*path, str(index)], child, inherited)
                return
            topic = ".".join(path)
            identifier = f"{category}.{topic}"
            entry = value if isinstance(value, dict) else {}
            summary = entry.get("summary") or (f"{topic}: {serialized}" if len(serialized) <= 650 else f"{topic}: {entry.get('description', entry.get('notes', 'structured reference entry'))}")
            # Long scalar prose is chunked losslessly rather than truncating rules.
            parts = [serialized[i:i + 1800] for i in range(0, len(serialized), 1800)] if len(serialized) > 2400 else [serialized]
            for index, part in enumerate(parts):
                record_id = identifier if len(parts) == 1 else f"{identifier}.chunk{index}"
                record_summary = summary if len(parts) == 1 else f"{topic} (part {index + 1}/{len(parts)})"
                meta = {"topic": topic, "summary": record_summary, "tier": category.split("_")[0],
                        "source": category, "source_sha256": source_hash,
                        "condition": "Applies when the task matches this entry's material/structure and stated conditions.",
                        "priority": "retrieval_rank", "principles": entry.get("principles", []) or [part]}
                if entry.get("code"):
                    meta["code"] = entry["code"]
                comp = next((formula(part) for part in path if formula(part)), None)
                records.append({"id": record_id, "source": category, "content": part, "metadata": meta,
                                "composition": comp, "entry_path": path, "entry": entry})
        for key, value in data.items():
            if not key.startswith("_"):
                visit([key], value)
    return records


class PaperRetriever:
    def __init__(self, store, categories, task_type=None, strict=True, asset_dir=ASSETS):
        self.store, self.categories, self.task_type = store, tuple(sorted(categories)), task_type
        self.records = knowledge_documents(store, self.categories)
        self.strict, self.asset_error = strict, None
        self.composition_encoder = self.structure_encoder = None
        self.asset_manifest = {}
        try:
            self.composition_encoder, self.structure_encoder, self.asset_manifest = get_encoders(str(asset_dir))
        except (OSError, ValueError, ImportError, RuntimeError) as exc:
            self.asset_error = f"{type(exc).__name__}: {exc}"
            if strict:
                raise RuntimeError(f"Paper RAG learned assets unavailable: {self.asset_error}") from exc
        self.prototypes, self.prototype_types = {}, {}
        for category in ("L2_structures.bulk.crystal_structures", "L2_structures.bulk.metal_properties"):
            if category not in self.categories:
                continue
            for name, info in store.get_data(category).items():
                if name.startswith("_") or not isinstance(info, dict):
                    continue
                prototype = make_prototype(name, info)
                if prototype is not None and name not in self.prototypes:
                    self.prototypes[name] = prototype
                    self.prototype_types[name] = str(info.get("structure_type", info.get("structure", ""))).lower()
        # Association is based only on explicit material/prototype words in the
        # stored document, independent of the current query or benchmark results.
        self.document_prototypes = {}
        for row in self.records:
            if not row["source"].startswith("L2_"):
                continue
            names = [name for name in self.prototypes if name in row["entry_path"]]
            if not names:
                words = tokens(" ".join(row["entry_path"]) + " " + row["content"])
                kinds = set(self.prototype_types.values()) & words
                names = [min(name for name, kind in self.prototype_types.items() if kind == value) for value in sorted(kinds)]
            if names:
                self.document_prototypes[row["id"]] = names
        properties = []
        for row in self.records:
            # Only explicitly labelled bulk band gaps, not adsorption/defect energies.
            if row["source"] != "L2_structures.bulk.crystal_structures":
                continue
            value = row["entry"].get("band_gap")
            if isinstance(value, (int, float)) and math.isfinite(value):
                properties.append({**row, "properties": {"Egap": float(value)},
                                   "property_provenance": "Existing KB band_gap in eV; electronic-structure method unspecified."})
        self.property_index = PropertyIndex(properties)
        self.last_metadata = {}
        self._record_tokens = {
            row["id"]: (tokens(row["metadata"]["topic"] + " " + row["source"]), tokens(row["content"]))
            for row in self.records
        }
        self._query_cache = {}

    def retrieve(self, query, k=8, task_input=None):
        if k <= 0:
            self.last_metadata = {"implementation": "paper_rag_v1", "retrieval_route": "no_results_requested", "selected_ids": []}
            return []
        task = query if isinstance(query, TaskQuery) else parse_task_query(str(query), task_input, self.task_type)
        cache_key = json.dumps([task.as_dict(), k], sort_keys=True)
        if cache_key in self._query_cache:
            records, metadata = self._query_cache[cache_key]
            self.last_metadata = {**deepcopy(metadata), "cache_hit": True}
            return deepcopy(records)
        words = tokens(task.text + " " + json.dumps(task.geometry) + " " + str(task.task_type or ""))
        rankings = {"L1_composition": [], "L2_structure": [], "L3_property": []}
        statuses = {}
        missing_elements = sorted(set(task.elements) - set(self.composition_encoder.vectors)) if self.composition_encoder else task.elements
        if self.composition_encoder and task.compositions:
            for row in self.records:
                if not row["source"].startswith("L1_") or not row["composition"]:
                    continue
                try:
                    score = max(self.composition_encoder.similarity(comp, row["composition"]) for comp in task.compositions)
                    rankings["L1_composition"].append({**row, "score": score})
                except ValueError:
                    continue
            statuses["L1_composition"] = "ready" if rankings["L1_composition"] else "no_supported_compositions"
        else:
            statuses["L1_composition"] = "no_query_composition" if not task.compositions else "missing_asset"
        structure_materials = task.structure_materials or task.materials
        query_prototypes = [name for name in structure_materials if name in self.prototypes]
        if not query_prototypes:
            kinds = {str(value).lower() for value in task.geometry.values() if isinstance(value, str)}
            query_prototypes = [min(name for name, kind in self.prototype_types.items() if kind == value) for value in sorted(kinds & set(self.prototype_types.values()))]
        if self.structure_encoder and query_prototypes:
            query_vectors = [self.structure_encoder.encode(self.prototypes[name]) for name in query_prototypes]
            for row in self.records:
                names = self.document_prototypes.get(row["id"], [])
                if names:
                    score = max(cosine(q, self.structure_encoder.encode(self.prototypes[name])) for q in query_vectors for name in names)
                    rankings["L2_structure"].append({**row, "score": score, "structure_prototypes": names})
            statuses["L2_structure"] = "ready"
        else:
            statuses["L2_structure"] = "no_supported_query_prototype" if self.structure_encoder else "missing_asset"
        rankings["L3_property"] = self.property_index.rank(task.properties)
        statuses["L3_property"] = ("ready_partial_property_corpus" if rankings["L3_property"] else
                                    "no_query_property_target" if not task.properties else "no_corpus_dimensions_for_target")
        for name, ranking in rankings.items():
            # Text only resolves equal scientific scores, never changes those
            # scores or supplies an additional RRF channel.
            rankings[name] = sorted(ranking, key=lambda row: (-row["score"], -self._lexical_score(row, words), row["id"]))[:max(30, 4 * k)]
        fused = reciprocal_rank_fusion(rankings)
        route = "three_channel_rrf"
        if not fused:
            route = "all_scientific_channels_empty_lexical_fallback"
            fused = [{**row, "channel_ranks": {}, "channel_scores": {}} for row in self._rank_rules(words)]
        result = []
        for index, row in enumerate(fused[:k], 1):
            # Do not expose full internal entry objects in prompt/log records.
            out = {key: value for key, value in row.items() if key not in {"entry", "entry_path", "composition"}}
            out["metadata"] = {**row["metadata"], "priority": index, "channel_ranks": row["channel_ranks"]}
            result.append(out)
        self.last_metadata = {
            "implementation": "paper_rag_v1", "query": task.as_dict(), "channel_status": statuses,
            "channel_result_counts": {name: len(rows) for name, rows in rankings.items()},
            "asset_error": self.asset_error, "missing_mat2vec_elements": missing_elements,
            "query_structure_prototypes": query_prototypes,
            "materials_without_explicit_prototype": [name for name in structure_materials if name not in self.prototypes],
            "prototype_corpus": "Standard cells instantiated from existing KB lattice constants and explicit bases; not a downloaded relaxed MP structure corpus.",
            "cgcnn_representation": "official pretrained final-convolution mean pooling; 64 dimensions",
            "property_statistics": self.property_index.statistics,
            "property_corpus_limitation": "Existing KB band_gap only; method unspecified; no complete Eform/Egap/Gshear/Kbulk corpus.",
            "rrf_k": 60, "rrf_rank_origin": 1, "retrieval_route": route,
            "lexical_rules": "Not fused; available separately for repair or when all scientific channels are empty.",
            "asset_hashes": self.asset_manifest.get("files", {}),
            "document_count": len(self.records), "selected_ids": [row["id"] for row in result],
            "cache_hit": False,
            "deprecated_retrieval_paths": {
                category: self.store.get_data(category).get("_metadata", {}).get("deprecated_retrieval_paths", [])
                for category in self.categories
                if self.store.get_data(category).get("_metadata", {}).get("deprecated_retrieval_paths")
            },
            "corpus_sha256": hashlib.sha256(json.dumps([(row["id"], row["metadata"]["source_sha256"]) for row in self.records], sort_keys=True).encode()).hexdigest(),
        }
        self._query_cache[cache_key] = deepcopy((result, self.last_metadata))
        if len(self._query_cache) > 128:
            self._query_cache.pop(next(iter(self._query_cache)))
        return result

    def _lexical_score(self, row, words):
        title, body = self._record_tokens[row["id"]]
        return 3 * len(words & title) + len(words & body)

    def _rank_rules(self, words):
        rows = [{**row, "score": float(self._lexical_score(row, words))} for row in self.records]
        return sorted((row for row in rows if row["score"] > 0), key=lambda row: (-row["score"], row["id"]))

    def retrieve_rules(self, query, k=3, task_input=None):
        task = parse_task_query(str(query), task_input, self.task_type)
        words = tokens(task.text + " " + json.dumps(task.geometry) + " " + str(task.task_type or ""))
        records = []
        for rank, row in enumerate(self._rank_rules(words)[:max(0, k)], 1):
            records.append({key: value for key, value in row.items() if key not in {"entry", "entry_path", "composition"}})
            records[-1]["metadata"] = {**row["metadata"], "priority": rank,
                                       "retrieval_route": "separate_lexical_repair_supplement"}
        self.last_metadata = {
            "implementation": "paper_rag_v1", "query": task.as_dict(),
            "retrieval_route": "separate_lexical_repair_supplement", "not_part_of_rrf": True,
            "selected_ids": [row["id"] for row in records], "asset_hashes": self.asset_manifest.get("files", {}),
            "document_count": len(self.records), "asset_error": self.asset_error,
        }
        return records


def space_group_reference(number):
    """All 230 groups, generated from pymatgen's crystallographic tables.

    Operations use pymatgen's conventional default setting; this does not claim
    a complete Wyckoff-coordinate catalogue or every alternative origin setting.
    """
    from importlib.metadata import version
    from pymatgen.symmetry.groups import SpaceGroup
    if isinstance(number, bool) or not isinstance(number, int) or not 1 <= number <= 230:
        raise ValueError("Space-group number must be an integer from 1 to 230")
    group = SpaceGroup.from_int_number(number)
    operations = sorted(group.symmetry_ops, key=lambda op: op.as_xyz_str())
    return {"number": number, "symbol": group.symbol, "crystal_system": group.crystal_system,
            "operations": [op.as_xyz_str() for op in operations],
            "affine_matrices": [op.affine_matrix.tolist() for op in operations],
            "source": "pymatgen.symmetry.groups.SpaceGroup", "pymatgen_version": version("pymatgen"),
            "setting": "pymatgen default conventional setting", "wyckoff_coverage": "not supplied by this API"}
