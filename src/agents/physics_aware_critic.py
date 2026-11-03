"""Generation-time critic for 7901 PDF Sec.3.3 / Appendix E.

Diagnostic availability is explicit. Missing MLIP/reference evidence never
receives a perfect score. See docs/implementation.md for frozen assumptions.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import math
import os
from typing import Any, Dict, List, Optional
from pymatgen.core import Element, Structure
from src.evaluation.mlip import MLIPResourceError, is_mlip_resource_error
from .paper_physics import (
    BVS_LAMBDA, BVS_NEIGHBOR_RADIUS, ENERGY_LAMBDA_PER_EV,
    ENERGY_MARGIN_EV_PER_ATOM, build_reference_structure, composition_signature,
    covalent_radius, geometry_diagnostics, infer_task_type, same_composition,
    semantic_components, task_record,
)

SEVERITY_RANK = {"critical": 0, "major": 1, "minor": 2}
CATEGORY_RANK = {"execution": 0, "geometric": 1, "identity": 2, "chemical": 2,
                 "thermodynamic": 3, "semantic_alignment": 4}


@dataclass
class CriticIssue:
    severity: str
    category: str
    description: str
    suggestion: str = ""
    metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self):
        return {"severity": self.severity, "category": self.category,
                "description": self.description, "suggestion": self.suggestion,
                "metrics": self.metrics}


@dataclass
class CriticState:
    geometric_indicator: float = 1.0
    chemical_score: float = 1.0
    thermodynamic_score: float = 1.0
    alignment_score: float = 1.0
    issues: List[CriticIssue] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)

    @property
    def overall_score(self):
        components = [self.chemical_score, self.thermodynamic_score, self.alignment_score]
        if self.geometric_indicator <= 0 or any(s <= 0 for s in components):
            return 0.0
        return float(self.geometric_indicator * math.exp(sum(math.log(s) for s in components) / 3))

    @property
    def confidence(self):
        return self.overall_score

    @property
    def recommendation(self):
        if not self.issues:
            return "accept"
        if any(i.category == "execution" and i.severity == "critical" for i in self.issues):
            return "regenerate"
        return "refine"

    def sorted_issues(self):
        return sorted(self.issues, key=lambda i: (CATEGORY_RANK.get(i.category, 99), SEVERITY_RANK.get(i.severity, 99)))

    def to_dict(self):
        return {"geometric_indicator": self.geometric_indicator, "chemical_score": self.chemical_score,
                "thermodynamic_score": self.thermodynamic_score, "alignment_score": self.alignment_score,
                "overall_score": self.overall_score, "confidence": self.confidence,
                "recommendation": self.recommendation, "issues": [i.to_dict() for i in self.sorted_issues()],
                "metrics": self.metrics}

    def to_evaluation(self):
        return {"overall_assessment": "pass" if not self.issues else "fail", "confidence": self.confidence,
                "issues": [i.to_dict() for i in self.sorted_issues()], "strengths": [],
                "recommendation": self.recommendation, "critic_state": self.to_dict()}


class PhysicsAwareCritic:
    """Task-aware geometry, local BVS, relative MLIP, and semantic diagnostics."""

    def __init__(self, overlap_tolerance: Optional[float] = None,
                 enable_chemical: bool = True, enable_thermo: bool = True,
                 mlip_model: str = "chgnet", mlip_device: str = "auto",
                 chemical_lambda: float = BVS_LAMBDA,
                 energy_lambda: float = ENERGY_LAMBDA_PER_EV,
                 energy_margin: float = ENERGY_MARGIN_EV_PER_ATOM,
                 max_mlip_atoms: Optional[int] = None):
        self.overlap_tolerance = overlap_tolerance
        self.enable_chemical = enable_chemical
        self.enable_thermo = enable_thermo
        self.mlip_model = mlip_model
        self.mlip_device = mlip_device
        self.chemical_lambda = float(chemical_lambda)
        self.energy_lambda = float(energy_lambda)
        self.energy_margin = float(energy_margin)
        self.max_mlip_atoms = int(max_mlip_atoms if max_mlip_atoms is not None
                                  else os.getenv("MATERIALSPILOT_MAX_MLIP_ATOMS", "300"))
        if self.max_mlip_atoms <= 0:
            raise ValueError("MLIP atom budget must be positive")
        if not all(math.isfinite(x) for x in (self.chemical_lambda, self.energy_lambda, self.energy_margin)):
            raise ValueError("Critic parameters must be finite")
        if self.chemical_lambda <= 0 or self.energy_lambda <= 0 or self.energy_margin < 0:
            raise ValueError("Penalty strengths must be positive and energy margin nonnegative")
        self._mlip_evaluator = None
        self._reference_energy_cache = {}

    def evaluate(self, structure: Optional[Structure], task_description: str,
                 task_input: Optional[Dict[str, Any]] = None,
                 deterministic_evaluation: Optional[Dict[str, Any]] = None) -> CriticState:
        task = task_record(task_description, task_input)
        state = CriticState()
        if structure is None:
            state.geometric_indicator = state.chemical_score = state.thermodynamic_score = state.alignment_score = 0.0
            state.issues.append(CriticIssue("critical", "execution", "No structure object was produced.",
                                           "Create a pymatgen Structure named 'structure'."))
            state.metrics["availability"] = {"geometric": False, "chemical": False, "thermodynamic": False, "alignment": False}
            state.metrics["quality_availability"] = {
                "complete": False, "score_interpretation": "zero_for_missing_structure",
                "unknown_components": ["chemical", "thermodynamic", "alignment"],
            }
            return state
        for name, check, attribute in (
            ("geometric", self._geometric_check, "geometric_indicator"),
            ("chemical", self._chemical_check, "chemical_score"),
            ("thermodynamic", self._thermodynamic_check, "thermodynamic_score"),
        ):
            try:
                score, issues, metrics = check(structure, task)
            except Exception as exc:
                if is_mlip_resource_error(exc):
                    raise MLIPResourceError(f"{name} diagnostic resource failure: {exc}") from exc
                score, issues, metrics = self._unavailable(name, f"{type(exc).__name__}: {exc}")
            setattr(state, attribute, score)
            state.issues.extend(issues)
            state.metrics[name] = metrics
            if name == "geometric" and score <= 0:
                # Severe overlaps make local valence/MLIP inference unreliable
                # and may create enormous periodic neighbor graphs. Geometry is
                # the dependency gate; only its actionable issue is returned.
                state.chemical_score = state.thermodynamic_score = state.alignment_score = 0.0
                for skipped in ("chemical", "thermodynamic", "alignment"):
                    state.metrics[skipped] = {
                        "available": False, "status": "not_evaluated_geometry_gate",
                        "reason": "Resolve geometric invalidity before dependent diagnostics",
                    }
                self._record_availability(state)
                return state
        try:
            score, issues, metrics = self._alignment_check(structure, task_description, task, deterministic_evaluation)
        except Exception as exc:
            if is_mlip_resource_error(exc):
                raise MLIPResourceError(f"Semantic diagnostic resource failure: {exc}") from exc
            score, issues, metrics = self._unavailable("semantic_alignment", f"{type(exc).__name__}: {exc}")
        state.alignment_score = score
        state.issues.extend(issues)
        state.metrics["alignment"] = metrics
        self._record_availability(state)
        return state

    @staticmethod
    def _record_availability(state):
        dimensions = ("geometric", "chemical", "thermodynamic", "alignment")
        available = {name: bool(state.metrics.get(name, {}).get("available", False)) for name in dimensions}
        unknown_quality = [name for name in dimensions[1:] if not available[name]]
        state.metrics["availability"] = available
        state.metrics["quality_availability"] = {
            "complete": not unknown_quality,
            "unknown_components": unknown_quality,
            "score_interpretation": ("conservative_zero_for_unavailable_component" if unknown_quality
                                     else "evaluated_diagnostic_score"),
            "chemical_bvs_available": bool(state.metrics.get("chemical", {}).get("bvs_available", False)),
        }

    @staticmethod
    def _unavailable(category, reason):
        metrics = {"available": False, "status": "unavailable", "reason": reason}
        issue = CriticIssue("major", category, f"Cannot evaluate {category}: {reason}",
                            "Supply the missing reference or diagnostic evidence; do not assume this check passed.", metrics)
        return 0.0, [issue], metrics

    def _geometric_check(self, structure, task=None):
        metrics = geometry_diagnostics(structure, task or {}, self.overlap_tolerance)
        metrics.update({"available": True, "status": "evaluated"})
        issues = []
        if metrics["indicator"] == 0:
            pair = metrics["overlap_pairs"][0]
            issues.append(CriticIssue("critical", "geometric",
                                     f"Atom pair {pair['i']}/{pair['j']} ({pair['species']}) overlaps: "
                                     f"{pair['distance']:.3f} A < {pair['threshold']:.3f} A.",
                                     "Increase the offending separation while preserving requested composition and geometry.", metrics))
        return float(metrics["indicator"]), issues, metrics

    def _chemical_check(self, structure, task=None):
        if not self.enable_chemical:
            return self._unavailable("chemical", "Chemical critic is disabled")
        if not len(structure) or not structure.is_ordered:
            return self._unavailable("chemical", "Nonempty ordered atomic species are required")
        task = task or {}
        explicit = all(getattr(site.specie, "oxi_state", None) is not None for site in structure)
        decorated, assignment_error = structure, None
        if not explicit and len(structure.composition.elements) > 1:
            try:
                from pymatgen.analysis.bond_valence import BVAnalyzer
                decorated = BVAnalyzer(distance_scale_factor=1.0).get_oxi_state_decorated_structure(structure)
            except MemoryError as exc:
                raise MLIPResourceError(f"Chemical diagnostic ran out of memory: {exc}") from exc
            except Exception as exc:
                assignment_error = f"{type(exc).__name__}: {exc}"
        from pymatgen.analysis.bond_valence import BV_PARAMS, ELECTRONEG, calculate_bv_sum
        records, unsupported = [], []
        for index, site in enumerate(decorated):
            target = getattr(site.specie, "oxi_state", None)
            element = Element(site.specie.symbol)
            if target is None or abs(float(target)) < 1e-12 or element not in BV_PARAMS:
                continue
            neighbors = decorated.get_neighbors(site, BVS_NEIGHBOR_RADIUS)
            usable = [neighbor for neighbor in neighbors
                      if Element(neighbor.specie.symbol) in BV_PARAMS
                      and neighbor.specie.symbol != site.specie.symbol
                      and (element in ELECTRONEG or Element(neighbor.specie.symbol) in ELECTRONEG)]
            if not usable:
                # A supported nominal ion with no bonds has BVS=0; it must
                # not disappear from the deviation denominator.
                unsupported.append(index)
            value = abs(float(calculate_bv_sum(site, usable, scale_factor=1.0)))
            records.append({"index": index, "species": site.specie.symbol,
                            "nominal_oxidation_state": float(target), "bond_valence_sum": value,
                            "deviation": abs(value - abs(float(target)))})
        if not records:
            return self._chemical_fallback(structure, task, assignment_error, explicit)
        deviation = sum(record["deviation"] for record in records) / len(records)
        score = math.exp(-self.chemical_lambda * deviation)
        metrics = {"available": True, "status": "evaluated", "method": "local_BVS_exponential",
                   "bvs_available": True, "lambda_c": self.chemical_lambda,
                   "mean_absolute_bvs_deviation": deviation, "evaluated_sites": len(records),
                   "site_fraction": len(records) / len(structure), "site_evidence": records,
                   "unsupported_sites": unsupported, "oxidation_assignment": "explicit" if explicit else "BVAnalyzer",
                   "parameter_source": "pymatgen BV_PARAMS empirical pair parameters; decay b=0.31 A",
                   "neighbor_radius_angstrom": BVS_NEIGHBOR_RADIUS}
        issues = []
        if deviation > 0.5:
            worst = max(records, key=lambda record: record["deviation"])
            issues.append(CriticIssue("minor", "chemical",
                                     f"Local bond-valence deviation is high at site {worst['index']} ({worst['species']}).",
                                     "Check nominal oxidation states, coordination, and bond distances while preserving stoichiometry.", metrics))
        return float(score), issues, metrics

    def _chemical_fallback(self, structure, task, assignment_error, explicit):
        # Appendix E.2 allows weaker checks when ionic BVS is inapplicable.
        valid_species = all(Element.is_valid_symbol(site.specie.symbol) for site in structure)
        explicit_charge = sum(float(site.specie.oxi_state) for site in structure) if explicit else None
        fields = task.get("input", task)
        constraints = fields.get("constraints", {}) or {}
        neutrality_required = bool(fields.get("charge_neutral", constraints.get("charge_neutral", False)))
        charge_ok = not neutrality_required or (explicit_charge is not None and abs(explicit_charge) <= 1e-6)
        geometry = geometry_diagnostics(structure, task, self.overlap_tolerance)
        checks = {"valid_species": valid_species, "no_severe_local_overlap": geometry["indicator"] == 1}
        if neutrality_required:
            checks["requested_charge_neutrality"] = charge_ok
        score = sum(bool(value) for value in checks.values()) / len(checks)
        metrics = {"available": True, "status": "weaker_fallback", "method": "species_charge_local_distance",
                   "bvs_available": False, "bvs_reason": assignment_error or "No ionic sites with usable nominal valences",
                   "fallback_checks": checks, "explicit_total_charge": explicit_charge,
                   "diagnostic_notes": (["Ionic oxidation-state assignment is unavailable; only the listed weaker checks were evaluated."]
                                        if assignment_error else []),
                   "limitation": "Fallback verifies only the listed checks; local BVS consistency is unverified"}
        issues = []
        if not all(checks.values()):
            issues.append(CriticIssue("major", "chemical", "A weaker chemical consistency check failed.",
                                     "Repair species, severe local contacts, or explicitly requested charge balance.", metrics))
        return float(score), issues, metrics

    def _thermodynamic_check(self, structure, task=None):
        if not self.enable_thermo:
            return self._unavailable("thermodynamic", "MLIP critic is disabled")
        if len(structure) > self.max_mlip_atoms:
            score, issues, metrics = self._unavailable(
                "thermodynamic", f"MLIP atom budget exceeded ({len(structure)} > {self.max_mlip_atoms})")
            metrics.update({"status": "not_evaluated_resource_budget", "atom_count": len(structure),
                            "max_mlip_atoms": self.max_mlip_atoms, "model": self.mlip_model,
                            "enabled_mlip": True, "mlip_evaluated": False})
            return score, issues, metrics
        task = task or {}
        # Even absent a reference, actually run the configured candidate MLIP.
        try:
            evaluator = self._get_mlip_evaluator()
            prediction = evaluator.predict_energy_only(structure)
            energy = float(prediction["energy_per_atom"])
            if not math.isfinite(energy):
                raise ValueError("MLIP energy is not finite")
        except Exception as exc:
            if is_mlip_resource_error(exc):
                raise MLIPResourceError(f"Candidate MLIP resource failure: {exc}") from exc
            return self._unavailable("thermodynamic", f"MLIP prediction failed: {type(exc).__name__}: {exc}")
        base = {"model": self.mlip_model, "device": self.mlip_device, "enabled_mlip": True,
                "candidate_energy_per_atom": energy, "energy_per_atom": energy,
                "lambda_E": self.energy_lambda, "epsilon_E": self.energy_margin,
                "composition": composition_signature(structure), "mlip_evaluated": True,
                "max_mlip_atoms": self.max_mlip_atoms,
                "prediction_mode": prediction.get("prediction_mode", "energy_only_no_grad"),
                "force_diagnostics": "not_requested_non_scoring"}
        reference, reference_metadata = build_reference_structure(structure, task)
        base["reference"] = reference_metadata
        if reference is not None:
            if len(reference) > self.max_mlip_atoms:
                score, issues, metrics = self._unavailable(
                    "thermodynamic", f"Reference MLIP atom budget exceeded ({len(reference)} > {self.max_mlip_atoms})")
                metrics.update(base)
                metrics["status"] = "not_evaluated_resource_budget"
                return score, issues, metrics
            try:
                cache_key = f"{self.mlip_model}:{reference_metadata['structure_sha256']}"
                if cache_key not in self._reference_energy_cache:
                    value = float(evaluator.predict_energy_only(reference)["energy_per_atom"])
                    if not math.isfinite(value):
                        raise ValueError("Reference MLIP energy is not finite")
                    self._reference_energy_cache[cache_key] = value
                reference_energy = self._reference_energy_cache[cache_key]
            except Exception as exc:
                if is_mlip_resource_error(exc):
                    raise MLIPResourceError(f"Reference MLIP resource failure: {exc}") from exc
                score, issues, metrics = self._unavailable("thermodynamic", f"Reference MLIP failed: {exc}")
                metrics.update(base)
                return score, issues, metrics
            delta = energy - reference_energy
            score = math.exp(-self.energy_lambda * max(0.0, delta - self.energy_margin))
            metrics = {**base, "available": True, "status": "evaluated", "method": "same_composition_relative_energy",
                       "reference_energy_per_atom": reference_energy, "delta_energy_per_atom": delta, "score": score}
        else:
            # No universal absolute energy window: task and composition must match.
            fields = task.get("input", task)
            specification = fields.get("thermodynamic_reference", task.get("thermodynamic_reference", {})) or {}
            bounds = specification.get("energy_window")
            try:
                if not bounds or not specification.get("provenance") or not specification.get("composition"):
                    raise ValueError("No provenance-bearing task-conditioned energy range")
                if specification.get("task_type") != infer_task_type(task):
                    raise ValueError("Reference energy range task family does not match")
                if not same_composition(structure, specification["composition"]):
                    raise ValueError("Reference energy range composition does not match")
                lower, upper = map(float, bounds)
                if not math.isfinite(lower) or not math.isfinite(upper) or lower > upper:
                    raise ValueError("Reference energy range is invalid")
                distance = max(lower - energy, 0.0, energy - upper)
                score = math.exp(-self.energy_lambda * distance)
                metrics = {**base, "available": True, "status": "evaluated", "method": "task_conditioned_admissible_range",
                           "energy_window": [lower, upper], "range_provenance": specification["provenance"], "score": score}
            except Exception as exc:
                score, issues, metrics = self._unavailable("thermodynamic", f"{reference_metadata.get('reason')}; {exc}")
                metrics.update(base)
                return score, issues, metrics
        issues = []
        if score < 0.5:
            issues.append(CriticIssue("major", "thermodynamic", "MLIP energy is unfavorable relative to the documented task reference.",
                                     "Reduce local strain, repair interface separation or adsorption placement while retaining task constraints.", metrics))
        return float(score), issues, metrics

    def _alignment_check(self, structure, task_description, task_input, deterministic_evaluation=None):
        from src.evaluation.paper_constraints import collect_task_constraints
        task = task_record(task_description, task_input)
        checks = collect_task_constraints(structure, task, infer_task_type(task))
        checks = [check.to_dict() if hasattr(check, "to_dict") else dict(check) for check in checks]
        score, components = semantic_components(checks)
        known = bool(components) and all(component["status"] != "unknown" for component in components)
        metrics = {"available": known, "status": "evaluated" if known else "partial_or_unavailable",
                   "method": "constraint_exponentials_geometric_mean", "components": components,
                   "constraint_count": len(components), "unknown_count": sum(c["status"] == "unknown" for c in components),
                   "violation_proxy": "1 - bounded diagnostic satisfaction when no explicit violation is supplied"}
        issues = []
        for component in components:
            if component["satisfied"] is not True or component["status"] == "unknown":
                evidence = component["evidence"]
                description = evidence.get("message") or f"Prompt constraint {component['name']} is not established."
                issues.append(CriticIssue("major", "semantic_alignment", description,
                                         f"Enforce and make observable the requested constraint: {component['name']}.", component))
        if not components:
            issues.append(CriticIssue("major", "semantic_alignment", "No auditable prompt constraints were produced.",
                                     "Provide explicit task constraints and a supported task family.", metrics))
        return float(score), issues, metrics

    def _get_mlip_evaluator(self):
        if self._mlip_evaluator is None:
            from src.evaluation.mlip import get_default_evaluator
            self._mlip_evaluator = get_default_evaluator(model_type=self.mlip_model, device=self.mlip_device)
        return self._mlip_evaluator

    def _atomic_radius(self, specie):
        """Compatibility name: this implementation always returns covalent radii."""
        return covalent_radius(specie)
