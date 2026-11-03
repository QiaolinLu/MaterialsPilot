"""
Feedback transformation for execution-feedback optimization.

The Methods section describes a transformation:

    f_t = Psi(e_t, R(S_t, tau), tau)

This module implements that Psi as a pluggable component. The default
RuleBasedFeedbackTransformer is deterministic and reproducible. The
LLMFeedbackTransformer keeps the same interface but lets an LLM rewrite the
evidence into a compact repair plan when desired.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
import re
from typing import Any, Dict, List, Optional

from .physics_aware_critic import CriticState


@dataclass
class FeedbackMessage:
    """Actionable feedback plus retrieval queries for the next state update."""

    text: str
    repair_queries: List[str] = field(default_factory=list)
    stop: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "repair_queries": self.repair_queries,
            "stop": self.stop,
            "metadata": self.metadata,
        }


class ConstraintEvidenceRenderer:
    """Render saved observations, never rescore or infer unknown measurements.

    Physical residuals below are display-only differences from the recorded
    target. The critic's dimensionless ``violation``/``rho`` are deliberately
    excluded: they are not distances, angles, or missing atoms.
    """

    @staticmethod
    def check(component):
        if not isinstance(component, dict) or "name" not in component:
            return None
        evidence = component.get("evidence")
        row = evidence if isinstance(evidence, dict) and "satisfied" in evidence else component
        return row if "satisfied" in row and row.get("hard", True) else None

    @staticmethod
    def number(value):
        if isinstance(value, (str, bool, dict, list, tuple)) or value is None:
            return None
        try:
            value = float(value)
            return value if math.isfinite(value) else None
        except (ValueError, TypeError):
            return None

    @classmethod
    def compact(cls, value):
        number = cls.number(value)
        if number is not None:
            return f"{number:.6g}"
        if isinstance(value, dict):
            # Fit internals and aggregate/proxy scores are not repair targets.
            hidden = {"score", "rho", "violation", "alpha", "left", "right"}
            return "{" + ", ".join(f"{k}: {cls.compact(v)}" for k, v in value.items()
                                    if k not in hidden) + "}"
        if isinstance(value, (list, tuple)):
            return "[" + ", ".join(cls.compact(v) for v in value) + "]"
        return "unavailable" if value is None else str(value)

    @staticmethod
    def unit(name, value, threshold):
        if isinstance(threshold, dict) and "tolerance_degrees" in threshold:
            return "degrees"
        if "layer" in name:
            return "layers"
        if any(part in name for part in ("vacuum", "distance", "height", "bonding")) or (
            isinstance(value, dict) and "lateral_mismatch" in value
        ):
            return "Å"
        if "count" in name or name in ("min_atoms", "max_atoms"):
            return "sites/atoms"
        return ""

    @classmethod
    def measurement(cls, row):
        name = str(row.get("name", "constraint"))
        value, target = row.get("value"), row.get("requirement")
        threshold = row.get("threshold", target)
        unit = cls.unit(name, value, threshold)
        suffix = f" {unit}" if unit else ""
        parts = [f"actual={cls.compact(value)}{suffix}",
                 f"requested target={cls.compact(target)}",
                 f"threshold={cls.compact(threshold)}{suffix}"]
        bounds = threshold if isinstance(threshold, dict) else {}
        if isinstance(value, dict) and "lateral_mismatch" in value:
            actual = cls.number(value.get("lateral_mismatch"))
            limit = cls.number(value.get("tolerance_angstrom"))
            if actual is not None and limit is not None:
                parts.append(f"lateral residual={actual:.6g} Å; allowed={limit:.6g} Å; "
                             f"excess={max(0., actual-limit):.6g} Å")
        elif isinstance(value, dict) and any(k in value for k in ("error_degrees", "measured_rotation_error_degrees")):
            error = cls.number(value.get("error_degrees", value.get("measured_rotation_error_degrees")))
            limit = cls.number(bounds.get("tolerance_degrees"))
            if error is not None:
                parts.append(f"angular residual={error:.6g} degrees")
                if limit is not None:
                    parts.append(f"allowed={limit:.6g} degrees; excess={max(0., error-limit):.6g} degrees")
        elif "layer" in name and isinstance(value, dict):
            required = cls.number(threshold)
            if required is not None:
                for role, count in value.items():
                    actual = cls.number(count)
                    if actual is not None:
                        parts.append(f"{role}: actual={actual:.6g} layers; target>={required:.6g} layers; "
                                     f"shortfall={max(0., required-actual):.6g} layers")
        else:
            actual = cls.number(value)
            if isinstance(value, dict):
                for key in ("height", "min_distance", "measured"):
                    actual = cls.number(value.get(key))
                    if actual is not None:
                        break
            required = cls.number(bounds.get("minimum", threshold))
            if actual is not None:
                interval = bounds.get("range")
                if isinstance(interval, (list, tuple)) and len(interval) == 2 and all(cls.number(x) is not None for x in interval):
                    lower, upper = map(float, interval)
                    parts.append(f"distance outside requested range={max(lower-actual, 0., actual-upper):.6g}{suffix} "
                                 "(before the separately reported tolerance)")
                elif required is not None:
                    if name.startswith("min_") or "minimum" in bounds:
                        parts.append(f"shortfall to requested minimum={max(0., required-actual):.6g}{suffix} "
                                     "(before the separately reported tolerance)")
                    elif name.startswith("max_") or "maximum" in bounds:
                        parts.append(f"excess over requested maximum={max(0., actual-required):.6g}{suffix}")
                    else:
                        parts.append(f"actual minus target={actual-required:.6g}{suffix}")
        return "; ".join(parts)

    @staticmethod
    def repair(name):
        if "layer" in name:
            return "Adjust only the deficient slab roles along their normal; recount layers on both sides and recheck contact and vacuum."
        if "vacuum" in name:
            return "Adjust the empty periodic gap along the slab normal while preserving slab Cartesian geometry and interfacial contact; remeasure the largest periodic gap."
        if "site" in name and ("adsorb" in name or "adsorption" in name):
            return "Move the indicated adsorbate laterally to the requested site using periodic surface images; preserve its molecular geometry and height."
        if "height" in name:
            return "Translate only the indicated adsorbate along the local surface normal into the requested height range; preserve its lateral site and internal bonds."
        if "bonding" in name or "interface_distance" in name:
            return "Adjust the relative slab separation to establish the requested cross-interface contact/pair; preserve each slab, its composition and layer count."
        if name in ("gb_plane", "rotation_axis", "rotation_angle", "csl_relation", "boundary_type", "bicrystal"):
            return "Correct the grain/lattice transformation and commensurate periodic cell, then independently remeasure the two grain orientations; changing a declared label is insufficient."
        if "count" in name:
            return "Check the reference-cell convention, supercell multiplicity and before/after site ledger; correct only the requested species or defect operation."
        return "Change only the construction operation responsible for this measured discrepancy and recheck its target and dependent constraints."

    @classmethod
    def constraint_lines(cls, component):
        row = cls.check(component)
        if row is None:
            return []
        name = str(row.get("name", "constraint"))
        unknown = (row.get("satisfied") is None or row.get("status") == "unknown"
                   or component.get("status") == "unknown")
        if unknown:
            return [f"[semantic_alignment] Evidence unavailable (not an observed violation): {name}.",
                    f"Requested target={cls.compact(row.get('requirement', row.get('threshold')))}. "
                    f"Evidence limitation: {row.get('reason') or row.get('message') or 'Required measurement/reference is unavailable.'}",
                    "Required evidence: use an independently supplied reference, site/role assignment or supported geometric measurement where available; report missing inputs explicitly. "
                    "Code literals and claimed labels do not establish the constraint. Do not alter an otherwise valid structure solely to address missing evidence."]
        if bool(row.get("satisfied")):
            return []
        source = row.get("evidence", {})
        source = source if isinstance(source, dict) else {}
        source_check = source.get("source_check") or {}
        note = source_check.get("message") or row.get("reason") or ""
        lines = [f"[semantic_alignment] Observed hard-constraint violation: {name}.",
                 "Localized evidence: " + cls.measurement(row)]
        if source.get("kind") or note:
            lines.append(f"Measurement provenance: {source.get('kind', 'saved diagnostic')}; {note}")
        lines.append("Required repair: " + cls.repair(name))
        return lines

    @classmethod
    def issue_lines(cls, issue):
        metrics = issue.metrics if isinstance(issue.metrics, dict) else {}
        if issue.category in ("alignment", "semantic", "semantic_alignment"):
            rendered = cls.constraint_lines(metrics)
            if rendered:
                return rendered
        if metrics.get("available") is False:
            reason = str(metrics.get("reason") or issue.description)
            resource = any(word in reason.lower() for word in ("out of memory", "outofmemory", "resource", "atom budget"))
            label = "Diagnostic resource failure" if resource else "Diagnostic evidence unavailable"
            return [f"[{issue.category}] {label}: {reason}",
                    "No physical violation is established by this unavailable diagnostic.",
                    "Required evidence: resolve the diagnostic resource/reference limitation and reevaluate; "
                    "do not invent an energy comparison or rebuild the structure solely because this evidence is missing."]
        lines = [f"[{issue.category}] {issue.description}"]
        if issue.suggestion:
            lines.append("Required repair: " + issue.suggestion)
        return lines

    @classmethod
    def invariant_lines(cls, critic_state):
        if critic_state is None:
            return []
        alignment = critic_state.metrics.get("alignment", critic_state.metrics.get("semantic", {}))
        checks = [cls.check(c) for c in alignment.get("components", []) if c.get("status") != "unknown"]
        passed = [row for row in checks if row and row.get("satisfied") is not None
                  and bool(row["satisfied"]) and row.get("status") != "unknown"]
        if not passed:
            return []
        return ["Preserve these currently satisfied hard constraints while applying local repairs; recheck them after every edit:"] + [
            f"- {row['name']}: target={cls.compact(row.get('requirement', row.get('threshold')))}; "
            f"current={cls.compact(row.get('value'))}" for row in passed]


class RuleBasedFeedbackTransformer:
    """Deterministic feedback verbalizer.

    It maps raw execution failures and critic-state violations into concise
    constraints that the generator/refiner can act on, while also emitting
    repair-specific retrieval queries for feedback-aware context expansion.
    """

    def __init__(self, max_issues: int = 6, include_metrics: bool = True):
        self.max_issues = max_issues
        self.include_metrics = include_metrics

    def transform(
        self,
        execution_result: Dict[str, Any],
        critic_state: Optional[CriticState],
        task_description: str,
        iteration: int = 1,
    ) -> FeedbackMessage:
        success = bool(execution_result.get("success", False))
        output = str(execution_result.get("output", "") or "").strip()

        lines: List[str] = []
        repair_queries: List[str] = []
        metadata: Dict[str, Any] = {
            "iteration": iteration,
            "execution_success": success,
        }

        if not success:
            error_summary = self._summarize_execution_error(output)
            lines.append("The previous generated program did not produce a valid structure.")
            lines.append(f"Execution evidence: {error_summary}")
            lines.append(
                "Repair priority: fix the Python/pymatgen failure first, and ensure the script "
                "creates a pymatgen Structure object named 'structure'."
            )
            repair_queries.extend(
                self._queries_for_execution_error(output, task_description)
            )
        elif execution_result.get("partial_execution_error"):
            error_summary = self._summarize_execution_error(output)
            lines.append(
                "The previous generated program created a Structure but failed a final self-check."
            )
            lines.append(f"Execution evidence: {error_summary}")
            lines.append(
                "Repair priority: keep the partial structure pattern, but fix the failed hard "
                "constraint before accepting the candidate."
            )
            repair_queries.extend(
                self._queries_for_execution_error(output, task_description)
            )

        if critic_state is not None:
            state_dict = critic_state.to_dict()
            metadata["critic_state"] = state_dict
            issues = critic_state.sorted_issues()

            if issues:
                if lines:
                    lines.append("")
                lines.append("Physics-aware critic repair constraints:")
                for idx, issue in enumerate(issues[: self.max_issues], 1):
                    lines.append(
                        f"{idx}. [{issue.severity.upper()}:{issue.category}] "
                        f"{issue.description}"
                    )
                    if issue.suggestion:
                        lines.append(f"   Required repair: {issue.suggestion}")
                    if self.include_metrics:
                        metric_hint = self._compact_metrics(issue.metrics)
                        if metric_hint:
                            lines.append(f"   Evidence: {metric_hint}")
                    repair_queries.extend(self._queries_for_issue(issue.category, issue.description))
                    repair_queries.extend(self._queries_for_issue_metrics(issue.metrics))
            elif success:
                metadata["reason"] = "no_execution_or_critic_issues"
                return FeedbackMessage(text="", repair_queries=[], stop=True, metadata=metadata)

            if self.include_metrics:
                score_line = (
                    "Current internal critic scores: "
                    f"I_geom={critic_state.geometric_indicator:.3f}, "
                    f"V_chem={critic_state.chemical_score:.3f}, "
                    f"V_thermo={critic_state.thermodynamic_score:.3f}, "
                    f"V_align={critic_state.alignment_score:.3f}, "
                    f"aggregate={critic_state.overall_score:.3f}."
                )
                if lines:
                    lines.append("")
                lines.append(score_line)

        if lines:
            lines.append("")
            lines.append(
                "For the next attempt, make the smallest code change that resolves the highest "
                "severity issue first; do not introduce a different material, task, or target "
                "constraint unless regeneration is explicitly needed."
            )

        return FeedbackMessage(
            text="\n".join(lines).strip(),
            repair_queries=self._dedupe(repair_queries),
            stop=False,
            metadata=metadata,
        )

    def _summarize_execution_error(self, output: str) -> str:
        if not output:
            return "unknown execution failure"

        lines = [line.strip() for line in output.splitlines() if line.strip()]
        if not lines:
            return output[:500]

        traceback_tail = lines[-4:]
        return " | ".join(traceback_tail)[:700]

    def _queries_for_execution_error(
        self,
        output: str,
        task_description: str = "",
    ) -> List[str]:
        lower = output.lower()
        task_lower = task_description.lower()
        queries = ["pymatgen Structure construction API requirements"]
        is_interface_task = (
            "interface" in task_lower
            or "heterostructure" in task_lower
            or ("film" in task_lower and "substrate" in task_lower)
        )
        is_adsorption_task = (
            "adsorption" in task_lower
            or "adsorbate" in task_lower
            or "adsorb" in task_lower
            or "surface" in task_lower
        )
        is_defect_task = (
            "defect" in task_lower
            or "vacancy" in task_lower
            or "interstitial" in task_lower
            or "substitution" in task_lower
            or "dopant" in task_lower
        )

        if "attributeerror" in lower or "typeerror" in lower:
            queries.append("pymatgen API usage pattern for the failing operation")
        if (
            "syntaxerror" in lower
            or "unterminated string" in lower
            or "never closed" in lower
            or "invalid syntax" in lower
            or "non-ascii python identifier" in lower
            or "ascii-only executable python" in lower
        ):
            queries.append("regenerate concise executable python code without markdown or prose")
        if "importerror" in lower or "modulenotfounderror" in lower:
            queries.append("available installed materials modeling APIs")
        if "indexerror" in lower:
            queries.append("safe site indexing for supercell and defect construction")
        if "valueerror" in lower:
            queries.append("valid lattice coordinates species and fractional coordinate constraints")
        if (
            "len(species)" in lower
            or "len(coords)" in lower
            or "species and coords" in lower
            or "species list and coords list" in lower
        ):
            queries.extend(
                [
                    "pymatgen Structure species coords length must match",
                    "fcc elemental conventional cell four species four coordinates",
                ]
            )
        if "cif" in lower:
            queries.append("pymatgen CIF writing and Structure serialization")
        if (
            "invalid cif" in lower
            or "occupancy" in lower
            or "equivalent site" in lower
            or "periodic duplicate" in lower
        ):
            queries.extend(
                [
                    "avoid CIF roundtrip for generated Structure diagnostics",
                    "adsorption lattice span periodic duplicate repair",
                ]
            )
        if "_output_structure.cif" in lower or "output file" in lower:
            queries.append("ensure generated code assigns final pymatgen Structure named structure")
        if "minimum_distance" in lower:
            queries.append("pymatgen compatibility guard minimum_distance replacement")
        if "interfacebuilder" in lower or "coherentinterfacebuilder" in lower:
            queries.append("pymatgen compatibility guard InterfaceBuilder CoherentInterfaceBuilder")
        if is_interface_task and (
            "no lattice match" in lower
            or "lattice match" in lower
            or "strain" in lower
            or "max_length_mismatch" in lower
        ):
            queries.extend(
                [
                    "interface best small-integer lattice matching and film in-plane strain",
                    "pymatgen compatibility guard interface manual stack fallback",
                ]
            )
        if is_interface_task and (
            "aligned with x-axis" in lower
            or "not aligned" in lower
            or "orientation" in lower
        ):
            queries.extend(
                [
                    "avoid brittle interface slab orientation assertions validate final geometry",
                    "pymatgen compatibility guard interface manual stack fallback",
                ]
            )
        if is_interface_task and (
            "interface distance" in lower
            or "bonding_type" in lower
            or "bonding type" in lower
            or "min_slab_layers" in lower
            or "slab layers" in lower
            or "layer_structure" in lower
        ):
            queries.extend(
                [
                    "interface z gap slab layer count and bonding-distance constraints",
                    "manual heterostructure stack target interface gap and vacuum",
                ]
            )
        if "grainboundarygenerator" in lower or "pymatgen.analysis.grain" in lower:
            queries.append("pymatgen compatibility guard grain boundary construction")
        if "mpr" in lower or "materials project" in lower:
            queries.append("pymatgen compatibility guard local prototype construction without MPRester")
        if "composition mismatch" in lower or "expected" in lower and "got" in lower:
            queries.append("validate generated structure composition and atom counts")
        if "singular matrix" in lower:
            queries.append("avoid singular lattice matrices and degenerate slab vectors")
            if is_interface_task or "make_supercell" in lower:
                queries.extend(
                    [
                        "interface make_supercell full-rank integer scaling matrix",
                        "pymatgen compatibility guard interface manual stack fallback",
                    ]
                )
        if "modify_lattice" in lower:
            queries.extend(
                [
                    "pymatgen lattice replacement by rebuilding Structure with new Lattice",
                    "pymatgen compatibility guard interface manual stack fallback",
                ]
            )
        if (
            "shapes" in lower and "not aligned" in lower
            or "matmul" in lower and "mismatch" in lower
            or "dimension mismatch" in lower
            or "vstack" in lower and "dimensions" in lower and "match" in lower
            or "concatenate" in lower and "dimensions" in lower and "match" in lower
        ):
            queries.extend(
                [
                    "numpy rotation matrix coordinate array shape compatibility",
                    "pymatgen compatibility guard grain boundary construction",
                ]
            )
            if is_interface_task:
                queries.extend(
                    [
                        "interface manual stack construct 3D lattice vectors from xy vectors",
                        "pymatgen compatibility guard interface manual stack fallback",
                    ]
                )
        if "slabgenerator" in lower:
            queries.append("pymatgen compatibility guard SlabGenerator current signature")
        if "symmetrize" in lower or "min_slab_thickness" in lower or "min_vacuum" in lower:
            queries.append("pymatgen compatibility guard SlabGenerator unsupported kwargs")
        if "get_slab_thickness" in lower:
            queries.append("pymatgen slab thickness from Cartesian coordinate z range")
        if "adsorbatesitefinder" in lower:
            queries.append("pymatgen compatibility guard AdsorbateSiteFinder import")
        if "read-only" in lower or "assignment destination is read-only" in lower:
            queries.append("pymatgen compatibility guard lattice matrix copy before edit")
        if "symmetryundeterminederror" in lower or "unable to determine symmetry" in lower:
            queries.append("pymatgen compatibility guard SlabGenerator symmetry fallback")
        if "too few atoms" in lower or "minimum 20" in lower:
            queries.append("pymatgen compatibility guard grain boundary minimum atom count")
        if "grain thickness insufficient" in lower:
            queries.append("pymatgen compatibility guard grain boundary slab thickness")
        if "c-axis" in lower and ("min_thickness" in lower or "minimum thickness" in lower):
            queries.extend(
                [
                    "pymatgen compatibility guard grain boundary slab thickness",
                    "grain boundary c axis lattice length minimum thickness repair",
                ]
            )
        if "host material is not" in lower or "reduced_formula" in lower:
            queries.append("pymatgen compatibility guard vacancy reduced formula validation")
        if "expected 16 pt atoms" in lower or "got 32" in lower:
            queries.append("pymatgen compatibility guard adsorption slab atom count")
        if "execution timeout" in lower or "timeout" in lower:
            queries.append("avoid expensive SlabGenerator search use bounded manual construction")
        if (
            "minimum distance" in lower
            or "min distance" in lower
            or "too close" in lower
            or "overlap" in lower
            or "too small" in lower and "distance" in lower
        ):
            queries.extend(
                [
                    "repair atom overlap by placement height lattice expansion or supercell spacing",
                    "minimum interatomic distance and overlap constraints",
                ]
            )
            if "grain boundary" in task_lower or "grain-boundary" in task_lower:
                queries.extend(
                    [
                        "pymatgen compatibility guard grain boundary construction",
                        "grain boundary interface gap periodic overlap pruning",
                    ]
                )
            if is_interface_task:
                queries.extend(
                    [
                        "interface overlap repair by z gap adjustment and periodic distance check",
                        "manual heterostructure stack target interface gap and vacuum",
                    ]
                )
            if is_adsorption_task:
                queries.extend(
                    [
                        "adsorbate height repair and surface nearest-neighbor distance check",
                        "surface adsorption placement with molecule orientation and vacuum",
                    ]
                )
            if is_defect_task:
                queries.extend(
                    [
                        "defect interstitial dopant local separation and nearest-neighbor relaxation",
                        "defect supercell local strain repair without changing composition",
                    ]
                )

        return queries

    def _queries_for_issue(self, category: str, description: str) -> List[str]:
        category = category.lower()
        description = description.lower()
        queries: List[str] = []

        if category in {"geometric", "physical_validity"}:
            queries.extend(
                [
                    "minimum interatomic distance and overlap constraints",
                    "repair atom overlap by placement height lattice expansion or supercell spacing",
                ]
            )
        if category in {"chemical", "charge_balance"}:
            queries.extend(
                [
                    "charge neutrality oxidation state bond valence repair rules",
                    "defect charge compensation and stoichiometry constraints",
                ]
            )
        if category in {"thermodynamic", "stability"}:
            queries.extend(
                [
                    "relax high strain generated structure before MLIP energy check",
                    "CHGNet MACE M3GNet plausibility checks for generated structures",
                ]
            )
            if "large structure" in description or "skipped" in description:
                queries.append("compact supercell size for generation-time MLIP feedback")
        if category in {"semantic_alignment", "task_alignment"}:
            queries.extend(
                [
                    "task-specific construction template matching prompt constraints",
                    "validate material operation composition supercell and numeric constraints",
                ]
            )
        if category in {"execution", "code_quality"}:
            queries.append("pymatgen code generation common errors and fixes")

        if "adsorb" in description or "surface" in description:
            queries.extend(
                [
                    "surface adsorption site placement height and molecular geometry",
                    "pymatgen compatibility guard adsorption slab atom count",
                ]
            )
        if "interface" in description or "lattice mismatch" in description:
            queries.extend(
                [
                    "heterostructure interface lattice matching and strain constraints",
                    "pymatgen compatibility guard InterfaceBuilder CoherentInterfaceBuilder",
                    "interface z gap slab layer count and bonding-distance constraints",
                    "manual heterostructure stack target interface gap and vacuum",
                ]
            )
        if "grain boundary" in description or "csl" in description:
            queries.extend(
                [
                    "grain boundary CSL rotation matrix construction",
                    "pymatgen compatibility guard grain boundary construction",
                ]
            )
        if "defect" in description or "vacancy" in description or "substitution" in description:
            queries.append("defect engineering supercell defect count and separation rules")
        if "interstitial" in description or "dopant" in description:
            queries.append("defect interstitial dopant local separation and nearest-neighbor relaxation")

        return queries

    def _queries_for_issue_metrics(self, metrics: Dict[str, Any]) -> List[str]:
        queries: List[str] = []
        if isinstance(metrics, dict) and (
            "top_force_sites" in metrics
            or "force_p95" in metrics
            or "max_force" in metrics
            or "score_force" in metrics
        ):
            queries.extend(
                [
                    "repair high-force atoms by increasing local separations and relaxing placement",
                    "adsorbate interface grain-boundary spacing rules for MLIP force reduction",
                    "use MLIP top-force nearest-neighbor diagnostics to target local repairs",
                ]
            )
        if isinstance(metrics, dict) and metrics.get("status") == "skipped_large":
            queries.append("compact supercell size for generation-time MLIP feedback")

        messages = self._collect_metric_messages(metrics)
        if not messages:
            return self._dedupe(queries)

        joined = " ".join(messages).lower()
        if "atom count" in joined or "composition" in joined or "expected" in joined:
            queries.extend(
                [
                    "validate generated structure composition and atom counts",
                    "pymatgen compatibility guard task-input hard constraints",
                ]
            )
        if "adsorb" in joined or "surface" in joined:
            queries.append("pymatgen compatibility guard adsorption slab atom count")
        if "interfacebuilder" in joined or "coherentinterfacebuilder" in joined:
            queries.append("pymatgen compatibility guard InterfaceBuilder CoherentInterfaceBuilder")
        if (
            "interface_distance" in joined
            or "interface distance" in joined
            or "bonding_type" in joined
            or "bonding type" in joined
            or "min_slab_layers" in joined
            or "layer_structure" in joined
            or "no lattice match" in joined
            or "strain" in joined
        ):
            queries.extend(
                [
                    "interface best small-integer lattice matching and film in-plane strain",
                    "interface z gap slab layer count and bonding-distance constraints",
                    "manual heterostructure stack target interface gap and vacuum",
                ]
            )
        if "grainboundarygenerator" in joined or "pymatgen.analysis.grain" in joined:
            queries.append("pymatgen compatibility guard grain boundary construction")
        if "too few atoms" in joined or "minimum 20" in joined:
            queries.append("pymatgen compatibility guard grain boundary minimum atom count")
        if "symmetrize" in joined or "slabgenerator" in joined:
            queries.append("pymatgen compatibility guard SlabGenerator current signature")
        if "reduced_formula" in joined or "host material is not" in joined:
            queries.append("pymatgen compatibility guard vacancy reduced formula validation")
        return self._dedupe(queries)

    def _collect_metric_messages(self, value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        if isinstance(value, dict):
            messages: List[str] = []
            for item in value.values():
                messages.extend(self._collect_metric_messages(item))
            return messages
        if isinstance(value, (list, tuple, set)):
            messages = []
            for item in value:
                messages.extend(self._collect_metric_messages(item))
            return messages
        return []

    def _compact_metrics(self, metrics: Dict[str, Any]) -> str:
        if not metrics:
            return ""

        if "top_force_sites" in metrics or "max_force" in metrics or "force_p95" in metrics:
            force_items = []
            for key in ("max_force", "force_p95", "mean_force", "score_force"):
                if key in metrics:
                    value = metrics[key]
                    if isinstance(value, (int, float)):
                        force_items.append(f"{key}={float(value):.3f}")
                    else:
                        force_items.append(f"{key}={value}")

            top_sites = metrics.get("top_force_sites") or []
            if top_sites:
                site_bits = []
                for site in top_sites[:3]:
                    nearest = site.get("nearest_neighbor") or {}
                    if nearest:
                        site_bits.append(
                            "#{idx} {species} F={force} near #{nidx} {nspecies} d={dist}A".format(
                                idx=site.get("index"),
                                species=site.get("species"),
                                force=site.get("force"),
                                nidx=nearest.get("index"),
                                nspecies=nearest.get("species"),
                                dist=nearest.get("distance"),
                            )
                        )
                    else:
                        site_bits.append(
                            "#{idx} {species} F={force}".format(
                                idx=site.get("index"),
                                species=site.get("species"),
                                force=site.get("force"),
                            )
                        )
                force_items.append("top_force_sites=[" + "; ".join(site_bits) + "]")

            if force_items:
                return "; ".join(force_items[:5])

        simple_items = []
        for key, value in metrics.items():
            if isinstance(value, (str, int, float, bool)):
                simple_items.append(f"{key}={value}")
            elif isinstance(value, list) and value:
                simple_items.append(f"{key}={value[:2]}")
            elif isinstance(value, dict):
                for subkey, subvalue in value.items():
                    if isinstance(subvalue, (str, int, float, bool)):
                        simple_items.append(f"{key}.{subkey}={subvalue}")
                    if len(simple_items) >= 4:
                        break
            if len(simple_items) >= 4:
                break

        return "; ".join(simple_items)[:500]

    @staticmethod
    def _dedupe(values: List[str]) -> List[str]:
        seen = set()
        result = []
        for value in values:
            normalized = value.strip()
            if not normalized:
                continue
            key = normalized.lower()
            if key in seen:
                continue
            seen.add(key)
            result.append(normalized)
        return result


class LLMFeedbackTransformer(RuleBasedFeedbackTransformer):
    """LLM-backed feedback transformer with deterministic fallback.

    This is intentionally a thin adapter: the deterministic transformer still
    gathers and prioritizes evidence, then the LLM can compress it into a repair
    message. If parsing or generation fails, the rule-based feedback is returned.
    """

    SYSTEM_PROMPT = """You transform execution and physics-critic evidence into repair feedback.
Return ONLY JSON with keys:
{
  "feedback": "concise actionable repair instructions",
  "repair_queries": ["short retrieval query", "..."],
  "stop": false
}

Rules:
- Do not invent new task requirements.
- Prioritize execution failures, atom overlaps, task mismatch, charge issues, then soft warnings.
- Keep feedback concrete enough for code regeneration.
- Set stop=true only when there are no execution or critic issues."""

    def __init__(self, llm_client, max_issues: int = 6, include_metrics: bool = True):
        super().__init__(max_issues=max_issues, include_metrics=include_metrics)
        self.llm_client = llm_client

    def transform(
        self,
        execution_result: Dict[str, Any],
        critic_state: Optional[CriticState],
        task_description: str,
        iteration: int = 1,
    ) -> FeedbackMessage:
        rule_feedback = super().transform(
            execution_result=execution_result,
            critic_state=critic_state,
            task_description=task_description,
            iteration=iteration,
        )
        if rule_feedback.stop or not rule_feedback.text:
            return rule_feedback

        prompt = {
            "task": task_description,
            "iteration": iteration,
            "execution_result": {
                "success": execution_result.get("success", False),
                "output": str(execution_result.get("output", ""))[:1500],
            },
            "critic_state": critic_state.to_dict() if critic_state else None,
            "rule_based_feedback": rule_feedback.to_dict(),
        }

        try:
            response, tokens = self.llm_client.generate(
                prompt=json.dumps(prompt, indent=2),
                system_prompt=self.SYSTEM_PROMPT,
                temperature=0.0,
            )
            payload = json.loads(self._extract_json_object(response))
            feedback = str(payload.get("feedback", "")).strip()
            if not feedback:
                return rule_feedback
            queries = payload.get("repair_queries", rule_feedback.repair_queries)
            if not isinstance(queries, list):
                queries = rule_feedback.repair_queries
            stop = bool(payload.get("stop", False))
            metadata = {
                **rule_feedback.metadata,
                "llm_transformer_tokens": tokens,
                "rule_based_feedback": rule_feedback.to_dict(),
            }
            return FeedbackMessage(
                text=feedback,
                repair_queries=self._dedupe([str(query) for query in queries]),
                stop=stop,
                metadata=metadata,
            )
        except Exception as exc:
            rule_feedback.metadata["llm_transformer_error"] = str(exc)
            return rule_feedback

    @staticmethod
    def _extract_json_object(text: str) -> str:
        stripped = text.strip()
        if stripped.startswith("```"):
            stripped = re.sub(r"^```(?:json)?", "", stripped).strip()
            stripped = re.sub(r"```$", "", stripped).strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            return stripped

        match = re.search(r"\{.*\}", stripped, re.DOTALL)
        if not match:
            raise ValueError("No JSON object found in LLM feedback transformer output")
        return match.group(0)
