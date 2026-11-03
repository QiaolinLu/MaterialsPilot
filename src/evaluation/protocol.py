"""Paper Eq. 23–28 benchmark metrics, independent of generation success flags.

One fixed prompt-derived hard set feeds flat CSR and conjunctive TSR. All
attempts remain in aggregate denominators. Unknown evidence is retained and
reported; primary values are conservative lower bounds, never invented passes.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import math
from typing import Any, Dict, Iterable, List, Optional
from pymatgen.core import Structure

from .paper_constraints import collect_task_constraints, infer_task_type
from .validity import structural_validity_checks, DISTANCES
from .paper_serialization import optional_bool, json_mapping
from .mlip_resources import MLIPResourceError


def _unit(value, default=0.0):
    try:
        number=float(value)
        return max(0., min(1., number)) if math.isfinite(number) else default
    except (ValueError, TypeError):
        return default


@dataclass
class ConstraintCheck:
    name: str
    satisfied: Optional[bool]
    value: Any = None
    threshold: Any = None
    message: str = ""
    group: str = "task"
    hard: bool = True
    score: Optional[float] = None
    evidence: Dict[str, Any] = field(default_factory=dict)
    requirement: Any = None
    evidence_available: bool = True
    status: str = ""

    def __post_init__(self):
        self.satisfied = optional_bool(self.satisfied)
        self.hard = optional_bool(self.hard)
        self.evidence_available = self.satisfied is not None
        self.status = "unknown" if self.satisfied is None else "pass" if self.satisfied else "fail"

    @classmethod
    def from_row(cls, row):
        fields = cls.__dataclass_fields__
        values = {k:v for k,v in row.items() if k in fields}
        values["message"] = row.get("message", row.get("reason", ""))
        values["evidence_available"] = row.get("satisfied") is not None
        values["status"] = "unknown" if row.get("satisfied") is None else "pass" if row["satisfied"] else "fail"
        return cls(**values)

    def to_dict(self):
        return json_mapping(asdict(self))

    def normalized_score(self):
        satisfied = optional_bool(self.satisfied)
        if satisfied is True:
            return 1.0
        if satisfied is None:
            return 0.0
        return _unit(self.score)


@dataclass
class ProtocolResult:
    structural_valid: bool
    task_satisfied: bool
    constraint_satisfaction: float
    avg_score: Optional[float] = None
    quality_components: Dict[str, float] = field(default_factory=dict)
    structural_checks: List[ConstraintCheck] = field(default_factory=list)
    task_constraints: List[ConstraintCheck] = field(default_factory=list)
    quality_availability: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.structural_valid = optional_bool(self.structural_valid)
        self.task_satisfied = optional_bool(self.task_satisfied)

    def to_dict(self):
        # Normalize even if a downstream diagnostic mutated a dataclass after
        # construction; identity comparisons must never see np.bool_.
        for check in self.structural_checks + self.task_constraints:
            check.__post_init__()
        hard = [c for c in self.task_constraints if c.hard]
        unknown = [c.name for c in hard if c.satisfied is None]
        structural_unknown = [c.name for c in self.structural_checks if c.satisfied is None]
        possible_valid = bool(self.structural_checks) and all(c.satisfied is not False for c in self.structural_checks)
        csr_upper = (sum(1. if c.satisfied is None else c.normalized_score() for c in hard)/len(hard)
                     if possible_valid and hard else 0.)
        tsr_upper = bool(possible_valid and hard and all(c.satisfied is not False for c in hard))
        return json_mapping({
            "structural_valid": self.structural_valid,
            "task_satisfied": self.task_satisfied,
            "constraint_satisfaction": self.constraint_satisfaction,
            "avg_score": self.avg_score,
            "quality_components": self.quality_components,
            "quality_availability": self.quality_availability,
            "structural_checks": [asdict(c) for c in self.structural_checks],
            "task_constraints": [asdict(c) for c in self.task_constraints],
            "unknown_constraints": unknown,
            "unknown_structural_checks": structural_unknown,
            "constraint_coverage": {"known":len(hard)-len(unknown), "total":len(hard),
                                    "fraction":(len(hard)-len(unknown))/len(hard) if hard else 0.},
            "score_interpretation": "conservative_lower_bound" if unknown or structural_unknown else "evaluated",
            "avg_score_interpretation": "evaluated" if self.quality_availability.get("complete") else "conservative_lower_bound",
            "sr_upper": float(possible_valid), "tsr_upper": float(tsr_upper), "csr_upper": csr_upper,
            "protocol_version": "paper-aligned-1.0.2",
        })


def evaluate_result_protocol(result, task, output_dir, enable_mlip=False, mlip_model="chgnet", mlip_device="auto", compute_quality=True):
    result_dict=_as_result_dict(result)
    structure=_load_structure_from_result(result_dict, output_dir)
    family=result_dict.get("task_type") or infer_task_type(task)
    if family not in DISTANCES:
        family=infer_task_type(task)
    return evaluate_structure_protocol(structure,task,family,
        deterministic_result=_deterministic_result_for_quality(result_dict),result_metadata=result_dict,
        enable_mlip=enable_mlip,mlip_model=mlip_model,mlip_device=mlip_device,compute_quality=compute_quality).to_dict()


def evaluate_structure_protocol(structure, task, task_type, deterministic_result=None, result_metadata=None,
                                enable_mlip=False, mlip_model="chgnet", mlip_device="auto", compute_quality=True):
    structural = [ConstraintCheck.from_row(c) for c in structural_validity_checks(structure,task_type)]
    valid = bool(structural) and all(c.satisfied is True for c in structural)
    # Required constraints are collected even for invalid/missing structures.
    # Measurements are safely suppressed by the pure collector in that case.
    constraints=[ConstraintCheck.from_row(c) for c in collect_task_constraints(structure,task,task_type,result_metadata)]
    hard=[c for c in constraints if c.hard]
    csr = sum(c.normalized_score() for c in hard)/len(hard) if valid and hard else 0.
    satisfied = bool(valid and hard and all(c.satisfied is True for c in hard))
    components,availability={},{}
    avg=0.0
    if valid and compute_quality:
        components,availability=_quality_components(structure,task,task_type,deterministic_result,
                                                      enable_mlip,mlip_model,mlip_device)
        values=[_unit(components.get(k)) for k in ("alignment","chemical","thermodynamic")]
        avg=math.prod(values)**(1./3.)
    elif not compute_quality:
        availability={"complete":False,"unknown_components":["alignment","chemical","thermodynamic"],
                      "reason":"Quality computation explicitly disabled", "score_interpretation":"conservative_lower_bound"}
    else:
        availability={"complete":False,"reason":"Invalid or unverified structural gate; Eq. 28 contributes zero"}
    return ProtocolResult(valid,satisfied,csr,avg,components,structural,constraints,availability)


def summarize_protocol_results(results: Iterable[Any]) -> Dict[str, Any]:
    protocols=[]
    missing=0
    for result in results:
        data=_as_result_dict(result)
        protocol=(data.get("metadata") or {}).get("evaluation_protocol")
        # Accept direct protocol dictionaries for offline consumers.
        if protocol is None and "structural_valid" in data:
            protocol=data
        if not isinstance(protocol,dict) or not protocol:
            missing+=1
            protocol={}
        protocols.append(json_mapping(protocol))
    total=len(protocols)
    valid=sum(p.get("structural_valid") is True for p in protocols)
    success=sum(p.get("structural_valid") is True and p.get("task_satisfied") is True for p in protocols)
    csr=sum(_unit(p.get("constraint_satisfaction")) if p.get("structural_valid") is True else 0. for p in protocols)
    avg=sum(_unit(p.get("avg_score")) if p.get("structural_valid") is True else 0. for p in protocols)
    denom=max(total,1)
    unknown=sum(bool(p.get("unknown_constraints") or p.get("unknown_structural_checks")) for p in protocols)+missing
    complete=sum(p.get("structural_valid") is True and (p.get("quality_availability") or {}).get("complete") is True for p in protocols)
    return {"total":total,"valid":valid,"task_satisfied":success,"sr":valid/denom,
            "tsr":success/denom,"csr":csr/denom,"avg_score":avg/denom,
            "missing_protocol":missing,"samples_with_unknown_evidence":unknown,
            "quality_complete":complete,"quality_complete_fraction":complete/denom,
            "score_interpretation":"conservative_lower_bound" if unknown or complete<valid else "evaluated",
            "sr_upper":sum(_unit(p.get("sr_upper",float(bool(p.get("structural_valid"))))) if p else 1. for p in protocols)/denom,
            "tsr_upper":sum(_unit(p.get("tsr_upper",float(bool(p.get("task_satisfied"))))) if p else 1. for p in protocols)/denom,
            "csr_upper":sum(_unit(p.get("csr_upper",p.get("constraint_satisfaction",0.) if p.get("structural_valid") else 0.)) if p else 1. for p in protocols)/denom}


def _as_result_dict(result):
    if isinstance(result,dict):return result
    if hasattr(result,"to_dict"):return result.to_dict()
    if hasattr(result,"__dict__"):return dict(result.__dict__)
    return {}


def _load_structure_from_result(result,output_dir):
    value=result.get("output_cif_path")
    if not value:return None
    path=Path(value)
    if not path.is_absolute():path=Path(output_dir)/path
    try:return Structure.from_file(str(path))
    except Exception:return None


def _deterministic_result_for_quality(result):
    return {k:result.get(k,default) for k,default in {"score":0.,"success":False,"errors":[],"warnings":[],"generated_code":""}.items()}


def _quality_components(structure,task,task_type,deterministic_result,enable_mlip,mlip_model,mlip_device):
    try:
        from src.agents.physics_aware_critic import PhysicsAwareCritic
        full_task=dict(task)
        full_task["task_type"]=task_type
        state=PhysicsAwareCritic(enable_thermo=enable_mlip,mlip_model=mlip_model,mlip_device=mlip_device).evaluate(
            structure=structure,task_description=task.get("description",""),task_input=full_task,
            deterministic_evaluation=deterministic_result)
        components={"alignment":_unit(state.alignment_score),"chemical":_unit(state.chemical_score),
                    "thermodynamic":_unit(state.thermodynamic_score)}
        availability=dict(state.metrics.get("quality_availability") or {})
        if not availability:
            availability={"complete":False,"reason":"Critic did not provide quality availability"}
        return components,availability
    except MLIPResourceError:
        raise
    except MemoryError as exc:
        raise MLIPResourceError(f"Post-hoc quality evaluation ran out of memory: {exc}") from exc
    except Exception as exc:
        return {"alignment":0.,"chemical":0.,"thermodynamic":0.}, {
            "complete":False,"unknown_components":["alignment","chemical","thermodynamic"],
            "reason":f"{type(exc).__name__}: {exc}","score_interpretation":"conservative_lower_bound"}


# Private compatibility names retained for existing local diagnostics.
_infer_task_type=infer_task_type

def _structural_min_distance(task,task_type):
    return DISTANCES.get(task_type,1.0)

def _structural_validity_checks(structure,task,task_type):
    return [ConstraintCheck.from_row(c) for c in structural_validity_checks(structure,task_type)]

def _task_constraint_checks(structure,task,task_type,result_metadata=None):
    return [ConstraintCheck.from_row(c) for c in collect_task_constraints(structure,task,task_type,result_metadata)]

def _group_normalized_constraint_satisfaction(checks):
    hard=[c for c in checks if c.hard]
    return sum(c.normalized_score() for c in hard)/len(hard) if hard else 0.
