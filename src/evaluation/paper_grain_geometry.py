"""Local cubic orientation evidence for Table 5 grain-boundary constraints.

Fits ideal nearest-neighbor shells on both sides of the a/b boundary plane.
It supports fcc, bcc and diamond. The fixed angular tolerances and CSL bounded
proxy are implementation conventions, not numbers specified by the paper.
Unsupported or unidentifiable geometry is unknown; code is never evidence.
"""
from __future__ import annotations

import itertools
import math
import re
import numpy as np
from pymatgen.core import Composition, Element
from scipy.optimize import linear_sum_assignment
from scipy.spatial.transform import Rotation
from .paper_serialization import optional_bool

FIT_TOLERANCE_DEG = 8.0
MATCH_TOLERANCE_DEG = 5.0
SYMMETRIES = Rotation.create_group("O").as_matrix()


def _template(kind):
    if kind == "fcc":
        vectors = [v for v in itertools.product((-1, 0, 1), repeat=3) if sum(x*x for x in v) == 2]
    elif kind == "bcc":
        vectors = list(itertools.product((-1, 1), repeat=3))
    elif kind == "diamond":
        vectors = [v for v in itertools.product((-1, 1), repeat=3) if math.prod(v) == 1]
    else:
        return None
    vectors = np.asarray(vectors, float)
    return vectors / np.linalg.norm(vectors, axis=1)[:, None]


def _basis(first, second):
    one = first / np.linalg.norm(first)
    two = second - np.dot(second, one) * one
    two /= np.linalg.norm(two)
    return np.column_stack((one, two, np.cross(one, two)))


def fit_local_orientation(vectors, kind):
    """Return (orientation mapping ideal crystal to Cartesian, residual deg)."""
    ideal = _template(kind)
    if ideal is None or len(vectors) < len(ideal):
        return None
    vectors = np.asarray(vectors, float)
    lengths = np.linalg.norm(vectors, axis=1)
    chosen = np.argsort(lengths)[:len(ideal)]
    lengths = lengths[chosen]
    # Avoid fitting a chemically unrelated broad shell to a cubic template.
    if lengths.min() <= 1e-8 or lengths.max()/lengths.min() > 1.3:
        return None
    observed = vectors[chosen] / lengths[:, None]
    target_dot = float(ideal[0] @ ideal[1])
    ideal_basis = _basis(ideal[0], ideal[1])
    best = None
    for i,j in itertools.permutations(range(len(observed)), 2):
        if abs(float(observed[i] @ observed[j])-target_dot) > 0.2:
            continue
        rotation = _basis(observed[i], observed[j]) @ ideal_basis.T
        fitted = ideal @ rotation.T
        cost = np.clip(1 - fitted @ observed.T, 0, 2)
        rows,cols = linear_sum_assignment(cost)
        # Refine by a proper Kabsch fit after correspondence is assigned.
        fitted_rotation,_ = Rotation.align_vectors(observed[cols], ideal[rows])
        rotation = fitted_rotation.as_matrix()
        cosine = np.sum((ideal[rows] @ rotation.T)*observed[cols],axis=1)
        error = float(np.sqrt(np.mean(np.degrees(np.arccos(np.clip(cosine,-1,1)))**2)))
        if best is None or error < best[1]:
            best = (rotation,error)
        if error < 1e-5:
            break
    return best if best is not None and best[1] <= FIT_TOLERANCE_DEG else None


def _neighbor_vectors(structure, index, number):
    frac = np.asarray(structure.frac_coords, float)
    delta = frac - frac[index]
    delta -= np.round(delta)
    shifts = np.asarray(list(itertools.product((-1,0,1), repeat=3)))
    vectors = ((delta[None,:,:]+shifts[:,None,:]) @ structure.lattice.matrix).reshape(-1,3)
    lengths = np.linalg.norm(vectors, axis=1)
    order = np.argsort(lengths)
    order = order[lengths[order] > 1e-6][:number]
    return vectors[order]


def _equivalent_angle(left, right):
    matrices = np.einsum("aij,jk->aik", SYMMETRIES, left.T @ right)
    traces = np.trace(matrices,axis1=1,axis2=2)
    return float(np.degrees(np.arccos(np.clip((max(traces)-1)/2,-1,1))))


def _side_orientation(structure, indices, kind):
    ideal = _template(kind)
    if ideal is None or len(indices) < len(ideal)+1:
        return None
    points=np.asarray(structure.cart_coords)[indices]
    center=np.median(points,axis=0)
    candidates=[indices[i] for i in np.argsort(np.linalg.norm(points-center,axis=1))[:24]]
    fits=[]
    for i in candidates:
        fit=fit_local_orientation(_neighbor_vectors(structure,i,len(ideal)),kind)
        if fit is not None:
            fits.append((fit[0],fit[1],i))
    if len(fits)<3:
        return None
    clusters=[]
    for r,error,index in fits:
        matching=[f for f in fits if _equivalent_angle(r,f[0]) <= MATCH_TOLERANCE_DEG]
        clusters.append((len(matching),-error,r,error,index))
    count,_,orientation,error,index=max(clusters,key=lambda row:row[:2])
    if count < 3 or count < len(fits)/2:
        return None
    return orientation,dict(fit_residual_degrees=error,consistent_atoms=count,
                           fitted_atoms=len(fits),sampled_atoms=len(candidates),representative_atom=index)


def _numbers(value):
    if isinstance(value,(list,tuple)):
        return np.array(value,float)
    text=str(value or "").strip("()[] ")
    parts=re.findall(r"-?\d+",text)
    if len(parts)==1 and len(parts[0])==3:
        parts=list(parts[0])
    return np.array([float(x) for x in parts],float) if len(parts)==3 else None


def _sigma_from_target(axis, angle):
    """Primitive integer quaternion Sigma for cubic CSL, if identifiable."""
    q=Rotation.from_rotvec(axis/np.linalg.norm(axis)*np.radians(angle)).as_quat()
    candidates=[]
    for scale in range(1,129):
        integers=np.rint(q*scale).astype(int)
        divisor=math.gcd(*[int(abs(x)) for x in integers])
        if not divisor:
            continue
        integers//=divisor
        error=float(np.linalg.norm(integers/np.linalg.norm(integers)-q))
        if error > 1e-3:
            continue
        sigma=int(integers@integers)
        while sigma and sigma%2==0:
            sigma//=2
        candidates.append((sigma,error))
    return min(candidates)[0] if candidates else None


def _row(name,passed,value,threshold,score=None,reason=""):
    passed=optional_bool(passed)
    return dict(name=name,satisfied=passed,value=value,threshold=threshold,
                hard=True,score=1. if passed is True else score,group="grain_geometry",message=reason)


def _material_identity(structure, material):
    """Fraction assignable to the requested stoichiometry, including compounds.

    For a single element this is the original elemental fraction exactly.
    For AB, only matched A:B formula units count toward the same 90% gate;
    extra species or off-stoichiometric excess do not count as host material.
    """
    threshold = {"material": material, "minimum_fraction": .9}
    try:
        formula = Composition(material, strict=True).get_el_amt_dict()
        if not formula or any(not Element.is_valid_symbol(e) or not math.isfinite(n) or n <= 0
                              for e, n in formula.items()):
            raise ValueError("Expected a nonempty formula with legal elements and positive amounts")
    except (ValueError, TypeError, KeyError) as exc:
        return _row("gb_material", None, None, threshold, None,
                    f"Requested material formula cannot be interpreted: {exc}")
    counts = structure.composition.get_el_amt_dict()
    units = min(counts.get(element, 0.) / amount for element, amount in formula.items())
    fraction = units * sum(formula.values()) / max(sum(counts.values()), 1.)
    return _row("gb_material", fraction >= .9, fraction, threshold, min(1., fraction/.9),
                "Fraction of atoms assignable to complete requested stoichiometric formula units")


def grain_boundary_constraints(structure,task):
    inp=task.get("input",task); expected=task.get("expected_properties") or {}
    material=inp.get("material"); kind=str(expected.get("structure_type",inp.get("structure",""))).lower()
    if kind in {"diamond_cubic","diamond-cubic"}:kind="diamond"
    rows=[_material_identity(structure, material)]
    matrix=np.asarray(structure.lattice.matrix)
    normal=np.cross(matrix[0],matrix[1]);normal/=np.linalg.norm(normal)
    thickness=abs(float(matrix[2]@normal))
    minimum=(inp.get("constraints") or {}).get("min_thickness")
    if minimum is not None:
        rows.append(_row("min_thickness",thickness>=minimum,thickness,minimum,min(1.,thickness/max(minimum,1e-6))))
    heights=np.asarray(structure.frac_coords)[:,2]%1
    split=float(np.median(heights))
    left=[i for i,h in enumerate(heights) if h<split]
    right=[i for i,h in enumerate(heights) if h>=split]
    first=_side_orientation(structure,left,kind)
    second=_side_orientation(structure,right,kind)
    if first is None or second is None:
        reason="Two consistent local cubic grain orientations could not be fitted (unsupported lattice or insufficient undistorted neighbors)."
        for name in ("gb_structure_type","bicrystal_structure","gb_plane","rotation_axis","rotation_angle","csl_relation","boundary_type"):
            rows.append(_row(name,None,None,{"fit_tolerance_degrees":FIT_TOLERANCE_DEG},None,reason))
        return rows
    o1,e1=first;o2,e2=second
    metadata={"left":e1,"right":e2,"lattice_type":kind,"boundary_normal_cartesian":normal.tolist()}
    rows.append(_row("gb_structure_type",True,metadata,kind))
    observed_angle=_equivalent_angle(o1,o2)
    rows.append(_row("bicrystal_structure",observed_angle>1.,{**metadata,"minimum_misorientation_degrees":observed_angle},
                      {"minimum_distinct_orientation_degrees":1.},min(1.,observed_angle)))
    axis=_numbers(inp.get("rotation_axis"));plane=_numbers(inp.get("gb_plane"))
    angle=expected.get("rotation_angle",inp.get("rotation_angle"))
    if axis is None or angle is None or np.linalg.norm(axis)==0:
        rows.extend(_row(name,None,None,None,None,"Requested angle and axis are needed to resolve the cubic symmetry branch")
                    for name in ("rotation_axis","rotation_angle","gb_plane","csl_relation","boundary_type"))
        return rows
    target=Rotation.from_rotvec(axis/np.linalg.norm(axis)*np.radians(float(angle))).as_matrix()
    candidates=[]
    for a,b in ((o1,o2),(o2,o1)):
        for s1 in SYMMETRIES:
            aa=a@s1
            relatives=np.einsum("ij,ajk->aik",aa.T@b,SYMMETRIES)
            traces=np.einsum("ij,aji->a",target.T,relatives)
            errors=np.degrees(np.arccos(np.clip((traces-1)/2,-1,1)))
            plane_error=0.
            if plane is not None and np.linalg.norm(plane)>0:
                cosine=abs(float((aa.T@normal)@(plane/np.linalg.norm(plane))))
                plane_error=float(np.degrees(np.arccos(np.clip(cosine,-1,1))))
            for index,error in enumerate(errors):
                candidates.append((max(float(error),plane_error),float(error),plane_error,relatives[index],aa))
    _,rotation_error,plane_error,best,aa=min(candidates,key=lambda x:x[0])
    rotvec=Rotation.from_matrix(best).as_rotvec();observed=float(np.degrees(np.linalg.norm(rotvec)))
    axis_error=90. if np.linalg.norm(rotvec)<1e-6 else float(np.degrees(np.arccos(np.clip(abs(float((rotvec/np.linalg.norm(rotvec))@(axis/np.linalg.norm(axis)))),-1,1))))
    angle_error=abs(observed-float(angle))
    for name,error,value in (("rotation_angle",angle_error,observed),("rotation_axis",axis_error,(rotvec/np.linalg.norm(rotvec)).tolist() if np.linalg.norm(rotvec)>1e-6 else None),
                             ("gb_plane",plane_error,(aa.T@normal).tolist())):
        rows.append(_row(name,error<=MATCH_TOLERANCE_DEG,{"measured":value,"error_degrees":error,**metadata},
                          {"tolerance_degrees":MATCH_TOLERANCE_DEG},math.exp(-max(0.,error-MATCH_TOLERANCE_DEG)/MATCH_TOLERANCE_DEG)))
    sigma=_sigma_from_target(axis,float(angle))
    if inp.get("sigma") is not None:
        if sigma is None:
            rows.append(_row("csl_relation",None,None,inp["sigma"],None,"Requested rotation has no identified small integer quaternion"))
        else:
            passed=sigma==int(inp["sigma"]) and rotation_error<=MATCH_TOLERANCE_DEG
            score=0. if sigma!=int(inp["sigma"]) else math.exp(-max(0.,rotation_error-MATCH_TOLERANCE_DEG)/MATCH_TOLERANCE_DEG)
            rows.append(_row("csl_relation",passed,{"integer_quaternion_sigma":sigma,"measured_rotation_error_degrees":rotation_error},
                              {"sigma":inp["sigma"],"tolerance_degrees":MATCH_TOLERANCE_DEG},score,
                              "Fixed bounded cubic CSL orientation proxy; not an exact coincidence-site enumeration"))
    boundary=str(inp.get("gb_type","")).lower()
    axis_cart=aa@(axis/np.linalg.norm(axis));cosine=abs(float(axis_cart@normal))
    if boundary in {"tilt","symmetric_tilt","asymmetric_tilt"}:
        error=abs(90.-float(np.degrees(np.arccos(np.clip(cosine,-1,1)))))
    elif boundary in {"twist"}:
        error=float(np.degrees(np.arccos(np.clip(cosine,-1,1))))
    elif boundary in {"twin","coherent_twin"}:
        error=rotation_error if sigma==3 else 180.
    else:
        error=None
    rows.append(_row("boundary_type",None if error is None else error<=MATCH_TOLERANCE_DEG,
                      {"type":boundary,"error_degrees":error},{"tolerance_degrees":MATCH_TOLERANCE_DEG},
                      None if error is None else math.exp(-max(0.,error-MATCH_TOLERANCE_DEG)/MATCH_TOLERANCE_DEG),
                      "Unsupported boundary subtype" if error is None else "Measured rotation-axis/boundary-normal relationship"))
    return rows
