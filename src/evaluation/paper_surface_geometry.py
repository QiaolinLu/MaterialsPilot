"""Periodic surface measurements. Constants are implementation conventions.

No construction-code evidence is used. The a/b plane defines the interface
normal; the largest empty periodic substrate gap unwraps the occupied slab.
"""
import itertools
import math
import numpy as np

LAYER_TOLERANCE = 0.5
SITE_TOLERANCE = 0.35


def normal_coordinates(structure, indices):
    matrix = np.asarray(structure.lattice.matrix, float)
    normal = np.cross(matrix[0], matrix[1])
    normal /= np.linalg.norm(normal)
    period = abs(float(matrix[2] @ normal))
    if period <= 0:
        raise ValueError("Undefined periodic surface normal")
    frac = np.asarray(structure.frac_coords)[:, 2] % 1
    values = np.sort(frac[list(indices)])
    if not len(values):
        raise ValueError("No assigned substrate atoms")
    gaps = np.diff(np.r_[values, values[0] + 1])
    start = values[(int(np.argmax(gaps)) + 1) % len(values)]
    heights = ((frac - start) % 1) * period
    heights[np.isclose(heights, period, atol=1e-7)] = 0
    return heights, normal, period, float(max(gaps) * period)


def layer_groups(heights, indices, tolerance=LAYER_TOLERANCE):
    layers = []
    for index in sorted(indices, key=lambda i: heights[i]):
        if not layers or heights[index] - np.mean([heights[i] for i in layers[-1]]) > tolerance:
            layers.append([index])
        else:
            layers[-1].append(index)
    return layers


def min_cross_distance(structure, left, right):
    if not left or not right:
        return None
    best = math.inf
    # Bounded temporary arrays, no size-based successful skip.
    for start in range(0, len(left), 64):
        distances = structure.lattice.get_all_distances(
            structure.frac_coords[left[start:start+64]], structure.frac_coords[right])
        if not np.isfinite(distances).all():
            raise ValueError("Nonfinite cross-interface distances")
        best = min(best, float(np.min(distances)))
    return best


def range_score(value, lower, upper, tolerance=0.0):
    lower, upper = max(0.0, lower - tolerance), upper + tolerance
    if lower <= value <= upper:
        return True, 1.0
    deficit = lower - value if value < lower else value - upper
    return False, float(math.exp(-deficit / max(upper-lower, 1e-6)))


def _lateral_images(structure, indices, binding, normal):
    a, b = np.asarray(structure.lattice.matrix)[:2]
    shifts = [i*a + j*b for i, j in itertools.product(range(-2, 3), repeat=2)]
    vectors = np.array([structure[i].coords + shift - binding for i in indices for shift in shifts])
    vectors -= (vectors @ normal)[:, None] * normal
    # Coordinates outside the central unit cell are first wrapped by callers.
    order = np.argsort(np.linalg.norm(vectors, axis=1))
    unique = []
    for vector in vectors[order]:
        if not any(np.linalg.norm(vector-other) < 1e-6 for other in unique):
            unique.append(vector)
        if len(unique) >= 12:
            break
    return np.asarray(unique)


def adsorption_site(structure, substrate, adsorbate, requested):
    """Resolve common top/bridge/hollow sites; distinguish fcc/hcp by stacking.

    Step-edge labels have no accepted generic classifier and remain unknown.
    """
    requested = str(requested).lower()
    if requested not in {"top", "ontop", "bridge", "hollow", "fcc", "hcp"}:
        return None, None, {"reason": "Unsupported site classifier", "site": requested}
    heights, normal, _, _ = normal_coordinates(structure, substrate)
    layers = layer_groups(heights, substrate)
    if not adsorbate or not layers:
        return False, 0.0, {"reason": "Missing adsorbate or substrate"}
    binding = structure[min(adsorbate, key=lambda i: heights[i])].coords
    top = _lateral_images(structure, layers[-1], binding, normal)
    if not len(top):
        return None, None, {"reason": "Missing top-layer geometry"}
    distances = np.linalg.norm(top, axis=1)
    mismatch = float(distances[0])
    if requested == "bridge":
        pair_lengths = [np.linalg.norm(a-b) for a, b in itertools.combinations(top, 2)]
        nearest = min((d for d in pair_lengths if d > 1e-6), default=math.inf)
        mismatch = min((np.linalg.norm((a+b)/2) for a, b in itertools.combinations(top, 2)
                        if abs(np.linalg.norm(a-b)-nearest) <= SITE_TOLERANCE), default=math.inf)
    elif requested in {"hollow", "fcc", "hcp"}:
        if len(top) < 3:
            return None, None, {"reason": "Fewer than three top-layer images"}
        # Require a triangle around the binding point, not simply off-top.
        tri = top[:3]
        area = np.linalg.norm(np.cross(tri[1]-tri[0], tri[2]-tri[0]))
        mismatch = max(float(np.ptp(distances[:3])), float(np.linalg.norm(np.mean(tri, axis=0))))
        if area < 1e-6:
            mismatch = math.inf
        if requested in {"fcc", "hcp"} and mismatch <= SITE_TOLERANCE:
            if len(layers) < 3:
                return None, None, {"reason": "fcc/hcp classification requires three resolved substrate layers"}
            second = _lateral_images(structure, layers[-2], binding, normal)
            third = _lateral_images(structure, layers[-3], binding, normal)
            second_distance = float(np.linalg.norm(second[0]))
            third_distance = float(np.linalg.norm(third[0]))
            actual = "hcp" if second_distance <= SITE_TOLERANCE else "fcc" if third_distance <= SITE_TOLERANCE else None
            if actual is None:
                return None, None, {"reason": "Subsurface stacking does not establish fcc or hcp", "second": second_distance, "third": third_distance}
            if actual != requested:
                return False, 0.0, {"observed_site": actual, "requested_site": requested}
    passed = bool(mismatch <= SITE_TOLERANCE)
    score = 1.0 if passed else math.exp(-(mismatch-SITE_TOLERANCE)/SITE_TOLERANCE)
    return passed, score, {"lateral_mismatch": float(mismatch) if math.isfinite(mismatch) else None,
                           "tolerance_angstrom": SITE_TOLERANCE, "site": requested}


def surface_facet(structure, substrate, metal, facet):
    """Check Miller normal from fitted local cubic order, not cell gamma/code."""
    from pymatgen.core import Structure
    from .paper_grain_geometry import _side_orientation, _numbers, SYMMETRIES, MATCH_TOLERANCE_DEG
    kinds={**dict.fromkeys(("Cu","Ni","Al","Ag","Au","Pt","Pd","Rh","Ir"),"fcc"),
           **dict.fromkeys(("Fe","W","Mo","Cr","V","Nb","Ta"),"bcc"),"Si":"diamond","Ge":"diamond"}
    kind=kinds.get(metal)
    plane=_numbers(facet)
    if kind is None or plane is None or np.linalg.norm(plane)==0 or not substrate:
        return None,None,{"reason":"Unsupported material lattice or unparsed Miller plane"}
    slab=Structure.from_sites([structure[i] for i in substrate])
    fit=_side_orientation(slab,list(range(len(slab))),kind)
    if fit is None:
        return None,None,{"reason":"No consistent local cubic orientation is identifiable in the substrate"}
    orientation,evidence=fit
    normal=np.cross(slab.lattice.matrix[0],slab.lattice.matrix[1]);normal/=np.linalg.norm(normal)
    requested=plane/np.linalg.norm(plane)
    normals=np.einsum("ij,ajk,k->ai",orientation,SYMMETRIES,requested)
    cosine=float(np.max(np.abs(normals@normal)))
    error=float(np.degrees(np.arccos(np.clip(cosine,-1,1))))
    passed=error<=MATCH_TOLERANCE_DEG
    return passed,math.exp(-max(0.,error-MATCH_TOLERANCE_DEG)/MATCH_TOLERANCE_DEG), {
        **evidence,"lattice_type":kind,"miller_plane":plane.tolist(),"normal_error_degrees":error,
        "tolerance_degrees":MATCH_TOLERANCE_DEG,"method":"local-neighbor orientation and cubic symmetry"}
