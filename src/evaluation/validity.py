"""Eq. 23 universal validity, independent of critic and task constraints.

The radius search includes periodic self images and distinct coincident sites.
Finite-input/bounded-allocation checks precede native spatial-tree operations.
An uncomputable measurement is unknown, never a successful validity check.
"""
from __future__ import annotations

import itertools
import math
import numpy as np
from pymatgen.core import Element
from scipy.spatial import cKDTree
from .paper_serialization import optional_bool

DISTANCES = {"defect": 1.5, "adsorption": 0.8, "grain_boundary": 1.0, "interface": 1.0}
EPSILON = 1e-8


def _check(name, satisfied, value=None, threshold=None, reason=""):
    satisfied = optional_bool(satisfied)
    return dict(name=name, satisfied=satisfied, value=value, threshold=threshold,
                message=reason, hard=True, group="validity",
                score=1.0 if satisfied is True else 0.0 if satisfied is False else None)


def periodic_min_distance_within(structure, cutoff):
    """Exact radius-limited MIC distance, including noncentral self images.

    A Cartesian vector shorter than r has fractional component i bounded by
    r*||inv(lattice)[:,i]||. Wrapped positions differ by <1, giving a complete
    finite image set for an arbitrary (including skew) nonsingular lattice.
    Return None when no pair/image is within the radius. Raise on unavailable
    geometry; callers must not convert that exception to a passing check.
    """
    matrix = np.asarray(structure.lattice.matrix, dtype=float)
    frac = np.asarray(structure.frac_coords, dtype=float)
    if not np.isfinite(matrix).all() or not np.isfinite(frac).all():
        raise ValueError("Nonfinite cell or fractional coordinates")
    inverse = np.linalg.inv(matrix)
    bounds = np.floor((cutoff + EPSILON) * np.linalg.norm(inverse, axis=0) + 1).astype(int)
    if np.any(bounds < 0) or np.any(bounds > 100):
        raise ValueError("Cell requires an unsafe number of periodic images")
    image_count = math.prod(2 * int(x) + 1 for x in bounds)
    if image_count * len(structure) > 2_000_000:
        raise ValueError("Radius search exceeds the fixed two-million-image safety limit")
    shifts = np.array(list(itertools.product(*(range(-int(x), int(x)+1) for x in bounds))))
    wrapped = frac % 1.0
    points = ((wrapped[None, :, :] + shifts[:, None, :]) @ matrix).reshape(-1, 3)
    centers = wrapped @ matrix
    if not np.isfinite(points).all():
        raise ValueError("Nonfinite periodic image coordinates")
    # Every center has exactly one central self entry. Its second nearest
    # entry includes coincident distinct sites and periodic self images.
    distances, _ = cKDTree(points).query(centers, k=2, distance_upper_bound=cutoff + EPSILON)
    near = np.asarray(distances)[:, 1]
    near = near[np.isfinite(near)]
    return float(near.min()) if len(near) else None


def structural_validity_checks(structure, task_type):
    checks = [_check("structure_exists", structure is not None, threshold="structure object")]
    if structure is None:
        checks.extend(_check(name, None, reason="No structure is available") for name in
                      ("nonempty_valid_species", "finite_coordinates", "steric_feasibility"))
        return checks
    try:
        atoms_ok = bool(len(structure)) and all(
            bool(site.species) and all(
                isinstance(getattr(specie, "element", specie), Element)
                and math.isfinite(float(occupancy)) and float(occupancy) > 0
                for specie, occupancy in site.species.items())
            for site in structure)
    except Exception:
        atoms_ok = False
    checks.append(_check("nonempty_valid_species", atoms_ok, value=len(structure),
                         threshold="nonempty legal elements with positive finite occupancy"))
    try:
        finite = bool(np.isfinite(structure.cart_coords).all() and np.isfinite(structure.frac_coords).all())
    except Exception:
        finite = False
    checks.append(_check("finite_coordinates", finite, threshold="all Cartesian and fractional coordinates finite"))
    cutoff = DISTANCES.get(task_type)
    if cutoff is None:
        checks.append(_check("steric_feasibility", None, reason="Unknown task family; no PDF steric threshold"))
    elif not atoms_ok or not finite:
        checks.append(_check("steric_feasibility", None, threshold=cutoff, reason="Atom/coordinate prerequisites failed"))
    else:
        try:
            minimum = periodic_min_distance_within(structure, cutoff)
            checks.append(_check("steric_feasibility", minimum is None or minimum + EPSILON >= cutoff,
                                 value=minimum, threshold=cutoff,
                                 reason="MIC radius search including distinct coincident atoms and periodic self images"))
        except Exception as exc:
            checks.append(_check("steric_feasibility", None, threshold=cutoff,
                                 reason=f"Steric evidence unavailable: {type(exc).__name__}: {exc}"))
    return checks
