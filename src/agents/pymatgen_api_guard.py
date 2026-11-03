"""
Pymatgen API guardrails for generated structure-construction code.

This module is intentionally separate from the default prompt templates so the
methods-inspired agent can use stronger compatibility hints without changing the
existing code path.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional


BASE_GUARD = """Pymatgen API compatibility guard for generated code:
- Always create a pymatgen Structure object named `structure`.
- Do not call Materials Project APIs or MPRester in generated benchmark code.
  Construct known prototype bulk structures locally from lattice constants.
- Do not import `minimum_distance` from `pymatgen.analysis.structure_analyzer`.
  For distance checks use `structure.distance_matrix`, set the diagonal to
  infinity, or call `structure.get_distance(i, j)`.
- Validate species counts before returning: `structure.composition.as_dict()`
  must match the task target, including any adsorbate or defect count.
- Avoid relying on optional pymatgen modules that may not exist in the local
  installation. Prefer `pymatgen.core`, `pymatgen.core.surface`, and explicit
  coordinate construction."""


SURFACE_API_GUARD = """Current pymatgen surface API guard:
- Installed `SlabGenerator` signature is:
  `SlabGenerator(initial_structure, miller_index, min_slab_size, min_vacuum_size, lll_reduce=False, center_slab=False, in_unit_planes=False, primitive=True, max_normal_search=None, reorient_lattice=True)`.
- Do not pass unsupported kwargs: `symmetrize`, `min_slab_thickness`, or
  `min_vacuum`.
- Do not mutate `structure.lattice.matrix` in place; it can be read-only.
  Copy the matrix, edit the copy, and create `Lattice(new_matrix)`.
- Do not call `slab.get_slab_thickness()`; that helper is not available here.
  Use `float(np.max(slab.cart_coords[:, 2]) - np.min(slab.cart_coords[:, 2]))`
  for a Cartesian z-thickness estimate.
- If using AdsorbateSiteFinder, import it from `pymatgen.core.adsorption`
  or fall back to manual site placement. Do not import AdsorbateSiteFinder from
  `pymatgen.core.surface`.
- If SlabGenerator raises `SymmetryUndeterminedError`, rebuild the bulk with a
  valid local prototype or construct the slab manually from repeated Cartesian
  layers instead of retrying the same invalid bulk."""


ADSORPTION_GUARD = """Surface adsorption construction guard:
- Use `from pymatgen.core.surface import SlabGenerator` for slabs.
- Choose the substrate prototype, composition, and surface orientation from
  the task and relevant material knowledge; do not assume every metal is fcc.
- Enforce explicitly requested atom counts and adsorbate stoichiometry. Keep
  molecule multiplicity and target coverage separate from total slab size;
  specify the exposed-surface-site convention when interpreting coverage.
- Establish the surface normal from the in-plane lattice vectors. Use a
  consistent periodic image for the slab and each intact molecule before
  identifying exposed layers, local step/terrace sites, or binding positions.
  Global Cartesian z extrema are not a general periodic surface definition.
- Place each molecule using its binding atom and local surface normal. Choose
  heights and orientations from explicit task requirements or documented
  chemical knowledge, not a universal height window or fixed-material recipe.
- Check finite coordinates and a nonsingular lattice before distance routines.
  Re-check periodic contacts after placement, distinguishing bonded molecular
  pairs from unintended slab/adsorbate or adsorbate/adsorbate collisions.
- If `surface_coords` and `surface_lattice` are provided in the structured
  task input, use that benchmark slab directly as the starting surface. Do not
  rebuild a smaller primitive slab, change the in-plane lattice, or call
  `make_supercell` on top of those already-expanded coordinates.
- If AdsorbateSiteFinder returns a dictionary, access named keys such as
  `all`, `ontop`, `bridge`, or `hollow`; do not treat it as a plain list."""


GRAIN_BOUNDARY_CONSTRUCTOR_CODE = """def select_cubic_csl_angle(sigma, rotation_axis, requested_angle, input_decimals=None):
    # Input rounding interval, NOT a structural/scoring tolerance.
    import math
    from decimal import Decimal
    from numbers import Integral
    from pymatgen.core.interface import GrainBoundaryGenerator
    axis = tuple(rotation_axis)
    if isinstance(sigma, bool) or not isinstance(sigma, Integral) or sigma < 2:
        raise ValueError("Sigma must be an integer >= 2")
    if len(axis) != 3 or not any(axis) or any(
        isinstance(v, bool) or not isinstance(v, Integral) for v in axis
    ):
        raise ValueError("Cubic axis must contain three nonzero-together integer indices")
    target = Decimal(str(requested_angle))
    if not target.is_finite() or not 0 < target <= 180:
        raise ValueError("Requested angle must be finite and in (0, 180] degrees")
    if input_decimals is not None and (
        isinstance(input_decimals, bool) or not isinstance(input_decimals, Integral)
        or input_decimals < 0
    ):
        raise ValueError("input_decimals must be a nonnegative integer")
    if isinstance(requested_angle, str) and input_decimals is not None \
            and target.as_tuple().exponent != -int(input_decimals):
        raise ValueError("input_decimals conflicts with the displayed angle precision")
    exponent = target.as_tuple().exponent if input_decimals is None else -int(input_decimals)
    half_unit = Decimal(5).scaleb(exponent - 1)
    # enum returns {Sigma: [numeric angles]}, not records or (Sigma, angle) pairs.
    angles = GrainBoundaryGenerator.enum_sigma_cubic(int(sigma), axis).get(int(sigma), [])
    roundoff = Decimal(str(8 * math.ulp(float(target))))
    matches = sorted({float(a) for a in angles
                      if abs(Decimal(str(float(a))) - target) <= half_unit + roundoff})
    if len(matches) != 1:
        raise ValueError(f"Expected one angle matching Sigma={sigma}, axis={axis}, "
                         f"input={requested_angle}; matches={matches}; candidates={angles}")
    return matches[0]


def construct_periodic_gb(bulk, rotation_axis, exact_angle=None, plane=None,
                          min_grain_thickness=None, expand_times=4, gap=0.0,
                          *, sigma=None, requested_angle=None, input_decimals=None,
                          ratio=None):
    import math
    import numpy as np
    from pymatgen.core import Structure
    from pymatgen.core.interface import GrainBoundaryGenerator
    generator = GrainBoundaryGenerator(bulk)
    if sigma is not None:
        if generator.lat_type.lower() != "c":
            raise ValueError("Cubic Sigma selection requires a cubic prototype")
        if exact_angle is not None or requested_angle is None:
            raise ValueError("Supply sigma + requested_angle, or exact_angle, not both")
        exact_angle = select_cubic_csl_angle(sigma, rotation_axis, requested_angle, input_decimals)
    elif requested_angle is not None or input_decimals is not None:
        raise ValueError("Rounded-angle selection requires the requested Sigma")
    if exact_angle is None or not math.isfinite(float(exact_angle)) or plane is None:
        raise ValueError("Supply a finite exact angle and an explicit boundary plane")
    def build(repeats):
        return generator.gb_from_parameters(
            rotation_axis=tuple(rotation_axis), rotation_angle=float(exact_angle),
            plane=tuple(plane), expand_times=int(repeats),
            vacuum_thickness=float(gap), normal=False, ratio=ratio,
        )
    if min_grain_thickness is not None:
        unit = build(1)
        a, b, c = unit.lattice.matrix
        normal = np.cross(a, b)
        normal /= np.linalg.norm(normal)
        # The periodic cell contains two grains and two requested gaps.
        repeat_length = abs(float(c @ normal)) / 2.0 - float(gap)
        if repeat_length <= 0 or not math.isfinite(repeat_length):
            raise ValueError("Invalid oriented grain repeat length")
        # One extra repeat covers the missing outer atomic plane of each grain.
        expand_times = max(1, math.ceil(float(min_grain_thickness) / repeat_length) + 1)
    bicrystal = build(expand_times)
    # Keep the CSL lattice and coordinates together. Metadata is not proof.
    return Structure(bicrystal.lattice, bicrystal.species, bicrystal.frac_coords)
"""


GRAIN_BOUNDARY_GUARD = """Installed pymatgen grain-boundary construction contract:
- Use a verified bulk prototype and elemental basis with
  `pymatgen.core.interface.GrainBoundaryGenerator`. Axis and plane indices must
  retain their specified basis and signs; a direction family is not a literal
  direction. The rotation axis need not be the boundary normal.
- For cubic Sigma tasks use the helper below directly:
  `structure = construct_periodic_gb(bulk, axis, plane=plane, sigma=sigma,
  requested_angle=angle_text, min_grain_thickness=thickness, gap=gap)`.
  `angle_text` preserves the input decimal precision (e.g. "36.87"); floats use
  their string representation, or explicit `input_decimals` from the input.
  This selects a full-precision member of the numeric enum list. It changes
  neither scoring tolerances nor the requested Sigma; do not round the result.
  Empty or ambiguous matches are errors, not permission to change axis/plane
  or use `get_rotation_angle_from_sigma` (which can fall back to smaller Sigma).
- The existing `(bulk, axis, exact_angle, plane, ...)` call remains available
  for an already verified exact angle. Noncubic tasks require the appropriate
  lattice-specific enumeration and axial `ratio`; never use cubic enumeration
  for hcp. Unspecified crystallographic choices must be identified as choices.
- Preserve the native periodic CSL lattice and its coordinates together;
  rotating atoms in a replacement orthorhombic box does not establish a CSL.
  `normal=False` retains the compact oblique cell. Thickness is measured along
  cross(a,b); the helper sizes repeats from the native normal period plus one
  outer-plane allowance. Verify final occupied thickness, both boundaries,
  periodic contacts/self images, composition and local grain orientations.
  Construction metadata alone is not geometric verification.
- Gap, thickness and cell size follow the task, not fixed Miller-index rules.
  The library's default `rm_ratio=0.7` handles close intergrain duplicates;
  do not prune atoms to fit an MLIP budget or add a new distance acceptance rule.
  If a compatible cell exceeds resources, report that limitation separately.
  The code returns a plain Structure without scoring or hidden optimization.
```python
""" + GRAIN_BOUNDARY_CONSTRUCTOR_CODE + "```"



INTERFACE_GUARD = """Interface construction guard:
- Do not use `InterfaceBuilder`; it is not present in this pymatgen runtime.
- If using `CoherentInterfaceBuilder`, use only the installed signature:
  `CoherentInterfaceBuilder(substrate_structure, film_structure, film_miller, substrate_miller, zslgen=None, termination_ftol=0.25, label_index=False, filter_out_sym_slabs=True)`.
  Do not pass `max_interface_area`, `max_length_mismatch`, or similar kwargs.
- Prefer a robust manual stack when the exact interface builder API is unclear:
  construct film and substrate bulk locally, make oriented slabs, choose the
  best small-integer match, strain the film in-plane to the substrate cell,
  shift the film above the substrate, and combine species plus Cartesian
  coordinates into one Structure.
- For interface slabs, use the current SlabGenerator kwargs from the surface API
  guard. Never pass `symmetrize=True`, `min_slab_thickness`, or `min_vacuum`.
- Do not assign into `slab.lattice.matrix` or `structure.lattice.matrix`.
  Build a copied matrix and then call `Structure(Lattice(new_matrix), ...)`.
- `Slab`/`Structure` objects in this runtime do not support
  `modify_lattice(...)`. To change the lattice, create a new object with
  `Structure(Lattice(new_matrix), old_species, old_cart_coords,
  coords_are_cartesian=True)`.
- `make_supercell` scaling matrices must be full-rank integer 3x3 matrices
  with positive determinant. Never use a matrix with a zero row/column, e.g.
  `[[m,0,0],[0,n,0],[0,0,1]]` is valid while `[[m,0,0],[0,n,0],[0,0,0]]`
  is singular and invalid.
- Do not encode matched in-plane vector coefficients as
  `[[n1,n2,0],[m1,m2,0],[0,0,1]]` unless you explicitly verify
  `n1*m2 - n2*m1 != 0`. For benchmark reliability, prefer diagonal
  in-plane repeats (`[[na,0,0],[0,nb,0],[0,0,1]]`) and then strain/replace
  the film Cartesian xy coordinates into the final substrate lattice.
- When constructing a lattice matrix manually, every row must be a 3-vector.
  If matched in-plane vectors are 2D xy arrays, convert them to
  `[ax, ay, 0.0]` and `[bx, by, 0.0]` before stacking with `[0.0, 0.0, c]`.
- Search commensurate in-plane cells within the explicit task strain limit
  and available resources. Accept a match only if it meets that limit; if none
  is found, report the unmet constraint. Do not silently relax the strain limit
  or present the closest incompatible match as satisfying the task.
- Keep the substrate as the lower slab and the film as the upper slab along the
  c axis. Target an interface gap from `interface_distance_range` when present,
  otherwise use 2.0-2.6 A for chemisorbed/ionic interfaces and 3.0-3.5 A for
  vdW interfaces.
- Include vacuum along c after stacking: use the requested `min_vacuum` when
  present, otherwise at least 12-15 A. Recompute the c lattice length from the
  combined slab thickness plus vacuum.
- Avoid brittle orientation assertions such as `MgO slab aligned with x-axis`.
  Validate the final stacked structure by material identity, slab layers,
  interface distance, vacuum, and minimum periodic distance.
- After combining the slabs, compute the periodic distance matrix. If the
  minimum nonzero distance is below 1.2 A, increase the interface gap or apply a
  small lateral offset before returning.
- Generated code should contain ASCII Python only; do not include unicode
  punctuation in comments or strings.
- Miller indices must be numeric Python tuples, for example `(1, 1, 1)`.
  Never place prose, placeholders, or non-ASCII words inside tuple literals.
- Never fetch film/substrate structures from MPRester during benchmark runs."""


DEFECT_GUARD = """Defect construction guard:
- Use local prototype structures and explicit site edits: `make_supercell`,
  `remove_sites`, `replace`, and `append`.
- For vacancy and substitution tasks, compute expected composition after the
  edit and assert it before returning.
- For vacancy tasks, do not assert that the final reduced formula equals the
  pristine host formula. A vacancy changes stoichiometry by definition. Validate
  expected atom counts and element counts instead.
- For interstitials, use Cartesian placement near a physically separated site
  and reject placements that create very short distances."""


TASK_CONSTRAINT_KEYS = (
    "expected_total_atoms",
    "expected_atoms",
    "expected_composition",
    "expected_properties",
    "validation",
    "constraints",
    "composition",
    "material",
    "host_material",
    "defect_element",
    "defect_type",
    "catalyst_metal",
    "metal",
    "adsorbate",
    "adsorption_site",
    "site",
    "film_material",
    "substrate_material",
    "film_miller",
    "substrate_miller",
    "gb_plane",
    "gb_type",
    "sigma",
    "rotation_axis",
    "rotation_angle",
    "structure_type",
    "crystal_type",
)

def infer_guard_sections(task_description: str = "") -> List[str]:
    """Select guard sections relevant to the task."""
    lower = (task_description or "").lower()
    sections = [BASE_GUARD]

    needs_surface_api = any(
        token in lower
        for token in (
            "adsorption",
            "adsorbate",
            "surface",
            "slab",
            "slabgenerator",
            "interface",
            "grain boundary",
        )
    )
    if needs_surface_api:
        sections.append(SURFACE_API_GUARD)
    if any(token in lower for token in ("adsorption", "adsorbate", "surface", "slab")):
        sections.append(ADSORPTION_GUARD)
    if any(token in lower for token in ("grain boundary", "csl", "sigma", "tilt boundary")):
        sections.append(GRAIN_BOUNDARY_GUARD)
    if any(token in lower for token in ("interface", "heterostructure", "film", "substrate")):
        sections.append(INTERFACE_GUARD)
    if any(token in lower for token in ("defect", "vacancy", "substitution", "interstitial")):
        sections.append(DEFECT_GUARD)

    return sections


def get_pymatgen_api_guard(
    task_description: str = "",
    task_input: Optional[Dict[str, Any]] = None,
) -> str:
    """Return a compact compatibility note for the current task."""
    sections = infer_guard_sections(task_description)
    fields = task_input or {}
    family = fields.get("task_type")
    is_grain = family == "grain_boundary" or (not family and fields.get("material") and any(
        key in fields for key in ("sigma", "gb_type", "gb_plane", "rotation_axis")
    ))
    if family and not is_grain:
        # An explicit domain outranks incidental text such as defect segregation
        # near a grain boundary. Text fallback is only for untyped callers.
        sections = [section for section in sections if section != GRAIN_BOUNDARY_GUARD]
    elif is_grain:
        # Twin/HCP descriptions need not contain the lexical "grain boundary".
        if GRAIN_BOUNDARY_GUARD not in sections:
            sections.append(GRAIN_BOUNDARY_GUARD)
    task_constraints = format_task_constraints(task_input or {})
    if task_constraints:
        sections.insert(1, task_constraints)
    return "\n\n".join(sections).strip()


def format_task_constraints(task_input: Dict[str, Any]) -> str:
    """Promote structured benchmark fields into hard generation constraints."""
    constraints = []
    for key in TASK_CONSTRAINT_KEYS:
        if key not in task_input:
            continue
        value = task_input[key]
        if value is None or value == "":
            continue
        constraints.append(f"- {key}: {_compact_value(value)}")

    constraints.extend(_derived_task_hints(task_input))

    if not constraints:
        return ""

    return (
        "Task-input hard constraints for this benchmark item:\n"
        + "\n".join(constraints)
        + "\nBefore returning, explicitly verify that the final `structure` satisfies "
        "these fields. For exact atom-count or composition fields, rebuild rather "
        "than accepting a near miss."
    )


def _compact_value(value: Any) -> str:
    if isinstance(value, dict):
        items = [f"{key}={_compact_value(val)}" for key, val in value.items()]
        return "{" + ", ".join(items) + "}"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_compact_value(item) for item in value) + "]"
    return str(value)


def _derived_task_hints(task_input: Dict[str, Any]) -> List[str]:
    hints: List[str] = []

    expected_total_atoms = task_input.get("expected_total_atoms") or task_input.get("expected_atoms")
    catalyst = task_input.get("catalyst_metal") or task_input.get("metal")
    adsorbate = task_input.get("adsorbate")
    if expected_total_atoms and catalyst and adsorbate:
        try:
            # A scalar count describes one adsorbate species. Multiple species
            # or coverage without an explicit molecule count are ambiguous.
            count = None
            if not task_input.get("adsorbates"):
                if "adsorbate_count" in task_input:
                    count = _nonnegative_integer(task_input["adsorbate_count"])
                elif (task_input.get("coverage") is None
                      and "coverage" not in str(task_input.get("task_type", "")).lower()):
                    count = 1
            catalyst_atoms = None
            if count is not None:
                adsorbate_atoms = count * _formula_atom_count(str(adsorbate))
                catalyst_atoms = _nonnegative_integer(expected_total_atoms) - adsorbate_atoms
            if catalyst_atoms is not None and catalyst_atoms > 0:
                hints.append(
                    f"- derived target catalyst atom count: {catalyst_atoms} {catalyst} atoms"
                )
                hints.append(
                    "- adsorption count rule: if slab generation gives a different catalyst "
                    "atom count, rebuild the slab before appending the adsorbate"
                )
        except (TypeError, ValueError):
            pass
    if catalyst and adsorbate:
        scaffold = _adsorption_reference_scaffold(task_input)
        if scaffold:
            hints.append(scaffold)

    defect_type = str(task_input.get("defect_type", "")).lower()
    host_material = task_input.get("host_material") or task_input.get("material")
    if "vacancy" in defect_type and host_material:
        hints.append(
            "- vacancy validation rule: do not assert final reduced_formula equals "
            f"{host_material}; use explicit post-vacancy composition/count checks"
        )

    if task_input.get("film_material") and task_input.get("substrate_material"):
        hints.append(
            "- interface matching rule: search small integer in-plane supercell pairs "
            "and use the best small-integer match before failing on strain"
        )
        hints.append(
            "- interface stack rule: strain film in-plane to the matched substrate "
            "cell, place substrate below film along c, and set the z gap from "
            "interface_distance_range when available"
        )
        scaffold = _interface_reference_scaffold(task_input)
        if scaffold:
            hints.append(scaffold)

    if task_input.get("material") and any(
        key in task_input for key in ("sigma", "gb_type", "gb_plane", "rotation_axis")
    ):
        scaffold = _grain_boundary_reference_scaffold(task_input)
        if scaffold:
            hints.append(scaffold)

    return hints


def _grain_boundary_rotation_angle(task_input: Dict[str, Any]) -> Optional[float]:
    expected = task_input.get("expected_properties") or {}
    if isinstance(expected, dict) and expected.get("rotation_angle") is not None:
        try:
            return float(expected["rotation_angle"])
        except (TypeError, ValueError):
            pass

    if task_input.get("rotation_angle") is not None:
        try:
            return float(task_input["rotation_angle"])
        except (TypeError, ValueError):
            pass

    # Sigma alone does not determine the axis, angle or symmetry branch.
    return None


def _grain_boundary_reference_scaffold(task_input: Dict[str, Any]) -> str:
    """Expose requested crystallographic inputs without fabricating a proxy."""
    material = str(task_input.get("material") or "").strip()
    if not material:
        return ""
    constraints = task_input.get("constraints") or {}
    expected = task_input.get("expected_properties") or {}
    constraints = constraints if isinstance(constraints, dict) else {}
    expected = expected if isinstance(expected, dict) else {}
    kind = expected.get("structure_type", task_input.get("structure_type", task_input.get("structure")))
    requested = {
        "material": material, "crystal_type": kind,
        "sigma": task_input.get("sigma"),
        "rotation_axis": task_input.get("rotation_axis"),
        "rotation_angle_degrees": _grain_boundary_rotation_angle(task_input),
        "boundary_plane": task_input.get("gb_plane"),
        "boundary_type": task_input.get("gb_type"),
        "min_grain_thickness_angstrom": constraints.get("min_thickness"),
        "intergrain_expansion_angstrom": constraints.get("gb_expansion"),
    }
    requested = {key: value for key, value in requested.items() if value is not None}
    return (
        "- Native grain-boundary constructor inputs (retain their crystallographic meaning): "
        + _compact_value(requested)
        + ". Use the installed pymatgen.core.interface.GrainBoundaryGenerator contract. "
        "Missing axis/angle/plane fields are task freedoms or missing evidence, not a license "
        "to claim a default [001] axis, (111) plane or zero-degree rotation was specified. "
        "Material identity, periodic geometry and requested targets must be checked independently."
    )


def _interface_reference_scaffold(task_input: Dict[str, Any]) -> str:
    """Return compact task-specific constants for manual interface stacking."""
    film = str(task_input.get("film_material") or "").strip()
    substrate = str(task_input.get("substrate_material") or "").strip()
    if not film or not substrate:
        return ""

    constraints = task_input.get("constraints") or {}
    if not isinstance(constraints, dict):
        constraints = {}
    expected = task_input.get("expected_properties") or {}
    if not isinstance(expected, dict):
        expected = {}

    film_facet = task_input.get("film_facet") or task_input.get("film_miller") or "(001)"
    substrate_facet = (
        task_input.get("substrate_facet")
        or task_input.get("substrate_miller")
        or "(001)"
    )
    max_strain = constraints.get("max_strain", 0.05)
    min_vacuum = constraints.get("min_vacuum", 15.0)
    min_slab_layers = constraints.get("min_slab_layers", 4)
    distance_range = expected.get("interface_distance_range", [2.0, 3.0])
    bonding_type = expected.get("bonding_type", "")

    return f"""- interface reference scaffold: keep these constants in generated code,
  then implement a bounded manual stack:
  `film_material = "{film}"; substrate_material = "{substrate}";
  film_facet = "{film_facet}"; substrate_facet = "{substrate_facet}";
  max_strain = {max_strain}; min_vacuum = {min_vacuum};
  min_slab_layers = {min_slab_layers};
  interface_distance_range = {_compact_value(distance_range)};
  bonding_type = "{bonding_type}"`.
  Build local prototype slabs, enumerate in-plane integer repeats 1..5 for
  film and substrate, select the lowest-area pair with strain <= max_strain
  when possible, otherwise select the minimum-strain pair instead of failing.
  Use the matched substrate xy vectors as the final xy lattice, transform film
  Cartesian xy coordinates by the strain matrix into that cell, shift the film
  so `film_z_min = substrate_z_max + target_gap`, where `target_gap` is the
  midpoint of `interface_distance_range`, set `c_length = slab_thickness +
  min_vacuum`, create `structure = Structure(Lattice(new_matrix), species,
  coords, coords_are_cartesian=True)`, and verify composition, layers, vacuum,
  interface gap, and periodic minimum distance before returning."""


def _adsorption_reference_scaffold(task_input: Dict[str, Any]) -> str:
    """Return compact task-specific constants for manual adsorption stacking."""
    surface_coords = task_input.get("surface_coords")
    surface_lattice = task_input.get("surface_lattice")
    if not surface_coords or not surface_lattice:
        return ""

    metal = str(
        task_input.get("catalyst_metal")
        or task_input.get("metal")
        or task_input.get("surface_metal")
        or ""
    ).strip()
    adsorbate = str(task_input.get("adsorbate") or "").strip()
    site = str(task_input.get("adsorption_site") or task_input.get("site") or "top").strip()
    expected_atoms = task_input.get("expected_total_atoms") or task_input.get("expected_atoms")
    surface_atom_count = len(surface_coords) if isinstance(surface_coords, list) else "provided"
    species_literal = (
        repr(task_input.get("surface_species"))
        if task_input.get("surface_species")
        else f"[{metal!r}] * len(surface_coords)"
    )

    return f"""- adsorption reference scaffold: the structured task input already
  contains the benchmark surface. Copy these Python literals into generated
  code before building the Structure:
  `surface_lattice = {repr(surface_lattice)}`;
  `surface_coords = {repr(surface_coords)}`;
  `surface_species = {species_literal}`.
  Then use `surface = Structure(Lattice(surface_lattice), surface_species,
  surface_coords, coords_are_cartesian=True)` as the starting slab; do not
  reconstruct or rescale the slab. It has {surface_atom_count} surface atoms.
  Keep `adsorbate = "{adsorbate}"`, `adsorption_site = "{site}"`, and
  expected_total_atoms = {expected_atoms}. Establish a consistent periodic
  image and local surface normal before selecting binding sites. Preserve each
  molecule's connectivity while placing its binding atom; choose height and
  orientation from the task or documented chemical knowledge. Verify requested
  counts, composition, local adsorption geometry, and periodic contacts before
  returning; do not infer the surface solely from raw Cartesian z extrema."""


def _formula_atom_count(formula: str) -> int:
    """Count atoms in a valid molecular formula, including parenthesized groups."""
    from pymatgen.core import Composition

    amounts = list(Composition(formula, strict=True).values())
    if not amounts or any(not math.isfinite(amount) or amount <= 0
                          or not float(amount).is_integer() for amount in amounts):
        raise ValueError("Adsorbate formula must contain positive integer atom counts")
    return sum(int(amount) for amount in amounts)


def _nonnegative_integer(value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError("Atom and molecule counts cannot be booleans")
    number = float(value)
    if not math.isfinite(number) or number < 0 or not number.is_integer():
        raise ValueError("Atom and molecule counts must be nonnegative integers")
    return int(number)
