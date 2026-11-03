"""Benchmark prompts for Pure Prompt vs Agent comparison.

This module contains all prompts used in the benchmark_prompt_vs_agent.py script.
Separating prompts from code makes them easier to maintain and modify.
"""

from typing import Dict
import re


# =============================================================================
# System Prompt for Pure Prompt Generator
# =============================================================================

PURE_PROMPT_SYSTEM = """You are an expert materials scientist and Python programmer.
Your task is to generate pymatgen code to create crystal structures.

Requirements:
1. ONLY use basic pymatgen.core modules: Structure, Lattice, Element, Composition
2. Do NOT use pymatgen.analysis.defects or any other advanced modules
3. For vacancies: create supercell then use structure.remove_sites([index])
4. For substitutions: create supercell then use structure.replace(index, new_element)
5. Output valid Python code that creates a Structure object
6. Store the final structure in a variable called 'structure'
7. Do NOT include any explanation, only output the Python code

Example for vacancy:
```python
from pymatgen.core import Structure, Lattice

# Create pristine structure
lattice = Lattice.cubic(5.43)
pristine = Structure(lattice, ['Si', 'Si'], [[0, 0, 0], [0.25, 0.25, 0.25]])

# Create supercell
supercell = pristine * [2, 2, 2]

# Remove atom to create vacancy
supercell.remove_sites([0])
structure = supercell
```

Example for substitution:
```python
from pymatgen.core import Structure, Lattice

lattice = Lattice.cubic(5.43)
pristine = Structure(lattice, ['Si', 'Si'], [[0, 0, 0], [0.25, 0.25, 0.25]])
supercell = pristine * [2, 2, 2]

# Replace Si with Ge
supercell.replace(0, 'Ge')
structure = supercell
```
"""


# =============================================================================
# Task-Specific Prompt Templates
# =============================================================================

def get_defect_task_prompt(task: Dict) -> str:
    """Generate prompt for defect engineering task."""
    # Handle challenging tasks format (natural language description)
    if 'pristine_formula' not in task and 'description' in task:
        return f"""Create a defect structure based on the following description:

{task['description']}

Requirements:
1. Use pymatgen to create the pristine crystal structure
2. Build the appropriate supercell as specified
3. Introduce the defect(s) as described
4. Verify finite geometry, requested stoichiometry, and periodic interatomic separations appropriate to the element pairs and bonding context
5. Store the final structure in variable 'structure'
"""

    # Handle standard JARVIS defect task format
    return f"""Create a defect structure with the following specifications:

Material: {task['pristine_formula']} (JARVIS ID: {task['jid']})
Defect Type: {task['defect_type']}
Defect Element: {task['defect_element']}
{f"Substitute Element: {task['substitute_element']}" if task.get('substitute_element') else ""}
Target Concentration: {task['target_concentration'] * 100:.1f}%
Supercell Size: {task['supercell_size'][0]}x{task['supercell_size'][1]}x{task['supercell_size'][2]}
Pristine Atoms per Unit Cell: {task['num_pristine_atoms']}
Expected Defects: {task['expected_defects']}

Requirements:
1. Create the pristine structure using pymatgen
2. Build the supercell
3. Introduce the specified defects (vacancy or substitution)
4. Verify the requested defect count, concentration, and any specified defect-defect separation against the pristine lattice
5. Store the final structure in variable 'structure'
"""


def get_adsorption_task_prompt(task: Dict) -> str:
    """Generate prompt for adsorption task."""
    # Handle different task formats (old vs new)
    metal = task.get('metal') or task.get('surface_metal', 'Pt')
    facet = task.get('facet') or task.get('miller_index', '111')
    site = task.get('site') or task.get('adsorption_site', 'top')

    # Handle single vs multiple adsorbates
    if 'adsorbates' in task:
        adsorbate = ', '.join(task['adsorbates'])
    else:
        adsorbate = task.get('adsorbate') or task.get('adsorbate_formula', 'H')

    # Build prompt based on task type
    task_type = task.get('task_type', 'adsorption')
    difficulty = task.get('difficulty', 'easy')

    if task_type == 'co_adsorption':
        return f"""Create a co-adsorption structure with the following specifications:

Surface Metal: {metal}
Miller Index: ({facet})
Adsorbates: {adsorbate}
Preferred Sites: {task.get('preferred_sites', ['top', 'fcc'])}

Task: {task.get('description', 'Place multiple adsorbates on the surface')}

Requirements:
1. Create the {metal}({facet}) surface slab using pymatgen (4 layers, 2x2 supercell)
2. Place each adsorbate at appropriate sites considering adsorbate-adsorbate interactions
3. Use the requested binding heights, or a justified adsorbate-specific reference when unspecified; measure height along the local surface normal
4. Verify each molecule's stoichiometry and site, preserving internal bonds while checking periodic adsorbate-surface and adsorbate-adsorbate separations
5. Store the final structure in variable 'structure'
"""
    elif task_type == 'stepped_adsorption':
        return f"""Create an adsorption structure on a stepped surface:

Surface Metal: {metal}
Miller Index: ({facet}) - stepped surface
Adsorbate: {adsorbate}

Task: {task.get('description', 'Place adsorbate on stepped surface')}

Requirements:
1. Create the {metal}({facet}) stepped surface using pymatgen SlabGenerator
2. Place the adsorbate at the most favorable site (step edge or terrace)
3. Measure the requested binding height along the local surface normal; locate step or terrace sites using local coordination
4. Check periodic minimum-image separations with element-pair and bonding context, preserving the adsorbate's internal geometry
5. Store the final structure in variable 'structure'
"""
    elif task_type == 'complex_adsorption':
        return f"""Create an adsorption structure with a complex adsorbate:

Surface Metal: {metal}
Miller Index: ({facet})
Adsorbate: {adsorbate}

Task: {task.get('description', 'Place complex adsorbate with correct geometry')}

Requirements:
1. Create the {metal}({facet}) surface slab using pymatgen
2. Build the {adsorbate} molecule with correct geometry
3. Place the intact molecule using an explicit binding anchor and orientation relative to the local surface normal
4. Verify molecular stoichiometry and internal bonds, and check periodic adsorbate-surface separations for steric overlaps
5. Store the final structure in variable 'structure'
"""
    elif task_type == 'coverage_adsorption':
        coverage = task.get('coverage', 0.25)
        return f"""Create a high-coverage adsorption structure:

Surface Metal: {metal}
Miller Index: ({facet})
Adsorbate: {adsorbate}
Target Coverage: {coverage} ML

Task: {task.get('description', 'Create surface with specified coverage')}

Requirements:
1. Create the {metal}({facet}) surface slab using pymatgen
2. Place {adsorbate} adsorbates to achieve {coverage} ML coverage
3. Define the surface-side site-count denominator for coverage and choose an integer molecule count consistent with the requested coverage
4. Preserve molecular bonds and check periodic lateral and adsorbate-surface separations with the applicable element-pair constraints
5. Store the final structure in variable 'structure'
"""
    else:
        # Standard single adsorbate task
        coords_info = ""
        if task.get('surface_coords'):
            coords_info = f"\nSurface Coordinates: {task['surface_coords'][:3]}... ({len(task['surface_coords'])} atoms)"

        return f"""Create an adsorption structure with the following specifications:

Surface Metal: {metal}
Miller Index: ({facet})
Adsorbate: {adsorbate}
Adsorption Site: {site}
{coords_info}

Requirements:
1. Create the {metal}({facet}) surface slab using pymatgen (4 layers, 2x2 supercell)
2. Place the {adsorbate} adsorbate at the {site} site
3. Use the requested binding height, or a justified adsorbate-specific reference when unspecified; height is a projection along the local surface normal, not a nearest-atom bond length
4. Identify the requested site from local surface topology and periodic neighbors, then check element-pair separations while preserving internal molecular bonds
5. Store the final structure in variable 'structure'
"""


def extract_metal_from_formula(formula: str) -> str:
    """Extract the primary metal element from a MOF formula.

    Common MOF metals: Zn, Cu, Fe, Co, Zr, Al, Mg, Ni, Cd, Mn, Cr, V, Ti
    """
    # Common MOF metals in order of prevalence
    mof_metals = ['Zn', 'Cu', 'Fe', 'Co', 'Zr', 'Al', 'Mg', 'Ni', 'Cd', 'Mn', 'Cr', 'V', 'Ti']

    for metal in mof_metals:
        # Match metal symbol followed by optional number
        if re.search(rf'\b{metal}\d*', formula):
            return metal

    return 'Zn'  # Default fallback


def get_mof_task_prompt(task: Dict) -> str:
    """Generate prompt for MOF reconstruction task."""
    target_props = task.get('target_properties', {})
    ground_truth = task.get('ground_truth', {})

    # Extract metal from formula
    formula = ground_truth.get('formula', '')
    metal = extract_metal_from_formula(formula)

    return f"""Design a Metal-Organic Framework (MOF) structure with the following target properties:

Task ID: {task['task_id']}
Metal Node: {metal}
Target Bandgap: {target_props.get('bandgap', 'N/A')} eV
Target PLD (Pore Limiting Diameter): {target_props.get('pld', 'N/A')} Å
Target LCD (Largest Cavity Diameter): {target_props.get('lcd', 'N/A')} Å
Target Density: {target_props.get('density', 'N/A')} g/cm³

Reference Information:
- Topology: {ground_truth.get('topology', 'pcu')}
- Formula: {ground_truth.get('formula', 'N/A')}

Requirements:
1. Create a MOF structure using pymatgen
2. Use {metal} as the metal node
3. Build a structure with the specified topology
4. Ensure the structure has at least 10 atoms
5. Ensure no atomic overlaps (minimum distance > 0.8 Å)
6. Store the final structure in variable 'structure'

Generate Python code using pymatgen to create this MOF structure.
"""


def get_grain_boundary_task_prompt(task: Dict) -> str:
    """Generate prompt for grain boundary task."""
    gb_type = task.get('input', {}).get('gb_type', 'tilt')
    material = task.get('input', {}).get('material', 'Cu')
    sigma = task.get('input', {}).get('sigma', 5)
    gb_plane = task.get('input', {}).get('gb_plane', '(310)')
    rotation_axis = task.get('input', {}).get('rotation_axis', [0, 0, 1])
    constraints = task.get('input', {}).get('constraints', {})
    expected = task.get('expected_properties', {})

    return f"""Create a grain boundary structure with the following specifications:

Material: {material}
Grain Boundary Type: {gb_type}
Sigma Value: Σ{sigma}
GB Plane: {gb_plane}
Rotation Axis: {rotation_axis}
Structure Type: {expected.get('structure_type', 'fcc')}
Expected Rotation Angle: {expected.get('rotation_angle', 'N/A')}°

Constraints:
- GB Expansion: {constraints.get('gb_expansion', 1.0)} Å
- Minimum Thickness: {constraints.get('min_thickness', 10.0)} Å

Requirements:
1. Create the bulk structure for {material} using pymatgen
2. Construct a commensurate periodic bicrystal that realizes the requested CSL relation;
   use a supported CSL constructor or derive compatible integer lattice transformations
3. Realize the specified misorientation axis and angle, boundary plane in the crystal
   frame, and boundary type; verify each from the two grain orientations
4. Enforce the requested thickness and expansion along the boundary normal while
   retaining crystalline interior regions on both sides
5. Check periodic boundary contacts and element-pair separations for overlaps
6. Verify composition and the requested crystallographic constraints; retain the
   grain transformations and site assignments as construction evidence when available
7. Store the final structure in variable 'structure'

Generate Python code using pymatgen to create this grain boundary structure.
"""


def get_interface_task_prompt(task: Dict) -> str:
    """Generate prompt for heterostructure interface task."""
    input_data = task.get('input', {})
    film_material = input_data.get('film_material', 'Pt')
    film_facet = input_data.get('film_facet', '(111)')
    substrate_material = input_data.get('substrate_material', 'MgO')
    substrate_facet = input_data.get('substrate_facet', '(001)')
    application = input_data.get('application', 'catalysis')
    constraints = input_data.get('constraints', {})
    expected = task.get('expected_properties', {})

    # Handle 2D materials
    is_2d_film = film_facet == 'monolayer'
    is_2d_substrate = substrate_facet == 'monolayer'

    if is_2d_film and is_2d_substrate:
        return f"""Create a 2D van der Waals heterostructure:

Film Material: {film_material} (monolayer)
Substrate Material: {substrate_material} (monolayer)
Application: {application}
Interface Type: {expected.get('interface_type', '2D_2D')}
Bonding Type: {expected.get('bonding_type', 'vdW')}

Constraints:
- Maximum Strain: {constraints.get('max_strain', 0.02) * 100:.1f}%
- Minimum Vacuum: {constraints.get('min_vacuum', 20.0)} Å

Requirements:
1. Create monolayer structures for both {film_material} and {substrate_material}
2. Find a commensurate supercell with strain < {constraints.get('max_strain', 0.02) * 100:.1f}%
3. Stack the film immediately above the substrate along z with interlayer distance {expected.get('interface_distance_range', [3.0, 3.5])} Å
4. Add one vacuum layer of at least {constraints.get('min_vacuum', 20.0)} Å outside the stacked bilayer, not between the two materials
5. Do not use pymatgen.analysis.interfaces, InterfaceBuilder, CoherentInterfaceBuilder, MPRester, or Materials Project downloads
6. Store the final structure in variable 'structure'

Generate Python code using pymatgen to create this heterostructure.
"""
    elif is_2d_film:
        return f"""Create a 2D material on 3D substrate interface:

Film Material: {film_material} (monolayer)
Substrate Material: {substrate_material}
Substrate Facet: {substrate_facet}
Application: {application}
Interface Type: {expected.get('interface_type', '2D_3D')}

Constraints:
- Maximum Strain: {constraints.get('max_strain', 0.05) * 100:.1f}%
- Minimum Vacuum: {constraints.get('min_vacuum', 20.0)} Å
- Minimum Slab Layers: {constraints.get('min_slab_layers', 4)}

Requirements:
1. Create the {substrate_material}{substrate_facet} substrate slab
2. Create the {film_material} monolayer
3. Match lattices with strain < {constraints.get('max_strain', 0.05) * 100:.1f}%
4. Place the 2D material immediately above the substrate along z with interface distance {expected.get('interface_distance_range', [2.8, 3.5])} Å
5. Add one vacuum layer of at least {constraints.get('min_vacuum', 20.0)} Å outside the combined stack, not between film and substrate
6. Do not use pymatgen.analysis.interfaces, InterfaceBuilder, CoherentInterfaceBuilder, MPRester, or Materials Project downloads
7. Store the final structure in variable 'structure'

Generate Python code using pymatgen to create this interface.
"""
    else:
        return f"""Create a heterostructure interface:

Film Material: {film_material}
Film Facet: {film_facet}
Substrate Material: {substrate_material}
Substrate Facet: {substrate_facet}
Application: {application}
Interface Type: {expected.get('interface_type', 'metal_oxide')}
Bonding Type: {expected.get('bonding_type', 'ionic')}

Constraints:
- Maximum Strain: {constraints.get('max_strain', 0.05) * 100:.1f}%
- Minimum Vacuum: {constraints.get('min_vacuum', 15.0)} Å
- Minimum Slab Layers: {constraints.get('min_slab_layers', 4)}

Requirements:
1. Create the {substrate_material}{substrate_facet} substrate slab using SlabGenerator
2. Create the {film_material}{film_facet} film slab
3. Find lattice matching with strain < {constraints.get('max_strain', 0.05) * 100:.1f}%
4. Stack the film immediately above the substrate along z with interface distance {expected.get('interface_distance_range', [2.0, 3.0])} Å
5. Add one vacuum layer of at least {constraints.get('min_vacuum', 15.0)} Å outside the combined stack, not between film and substrate
6. Verify the closest intended film-substrate contact is within the requested interface-distance range
7. Do not use pymatgen.analysis.interfaces, InterfaceBuilder, CoherentInterfaceBuilder, MPRester, or Materials Project downloads
8. Store the final structure in variable 'structure'

Generate Python code using pymatgen to create this heterostructure interface.
"""
