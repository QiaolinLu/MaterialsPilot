"""
Adsorption Structure Evaluator

Evaluates adsorption structures for physical validity and accuracy.
Migrated and enhanced from tasks/ocp_adsorption/evaluator.py and evaluator_v2.py
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from pymatgen.core import Structure

from .base import BaseEvaluator, EvaluationResult
from .config import (
    AdsorptionConfig,
    DEFAULT_ADSORPTION_CONFIG,
    LATTICE_CONSTANTS,
    ADSORBATE_ELEMENTS,
    ADSORBATE_ATOM_COUNTS
)


class AdsorptionEvaluator(BaseEvaluator):
    """Evaluator for adsorption structure generation tasks.

    Validates:
    - Atom count and composition
    - Atom overlaps
    - Adsorbate height above surface
    - Adsorbate geometry preservation
    - Surface/slab structure integrity
    - Lattice parameters
    """

    def __init__(self, validation_level: int = 1,
                 success_threshold: Optional[float] = None,
                 config: Optional[AdsorptionConfig] = None):
        """Initialize adsorption evaluator.

        Args:
            validation_level: 1=basic, 2=intermediate, 3=advanced
            success_threshold: Minimum score for success (overrides config if provided)
            config: AdsorptionConfig instance (uses default if not provided)
        """
        self.config = config or DEFAULT_ADSORPTION_CONFIG
        threshold = success_threshold if success_threshold is not None else self.config.success_threshold
        enable_relaxation = self.config.enable_relaxation_check
        super().__init__(threshold, enable_relaxation_check=enable_relaxation)
        self.validation_level = validation_level

        # Set thresholds from config
        self.MIN_INTERATOMIC_DISTANCE = self.config.min_interatomic_distance
        self.CLOSE_CONTACT_DISTANCE = self.config.close_contact_distance
        self.HEIGHT_TOLERANCE = self.config.height_tolerance
        self.GEOMETRY_TOLERANCE = self.config.geometry_tolerance
        self.LATTICE_TOLERANCE = self.config.lattice_tolerance
        self.SITE_TOLERANCES = self.config.site_tolerances

    def evaluate(self, structure: Structure, task: Dict[str, Any]) -> EvaluationResult:
        """Evaluate an adsorption structure.

        Args:
            structure: Generated adsorption structure
            task: Task specification with adsorption parameters

        Returns:
            EvaluationResult with detailed metrics
        """
        if structure is None:
            return EvaluationResult(
                score=0.0,
                success=False,
                errors=["No structure generated"],
                metrics={'structure': None}
            )

        if len(structure) == 0:
            return EvaluationResult(
                score=0.0,
                success=False,
                errors=["Generated structure is empty"],
                metrics={'atom_count': 0}
            )

        errors = []
        warnings = []
        scores = []
        metrics = {}

        # Extract task parameters
        task_input = task.get('input', task)
        catalyst_metal = task_input.get('catalyst_metal', task_input.get('metal', 'Pt'))
        adsorbate = task_input.get('adsorbate', 'CO')
        adsorption_site = task_input.get('adsorption_site', task_input.get('site', 'top'))
        expected_atoms = task_input.get('expected_total_atoms', task_input.get('expected_atoms'))
        target_height = task_input.get('target_height', 2.0)
        expected_height_range = task_input.get('expected_height_range', (1.5, 2.5))

        metrics['catalyst_metal'] = catalyst_metal
        metrics['adsorbate'] = adsorbate
        metrics['adsorption_site'] = adsorption_site

        # ==================== Level 1: Basic Checks ====================

        # Check atom overlaps
        overlap_score, overlap_msg = self.check_atom_overlaps(structure)
        scores.append(overlap_score)
        metrics['min_interatomic_distance'] = self._get_min_distance(structure)
        if overlap_msg:
            if overlap_score == 0:
                errors.append(overlap_msg)
            else:
                warnings.append(overlap_msg)

        # Check atom count
        atom_score, atom_msg = self._check_atom_count_adsorption(
            structure, expected_atoms, catalyst_metal, adsorbate
        )
        scores.append(atom_score)
        metrics['atom_count'] = len(structure)
        metrics['expected_atoms'] = expected_atoms
        if atom_msg:
            if atom_score < 0.5:
                errors.append(atom_msg)
            else:
                warnings.append(atom_msg)

        # Check elements present
        elem_score, elem_msg = self._check_elements(structure, catalyst_metal, adsorbate)
        scores.append(elem_score)
        metrics['composition'] = structure.composition.as_dict()
        if elem_msg:
            if elem_score == 0:
                errors.append(elem_msg)
            else:
                warnings.append(elem_msg)

        # ==================== Level 2: Intermediate Checks ====================

        if self.validation_level >= 2:
            # Check adsorbate height
            height_score, height_msg, height_metrics = self._check_adsorbate_height(
                structure, catalyst_metal, target_height, expected_height_range
            )
            scores.append(height_score)
            metrics.update(height_metrics)
            if height_msg:
                if height_score < 0.5:
                    errors.append(height_msg)
                else:
                    warnings.append(height_msg)

            # Check lattice parameters
            lattice_score, lattice_msg = self._check_lattice(structure, catalyst_metal)
            scores.append(lattice_score)
            metrics['lattice_abc'] = list(structure.lattice.abc)
            if lattice_msg:
                warnings.append(lattice_msg)

        # ==================== Level 3: Advanced Checks ====================

        if self.validation_level >= 3:
            # Check slab structure (layers)
            slab_score, slab_msg, slab_metrics = self._check_slab_structure(
                structure, catalyst_metal
            )
            scores.append(slab_score)
            metrics.update(slab_metrics)
            if slab_msg:
                warnings.append(slab_msg)

            # Check adsorbate geometry (for multi-atom adsorbates)
            geom_score, geom_msg = self._check_adsorbate_geometry(
                structure, catalyst_metal, adsorbate
            )
            scores.append(geom_score)
            if geom_msg:
                warnings.append(geom_msg)

            # Check adsorption site
            site_score, site_msg = self._check_adsorption_site(
                structure, catalyst_metal, adsorption_site
            )
            scores.append(site_score)
            metrics['adsorption_site_score'] = site_score
            if site_msg:
                if site_score < 0.5:
                    errors.append(site_msg)
                else:
                    warnings.append(site_msg)

        # ==================== MLIP Energy Check ====================

        # Only run MLIP check if enabled
        if self.config.enable_mlip_check:
            mlip_score, mlip_msg, mlip_metrics = self._check_adsorption_stability(
                structure, catalyst_metal
            )
            scores.append(mlip_score)
            metrics['mlip'] = mlip_metrics
            if mlip_msg:
                if mlip_score < 0.5:
                    warnings.append(mlip_msg)
                else:
                    warnings.append(mlip_msg)

        # Calculate overall score
        # Index 0 is atom overlap check - mark as critical
        overall_score = self._aggregate_scores(scores, critical_indices=[0])
        success = overall_score >= self.success_threshold

        return EvaluationResult(
            score=overall_score,
            success=success,
            errors=errors,
            warnings=warnings,
            metrics=metrics
        )

    def _check_atom_count_adsorption(self, structure: Structure,
                                      expected_atoms: Optional[int],
                                      metal: str,
                                      adsorbate: str) -> Tuple[float, Optional[str]]:
        """Check atom count for adsorption structure."""
        actual = len(structure)

        if expected_atoms is not None:
            # Allow some tolerance (±4 atoms for different supercell sizes)
            tolerance = 4
            diff = abs(actual - expected_atoms)

            if diff == 0:
                return 1.0, None
            elif diff <= tolerance:
                return 0.8, f"Atom count slightly off: {actual} vs expected {expected_atoms}"
            else:
                return 0.3, f"Atom count mismatch: {actual} vs expected {expected_atoms}"

        # If no expected count, check minimum reasonable size
        min_atoms = 10  # At least some surface atoms + adsorbate
        if actual >= min_atoms:
            return 1.0, None
        else:
            return 0.5, f"Low atom count: {actual}"

    def _check_elements(self, structure: Structure,
                        metal: str,
                        adsorbate: str) -> Tuple[float, Optional[str]]:
        """Check if structure contains correct elements."""
        actual_elements = set(str(site.specie) for site in structure)

        # Expected elements: metal + adsorbate atoms
        expected_elements = {metal}
        adsorbate_elems = ADSORBATE_ELEMENTS.get(adsorbate, set())
        if adsorbate_elems:
            expected_elements |= adsorbate_elems
        else:
            # Single element adsorbate
            expected_elements.add(adsorbate)

        missing = expected_elements - actual_elements

        if not missing:
            return 1.0, None
        elif len(missing) == 1:
            return 0.5, f"Missing element: {missing}"
        else:
            return 0.0, f"Missing elements: {missing}"

    def _check_adsorbate_height(self, structure: Structure,
                                 metal: str,
                                 target_height: float,
                                 expected_range: Tuple[float, float]) -> Tuple[float, Optional[str], Dict]:
        """Check if adsorbate is at correct height above surface."""
        metrics = {}

        # Identify metal atoms (surface) and adsorbate atoms
        metal_z = []
        adsorbate_z = []

        for site in structure:
            elem = str(site.specie)
            z = site.coords[2]

            if elem == metal:
                metal_z.append(z)
            else:
                adsorbate_z.append(z)

        if not metal_z or not adsorbate_z:
            return 0.5, "Could not identify surface/adsorbate atoms", metrics

        surface_top = max(metal_z)
        adsorbate_bottom = min(adsorbate_z)
        actual_height = adsorbate_bottom - surface_top

        metrics['surface_top_z'] = surface_top
        metrics['adsorbate_bottom_z'] = adsorbate_bottom
        metrics['actual_height'] = actual_height
        metrics['target_height'] = target_height

        # Check against expected range
        min_h, max_h = expected_range
        if min_h - self.HEIGHT_TOLERANCE <= actual_height <= max_h + self.HEIGHT_TOLERANCE:
            return 1.0, None, metrics
        elif actual_height > 0:
            deviation = min(abs(actual_height - min_h), abs(actual_height - max_h))
            score = max(0.3, 1.0 - deviation / self.HEIGHT_TOLERANCE)
            return score, f"Adsorbate height {actual_height:.2f} Å outside expected range {expected_range}", metrics
        else:
            return 0.0, f"Adsorbate below surface: height = {actual_height:.2f} Å", metrics

    def _check_lattice(self, structure: Structure,
                       metal: str) -> Tuple[float, Optional[str]]:
        """Check if lattice constant is reasonable for the metal."""
        expected_a = LATTICE_CONSTANTS.get(metal, 3.9)

        lattice = structure.lattice
        actual_a = lattice.a
        actual_b = lattice.b

        # For supercells, expect multiples of base lattice constant
        possible_a = [
            expected_a,
            2 * expected_a,
            3 * expected_a,
            expected_a * np.sqrt(2),
            2 * expected_a * np.sqrt(2),
        ]

        min_deviation_a = min(abs(actual_a - pa) for pa in possible_a)
        min_deviation_b = min(abs(actual_b - pa) for pa in possible_a)
        min_deviation = min(min_deviation_a, min_deviation_b)

        if min_deviation < self.LATTICE_TOLERANCE:
            return 1.0, None
        elif min_deviation < self.LATTICE_TOLERANCE * 2:
            return 0.7, f"Lattice constant slightly off: deviation {min_deviation:.2f} Å"
        else:
            return 0.5, f"Lattice constant deviation: {min_deviation:.2f} Å"

    def _check_slab_structure(self, structure: Structure,
                               metal: str) -> Tuple[float, Optional[str], Dict]:
        """Check if slab has reasonable layer structure."""
        metrics = {}

        # Get z-coordinates of metal atoms
        metal_z = []
        for site in structure:
            if str(site.specie) == metal:
                metal_z.append(site.coords[2])

        if len(metal_z) < 4:
            metrics['num_metal_atoms'] = len(metal_z)
            return 0.3, f"Too few metal atoms: {len(metal_z)}", metrics

        # Count distinct layers (cluster z-coordinates)
        metal_z = sorted(metal_z)
        layers = []
        current_layer = [metal_z[0]]

        for z in metal_z[1:]:
            if z - current_layer[-1] < 0.5:  # Same layer
                current_layer.append(z)
            else:
                layers.append(current_layer)
                current_layer = [z]
        layers.append(current_layer)

        num_layers = len(layers)
        atoms_per_layer = [len(layer) for layer in layers]

        metrics['num_layers'] = num_layers
        metrics['atoms_per_layer'] = atoms_per_layer
        metrics['num_metal_atoms'] = len(metal_z)

        # Expect 3-6 layers for a reasonable slab
        if 3 <= num_layers <= 6:
            return 1.0, None, metrics
        elif 2 <= num_layers <= 8:
            return 0.7, f"Unusual number of layers: {num_layers}", metrics
        else:
            return 0.4, f"Unexpected slab structure: {num_layers} layers", metrics

    def _check_adsorbate_geometry(self, structure: Structure,
                                   metal: str,
                                   adsorbate: str) -> Tuple[float, Optional[str]]:
        """Check if adsorbate geometry is preserved (for multi-atom adsorbates)."""
        expected_atoms = ADSORBATE_ATOM_COUNTS.get(adsorbate, 1)

        if expected_atoms <= 1:
            return 1.0, None  # Single atom, no geometry to check

        # Get adsorbate atom positions
        adsorbate_coords = []
        for site in structure:
            if str(site.specie) != metal:
                adsorbate_coords.append(site.coords)

        if len(adsorbate_coords) < 2:
            return 1.0, None  # Not enough atoms to check geometry

        # Calculate internal distances
        distances = []
        for i in range(len(adsorbate_coords)):
            for j in range(i + 1, len(adsorbate_coords)):
                dist = np.linalg.norm(
                    np.array(adsorbate_coords[i]) - np.array(adsorbate_coords[j])
                )
                distances.append(dist)

        if not distances:
            return 1.0, None

        # Check for reasonable bond lengths (0.8 - 3.0 Å typical)
        min_dist = min(distances)
        max_dist = max(distances)

        if min_dist < 0.5:
            return 0.3, f"Adsorbate atoms too close: {min_dist:.2f} Å"
        elif min_dist < 0.8:
            return 0.6, f"Short adsorbate bond: {min_dist:.2f} Å"
        elif max_dist > 5.0:
            return 0.5, f"Adsorbate atoms far apart: {max_dist:.2f} Å"

        return 1.0, None

    def _check_adsorption_site(self, structure: Structure,
                                metal: str,
                                expected_site: str) -> Tuple[float, Optional[str]]:
        """Check if adsorbate is at the correct adsorption site.

        Validates:
        - top/ontop: adsorbate directly above a surface atom
        - bridge: adsorbate between two surface atoms
        - hollow/fcc/hcp: adsorbate in a hollow site (center of 3+ atoms)
        """
        # Get surface (metal) atoms and adsorbate atoms
        metal_coords = []
        metal_z = []
        adsorbate_coords = []

        for site in structure:
            if str(site.specie) == metal:
                metal_coords.append(site.coords[:2])  # x, y only
                metal_z.append(site.coords[2])
            else:
                adsorbate_coords.append(site.coords)

        if not metal_coords or not adsorbate_coords:
            return 0.5, "Could not identify surface/adsorbate atoms"

        # Find top surface layer atoms
        if not metal_z:
            return 0.5, "No metal atoms found"

        surface_top_z = max(metal_z)
        top_layer_threshold = 0.5  # Å

        top_surface_coords = []
        for i, z in enumerate(metal_z):
            if abs(z - surface_top_z) < top_layer_threshold:
                top_surface_coords.append(metal_coords[i])

        if len(top_surface_coords) < 1:
            return 0.5, "Could not identify top surface layer"

        # Get adsorbate binding position (lowest adsorbate atom)
        adsorbate_z = [c[2] for c in adsorbate_coords]
        binding_idx = np.argmin(adsorbate_z)
        binding_pos = np.array(adsorbate_coords[binding_idx][:2])  # x, y only

        # Get tolerance for this site type
        tolerance = self.SITE_TOLERANCES.get(expected_site.lower(), 0.2)

        # Convert to numpy arrays for calculations
        top_surface_coords = [np.array(c) for c in top_surface_coords]

        if expected_site.lower() in ['top', 'ontop']:
            # Should be directly above a surface atom
            min_dist = min(np.linalg.norm(binding_pos - sc) for sc in top_surface_coords)

            # Convert to fractional for comparison
            lattice_a = structure.lattice.a
            min_dist_frac = min_dist / lattice_a

            if min_dist_frac < tolerance:
                return 1.0, None
            elif min_dist_frac < tolerance * 2:
                return 0.7, f"Adsorbate slightly off top site (dist: {min_dist:.2f} Å)"
            else:
                return 0.4, f"Adsorbate not at top site (dist: {min_dist:.2f} Å)"

        elif expected_site.lower() == 'bridge':
            # Should be between two surface atoms
            min_bridge_dist = float('inf')

            for i, pos1 in enumerate(top_surface_coords):
                for pos2 in top_surface_coords[i+1:]:
                    midpoint = (pos1 + pos2) / 2
                    dist = np.linalg.norm(binding_pos - midpoint)
                    min_bridge_dist = min(min_bridge_dist, dist)

            lattice_a = structure.lattice.a
            min_dist_frac = min_bridge_dist / lattice_a

            if min_dist_frac < tolerance:
                return 1.0, None
            elif min_dist_frac < tolerance * 2:
                return 0.7, f"Adsorbate slightly off bridge site"
            else:
                return 0.4, f"Adsorbate not at bridge site"

        elif expected_site.lower() in ['hollow', 'fcc', 'hcp']:
            # Should be in a hollow site (not directly above any single atom)
            min_dist_to_atom = min(np.linalg.norm(binding_pos - sc) for sc in top_surface_coords)

            lattice_a = structure.lattice.a
            min_dist_frac = min_dist_to_atom / lattice_a

            # For hollow sites, should NOT be too close to any single atom
            if min_dist_frac > 0.1:  # Not directly on top
                # Check if it's roughly equidistant from multiple atoms (hollow characteristic)
                distances = [np.linalg.norm(binding_pos - sc) for sc in top_surface_coords]
                distances.sort()

                if len(distances) >= 3:
                    # Check if closest 3 atoms are roughly equidistant
                    d1, d2, d3 = distances[:3]
                    if max(d1, d2, d3) - min(d1, d2, d3) < 0.5:  # Within 0.5 Å
                        return 1.0, None
                    else:
                        return 0.7, "Hollow site geometry not ideal"

                return 0.8, None  # Acceptable hollow position
            else:
                return 0.4, f"Adsorbate too close to surface atom for hollow site"

        # Unknown site type - just check it's not overlapping
        return 0.6, f"Unknown site type: {expected_site}"

    def _check_adsorption_stability(self, structure: Structure,
                                     metal: str) -> Tuple[float, Optional[str], Dict]:
        """Check adsorption stability using MLIP.

        Args:
            structure: Adsorption structure (slab + adsorbate)
            metal: Metal element of the slab

        Returns:
            Tuple of (score, message, metrics)
        """
        try:
            from .mlip import MLIPEvaluator

            mlip = MLIPEvaluator(model_type=self.config.mlip_model_type)

            # Evaluate overall stability based on forces
            stability_score, stability_msg, metrics = mlip.evaluate_stability(structure)

            # For adsorption, also check if forces are reasonable
            max_force = metrics.get('max_force', 0.0)

            # Score based on maximum force (similar to stability check)
            if max_force < 0.1:
                score = 1.0
                message = f"Structure stable (max force: {max_force:.3f} eV/Å)"
            elif max_force < 0.3:
                score = 0.8
                message = f"Structure mostly stable (max force: {max_force:.3f} eV/Å)"
            elif max_force < 0.5:
                score = 0.6
                message = f"Moderate forces detected (max force: {max_force:.3f} eV/Å)"
            else:
                score = 0.3
                message = f"High forces - structure may be unstable (max force: {max_force:.3f} eV/Å)"

            return score, message, metrics

        except ImportError:
            return 1.0, None, {'skipped': f'{self.config.mlip_model_type} not available'}
        except Exception as e:
            return 0.5, f"MLIP evaluation failed: {e}", {}


# Convenience function for legacy compatibility
def evaluate_adsorption_structure(structure: Structure,
                                   task: Dict[str, Any],
                                   validation_level: int = 1) -> Tuple[float, List[str], List[str]]:
    """Legacy interface for adsorption evaluation.

    Args:
        structure: Generated structure
        task: Task specification
        validation_level: 1=basic, 2=intermediate, 3=advanced

    Returns:
        Tuple of (score, errors, warnings)
    """
    evaluator = AdsorptionEvaluator(validation_level=validation_level)
    result = evaluator.evaluate(structure, task)
    return result.to_tuple()
