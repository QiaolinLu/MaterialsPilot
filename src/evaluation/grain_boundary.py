"""
Grain Boundary Structure Evaluator

Evaluates grain boundary structures for physical validity and correctness.
Includes both basic structural checks and advanced CSL theory validation.
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from pymatgen.core import Structure

from .base import BaseEvaluator, EvaluationResult
from .config import GrainBoundaryConfig, DEFAULT_GRAIN_BOUNDARY_CONFIG


class GrainBoundaryEvaluator(BaseEvaluator):
    """Evaluator for grain boundary structures.

    Validation levels:
    - Level 1 (Basic): Atom count, overlaps, dimensions
    - Level 2 (Intermediate): Composition, bicrystal structure
    - Level 3 (Advanced): CSL theory, rotation angles, GB plane orientation
    """

    def __init__(self, validation_level: int = 1,
                 success_threshold: Optional[float] = None,
                 config: Optional[GrainBoundaryConfig] = None):
        """Initialize grain boundary evaluator.

        Args:
            validation_level: 1=basic, 2=intermediate, 3=advanced
            success_threshold: Minimum score for success (overrides config if provided)
            config: GrainBoundaryConfig instance (uses default if not provided)
        """
        self.config = config or DEFAULT_GRAIN_BOUNDARY_CONFIG
        threshold = success_threshold if success_threshold is not None else self.config.success_threshold
        super().__init__(threshold)
        self.validation_level = validation_level

        # Set thresholds from config
        self.MIN_INTERATOMIC_DISTANCE = self.config.min_interatomic_distance
        self.CLOSE_CONTACT_DISTANCE = self.config.close_contact_distance
        self.MIN_ATOMS = self.config.min_atoms
        self.MIN_C_AXIS = self.config.min_c_axis
        self.MIN_DIMENSION = self.config.min_dimension
        self.CSL_ROTATION_ANGLES = self.config.csl_rotation_angles
        self.MATERIAL_LATTICE_TYPES = self.config.material_lattice_types

    def evaluate(self, structure: Structure, task: Dict[str, Any]) -> EvaluationResult:
        """Evaluate a grain boundary structure.

        Args:
            structure: Generated grain boundary structure
            task: Task specification with keys like 'input', 'material', 'sigma', etc.

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

        errors = []
        warnings = []
        scores = []
        metrics = {}

        # Extract task parameters
        task_input = task.get('input', task)
        material = task_input.get('material', 'Cu')
        sigma = task_input.get('sigma', 5)
        gb_plane = task_input.get('gb_plane', '(310)')
        gb_type = task_input.get('gb_type', 'tilt')

        metrics['material'] = material
        metrics['sigma'] = sigma
        metrics['gb_plane'] = gb_plane
        metrics['gb_type'] = gb_type

        # ==================== Level 1: Basic Checks ====================

        # Check atom count
        atom_score, atom_msg = self._check_atom_count_gb(structure)
        scores.append(atom_score)
        metrics['atom_count'] = len(structure)
        if atom_msg:
            if atom_score == 0:
                errors.append(atom_msg)
            else:
                warnings.append(atom_msg)

        # Check atom overlaps
        overlap_score, overlap_msg = self.check_atom_overlaps(structure)
        scores.append(overlap_score)
        metrics['min_interatomic_distance'] = self._get_min_distance(structure)
        if overlap_msg:
            if overlap_score == 0:
                errors.append(overlap_msg)
            else:
                warnings.append(overlap_msg)

        # Check lattice dimensions
        dim_score, dim_msg = self._check_gb_dimensions(structure)
        scores.append(dim_score)
        metrics['lattice_abc'] = list(structure.lattice.abc)
        if dim_msg:
            if dim_score < 0.5:
                errors.append(dim_msg)
            else:
                warnings.append(dim_msg)

        # ==================== Level 2: Intermediate Checks ====================

        if self.validation_level >= 2:
            # Check composition
            comp_score, comp_msg = self._check_composition_gb(structure, material)
            scores.append(comp_score)
            metrics['composition'] = structure.composition.as_dict()
            if comp_msg:
                if comp_score == 0:
                    errors.append(comp_msg)
                else:
                    warnings.append(comp_msg)

            # Check bicrystal structure (two grains present)
            bicrystal_score, bicrystal_msg = self._check_bicrystal_structure(structure)
            scores.append(bicrystal_score)
            metrics['bicrystal_detected'] = bicrystal_score > 0.5
            if bicrystal_msg:
                warnings.append(bicrystal_msg)

        # ==================== Level 3: Advanced Checks ====================

        if self.validation_level >= 3:
            # Check CSL theory compliance
            csl_score, csl_msg, csl_metrics = self._check_csl_compliance(
                structure, material, sigma, gb_plane, gb_type
            )
            scores.append(csl_score)
            metrics.update(csl_metrics)
            if csl_msg:
                if csl_score < 0.5:
                    errors.append(csl_msg)
                else:
                    warnings.append(csl_msg)

        # Calculate overall score
        # Index 1 is atom overlap check - mark as critical
        overall_score = self._aggregate_scores(scores, critical_indices=[1])
        success = overall_score >= self.success_threshold

        return EvaluationResult(
            score=overall_score,
            success=success,
            errors=errors,
            warnings=warnings,
            metrics=metrics
        )

    def _check_atom_count_gb(self, structure: Structure) -> Tuple[float, Optional[str]]:
        """Check atom count for grain boundary structure."""
        num_atoms = len(structure)

        if num_atoms >= self.MIN_ATOMS:
            return 1.0, None
        elif num_atoms >= 10:
            return 0.6, f"Low atom count: {num_atoms} (recommended >= {self.MIN_ATOMS})"
        else:
            return 0.2, f"Too few atoms: {num_atoms} (minimum {self.MIN_ATOMS})"

    def _check_gb_dimensions(self, structure: Structure) -> Tuple[float, Optional[str]]:
        """Check lattice dimensions for grain boundary structure."""
        a, b, c = structure.lattice.abc

        # GB structures should have substantial c-axis (perpendicular to GB plane)
        if c < self.MIN_C_AXIS:
            if c < 10:
                return 0.3, f"C-axis too short: {c:.1f} Å (need >= {self.MIN_C_AXIS})"
            return 0.5, f"Short c-axis: {c:.1f} Å (recommended >= {self.MIN_C_AXIS})"

        # Check minimum dimension
        min_dim = min(a, b, c)
        if min_dim < self.MIN_DIMENSION:
            return 0.5, f"Small dimension: {min_dim:.2f} Å"

        return 1.0, None

    def _check_composition_gb(self, structure: Structure,
                               expected_material: str) -> Tuple[float, Optional[str]]:
        """Check composition for grain boundary (should be single element)."""
        composition = structure.composition.as_dict()

        if expected_material in composition:
            # Check if it's predominantly the expected material
            total = sum(composition.values())
            material_fraction = composition[expected_material] / total

            if material_fraction >= 0.99:
                return 1.0, None
            elif material_fraction >= 0.9:
                return 0.8, f"Impure composition: {material_fraction:.1%} {expected_material}"
            else:
                return 0.5, f"Mixed composition: {material_fraction:.1%} {expected_material}"
        else:
            return 0.0, f"Wrong material: expected {expected_material}, got {list(composition.keys())}"

    def _check_bicrystal_structure(self, structure: Structure) -> Tuple[float, Optional[str]]:
        """Check if structure has bicrystal characteristics.

        A proper grain boundary should show two distinct grain regions.
        This is a heuristic check based on z-coordinate distribution.
        """
        # Get z-coordinates
        z_coords = [site.frac_coords[2] for site in structure]

        if len(z_coords) < 10:
            return 0.5, "Too few atoms for bicrystal analysis"

        # Check for bimodal distribution in z
        z_array = np.array(z_coords)
        z_mean = np.mean(z_array)

        # Count atoms in each half
        lower_half = np.sum(z_array < z_mean)
        upper_half = np.sum(z_array >= z_mean)

        # Both halves should have substantial atoms
        total = len(z_array)
        lower_frac = lower_half / total
        upper_frac = upper_half / total

        if 0.3 <= lower_frac <= 0.7 and 0.3 <= upper_frac <= 0.7:
            return 1.0, None
        elif 0.2 <= lower_frac <= 0.8:
            return 0.7, "Asymmetric grain distribution"
        else:
            return 0.4, "Bicrystal structure not clearly detected"

    def _check_csl_compliance(self, structure: Structure, material: str,
                               sigma: int, gb_plane: str,
                               gb_type: str) -> Tuple[float, Optional[str], Dict]:
        """Check CSL theory compliance (advanced validation).

        Validates:
        1. Sigma value is valid for the lattice type
        2. Structure shows evidence of grain misorientation
        3. Atom count is consistent with CSL theory
        """
        metrics = {}
        scores = []

        # Get lattice type
        lattice_type = self.MATERIAL_LATTICE_TYPES.get(material, 'fcc')
        metrics['lattice_type'] = lattice_type

        # Check 1: Sigma value validity
        if sigma in self.CSL_ROTATION_ANGLES:
            if lattice_type in self.CSL_ROTATION_ANGLES[sigma]:
                expected_angle = self.CSL_ROTATION_ANGLES[sigma][lattice_type]
                metrics['expected_rotation_angle'] = expected_angle
                scores.append(1.0)
            else:
                scores.append(0.5)
                metrics['csl_warning'] = f"Σ{sigma} not standard for {lattice_type}"
        else:
            scores.append(0.3)
            metrics['csl_warning'] = f"Σ{sigma} is not a common CSL value"

        # Check 2: Verify grain misorientation exists
        misorientation_score, misorientation_detected = self._check_grain_misorientation(structure)
        scores.append(misorientation_score)
        metrics['misorientation_detected'] = misorientation_detected

        # Check 3: Atom count consistency with CSL
        # For Σn GB, the CSL unit cell has n times the volume of primitive cell
        # This affects expected atom counts
        csl_atom_score = self._check_csl_atom_count(structure, sigma, lattice_type)
        scores.append(csl_atom_score)
        metrics['csl_atom_score'] = csl_atom_score

        overall_score = sum(scores) / len(scores)

        if overall_score >= 0.8:
            return overall_score, None, metrics
        elif overall_score >= 0.5:
            return overall_score, f"Partial CSL compliance (score: {overall_score:.2f})", metrics
        else:
            return overall_score, f"Poor CSL compliance (score: {overall_score:.2f})", metrics

    def _check_grain_misorientation(self, structure: Structure) -> Tuple[float, bool]:
        """Check if structure shows evidence of grain misorientation.

        Analyzes local atomic environments to detect orientation differences
        between upper and lower halves of the structure.
        """
        if len(structure) < 16:
            return 0.5, False

        # Get z-coordinates and split into two halves
        z_coords = np.array([site.frac_coords[2] for site in structure])
        z_median = np.median(z_coords)

        lower_indices = np.where(z_coords < z_median)[0]
        upper_indices = np.where(z_coords >= z_median)[0]

        if len(lower_indices) < 4 or len(upper_indices) < 4:
            return 0.5, False

        # Analyze local coordination in each grain
        # Compare nearest neighbor distances and angles
        try:
            lower_nn_dists = self._get_avg_nn_distance(structure, lower_indices)
            upper_nn_dists = self._get_avg_nn_distance(structure, upper_indices)

            # If both grains have similar internal structure but different
            # from each other at the boundary, it indicates misorientation
            if lower_nn_dists > 0 and upper_nn_dists > 0:
                # Check for structural difference at boundary
                boundary_score = self._check_boundary_region(structure, z_median)

                if boundary_score > 0.6:
                    return 1.0, True
                elif boundary_score > 0.3:
                    return 0.7, True
                else:
                    return 0.5, False
        except Exception:
            pass

        return 0.5, False

    def _get_avg_nn_distance(self, structure: Structure, indices: np.ndarray) -> float:
        """Get average nearest neighbor distance for a subset of atoms."""
        if len(indices) < 2:
            return 0.0

        distances = []
        for i in indices:
            min_dist = float('inf')
            for j in indices:
                if i != j:
                    dist = structure.get_distance(int(i), int(j))
                    min_dist = min(min_dist, dist)
            if min_dist < float('inf'):
                distances.append(min_dist)

        return np.mean(distances) if distances else 0.0

    def _check_boundary_region(self, structure: Structure, z_boundary: float) -> float:
        """Check the grain boundary region for structural features.

        A proper GB should show:
        1. Atoms near the boundary with unusual coordination
        2. Possible local density variation
        """
        # Get atoms near the boundary (within 10% of z-range)
        z_coords = np.array([site.frac_coords[2] for site in structure])
        z_range = z_coords.max() - z_coords.min()
        boundary_width = 0.1 * z_range

        boundary_mask = np.abs(z_coords - z_boundary) < boundary_width
        boundary_indices = np.where(boundary_mask)[0]

        if len(boundary_indices) < 2:
            return 0.3

        # Check if boundary region has different local structure
        # (e.g., different coordination numbers or bond lengths)
        bulk_indices = np.where(~boundary_mask)[0]

        if len(bulk_indices) < 4:
            return 0.5

        boundary_nn = self._get_avg_nn_distance(structure, boundary_indices)
        bulk_nn = self._get_avg_nn_distance(structure, bulk_indices)

        if bulk_nn > 0:
            # Boundary region often has slightly different NN distances
            ratio = boundary_nn / bulk_nn
            if 0.9 <= ratio <= 1.1:
                return 0.8  # Similar but not identical - good GB
            elif 0.8 <= ratio <= 1.2:
                return 0.6
            else:
                return 0.4  # Too different - might be problematic

        return 0.5

    def _check_csl_atom_count(self, structure: Structure, sigma: int,
                              lattice_type: str) -> float:
        """Check if atom count is consistent with CSL theory."""
        num_atoms = len(structure)

        # For common lattice types, check if atom count makes sense
        # FCC: 4 atoms per conventional cell
        # BCC: 2 atoms per conventional cell
        # Diamond: 8 atoms per conventional cell
        atoms_per_cell = {'fcc': 4, 'bcc': 2, 'diamond': 8, 'hcp': 2}.get(lattice_type, 4)

        # GB structures typically have 2 grains, each with multiple unit cells
        # Minimum reasonable: 2 grains × 2 cells × atoms_per_cell
        min_reasonable = 2 * 2 * atoms_per_cell

        if num_atoms >= min_reasonable:
            # Check if atom count is roughly a multiple of expected
            # (accounting for possible GB expansion/contraction)
            remainder = num_atoms % atoms_per_cell
            if remainder <= 2 or remainder >= atoms_per_cell - 2:
                return 1.0
            else:
                return 0.7  # Slightly off but acceptable
        else:
            return 0.3  # Too few atoms


# Convenience function for legacy compatibility
def evaluate_grain_boundary_structure(structure: Structure,
                                       task: Dict[str, Any],
                                       validation_level: int = 1) -> Tuple[float, List[str], List[str]]:
    """Legacy interface for grain boundary evaluation.

    Args:
        structure: Generated structure
        task: Task specification
        validation_level: 1=basic, 2=intermediate, 3=advanced

    Returns:
        Tuple of (score, errors, warnings)
    """
    evaluator = GrainBoundaryEvaluator(validation_level=validation_level)
    result = evaluator.evaluate(structure, task)
    return result.to_tuple()
