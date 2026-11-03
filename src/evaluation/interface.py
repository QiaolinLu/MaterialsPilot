"""
Interface/Heterostructure Structure Evaluator

Evaluates heterostructure interface structures for physical validity.
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from pymatgen.core import Structure

from .base import BaseEvaluator, EvaluationResult
from .config import InterfaceConfig, DEFAULT_INTERFACE_CONFIG


class InterfaceEvaluator(BaseEvaluator):
    """Evaluator for heterostructure interface structures.

    Validates:
    - Atom count and composition
    - Atom overlaps
    - Interface geometry (vacuum layer, dimensions)
    - Material layer structure
    """

    def __init__(self, validation_level: int = 1,
                 success_threshold: Optional[float] = None,
                 config: Optional[InterfaceConfig] = None):
        """Initialize interface evaluator.

        Args:
            validation_level: 1=basic, 2=intermediate, 3=advanced
            success_threshold: Minimum score for success (overrides config if provided)
            config: InterfaceConfig instance (uses default if not provided)
        """
        self.config = config or DEFAULT_INTERFACE_CONFIG
        threshold = success_threshold if success_threshold is not None else self.config.success_threshold
        super().__init__(threshold)
        self.validation_level = validation_level

        # Set thresholds from config
        self.MIN_INTERATOMIC_DISTANCE = self.config.min_interatomic_distance
        self.CLOSE_CONTACT_DISTANCE = self.config.close_contact_distance
        self.MIN_ATOMS = self.config.min_atoms
        self.MIN_C_AXIS = self.config.min_c_axis
        self.MIN_VACUUM = self.config.min_vacuum
        self.MIN_DIMENSION = self.config.min_dimension
        self.METAL_ELEMENTS = self.config.metal_elements
        self.OXIDE_ELEMENTS = self.config.oxide_elements
        self.SEMICONDUCTOR_ELEMENTS = self.config.semiconductor_elements

    def evaluate(self, structure: Structure, task: Dict[str, Any]) -> EvaluationResult:
        """Evaluate an interface structure.

        Args:
            structure: Generated interface structure
            task: Task specification

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

        # Extract task parameters from standard schema
        # Schema: task.category, task.input.film_material, task.input.substrate_material
        task_input = task.get('input', {})

        # Interface type from top-level category
        interface_type = task.get('category', 'metal_oxide')

        # Materials from input section
        material_a = task_input.get('film_material', '')
        material_b = task_input.get('substrate_material', '')

        metrics['interface_type'] = interface_type
        metrics['material_a'] = material_a
        metrics['material_b'] = material_b

        # ==================== Level 1: Basic Checks ====================

        # Check atom count
        atom_score, atom_msg = self._check_atom_count_interface(structure)
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
        dim_score, dim_msg = self._check_interface_dimensions(structure)
        scores.append(dim_score)
        metrics['lattice_abc'] = list(structure.lattice.abc)
        if dim_msg:
            if dim_score < 0.5:
                errors.append(dim_msg)
            else:
                warnings.append(dim_msg)

        # ==================== Level 2: Intermediate Checks ====================

        if self.validation_level >= 2:
            # Check composition for interface type
            comp_score, comp_msg = self._check_interface_composition(
                structure, interface_type, material_a, material_b
            )
            scores.append(comp_score)
            metrics['composition'] = structure.composition.as_dict()
            if comp_msg:
                if comp_score == 0:
                    errors.append(comp_msg)
                else:
                    warnings.append(comp_msg)

            # Check layer structure
            layer_score, layer_msg = self._check_layer_structure(structure)
            scores.append(layer_score)
            metrics['layer_structure_detected'] = layer_score > 0.5
            if layer_msg:
                warnings.append(layer_msg)

        # ==================== Level 3: Advanced Checks ====================

        if self.validation_level >= 3:
            # Check interface region
            interface_score, interface_msg, interface_metrics = self._check_interface_region(
                structure, interface_type
            )
            scores.append(interface_score)
            metrics.update(interface_metrics)
            if interface_msg:
                if interface_score < 0.5:
                    errors.append(interface_msg)
                else:
                    warnings.append(interface_msg)

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

    def _check_atom_count_interface(self, structure: Structure) -> Tuple[float, Optional[str]]:
        """Check atom count for interface structure."""
        num_atoms = len(structure)

        if num_atoms >= self.MIN_ATOMS:
            return 1.0, None
        elif num_atoms >= 10:
            return 0.6, f"Low atom count: {num_atoms} (recommended >= {self.MIN_ATOMS})"
        else:
            return 0.2, f"Too few atoms: {num_atoms} (minimum {self.MIN_ATOMS})"

    def _check_interface_dimensions(self, structure: Structure) -> Tuple[float, Optional[str]]:
        """Check lattice dimensions for interface structure."""
        a, b, c = structure.lattice.abc

        # Interface structures should have substantial c-axis
        if c < self.MIN_C_AXIS:
            if c < 10:
                return 0.3, f"C-axis too short: {c:.1f} Å (need >= {self.MIN_C_AXIS})"
            return 0.5, f"Short c-axis: {c:.1f} Å (recommended >= {self.MIN_C_AXIS})"

        # Check minimum dimension
        min_dim = min(a, b)
        if min_dim < self.MIN_DIMENSION:
            return 0.5, f"Small lateral dimension: {min_dim:.2f} Å"

        return 1.0, None

    def _check_interface_composition(self, structure: Structure,
                                      interface_type: str,
                                      material_a: str,
                                      material_b: str) -> Tuple[float, Optional[str]]:
        """Check composition for interface type."""
        composition = structure.composition.as_dict()
        elements = set(composition.keys())

        if interface_type == 'metal_oxide':
            # Should have metal and oxygen
            has_metal = bool(elements & self.METAL_ELEMENTS)
            has_oxygen = 'O' in elements

            if has_metal and has_oxygen:
                return 1.0, None
            elif has_metal or has_oxygen:
                return 0.5, f"Incomplete metal-oxide interface: {elements}"
            else:
                return 0.0, f"Missing metal and oxygen: {elements}"

        elif interface_type == 'metal_semiconductor':
            has_metal = bool(elements & self.METAL_ELEMENTS)
            has_semiconductor = bool(elements & self.SEMICONDUCTOR_ELEMENTS)

            if has_metal and has_semiconductor:
                return 1.0, None
            elif has_metal or has_semiconductor:
                return 0.5, f"Incomplete metal-semiconductor interface: {elements}"
            else:
                return 0.0, f"Missing expected elements: {elements}"

        else:
            # Generic check - should have at least 2 different element types
            if len(elements) >= 2:
                return 1.0, None
            else:
                return 0.5, f"Single element interface: {elements}"

    def _check_layer_structure(self, structure: Structure) -> Tuple[float, Optional[str]]:
        """Check if structure has layered characteristics."""
        # Get z-coordinates
        z_coords = [site.frac_coords[2] for site in structure]

        if len(z_coords) < 10:
            return 0.5, "Too few atoms for layer analysis"

        z_array = np.array(z_coords)

        # Check for distinct layers by looking at z-distribution
        z_sorted = np.sort(z_array)
        z_gaps = np.diff(z_sorted)

        # Large gaps indicate layer separation or vacuum
        large_gaps = np.sum(z_gaps > 0.1)  # Fractional coordinate gap

        if large_gaps >= 1:
            return 1.0, None
        else:
            return 0.6, "Layer structure not clearly detected"

    def _check_interface_region(self, structure: Structure,
                                 interface_type: str) -> Tuple[float, Optional[str], Dict]:
        """Check interface region characteristics (advanced)."""
        metrics = {}

        # Analyze element distribution along z-axis
        z_coords = np.array([site.frac_coords[2] for site in structure])
        elements = [site.species_string for site in structure]

        # Group by element
        element_z = {}
        for z, elem in zip(z_coords, elements):
            if elem not in element_z:
                element_z[elem] = []
            element_z[elem].append(z)

        # Check if different elements are in different z-regions
        element_means = {elem: np.mean(zs) for elem, zs in element_z.items()}
        metrics['element_z_means'] = element_means

        if len(element_means) >= 2:
            z_values = list(element_means.values())
            z_separation = max(z_values) - min(z_values)
            metrics['element_z_separation'] = z_separation

            if z_separation > 0.2:  # Good separation
                return 1.0, None, metrics
            elif z_separation > 0.1:
                return 0.7, "Moderate element separation", metrics
            else:
                return 0.5, "Elements not well separated", metrics
        else:
            return 0.5, "Single element type", metrics


# Convenience function for legacy compatibility
def evaluate_interface_structure(structure: Structure,
                                  task: Dict[str, Any],
                                  validation_level: int = 1) -> Tuple[float, List[str], List[str]]:
    """Legacy interface for interface evaluation.

    Args:
        structure: Generated structure
        task: Task specification
        validation_level: 1=basic, 2=intermediate, 3=advanced

    Returns:
        Tuple of (score, errors, warnings)
    """
    evaluator = InterfaceEvaluator(validation_level=validation_level)
    result = evaluator.evaluate(structure, task)
    return result.to_tuple()
