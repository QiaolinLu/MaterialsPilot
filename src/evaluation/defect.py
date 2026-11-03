"""
Defect Structure Evaluator

Evaluates defect engineering structures (vacancies, substitutions, interstitials).
Migrated and enhanced from tasks/jarvis_defect/evaluator.py
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from pymatgen.core import Structure

from .base import BaseEvaluator, EvaluationResult
from .config import DefectConfig, DEFAULT_DEFECT_CONFIG


class DefectEvaluator(BaseEvaluator):
    """Evaluator for defect engineering structures.

    Validates:
    - Supercell size correctness
    - Defect concentration accuracy
    - Minimum defect-defect distance
    - Atom overlaps
    - Composition correctness
    """

    def __init__(self, validation_level: int = 1,
                 success_threshold: Optional[float] = None,
                 config: Optional[DefectConfig] = None):
        """Initialize defect evaluator.

        Args:
            validation_level: 1=basic, 2=intermediate, 3=advanced
            success_threshold: Minimum score for success (overrides config if provided)
            config: DefectConfig instance (uses default if not provided)
        """
        self.config = config or DEFAULT_DEFECT_CONFIG
        threshold = success_threshold if success_threshold is not None else self.config.success_threshold
        enable_relaxation = self.config.enable_relaxation_check
        super().__init__(threshold, enable_relaxation_check=enable_relaxation)
        self.validation_level = validation_level

        # Set thresholds from config
        self.MIN_INTERATOMIC_DISTANCE = self.config.min_interatomic_distance
        self.CLOSE_CONTACT_DISTANCE = self.config.close_contact_distance
        self.MIN_DEFECT_DISTANCE = self.config.min_defect_distance
        self.CONCENTRATION_TOLERANCE = self.config.concentration_tolerance
        self.SUPERCELL_TOLERANCE = self.config.supercell_tolerance

    def evaluate(self, structure: Structure, task: Dict[str, Any],
                 pristine_structure: Optional[Structure] = None) -> EvaluationResult:
        """Evaluate a defect structure.

        Args:
            structure: Generated defect structure
            task: Task specification with defect parameters
            pristine_structure: Original pristine structure (optional)

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
        defect_type = task_input.get('defect_type', 'vacancy')
        defect_element = task_input.get('defect_element', task_input.get('element', ''))
        substitute_element = task_input.get('substitute_element')
        supercell_size = task_input.get('supercell_size', [2, 2, 2])
        target_concentration = task_input.get('target_concentration', 0.0)
        expected_defects = task_input.get('expected_defects', 1)

        metrics['defect_type'] = defect_type
        metrics['defect_element'] = defect_element
        metrics['supercell_size'] = supercell_size

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

        # Check atom count (basic sanity check)
        atom_score, atom_msg = self._check_atom_count_defect(structure, supercell_size)
        scores.append(atom_score)
        metrics['atom_count'] = len(structure)
        if atom_msg:
            if atom_score == 0:
                errors.append(atom_msg)
            else:
                warnings.append(atom_msg)

        # ==================== Level 2: Intermediate Checks ====================

        if self.validation_level >= 2:
            if pristine_structure is not None:
                # Check supercell size with pristine structure
                supercell_score, supercell_msg, supercell_details = self._check_supercell_size(
                    structure, pristine_structure, supercell_size
                )
                scores.append(supercell_score)
                metrics['supercell_details'] = supercell_details
                if supercell_msg:
                    if supercell_score == 0:
                        errors.append(supercell_msg)
                    else:
                        warnings.append(supercell_msg)

                # Check composition with pristine structure
                comp_score, comp_msg, comp_details = self._check_defect_composition(
                    structure, pristine_structure, supercell_size,
                    defect_type, defect_element, substitute_element, expected_defects
                )
                scores.append(comp_score)
                metrics['composition_details'] = comp_details
                if comp_msg:
                    if comp_score == 0:
                        errors.append(comp_msg)
                    else:
                        warnings.append(comp_msg)
            else:
                # Fallback: Check composition without pristine structure
                comp_score, comp_msg = self._check_composition_without_pristine(
                    structure, defect_type, defect_element, substitute_element
                )
                scores.append(comp_score)
                metrics['composition'] = structure.composition.as_dict()
                if comp_msg:
                    if comp_score == 0:
                        errors.append(comp_msg)
                    else:
                        warnings.append(comp_msg)

                # Check expected atom count from task
                expected_atom_score, expected_atom_msg = self._check_expected_atom_count(
                    structure, task_input
                )
                scores.append(expected_atom_score)
                if expected_atom_msg:
                    if expected_atom_score < 0.5:
                        errors.append(expected_atom_msg)
                    else:
                        warnings.append(expected_atom_msg)

        # ==================== Level 3: Advanced Checks ====================

        if self.validation_level >= 3 and pristine_structure is not None:
            # Check defect concentration
            conc_score, conc_msg, conc_metrics = self._check_defect_concentration(
                structure, pristine_structure, supercell_size,
                defect_type, defect_element, substitute_element, target_concentration
            )
            scores.append(conc_score)
            metrics.update(conc_metrics)
            if conc_msg:
                if conc_score < 0.5:
                    errors.append(conc_msg)
                else:
                    warnings.append(conc_msg)

            # Check defect-defect distance (for substitution)
            if defect_type == 'substitution' and substitute_element:
                dist_score, dist_msg = self._check_defect_distance(
                    structure, substitute_element
                )
                scores.append(dist_score)
                if dist_msg:
                    warnings.append(dist_msg)

        # ==================== MLIP Energy Check ====================

        # Only run MLIP check if enabled and pristine structure is available
        if self.config.enable_mlip_check and pristine_structure is not None:
            mlip_score, mlip_msg, mlip_metrics = self._check_defect_energy(
                structure, pristine_structure, defect_element
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

    def _check_atom_count_defect(self, structure: Structure,
                                  supercell_size: List[int]) -> Tuple[float, Optional[str]]:
        """Check atom count for defect structure."""
        num_atoms = len(structure)
        min_expected = np.prod(supercell_size) * 2  # At least 2 atoms per unit cell

        if num_atoms >= min_expected:
            return 1.0, None
        elif num_atoms >= min_expected // 2:
            return 0.5, f"Low atom count: {num_atoms}"
        else:
            return 0.0, f"Too few atoms: {num_atoms}"

    def _check_composition_without_pristine(self, structure: Structure,
                                             defect_type: str,
                                             defect_element: str,
                                             substitute_element: Optional[str]) -> Tuple[float, Optional[str]]:
        """Check composition when pristine structure is not available.

        Validates:
        1. For vacancy: defect_element should be present (but reduced)
        2. For substitution: both defect_element and substitute_element should be present
        """
        composition = structure.composition.as_dict()
        elements = set(composition.keys())

        if defect_type == 'vacancy':
            # For vacancy, the defect element should still be present (unless all removed)
            if defect_element and defect_element in elements:
                return 1.0, None
            elif defect_element:
                # Element completely removed - might be valid for high concentration
                return 0.7, f"Defect element {defect_element} not found (may be valid for high vacancy concentration)"
            else:
                return 0.5, "No defect element specified"

        elif defect_type == 'substitution':
            if not substitute_element:
                return 0.5, "No substitute element specified"

            has_substitute = substitute_element in elements

            if has_substitute:
                return 1.0, None
            else:
                return 0.0, f"Substitute element {substitute_element} not found in structure"

        elif defect_type == 'interstitial':
            # For interstitial, the added element should be present
            interstitial_element = substitute_element or defect_element
            if interstitial_element and interstitial_element in elements:
                return 1.0, None
            else:
                return 0.5, f"Interstitial element not clearly identified"

        return 0.5, f"Unknown defect type: {defect_type}"

    def _check_expected_atom_count(self, structure: Structure,
                                    task_input: Dict[str, Any]) -> Tuple[float, Optional[str]]:
        """Check atom count against expected values from task specification."""
        actual_atoms = len(structure)

        # Try to get expected atom count from various task fields
        num_pristine = task_input.get('num_pristine_atoms', 0)
        supercell_size = task_input.get('supercell_size', [2, 2, 2])
        defect_type = task_input.get('defect_type', 'vacancy')
        expected_defects = task_input.get('expected_defects', 1)

        if num_pristine > 0:
            # Calculate expected atoms
            scale = np.prod(supercell_size)
            expected_atoms = num_pristine * scale

            if defect_type == 'vacancy':
                expected_atoms -= expected_defects
            elif defect_type == 'interstitial':
                expected_atoms += expected_defects
            # substitution doesn't change total count

            diff = abs(actual_atoms - expected_atoms)

            if diff == 0:
                return 1.0, None
            elif diff <= 2:
                return 0.8, f"Atom count slightly off: {actual_atoms} vs expected {expected_atoms}"
            elif diff <= expected_defects * 2:
                return 0.5, f"Atom count differs: {actual_atoms} vs expected {expected_atoms}"
            else:
                return 0.2, f"Atom count mismatch: {actual_atoms} vs expected {expected_atoms}"

        # Fallback: just check reasonable size
        min_atoms = np.prod(supercell_size) * 2
        if actual_atoms >= min_atoms:
            return 0.8, None
        else:
            return 0.4, f"Structure may be too small: {actual_atoms} atoms"

    def _check_supercell_size(self, structure: Structure,
                               pristine_structure: Structure,
                               expected_supercell: List[int]) -> Tuple[float, Optional[str], Dict]:
        """Verify supercell dimensions."""
        pristine_lattice = pristine_structure.lattice
        generated_lattice = structure.lattice

        expected_a = pristine_lattice.a * expected_supercell[0]
        expected_b = pristine_lattice.b * expected_supercell[1]
        expected_c = pristine_lattice.c * expected_supercell[2]

        details = {
            'expected': [expected_a, expected_b, expected_c],
            'actual': [generated_lattice.a, generated_lattice.b, generated_lattice.c],
        }

        # Check with tolerance
        a_error = abs(generated_lattice.a - expected_a) / expected_a
        b_error = abs(generated_lattice.b - expected_b) / expected_b
        c_error = abs(generated_lattice.c - expected_c) / expected_c

        details['errors'] = [a_error, b_error, c_error]

        if max(a_error, b_error, c_error) < self.SUPERCELL_TOLERANCE:
            return 1.0, None, details
        elif max(a_error, b_error, c_error) < self.SUPERCELL_TOLERANCE * 2:
            return 0.7, "Supercell dimensions slightly off", details
        else:
            return 0.0, "Supercell dimensions incorrect", details

    def _check_defect_composition(self, structure: Structure,
                                   pristine_structure: Structure,
                                   supercell_size: List[int],
                                   defect_type: str,
                                   defect_element: str,
                                   substitute_element: Optional[str],
                                   expected_defects: int) -> Tuple[float, Optional[str], Dict]:
        """Verify composition matches expected defect structure."""
        # Calculate expected composition
        pristine_comp = pristine_structure.composition.as_dict()
        scale = np.prod(supercell_size)

        expected_comp = {elem: count * scale for elem, count in pristine_comp.items()}

        if defect_type == 'vacancy':
            if defect_element in expected_comp:
                expected_comp[defect_element] -= expected_defects
        elif defect_type == 'substitution':
            if defect_element in expected_comp:
                expected_comp[defect_element] -= expected_defects
            if substitute_element:
                expected_comp[substitute_element] = expected_comp.get(substitute_element, 0) + expected_defects

        # Get actual composition
        actual_comp = structure.composition.as_dict()

        details = {
            'expected_composition': expected_comp,
            'actual_composition': actual_comp,
        }

        # Compare
        all_elements = set(expected_comp.keys()) | set(actual_comp.keys())
        differences = {}
        for elem in all_elements:
            expected = expected_comp.get(elem, 0)
            actual = actual_comp.get(elem, 0)
            if abs(expected - actual) > 0.5:
                differences[elem] = {'expected': expected, 'actual': actual}

        details['differences'] = differences

        if not differences:
            return 1.0, None, details
        elif len(differences) == 1:
            return 0.7, f"Minor composition difference: {differences}", details
        else:
            return 0.0, f"Composition mismatch: {differences}", details

    def _check_defect_concentration(self, structure: Structure,
                                     pristine_structure: Structure,
                                     supercell_size: List[int],
                                     defect_type: str,
                                     defect_element: str,
                                     substitute_element: Optional[str],
                                     target_concentration: float) -> Tuple[float, Optional[str], Dict]:
        """Check defect concentration accuracy."""
        # Count atoms in pristine supercell
        pristine_supercell_atoms = len(pristine_structure) * np.prod(supercell_size)

        # Count defect element in pristine
        pristine_defect_count = sum(
            1 for site in pristine_structure
            if str(site.specie) == defect_element
        ) * np.prod(supercell_size)

        # Count in generated structure
        generated_defect_count = sum(
            1 for site in structure
            if str(site.specie) == defect_element
        )

        if defect_type == 'vacancy':
            n_defects = pristine_defect_count - generated_defect_count
        else:  # substitution
            substitute_count = sum(
                1 for site in structure
                if str(site.specie) == substitute_element
            ) if substitute_element else 0
            pristine_substitute = sum(
                1 for site in pristine_structure
                if str(site.specie) == substitute_element
            ) * np.prod(supercell_size) if substitute_element else 0
            n_defects = substitute_count - pristine_substitute

        actual_concentration = n_defects / pristine_supercell_atoms if pristine_supercell_atoms > 0 else 0

        metrics = {
            'actual_concentration': actual_concentration,
            'target_concentration': target_concentration,
            'n_defects': int(n_defects),
            'concentration_error': abs(actual_concentration - target_concentration),
        }

        conc_error = abs(actual_concentration - target_concentration)

        if conc_error <= self.CONCENTRATION_TOLERANCE:
            return 1.0, None, metrics
        elif conc_error <= self.CONCENTRATION_TOLERANCE * 2:
            return 0.7, f"Concentration slightly off: {actual_concentration:.3f} vs {target_concentration:.3f}", metrics
        else:
            return 0.3, f"Concentration error: {actual_concentration:.3f} vs {target_concentration:.3f}", metrics

    def _check_defect_distance(self, structure: Structure,
                                substitute_element: str) -> Tuple[float, Optional[str]]:
        """Check minimum distance between defect sites."""
        defect_positions = []
        for site in structure:
            if str(site.specie) == substitute_element:
                defect_positions.append(site.coords)

        if len(defect_positions) < 2:
            return 1.0, None

        # Calculate pairwise distances
        min_dist = float('inf')
        for i in range(len(defect_positions)):
            for j in range(i + 1, len(defect_positions)):
                dist = np.linalg.norm(
                    np.array(defect_positions[i]) - np.array(defect_positions[j])
                )
                min_dist = min(min_dist, dist)

        if min_dist >= self.MIN_DEFECT_DISTANCE:
            return 1.0, None
        elif min_dist >= self.MIN_DEFECT_DISTANCE / 2:
            return 0.7, f"Defects close together: {min_dist:.2f} Å"
        else:
            return 0.4, f"Defects too close: {min_dist:.2f} Å"

    def _check_defect_energy(self, structure: Structure,
                             pristine_structure: Structure,
                             defect_element: str) -> Tuple[float, Optional[str], Dict]:
        """Check defect formation energy using MLIP.

        Args:
            structure: Defect structure
            pristine_structure: Pristine structure
            defect_element: Element involved in defect

        Returns:
            Tuple of (score, message, metrics)
        """
        try:
            from .mlip import MLIPEvaluator

            mlip = MLIPEvaluator(model_type=self.config.mlip_model_type)

            # Calculate formation energy
            formation_energy, metrics = mlip.calculate_formation_energy(
                defect_structure=structure,
                pristine_structure=pristine_structure,
                removed_element=defect_element,
                chemical_potential=0.0  # Simplified: use 0 for relative comparison
            )

            # Score based on formation energy thresholds
            if formation_energy < self.config.formation_energy_excellent:
                score = 1.0
                message = f"Formation energy: {formation_energy:.2f} eV (excellent)"
            elif formation_energy < self.config.formation_energy_acceptable:
                score = 0.7
                message = f"Formation energy: {formation_energy:.2f} eV (acceptable)"
            else:
                score = 0.3
                message = f"Formation energy: {formation_energy:.2f} eV (high - may be unstable)"

            return score, message, metrics

        except ImportError:
            return 1.0, None, {'skipped': f'{self.config.mlip_model_type} not available'}
        except Exception as e:
            return 0.5, f"MLIP evaluation failed: {e}", {}


# Convenience function for legacy compatibility
def evaluate_defect_structure(structure: Structure,
                               task: Dict[str, Any],
                               pristine_structure: Optional[Structure] = None,
                               validation_level: int = 1) -> Tuple[float, List[str], List[str]]:
    """Legacy interface for defect evaluation.

    Args:
        structure: Generated structure
        task: Task specification
        pristine_structure: Original pristine structure
        validation_level: 1=basic, 2=intermediate, 3=advanced

    Returns:
        Tuple of (score, errors, warnings)
    """
    evaluator = DefectEvaluator(validation_level=validation_level)
    result = evaluator.evaluate(structure, task, pristine_structure)
    return result.to_tuple()
