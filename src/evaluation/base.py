"""
Base Evaluator Module

Provides abstract base class and common utilities for structure evaluation.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
import math
import os
import numpy as np
from pymatgen.core import Structure

# Lazy import for relaxation checker to avoid hard dependency on ASE
_relaxation_checker = None


def _distance_matrix_atom_limit() -> int:
    """Maximum atoms for full periodic distance matrices in benchmark checks."""
    return int(os.getenv("MATERIALSPILOT_MAX_DISTANCE_MATRIX_ATOMS", "512"))


@dataclass
class EvaluationResult:
    """Standardized evaluation result."""
    score: float  # Overall score 0-1
    success: bool  # Whether evaluation passed (score > threshold)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)

    def to_tuple(self) -> Tuple[float, List[str], List[str]]:
        """Convert to legacy tuple format (score, errors, warnings)."""
        return (self.score, self.errors, self.warnings)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'score': self.score,
            'success': self.success,
            'errors': self.errors,
            'warnings': self.warnings,
            'metrics': self.metrics,
        }


class BaseEvaluator(ABC):
    """Abstract base class for structure evaluators.

    All evaluators should inherit from this class and implement
    the evaluate() method.
    """

    # Default thresholds - can be overridden by subclasses
    MIN_INTERATOMIC_DISTANCE = 1.0  # Angstroms
    CLOSE_CONTACT_DISTANCE = 1.5  # Angstroms
    SUCCESS_THRESHOLD = 0.5  # Score threshold for success

    def __init__(self, success_threshold: float = 0.5,
                 enable_relaxation_check: bool = False):
        """Initialize evaluator.

        Args:
            success_threshold: Minimum score to consider evaluation successful
            enable_relaxation_check: Enable fast force field relaxation check
        """
        self.success_threshold = success_threshold
        self.enable_relaxation_check = enable_relaxation_check

    @abstractmethod
    def evaluate(self, structure: Structure, task: Dict[str, Any]) -> EvaluationResult:
        """Evaluate a generated structure against task requirements.

        Args:
            structure: Generated pymatgen Structure
            task: Task specification dictionary

        Returns:
            EvaluationResult with score, errors, warnings, and metrics
        """
        pass

    def evaluate_legacy(self, structure: Structure, task: Dict[str, Any]) -> Tuple[float, List[str], List[str]]:
        """Legacy interface returning (score, errors, warnings) tuple.

        For backward compatibility with existing benchmark scripts.
        """
        result = self.evaluate(structure, task)
        return result.to_tuple()

    # ==================== Common Validation Methods ====================

    def check_atom_overlaps(self, structure: Structure) -> Tuple[float, Optional[str]]:
        """Check for overlapping atoms.

        Args:
            structure: Structure to check

        Returns:
            Tuple of (score, error_message or None)
            - score: 1.0 if no overlaps, 0.7 if close contacts, 0.0 if overlaps
        """
        min_dist = self._get_min_distance(
            structure,
            cutoff=max(self.MIN_INTERATOMIC_DISTANCE, self.CLOSE_CONTACT_DISTANCE),
        )

        if min_dist < self.MIN_INTERATOMIC_DISTANCE:
            return 0.0, f"Atom overlap: min distance = {min_dist:.2f} Å"
        elif min_dist < self.CLOSE_CONTACT_DISTANCE:
            return 0.7, f"Close contact: min distance = {min_dist:.2f} Å (warning)"
        return 1.0, None

    def check_atom_count(self, structure: Structure,
                         min_atoms: int = 0,
                         max_atoms: Optional[int] = None,
                         expected_atoms: Optional[int] = None,
                         tolerance: float = 0.1) -> Tuple[float, Optional[str]]:
        """Check atom count against requirements.

        Args:
            structure: Structure to check
            min_atoms: Minimum required atoms
            max_atoms: Maximum allowed atoms (optional)
            expected_atoms: Expected atom count (optional, for ratio check)
            tolerance: Tolerance for expected_atoms ratio check

        Returns:
            Tuple of (score, error_message or None)
        """
        actual = len(structure)

        if expected_atoms is not None:
            ratio = actual / expected_atoms
            if 1 - tolerance <= ratio <= 1 + tolerance:
                return 1.0, None
            elif 0.8 <= ratio <= 1.2:
                return 0.6, f"Atom count mismatch: {actual} vs expected {expected_atoms}"
            else:
                return 0.0, f"Atom count error: {actual} vs expected {expected_atoms}"

        if actual < min_atoms:
            return 0.0, f"Too few atoms: {actual} < {min_atoms}"

        if max_atoms is not None and actual > max_atoms:
            return 0.0, f"Too many atoms: {actual} > {max_atoms}"

        return 1.0, None

    def check_composition(self, structure: Structure,
                          required_elements: List[str]) -> Tuple[float, Optional[str]]:
        """Check that structure contains required elements.

        Args:
            structure: Structure to check
            required_elements: List of element symbols that must be present

        Returns:
            Tuple of (score, error_message or None)
        """
        composition = structure.composition.as_dict()
        present_elements = set(composition.keys())
        required_set = set(required_elements)

        missing = required_set - present_elements
        if missing:
            return 0.0, f"Missing elements: {missing}"

        return 1.0, None

    def check_lattice_dimensions(self, structure: Structure,
                                  min_a: float = 0, min_b: float = 0, min_c: float = 0,
                                  max_a: Optional[float] = None,
                                  max_b: Optional[float] = None,
                                  max_c: Optional[float] = None) -> Tuple[float, Optional[str]]:
        """Check lattice dimensions.

        Args:
            structure: Structure to check
            min_a, min_b, min_c: Minimum lattice parameters
            max_a, max_b, max_c: Maximum lattice parameters (optional)

        Returns:
            Tuple of (score, error_message or None)
        """
        a, b, c = structure.lattice.abc

        if a < min_a:
            return 0.5, f"Lattice a too small: {a:.2f} < {min_a}"
        if b < min_b:
            return 0.5, f"Lattice b too small: {b:.2f} < {min_b}"
        if c < min_c:
            return 0.5, f"Lattice c too small: {c:.2f} < {min_c}"

        if max_a is not None and a > max_a:
            return 0.5, f"Lattice a too large: {a:.2f} > {max_a}"
        if max_b is not None and b > max_b:
            return 0.5, f"Lattice b too large: {b:.2f} > {max_b}"
        if max_c is not None and c > max_c:
            return 0.5, f"Lattice c too large: {c:.2f} > {max_c}"

        return 1.0, None

    def _aggregate_scores(self, scores: List[float],
                          weights: Optional[List[float]] = None,
                          critical_indices: Optional[List[int]] = None) -> float:
        """Aggregate multiple scores into overall score.

        Args:
            scores: List of individual scores (0-1)
            weights: Optional weights for each score
            critical_indices: Indices of critical checks (e.g., atom overlaps)
                             If any critical check fails (score=0), overall score is capped

        Returns:
            Weighted average score, potentially capped if critical checks fail
        """
        if not scores:
            return 0.0

        # Check for critical failures
        if critical_indices:
            for idx in critical_indices:
                if idx < len(scores) and scores[idx] == 0:
                    # Critical failure - cap the maximum possible score
                    # Even if other checks pass, a critical failure limits success
                    max_possible = 0.3
                    break
            else:
                max_possible = 1.0
        else:
            max_possible = 1.0

        # Calculate weighted average
        if weights is None:
            avg_score = sum(scores) / len(scores)
        else:
            if len(weights) != len(scores):
                raise ValueError("Weights must match scores length")

            total_weight = sum(weights)
            if total_weight == 0:
                return 0.0

            avg_score = sum(s * w for s, w in zip(scores, weights)) / total_weight

        return min(avg_score, max_possible)

    def _get_min_distance(self, structure: Structure, cutoff: Optional[float] = None) -> float:
        """Get minimum interatomic distance in structure.

        Args:
            structure: Structure to analyze

        Returns:
            Minimum distance between any two atoms in Angstroms
        """
        if len(structure) < 2:
            return math.inf

        if cutoff is not None:
            try:
                _, _, _, distances = structure.get_neighbor_list(r=float(cutoff))
                distances = np.asarray(distances, dtype=float)
                distances = distances[distances > 1e-8]
                if len(distances) > 0:
                    return float(np.min(distances))

                if len(structure) > _distance_matrix_atom_limit():
                    return float(cutoff)
            except Exception:
                if len(structure) > _distance_matrix_atom_limit():
                    return float(cutoff)

        dist_matrix = np.array(structure.distance_matrix, dtype=float)
        np.fill_diagonal(dist_matrix, np.inf)
        value = float(np.min(dist_matrix))
        return value if math.isfinite(value) else math.inf

    def _process_check_result(self, score: float, message: Optional[str],
                               scores: List[float], errors: List[str],
                               warnings: List[str]) -> None:
        """Process a check result and update scores/errors/warnings lists.

        Args:
            score: Score from check (0-1)
            message: Optional error/warning message
            scores: List to append score to
            errors: List to append error messages to
            warnings: List to append warning messages to
        """
        scores.append(score)
        if message:
            if score == 0:
                errors.append(message)
            else:
                warnings.append(message)

    def check_structural_stability(self, structure: Structure) -> Tuple[float, Optional[str], Dict]:
        """Check structural stability via fast force field relaxation.

        This method uses ASE with EMT calculator to perform quick relaxation
        and assess whether the structure is in a stable configuration.

        Note: Only works for metallic systems supported by EMT (Cu, Ni, Pd, Pt, Ag, Au, Al).
        For unsupported systems, returns neutral score without error.

        Args:
            structure: Structure to check

        Returns:
            Tuple of (score, message, metrics)
            - score: 0-1, based on RMSD after relaxation
            - message: Warning/error message or None
            - metrics: Detailed relaxation metrics
        """
        global _relaxation_checker

        if not self.enable_relaxation_check:
            return 1.0, None, {'skipped': 'Relaxation check disabled'}

        # Lazy import relaxation checker
        if _relaxation_checker is None:
            try:
                from .relaxation import get_default_checker
                _relaxation_checker = get_default_checker()
            except ImportError:
                return 1.0, None, {'skipped': 'ASE not available'}

        return _relaxation_checker.check_stability(structure)
