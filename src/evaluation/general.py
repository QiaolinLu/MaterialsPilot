"""
General Evaluator

Universal structure evaluator for arbitrary crystal structures.
Runs baseline physical validity checks without task-specific assumptions.
"""

from typing import Dict, Any
from pymatgen.core import Structure

from .base import BaseEvaluator, EvaluationResult


class GeneralEvaluator(BaseEvaluator):
    """
    General-purpose evaluator for any crystal structure.

    Runs universal checks that apply to all structures:
    - Atom overlap detection
    - Minimum atom count
    - Lattice dimension sanity bounds
    """

    # Generous bounds for arbitrary structures
    MAX_LATTICE_PARAM = 500.0  # Angstroms (covers MOFs, supercells)
    MIN_LATTICE_PARAM = 1.0    # Angstroms

    def evaluate(self, structure: Structure, task: Dict[str, Any]) -> EvaluationResult:
        """Evaluate structure with universal physical validity checks."""
        scores = []
        errors = []
        warnings = []

        # Check 1: Atom overlaps (critical)
        score, msg = self.check_atom_overlaps(structure)
        self._process_check_result(score, msg, scores, errors, warnings)

        # Check 2: Minimum atom count
        score, msg = self.check_atom_count(structure, min_atoms=1)
        self._process_check_result(score, msg, scores, errors, warnings)

        # Check 3: Lattice dimension sanity
        score, msg = self.check_lattice_dimensions(
            structure,
            min_a=self.MIN_LATTICE_PARAM,
            min_b=self.MIN_LATTICE_PARAM,
            min_c=self.MIN_LATTICE_PARAM,
            max_a=self.MAX_LATTICE_PARAM,
            max_b=self.MAX_LATTICE_PARAM,
            max_c=self.MAX_LATTICE_PARAM,
        )
        self._process_check_result(score, msg, scores, errors, warnings)

        # Aggregate with atom overlap as critical
        overall = self._aggregate_scores(scores, critical_indices=[0])
        success = overall >= self.success_threshold

        return EvaluationResult(
            score=overall,
            success=success,
            errors=errors,
            warnings=warnings,
            metrics={
                'num_atoms': len(structure),
                'composition': str(structure.composition.reduced_formula),
                'lattice_abc': list(structure.lattice.abc),
                'min_distance': self._get_min_distance(structure),
            },
        )
