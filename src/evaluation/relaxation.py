"""
Structure Relaxation Checker Module

Provides fast force field relaxation to assess structural stability.
Uses ASE with EMT calculator for quick energy minimization.
"""

import warnings
from typing import Dict, Optional, Tuple
import numpy as np
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor


class RelaxationChecker:
    """Fast force field relaxation checker for structure stability assessment.

    Uses ASE's EMT (Effective Medium Theory) calculator for rapid relaxation.
    EMT is suitable for metallic systems (Cu, Ni, Pd, Pt, Ag, Au, Al).
    """

    # Elements supported by EMT calculator
    EMT_SUPPORTED_ELEMENTS = {'Cu', 'Ni', 'Pd', 'Pt', 'Ag', 'Au', 'Al'}

    def __init__(self,
                 max_steps: int = 50,
                 fmax: float = 0.1,
                 displacement_threshold: float = 0.5):
        """Initialize relaxation checker.

        Args:
            max_steps: Maximum optimization steps
            fmax: Force convergence criterion (eV/Å)
            displacement_threshold: RMSD threshold for stability (Å)
        """
        self.max_steps = max_steps
        self.fmax = fmax
        self.displacement_threshold = displacement_threshold

        # Lazy import to avoid dependency issues
        self._ase_available = None
        self._emt_calculator = None
        self._optimizer_class = None

    def _check_ase_availability(self) -> bool:
        """Check if ASE is available and cache the result."""
        if self._ase_available is None:
            try:
                from ase.calculators.emt import EMT
                from ase.optimize import BFGS
                self._emt_calculator = EMT
                self._optimizer_class = BFGS
                self._ase_available = True
            except ImportError:
                self._ase_available = False
        return self._ase_available

    def is_supported(self, structure: Structure) -> bool:
        """Check if structure contains only EMT-supported elements.

        Args:
            structure: Structure to check

        Returns:
            True if all elements are supported by EMT
        """
        elements = set(site.specie.symbol for site in structure)
        return elements.issubset(self.EMT_SUPPORTED_ELEMENTS)

    def relax_structure(self, structure: Structure) -> Tuple[Optional[Structure], Dict]:
        """Relax structure using EMT calculator.

        Args:
            structure: Structure to relax

        Returns:
            Tuple of (relaxed_structure, metrics_dict)
            - relaxed_structure: Relaxed structure or None if failed
            - metrics_dict: Contains 'success', 'rmsd', 'max_displacement', etc.
        """
        metrics = {
            'success': False,
            'rmsd': None,
            'max_displacement': None,
            'converged': False,
            'error': None
        }

        # Check ASE availability
        if not self._check_ase_availability():
            metrics['error'] = 'ASE not available'
            return None, metrics

        # Check element support
        if not self.is_supported(structure):
            unsupported = set(str(site.specie) for site in structure) - self.EMT_SUPPORTED_ELEMENTS
            metrics['error'] = f'Unsupported elements: {unsupported}'
            return None, metrics

        try:
            # Convert to ASE Atoms
            adaptor = AseAtomsAdaptor()
            atoms = adaptor.get_atoms(structure)

            # Store original positions
            original_positions = atoms.get_positions().copy()

            # Set up calculator
            atoms.calc = self._emt_calculator()

            # Run optimization with selective warning suppression
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=DeprecationWarning)
                warnings.filterwarnings("ignore", category=UserWarning)
                optimizer = self._optimizer_class(atoms, logfile=None)
                optimizer.run(fmax=self.fmax, steps=self.max_steps)

            # Get relaxed positions
            relaxed_positions = atoms.get_positions()

            # Calculate displacements
            displacements = np.linalg.norm(relaxed_positions - original_positions, axis=1)
            rmsd = np.sqrt(np.mean(displacements**2))
            max_displacement = np.max(displacements)

            # Convert back to pymatgen Structure
            relaxed_structure = adaptor.get_structure(atoms)

            # Update metrics
            metrics['success'] = True
            metrics['rmsd'] = float(rmsd)
            metrics['max_displacement'] = float(max_displacement)
            metrics['converged'] = optimizer.converged()
            metrics['n_steps'] = optimizer.get_number_of_steps()

            return relaxed_structure, metrics

        except Exception as e:
            metrics['error'] = str(e)
            return None, metrics

    def check_stability(self, structure: Structure) -> Tuple[float, Optional[str], Dict]:
        """Check structural stability via relaxation.

        Args:
            structure: Structure to check

        Returns:
            Tuple of (score, message, metrics)
            - score: 0-1, based on RMSD after relaxation
            - message: Warning/error message or None
            - metrics: Detailed relaxation metrics
        """
        # Check if relaxation is applicable
        if not self._check_ase_availability():
            return 1.0, None, {'skipped': 'ASE not available'}

        if not self.is_supported(structure):
            return 1.0, None, {'skipped': 'Unsupported elements'}

        # Perform relaxation
        relaxed_structure, metrics = self.relax_structure(structure)

        if not metrics['success']:
            return 0.5, f"Relaxation failed: {metrics['error']}", metrics

        rmsd = metrics['rmsd']
        max_disp = metrics['max_displacement']

        # Score based on RMSD
        if rmsd < self.displacement_threshold * 0.3:
            # Very stable - minimal relaxation
            score = 1.0
            message = None
        elif rmsd < self.displacement_threshold * 0.6:
            # Stable - small relaxation
            score = 0.9
            message = f"Small structural relaxation: RMSD={rmsd:.3f}Å"
        elif rmsd < self.displacement_threshold:
            # Moderately stable
            score = 0.7
            message = f"Moderate structural relaxation: RMSD={rmsd:.3f}Å"
        elif rmsd < self.displacement_threshold * 1.5:
            # Questionable stability
            score = 0.5
            message = f"Large structural relaxation: RMSD={rmsd:.3f}Å (max={max_disp:.3f}Å)"
        else:
            # Unstable structure
            score = 0.3
            message = f"Unstable structure: RMSD={rmsd:.3f}Å (max={max_disp:.3f}Å)"

        return score, message, metrics


# Global instance for convenience
_default_checker = None

def get_default_checker() -> RelaxationChecker:
    """Get default relaxation checker instance."""
    global _default_checker
    if _default_checker is None:
        _default_checker = RelaxationChecker()
    return _default_checker
