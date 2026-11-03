"""
MLIP-based Energy Evaluator Module

Provides energy evaluation using Machine Learning Interatomic Potentials (MLIPs).
Supports M3GNet, CHGNet, and MACE for fast energy and force predictions.
"""

import warnings
import os
import errno
from typing import Dict, Optional, Tuple, Any
from pymatgen.core import Structure
import numpy as np
from .mlip_resources import MLIPResourceError, mlip_resource_slot


def is_mlip_resource_error(error: BaseException) -> bool:
    """Recognize explicit allocation/device failures, including wrapped causes."""
    seen = set()
    current = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, (MLIPResourceError, MemoryError)):
            return True
        if isinstance(current, OSError) and current.errno == errno.ENOMEM:
            return True
        message = str(current).lower()
        if any(marker in message for marker in (
            "out of memory", "cannot allocate memory", "can't allocate memory",
            "defaultcpuallocator", "std::bad_alloc", "memory allocation failed",
            "cublas_status_alloc_failed", "cudnn_status_alloc_failed",
            "cuda initialization error", "cuda driver version is insufficient",
            "found no nvidia driver", "invalid device ordinal",
        )):
            return True
        current = current.__cause__ or current.__context__
    return False


class MLIPEvaluator:
    """MLIP-based energy evaluator using M3GNet, CHGNet, or MACE.

    Provides fast energy predictions for structure stability assessment.
    Uses lazy loading to avoid dependency issues when MLIP is not needed.
    """

    def __init__(self, model_type: str = 'm3gnet', device: str = None):
        """Initialize MLIP evaluator.

        Args:
            model_type: 'm3gnet', 'chgnet', or 'mace'
            device: Device to use ('auto', 'cuda', 'cpu'). 'auto' selects GPU if available.
        """
        if device is None:
            device = os.getenv('MATERIALSPILOT_MLIP_DEVICE', 'auto')

        if model_type not in ['m3gnet', 'chgnet', 'mace']:
            raise ValueError(f"Unsupported model_type: {model_type}. Use 'm3gnet', 'chgnet', or 'mace'")

        if device not in ['auto', 'cuda', 'cpu']:
            raise ValueError(f"Unsupported device: {device}. Use 'auto', 'cuda', or 'cpu'")

        self.model_type = model_type
        self.device = device
        self._model = None
        self._model_available = None

    def _check_model_availability(self) -> bool:
        """Check if the required MLIP model is available."""
        if self._model_available is not None:
            return self._model_available

        try:
            if self.model_type == 'm3gnet':
                import matgl
                self._model_available = True
            elif self.model_type == 'chgnet':
                import chgnet
                self._model_available = True
            elif self.model_type == 'mace':
                import mace
                self._model_available = True
        except ImportError:
            self._model_available = False

        return self._model_available

    def _get_device(self) -> str:
        """Determine the device to use for model inference.

        Returns:
            Device string ('cuda' or 'cpu')
        """
        if self.device == 'auto':
            try:
                import torch
                return 'cuda' if torch.cuda.is_available() else 'cpu'
            except ImportError:
                return 'cpu'
        return self.device

    def _load_model(self):
        """Lazy load MLIP model."""
        if self._model is not None:
            return

        if not self._check_model_availability():
            install_cmd = {
                'm3gnet': 'matgl',
                'chgnet': 'chgnet',
                'mace': 'mace-torch'
            }.get(self.model_type, self.model_type)
            raise ImportError(
                f"{self.model_type} not available. "
                f"Install with: pip install {install_cmd}"
            )

        device = self._get_device()

        try:
            if self.model_type == 'm3gnet':
                from matgl import load_model
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=DeprecationWarning)
                    warnings.filterwarnings("ignore", category=UserWarning)
                    self._model = load_model("M3GNet-MP-2021.2.8-PES")
                    # M3GNet models can be moved to device
                    if hasattr(self._model, 'to'):
                        self._model = self._model.to(device)
            elif self.model_type == 'chgnet':
                from chgnet.model import CHGNet
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=DeprecationWarning)
                    warnings.filterwarnings("ignore", category=UserWarning)
                    # Force the selected device at load time. CHGNet probes CUDA
                    # internally by default, which can fail in multiprocessing
                    # post-hoc evaluation even when CPU inference is requested.
                    try:
                        self._model = CHGNet.load(use_device=device)
                    except TypeError:
                        self._model = CHGNet.load()
                        if hasattr(self._model, 'to'):
                            self._model = self._model.to(device)
            elif self.model_type == 'mace':
                from mace.calculators import mace_mp
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=DeprecationWarning)
                    warnings.filterwarnings("ignore", category=UserWarning)
                    # Load MACE-MP-0 (medium, universal) model with GPU support
                    self._model = mace_mp(
                        model="medium",
                        dispersion=False,
                        default_dtype="float32",
                        device=device
                    )
        except Exception as e:
            raise RuntimeError(f"Failed to load {self.model_type} model: {e}")

    def predict_energy_only(self, structure: Structure) -> Dict[str, Any]:
        """CHGNet energy without force/stress gradients for the paper critic.

        This separate entry point leaves ``predict_energy``'s energy/force/stress
        contract intact. It uses the same model energy, normalization and weights.
        Resource failures propagate distinctly and must not become scored samples.
        """
        if self.model_type != 'chgnet':
            raise NotImplementedError("Energy-only paper inference currently supports CHGNet")
        try:
            with mlip_resource_slot():
                self._load_model()
                import torch
                with torch.no_grad():
                    prediction = self._model.predict_structure(structure, task='e')
                energy_per_atom = float(prediction['e'])
                if not np.isfinite(energy_per_atom):
                    raise ValueError("MLIP energy is not finite")
                return {
                    'energy': energy_per_atom * len(structure),
                    'energy_per_atom': energy_per_atom,
                    'prediction_mode': 'energy_only_no_grad',
                    'force_diagnostics': 'not_requested_non_scoring',
                }
        except Exception as exc:
            if is_mlip_resource_error(exc):
                raise MLIPResourceError(f"MLIP resource failure during energy-only prediction: {exc}") from exc
            raise

    def predict_energy(self, structure: Structure) -> Dict[str, Any]:
        """Predict energy and forces for a structure.

        Args:
            structure: Structure to evaluate

        Returns:
            Dictionary containing:
                - energy: Total energy (eV)
                - energy_per_atom: Energy per atom (eV/atom)
                - forces: Forces on atoms (eV/Å)
                - max_force: Maximum force magnitude (eV/Å)
                - stress: Stress tensor (if available)
        """
        self._load_model()

        try:
            if self.model_type == 'm3gnet':
                # M3GNet prediction
                graph = self._model.graph_converter(structure)
                prediction = self._model(graph)

                energy = float(prediction['energy'].detach().cpu().numpy())
                forces = prediction['forces'].detach().cpu().numpy()
                stress = prediction.get('stress', None)
                if stress is not None:
                    stress = stress.detach().cpu().numpy()

            elif self.model_type == 'chgnet':
                # CHGNet prediction
                prediction = self._model.predict_structure(structure)
                # CHGNet returns energy in eV/atom. Convert to total energy so
                # the common result schema stays consistent across MLIP models.
                energy = float(prediction['e']) * len(structure)
                forces = prediction['f']
                stress = prediction.get('s', None)

            elif self.model_type == 'mace':
                # MACE prediction (uses ASE calculator interface)
                from pymatgen.io.ase import AseAtomsAdaptor

                # Convert pymatgen Structure to ASE Atoms
                adaptor = AseAtomsAdaptor()
                atoms = adaptor.get_atoms(structure)

                # Set calculator and compute properties
                atoms.calc = self._model
                energy = float(atoms.get_potential_energy())
                forces = atoms.get_forces()

                # Get stress if available
                try:
                    stress = atoms.get_stress(voigt=False)  # 3x3 tensor
                except Exception:
                    stress = None

            # Calculate derived metrics
            n_atoms = len(structure)
            energy_per_atom = energy / n_atoms
            force_array = np.asarray(forces, dtype=float)
            force_norms = np.linalg.norm(force_array, axis=1)
            max_force = float(np.max(force_norms))
            mean_force = float(np.mean(force_norms))
            force_p95 = float(np.percentile(force_norms, 95))

            result = {
                'energy': energy,
                'energy_per_atom': energy_per_atom,
                'forces': force_array,
                'max_force': max_force,
                'mean_force': mean_force,
                'force_p95': force_p95,
            }

            if stress is not None:
                result['stress'] = stress

            return result

        except Exception as e:
            raise RuntimeError(f"MLIP prediction failed: {e}")

    def relax_structure(
        self,
        structure: Structure,
        fmax: float = 0.5,
        steps: int = 8,
        relax_cell: bool = False,
        verbose: bool = False,
    ) -> Tuple[Structure, Dict[str, Any]]:
        """Run a short MLIP relaxation and return the relaxed structure.

        The generation-time feedback loop only needs a small local repair, not
        a production relaxation.  CHGNet exposes a pymatgen-native optimizer, so
        use it here and keep other MLIP backends prediction-only unless they add
        an equally direct relaxation API.
        """
        if self.model_type != 'chgnet':
            raise NotImplementedError(
                "MLIP relaxation is currently implemented only for CHGNet"
            )

        self._load_model()
        try:
            from chgnet.model.dynamics import StructOptimizer

            optimizer = StructOptimizer(
                model=self._model,
                use_device=self._get_device(),
            )
            result = optimizer.relax(
                structure,
                fmax=fmax,
                steps=max(1, int(steps)),
                relax_cell=bool(relax_cell),
                verbose=verbose,
                loginterval=max(1, int(steps)),
            )
            relaxed = result.get('final_structure')
            if relaxed is None:
                raise RuntimeError("CHGNet relaxation did not return final_structure")

            metrics = self.predict_energy(relaxed)
            metrics.update(
                {
                    'relax_steps_requested': int(steps),
                    'relax_fmax': float(fmax),
                    'relax_cell': bool(relax_cell),
                }
            )
            return relaxed, metrics
        except Exception as e:
            raise RuntimeError(f"MLIP relaxation failed: {e}")

    def evaluate_stability(self, structure: Structure) -> Tuple[float, Optional[str], Dict]:
        """Evaluate structure stability based on forces.

        Args:
            structure: Structure to evaluate

        Returns:
            Tuple of (score, message, metrics)
            - score: 0-1, based on maximum force
            - message: Warning message or None
            - metrics: Detailed prediction metrics
        """
        if not self._check_model_availability():
            return 1.0, None, {'skipped': f'{self.model_type} not available'}

        try:
            metrics = self.predict_energy(structure)
            max_force = metrics['max_force']

            # Score based on maximum force
            if max_force < 0.1:
                score = 1.0
                message = None
            elif max_force < 0.3:
                score = 0.8
                message = f"Moderate forces: {max_force:.3f} eV/Å"
            elif max_force < 0.5:
                score = 0.6
                message = f"High forces: {max_force:.3f} eV/Å"
            else:
                score = 0.3
                message = f"Very high forces: {max_force:.3f} eV/Å - structure unstable"

            return score, message, metrics

        except Exception as e:
            return 0.5, f"MLIP evaluation failed: {e}", {}

    def calculate_formation_energy(
        self,
        defect_structure: Structure,
        pristine_structure: Structure,
        removed_element: Optional[str] = None,
        chemical_potential: float = 0.0
    ) -> Tuple[float, Dict]:
        """Calculate defect formation energy.

        Args:
            defect_structure: Structure with defect
            pristine_structure: Pristine structure
            removed_element: Element removed (for vacancy)
            chemical_potential: Chemical potential of removed element (eV)

        Returns:
            Tuple of (formation_energy, metrics)
        """
        if not self._check_model_availability():
            raise RuntimeError(f"{self.model_type} not available")

        # Predict energies
        defect_pred = self.predict_energy(defect_structure)
        pristine_pred = self.predict_energy(pristine_structure)

        # Calculate formation energy
        # E_f = E_defect - E_pristine + n_removed * μ
        n_removed = len(pristine_structure) - len(defect_structure)
        formation_energy = (
            defect_pred['energy'] - pristine_pred['energy'] +
            n_removed * chemical_potential
        )

        metrics = {
            'formation_energy': formation_energy,
            'defect_energy': defect_pred['energy'],
            'pristine_energy': pristine_pred['energy'],
            'defect_max_force': defect_pred['max_force'],
            'pristine_max_force': pristine_pred['max_force'],
        }

        return formation_energy, metrics

    def calculate_adsorption_energy(
        self,
        total_structure: Structure,
        slab_structure: Structure,
        molecule_energy: Optional[float] = None
    ) -> Tuple[float, Dict]:
        """Calculate adsorption energy.

        Args:
            total_structure: Slab + adsorbate structure
            slab_structure: Clean slab structure
            molecule_energy: Gas phase molecule energy (eV), if known

        Returns:
            Tuple of (adsorption_energy, metrics)
        """
        if not self._check_model_availability():
            raise RuntimeError(f"{self.model_type} not available")

        # Predict energies
        total_pred = self.predict_energy(total_structure)
        slab_pred = self.predict_energy(slab_structure)

        # Calculate adsorption energy
        # E_ads = E_total - E_slab - E_molecule
        # If molecule_energy not provided, approximate from atom count difference
        if molecule_energy is None:
            # Rough approximation: assume molecule energy is proportional to atoms
            n_adsorbate = len(total_structure) - len(slab_structure)
            molecule_energy = 0.0  # Will give relative adsorption strength

        adsorption_energy = (
            total_pred['energy'] - slab_pred['energy'] - molecule_energy
        )

        metrics = {
            'adsorption_energy': adsorption_energy,
            'total_energy': total_pred['energy'],
            'slab_energy': slab_pred['energy'],
            'molecule_energy': molecule_energy,
            'total_max_force': total_pred['max_force'],
            'slab_max_force': slab_pred['max_force'],
        }

        return adsorption_energy, metrics


# Global instance for convenience
_default_evaluator = None

def get_default_evaluator(model_type: str = 'm3gnet', device: str = 'auto') -> MLIPEvaluator:
    """Get default MLIP evaluator instance.

    Args:
        model_type: 'm3gnet', 'chgnet', or 'mace'
        device: Device to use ('auto', 'cuda', 'cpu')

    Returns:
        MLIPEvaluator instance
    """
    global _default_evaluator
    if (_default_evaluator is None or
        _default_evaluator.model_type != model_type or
        _default_evaluator.device != device):
        _default_evaluator = MLIPEvaluator(model_type=model_type, device=device)
    return _default_evaluator
