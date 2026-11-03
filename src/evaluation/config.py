"""
Evaluation Configuration Module

Centralizes all constants and thresholds used by structure evaluators.
This allows easy tuning of evaluation criteria without modifying evaluator code.
"""

from dataclasses import dataclass
from typing import Dict, Set


# ==================== Base Evaluation Thresholds ====================

@dataclass
class BaseThresholds:
    """Base thresholds applicable to all evaluators."""
    min_interatomic_distance: float = 1.0  # Angstroms
    close_contact_distance: float = 1.5  # Angstroms
    success_threshold: float = 0.5  # Minimum score for success

    # Relaxation check settings
    enable_relaxation_check: bool = False  # Disabled by default (requires ASE)
    relaxation_max_steps: int = 50  # Maximum optimization steps
    relaxation_fmax: float = 0.1  # Force convergence criterion (eV/Å)
    relaxation_displacement_threshold: float = 0.5  # RMSD threshold (Å)

    # MLIP energy check settings
    enable_mlip_check: bool = False  # Disabled by default (requires matgl/chgnet)
    mlip_model_type: str = 'chgnet'  # 'chgnet' or 'm3gnet' (chgnet recommended for macOS)


# ==================== Defect Evaluation Configuration ====================

@dataclass
class DefectConfig:
    """Configuration for defect structure evaluation."""
    min_interatomic_distance: float = 1.5  # Angstroms
    close_contact_distance: float = 2.0  # Angstroms
    min_defect_distance: float = 5.0  # Minimum distance between defects (Angstroms)
    concentration_tolerance: float = 0.02  # 2% tolerance for defect concentration
    supercell_tolerance: float = 0.05  # 5% tolerance for lattice parameters
    success_threshold: float = 0.5
    enable_relaxation_check: bool = False  # Disabled by default (requires ASE)
    enable_mlip_check: bool = False  # Disabled by default (requires matgl/chgnet)
    mlip_model_type: str = 'chgnet'  # 'chgnet' or 'm3gnet' (chgnet recommended for macOS)

    # Formation energy thresholds (eV)
    formation_energy_excellent: float = 3.0  # < 3 eV: score 1.0
    formation_energy_acceptable: float = 5.0  # 3-5 eV: score 0.7
    # > 5 eV: score 0.3


# ==================== Adsorption Evaluation Configuration ====================

@dataclass
class AdsorptionConfig:
    """Configuration for adsorption structure evaluation."""
    # Lower threshold because adsorbate molecules have short internal bonds
    min_interatomic_distance: float = 0.8  # Angstroms (C-O ~1.13Å, O-H ~0.96Å)
    close_contact_distance: float = 1.0  # Angstroms
    height_tolerance: float = 0.5  # Angstroms
    geometry_tolerance: float = 0.3  # Angstroms
    lattice_tolerance: float = 0.3  # Angstroms
    success_threshold: float = 0.5
    enable_relaxation_check: bool = False  # Disabled by default (requires ASE)
    enable_mlip_check: bool = False  # Disabled by default (requires matgl/chgnet)
    mlip_model_type: str = 'chgnet'  # 'chgnet' or 'm3gnet' (chgnet recommended for macOS)

    # Adsorption energy thresholds (eV)
    adsorption_energy_min: float = -3.0  # Chemisorption range
    adsorption_energy_max: float = -0.5  # Weak adsorption threshold
    # -3.0 to -0.5 eV: score 1.0 (reasonable adsorption)
    # -0.5 to 0 eV: score 0.7 (weak adsorption)
    # > 0 eV: score 0.3 (unstable)

    # Site position tolerances (fractional coordinates)
    site_tolerances: Dict[str, float] = None

    def __post_init__(self):
        if self.site_tolerances is None:
            self.site_tolerances = {
                'top': 0.1,
                'bridge': 0.15,
                'hollow': 0.2,
                'fcc': 0.2,
                'hcp': 0.2,
                'ontop': 0.1,
            }


# ==================== Grain Boundary Evaluation Configuration ====================

@dataclass
class GrainBoundaryConfig:
    """Configuration for grain boundary structure evaluation."""
    min_interatomic_distance: float = 1.0  # Angstroms
    close_contact_distance: float = 1.5  # Angstroms
    min_atoms: int = 20
    min_c_axis: float = 15.0  # Minimum c-axis length (Angstroms)
    min_dimension: float = 5.0  # Minimum for any lattice dimension (Angstroms)
    success_threshold: float = 0.5
    enable_mlip_check: bool = False  # Disabled by default (requires matgl/chgnet)
    mlip_model_type: str = 'chgnet'  # 'chgnet' or 'm3gnet' (chgnet recommended for macOS)

    # CSL theory parameters - rotation angles for different sigma values
    csl_rotation_angles: Dict[int, Dict[str, float]] = None

    # Material lattice types
    material_lattice_types: Dict[str, str] = None

    def __post_init__(self):
        if self.csl_rotation_angles is None:
            self.csl_rotation_angles = {
                3: {'fcc': 60.0, 'bcc': 60.0, 'diamond': 60.0},
                5: {'fcc': 36.87, 'bcc': 36.87},
                7: {'fcc': 38.21, 'bcc': 38.21},
                9: {'fcc': 38.94, 'bcc': 38.94, 'diamond': 38.94},
                11: {'fcc': 50.48, 'bcc': 50.48},
                13: {'fcc': 22.62, 'bcc': 22.62},
                17: {'fcc': 28.07, 'bcc': 28.07},
                19: {'fcc': 26.53, 'bcc': 26.53},
                21: {'fcc': 21.79, 'bcc': 21.79},
                27: {'fcc': 31.59, 'bcc': 31.59, 'diamond': 31.59},
            }

        if self.material_lattice_types is None:
            self.material_lattice_types = {
                'Cu': 'fcc', 'Al': 'fcc', 'Ni': 'fcc', 'Au': 'fcc',
                'Ag': 'fcc', 'Pt': 'fcc', 'Pd': 'fcc',
                'Fe': 'bcc', 'W': 'bcc', 'Mo': 'bcc', 'Cr': 'bcc',
                'V': 'bcc', 'Nb': 'bcc', 'Ta': 'bcc',
                'Si': 'diamond', 'Ge': 'diamond', 'C': 'diamond',
                'Ti': 'hcp', 'Zr': 'hcp', 'Mg': 'hcp',
            }


# ==================== Interface Evaluation Configuration ====================

@dataclass
class InterfaceConfig:
    """Configuration for interface/heterostructure evaluation."""
    min_interatomic_distance: float = 1.0  # Angstroms
    close_contact_distance: float = 1.5  # Angstroms
    min_atoms: int = 15
    min_c_axis: float = 15.0  # Minimum c-axis for interface structures (Angstroms)
    min_vacuum: float = 10.0  # Minimum vacuum layer thickness (Angstroms)
    min_dimension: float = 5.0  # Angstroms
    success_threshold: float = 0.5
    enable_mlip_check: bool = False  # Disabled by default (requires matgl/chgnet)
    mlip_model_type: str = 'chgnet'  # 'chgnet' or 'm3gnet' (chgnet recommended for macOS)

    # Element sets for interface type validation
    metal_elements: Set[str] = None
    oxide_elements: Set[str] = None
    semiconductor_elements: Set[str] = None

    def __post_init__(self):
        if self.metal_elements is None:
            self.metal_elements = {
                'Cu', 'Al', 'Ni', 'Au', 'Ag', 'Pt', 'Pd',
                'Fe', 'W', 'Mo', 'Cr', 'Ti', 'Zr'
            }

        if self.oxide_elements is None:
            self.oxide_elements = {'O'}

        if self.semiconductor_elements is None:
            self.semiconductor_elements = {
                'Si', 'Ge', 'Ga', 'As', 'In', 'P', 'N'
            }


# ==================== MOF Evaluation Configuration ====================

@dataclass
class MOFConfig:
    """Configuration for MOF (Metal-Organic Framework) evaluation."""
    # MOFs can have closer contacts due to organic linkers
    min_interatomic_distance: float = 0.8  # Angstroms
    close_contact_distance: float = 1.2  # Angstroms
    min_atoms: int = 10
    max_atoms: int = 1000
    min_density: float = 0.1  # g/cm³
    max_density: float = 3.0  # g/cm³
    min_lattice: float = 5.0  # Angstroms
    max_lattice: float = 100.0  # Angstroms
    typical_min_lattice: float = 10.0  # Typical MOF unit cells
    typical_max_lattice: float = 50.0  # Angstroms
    success_threshold: float = 0.5

    # Common MOF metals
    mof_metals: Set[str] = None

    # Organic elements (must have at least C and O for carboxylate linkers)
    organic_elements: Set[str] = None

    # Scoring weights for different checks
    weights: Dict[str, float] = None

    def __post_init__(self):
        if self.mof_metals is None:
            self.mof_metals = {
                'Zn', 'Cu', 'Fe', 'Co', 'Ni', 'Zr', 'Al', 'Mg', 'Cd', 'Mn',
                'Cr', 'V', 'Ti', 'Ca', 'Ba', 'Sr', 'Pb', 'Ag', 'Au', 'Pt', 'Pd'
            }

        if self.organic_elements is None:
            self.organic_elements = {
                'C', 'O', 'N', 'H', 'S', 'P', 'F', 'Cl', 'Br', 'I'
            }

        if self.weights is None:
            self.weights = {
                'overlap': 0.3,
                'count': 0.15,
                'composition': 0.25,
                'density': 0.15,
                'lattice': 0.1,
                'properties': 0.05,
            }


# ==================== Chemical Data ====================

# Lattice constants for common metals (Angstroms)
LATTICE_CONSTANTS: Dict[str, float] = {
    'Pt': 3.92, 'Pd': 3.89, 'Cu': 3.61, 'Ni': 3.52,
    'Au': 4.08, 'Ag': 4.09, 'Rh': 3.80, 'Ir': 3.84,
    'Fe': 2.87, 'W': 3.16, 'Mo': 3.15, 'Co': 2.51,
}

# Adsorbate element composition
ADSORBATE_ELEMENTS: Dict[str, Set[str]] = {
    'H': {'H'}, 'O': {'O'}, 'N': {'N'}, 'C': {'C'},
    'CO': {'C', 'O'}, 'NO': {'N', 'O'}, 'OH': {'O', 'H'},
    'H2O': {'O', 'H'}, 'NH3': {'N', 'H'}, 'CO2': {'C', 'O'},
    'NH': {'N', 'H'}, 'CH': {'C', 'H'}, 'NH2': {'N', 'H'},
    'CH3': {'C', 'H'}, 'COOH': {'C', 'O', 'H'}, 'HCOO': {'H', 'C', 'O'},
    'OCHO': {'O', 'C', 'H'}, 'HCO': {'H', 'C', 'O'}, 'OOH': {'O', 'H'},
    'CH4': {'C', 'H'},
}

# Adsorbate atom counts
ADSORBATE_ATOM_COUNTS: Dict[str, int] = {
    'H': 1, 'O': 1, 'N': 1, 'C': 1,
    'CO': 2, 'NO': 2, 'OH': 2,
    'H2O': 3, 'NH3': 4, 'CO2': 3,
    'NH': 2, 'CH': 2, 'NH2': 3,
    'CH3': 4, 'COOH': 4, 'HCOO': 4, 'OCHO': 4, 'HCO': 3, 'OOH': 3,
    'CH4': 5,
}


# ==================== Default Configuration Instances ====================

# Create default configuration instances
DEFAULT_BASE_CONFIG = BaseThresholds()
DEFAULT_DEFECT_CONFIG = DefectConfig()
DEFAULT_ADSORPTION_CONFIG = AdsorptionConfig()
DEFAULT_GRAIN_BOUNDARY_CONFIG = GrainBoundaryConfig()
DEFAULT_INTERFACE_CONFIG = InterfaceConfig()
DEFAULT_MOF_CONFIG = MOFConfig()
