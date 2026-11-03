"""
Knowledge Domain Definitions - Hierarchical Text-to-Structure Framework

Defines a three-level hierarchical knowledge structure:
- L1_foundation: Universal data and physical/chemical principles (always loaded)
- L2_structures: Structure-type specific knowledge
- L3_applications: Application-specific knowledge

This enables automatic knowledge selection based on the task being performed.
"""

from enum import Enum
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field


class KnowledgeLevel(Enum):
    """Hierarchical knowledge levels for text-to-structure generation."""
    L1_FOUNDATION = "L1_foundation"  # Original L0 + L1; always loaded
    L2_STRUCTURES = "L2_structures"  # Structure-type specific
    L3_APPLICATIONS = "L3_applications"  # Application-specific
    # Compatibility names are aliases, not additional hierarchy levels.
    SHARED = "L1_foundation"
    CORE = "L1_foundation"
    DOMAIN = "L2_structures"
    TASK = "L3_applications"


@dataclass
class DomainConfig:
    """Configuration for a knowledge domain."""
    name: str
    level: KnowledgeLevel
    categories: List[str]  # Knowledge categories in this domain
    subcategory: Optional[str] = None  # For L2/L3 subcategories (e.g., "bulk", "catalysis")
    dependencies: List[str] = field(default_factory=list)  # Other domains this depends on
    description: str = ""


# ============================================================================
# NEW HIERARCHICAL DOMAIN CONFIGURATIONS
# ============================================================================

DOMAIN_CONFIGS: Dict[str, DomainConfig] = {
    # L1 combines universal data and physical/chemical reasoning principles.
    "L1_foundation": DomainConfig(
        name="L1_foundation",
        level=KnowledgeLevel.L1_FOUNDATION,
        categories=[
            "L1_foundation.periodic_table",
            "L1_foundation.ionic_radii",
            "L1_foundation.bond_lengths",
            "L1_foundation.oxidation_states",
            "L1_foundation.physical_principles",
            "L1_foundation.chemical_trends",
            "L1_foundation.estimation_rules",
            "L1_foundation.interface_principles",
            "L1_foundation.material_inference_rules",
            "L1_foundation.shared.pymatgen_api",
            "L1_foundation.shared.physical_constraints",
        ],
        description="Universal element data and physical/chemical principles"
    ),

    # ==================== L2: Structure-Type Specific ====================
    "L2_bulk": DomainConfig(
        name="L2_bulk",
        level=KnowledgeLevel.L2_STRUCTURES,
        subcategory="bulk",
        categories=[
            "L2_structures.bulk.crystal_structures",
            "L2_structures.bulk.structure_types",
            "L2_structures.bulk.metal_properties",
            "L2_structures.bulk.space_groups",
            "L2_structures.bulk.wyckoff_positions",
        ],
        description="Bulk crystal structure knowledge"
    ),
    "L2_surface": DomainConfig(
        name="L2_surface",
        level=KnowledgeLevel.L2_STRUCTURES,
        subcategory="surface",
        categories=[
            "L2_structures.surface.surface_sites",
            "L2_structures.surface.surface_structures",
            "L2_structures.surface.periodic_geometry",
        ],
        dependencies=["L2_bulk"],
        description="Surface structure and adsorption sites"
    ),
    "L2_defect": DomainConfig(
        name="L2_defect",
        level=KnowledgeLevel.L2_STRUCTURES,
        subcategory="defect",
        categories=[
            "L2_structures.defect.defect_physics",
            "L2_structures.defect.defect_construction",
            "L2_structures.defect.complex_defects",
            "L2_structures.defect.interstitial_sites",
            "L2_structures.defect.advanced_structures",
            "L2_structures.defect.formation_energies",
            "L2_structures.defect.substitution_rules",
        ],
        dependencies=["L2_bulk"],
        description="Defect structure knowledge (vacancies, substitutions, interstitials)"
    ),
    "L2_interface": DomainConfig(
        name="L2_interface",
        level=KnowledgeLevel.L2_STRUCTURES,
        subcategory="interface",
        categories=[
            "L2_structures.interface.interface_types",
            "L2_structures.interface.construction_patterns",
        ],
        dependencies=["L2_bulk"],
        description="Interface and heterostructure knowledge"
    ),
    "L2_grain_boundary": DomainConfig(
        name="L2_grain_boundary",
        level=KnowledgeLevel.L2_STRUCTURES,
        subcategory="interface",
        categories=[
            "L2_structures.interface.grain_boundary",
            "L2_structures.interface.construction_patterns",
            "L2_structures.interface.interface_types",
        ],
        dependencies=["L2_bulk"],
        description="Grain boundary construction knowledge with performance optimization"
    ),
    "L2_framework": DomainConfig(
        name="L2_framework",
        level=KnowledgeLevel.L2_STRUCTURES,
        subcategory="framework",
        categories=[
            "L2_structures.framework.topologies",
            "L2_structures.framework.common_mofs",
            "L2_structures.framework.sbu_library",
            "L2_structures.framework.linker_library",
        ],
        dependencies=["L2_bulk"],
        description="Framework structure knowledge (MOFs, COFs, zeolites)"
    ),

    "L2_atom_operations": DomainConfig(
        name="L2_atom_operations",
        level=KnowledgeLevel.L2_STRUCTURES,
        subcategory="atom_operations",
        categories=["L2_structures.atom_operations.geometry", "L2_structures.atom_operations.operations"],
        description="Cartesian atom editing, periodic geometry, and AtomWorld operation semantics",
    ),

    # ==================== L3: Application-Specific ====================
    "L3_catalysis": DomainConfig(
        name="L3_catalysis",
        level=KnowledgeLevel.L3_APPLICATIONS,
        subcategory="catalysis",
        categories=[
            "L3_applications.catalysis.adsorbate_chemistry",
            "L3_applications.catalysis.adsorbate_placement",
            "L3_applications.catalysis.adsorption_construction",
            "L3_applications.catalysis.coadsorption_interactions",
            "L3_applications.catalysis.adsorption_energies",
            "L3_applications.catalysis.scaling_relations",
            "L3_applications.catalysis.surface_energies",
        ],
        dependencies=["L2_surface"],
        description="Catalysis and adsorption application knowledge"
    ),
    "L3_gas_storage": DomainConfig(
        name="L3_gas_storage",
        level=KnowledgeLevel.L3_APPLICATIONS,
        subcategory="gas_storage",
        categories=[
        ],
        dependencies=["L2_framework"],
        description="Gas storage application knowledge"
    ),
    "L3_electronics": DomainConfig(
        name="L3_electronics",
        level=KnowledgeLevel.L3_APPLICATIONS,
        subcategory="electronics",
        categories=[
        ],
        dependencies=["L2_defect"],
        description="Electronics and doping application knowledge"
    ),

    # Compatibility domain; these resources belong to L1 and are loaded there.
    "shared": DomainConfig(
        name="shared",
        level=KnowledgeLevel.L1_FOUNDATION,
        categories=[
            "L1_foundation.shared.pymatgen_api",
            "L1_foundation.shared.physical_constraints",
        ],
        description="[Compatibility] L1 API reference and physical constraints"
    ),

    # ==================== Legacy Domain Aliases (Backward Compatibility) ====================
    "elements": DomainConfig(
        name="elements",
        level=KnowledgeLevel.CORE,
        categories=[
            "L1_foundation.periodic_table",
            "L1_foundation.ionic_radii",
            "L1_foundation.oxidation_states",
            "L1_foundation.bond_lengths",
        ],
        description="[Legacy] Fundamental element properties"
    ),
    "materials": DomainConfig(
        name="materials",
        level=KnowledgeLevel.L2_STRUCTURES,
        categories=[
            "L2_structures.bulk.crystal_structures",
            "L2_structures.bulk.structure_types",
            "L2_structures.bulk.metal_properties",
        ],
        description="[Legacy] Crystal structure and material property data"
    ),
    "symmetry": DomainConfig(
        name="symmetry",
        level=KnowledgeLevel.L2_STRUCTURES,
        categories=[
            "L2_structures.bulk.space_groups",
            "L2_structures.bulk.wyckoff_positions",
        ],
        description="[Legacy] Crystallographic symmetry information"
    ),
    "principles": DomainConfig(
        name="principles",
        level=KnowledgeLevel.CORE,
        categories=[
            "L1_foundation.physical_principles",
            "L1_foundation.chemical_trends",
            "L1_foundation.estimation_rules",
            "L1_foundation.interface_principles",
        ],
        description="[Legacy] Physical and chemical principles"
    ),
    "defects": DomainConfig(
        name="defects",
        level=KnowledgeLevel.DOMAIN,
        categories=[
            "L2_structures.defect.defect_physics",
            "L2_structures.defect.defect_construction",
            "L2_structures.defect.complex_defects",
            "L2_structures.defect.interstitial_sites",
            "L2_structures.defect.advanced_structures",
            "L2_structures.defect.formation_energies",
            "L2_structures.defect.substitution_rules",
        ],
        dependencies=["materials", "elements"],
        description="[Legacy] Defect engineering knowledge"
    ),
    "surfaces": DomainConfig(
        name="surfaces",
        level=KnowledgeLevel.DOMAIN,
        categories=[
            "L2_structures.surface.surface_sites",
            "L2_structures.surface.periodic_geometry",
            "L2_structures.surface.surface_structures",
        ],
        dependencies=["materials"],
        description="[Legacy] Surface science knowledge"
    ),
    "adsorption": DomainConfig(
        name="adsorption",
        level=KnowledgeLevel.L3_APPLICATIONS,
        categories=[
            "L3_applications.catalysis.adsorbate_chemistry",
            "L3_applications.catalysis.adsorbate_placement",
            "L3_applications.catalysis.adsorption_construction",
            "L3_applications.catalysis.coadsorption_interactions",
            "L3_applications.catalysis.adsorption_energies",
            "L3_applications.catalysis.scaling_relations",
            "L3_applications.catalysis.surface_energies",
        ],
        dependencies=["surfaces", "elements"],
        description="[Legacy] Adsorption chemistry and catalysis"
    ),
    "mof": DomainConfig(
        name="mof",
        level=KnowledgeLevel.DOMAIN,
        categories=[
            "L2_structures.framework.topologies",
            "L2_structures.framework.common_mofs",
            "L2_structures.framework.sbu_library",
            "L2_structures.framework.linker_library",
        ],
        dependencies=["materials", "symmetry"],
        description="[Legacy] Metal-Organic Framework design knowledge"
    ),
    "interfaces": DomainConfig(
        name="interfaces",
        level=KnowledgeLevel.DOMAIN,
        categories=[
            "L2_structures.interface.interface_types",
        ],
        dependencies=["materials", "symmetry"],
        description="[Legacy] Interface and heterostructure knowledge"
    ),
    "grain_boundaries": DomainConfig(
        name="grain_boundaries",
        level=KnowledgeLevel.DOMAIN,
        categories=[
            "L2_structures.interface.interface_types",
        ],
        dependencies=["materials", "symmetry"],
        description="[Legacy] Grain boundary structure knowledge"
    ),
    "pymatgen_api": DomainConfig(
        name="pymatgen_api",
        level=KnowledgeLevel.L1_FOUNDATION,
        categories=[
            "L1_foundation.shared.pymatgen_api",
        ],
        description="[Legacy] Pymatgen API usage patterns"
    ),
    "constraints": DomainConfig(
        name="constraints",
        level=KnowledgeLevel.L1_FOUNDATION,
        categories=[
            "L1_foundation.shared.physical_constraints",
        ],
        description="[Legacy] Physical constraints for structure validation"
    ),
}


# ============================================================================
# TASK TO KNOWLEDGE MAPPING (New Hierarchical System)
# ============================================================================

TASK_KNOWLEDGE_MAP: Dict[str, Dict[str, Any]] = {
    "atomworld_structure_edit": {"L2": ["L2_atom_operations"], "L3": []},
    "atomworld": {"inherits": "atomworld_structure_edit"},
    "general": {"inherits": "bulk"},
    "defect_complex": {"inherits": "defect"},
    # Bulk structure generation
    "bulk": {
        "description": "Bulk crystal structure generation",
        "L2": ["L2_bulk"],
        "L3": [],
    },

    # Defect tasks
    "defect": {
        "description": "Point defect generation (vacancies, substitutions, interstitials)",
        "L2": ["L2_bulk", "L2_defect"],
        "L3": [],
    },
    "vacancy": {"inherits": "defect"},
    "substitution": {"inherits": "defect"},
    "interstitial": {"inherits": "defect"},

    # Surface/adsorption tasks
    "surface": {
        "description": "Surface slab generation",
        "L2": ["L2_bulk", "L2_surface"],
        "L3": [],
    },
    "adsorption": {
        "description": "Surface adsorption structure generation",
        "L2": ["L2_bulk", "L2_surface"],
        "L3": ["L3_catalysis"],
    },
    "coadsorption": {"inherits": "adsorption"},
    "co_adsorption": {"inherits": "adsorption"},
    "stepped_adsorption": {"inherits": "adsorption"},
    "complex_adsorption": {"inherits": "adsorption"},
    "coverage_adsorption": {"inherits": "adsorption"},
    "catalysis": {"inherits": "adsorption"},

    # MOF tasks
    "mof": {
        "description": "Metal-Organic Framework structure generation",
        "L2": ["L2_bulk", "L2_framework"],
        "L3": ["L3_gas_storage"],
    },
    "mof_design": {"inherits": "mof"},
    "mof_reconstruction": {"inherits": "mof"},

    # Interface tasks
    "interface": {
        "description": "Interface/heterostructure generation",
        "L2": ["L2_bulk", "L2_interface"],
        "L3": [],
    },
    "heterostructure": {"inherits": "interface"},
    "grain_boundary": {
        "description": "Grain boundary structure generation with CSL theory",
        "L2": ["L2_bulk", "L2_grain_boundary"],
        "L3": [],
    },
    # Expanded benchmark task variants
    "interface_expanded": {"inherits": "interface"},
}


# Legacy task-domain mapping for backward compatibility
TASK_DOMAIN_MAP: Dict[str, List[str]] = {
    "atomworld_structure_edit": ["L2_atom_operations"],
    "atomworld": ["L2_atom_operations"],
    # Defect engineering tasks
    "defect": ["defects", "pymatgen_api", "constraints"],
    "vacancy": ["defects", "pymatgen_api", "constraints"],
    "substitution": ["defects", "pymatgen_api", "constraints"],
    "interstitial": ["defects", "pymatgen_api", "constraints"],

    # Surface/adsorption tasks
    "adsorption": ["adsorption", "surfaces", "pymatgen_api", "constraints"],
    "coadsorption": ["adsorption", "surfaces", "pymatgen_api", "constraints"],
    "surface": ["surfaces", "pymatgen_api", "constraints"],
    "catalysis": ["adsorption", "surfaces", "pymatgen_api", "constraints"],

    # MOF tasks
    "mof": ["mof", "pymatgen_api", "constraints"],
    "mof_design": ["mof", "pymatgen_api", "constraints"],
    "mof_reconstruction": ["mof", "pymatgen_api", "constraints"],

    # Interface tasks
    "interface": ["interfaces", "pymatgen_api", "constraints"],
    "heterostructure": ["interfaces", "pymatgen_api", "constraints"],
    "grain_boundary": ["grain_boundaries", "pymatgen_api", "constraints"],

    # Bulk tasks
    "bulk": ["materials", "symmetry", "pymatgen_api", "constraints"],
}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_always_loaded_domains() -> List[str]:
    """Get the always-loaded foundation, including API and constraint data."""
    return ["L1_foundation"]


def get_core_domains() -> List[str]:
    """Get all core domain names (legacy compatibility)."""
    return ["L1_foundation", "shared", "elements", "materials", "symmetry", "principles"]


def get_domain_config(domain_name: str) -> Optional[DomainConfig]:
    """Get configuration for a specific domain."""
    return DOMAIN_CONFIGS.get(domain_name)


def get_all_categories_for_domain(domain_name: str) -> List[str]:
    """Resolve all transitive dependencies in deterministic order."""
    from .registry import KnowledgeRegistry
    registry = KnowledgeRegistry()
    return registry.get_categories_for_domains(registry.resolve_domains([domain_name]))


def get_domains_for_task(task_type: str) -> List[str]:
    from .registry import KnowledgeRegistry
    return KnowledgeRegistry().get_domains_for_task(task_type)


def get_categories_for_task(task_type: str) -> List[str]:
    from .registry import KnowledgeRegistry
    return KnowledgeRegistry().get_categories_for_task(task_type)
