"""
Unified Hierarchical RAG Framework

Single RAG interface that works for all task types with automatic
knowledge domain selection and dynamic tool generation.

This is the main entry point for the RAG system. It provides:
- Automatic knowledge domain selection based on task type
- Dynamic tool generation based on active domains
- Unified query interface for all knowledge types
- Backward compatibility with existing task-specific RAG classes

Knowledge Hierarchy (L1-L3):
- L1_foundation: Element data and physical/chemical principles (always loaded)
- L2_structures: Structure-type specific - bulk, surface, defect, interface, framework
- L3_applications: Application-specific - catalysis, gas_storage, electronics
API reference and physical constraints are included in L1_foundation/shared.
"""

import json
import logging
import re
from typing import Dict, List, Optional, Any
from pathlib import Path

from src.knowledge.knowledge_store import KnowledgeStore
from .registry import KnowledgeRegistry
from .tools import ToolRegistry

logger = logging.getLogger(__name__)

# Default data directories
DEFAULT_GENERAL_DIR = Path(__file__).parent.parent.parent / "data" / "knowledge"
DEFAULT_OVERRIDES_DIR = Path(__file__).parent.parent.parent / "data" / "knowledge_overrides"


class UnifiedRAG:
    """
    Unified RAG system with hierarchical knowledge organization.

    Features:
    - Automatic knowledge domain selection based on task type
    - Dynamic tool generation based on active domains
    - Hierarchical knowledge loading (L1 -> L2 -> L3)
    - Backward compatible with existing task agents

    Knowledge Levels:
    - L1_foundation: Universal data and physical/chemical principles (always loaded)
      - periodic_table, ionic_radii, bond_lengths, oxidation_states
      - physical_principles, chemical_trends, estimation_rules
    - L2_structures: Structure-type specific (loaded based on task)
      - bulk: crystal_structures, structure_types, metal_properties, space_groups
      - surface: surface_sites
      - defect: defect_physics, formation_energies, substitution_rules
      - interface: interface_types, lattice_matching
      - framework: topologies, common_mofs, sbu_library, linker_library
    - L3_applications: Application-specific (loaded based on task)
      - catalysis: adsorbate_chemistry, adsorption_energies, scaling_relations
      - gas_storage: gas_properties
      - electronics: doping_rules
    L1 also includes the pymatgen API reference and physical constraints.

    Usage:
        # For defect engineering tasks
        rag = UnifiedRAG(task_type="defect")

        # For adsorption tasks
        rag = UnifiedRAG(task_type="adsorption")

        # Auto-infer task type from user input
        rag = UnifiedRAG()
        rag.set_task_type(rag.infer_task_type(user_input))

        # Get tools for LLM
        tools = rag.get_tool_definitions()

        # Execute a tool call
        result = rag.execute_tool("query_material_properties", {"material": "Si"})

        # Get knowledge organized by level
        knowledge_by_level = rag.get_knowledge_by_level()
    """

    # Eight short summaries fit in the initial context budget. Other domains
    # continue using their existing tool/description fallback when retrieval is empty.
    initial_retrieval_k = 8
    construction_categories = (
        "L2_structures.surface.periodic_geometry",
        "L3_applications.catalysis.adsorption_construction",
    )

    def __init__(
        self,
        task_type: Optional[str] = None,
        additional_domains: Optional[List[str]] = None,
        task_knowledge_dir: Optional[Path] = None,
        use_hierarchical: bool = True,
        strict_retrieval_assets: bool = True,
        retrieval_asset_dir: Optional[Path] = None,
    ):
        """
        Initialize the unified RAG system.

        Args:
            task_type: Type of task (e.g., 'defect', 'adsorption', 'mof')
                      If None, will be inferred from user input or use always-loaded domains only
            additional_domains: Extra domains to load beyond automatic selection
            task_knowledge_dir: Directory for task-specific knowledge files
            use_hierarchical: If True, use new L1-L3 hierarchical system (default).
                            If False, use legacy core/domain/task system.
        """
        self.task_type = task_type
        self.additional_domains = additional_domains or []
        self.task_knowledge_dir = task_knowledge_dir
        self.use_hierarchical = use_hierarchical
        self.strict_retrieval_assets = strict_retrieval_assets
        self.retrieval_asset_dir = retrieval_asset_dir
        self._paper_retriever = None
        self._paper_store_signature = None
        self.last_retrieval_metadata = {}

        # Initialize registries
        self.knowledge_registry = KnowledgeRegistry(use_hierarchical=use_hierarchical)
        self.tool_registry = ToolRegistry()

        # Determine active domains
        self.active_domains = self._resolve_domains()

        # Initialize knowledge store with appropriate directories
        self.store = self._init_knowledge_store()

        # Query history for debugging/logging
        self.query_history: List[Dict] = []

        logger.info(f"UnifiedRAG initialized with task_type={task_type}, "
                   f"hierarchical={use_hierarchical}, domains={self.active_domains}")

    def _resolve_domains(self) -> List[str]:
        """Resolve which knowledge domains to load based on task type."""
        if self.task_type:
            domains = self.knowledge_registry.get_domains_for_task(self.task_type)
        elif self.use_hierarchical:
            domains = self.knowledge_registry.always_loaded_domains
        else:
            domains = self.knowledge_registry.core_domains
        return self.knowledge_registry.resolve_domains([*domains, *self.additional_domains])

    def _init_knowledge_store(self) -> KnowledgeStore:
        """Initialize knowledge store with appropriate directories."""
        additional_dirs = []

        # Add task-specific knowledge directory if provided
        if self.task_knowledge_dir and self.task_knowledge_dir.exists():
            additional_dirs.append(self.task_knowledge_dir)

        # Preserve the source task's API subset rather than reintroducing
        # dormant manual site templates when duplicate overlays are removed.
        task_key = (self.task_type or "").strip().lower()
        task_config = self.knowledge_registry.get_task_knowledge_config(task_key) or {}
        if task_key == "surface" or "L3_catalysis" in task_config.get("L3", []):
            task_dir = DEFAULT_OVERRIDES_DIR / "adsorption"
            if task_dir.exists():
                additional_dirs.append(task_dir)

        return KnowledgeStore(
            data_dir=DEFAULT_GENERAL_DIR,
            additional_dirs=additional_dirs
        )

    def set_task_type(self, task_type: str):
        """
        Set or change the task type, reloading domains as needed.

        Args:
            task_type: New task type
        """
        if task_type != self.task_type:
            self.task_type = task_type
            self.active_domains = self._resolve_domains()
            self.store = self._init_knowledge_store()
            logger.info(f"Task type set to '{task_type}', active domains: {self.active_domains}")

    def infer_task_type(self, user_input: str) -> Optional[str]:
        """Infer task type from natural language input."""
        return self.knowledge_registry.infer_task_type(user_input)

    def _query_atom_operation(self, operation: str) -> str:
        from .atom_operations import AtomOperationRAG
        return AtomOperationRAG(self.store).describe_operation(operation)

    # ==================== Tool Interface ====================

    def get_tool_definitions(self) -> List[Dict]:
        """Get tool definitions for active domains."""
        tools = self.tool_registry.get_tools_for_domains(self.active_domains)
        if ("L2_structures.bulk.space_groups" in self.get_available_categories()
                and self.store.get_data("L2_structures.bulk.space_groups")):
            tools.append({
                "name": "query_space_group",
                "description": "Retrieve symmetry operations and affine matrices for any space group 1–230 in pymatgen's default setting.",
                "parameters": {"type": "object", "properties": {"number": {"type": "integer", "minimum": 1, "maximum": 230}}, "required": ["number"]},
            })
        return tools

    def execute_tool(self, tool_name: str, arguments: Dict) -> str:
        """
        Execute a tool call and return result.

        Args:
            tool_name: Name of the tool to execute
            arguments: Tool arguments

        Returns:
            Formatted result string
        """
        self.query_history.append({'tool': tool_name, 'args': arguments})

        # Map tool names to handler methods
        handlers = {
            # Materials domain
            "query_material_properties": self._query_material_properties,
            "query_metal_properties": self._query_metal_properties,
            # Defects domain
            "query_defect_physics": self._query_defect_physics,
            "query_substitution_rules": self._query_substitution_rules,
            "calculate_defect_count": self._calculate_defect_count,
            # Surfaces domain
            "query_surface_construction": self._query_surface_construction,
            "query_surface_sites": self._query_surface_sites,
            # Adsorption domain
            "query_adsorbate_chemistry": self._query_adsorbate_chemistry,
            "query_coadsorption_interaction": self._query_coadsorption_interaction,
            "query_adsorption_energies": self._query_adsorption_energies,
            # MOF domain
            "query_mof_topology": self._query_mof_topology,
            "query_mof_linker": self._query_mof_linker,
            "query_mof_sbu": self._query_mof_sbu,
            # Common tools
            "query_pymatgen_api": self._query_pymatgen_api,
            "query_physical_constraints": self._query_physical_constraints,
            "query_element_properties": self._query_element_properties,
            "query_space_group": self._query_space_group,
        }

        handler = handlers.get(tool_name)
        if handler:
            try:
                return handler(**arguments)
            except Exception as e:
                logger.error(f"Error executing tool {tool_name}: {e}")
                return f"Error executing {tool_name}: {str(e)}"
        else:
            return f"Unknown tool: {tool_name}. Available tools: {list(handlers.keys())}"

    # ==================== Materials Domain Queries ====================

    def _query_material_properties(self, material: str) -> str:
        """Query crystal structure properties of a material."""
        result = self.store.query_with_principles(
            "L2_structures.bulk.crystal_structures", material, context=self.task_type
        )

        if result["exact_match"]:
            props = result["data"]
            output = f"Material: {material}\n"
            output += f"  Crystal structure: {props.get('structure_type', 'unknown')}\n"

            if 'lattice_constant' in props:
                output += f"  Lattice constant: {props['lattice_constant']} Å\n"
            if 'lattice_constant_a' in props:
                output += f"  Lattice constant a: {props['lattice_constant_a']} Å\n"
                output += f"  Lattice constant c: {props.get('lattice_constant_c', 'N/A')} Å\n"

            output += f"  Space group: {props.get('space_group', 'unknown')}\n"
            output += f"  Atoms per unit cell: {props.get('atoms_per_unit_cell', 'unknown')}\n"

            if 'basis_positions' in props:
                output += f"  Basis positions: {props['basis_positions']}\n"
            if 'sublattices' in props:
                output += f"  Sublattices: {props['sublattices']}\n"

            return output
        else:
            # Return estimation principles
            output = f"Material '{material}' not in knowledge base.\n"
            if result.get("similar_materials"):
                output += f"Similar materials: {result['similar_materials']}\n"
            if result.get("principles"):
                output += "\nEstimation principles:\n"
                output += self.store.format_principles_for_llm(result["principles"], max_length=1000)
            return output

    def _query_metal_properties(self, metal: str) -> str:
        """Query properties of a catalyst metal."""
        props = self.store.query("L2_structures.bulk.metal_properties", metal)

        if props is None:
            available = self.store.list_keys("L2_structures.bulk.metal_properties")
            similar = self.store.get_similar_keys("L2_structures.bulk.metal_properties", metal, 3)
            if similar and similar[0].lower() != metal.lower():
                return f"Metal '{metal}' not found. Did you mean: {similar}? Available: {available[:10]}"
            return f"Metal '{metal}' not in knowledge base. Available: {available}"

        # Get crystal structure type info
        struct_type = props.get('structure', 'fcc')
        struct = self.store.query("L2_structures.bulk.structure_types", struct_type) or {}

        result = f"Metal: {metal}\n"
        result += f"  Crystal structure: {struct_type.upper()}\n"

        if 'lattice_constant' in props:
            result += f"  Lattice constant: {props['lattice_constant']} Å\n"
        if 'lattice_constant_a' in props:
            result += f"  Lattice constant a: {props['lattice_constant_a']} Å\n"
            result += f"  Lattice constant c: {props.get('lattice_constant_c', 'N/A')} Å\n"

        if 'd_band_center' in props:
            result += f"  d-band center: {props['d_band_center']} eV\n"
        result += f"  Description: {props.get('description', 'N/A')}\n"
        result += f"  Common facets: {props.get('common_facets', [])}\n"

        if struct:
            result += f"  Atoms per unit cell: {struct.get('atoms_per_unit_cell', 'N/A')}\n"
            result += f"  Basis positions: {struct.get('basis_positions', [])}"

        return result

    # ==================== Defects Domain Queries ====================

    def _query_defect_physics(self, defect_type: str) -> str:
        """Query physics knowledge about a defect type."""
        # Try new hierarchical category first
        physics = self.store.query("L2_structures.defect.defect_physics", defect_type)

        # Fall back to legacy category
        if physics is None:
            physics = self.store.query("defects.defect_physics", defect_type)

        if physics is None:
            available = self.store.list_keys("L2_structures.defect.defect_physics") or \
                       self.store.list_keys("defects.defect_physics")
            return f"Defect type '{defect_type}' not found. Available: {available}"

        result = f"Defect type: {defect_type}\n"
        result += f"  Description: {physics.get('description', 'N/A')}\n"
        result += f"  Formation: {physics.get('formation_mechanism', 'N/A')}\n"

        if 'concentration_formula' in physics:
            result += f"  Concentration: {physics['concentration_formula']}\n"
        if 'minimum_separation_rule' in physics:
            result += f"  Separation rule: {physics['minimum_separation_rule']}\n"

        if 'typical_formation_energy_eV' in physics:
            result += f"  Formation energies: {physics['typical_formation_energy_eV']}\n"

        if 'common_sites' in physics:
            result += f"  Common sites: {physics['common_sites']}\n"

        if 'common_dopants' in physics:
            result += f"  Common dopants: {physics['common_dopants']}\n"

        return result

    def _query_substitution_rules(self, host_element: str = None, host: str = None) -> str:
        """Query substitution/doping rules for a host element."""
        # Accept both 'host_element' and 'host' for robustness
        element = host_element or host
        if not element:
            return "Error: Please provide 'host_element' parameter"

        # Try new hierarchical category first
        rules = self.store.query("L2_structures.defect.substitution_rules", element)

        # Fall back to legacy category
        if rules is None:
            rules = self.store.query("defects.substitution_rules", element)

        if rules is None:
            available = self.store.list_keys("L2_structures.defect.substitution_rules") or \
                       self.store.list_keys("defects.substitution_rules")
            return f"No substitution rules for '{element}'. Available: {available}"

        result = f"Substitution rules for {element}:\n"
        result += f"  Compatible dopants: {rules.get('compatible_dopants', [])}\n"
        result += f"  Ionic radius: {rules.get('ionic_radius', 'N/A')} Å\n"
        result += f"  Oxidation states: {rules.get('oxidation_states', [])}\n"

        if 'size_mismatch_tolerance' in rules:
            result += f"  Size mismatch tolerance: {rules['size_mismatch_tolerance']}\n"

        return result

    def _calculate_defect_count(self, n_atoms: int, target_concentration: float) -> str:
        """Calculate number of defects needed for target concentration."""
        n_defects = max(1, round(n_atoms * target_concentration))
        actual_concentration = n_defects / n_atoms

        result = f"Defect count calculation:\n"
        result += f"  Total atoms: {n_atoms}\n"
        result += f"  Target concentration: {target_concentration * 100:.2f}%\n"
        result += f"  Calculated defects: {n_defects}\n"
        result += f"  Actual concentration: {actual_concentration * 100:.2f}%\n"
        result += f"  Deviation: {abs(actual_concentration - target_concentration) * 100:.2f}%"

        return result

    # ==================== Surfaces Domain Queries ====================

    def _query_surface_sites(self, facet: str = None, site_type: str = None, **kwargs) -> str:
        """Query available adsorption sites on a surface facet.

        Args:
            facet: Surface facet (e.g., '(111)', '(100)', '(110)')
            site_type: Optional filter for specific site type (e.g., 'top', 'bridge', 'fcc')
        """
        # Handle missing facet parameter - try to infer from site_type or provide guidance
        if facet is None:
            # Check if facet was passed with wrong key name
            facet = kwargs.get('surface') or kwargs.get('miller_index') or kwargs.get('plane')
            if facet is None:
                available = self.store.list_keys("L2_structures.surface.surface_sites") or \
                           self.store.list_keys("surfaces.surface_sites")
                return (f"Error: 'facet' parameter is required. Please specify a surface facet.\n"
                       f"Available facets: {available}\n"
                       f"Example: query_surface_sites(facet='(111)', site_type='top')")
        # Try new hierarchical category first
        surface = self.store.query("L2_structures.surface.surface_sites", facet)

        # Fall back to legacy category
        if surface is None:
            surface = self.store.query("surfaces.surface_sites", facet)

        if surface is None:
            available = self.store.list_keys("L2_structures.surface.surface_sites") or \
                       self.store.list_keys("surfaces.surface_sites")
            similar = self.store.get_similar_keys("L2_structures.surface.surface_sites", facet, 3) or \
                     self.store.get_similar_keys("surfaces.surface_sites", facet, 3)
            if similar and similar[0] != facet:
                return f"Facet '{facet}' not found. Did you mean: {similar}? Available: {available}"
            return f"Facet '{facet}' not in knowledge base. Available: {available}"

        result = f"Surface: {facet}\n"
        result += f"  Description: {surface.get('description', 'N/A')}\n"
        result += f"  Structure: {surface.get('structure', 'N/A')}\n"

        sites = surface.get('sites', {})

        # Filter by site_type if specified
        if site_type:
            # Normalize site_type for matching (handle variations like 'ontop' -> 'top')
            site_type_normalized = site_type.lower().replace('ontop', 'top')
            filtered_sites = {
                name: info for name, info in sites.items()
                if site_type_normalized in name.lower() or name.lower() in site_type_normalized
            }
            if filtered_sites:
                sites = filtered_sites
                result += f"  Filtered by site type: {site_type}\n"
            else:
                result += f"  Note: Site type '{site_type}' not found, showing all available sites.\n"

        result += f"  Available sites:\n"
        for site_name, site_info in sites.items():
            result += f"    {site_name}:\n"
            result += f"      Coordination: {site_info.get('coordination', 'N/A')}-fold\n"
            result += f"      Position: {site_info.get('position', 'N/A')}\n"
            result += f"      Fractional offset: {site_info.get('fractional_offset', [])}\n"

        return result

    # ==================== Adsorption Domain Queries ====================

    def _query_adsorbate_chemistry(self, adsorbate: str) -> str:
        """Query chemistry and binding properties of an adsorbate."""
        # Try new hierarchical category first (L3_applications.catalysis)
        chem = self.store.query("L3_applications.catalysis.adsorbate_chemistry", adsorbate)

        # Fall back to legacy category
        if chem is None:
            chem = self.store.query("adsorbates.adsorbate_chemistry", adsorbate)

        if chem is None:
            available = self.store.list_keys("L3_applications.catalysis.adsorbate_chemistry") or \
                       self.store.list_keys("adsorbates.adsorbate_chemistry")
            similar = self.store.get_similar_keys("L3_applications.catalysis.adsorbate_chemistry", adsorbate, 3) or \
                     self.store.get_similar_keys("adsorbates.adsorbate_chemistry", adsorbate, 3)
            if similar and similar[0] != adsorbate:
                return f"Adsorbate '{adsorbate}' not found. Did you mean: {similar}? Available: {available}"
            return f"Adsorbate '{adsorbate}' not in knowledge base. Available: {available}"

        result = f"Adsorbate: {adsorbate}\n"
        result += f"  Atoms: {chem.get('atoms', [])}\n"
        result += f"  Geometry: {chem.get('geometry', 'N/A')}\n"
        result += f"  Binding atom: {chem.get('binding_atom', 'N/A')}\n"

        if 'bond_length' in chem:
            result += f"  Internal bond length: {chem['bond_length']} Å\n"
        if 'bond_angle' in chem:
            result += f"  Bond angle: {chem['bond_angle']}°\n"

        if 'binding_mechanism' in chem:
            result += f"  Binding mechanism: {chem['binding_mechanism']}\n"

        result += f"  Typical height above surface: {chem.get('typical_height', 'N/A')} Å\n"

        pref = chem.get('site_preference', {})
        if pref:
            result += f"  Site preference: {pref.get('general', [])}\n"
            result += f"  Reasoning: {pref.get('reasoning', 'N/A')}"

            if 'metal_specific' in pref:
                result += f"\n  Metal-specific preferences: {pref['metal_specific']}"

        return result

    def _query_coadsorption_interaction(self, adsorbate1: str, adsorbate2: str) -> str:
        """Query interaction rules between two co-adsorbed species."""
        # Try both orderings of the key
        key1 = f"{adsorbate1}+{adsorbate2}"
        key2 = f"{adsorbate2}+{adsorbate1}"

        # Try new hierarchical category first
        interaction = self.store.query("L3_applications.catalysis.coadsorption_interactions", key1)
        if interaction is None:
            interaction = self.store.query("L3_applications.catalysis.coadsorption_interactions", key2)

        # Fall back to legacy category
        if interaction is None:
            interaction = self.store.query("adsorbates.coadsorption_interactions", key1)
        if interaction is None:
            interaction = self.store.query("adsorbates.coadsorption_interactions", key2)

        if interaction is None:
            return f"No specific interaction data for {adsorbate1}-{adsorbate2}. Use general rule: maintain >2.5 Å separation."

        result = f"Co-adsorption: {adsorbate1} + {adsorbate2}\n"
        result += f"  Interaction type: {interaction.get('interaction_type', 'N/A')}\n"
        result += f"  Minimum separation: {interaction.get('minimum_separation', 2.5)} Å\n"
        result += f"  Site competition: {interaction.get('site_competition', 'N/A')}\n"
        result += f"  Notes: {interaction.get('notes', 'N/A')}"

        if interaction.get('reaction_possible'):
            result += f"\n  Reaction possible: Yes -> {interaction.get('reaction_product', 'unknown')}"

        return result

    def _query_adsorption_energies(self, adsorbate: str, metal: str = None) -> str:
        """Query typical adsorption energies."""
        # Try new hierarchical category first
        energies = self.store.query("L3_applications.catalysis.adsorption_energies", adsorbate)

        # Fall back to legacy category
        if energies is None:
            energies = self.store.query("catalysis.adsorption_energies", adsorbate)

        if energies is None:
            available = self.store.list_keys("L3_applications.catalysis.adsorption_energies") or \
                       self.store.list_keys("catalysis.adsorption_energies")
            return f"No adsorption energy data for '{adsorbate}'. Available: {available}"

        result = f"Adsorption energies for {adsorbate}:\n"

        if metal and metal in energies:
            result += f"  On {metal}: {energies[metal]} eV\n"
        else:
            for m, e in energies.items():
                if m != '_metadata':
                    result += f"  On {m}: {e} eV\n"

        return result

    # ==================== MOF Domain Queries ====================

    def _query_mof_topology(self, topology: str) -> str:
        """Query information about a MOF topology."""
        # Try new hierarchical category first
        topo = self.store.query("L2_structures.framework.topologies", topology)

        # Fall back to legacy category
        if topo is None:
            topo = self.store.query("mof.topologies", topology)

        if topo is None:
            available = self.store.list_keys("L2_structures.framework.topologies") or \
                       self.store.list_keys("mof.topologies")
            return f"Topology '{topology}' not found. Available: {available}"

        result = f"MOF Topology: {topology}\n"
        result += f"  Description: {topo.get('description', 'N/A')}\n"
        result += f"  Connectivity: {topo.get('connectivity', 'N/A')}\n"
        result += f"  Coordination: {topo.get('coordination', 'N/A')}\n"

        if 'example_mofs' in topo:
            result += f"  Example MOFs: {topo['example_mofs']}\n"
        if 'typical_pld' in topo:
            result += f"  Typical PLD: {topo['typical_pld']} Å\n"

        return result

    def _query_mof_linker(self, linker: str) -> str:
        """Query information about a MOF linker."""
        # Try new hierarchical category first
        link = self.store.query("L2_structures.framework.linker_library", linker)

        # Fall back to legacy category
        if link is None:
            link = self.store.query("mof.linker_library", linker)

        if link is None:
            available = self.store.list_keys("L2_structures.framework.linker_library") or \
                       self.store.list_keys("mof.linker_library")
            return f"Linker '{linker}' not found. Available: {available}"

        result = f"MOF Linker: {linker}\n"
        result += f"  Full name: {link.get('full_name', link.get('name', 'N/A'))}\n"
        result += f"  Formula: {link.get('formula', 'N/A')}\n"
        result += f"  Connectivity: {link.get('connectivity', 'N/A')}\n"
        result += f"  Length: {link.get('length', link.get('length_angstrom', 'N/A'))} Å\n"

        if 'functional_groups' in link:
            result += f"  Functional groups: {link['functional_groups']}\n"

        return result

    def _query_mof_sbu(self, metal: str) -> str:
        """Query information about MOF secondary building units."""
        # Try new hierarchical category first
        sbu = self.store.query("L2_structures.framework.sbu_library", metal)

        # Fall back to legacy category
        if sbu is None:
            sbu = self.store.query("mof.sbu_library", metal)

        if sbu is None:
            available = self.store.list_keys("L2_structures.framework.sbu_library") or \
                       self.store.list_keys("mof.sbu_library")
            return f"No SBU data for metal '{metal}'. Available: {available}"

        result = f"MOF SBU for {metal}:\n"
        result += f"  Name: {sbu.get('name', 'N/A')}\n"
        result += f"  Formula: {sbu.get('formula', 'N/A')}\n"
        result += f"  Coordination: {sbu.get('coordination', 'N/A')}\n"
        result += f"  Geometry: {sbu.get('geometry', 'N/A')}\n"
        result += f"  Connectivity: {sbu.get('connectivity', 'N/A')}\n"

        if 'example_mofs' in sbu:
            result += f"  Example MOFs: {sbu['example_mofs']}\n"

        return result

    # ==================== Common Tool Queries ====================

    def _query_pymatgen_api(self, operation: str) -> str:
        """Query pymatgen API usage for a specific operation."""
        # Try new hierarchical category first
        api = self.store.query("shared.pymatgen_api", operation)

        # Fall back to legacy category
        if api is None:
            api = self.store.query("pymatgen_api.api_reference", operation)

        if api is None:
            available = self.store.list_keys("shared.pymatgen_api") or \
                       self.store.list_keys("pymatgen_api.api_reference")
            return f"Operation '{operation}' not found. Available: {available}"

        result = f"pymatgen API: {operation}\n"
        result += f"  Description: {api.get('description', 'N/A')}\n"

        if 'imports' in api:
            imports = api['imports']
            if isinstance(imports, list):
                result += f"  Imports: {'; '.join(imports)}\n"
            else:
                result += f"  Imports: {imports}\n"

        if 'method' in api:
            result += f"  Method: {api['method']}\n"
        if 'pattern' in api:
            result += f"  Pattern: {api['pattern']}\n"

        if 'parameters' in api:
            result += "  Parameters:\n"
            for param, desc in api['parameters'].items():
                result += f"    {param}: {desc}\n"

        if 'example' in api:
            result += f"  Example: {api['example']}\n"

        if 'notes' in api:
            notes = api['notes']
            if isinstance(notes, list):
                result += "  Notes:\n"
                for note in notes:
                    result += f"    - {note}\n"
            else:
                result += f"  Notes: {notes}\n"

        return result.strip()

    def _query_physical_constraints(self, constraint_type: str = None) -> str:
        """Query physical constraints for structure validation."""
        # Try new hierarchical category first
        constraints = self.store.get_data("shared.physical_constraints")

        # Fall back to legacy category
        if not constraints:
            constraints = self.store.get_data("constraints.physical_constraints")

        if not constraints:
            return "Physical constraints data not available."

        # Select constraints based on task type
        if self.task_type in ["defect", "vacancy", "substitution", "interstitial"]:
            section = constraints.get("defect_structures", {})
            title = "Defect Structures"
        elif self.task_type in ["adsorption", "surface", "coadsorption"]:
            section = constraints.get("surface_adsorption", {})
            title = "Surface Adsorption"
        elif self.task_type in ["mof", "mof_design"]:
            section = constraints.get("mof_structures", constraints)
            title = "MOF Structures"
        else:
            section = constraints
            title = "General"

        result = f"Physical Constraints for {title}:\n"

        for key, value in section.items():
            if isinstance(value, dict) and 'description' in value:
                result += f"\n{key}:\n"
                result += f"  {value.get('description', 'N/A')}\n"
                if 'value' in value:
                    result += f"  value: {value['value']} {value.get('unit', '')}\n"
                if 'minimum_atoms' in value:
                    result += f"  minimum_atoms: {value['minimum_atoms']}\n"

        return result.strip()

    def _query_element_properties(self, element: str) -> str:
        """Query properties of an element."""
        result = f"Element: {element}\n"

        # Try new hierarchical categories first, then fall back to legacy
        # Periodic table
        props = self.store.query("L1_foundation.periodic_table", element)
        if props is None:
            props = self.store.query("elements.periodic_table", element)

        if props:
            result += f"  Atomic number: {props.get('atomic_number', 'N/A')}\n"
            result += f"  Atomic mass: {props.get('atomic_mass', 'N/A')}\n"
            result += f"  Electronegativity: {props.get('electronegativity', 'N/A')}\n"

        # Ionic radii
        radii = self.store.query("L1_foundation.ionic_radii", element)
        if radii is None:
            radii = self.store.query("elements.ionic_radii", element)
        if radii:
            result += f"  Ionic radii: {radii}\n"

        # Oxidation states
        ox_states = self.store.query("L1_foundation.oxidation_states", element)
        if ox_states is None:
            ox_states = self.store.query("elements.oxidation_states", element)
        if ox_states:
            result += f"  Oxidation states: {ox_states}\n"

        if not props and not radii and not ox_states:
            return f"No data found for element '{element}'."

        return result

    # ==================== Context Building ====================

    def _retrieve_construction_principles(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve general construction principles within the active knowledge levels.

        This small, deterministic index contains knowledge only, never benchmark
        examples or results. Access through get_data preserves ablation filters
        and task overlays. An empty result leaves legacy callers' fallbacks intact.
        """
        if k <= 0:
            return []
        active = set(self.get_available_categories())
        tokens = set(re.findall(r"\w+", str(query).lower().replace("_", " ")))
        tokens -= set("a an the and or of for to in on at by from with as is are be not do it this that use".split())
        records = []
        for category in self.construction_categories:
            if category not in active:
                continue
            for topic, entry in self.store.get_data(category).items():
                if topic.startswith("_") or not isinstance(entry, dict) or not entry.get("summary"):
                    continue
                search_text = " ".join([topic.replace("_", " "), *entry.get("keywords", [])])
                terms = set(re.findall(r"\w+", search_text.lower()))
                content = "\n".join([entry["summary"], *entry.get("principles", [])])
                body_terms = set(re.findall(r"\w+", content.lower()))
                score = 3 * len(tokens & terms) + len(tokens & body_terms)
                if entry.get("code"):
                    content += f"\n```python\n{entry['code']}\n```"
                records.append({
                    "id": f"{category}.{topic}", "source": category,
                    "content": content, "score": float(score),
                    "metadata": {"topic": topic, **entry},
                })
        if str(query).strip() and not any(record["score"] for record in records):
            return []
        return sorted(records, key=lambda record: -record["score"])[:k]

    def parse_task_query(self, task_description: str, task_input: Optional[Dict] = None):
        from .paper_retrieval import parse_task_query
        return parse_task_query(task_description, task_input, self.task_type)

    def retrieve(self, query: str, k: int = 5, task_input: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """Retrieve with real Mat2Vec/CGCNN, explicit property RBF, and RRF.

        ``last_retrieval_metadata`` records query parsing, assets, corpus hashes,
        channel skips and explicit all-channels-empty fallbacks for each call.
        """
        from .paper_retrieval import PaperRetriever
        categories = tuple(sorted(self.get_available_categories()))
        signature = (id(self.store), categories, self.task_type)
        if self._paper_retriever is None or signature != self._paper_store_signature:
            options = {"task_type": self.task_type, "strict": self.strict_retrieval_assets}
            if self.retrieval_asset_dir is not None:
                options["asset_dir"] = self.retrieval_asset_dir
            self._paper_retriever = PaperRetriever(self.store, categories, **options)
            self._paper_store_signature = signature
        records = self._paper_retriever.retrieve(query, k=k, task_input=task_input)
        self.last_retrieval_metadata = self._paper_retriever.last_metadata
        if k > 0:
            self.query_history.append({"retrieval": self.last_retrieval_metadata})
        return records

    def retrieve_rules(self, query: str, k: int = 3, task_input: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """Explicit repair supplement; never contributes a fourth RRF channel."""
        self.retrieve(query, k=0, task_input=task_input)
        records = self._paper_retriever.retrieve_rules(query, k=k, task_input=task_input)
        self.last_retrieval_metadata = self._paper_retriever.last_metadata
        self.query_history.append({"retrieval": self.last_retrieval_metadata})
        return records

    def _query_space_group(self, number: int) -> str:
        if ("L2_structures.bulk.space_groups" not in self.get_available_categories()
                or not self.store.get_data("L2_structures.bulk.space_groups")):
            return "Space-group knowledge is not available in the active domain filter."
        from .paper_retrieval import space_group_reference
        return json.dumps(space_group_reference(number), ensure_ascii=False)

    @staticmethod
    def format_results_for_llm(results: List[Dict], max_length: int = 4000) -> str:
        """Fit summaries first, then whole principles and complete code blocks."""
        if max_length <= 0:
            return ""
        blocks, selected = [], []

        def fits(candidate):
            return len("\n\n".join(candidate)) <= max_length

        for record in results:
            meta = record.get("metadata", {})
            level = record.get("source", "knowledge").split("_")[0]
            heading = meta.get("topic", record.get("id", "principle"))
            block = f"[{level}: {heading}]\n{meta.get('summary', record.get('content', ''))}"
            if meta.get("condition"):
                block += f"\nCondition: {meta['condition']}\nPriority: {meta.get('priority', 'retrieval rank')}"
            if fits([*blocks, block]):
                blocks.append(block)
                selected.append(meta)
        # Round-robin details keep one long topic from consuming every spare character.
        for index in range(max((len(m.get("principles", [])) for m in selected), default=0)):
            for pos, meta in enumerate(selected):
                principles = meta.get("principles", [])
                if index < len(principles):
                    expanded = blocks.copy()
                    expanded[pos] += f"\n- {principles[index]}"
                    if fits(expanded):
                        blocks = expanded
        for pos, meta in enumerate(selected):
            if meta.get("code"):
                expanded = blocks.copy()
                expanded[pos] += f"\n```python\n{meta['code']}\n```"
                if fits(expanded):
                    blocks = expanded
        return "\n\n".join(blocks)

    def _query_surface_construction(self, query: str = "") -> str:
        """Expose the same scoped principles to explicit tool callers."""
        records = self._retrieve_construction_principles(query, k=8) or self._retrieve_construction_principles("", k=8)
        return self.format_results_for_llm(records, max_length=6500)

    def build_initial_context(self, task_description: str, task_params: Dict = None) -> str:
        """
        Build initial context for LLM based on task type and parameters.

        Args:
            task_description: Natural language task description
            task_params: Optional structured task parameters

        Returns:
            Formatted context string for LLM
        """
        # Infer task type if not set
        if not self.task_type:
            inferred = self.infer_task_type(task_description)
            if inferred:
                self.set_task_type(inferred)

        query = task_description + "\n" + json.dumps(task_params or {}, ensure_ascii=False, default=str)
        results = self.retrieve(query, k=self.initial_retrieval_k, task_input=task_params) or self.retrieve("", k=self.initial_retrieval_k, task_input=task_params)
        principles = self.format_results_for_llm(results, max_length=2500)
        context_parts = [
            f"TASK: {task_description}",
            principles,
            "",
            "AVAILABLE KNOWLEDGE TOOLS:",
        ]

        # List available tools
        for tool in self.get_tool_definitions():
            context_parts.append(f"  - {tool['name']}: {tool['description']}")

        context_parts.extend([
            "",
            "INSTRUCTIONS:",
            "1. First, query the knowledge tools to understand the task requirements",
            "2. Plan your approach based on the knowledge",
            "3. Generate Python code that satisfies all requirements",
            "4. Ensure physical constraints are satisfied",
            "5. Save output to 'output.cif'",
            "",
            "Think step by step and explain your reasoning.",
        ])

        return "\n".join(context_parts)

    # ==================== Utility Methods ====================

    def get_query_history(self) -> List[Dict]:
        """Get the history of tool queries made."""
        return self.query_history.copy()

    def clear_query_history(self):
        """Clear the query history."""
        self.query_history.clear()

    def get_active_domains(self) -> List[str]:
        """Get list of currently active domains."""
        return self.active_domains.copy()

    def get_available_categories(self) -> List[str]:
        """Get all knowledge categories available for current task."""
        return self.knowledge_registry.get_categories_for_domains(self.active_domains)

    def get_knowledge_by_level(self) -> Dict[str, List[str]]:
        """
        Get knowledge categories organized by level for current task.

        Returns:
            Dictionary mapping level names to category lists:
            {
                "L1_foundation": [...],
                "L2_structures": [...],
                "L3_applications": [...]
            }
        """
        if self.task_type:
            return self.knowledge_registry.get_knowledge_by_level(self.task_type)
        else:
            # Return only always-loaded knowledge
            return {
                "L1_foundation": self.knowledge_registry.get_categories_for_domains(["L1_foundation"]),
                "L2_structures": [],
                "L3_applications": [],
            }

    def describe_knowledge(self) -> str:
        """
        Get a human-readable description of loaded knowledge.

        Returns:
            Formatted string describing the knowledge hierarchy
        """
        if self.task_type:
            return self.knowledge_registry.describe_task_knowledge(self.task_type)
        else:
            return self.knowledge_registry.get_hierarchy_summary()

    def reload_knowledge(self):
        """Reload knowledge from files."""
        self.store.reload()
        self._paper_retriever = None
        self._paper_store_signature = None
        logger.info("Knowledge store reloaded")
