"""
Dynamic Tool Definition System

Generates tool definitions based on active knowledge domains.
Each domain can register its own tools, and the system dynamically
generates the appropriate tool definitions for the LLM.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field


@dataclass
class ToolDefinition:
    """Definition for a RAG tool."""
    name: str
    description: str
    parameters: Dict[str, Any]
    domain: str  # Which domain this tool belongs to
    handler: str  # Method name to call in UnifiedRAG


# Base tool definitions organized by domain
DOMAIN_TOOLS: Dict[str, List[ToolDefinition]] = {
    # ==================== Materials Domain ====================
    "materials": [
        ToolDefinition(
            name="query_material_properties",
            description="Query crystal structure properties of a material (lattice constant, structure type, space group, atoms per unit cell)",
            parameters={
                "type": "object",
                "properties": {
                    "material": {
                        "type": "string",
                        "description": "Material formula (e.g., 'Si', 'GaAs', 'Fe', 'NaCl')"
                    }
                },
                "required": ["material"]
            },
            domain="materials",
            handler="_query_material_properties"
        ),
        ToolDefinition(
            name="query_metal_properties",
            description="Query properties of a metal including crystal structure, lattice constant, and d-band center",
            parameters={
                "type": "object",
                "properties": {
                    "metal": {
                        "type": "string",
                        "description": "Metal symbol (e.g., 'Pt', 'Cu', 'Ni', 'Fe')"
                    }
                },
                "required": ["metal"]
            },
            domain="materials",
            handler="_query_metal_properties"
        ),
    ],

    # ==================== Defects Domain ====================
    "defects": [
        ToolDefinition(
            name="query_defect_physics",
            description="Query physics knowledge about defect types (vacancy, substitution, interstitial) including formation mechanisms and typical energies",
            parameters={
                "type": "object",
                "properties": {
                    "defect_type": {
                        "type": "string",
                        "description": "Type of defect: 'vacancy', 'substitution', or 'interstitial'"
                    }
                },
                "required": ["defect_type"]
            },
            domain="defects",
            handler="_query_defect_physics"
        ),
        ToolDefinition(
            name="query_substitution_rules",
            description="Query substitution/doping rules for a host material",
            parameters={
                "type": "object",
                "properties": {
                    "host_element": {
                        "type": "string",
                        "description": "Host element to be substituted (e.g., 'Si', 'Ga')"
                    }
                },
                "required": ["host_element"]
            },
            domain="defects",
            handler="_query_substitution_rules"
        ),
        ToolDefinition(
            name="calculate_defect_count",
            description="Calculate number of defects needed for a target concentration",
            parameters={
                "type": "object",
                "properties": {
                    "n_atoms": {
                        "type": "integer",
                        "description": "Total number of atoms in supercell"
                    },
                    "target_concentration": {
                        "type": "number",
                        "description": "Target defect concentration (e.g., 0.02 for 2%)"
                    }
                },
                "required": ["n_atoms", "target_concentration"]
            },
            domain="defects",
            handler="_calculate_defect_count"
        ),
    ],

    # ==================== Surfaces Domain ====================
    "surfaces": [
        ToolDefinition(
            name="query_surface_construction",
            description="Retrieve general periodic surface geometry and, for adsorption tasks, molecular placement and composition principles.",
            parameters={
                "type": "object",
                "properties": {"query": {"type": "string", "description": "Task requirements or the geometry/composition problem to repair"}},
                "required": []
            },
            domain="surfaces",
            handler="_query_surface_construction"
        ),
        ToolDefinition(
            name="query_surface_sites",
            description="Query available adsorption sites on a surface facet. Returns site information including fractional offsets for placing adsorbates.",
            parameters={
                "type": "object",
                "properties": {
                    "facet": {
                        "type": "string",
                        "description": "Surface facet (e.g., '(111)', '(100)', '(110)')"
                    },
                    "site_type": {
                        "type": "string",
                        "description": "Optional: Filter by specific site type (e.g., 'top', 'bridge', 'fcc', 'hcp', 'hollow'). If not provided, returns all available sites."
                    }
                },
                "required": ["facet"]
            },
            domain="surfaces",
            handler="_query_surface_sites"
        ),
    ],

    # ==================== Adsorption Domain ====================
    "adsorption": [
        ToolDefinition(
            name="query_adsorbate_chemistry",
            description="Query chemistry, binding properties, and site preferences of an adsorbate",
            parameters={
                "type": "object",
                "properties": {
                    "adsorbate": {
                        "type": "string",
                        "description": "Adsorbate formula (e.g., 'CO', 'H', 'O', 'OH', 'NO')"
                    }
                },
                "required": ["adsorbate"]
            },
            domain="adsorption",
            handler="_query_adsorbate_chemistry"
        ),
        ToolDefinition(
            name="query_coadsorption_interaction",
            description="Query interaction rules between two co-adsorbed species",
            parameters={
                "type": "object",
                "properties": {
                    "adsorbate1": {
                        "type": "string",
                        "description": "First adsorbate formula"
                    },
                    "adsorbate2": {
                        "type": "string",
                        "description": "Second adsorbate formula"
                    }
                },
                "required": ["adsorbate1", "adsorbate2"]
            },
            domain="adsorption",
            handler="_query_coadsorption_interaction"
        ),
        ToolDefinition(
            name="query_adsorption_energies",
            description="Query typical adsorption energies for an adsorbate on different metals",
            parameters={
                "type": "object",
                "properties": {
                    "adsorbate": {
                        "type": "string",
                        "description": "Adsorbate formula"
                    },
                    "metal": {
                        "type": "string",
                        "description": "Metal symbol (optional)"
                    }
                },
                "required": ["adsorbate"]
            },
            domain="adsorption",
            handler="_query_adsorption_energies"
        ),
    ],

    # ==================== MOF Domain ====================
    "mof": [
        ToolDefinition(
            name="query_mof_topology",
            description="Query information about a MOF topology (connectivity, typical structures)",
            parameters={
                "type": "object",
                "properties": {
                    "topology": {
                        "type": "string",
                        "description": "Topology code (e.g., 'pcu', 'dia', 'sql', 'nbo')"
                    }
                },
                "required": ["topology"]
            },
            domain="mof",
            handler="_query_mof_topology"
        ),
        ToolDefinition(
            name="query_mof_linker",
            description="Query information about a MOF linker molecule",
            parameters={
                "type": "object",
                "properties": {
                    "linker": {
                        "type": "string",
                        "description": "Linker name or abbreviation (e.g., 'BDC', 'BTC', 'NDC')"
                    }
                },
                "required": ["linker"]
            },
            domain="mof",
            handler="_query_mof_linker"
        ),
        ToolDefinition(
            name="query_mof_sbu",
            description="Query information about a MOF secondary building unit (SBU)",
            parameters={
                "type": "object",
                "properties": {
                    "metal": {
                        "type": "string",
                        "description": "Metal in the SBU (e.g., 'Zn', 'Cu', 'Zr')"
                    }
                },
                "required": ["metal"]
            },
            domain="mof",
            handler="_query_mof_sbu"
        ),
    ],

    # ==================== Common Tools (Always Available) ====================
    "_common": [
        ToolDefinition(
            name="query_pymatgen_api",
            description="Query pymatgen API usage patterns for structure manipulation",
            parameters={
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "description": "Operation type (e.g., 'structure_creation', 'supercell', 'slab_generation', 'add_site', 'file_io')"
                    }
                },
                "required": ["operation"]
            },
            domain="_common",
            handler="_query_pymatgen_api"
        ),
        ToolDefinition(
            name="query_physical_constraints",
            description="Query physical constraints for validating structures (minimum distances, vacuum size, etc.)",
            parameters={
                "type": "object",
                "properties": {
                    "constraint_type": {
                        "type": "string",
                        "description": "Type of constraint (optional, returns all if not specified)"
                    }
                },
                "required": []
            },
            domain="_common",
            handler="_query_physical_constraints"
        ),
        ToolDefinition(
            name="query_element_properties",
            description="Query properties of an element (atomic radius, electronegativity, oxidation states)",
            parameters={
                "type": "object",
                "properties": {
                    "element": {
                        "type": "string",
                        "description": "Element symbol (e.g., 'Fe', 'O', 'C')"
                    }
                },
                "required": ["element"]
            },
            domain="_common",
            handler="_query_element_properties"
        ),
    ],
}


class ToolRegistry:
    """
    Registry for dynamically generating tool definitions based on active domains.

    This class manages tool definitions and generates the appropriate
    tool list for LLM function calling based on which knowledge domains
    are active for the current task.
    """

    def __init__(self):
        """Initialize with default domain tools."""
        self._domain_tools: Dict[str, List[ToolDefinition]] = {}
        # Deep copy to avoid modifying the original
        for domain, tools in DOMAIN_TOOLS.items():
            self._domain_tools[domain] = list(tools)
        # Hierarchy names must resolve to the domains used by tool definitions.
        aliases = {
            "L2_bulk": ["materials"],
            "L2_defect": ["defects"],
            "L2_surface": ["surfaces"],
            "L2_framework": ["mof"],
            "L3_catalysis": ["adsorption"],
        }
        for domain, legacy_domains in aliases.items():
            self._domain_tools[domain] = [
                tool for legacy in legacy_domains
                for tool in self._domain_tools.get(legacy, [])
            ]
        self.register_tool("L2_atom_operations", ToolDefinition(
            name="query_atom_operation",
            description="Query reference-preserving atomic edit knowledge and a parameterized pymatgen recipe.",
            parameters={"type": "object", "properties": {"operation": {"type": "string"}}, "required": ["operation"]},
            domain="L2_atom_operations",
            handler="_query_atom_operation",
        ))

    def get_tools_for_domains(self, domains: List[str]) -> List[Dict]:
        """
        Get tool definitions for a list of domains.

        Args:
            domains: List of domain names

        Returns:
            List of tool definitions in LLM-compatible format
        """
        tools = []
        seen_names = set()

        # Always include common tools first
        for tool_def in self._domain_tools.get("_common", []):
            if tool_def.name not in seen_names:
                tools.append(self._to_dict(tool_def))
                seen_names.add(tool_def.name)

        # Add domain-specific tools
        for domain in domains:
            for tool_def in self._domain_tools.get(domain, []):
                if tool_def.name not in seen_names:
                    tools.append(self._to_dict(tool_def))
                    seen_names.add(tool_def.name)

        return tools

    def get_all_tools(self) -> List[Dict]:
        """Get all registered tool definitions."""
        tools = []
        seen_names = set()

        for domain_tools in self._domain_tools.values():
            for tool_def in domain_tools:
                if tool_def.name not in seen_names:
                    tools.append(self._to_dict(tool_def))
                    seen_names.add(tool_def.name)

        return tools

    def _to_dict(self, tool_def: ToolDefinition) -> Dict:
        """Convert ToolDefinition to dictionary format for LLM."""
        return {
            "name": tool_def.name,
            "description": tool_def.description,
            "parameters": tool_def.parameters,
        }

    def get_handler_name(self, tool_name: str) -> Optional[str]:
        """
        Get the handler method name for a tool.

        Args:
            tool_name: Name of the tool

        Returns:
            Handler method name or None if not found
        """
        for domain_tools in self._domain_tools.values():
            for tool_def in domain_tools:
                if tool_def.name == tool_name:
                    return tool_def.handler
        return None

    def register_tool(self, domain: str, tool_def: ToolDefinition):
        """
        Register a new tool for a domain.

        Args:
            domain: Domain name
            tool_def: Tool definition
        """
        if domain not in self._domain_tools:
            self._domain_tools[domain] = []
        self._domain_tools[domain].append(tool_def)

    def list_tools_for_domain(self, domain: str) -> List[str]:
        """List tool names for a specific domain."""
        return [t.name for t in self._domain_tools.get(domain, [])]

    def list_all_domains(self) -> List[str]:
        """List all domains that have tools registered."""
        return [d for d in self._domain_tools.keys() if d != "_common"]
