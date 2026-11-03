"""
Generic Knowledge Store for RAG modules.

This module provides a data-agnostic knowledge store that loads data from JSON files.
The RAG code is separated from the data content, allowing easy extension and maintenance.

Enhanced with:
- Principle-based estimation when exact data is unavailable
- Material family detection for unknown materials
- Fallback to estimation rules and physical principles
- Support for multiple data directories (general + task-specific)
"""

import json
import logging
import re
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional, Any, Union

from .categories import canonical_category

logger = logging.getLogger(__name__)

# Default data directory (general knowledge)
DEFAULT_DATA_DIR = Path(__file__).parent.parent.parent / "data" / "knowledge"


class KnowledgeStore:
    """
    Generic knowledge store that loads and queries JSON data files.

    Separates RAG code from data content for better maintainability and extensibility.
    Supports loading from multiple directories (general knowledge + task-specific).
    """

    def __init__(self, data_dir: Optional[Union[Path, str]] = None,
                 additional_dirs: Optional[List[Union[Path, str]]] = None):
        """
        Initialize the knowledge store.

        Args:
            data_dir: Primary knowledge data directory. Uses default if not provided.
            additional_dirs: Additional directories to load data from (e.g., task-specific).
                            Data from additional dirs takes precedence over primary.
        """
        self.data_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
        self.additional_dirs = [Path(d) for d in (additional_dirs or [])]
        self._cache: Dict[str, Dict] = {}
        self._load_all_data()

    def _load_all_data(self):
        """Load all JSON data files from all data directories."""
        # Load from primary directory first
        self._load_from_directory(self.data_dir)

        # Load from additional directories (these take precedence)
        for additional_dir in self.additional_dirs:
            self._load_from_directory(additional_dir)

    def _load_from_directory(self, data_dir: Path):
        """Load all JSON data files from a single directory."""
        if not data_dir.exists():
            logger.warning(f"Knowledge data directory not found: {data_dir}")
            return

        # Load all JSON files recursively
        # A canonical file wins over a legacy alias within one directory.
        # Additional directories still override the primary directory.
        def path_category(path):
            return ".".join(path.relative_to(data_dir).with_suffix("").parts)

        files = sorted(data_dir.rglob("*.json"))
        files.sort(key=lambda p: canonical_category(path_category(p)) == path_category(p))
        for json_file in files:
            if json_file.name.startswith(("_", ".")):
                continue
            try:
                key = canonical_category(path_category(json_file))

                with open(json_file, "r", encoding="utf-8") as f:
                    self._cache[key] = json.load(f)

                logger.debug(f"Loaded knowledge file: {key} from {data_dir}")
            except Exception as e:
                logger.error(f"Failed to load {json_file}: {e}")

    def get_data(self, category: str) -> Dict:
        """
        Get all data for a category.

        Args:
            category: Data category key (e.g., 'materials.metal_properties')

        Returns:
            Dictionary of data for the category, or empty dict if not found.
        """
        return self._cache.get(canonical_category(category), {})

    def query(self, category: str, key: str) -> Optional[Dict]:
        """
        Query a specific item from a category.

        Args:
            category: Data category key
            key: Item key within the category

        Returns:
            Item data or None if not found.
        """
        data = self.get_data(category)
        return data.get(key)

    def query_with_fallback(self, category: str, key: str,
                           fallback_categories: Optional[List[str]] = None) -> Optional[Dict]:
        """
        Query with fallback to other categories if not found.

        Args:
            category: Primary category to search
            key: Item key to find
            fallback_categories: List of fallback categories to try

        Returns:
            Item data or None if not found in any category.
        """
        result = self.query(category, key)
        if result is not None:
            return result

        if fallback_categories:
            for fallback in fallback_categories:
                result = self.query(fallback, key)
                if result is not None:
                    return result

        return None

    def list_keys(self, category: str) -> List[str]:
        """
        List all keys in a category.

        Args:
            category: Data category key

        Returns:
            List of keys in the category.
        """
        data = self.get_data(category)
        return list(data.keys())

    def search(self, category: str, query: str,
               fields: Optional[List[str]] = None) -> List[Dict]:
        """
        Search for items containing the query string.

        Args:
            category: Data category to search
            query: Search string (case-insensitive)
            fields: Specific fields to search in (searches all if None)

        Returns:
            List of matching items with their keys.
        """
        data = self.get_data(category)
        query_lower = query.lower()
        results = []

        for key, item in data.items():
            if self._item_matches(item, query_lower, fields) or query_lower in key.lower():
                results.append({"key": key, "data": item})

        return results

    def _item_matches(self, item: Any, query: str, fields: Optional[List[str]]) -> bool:
        """Check if an item matches the search query."""
        if isinstance(item, dict):
            for field_key, value in item.items():
                if fields and field_key not in fields:
                    continue
                if self._value_matches(value, query):
                    return True
        elif isinstance(item, (list, tuple)):
            for value in item:
                if self._value_matches(value, query):
                    return True
        else:
            return self._value_matches(item, query)
        return False

    def _value_matches(self, value: Any, query: str) -> bool:
        """Check if a value matches the search query."""
        if isinstance(value, str):
            return query in value.lower()
        elif isinstance(value, (list, tuple)):
            return any(self._value_matches(v, query) for v in value)
        elif isinstance(value, dict):
            return self._item_matches(value, query, None)
        return False

    def get_similar_keys(self, category: str, key: str, max_results: int = 5) -> List[str]:
        """
        Find similar keys using simple string matching.

        Args:
            category: Data category
            key: Key to find similar matches for
            max_results: Maximum number of results to return

        Returns:
            List of similar keys.
        """
        all_keys = self.list_keys(category)
        key_lower = key.lower()

        # Score keys by similarity
        scored = []
        for k in all_keys:
            k_lower = k.lower()
            # Simple scoring: exact match > starts with > contains > partial overlap
            if k_lower == key_lower:
                score = 100
            elif k_lower.startswith(key_lower) or key_lower.startswith(k_lower):
                score = 80
            elif key_lower in k_lower or k_lower in key_lower:
                score = 60
            else:
                # Count common characters
                common = sum(1 for c in key_lower if c in k_lower)
                score = common * 10 / max(len(key_lower), 1)

            if score > 0:
                scored.append((k, score))

        # Sort by score descending
        scored.sort(key=lambda x: x[1], reverse=True)
        return [k for k, _ in scored[:max_results]]

    def format_as_text(self, category: str, key: str,
                       include_key: bool = True) -> str:
        """
        Format an item as human-readable text.

        Args:
            category: Data category
            key: Item key
            include_key: Whether to include the key in output

        Returns:
            Formatted text representation.
        """
        item = self.query(category, key)
        if item is None:
            available = self.list_keys(category)
            similar = self.get_similar_keys(category, key, 3)
            msg = f"'{key}' not found in {category}."
            if similar:
                msg += f" Similar: {similar}"
            elif available:
                msg += f" Available: {available[:10]}"
            return msg

        lines = []
        if include_key:
            lines.append(f"{key}:")

        lines.extend(self._format_dict(item, indent=2))
        return "\n".join(lines)

    def _format_dict(self, data: Any, indent: int = 0) -> List[str]:
        """Recursively format a dictionary as text lines."""
        lines = []
        prefix = " " * indent

        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, dict):
                    lines.append(f"{prefix}{key}:")
                    lines.extend(self._format_dict(value, indent + 2))
                elif isinstance(value, list):
                    if all(isinstance(v, (str, int, float)) for v in value):
                        lines.append(f"{prefix}{key}: {value}")
                    else:
                        lines.append(f"{prefix}{key}:")
                        for item in value:
                            lines.extend(self._format_dict(item, indent + 2))
                else:
                    lines.append(f"{prefix}{key}: {value}")
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    lines.extend(self._format_dict(item, indent))
                else:
                    lines.append(f"{prefix}- {item}")
        else:
            lines.append(f"{prefix}{data}")

        return lines

    @property
    def categories(self) -> List[str]:
        """List all available data categories."""
        return list(self._cache.keys())

    def reload(self):
        """Reload all data from files."""
        self._cache.clear()
        self._load_all_data()

    # ==================== Enhanced Query Methods ====================

    def query_with_principles(self, category: str, key: str,
                              context: Optional[str] = None) -> Dict[str, Any]:
        """
        Query with fallback to estimation principles when exact data is unavailable.

        Args:
            category: Data category key (e.g., 'materials.crystal_structures')
            key: Item key to find (e.g., 'InP')
            context: Optional context for better principle selection (e.g., 'defect', 'adsorption')

        Returns:
            Dictionary with either exact data or estimation guidance.
        """
        # 1. Try exact match in static data
        result = self.query(category, key)
        if result is not None:
            return {
                "source": "static_data",
                "data": result,
                "exact_match": True
            }

        # 2. Try to find in related categories
        related_categories = self._get_related_categories(category)
        for related_cat in related_categories:
            result = self.query(related_cat, key)
            if result is not None:
                return {
                    "source": "related_category",
                    "category": related_cat,
                    "data": result,
                    "exact_match": True
                }

        # 3. Try to find in material families (for materials)
        if "material" in category or "crystal" in category:
            family_data = self._find_in_material_families(key)
            if family_data:
                return {
                    "source": "material_family",
                    "data": family_data,
                    "exact_match": False,
                    "note": f"Data from material family: {family_data.get('family', 'unknown')}"
                }

        # 4. Return estimation principles
        principles = self._get_estimation_principles(category, key, context)
        return {
            "source": "estimation_principles",
            "exact_match": False,
            "warning": f"No exact data for '{key}' in {category}. Use these principles to estimate.",
            "principles": principles,
            "similar_materials": self.get_similar_keys(category, key, 5)
        }

    def _get_related_categories(self, category: str) -> List[str]:
        """Get related categories for fallback search."""
        related = []

        if "crystal" in category or "material" in category:
            related.extend([
                "L2_structures.bulk.crystal_structures",
                "L2_structures.bulk.metal_properties",
                "L2_structures.bulk.structure_types"
            ])
        if "adsorption" in category or "surface" in category:
            related.extend([
                "catalysis.adsorption_energies",
                "surfaces.surface_sites",
                "adsorbates.adsorbate_chemistry"
            ])
        if "defect" in category:
            related.extend([
                "defects.defect_physics",
                "defects.formation_energies",
                "defects.substitution_rules"
            ])

        # Remove the original category
        return [c for c in related if c != category]

    def _find_in_material_families(self, material: str) -> Optional[Dict]:
        """
        Try to find material in chemical_trends material families.

        Args:
            material: Material formula (e.g., 'InP', 'ZnO')

        Returns:
            Material data from family if found, None otherwise.
        """
        chemical_trends = self.get_data("principles.chemical_trends")
        if not chemical_trends:
            return None

        material_families = chemical_trends.get("material_families", {})

        for family_name, family_data in material_families.items():
            members = family_data.get("members", {})
            if material in members:
                member_data = deepcopy(members[material])
                member_data["family"] = family_name
                member_data["family_properties"] = {
                    k: v for k, v in family_data.items()
                    if k not in ["members", "description"]
                }
                return member_data

        return None

    def _get_estimation_principles(self, category: str, key: str,
                                   context: Optional[str] = None) -> Dict[str, Any]:
        """
        Get relevant estimation principles for the query.

        Args:
            category: Data category
            key: Item key
            context: Optional context for better principle selection

        Returns:
            Dictionary of relevant principles.
        """
        principles = {}

        # Load estimation rules
        estimation_rules = self.get_data("principles.estimation_rules")
        physical_principles = self.get_data("principles.physical_principles")
        chemical_trends = self.get_data("principles.chemical_trends")

        # Select relevant principles based on category and context
        if "crystal" in category or "material" in category or "lattice" in str(key).lower():
            if estimation_rules:
                principles["lattice_estimation"] = estimation_rules.get(
                    "lattice_constant_estimation", {})
                principles["structure_prediction"] = estimation_rules.get(
                    "structure_type_prediction", {})
            if physical_principles:
                principles["crystal_principles"] = physical_principles.get(
                    "crystal_structure_principles", {})
            if chemical_trends:
                principles["periodic_trends"] = chemical_trends.get(
                    "periodic_trends", {})
                # Try to identify material family
                family = self._identify_material_family(key)
                if family and family in chemical_trends.get("material_families", {}):
                    principles["material_family"] = chemical_trends["material_families"][family]

        if "adsorption" in category or context == "adsorption":
            if estimation_rules:
                principles["adsorption_estimation"] = estimation_rules.get(
                    "adsorption_energy_estimation", {})
            if physical_principles:
                principles["surface_chemistry"] = physical_principles.get(
                    "surface_chemistry_principles", {})

        if "defect" in category or context == "defect":
            if estimation_rules:
                principles["defect_estimation"] = estimation_rules.get(
                    "defect_formation_energy_estimation", {})
            if physical_principles:
                principles["defect_physics"] = physical_principles.get(
                    "defect_physics_principles", {})
            if chemical_trends:
                principles["dopant_compatibility"] = chemical_trends.get(
                    "dopant_compatibility", {})

        # Always include usage guidelines if available
        if estimation_rules and "usage_guidelines" in estimation_rules:
            principles["usage_guidelines"] = estimation_rules["usage_guidelines"]

        return principles

    def _identify_material_family(self, material: str) -> Optional[str]:
        """
        Identify the material family based on formula.

        Args:
            material: Material formula

        Returns:
            Family name or None.
        """
        # Common patterns for material families
        iii_v_elements = {"Ga", "In", "Al", "As", "P", "N", "Sb"}
        ii_vi_elements = {"Zn", "Cd", "Hg", "O", "S", "Se", "Te"}
        transition_metals = {"Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
                           "Zr", "Nb", "Mo", "Ru", "Rh", "Pd", "Ag", "Hf", "Ta", "W",
                           "Re", "Os", "Ir", "Pt", "Au"}

        # Parse formula to get elements
        elements = set(re.findall(r'[A-Z][a-z]?', material))

        # Check for III-V semiconductors
        if elements & {"Ga", "In", "Al"} and elements & {"As", "P", "N", "Sb"}:
            return "III_V_semiconductors"

        # Check for II-VI semiconductors
        if elements & {"Zn", "Cd", "Hg"} and elements & {"O", "S", "Se", "Te"}:
            return "II_VI_semiconductors"

        # Check for transition metal oxides
        if elements & transition_metals and "O" in elements:
            return "transition_metal_oxides"

        # Check for perovskites (ABO3 pattern)
        if len(elements) == 3 and "O" in elements:
            # Could be perovskite
            return "perovskites"

        # Check for layered materials
        layered_patterns = {"MoS", "WS", "MoSe", "WSe", "graphene", "hBN"}
        if any(p in material for p in layered_patterns):
            return "layered_materials_2D"

        # Check for pure metals
        if len(elements) == 1 and elements & transition_metals:
            # Determine FCC, BCC, or HCP based on element
            fcc_metals = {"Cu", "Ag", "Au", "Al", "Ni", "Pd", "Pt", "Rh", "Ir"}
            bcc_metals = {"Fe", "Cr", "Mo", "W", "V", "Nb", "Ta"}
            hcp_metals = {"Ti", "Zr", "Co", "Zn", "Mg", "Ru", "Os"}

            elem = list(elements)[0]
            if elem in fcc_metals:
                return "fcc_metals"
            elif elem in bcc_metals:
                return "bcc_metals"
            elif elem in hcp_metals:
                return "hcp_metals"

        return None

    def get_principle(self, principle_category: str, principle_name: str) -> Optional[Dict]:
        """
        Get a specific principle from the principles knowledge base.

        Args:
            principle_category: Category in principles (e.g., 'estimation_rules', 'physical_principles')
            principle_name: Name of the principle

        Returns:
            Principle data or None.
        """
        data = self.get_data(f"principles.{principle_category}")
        if data:
            return data.get(principle_name)
        return None

    def list_principles(self) -> Dict[str, List[str]]:
        """
        List all available principles.

        Returns:
            Dictionary mapping principle categories to their available principles.
        """
        result = {}
        principle_names = {"physical_principles", "chemical_trends", "estimation_rules",
                           "interface_principles", "material_inference_rules"}
        for category in self.categories:
            if category.startswith("L1_foundation.") and category.rsplit(".", 1)[-1] in principle_names:
                principle_name = category.rsplit(".", 1)[-1]
                data = self.get_data(category)
                if data:
                    # Filter out metadata keys
                    keys = [k for k in data.keys() if not k.startswith("_")]
                    result[principle_name] = keys
        return result

    def format_principles_for_llm(self, principles: Dict[str, Any],
                                  max_length: int = 2000) -> str:
        """
        Format principles as a concise string for LLM context.

        Args:
            principles: Dictionary of principles
            max_length: Maximum length of output

        Returns:
            Formatted string.
        """
        lines = ["ESTIMATION PRINCIPLES (use when exact data unavailable):"]

        for category, content in principles.items():
            if category == "usage_guidelines":
                continue

            lines.append(f"\n## {category.replace('_', ' ').title()}")

            if isinstance(content, dict):
                # Extract key information
                if "description" in content:
                    lines.append(f"  {content['description']}")

                if "methods" in content:
                    for method in content["methods"][:3]:  # Limit to 3 methods
                        if isinstance(method, dict):
                            name = method.get("name", "")
                            desc = method.get("description", "")
                            formula = method.get("formula", "")
                            lines.append(f"  - {name}: {desc}")
                            if formula:
                                lines.append(f"    Formula: {formula}")

                if "rules" in content and isinstance(content["rules"], list):
                    for rule in content["rules"][:3]:
                        if isinstance(rule, str):
                            lines.append(f"  - {rule}")
                        elif isinstance(rule, dict):
                            lines.append(f"  - {rule.get('name', '')}: {rule.get('rule', '')}")

        result = "\n".join(lines)

        # Truncate if too long
        if len(result) > max_length:
            result = result[:max_length - 50] + "\n\n[Truncated - see full principles in knowledge base]"

        return result


# Singleton instance for shared access
_default_store: Optional[KnowledgeStore] = None


def get_knowledge_store(data_dir: Optional[Path] = None) -> KnowledgeStore:
    """
    Get the default knowledge store instance.

    Args:
        data_dir: Optional custom data directory

    Returns:
        KnowledgeStore instance.
    """
    global _default_store

    if data_dir is not None:
        # Return a new instance for custom data directory
        return KnowledgeStore(data_dir)

    if _default_store is None:
        _default_store = KnowledgeStore()

    return _default_store
