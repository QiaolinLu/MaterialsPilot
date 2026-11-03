"""
Knowledge Registry

Central registry for managing knowledge domains and automatic selection based on task type.
Provides the mapping between tasks and their required knowledge domains.

Supports both:
- Three-level system (L1_foundation, L2_structures, L3_applications)
- Legacy system (core, domain, task) for backward compatibility
"""

import re
from copy import deepcopy
import logging
from typing import Dict, List, Set, Optional, Any
from src.knowledge.categories import canonical_category

from .domains import (
    KnowledgeLevel,
    DomainConfig,
    DOMAIN_CONFIGS,
    TASK_DOMAIN_MAP,
    TASK_KNOWLEDGE_MAP,
    get_core_domains,
    get_always_loaded_domains,
)

logger = logging.getLogger(__name__)


class KnowledgeRegistry:
    """
    Registry for knowledge domains with automatic selection based on task type.

    This class manages:
    - Task type to domain mappings (both new hierarchical and legacy systems)
    - Automatic domain resolution including dependencies
    - Task type inference from natural language

    Supports two knowledge organization systems:
    1. New Hierarchical System (L1-L3):
       - L1_foundation: Universal data and physical/chemical principles (always loaded)
       - L2_structures: Structure-type specific (bulk, surface, defect, interface, framework)
       - L3_applications: Application-specific (catalysis, gas_storage, electronics)

    2. Legacy System (for backward compatibility):
       - core: Core knowledge (elements, materials, symmetry, principles)
       - domain: Domain-specific knowledge (defects, surfaces, adsorption, mof, interfaces)
       - task: Task-specific overrides
    """

    def __init__(self, use_hierarchical: bool = True):
        """
        Initialize the registry with default configurations.

        Args:
            use_hierarchical: If True, use new L1-L3 hierarchical system.
                            If False, use legacy core/domain/task system.
        """
        self._domain_configs = deepcopy(DOMAIN_CONFIGS)
        self._task_domain_map = deepcopy(TASK_DOMAIN_MAP)
        self._task_knowledge_map = deepcopy(TASK_KNOWLEDGE_MAP)
        self._registered_tasks = {}
        self._core_domains = get_core_domains()
        self._always_loaded = get_always_loaded_domains()
        self._use_hierarchical = use_hierarchical

    def get_domains_for_task(self, task_type: str) -> List[str]:
        """
        Get all relevant knowledge domains for a task type.

        Uses the new hierarchical system (L1-L3) by default, with fallback
        to legacy system for backward compatibility.

        Automatically includes:
        1. Always-loaded L1 foundation (including API and constraint knowledge)
        2. Task-specific domains (L2, L3 OR domain-specific)
        3. Dependencies of task-specific domains

        Args:
            task_type: Type of task (e.g., 'defect', 'adsorption', 'mof')

        Returns:
            List of domain names to load
        """
        if self._use_hierarchical:
            return self._get_domains_hierarchical(task_type)
        else:
            return self._get_domains_legacy(task_type)

    def resolve_domains(self, domains: List[str]) -> List[str]:
        """Resolve transitive dependencies, detecting invalid registration early."""
        resolved, visiting = [], set()

        def visit(name):
            if name in visiting:
                raise ValueError(f"Knowledge domain dependency cycle at {name}")
            if name in resolved:
                return
            if name not in self._domain_configs:
                raise ValueError(f"Unknown knowledge domain: {name}")
            visiting.add(name)
            for dep in self._domain_configs[name].dependencies:
                visit(dep)
            visiting.remove(name)
            resolved.append(name)

        for name in domains:
            visit(name)
        return resolved

    def _get_domains_hierarchical(self, task_type: str) -> List[str]:
        key = task_type.strip().lower()
        if key in self._registered_tasks:
            selected = self._registered_tasks[key]
        else:
            config = self.get_task_knowledge_config(key) or {}
            selected = config.get("L2", []) + config.get("L3", [])
        return self.resolve_domains([*self._always_loaded, *selected])

    def _get_domains_legacy(self, task_type: str) -> List[str]:
        """Get domains using legacy core/domain/task system."""
        domains: Set[str] = set()

        # Always include core domains
        domains.update(self._core_domains)

        # Add task-specific domains
        task_domains = self._task_domain_map.get(task_type.lower(), [])
        for domain in task_domains:
            domains.add(domain)
            # Add dependencies recursively
            self._add_dependencies(domain, domains)

        return sorted(domains)

    def _add_dependencies(self, domain: str, domains: Set[str]):
        """Recursively add domain dependencies."""
        config = self._domain_configs.get(domain)
        if config and config.dependencies:
            for dep in config.dependencies:
                if dep not in domains:
                    domains.add(dep)
                    self._add_dependencies(dep, domains)

    def get_categories_for_domains(self, domains: List[str]) -> List[str]:
        """
        Get all knowledge categories for a list of domains.

        Args:
            domains: List of domain names

        Returns:
            List of category names (e.g., 'L1_foundation.periodic_table')
        """
        categories = []
        seen = set()

        for domain in domains:
            config = self._domain_configs.get(domain)
            if config:
                for cat in config.categories:
                    cat = canonical_category(cat)
                    if cat not in seen:
                        categories.append(cat)
                        seen.add(cat)

        return categories

    def get_categories_for_task(self, task_type: str) -> List[str]:
        """
        Get all knowledge categories for a task type.

        Uses the new hierarchical system by default.

        Args:
            task_type: Type of task

        Returns:
            List of category names
        """
        return self.get_categories_for_domains(self.get_domains_for_task(task_type))

    def get_knowledge_by_level(self, task_type: str) -> Dict[str, List[str]]:
        """
        Get knowledge categories organized by level for a task type.

        This is useful for understanding what knowledge is loaded at each level.

        Args:
            task_type: Type of task

        Returns:
            Dictionary mapping level names to category lists
        """
        result = {
            "L1_foundation": [],
            "L2_structures": [],
            "L3_applications": [],
        }

        domains = self.get_domains_for_task(task_type)

        for category in self.get_categories_for_domains(domains):
            level = category.split(".", 1)[0]
            if level in result:
                result[level].append(category)

        # Remove duplicates while preserving order
        for key in result:
            result[key] = list(dict.fromkeys(result[key]))

        return result

    def get_task_knowledge_config(self, task_type: str) -> Optional[Dict[str, Any]]:
        """
        Get the knowledge configuration for a task type from TASK_KNOWLEDGE_MAP.

        Args:
            task_type: Type of task

        Returns:
            Task knowledge configuration or None
        """
        key = task_type.strip().lower()
        seen = set()
        while key in self._task_knowledge_map:
            if key in seen:
                raise ValueError(f"Knowledge task inheritance cycle at {key}")
            seen.add(key)
            config = self._task_knowledge_map[key]
            if "inherits" not in config:
                return deepcopy(config)
            key = config["inherits"].strip().lower()
        return None

    def register_domain(self, name: str, config: DomainConfig):
        """
        Register a new knowledge domain.

        Args:
            name: Domain name
            config: Domain configuration
        """
        self._domain_configs[name] = deepcopy(config)

        # Update appropriate lists based on level
        if config.level == KnowledgeLevel.L1_FOUNDATION:
            if name not in self._core_domains:
                self._core_domains.append(name)
            if name not in self._always_loaded:
                self._always_loaded.append(name)

        logger.info(f"Registered domain: {name} (level: {config.level.value})")

    def register_task_mapping(self, task_type: str, domains: List[str]):
        """
        Register a task type to domain mapping.

        Args:
            task_type: Task type name
            domains: List of domain names for this task
        """
        self.resolve_domains(domains)
        key = task_type.strip().lower()
        self._task_domain_map[key] = list(domains)
        self._registered_tasks[key] = list(domains)
        logger.info(f"Registered task mapping: {task_type} -> {domains}")

    def infer_task_type(self, user_input: str) -> Optional[str]:
        """
        Infer task type from natural language input.

        Uses keyword matching to determine the most likely task type.

        Args:
            user_input: Natural language description of the task

        Returns:
            Inferred task type or None if cannot determine
        """
        user_lower = user_input.lower()
        from .atom_operations import infer_operations
        if "atomworld" in user_lower or (
            re.search(r"\batom(?:s)?\b.*\b(?:index|indices)\b", user_lower)
            and infer_operations(user_input)
        ):
            return "atomworld_structure_edit"

        # Priority-ordered keyword matching
        # More specific patterns first
        task_keywords = {
            # Defect tasks
            "vacancy": [r"\bvacanc", r"\bvoid\b"],
            "substitution": [r"\bsubstitut", r"\bdop(?:e|ing|ant)", r"\breplace\b.*\batom"],
            "interstitial": [r"\binterstitial"],
            "defect": [r"\bdefect", r"\bpoint\s+defect"],

            # Surface/adsorption tasks
            "coadsorption": [r"\bco-?adsorb", r"\bmultiple\s+adsorb"],
            "adsorption": [r"\badsorb", r"\badsorbate", r"\bcatalyst", r"\bcatalysis"],
            "surface": [r"\bsurface\b", r"\bslab\b", r"\bfacet"],

            # MOF tasks
            "mof_design": [r"\bmof\s+design", r"\bdesign.*mof"],
            "mof": [r"\bmof\b", r"\bmetal.?organic", r"\bframework", r"\btopology", r"\blinker"],

            # Interface tasks
            "heterostructure": [r"\bheterostructure", r"\bheterojunction"],
            "grain_boundary": [r"\bgrain\s*boundar", r"\bgb\b"],
            "interface": [r"\binterface\b"],
        }

        for task_type, patterns in task_keywords.items():
            for pattern in patterns:
                if re.search(pattern, user_lower):
                    logger.debug(f"Inferred task type '{task_type}' from pattern '{pattern}'")
                    return task_type

        return None

    def get_domain_info(self, domain: str) -> Optional[Dict]:
        """
        Get information about a domain.

        Args:
            domain: Domain name

        Returns:
            Dictionary with domain info or None
        """
        config = self._domain_configs.get(domain)
        if not config:
            return None

        return {
            "name": config.name,
            "level": config.level.value,
            "categories": config.categories,
            "dependencies": config.dependencies,
            "description": config.description,
        }

    def list_domains(self, level: Optional[KnowledgeLevel] = None) -> List[str]:
        """
        List all registered domains.

        Args:
            level: Optional filter by knowledge level

        Returns:
            List of domain names
        """
        if level is None:
            return list(self._domain_configs.keys())

        return [
            name for name, config in self._domain_configs.items()
            if config.level == level
        ]

    def list_task_types(self) -> List[str]:
        """List all registered task types."""
        return sorted(set(self._task_domain_map) | set(self._task_knowledge_map) | set(self._registered_tasks))

    @property
    def core_domains(self) -> List[str]:
        """Get list of core domain names (legacy system)."""
        return self._core_domains.copy()

    @property
    def always_loaded_domains(self) -> List[str]:
        """Get list of always-loaded L1 domain names."""
        return self._always_loaded.copy()

    @property
    def use_hierarchical(self) -> bool:
        """Check if using hierarchical system."""
        return self._use_hierarchical

    @use_hierarchical.setter
    def use_hierarchical(self, value: bool):
        """Set whether to use hierarchical system."""
        self._use_hierarchical = value

    def list_domains_by_level(self) -> Dict[str, List[str]]:
        """
        List all domains organized by their knowledge level.

        Returns:
            Dictionary mapping level names to domain lists
        """
        result = {
            "L1_foundation": [],
            "L2_structures": [],
            "L3_applications": [],
        }

        for name, config in self._domain_configs.items():
            level = config.level
            if level == KnowledgeLevel.L1_FOUNDATION:
                result["L1_foundation"].append(name)
            elif level == KnowledgeLevel.L2_STRUCTURES:
                result["L2_structures"].append(name)
            elif level == KnowledgeLevel.L3_APPLICATIONS:
                result["L3_applications"].append(name)

        return result

    def get_hierarchy_summary(self) -> str:
        """
        Get a human-readable summary of the knowledge hierarchy.

        Returns:
            Formatted string describing the hierarchy
        """
        domains_by_level = self.list_domains_by_level()

        lines = [
            "Knowledge Hierarchy Summary",
            "=" * 40,
            "",
            "Always Loaded:",
            f"  L1 Foundation: {', '.join(domains_by_level['L1_foundation']) or 'None'}",
            "",
            "Task-Specific:",
            f"  L2 Structures: {', '.join(domains_by_level['L2_structures']) or 'None'}",
            f"  L3 Applications: {', '.join(domains_by_level['L3_applications']) or 'None'}",
        ]

        return "\n".join(lines)

    def describe_task_knowledge(self, task_type: str) -> str:
        """
        Get a human-readable description of knowledge loaded for a task.

        Args:
            task_type: Type of task

        Returns:
            Formatted string describing the knowledge
        """
        knowledge_by_level = self.get_knowledge_by_level(task_type)

        lines = [
            f"Knowledge for task: {task_type}",
            "=" * 40,
        ]

        for level, categories in knowledge_by_level.items():
            if categories:
                lines.append(f"\n{level}:")
                for cat in categories:
                    lines.append(f"  - {cat}")

        return "\n".join(lines)
