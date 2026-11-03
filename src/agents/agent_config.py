"""
Agent Configuration: Dynamic configuration system for multi-agent components.

Follows Data-Code Separation principle - all agent parameters are loaded
from external configuration files rather than hardcoded in source code.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class AgentConfig:
    """
    Dynamic configuration for agents.

    Loads validation rules, thresholds, and templates from external config files,
    eliminating hardcoded values in agent implementations.

    Usage:
        config = AgentConfig.load()

        # Get validation threshold
        min_dist = config.get_min_distance("metals")  # → 2.0

        # Get common issues for prompt
        issues = config.get_common_issues()

        # Get structure types for prompt
        structure_types = config.get_structure_types()
    """

    _instance = None
    _config: Dict[str, Any] = {}

    def __new__(cls):
        """Singleton pattern for global config access."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._config = {}
            cls._instance._loaded = False
        return cls._instance

    @classmethod
    def load(cls, config_path: Optional[Path] = None) -> 'AgentConfig':
        """
        Load configuration from file.

        Args:
            config_path: Path to config file. If None, uses default location.

        Returns:
            AgentConfig instance
        """
        instance = cls()

        if instance._loaded and config_path is None:
            return instance

        # Default config path
        if config_path is None:
            # Try multiple locations
            possible_paths = [
                Path(__file__).parent.parent.parent / "config" / "agent_config.json",
                Path.cwd() / "config" / "agent_config.json",
            ]
            for path in possible_paths:
                if path.exists():
                    config_path = path
                    break

        if config_path and config_path.exists():
            with open(config_path) as f:
                instance._config = json.load(f)
            instance._loaded = True
            logger.info(f"Loaded agent config from {config_path}")
        else:
            instance._config = cls._get_defaults()
            instance._loaded = True
            logger.warning("Using default agent config (no config file found)")

        return instance

    @staticmethod
    def _get_defaults() -> Dict[str, Any]:
        """Get default configuration if no file is found."""
        return {
            "physical_validation": {
                "min_interatomic_distance": {
                    "default": 1.2,
                    "by_bond_type": {
                        "metallic": 2.0,
                        "ionic": 1.8,
                        "covalent": 1.2
                    }
                },
                "energy_per_atom_range": {"min": -15.0, "max": 0.0},
                "charge_tolerance": 0.1
            },
            "structure_types": [
                "rocksalt", "zincblende", "wurtzite", "diamond", "fluorite",
                "perovskite", "spinel", "corundum", "rutile"
            ],
            "operations": [
                "vacancy", "substitution", "interstitial", "adsorption",
                "surface_generation", "interface_construction"
            ],
            "common_issues": []
        }

    # =========================================================================
    # Physical Validation Getters
    # =========================================================================

    def get_min_distance(self, material_class: Optional[str] = None,
                         bond_type: Optional[str] = None) -> float:
        """
        Get minimum interatomic distance threshold.

        Args:
            material_class: Optional material class (metals, oxides, etc.)
            bond_type: Optional bond type (metallic, ionic, covalent)

        Returns:
            Minimum distance in Angstroms
        """
        phys = self._config.get("physical_validation", {})
        dist_config = phys.get("min_interatomic_distance", {})

        # Try bond type first
        if bond_type:
            by_bond = dist_config.get("by_bond_type", {})
            if bond_type in by_bond:
                return by_bond[bond_type]

        # Try material class
        if material_class:
            by_material = dist_config.get("by_material_class", {})
            if material_class in by_material:
                return by_material[material_class]

        # Return default
        return dist_config.get("default", 1.2)

    def get_energy_range(self) -> tuple:
        """Get valid energy per atom range."""
        phys = self._config.get("physical_validation", {})
        energy = phys.get("energy_per_atom_range", {})
        return (energy.get("min", -15.0), energy.get("max", 0.0))

    def get_charge_tolerance(self) -> float:
        """Get charge balance tolerance."""
        phys = self._config.get("physical_validation", {})
        return phys.get("charge_tolerance", 0.1)

    # =========================================================================
    # Structure and Operation Getters
    # =========================================================================

    def get_structure_types(self) -> List[str]:
        """Get list of known structure types."""
        return self._config.get("structure_types", [])

    def get_operations(self) -> List[str]:
        """Get list of known operations."""
        return self._config.get("operations", [])

    def get_common_issues(self, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get common issues for agent prompts.

        Args:
            category: Optional filter by category (physical_validity, task_alignment, etc.)

        Returns:
            List of issue definitions
        """
        issues = self._config.get("common_issues", [])

        if category:
            issues = [i for i in issues if i.get("category") == category]

        return issues

    def get_validation_criteria(self, level: Optional[str] = None) -> List[str]:
        """Get validation criteria by level."""
        criteria = self._config.get("validation_criteria", {})

        if level:
            return criteria.get(level, [])

        # Return all criteria flattened
        all_criteria = []
        for level_criteria in criteria.values():
            all_criteria.extend(level_criteria)
        return all_criteria

    # =========================================================================
    # Prompt Building Helpers
    # =========================================================================

    def format_structure_types_for_prompt(self) -> str:
        """Format structure types as prompt text."""
        types = self.get_structure_types()
        return ", ".join(types)

    def format_operations_for_prompt(self) -> str:
        """Format operations as prompt text."""
        ops = self.get_operations()
        return ", ".join(ops)

    def format_common_issues_for_prompt(self) -> str:
        """Format common issues as prompt text."""
        issues = self.get_common_issues()
        if not issues:
            return ""

        lines = []
        for issue in issues:
            issue_type = issue.get("type", "unknown")
            suggestion = issue.get("suggestion", "")
            lines.append(f"- **{issue_type}**: {suggestion}")

        return "\n".join(lines)

    def format_validation_rules_for_prompt(self) -> str:
        """Describe the paper's physical checks without legacy global proxies.

        This changes generation instructions only. Validator thresholds and
        legacy configuration accessors remain independent and unchanged.
        """
        return """- Enforce the explicit task composition, atom counts, sites, orientations, and geometric constraints; preserve requirements already satisfied during repair.
- Verify finite coordinates and a nonsingular periodic cell. Check periodic minimum-image distances using element-pair and bonding-context constraints; distinguish molecular bonds from steric overlaps.
- Assess MLIP energy against a provenance-bearing task- or same-composition reference, or a justified task-conditioned range. If no reference is available, report missing evidence instead of inventing an absolute energy target.
- For ionic materials, verify justified oxidation states, stoichiometry, and charge balance. When oxidation states are not applicable or established, use species, stoichiometry, and local coordination checks without inventing ionic charges.
- Use measured geometric quantities and explicit units in construction checks; a requested surface-normal height is distinct from a three-dimensional bond distance."""

    # =========================================================================
    # Raw Config Access
    # =========================================================================

    def get(self, key: str, default: Any = None) -> Any:
        """Get raw config value by key."""
        return self._config.get(key, default)

    def get_nested(self, *keys, default: Any = None) -> Any:
        """Get nested config value."""
        value = self._config
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
            else:
                return default
            if value is None:
                return default
        return value

    # =========================================================================
    # Prompt Template Access
    # =========================================================================

    _prompt_templates: Dict[str, Any] = {}
    _templates_loaded: bool = False

    def get_prompt_template(self, agent_name: str) -> Optional[Dict[str, Any]]:
        """Get prompt template for an agent.

        Args:
            agent_name: Agent name (planner, generator, critic, reflection, refiner)

        Returns:
            Template dict with 'principles', 'role', 'exemplar_slots', etc.
            None if no template found.
        """
        if not self._templates_loaded:
            self._load_prompt_templates()
        return self._prompt_templates.get(agent_name)

    def _load_prompt_templates(self) -> None:
        """Load prompt templates from config/prompt_templates.json."""
        possible_paths = [
            Path(__file__).parent.parent.parent / "config" / "prompt_templates.json",
            Path.cwd() / "config" / "prompt_templates.json",
        ]
        for path in possible_paths:
            if path.exists():
                try:
                    with open(path) as f:
                        data = json.load(f)
                    # Strip metadata key
                    self._prompt_templates = {
                        k: v for k, v in data.items() if not k.startswith('_')
                    }
                    self._templates_loaded = True
                    logger.info(f"Loaded prompt templates from {path}")
                    return
                except Exception as e:
                    logger.error(f"Failed to load prompt templates: {e}")

        self._templates_loaded = True  # Don't retry
        logger.info("No prompt templates found, agents will use built-in prompts")


# Global config instance
agent_config = AgentConfig.load()
