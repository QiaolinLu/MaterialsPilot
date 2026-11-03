"""Canonical knowledge paths, including aliases used by older task agents."""

PREFIX_ALIASES = {
    "L0_universal.": "L1_foundation.",
    "L1_principles.": "L1_foundation.",
    "elements.": "L1_foundation.",
    "principles.": "L1_foundation.",
    "materials.": "L2_structures.bulk.",
    "symmetry.": "L2_structures.bulk.",
    "defects.": "L2_structures.defect.",
    "surfaces.": "L2_structures.surface.",
    "interfaces.": "L2_structures.interface.",
    "mof.": "L2_structures.framework.",
    "catalysis.": "L3_applications.catalysis.",
    "adsorbates.": "L3_applications.catalysis.",
    "shared.": "L1_foundation.shared.",
    "constraints.": "L1_foundation.shared.",
}


def canonical_category(category: str) -> str:
    # Bare shared prefixes are also used by FilteringKnowledgeStore. Keep
    # that boundary narrower than all L1 foundation data.
    if category == "shared":
        return "L1_foundation.shared"
    if category == "constraints":
        return "L1_foundation.shared.physical_constraints"
    if category == "pymatgen_api":
        return "L1_foundation.shared.pymatgen_api"
    if category == "pymatgen_api.api_reference":
        return "L1_foundation.shared.pymatgen_api"
    for old, new in PREFIX_ALIASES.items():
        if category.startswith(old):
            return new + category[len(old):]
    return category
