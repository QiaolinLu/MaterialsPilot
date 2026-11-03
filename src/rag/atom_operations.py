"""Offline, target-independent retrieval of atomic editing knowledge.

No benchmark targets, example IDs, evaluator results, or vector database are
inputs to this retriever. It implements the execution-feedback RAG interface.
"""

import re
from typing import Optional

from src.knowledge.knowledge_store import KnowledgeStore


OPERATION_PATTERNS = {
    "change_atom_action": r"\b(?:change|replace|substitute)\b[^\n]*\batom\b|替换.*原子",
    "remove_atom_action": r"\bremove\b[^\n]*\batom\b|删除.*索引.*原子",
    "add_atom_action": r"\badd\b[^\n]*\batom\b|添加.*原子",
    "move_towards_atom_action": r"\bmove\b[^\n]*\btowards?\b|朝.*原子.*移动",
    "move_atom_action": r"\bmove\s+(?:the\s+)?atom\s+at\s+index\s+\d+\s+by\b|移动.*原子",
    "insert_between_atoms_action": r"\binsert\b[^\n]*\bbetween\b|插入.*原子",
    "swap_atoms_action": r"\bswap\b[^\n]*\b(?:atoms?|positions)\b|交换.*原子",
    "delete_below_atom_action": r"\bdelete\b[^\n]*\b(?:below|lower)\b|删除.*下方",
    "rotate_around_atom_action": r"\brotate\b[^\n]*\b(?:surrounding|around|within)\b|旋转.*原子",
    "super_cell_action": r"\bsuper\s*cell\b|超胞",
}


def infer_operations(prompt: str) -> list[str]:
    """Search the requested action, excluding CIF text and the adapter contract."""
    match = re.search(r"Action prompt:\s*(.*?)(?:\n\s*\n|$)", prompt, re.I | re.S)
    text = match.group(1) if match else prompt
    explicit = [name for name in OPERATION_PATTERNS if re.search(r"\b" + name + r"\b", text)]
    if explicit:
        return explicit
    return [name for name, pattern in OPERATION_PATTERNS.items() if re.search(pattern, text, re.I)]


class AtomOperationRAG:
    def __init__(self, store: Optional[KnowledgeStore] = None):
        self.store = store if store is not None else KnowledgeStore()

    def describe_operation(self, operation: str) -> str:
        record = self.store.query("L2_structures.atom_operations.operations", operation)
        if record is None:
            matches = infer_operations(operation)
            record = self.store.query("L2_structures.atom_operations.operations", matches[0]) if matches else None
        if not record:
            return f"Unknown or disabled atom operation: {operation}"
        return self._format_record(record)

    @staticmethod
    def _format_record(record: dict) -> str:
        parts = [record["summary"], "Parameters: " + record["parameters"]]
        parts.extend(record.get("pitfalls", []))
        parts.append("Python recipe (bind parameters from the action prompt):\n" + record["code"])
        return "\n".join(parts)

    def retrieve(self, prompt: str, k: int = 5, **kwargs) -> list[dict]:
        if k <= 0:
            return []
        results = []
        for operation in infer_operations(prompt):
            record = self.store.query("L2_structures.atom_operations.operations", operation)
            if record:
                results.append({"id": operation, "content": self._format_record(record),
                                "source": "atom_operation_knowledge", "score": 1.0,
                                "metadata": {"sources": record["sources"]}})
        geometry = self.store.get_data("L2_structures.atom_operations.geometry")
        if geometry:
            results.append({"id": "atomic_geometry", "content": "\n".join(geometry["rules"]),
                            "source": "atom_operation_knowledge", "score": 0.5})
        return results[:k]

    @staticmethod
    def format_results_for_llm(results: list[dict], max_length: int = 2500) -> str:
        """Keep recipes intact; never truncate a Python statement mid-line."""
        parts = []
        remaining = max(0, max_length)
        for record in results:
            block = f"[{record['id']}]\n{record['content']}"
            separator = "\n\n" if parts else ""
            if len(separator) + len(block) <= remaining:
                parts.append(block)
                remaining -= len(separator) + len(block)
        return "\n\n".join(parts)
