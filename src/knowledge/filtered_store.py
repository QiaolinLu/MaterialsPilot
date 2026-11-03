"""A knowledge view whose fallbacks obey the same ablation boundary as queries."""

from .categories import canonical_category
from .knowledge_store import KnowledgeStore


class FilteringKnowledgeStore(KnowledgeStore):
    """Bind inherited lookup methods to the filtered view, including fallbacks.

    A proxy using __getattr__ returns methods bound to the unfiltered store;
    query_with_fallback and related-category searches can then leak disabled data.
    """

    def __init__(self, base: KnowledgeStore, disabled_prefixes):
        self.base = base
        self.data_dir = base.data_dir
        self.additional_dirs = base.additional_dirs
        self.disabled_prefixes = tuple(canonical_category(p) for p in disabled_prefixes)

    def _enabled(self, category: str) -> bool:
        return not canonical_category(category).startswith(self.disabled_prefixes)

    def get_data(self, category: str) -> dict:
        return self.base.get_data(category) if self._enabled(category) else {}

    @property
    def categories(self):
        return [c for c in self.base.categories if self._enabled(c)]

    def reload(self):
        self.base.reload()
