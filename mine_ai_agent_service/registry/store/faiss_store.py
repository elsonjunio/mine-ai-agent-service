import json
import os

import faiss
import numpy as np

from mine_ai_agent_service.registry.store.base import BaseVectorStore

_INDEX_FILE = 'registry.faiss'
_IDS_FILE = 'registry_ids.json'


class FaissVectorStore(BaseVectorStore):
    """Store de vetores usando faiss-cpu com busca por similaridade de cosseno.

    Persiste em `data_dir/`:
      registry.faiss    — índice IndexFlatIP com vetores L2-normalizados
      registry_ids.json — lista de IDs na mesma ordem do índice
    """

    def __init__(self, data_dir: str) -> None:
        self._data_dir = data_dir
        self._index: faiss.Index | None = None
        self._ids: list[str] = []

    @property
    def ids(self) -> list[str]:
        return list(self._ids)

    def add(self, ids: list[str], vectors: list[list[float]]) -> None:
        vecs = np.array(vectors, dtype=np.float32)
        faiss.normalize_L2(vecs)

        if self._index is None:
            self._index = faiss.IndexFlatIP(vecs.shape[1])

        self._index.add(vecs)
        self._ids.extend(ids)

    def search(self, query_vector: list[float], top_k: int = 5) -> list[str]:
        if self._index is None or self._index.ntotal == 0:
            return []

        q = np.array([query_vector], dtype=np.float32)
        faiss.normalize_L2(q)

        k = min(top_k, self._index.ntotal)
        _, indices = self._index.search(q, k)

        return [self._ids[i] for i in indices[0] if i >= 0]

    def save(self) -> None:
        os.makedirs(self._data_dir, exist_ok=True)
        faiss.write_index(
            self._index, os.path.join(self._data_dir, _INDEX_FILE)
        )
        with open(os.path.join(self._data_dir, _IDS_FILE), 'w') as f:
            json.dump(self._ids, f)

    def load(self) -> None:
        self._index = faiss.read_index(
            os.path.join(self._data_dir, _INDEX_FILE)
        )
        with open(os.path.join(self._data_dir, _IDS_FILE)) as f:
            self._ids = json.load(f)

    def clear(self) -> None:
        self._index = None
        self._ids = []

    def is_persisted(self) -> bool:
        return os.path.exists(
            os.path.join(self._data_dir, _INDEX_FILE)
        ) and os.path.exists(os.path.join(self._data_dir, _IDS_FILE))
