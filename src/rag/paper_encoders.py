"""Real pretrained encoders; missing assets never become random embeddings."""
from functools import lru_cache
from pathlib import Path
import hashlib
import json
import numpy as np

ASSETS = Path(__file__).resolve().parents[2] / "data/retrieval_assets"


def verify_assets(directory=ASSETS):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    for name, expected in manifest["files"].items():
        actual = hashlib.sha256((directory / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Retrieval asset checksum mismatch: {name}")
    source = Path(__file__).parent / "_cgcnn_upstream.py"
    if hashlib.sha256(source.read_bytes()).hexdigest() != manifest["cgcnn_source_sha256"]:
        raise ValueError("CGCNN source checksum mismatch")
    return manifest


def cosine(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.clip(np.dot(a, b) / denominator, -1, 1)) if denominator else 0.0


class Mat2VecEncoder:
    def __init__(self, directory=ASSETS):
        data = json.loads((Path(directory) / "mat2vec_elements.json").read_text())
        self.vectors = {k: np.asarray(v, dtype=float) for k, v in data["vectors"].items()}
        self._cache = {}

    def encode(self, composition):
        from pymatgen.core import Composition
        comp = Composition(composition).fractional_composition
        key = tuple(sorted(comp.as_dict().items()))
        if key in self._cache:
            return self._cache[key].copy()
        missing = [element.symbol for element in comp.elements if element.symbol not in self.vectors]
        if missing:
            raise ValueError(f"No pretrained Mat2Vec vectors for: {', '.join(missing)}")
        result = sum(float(amount) * self.vectors[element.symbol] for element, amount in comp.items())
        self._cache[key] = result
        return result.copy()

    def similarity(self, left, right, continuous_weight=0.7):
        from pymatgen.core import Composition
        a, b = Composition(left), Composition(right)
        left_elements, right_elements = set(a.as_dict()), set(b.as_dict())
        jaccard = len(left_elements & right_elements) / len(left_elements | right_elements)
        return continuous_weight * cosine(self.encode(a), self.encode(b)) + (1 - continuous_weight) * jaccard


class CGCNNEncoder:
    """CGCNN final atom states pooled to 64 dimensions, using official trained weights.

    Neighbor preprocessing follows upstream CIFData: radius 8 Å, 12 nearest
    periodic neighbors, 0.2 Å Gaussian spacing, missing-neighbor index 0/distance 9.
    The pooled state is read before the 32-dimensional prediction head.
    """
    def __init__(self, directory=ASSETS):
        import torch
        from ._cgcnn_upstream import CrystalGraphConvNet
        self.torch = torch
        directory = Path(directory)
        data = torch.load(directory / "cgcnn_formation_energy.pt", map_location="cpu", weights_only=True)
        self.atom_init = json.loads((directory / "atom_init.json").read_text())
        with torch.random.fork_rng(devices=[]):
            self.model = CrystalGraphConvNet(92, 41, **data["architecture"])
        # New torch versions add num_batches_tracked automatically for old checkpoints.
        self.model.load_state_dict(data["state_dict"], strict=True)
        self.model.eval()
        self._cache = {}

    def encode(self, structure):
        torch = self.torch
        if not structure.is_ordered or len(structure) == 0:
            raise ValueError("CGCNN requires a nonempty ordered crystal")
        key = hashlib.sha256(json.dumps(structure.as_dict(), sort_keys=True).encode()).hexdigest()
        if key in self._cache:
            return self._cache[key].copy()
        atom_features = np.asarray([self.atom_init[str(site.specie.Z)] for site in structure], dtype=np.float32)
        indices, distances = [], []
        for neighbors in structure.get_all_neighbors(8.0, include_index=True):
            ordered = sorted(neighbors, key=lambda x: (round(float(x.nn_distance), 10), int(x.index)))[:12]
            indices.append([int(n.index) for n in ordered] + [0] * (12 - len(ordered)))
            distances.append([float(n.nn_distance) for n in ordered] + [9.0] * (12 - len(ordered)))
        gaussian = np.exp(-((np.asarray(distances)[..., None] - np.arange(0, 8.1, 0.2)) / 0.2) ** 2)
        with torch.inference_mode():
            state = self.model.embedding(torch.tensor(atom_features, dtype=torch.float32))
            neighbor_features = torch.tensor(gaussian, dtype=torch.float32)
            neighbor_indices = torch.tensor(indices, dtype=torch.long)
            for convolution in self.model.convs:
                state = convolution(state, neighbor_features, neighbor_indices)
            result = state.mean(dim=0).cpu().numpy()
        if result.shape != (64,) or not np.isfinite(result).all():
            raise ValueError("Invalid CGCNN pooled representation")
        self._cache[key] = result
        return result.copy()


@lru_cache(maxsize=4)
def get_encoders(directory=str(ASSETS)):
    manifest = verify_assets(directory)
    return Mat2VecEncoder(directory), CGCNNEncoder(directory), manifest
