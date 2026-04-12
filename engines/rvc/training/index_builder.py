"""
FAISS index building helpers for RVC training artifacts.
"""

from __future__ import annotations

import hashlib
import os
from typing import Optional


def build_faiss_index(
    dataset_dir: str,
    sample_rate: str,
    model_name: str,
    index_dir: str,
    overwrite: bool = False,
) -> Optional[str]:
    try:
        import faiss
        import numpy as np
        from sklearn.cluster import MiniBatchKMeans
    except ImportError as exc:
        raise RuntimeError(
            "RVC index building requires a Python faiss package and scikit-learn. "
            "Install 'faiss-cpu' (or another compatible faiss build), not 'faiss'. "
            f"Underlying import error: {exc}"
        ) from exc

    missing_faiss_api = [name for name in ("index_factory", "extract_index_ivf", "write_index") if not hasattr(faiss, name)]
    if missing_faiss_api:
        module_file = getattr(faiss, "__file__", None)
        loader_name = type(getattr(faiss, "__loader__", None)).__name__ if getattr(faiss, "__loader__", None) else None
        raise RuntimeError(
            "RVC index building found a broken Python faiss package. "
            f"Missing required API: {', '.join(missing_faiss_api)}. "
            f"Imported module file: {module_file!r}, loader: {loader_name!r}. "
            "This usually means a stale namespace/stub package. Uninstall faiss/faiss-cpu/faiss-gpu, remove any stray site-packages/faiss directory, and reinstall 'faiss-cpu'."
        )

    os.makedirs(index_dir, exist_ok=True)

    dataset_key = hashlib.md5(f"{dataset_dir}|{sample_rate}|{model_name}".encode()).hexdigest()[:10]
    index_path = os.path.join(index_dir, f"{model_name}_v2_{sample_rate}_{dataset_key}.index")
    if os.path.isfile(index_path) and not overwrite:
        return index_path

    feature_dir = os.path.join(dataset_dir, "3_feature768")
    if not os.path.isdir(feature_dir):
        raise FileNotFoundError(f"Missing RVC feature directory: {feature_dir}")

    features = []
    for filename in sorted(os.listdir(feature_dir)):
        if filename.endswith(".npy"):
            features.append(np.load(os.path.join(feature_dir, filename)))

    if not features:
        raise RuntimeError(f"No feature files found in {feature_dir}")

    big_npy = np.concatenate(features, axis=0)
    shuffled_indices = np.arange(big_npy.shape[0])
    np.random.shuffle(shuffled_indices)
    big_npy = big_npy[shuffled_indices]

    if big_npy.shape[0] > 200000:
        big_npy = MiniBatchKMeans(
            n_clusters=10000,
            verbose=True,
            batch_size=256,
            compute_labels=False,
            init="random",
        ).fit(big_npy).cluster_centers_

    n_ivf = min(int(16 * (big_npy.shape[0] ** 0.5)), max(1, big_npy.shape[0] // 39))
    index = faiss.index_factory(768, f"IVF{n_ivf},Flat")
    index_ivf = faiss.extract_index_ivf(index)
    index_ivf.nprobe = 1
    index.train(big_npy)

    batch_size_add = 8192
    for start in range(0, big_npy.shape[0], batch_size_add):
        index.add(big_npy[start : start + batch_size_add])

    faiss.write_index(index, index_path)
    return index_path
