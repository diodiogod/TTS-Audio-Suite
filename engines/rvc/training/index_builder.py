"""
FAISS index building helpers for RVC training artifacts.
"""

from __future__ import annotations

import hashlib
import os
from typing import Optional


RVC_V2_FEATURE_DIMS = (768, 1024)


def build_faiss_index(
    dataset_dir: str,
    sample_rate: str,
    model_name: str,
    index_dir: str,
    overwrite: bool = False,
    feature_dim: Optional[int] = None,
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

    requested_dim = int(feature_dim or 768)
    if requested_dim not in RVC_V2_FEATURE_DIMS:
        raise ValueError(f"Unsupported RVC v2 feature dimension: {requested_dim}")

    dataset_key = hashlib.md5(
        f"{dataset_dir}|{sample_rate}|{model_name}|{requested_dim}".encode()
    ).hexdigest()[:10]
    index_path = os.path.join(index_dir, f"{model_name}_v2_{sample_rate}_{dataset_key}.index")
    if os.path.isfile(index_path) and not overwrite:
        return index_path

    feature_dir = os.path.join(dataset_dir, f"3_feature{requested_dim}")
    if not os.path.isdir(feature_dir):
        raise FileNotFoundError(f"Missing RVC feature directory: {feature_dir}")

    features = []
    for filename in sorted(os.listdir(feature_dir)):
        if filename.endswith(".npy"):
            features.append(np.load(os.path.join(feature_dir, filename)))

    if not features:
        raise RuntimeError(f"No feature files found in {feature_dir}")

    for feature in features:
        if feature.ndim != 2 or int(feature.shape[1]) != requested_dim:
            actual_dim = int(feature.shape[1]) if feature.ndim == 2 else None
            raise RuntimeError(
                f"RVC index expected {requested_dim}-dimensional features, but found "
                f"{actual_dim or 'invalid-shape'} in {feature_dir}. Recreate the dataset "
                "with the HuBERT model selected for this RVC model."
            )

    big_npy = np.concatenate(features, axis=0)
    if big_npy.ndim != 2:
        raise RuntimeError(
            f"RVC index features must be a 2D array, but found shape {tuple(big_npy.shape)}. "
            "Recreate the RVC dataset with the selected HuBERT model."
        )
    feature_dim = int(big_npy.shape[1])
    if feature_dim != requested_dim:
        raise RuntimeError(
            f"RVC index expected {requested_dim}-dimensional HuBERT features, but the prepared dataset "
            f"contains {feature_dim}-dimensional features. Recreate the dataset with the HuBERT model "
            "selected for this RVC model."
        )
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
    index = faiss.index_factory(requested_dim, f"IVF{n_ivf},Flat")
    index_ivf = faiss.extract_index_ivf(index)
    index_ivf.nprobe = 1
    index.train(big_npy)

    batch_size_add = 8192
    for start in range(0, big_npy.shape[0], batch_size_add):
        index.add(big_npy[start : start + batch_size_add])

    faiss.write_index(index, index_path)
    return index_path
