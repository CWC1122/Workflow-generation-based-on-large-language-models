import os

import numpy as np
from tqdm import tqdm

from embedding_client import EmbeddingClient


def _build_valid_indices_path(cache_path):
    root, _ = os.path.splitext(cache_path)
    return f"{root}_valid_indices.npy"


def build_or_load_embeddings(tasks, cache_path):
    valid_indices_path = _build_valid_indices_path(cache_path)

    if os.path.exists(cache_path):
        print("Loading embedding cache...")
        embeddings = np.load(cache_path)
        embeddings = np.asarray(embeddings, dtype=np.float32)
        embeddings = np.ascontiguousarray(embeddings)

        if embeddings.ndim != 2:
            raise ValueError(f"cache embeddings must be 2D, got shape={embeddings.shape}")

        if os.path.exists(valid_indices_path):
            valid_indices = np.load(valid_indices_path).tolist()
            valid_tasks = [tasks[i] for i in valid_indices]
        else:
            if len(tasks) != embeddings.shape[0]:
                raise ValueError(
                    "Found old embedding cache without valid indices, and task count "
                    f"does not match cache size: tasks={len(tasks)}, embeddings={embeddings.shape[0]}"
                )
            valid_tasks = tasks

        return embeddings, valid_tasks

    print("Building embeddings for confirmed_task...")

    embedder = EmbeddingClient()

    valid_embeddings = []
    valid_tasks = []
    valid_indices = []

    total = len(tasks)
    nan_count = 0
    error_count = 0

    for idx, task in enumerate(tqdm(tasks, desc="Embedding tasks", total=total)):
        text = task["confirmed_task"]

        try:
            emb = embedder.get_embedding(text)

            if emb is None:
                nan_count += 1
                continue

            if hasattr(emb, "detach"):
                emb = emb.detach().cpu().numpy()

            emb = np.asarray(emb, dtype=np.float32)

            if emb.ndim != 1:
                emb = emb.reshape(-1)

            if emb.size == 0:
                nan_count += 1
                continue

            if np.isnan(emb).any() or np.isinf(emb).any():
                nan_count += 1
                continue

            valid_embeddings.append(emb)
            valid_tasks.append(task)
            valid_indices.append(idx)

        except Exception as exc:
            error_count += 1
            print(f"[Embedding Error] {exc}")

    if not valid_embeddings:
        raise ValueError("No valid embeddings were generated from training tasks.")

    try:
        valid_embeddings = np.stack(valid_embeddings, axis=0).astype(np.float32)
    except Exception as exc:
        raise ValueError(f"Embedding dimensions are inconsistent: {exc}")

    valid_embeddings = np.ascontiguousarray(valid_embeddings)

    np.save(cache_path, valid_embeddings)
    np.save(valid_indices_path, np.asarray(valid_indices, dtype=np.int64))

    print("\n========== Embedding Statistics ==========")
    print(f"Total tasks         : {total}")
    print(f"Valid embeddings    : {len(valid_embeddings)}")
    print(f"Valid task indices  : {len(valid_indices)}")
    print(f"Filtered (NaN/Inf)  : {nan_count}")
    print(f"Error count         : {error_count}")
    print(f"NaN ratio           : {nan_count / total:.4f}")
    print("=========================================\n")

    return valid_embeddings, valid_tasks
