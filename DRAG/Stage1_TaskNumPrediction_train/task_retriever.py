import numpy as np

try:
    import faiss
except ImportError:
    faiss = None


class TaskRetriever:
    def __init__(self, tasks, embeddings, have_self=False):
        self.tasks = tasks
        self.have_self = have_self
        self.lengths = [len(task["action_id_list"]) for task in tasks]

        if not self.lengths:
            raise ValueError("TaskRetriever received an empty training task list.")

        if hasattr(embeddings, "detach"):
            embeddings = embeddings.detach().cpu().numpy()

        embeddings = np.asarray(embeddings, dtype=np.float32)
        embeddings = np.ascontiguousarray(embeddings)

        if embeddings.ndim != 2:
            raise ValueError(f"embeddings must be 2D, got shape={embeddings.shape}")

        if embeddings.shape[0] == 0:
            raise ValueError("embeddings is empty, cannot build faiss index.")

        if len(tasks) != embeddings.shape[0]:
            raise ValueError(
                f"tasks and embeddings size mismatch: {len(tasks)} vs {embeddings.shape[0]}"
            )

        if np.isnan(embeddings).any() or np.isinf(embeddings).any():
            raise ValueError("embeddings contains NaN or Inf, cannot build faiss index.")

        self.embeddings = embeddings
        self.ntotal = embeddings.shape[0]
        self.default_length = max(1, int(round(float(np.median(self.lengths)))))
        self.min_length = max(1, int(min(self.lengths)))
        self.max_length = max(1, int(max(self.lengths)))

        if faiss is not None:
            dim = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dim)
            self.index.add(embeddings)
        else:
            self.index = None

        self.nan_query_count = 0
        self.nan_sim_count = 0
        self.default_fallback_count = 0

    def _fallback_prediction(self, recom_num=None):
        self.default_fallback_count += 1
        if recom_num is not None:
            try:
                recom_num = int(recom_num)
                return max(self.min_length, min(self.max_length, recom_num))
            except (TypeError, ValueError):
                pass
        return self.default_length

    def predict_task_num(self, query_embedding, recom_num=None, top_k=4):
        if query_embedding is None:
            self.nan_query_count += 1
            return self._fallback_prediction(recom_num)

        if hasattr(query_embedding, "detach"):
            query_embedding = query_embedding.detach().cpu().numpy()

        query_embedding = np.asarray(query_embedding, dtype=np.float32)

        if query_embedding.ndim == 1:
            query_embedding = np.expand_dims(query_embedding, axis=0)

        query_embedding = np.ascontiguousarray(query_embedding)

        if query_embedding.size == 0:
            self.nan_query_count += 1
            return self._fallback_prediction(recom_num)

        if np.isnan(query_embedding).any() or np.isinf(query_embedding).any():
            self.nan_query_count += 1
            return self._fallback_prediction(recom_num)

        search_k = top_k if self.have_self else top_k + 1
        search_k = min(search_k, self.ntotal)

        if search_k <= 0:
            return self._fallback_prediction(recom_num)

        sims, ids = self._search(query_embedding, search_k)
        sims = sims[0]
        ids = ids[0]

        if not self.have_self and len(ids) > 0:
            sims = sims[1:]
            ids = ids[1:]

        valid_mask = ids != -1
        sims = sims[valid_mask]
        ids = ids[valid_mask]

        sims = sims[:top_k]
        ids = ids[:top_k]

        if len(ids) == 0:
            return self._fallback_prediction(recom_num)

        if np.isnan(sims).any() or np.isinf(sims).any():
            self.nan_sim_count += 1
            return self._fallback_prediction(recom_num)

        lengths = np.array([self.lengths[i] for i in ids], dtype=np.float32)
        sim_sum = float(np.sum(sims))

        if (not np.isfinite(sim_sum)) or sim_sum <= 0:
            avg_len = np.mean(lengths)
        else:
            weights = sims / sim_sum
            avg_len = np.sum(weights * lengths)

        prediction = int(round(float(avg_len)))
        return max(self.min_length, min(self.max_length, prediction))

    def _search(self, query_embedding, search_k):
        if self.index is not None:
            return self.index.search(query_embedding, search_k)

        query_vec = query_embedding[0]
        sims = self.embeddings @ query_vec
        top_indices = np.argsort(-sims)[:search_k]
        top_sims = sims[top_indices].astype(np.float32)
        top_ids = top_indices.astype(np.int64)
        return top_sims.reshape(1, -1), top_ids.reshape(1, -1)

    def print_stats(self):
        print("\n========== Retrieval Statistics ==========")
        print(f"Allow exact-match hit  : {self.have_self}")
        print(f"Default length prior   : {self.default_length}")
        print(f"Invalid query fallback : {self.nan_query_count}")
        print(f"Invalid sim fallback   : {self.nan_sim_count}")
        print(f"Total fallback count   : {self.default_fallback_count}")
        print("=========================================\n")
