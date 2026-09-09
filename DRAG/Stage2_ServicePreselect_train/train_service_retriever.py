from collections import Counter, defaultdict

import numpy as np

try:
    import faiss
except ImportError:
    faiss = None


class TrainTaskServiceRetriever:
    """
    先在训练集 task 上做检索，再从相似训练样本的 action_id_list 中聚合候选服务。
    """

    def __init__(self, train_tasks, embeddings, valid_service_ids):
        self.train_tasks = train_tasks
        self.valid_service_ids = set(valid_service_ids)

        if hasattr(embeddings, "detach"):
            embeddings = embeddings.detach().cpu().numpy()

        embeddings = np.asarray(embeddings, dtype=np.float32)
        embeddings = np.ascontiguousarray(embeddings)

        if embeddings.ndim != 2:
            raise ValueError(f"embeddings must be 2D, got shape={embeddings.shape}")

        if len(train_tasks) != embeddings.shape[0]:
            raise ValueError(
                f"train_tasks and embeddings size mismatch: {len(train_tasks)} vs {embeddings.shape[0]}"
            )

        if embeddings.shape[0] == 0:
            raise ValueError("Training embeddings are empty.")

        if np.isnan(embeddings).any() or np.isinf(embeddings).any():
            raise ValueError("Training embeddings contain NaN or Inf.")

        self.embeddings = embeddings
        self.ntotal = embeddings.shape[0]
        self.global_service_counts = Counter()
        self.default_services = []

        for task in train_tasks:
            unique_services = []
            seen = set()
            for service_id in task.get("action_id_list", []):
                if service_id in self.valid_service_ids and service_id not in seen:
                    unique_services.append(service_id)
                    seen.add(service_id)

            self.global_service_counts.update(unique_services)

        self.default_services = [
            service_id
            for service_id, _ in self.global_service_counts.most_common()
        ]

        if faiss is not None:
            dim = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dim)
            self.index.add(embeddings)
        else:
            self.index = None

        self.invalid_query_count = 0
        self.empty_result_count = 0

    def _normalize_query_embedding(self, query_embedding):
        if query_embedding is None:
            return None

        if hasattr(query_embedding, "detach"):
            query_embedding = query_embedding.detach().cpu().numpy()

        query_embedding = np.asarray(query_embedding, dtype=np.float32)

        if query_embedding.ndim == 1:
            query_embedding = np.expand_dims(query_embedding, axis=0)

        query_embedding = np.ascontiguousarray(query_embedding)

        if query_embedding.size == 0:
            return None

        if np.isnan(query_embedding).any() or np.isinf(query_embedding).any():
            return None

        return query_embedding

    def _default_candidates(self, k):
        self.empty_result_count += 1
        return [
            {
                "action_uid": service_id,
                "score": 0.0,
                "source": "train_default",
            }
            for service_id in self.default_services[:k]
        ]

    def recall(self, query_embedding, k, top_task_k=None):
        query_embedding = self._normalize_query_embedding(query_embedding)
        if query_embedding is None:
            self.invalid_query_count += 1
            return self._default_candidates(k)

        search_k = top_task_k if top_task_k is not None else max(8, min(32, k))
        search_k = max(1, min(search_k, self.ntotal))

        sims, ids = self._search(query_embedding, search_k)
        sims = sims[0]
        ids = ids[0]

        valid_mask = ids != -1
        sims = sims[valid_mask]
        ids = ids[valid_mask]

        if len(ids) == 0:
            return self._default_candidates(k)

        score_map = defaultdict(float)
        freq_map = Counter()

        for rank, (sim, task_id) in enumerate(zip(sims, ids)):
            task = self.train_tasks[int(task_id)]
            raw_services = task.get("action_id_list", [])

            unique_services = []
            seen = set()
            for service_id in raw_services:
                if service_id in self.valid_service_ids and service_id not in seen:
                    unique_services.append(service_id)
                    seen.add(service_id)

            if not unique_services:
                continue

            base_score = float(sim)
            if not np.isfinite(base_score):
                continue

            if base_score <= 0:
                base_score = 1e-6 * max(1, search_k - rank)

            for position, service_id in enumerate(unique_services):
                position_weight = 1.0 / (position + 1)
                score_map[service_id] += base_score * position_weight
                freq_map[service_id] += 1

        if not score_map:
            return self._default_candidates(k)

        ranked = sorted(
            score_map.items(),
            key=lambda item: (
                -item[1],
                -freq_map[item[0]],
                -self.global_service_counts[item[0]],
                item[0],
            ),
        )

        candidate_ids = [service_id for service_id, _ in ranked]

        if len(candidate_ids) < k:
            seen = set(candidate_ids)
            for service_id in self.default_services:
                if service_id not in seen:
                    candidate_ids.append(service_id)
                    seen.add(service_id)
                if len(candidate_ids) >= k:
                    break

        return [
            {
                "action_uid": service_id,
                "score": float(round(score_map.get(service_id, 0.0), 4)),
                "source": "train_retrieval" if service_id in score_map else "train_default",
            }
            for service_id in candidate_ids[:k]
        ]

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
        print("\n========== Stage2 Train Retrieval Statistics ==========")
        print(f"Invalid query fallback : {self.invalid_query_count}")
        print(f"Empty result fallback  : {self.empty_result_count}")
        print(f"Default pool size      : {len(self.default_services)}")
        print("======================================================\n")
