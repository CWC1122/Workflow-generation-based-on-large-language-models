import numpy as np


class ServiceRecallStage15:
    """
    Stage1.5：基于用户任务做粗粒度服务召回
    """

    def __init__(self, services, embedder):
        """
        :param services: tool_desc.json
        :param embedder: EmbeddingClient
        """

        self.services = services
        self.embedder = embedder

        # ===== 预计算服务 embedding（只做一次！）=====
        self.service_embeddings = []
        self.service_ids = []

        print("🔄 正在构建服务向量库...")

        for s in services:

            text = s.get("target_action_reprs", "")

            if not text:
                continue

            emb = self.embedder.get_embedding(text)

            self.service_embeddings.append(emb)
            self.service_ids.append(s["action_uid"])

        self.service_embeddings = np.array(self.service_embeddings)

        print(f"✅ 服务向量库完成，共 {len(self.service_ids)} 个服务")

    # =============================
    # 计算相似度（cosine）
    # =============================
    def cosine_similarity(self, a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    # =============================
    # 单任务召回
    # =============================
    def recall(self, query, k):

        query_emb = self.embedder.get_embedding(query)

        scores = []

        for i, emb in enumerate(self.service_embeddings):

            sim = self.cosine_similarity(query_emb, emb)

            scores.append((self.service_ids[i], sim))

        # 排序
        scores.sort(key=lambda x: x[1], reverse=True)

        # 取 top-k
        topk = scores[:k]

        return [
            {
                "action_uid": uid,
                "score": float(round(score, 4))
            }
            for uid, score in topk
        ]