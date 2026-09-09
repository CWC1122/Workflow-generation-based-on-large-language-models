import os
import requests

# ==================== Embedding 配置区域 ====================
EMBEDDING_URL = os.getenv("DRAG_EMBEDDING_URL") or "http://127.0.0.1:11434/v1/embeddings"
EMBEDDING_MODEL = os.getenv("DRAG_STAGE1_EMBEDDING_MODEL") or os.getenv("DRAG_EMBEDDING_MODEL") or "bge-m3:latest"
EMBEDDING_TIMEOUT = int(os.getenv("DRAG_EMBEDDING_TIMEOUT") or 120)
# ==================== Embedding 配置结束 ====================


class EmbeddingClient:
    def __init__(self, url=None, model=None, timeout=None):
        self.url = url or EMBEDDING_URL
        self.model = model or EMBEDDING_MODEL
        self.timeout = float(timeout if timeout is not None else EMBEDDING_TIMEOUT)

    def get_embedding(self, text):
        payload = {
            "model": self.model,
            "input": text or "",
        }

        response = requests.post(self.url, json=payload, timeout=self.timeout)
        response.raise_for_status()

        data = response.json()
        if "data" not in data or not data["data"]:
            raise RuntimeError(f"Unexpected embedding response: {data}")
        return data["data"][0]["embedding"]
