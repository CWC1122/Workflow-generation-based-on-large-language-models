import os
import requests


# ==================== Embedding 配置区域 ====================
EMBEDDING_URL = os.getenv("DRAG_EMBEDDING_URL") or "http://127.0.0.1:11434/v1/embeddings"
EMBEDDING_MODEL = os.getenv("DRAG_STAGE4_EMBEDDING_MODEL") or os.getenv("DRAG_EMBEDDING_MODEL") or "mxbai-embed-large"
EMBEDDING_TIMEOUT = int(os.getenv("DRAG_EMBEDDING_TIMEOUT") or 120)
# ==================== Embedding 配置结束 ====================


class EmbeddingClient:

    def __init__(self, url=None, model=None, timeout=None):
        # 👇 强制 CPU
        # os.environ["OLLAMA_NUM_GPU"] = "0"

        self.url = url or EMBEDDING_URL
        self.model = model or EMBEDDING_MODEL
        self.timeout = float(timeout if timeout is not None else EMBEDDING_TIMEOUT)
        # self.model = "bge-m3:latest"
    def get_embedding(self, text):

        payload = {
            "model": self.model,
            "input": text
        }

        response = requests.post(self.url, json=payload, timeout=self.timeout)
        response.raise_for_status()
        response = response.json()
        # print(text)
        # print(response)
        # ✅ 加错误处理（必须）
        if 'error' in response:
            raise RuntimeError(response['error']['message'])

        return response['data'][0]['embedding']
