import requests
import sys
from pathlib import Path


BASELINE_ROOT = Path(__file__).resolve().parents[2]
if str(BASELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(BASELINE_ROOT))

from baseline_runtime import (
    get_baseline_embedding_model,
    get_baseline_embedding_timeout,
    get_baseline_embedding_url,
)


def get_embedding(text, model=None):
    """
    获取文本的嵌入向量（embedding）
    
    参数:
    text: str - 需要获取嵌入向量的文本内容
    model: str - 使用的嵌入模型名称，默认为 "bge-m3:latest"
    
    返回:
    list - 文本的嵌入向量，通常是一个浮点数列表
    
    异常:
    requests.exceptions.HTTPError - 当 API 请求失败时抛出
    KeyError - 当 API 响应中不包含 "embedding" 字段时抛出
    """
    resolved_model = model or get_baseline_embedding_model("bge-m3:latest")
    url = get_baseline_embedding_url()
    
    # 设置请求头，指定内容类型为 JSON
    headers = {"Content-Type": "application/json"}
    
    if url.rstrip("/").endswith("/api/embeddings"):
        payload = {"model": resolved_model, "prompt": text or ""}
    else:
        payload = {"model": resolved_model, "input": text or ""}
    
    response = requests.post(
        url,
        headers=headers,
        json=payload,
        timeout=get_baseline_embedding_timeout(),
    )
    
    # 如果请求失败（状态码不是 200），则抛出 HTTPError 异常
    response.raise_for_status()
    
    # 解析响应 JSON 并返回嵌入向量
    data = response.json()
    if "embedding" in data:
        return data["embedding"]
    if "data" in data and data["data"]:
        return data["data"][0]["embedding"]
    raise RuntimeError(f"Unexpected embedding response: {data}")


if __name__ == "__main__":
    text = "这是一个测试"
    emb = get_embedding(text)
    if emb:
        print(f"Embedding 长度: {len(emb)}")
        print(emb[:10])
