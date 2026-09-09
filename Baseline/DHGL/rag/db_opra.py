from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility, Index
import requests
import json
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

def limit_length(text, max_len=30000):
    if text is None:
        return ""
    text = str(text)
    if len(text) > max_len:
        return text[:max_len]
    return text



class VectorDB:
    def __init__(self, host="localhost", port="19530", collection_name="SSC1"):
        self.collection_name = collection_name
        print(collection_name)
        test_embedding = self.get_embedding("测试文本")
        print(len(test_embedding))
        embedding_dim = len(test_embedding)
        self.embedding_dim = embedding_dim

        # 连接 Milvus
        connections.connect("default", host=host, port=port)

        # 检查 Collection 是否存在
        existing_collections = utility.list_collections()
        if collection_name in existing_collections:
            self.collection = Collection(name=collection_name)
        else:
            # 定义 Collection schema
            fields = [
                FieldSchema(
                    name="id",
                    dtype=DataType.INT64,
                    is_primary=True,
                    auto_id=False,
                    description="Unique service identifier"
                ),

                FieldSchema(
                    name="service_name",
                    dtype=DataType.VARCHAR,
                    max_length=256,
                    description="Service name"
                ),

                FieldSchema(
                    name="author",
                    dtype=DataType.VARCHAR,
                    max_length=256,
                    description="Author or organization"
                ),

                FieldSchema(
                    name="function",
                    dtype=DataType.VARCHAR,
                    max_length=128,
                    description="Service function type"
                ),

                FieldSchema(
                    name="descriptions",
                    dtype=DataType.VARCHAR,
                    max_length=30005,
                    description="Full service description"
                ),

                FieldSchema(
                    name="url",
                    dtype=DataType.VARCHAR,
                    max_length=512,
                    description="Service endpoint url"
                ),

                FieldSchema(
                    name="last_modified",
                    dtype=DataType.VARCHAR,
                    max_length=64,
                    description="Last modified time"
                ),

                FieldSchema(
                    name="input_parameter",
                    dtype=DataType.VARCHAR,
                    max_length=1024,
                    description="Input parameter list (JSON)"
                ),

                FieldSchema(
                    name="output_parameter",
                    dtype=DataType.VARCHAR,
                    max_length=1024,
                    description="Output parameter list (JSON)"
                ),

                FieldSchema(
                    name="downloads",
                    dtype=DataType.INT64,
                    description="Download count"
                ),

                FieldSchema(
                    name="likes",
                    dtype=DataType.INT64,
                    description="Like count"
                ),

                FieldSchema(
                    name="response_time",
                    dtype=DataType.FLOAT,
                    description="Response time"
                ),

                FieldSchema(
                    name="waiting_time",
                    dtype=DataType.VARCHAR,
                    max_length=512   # 可以大一点，例如 512 / 1024
                ),

                FieldSchema(
                    name="reliability",
                    dtype=DataType.FLOAT,
                    description="Reliability"
                ),

                FieldSchema(
                    name="successability",
                    dtype=DataType.FLOAT,
                    description="Success rate"
                ),

                FieldSchema(
                    name="embedding",
                    dtype=DataType.FLOAT_VECTOR,
                    dim=embedding_dim,
                    description="Vector embedding"
                )
            ]
            schema = CollectionSchema(fields, description="API 服务向量化存储")
            self.collection = Collection(name=collection_name, schema=schema)

    def get_embedding(self, text):
        """调用配置好的 embedding 服务生成向量。"""
        url = get_baseline_embedding_url()
        model = get_baseline_embedding_model("bge-m3:latest")
        if url.rstrip("/").endswith("/api/embeddings"):
            payload = {"model": model, "prompt": text or ""}
        else:
            payload = {"model": model, "input": text or ""}

        response = requests.post(
            url,
            json=payload,
            timeout=get_baseline_embedding_timeout(),
        )
        response.raise_for_status()

        data = response.json()
        if "embedding" in data:
            return data["embedding"]
        if "data" in data and data["data"]:
            return data["data"][0]["embedding"]
        raise RuntimeError(f"Unexpected embedding response: {data}")

    

    def insert(self, data_json, batch_size=500):
        """
        分批插入新格式的服务数据，description 字段进行向量化
        """

        total = len(data_json)
        print(f"📌 共 {total} 条数据，准备分批插入（每批 {batch_size} 条）...")

        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            current_batch = data_json[start:end]

            print(f"\n🚀 当前批次：第 {start + 1} 条 到 第 {end} 条")

            # ============ 字段准备 ============
            ids = list(range(start + 1, end + 1))

            service_name = [item.get("service_name", "") for item in current_batch]
            author = [item.get("author", "") for item in current_batch]
            function = [item.get("function", "") for item in current_batch]

            # 你原来的截断逻辑保留
            descriptions = [limit_length(item.get("description", "")) for item in current_batch]

            url = [item.get("url", "") for item in current_batch]
            last_modified = [item.get("last_modified", "") for item in current_batch]

            # 数组转字符串（Milvus 不支持 list）
            input_parameter = [
                json.dumps(item.get("input_parameter", []), ensure_ascii=False)
                for item in current_batch
            ]

            output_parameter = [
                json.dumps(item.get("output_parameter", []), ensure_ascii=False)
                for item in current_batch
            ]

            downloads = [int(item.get("downloads", 0)) for item in current_batch]
            likes = [int(item.get("likes", 0)) for item in current_batch]
            response_time = [float(item.get("response_time", 0)) for item in current_batch]

            # 按你要求：存为字符串
            waiting_time = [str(item.get("waiting_time", "")) for item in current_batch]

            reliability = [float(item.get("reliability", 0)) for item in current_batch]
            successability = [float(item.get("successability", 0)) for item in current_batch]

            # ============ Description 向量化 ============
            print("🧠 正在向量化 description...")
            embeddings = []

            for i, desc in enumerate(descriptions):
                real_index = start + i + 1
                try:
                    print(f"   ▶ 正在向量化第 {real_index}/{total} 条：{service_name[i]}  function:{function[i]}")
                    embedding = self.get_embedding(desc)
                    embeddings.append(embedding)
                except Exception as e:
                    print(f"❌ 第 {real_index} 条数据向量化失败")
                    print(f"错误原因：{e}")
                    print(f"问题服务：{service_name[i]}")
                    embeddings.append([0.0] * self.embedding_dim)  # 占位向量

            if len(embeddings) == 0:
                print(f"⚠️ 第 {start + 1} 到 {end} 条未生成有效向量，已跳过")
                continue

            # ============ 按 schema 顺序插入 ============
            entities = [
                ids,
                service_name,
                author,
                function,
                descriptions,
                url,
                last_modified,
                input_parameter,
                output_parameter,
                downloads,
                likes,
                response_time,
                waiting_time,
                reliability,
                successability,
                embeddings
            ]

            print("✅ 数据格式检查：")
            print(f"当前批次条数: {len(ids)}")
            print(f"向量维度: {len(embeddings[0])}")

            # ============ 插入 ============
            try:
                self.collection.insert(entities)
                self.collection.flush()
                print(f"✅ 已成功插入第 {start + 1} 条 到 第 {end} 条")
            except Exception as e:
                print(f"❌ 插入第 {start + 1} 到 {end} 条失败")
                print(f"错误原因：{e}")

        print("\n📌 正在检查现有索引...")
        existing_index_fields = [idx.field_name for idx in self.collection.indexes]
        # ========== 1️⃣ embedding 向量索引 ==========
        if "embedding" not in existing_index_fields:
            print("🚀 正在创建向量索引 (embedding)...")
            index_params = {
                "index_type": "IVF_FLAT",
                "metric_type": "IP",
                "params": {"nlist": 128}
            }

            self.collection.create_index(
                field_name="embedding",
                index_params=index_params
            )
            print("✅ embedding 向量索引创建完成")
        else:
            print("✅ embedding 向量索引已存在")

        # ========== 2️⃣ function 标量索引 ==========
        if "function" not in existing_index_fields:
            print("🚀 正在创建 function 标量索引...")

            scalar_index_params = {
                "index_type": "INVERTED",
                "params": {}
            }

            self.collection.create_index(
                field_name="function",
                index_params=scalar_index_params
            )
            print("✅ function 标量索引创建完成")
        else:
            print("✅ function 标量索引已存在")

        print("🎉 所有数据插入完成 + 向量索引创建完成！")


    def search(self, function_type, query_text, limit=10):
        """
        先根据 function 过滤，再根据 description 进行向量匹配
        """

        # 1. 加载 collection
        
        self.collection.load()

        # 2. 向量化 query
        query_embedding = self.get_embedding(query_text)

        # 3. 构造过滤表达式
        expr = f'function == "{function_type}"'

        # 4. 搜索参数
        search_params = {
            "metric_type": "IP",   # 要与你建索引时一致
            "params": {"nprobe": 10}
        }

        # 5. 搜索
        results = self.collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=limit,
            expr=expr,   # ✅ 核心：先按 function 过滤
            output_fields=[
                "id",
                "service_name",
                "author",
                "function",
                "descriptions",
                "url",
                "input_parameter",
                "output_parameter",
                "downloads",
                "likes",
                "response_time",
                "waiting_time",
                "reliability",
                "successability"
            ]
        )

        output = []

        for hits in results:
            for hit in hits:

                entity = hit.entity

                output.append({
                    "id": entity.get("id"),
                    "service_name": entity.get("service_name"),
                    "author": entity.get("author"),
                    "function": entity.get("function"),
                    "descriptions": entity.get("descriptions"),
                    "url": entity.get("url"),
                    "downloads": entity.get("downloads"),
                    "likes": entity.get("likes"),
                    "response_time": entity.get("response_time"),
                    "waiting_time": entity.get("waiting_time"),
                    "reliability": entity.get("reliability"),
                    "successability": entity.get("successability"),

                    # 相似度得分（非常重要）
                    "score": hit.score
                })

        return output
    def create_indexes(self):
        """为 embedding 和 function 创建索引（幂等，不重复创建）"""

        # ✅ 新增：创建索引前先 release
        try:
            print("📤 Releasing collection before creating index...")
            self.collection.release()
        except:
            pass

        print("\n📌 正在检查现有索引...")

        existing_index_fields = [idx.field_name for idx in self.collection.indexes]
        print("当前已有索引字段:", existing_index_fields)

        # ========== 1️⃣ embedding 向量索引 ==========
        if "embedding" not in existing_index_fields:
            print("🚀 正在创建向量索引 (embedding)...")
            index_params = {
                "index_type": "IVF_FLAT",
                "metric_type": "IP",
                "params": {"nlist": 128}
            }

            self.collection.create_index(
                field_name="embedding",
                index_params=index_params
            )
            print("✅ embedding 向量索引创建完成")
        else:
            print("✅ embedding 向量索引已存在")

        # ========== 2️⃣ function 标量索引 ==========
        if "function" not in existing_index_fields:
            print("🚀 正在创建 function 标量索引...")

            scalar_index_params = {
                "index_type": "INVERTED",
                "params": {}
            }

            self.collection.create_index(
                field_name="function",
                index_params=scalar_index_params
            )
            print("✅ function 标量索引创建完成")
        else:
            print("✅ function 标量索引已存在")

        # ✅ 建完以后再 load
        print("📥 Loading collection ...")
        self.collection.load()
