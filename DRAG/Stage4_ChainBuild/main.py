# =============================
# main.py
# =============================

import os
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from utils import read_json, write_json
from embedding_client import EmbeddingClient
from chain_builder import SemanticChainBuilder


# =============================
# 配置
# =============================

# INPUT_FILE = "../../Data/mul303/stage2_dag_gema27.json"
# SERVICE_FILE = "../../Data/mul303/tool_desc.json"

# OUTPUT_FILE = "../../Data/mul303/stage4_chains.json"
# UNFINISHED_FILE = "../../Data/mul303/unfinished_stage4.json"

# ==================== 路径配置区域 ====================
# 基础目录，统一管理
# BASE_DATA_DIR = "../../Data_no_self/Gemma27b_bge/mul"
BASE_DATA_DIR = "../../Data_no_self/Llama3.1:8b-bge/mul"

# 输入输出文件路径（标准化拼接）
INPUT_FILE = os.path.join(BASE_DATA_DIR, "stage2_dag_soft_num.json")
SERVICE_FILE = os.path.join(BASE_DATA_DIR, "tool_desc.json")
OUTPUT_FILE = os.path.join(BASE_DATA_DIR, "stage4_chains_weight_sorted.json")
UNFINISHED_FILE = os.path.join(BASE_DATA_DIR, "unfinished_stage4.json")
# ==================== 路径配置结束 ====================

# 打印当前基础路径（方便确认目录）
print(f"📂 当前数据基础目录: {os.path.abspath(BASE_DATA_DIR)}")

SAVE_INTERVAL = 100
MAX_WORKERS = min(6, os.cpu_count())


# =============================
# 全局变量
# =============================

builder = None

processed_tasks = []
processed_ids = set()

save_lock = threading.Lock()
task_counter = 0


def build_structured_chains(chains, nodes):
    """
    把 chain 的 node id 转成结构化格式
    """
    node_map = {n["id"]: n for n in nodes if "id" in n}

    structured = []

    for chain in chains:

        cur_chain = []

        for idx, nid in enumerate(chain):

            node = node_map.get(nid)
            if not node:
                continue

            text = node.get("requirement") or node.get("task") or ""

            # 跳过空节点（你之前提到的坑）
            if not text.strip():
                continue

            cur_chain.append({
                "id": nid,
                "requirement": text,
                "index": idx   # ⭐ 新增 index
            })

        if cur_chain:
            structured.append(cur_chain)

    return structured


# =============================
# 保存
# =============================

def save_progress():

    processed_tasks.sort(key=lambda x: x["annotation_id"])

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(processed_tasks, f, indent=4, ensure_ascii=False)

    print(f"💾 自动保存 {len(processed_tasks)} 条")


# =============================
# 单任务处理
# =============================

def process_single_task(task):

    global task_counter

    nodes = task.get("nodes", [])
    edges = task.get("edges", [])
    service_ids = task.get("st1.5_service", [])

    query = task.get("confirmed_task", "")

    # ===== 权重 =====
    weights = builder.compute_node_weights(
        nodes, edges, query, service_ids
    )

    # ===== 链分解 =====
    chains = builder.build_chains(nodes, edges, weights)

    # ===== 【新增】链排序 =====
    chains = builder.sort_chains(chains, weights) 
    # ===== 转结构化 =====
    chains_struct = build_structured_chains(chains, nodes)
    task["chains"] = chains_struct

    need_save = False

    with save_lock:

        processed_tasks.append(task)
        processed_ids.add(task["annotation_id"])

        task_counter += 1

        if task_counter % 10 == 0:
            print(f"📊 已处理 {task_counter}")

        if len(processed_tasks) % SAVE_INTERVAL == 0:
            need_save = True

    if need_save:
        save_progress()

    return task


# =============================
# 批处理
# =============================

def process_tasks(tasks):

    unfinished = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:

        futures = {
            executor.submit(process_single_task, t): t
            for t in tasks
        }

        for future in as_completed(futures):

            task = futures[future]

            try:
                future.result(timeout=120)

            except Exception as e:

                print(f"❌ fail: {e}")
                print(f"task: {task.get('confirmed_task')}")

                unfinished.append(task)

    return unfinished


# =============================
# 主函数
# =============================

def main():

    global builder
    global processed_tasks
    global processed_ids
    global task_counter

    tasks = read_json(INPUT_FILE)
    services = read_json(SERVICE_FILE)

    embedder = EmbeddingClient()

    service_dict = {
        s["action_uid"]: s for s in services
    }

    builder = SemanticChainBuilder(embedder, service_dict)

    # ===== 断点恢复 =====
    if os.path.exists(OUTPUT_FILE):

        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            processed_tasks = json.load(f)

        processed_ids = set(
            t["annotation_id"] for t in processed_tasks
        )

        task_counter = len(processed_tasks)

        print(f"🔁 已恢复 {task_counter}")

    else:
        processed_tasks = []
        processed_ids = set()

    # ===== 过滤 =====
    remaining = [
        t for t in tasks
        if t["annotation_id"] not in processed_ids
    ]

    print(f"📊 总任务: {len(tasks)}")
    print(f"✅ 已完成: {len(processed_tasks)}")
    print(f"⏳ 剩余: {len(remaining)}")

    if not remaining:
        print("🎉 已完成")
        return

    unfinished = process_tasks(remaining)

    save_progress()

    if unfinished:
        write_json(unfinished, UNFINISHED_FILE)
        print(f"⚠️ 未完成: {len(unfinished)}")

    print("\n🎉 Stage4 finished")

if __name__ == "__main__":
    main()