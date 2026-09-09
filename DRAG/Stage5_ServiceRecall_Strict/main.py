import os
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from utils import read_json, write_json
from embedding_client import EmbeddingClient
from recall_filter import RecallFilter


# INPUT_FILE = "../../Data/mul303/topo/stage3_sequence_topo.json"
# SERVICE_FILE = "../../Data/mul303/tool_desc.json"
# GRAPH_FILE = "../../Data/mul303/graph_desc.json"

# OUTPUT_FILE = "../../Data/mul303/topo/stage5_candidates_topo_no_filter.json"
# UNFINISHED_FILE = "../../Data/mul303/unfinished_stage5.json"

# ==================== 路径配置区域 ====================
# 基础目录，统一管理
# BASE_DATA_DIR = "../../Data_no_self/Gemma12b_bge/mul"
BASE_DATA_DIR = "../../Data_no_self/Llama3.1:70b-bge/ultratool"

# 输入输出文件路径（标准化拼接）
INPUT_FILE = os.path.join(BASE_DATA_DIR, "stage4_chains.json")
SERVICE_FILE = os.path.join(BASE_DATA_DIR, "tool_desc.json")
GRAPH_FILE = os.path.join(BASE_DATA_DIR, "graph_desc.json")
OUTPUT_FILE = os.path.join(BASE_DATA_DIR, "stage5_candidates_strict.json")
UNFINISHED_FILE = os.path.join(BASE_DATA_DIR, "unfinished_stage5.json")
# ==================== 路径配置结束 ====================

# 打印当前基础路径（方便确认目录）
print(f"📂 当前数据基础目录: {os.path.abspath(BASE_DATA_DIR)}")
SAVE_INTERVAL = 50
MAX_WORKERS = min(6, os.cpu_count())


filter_engine = None

processed_tasks = []
processed_ids = set()

save_lock = threading.Lock()
task_counter = 0


def save_progress():

    processed_tasks.sort(key=lambda x: x["annotation_id"])

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(processed_tasks, f, indent=4, ensure_ascii=False)

    print(f"💾 自动保存 {len(processed_tasks)} 条")


def process_single_task(task):

    global task_counter

    result = filter_engine.process_task(task)

    need_save = False

    with save_lock:

        processed_tasks.append(result)
        processed_ids.add(task["annotation_id"])

        task_counter += 1

        if task_counter % 10 == 0:
            print(f"📊 已处理 {task_counter}")

        if len(processed_tasks) % SAVE_INTERVAL == 0:
            need_save = True

    if need_save:
        save_progress()

    return result


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


def main():

    global filter_engine
    global processed_tasks
    global processed_ids
    global task_counter

    tasks = read_json(INPUT_FILE)
    services = read_json(SERVICE_FILE)
    graph = read_json(GRAPH_FILE)

    embedder = EmbeddingClient()

    filter_engine = RecallFilter(embedder, services, graph)

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

    print("\n🎉 Stage5 finished")


if __name__ == "__main__":
    main()