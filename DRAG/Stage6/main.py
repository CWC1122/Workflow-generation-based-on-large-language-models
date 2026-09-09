# =============================
# main.py - Stage6 主入口
# 功能：批量为任务选择具体的服务ID
# =============================
from embedding_client import EmbeddingClient
import os
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from file_utils import read_json_file
from service_selector import ServiceSelector


# ==================== 路径配置区域 ====================
# 基础目录，统一管理
BASE_DATA_DIR = "../../Data_no_self/Llama3.1:70b-bge/hug"
# BASE_DATA_DIR = "../../Data_no_self/Gemma12b_bge/hug"
# 输入输出文件路径（标准化拼接）
INPUT_FILE = os.path.join(BASE_DATA_DIR, "stage5_candidates_weight_sorted.json")
SERVICE_FILE = os.path.join(BASE_DATA_DIR, "tool_desc.json")
GRAPH_FILE = os.path.join(BASE_DATA_DIR, "graph_desc.json")
OUTPUT_FILE = os.path.join(BASE_DATA_DIR, "stage6_selected_weight_sorted.json")
UNFINISHED_FILE = os.path.join(BASE_DATA_DIR, "unfinished_stage6_weight_sorted.json")
# ==================== 路径配置结束 ====================

# 打印当前基础路径（方便确认目录）
print(f"📂 当前数据基础目录: {os.path.abspath(BASE_DATA_DIR)}")


SAVE_INTERVAL = 50
MAX_WORKERS = min(6, os.cpu_count())

selector = None
processed_tasks = []
processed_ids = set()
save_lock = threading.Lock()
task_counter = 0


def save_progress():
    processed_tasks.sort(key=lambda x: x["annotation_id"])
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(processed_tasks, f, indent=4, ensure_ascii=False)
    print(f"💾 saved {len(processed_tasks)}")


def load_graph(data):
    graph = {}
    for g in data:
        u = g["source"]
        v = g["target"]
        graph.setdefault(u, set()).add(v)
    return graph


def process_single(task):
    global task_counter

    result = selector.select_for_task(task)

    need_save = False
    with save_lock:
        processed_tasks.append(result)
        processed_ids.add(task["annotation_id"])
        task_counter += 1

        if task_counter % 10 == 0:
            print(f"processed {task_counter}")

        if len(processed_tasks) % SAVE_INTERVAL == 0:
            need_save = True

    if need_save:
        save_progress()

    return result


def process_all(tasks):
    unfinished = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_single, t): t for t in tasks}

        for future in as_completed(futures):
            task = futures[future]
            try:
                future.result(timeout=600)
            except Exception as e:
                print(f"❌ fail: {e}")
                unfinished.append(task)

    return unfinished


def main():
    global selector, processed_tasks, processed_ids, task_counter

    tasks = read_json_file(INPUT_FILE)
    services = read_json_file(SERVICE_FILE)
    graph_data = read_json_file(GRAPH_FILE)

    service_dict = {s["action_uid"]: s for s in services}
    graph = load_graph(graph_data)

    embedder = EmbeddingClient()

    selector = ServiceSelector(service_dict, graph, embedder)

    # ===== 断点恢复 =====
    if os.path.exists(OUTPUT_FILE):
        processed_tasks = json.load(open(OUTPUT_FILE))
        processed_ids = set(t["annotation_id"] for t in processed_tasks)
        task_counter = len(processed_tasks)
        print(f"resume {task_counter}")
    else:
        processed_tasks = []
        processed_ids = set()

    remaining = [t for t in tasks if t["annotation_id"] not in processed_ids]

    print(f"total={len(tasks)} done={len(processed_tasks)} left={len(remaining)}")

    if not remaining:
        return

    unfinished = process_all(remaining)

    save_progress()

    if unfinished:
        json.dump(unfinished, open(UNFINISHED_FILE, "w"), indent=4)

    print("🎉 Stage6 done")


if __name__ == "__main__":
    main()