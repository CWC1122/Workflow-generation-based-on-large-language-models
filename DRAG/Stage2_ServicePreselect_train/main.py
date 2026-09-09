import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from embedding_cache import build_or_load_embeddings
from embedding_client import EmbeddingClient
from file_utils import read_json_file
from train_service_retriever import TrainTaskServiceRetriever

# ==================== 路径配置区域 ====================
# 训练集目录：用于检索相似历史任务并聚合候选服务
TRAIN_DATA_DIR = "../../new_Data/train/mul"

# 测试集目录：读取 stage1 输出并生成 stage2 候选服务
TEST_DATA_DIR = "../../new_Data/test/mul"

TRAIN_TASKS_FILE = os.path.join(TRAIN_DATA_DIR, "data.json")
TRAIN_EMBEDDINGS_PATH = os.path.join(TRAIN_DATA_DIR, "stage2_train_task_embeddings.npy")
INPUT_FILE = os.path.join(TEST_DATA_DIR, "stage1_tasknum.json")
SERVICE_FILE = os.path.join(TEST_DATA_DIR, "tool_desc.json")
OUTPUT_FILE = os.path.join(TEST_DATA_DIR, "stage15_services.json")
UNFINISHED_FILE = os.path.join(TEST_DATA_DIR, "unfinished_stage15.json")
# ==================== 路径配置结束 ====================

SAVE_INTERVAL = 50
MAX_WORKERS = min(6, os.cpu_count() or 4)

recall_engine = None
embedder = None

processed_tasks = []
processed_ids = set()

save_lock = threading.Lock()
task_counter = 0


def compute_k(task_num):
    return min(max(12, task_num * 4), 24)


def save_progress():
    processed_tasks.sort(key=lambda item: item["annotation_id"])

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file_obj:
        json.dump(processed_tasks, file_obj, indent=4, ensure_ascii=False)

    print(f"Auto-saved {len(processed_tasks)} tasks")


def process_single_task(task):
    global task_counter

    try:
        query = task.get("confirmed_task", "")
        task_num = task.get("pred_task_num", 3)
        candidate_size = compute_k(task_num)

        try:
            query_embedding = embedder.get_embedding(query)
        except Exception as exc:
            print(f"[Query Embedding Error] {exc}")
            query_embedding = None

        top_services = recall_engine.recall(query_embedding, candidate_size)
        task["st1.5_service"] = [service["action_uid"] for service in top_services]

        need_save = False

        with save_lock:
            processed_tasks.append(task)
            processed_ids.add(task["annotation_id"])
            task_counter += 1

            if task_counter % 10 == 0:
                print(f"Processed {task_counter}")

            if len(processed_tasks) % SAVE_INTERVAL == 0:
                need_save = True

        if need_save:
            save_progress()

        return task

    except Exception as exc:
        raise RuntimeError(f"Stage2 error: {exc}")


def process_tasks(tasks):
    unfinished = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_single_task, task): task for task in tasks}

        for future in as_completed(futures):
            task = futures[future]
            try:
                future.result(timeout=300)
            except Exception as exc:
                print(f"fail: {exc}")
                print(f"task: {task.get('confirmed_task')}")
                unfinished.append(task)

    return unfinished


def main():
    global recall_engine
    global embedder
    global processed_tasks
    global processed_ids
    global task_counter

    print("=== ENTER STAGE2 ===")
    print(f"Train split  : {TRAIN_DATA_DIR}")
    print(f"Test split   : {TEST_DATA_DIR}")
    print(f"Input file   : {INPUT_FILE}")
    print(f"Output file  : {OUTPUT_FILE}")

    tasks = read_json_file(INPUT_FILE)
    train_tasks = read_json_file(TRAIN_TASKS_FILE)
    services = read_json_file(SERVICE_FILE)

    print(f"Loaded test tasks  : {len(tasks)}")
    print(f"Loaded train tasks : {len(train_tasks)}")
    print(f"Loaded services    : {len(services)}")

    embedder = EmbeddingClient()

    embeddings, valid_train_tasks = build_or_load_embeddings(
        train_tasks,
        TRAIN_EMBEDDINGS_PATH,
    )

    valid_service_ids = [service["action_uid"] for service in services]
    recall_engine = TrainTaskServiceRetriever(
        valid_train_tasks,
        embeddings,
        valid_service_ids,
    )

    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r", encoding="utf-8") as file_obj:
            processed_tasks = json.load(file_obj)

        processed_ids = {task["annotation_id"] for task in processed_tasks}
        task_counter = len(processed_tasks)

        print(f"Resume from {task_counter} processed tasks")
    else:
        processed_tasks = []
        processed_ids = set()
        task_counter = 0

    remaining = [task for task in tasks if task["annotation_id"] not in processed_ids]

    print(f"Total tasks : {len(tasks)}")
    print(f"Done        : {len(processed_tasks)}")
    print(f"Remaining   : {len(remaining)}")

    if not remaining:
        print("All tasks already finished")
        recall_engine.print_stats()
        return

    unfinished = process_tasks(remaining)

    save_progress()

    if unfinished:
        with open(UNFINISHED_FILE, "w", encoding="utf-8") as file_obj:
            json.dump(unfinished, file_obj, indent=4, ensure_ascii=False)
        print(f"Unfinished tasks saved: {len(unfinished)}")

    recall_engine.print_stats()
    print("Stage2 finished")
    print(f"Processed tasks: {len(processed_tasks)}")


if __name__ == "__main__":
    main()
