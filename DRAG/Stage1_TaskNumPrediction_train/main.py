import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from file_utils import read_json_file, write_json_file
from embedding_client import EmbeddingClient
from embedding_cache import build_or_load_embeddings
from task_retriever import TaskRetriever

# ==================== 路径配置区域 ====================
# 训练集目录：用于建立历史任务库
TRAIN_DATA_DIR = "../../new_Data/train/mul"

# 测试集目录：用于做预测并输出结果
TEST_DATA_DIR = "../../new_Data/test/mul"

TRAIN_TASKS_FILE_PATH = os.path.join(TRAIN_DATA_DIR, "data.json")
TASKS_FILE_PATH = os.path.join(TEST_DATA_DIR, "data.json")
HISTORY_EMBEDDINGS_PATH = os.path.join(TRAIN_DATA_DIR, "stage1_history_embeddings.npy")
UNFINISHED_TASKS_PATH = os.path.join(TEST_DATA_DIR, "unfinished_stage1.json")
PROCESSED_TASKS_PATH = os.path.join(TEST_DATA_DIR, "stage1_tasknum.json")
# ==================== 路径配置结束 ====================

retriever = None
embedder = None

task_counter = 0
counter_lock = threading.Lock()


def update_task_counter():
    global task_counter
    with counter_lock:
        task_counter += 1
        if task_counter % 50 == 0:
            print(f"Processed {task_counter} tasks")


def process_single_task(task):
    query = task["confirmed_task"]

    try:
        query_embedding = embedder.get_embedding(query)
    except Exception as exc:
        print(f"[Query Embedding Error] {exc}")
        query_embedding = None

    task_num = retriever.predict_task_num(
        query_embedding,
        recom_num=task.get("recom_num", 1),
    )
    task["pred_task_num"] = task_num

    update_task_counter()
    return task


def process_tasks(tasks):
    processed_tasks = []
    unfinished_tasks = []

    max_workers = os.cpu_count() or 4

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_single_task, task): task for task in tasks}

        for future in as_completed(futures):
            task = futures[future]
            try:
                result = future.result(timeout=600)
                processed_tasks.append(result)
            except Exception as exc:
                print(f"fail: {exc}, task: {task['confirmed_task']}")
                unfinished_tasks.append(task)

    if unfinished_tasks:
        write_json_file(UNFINISHED_TASKS_PATH, unfinished_tasks)
        print("unfinished saved:", len(unfinished_tasks))

    return processed_tasks


def main():
    global retriever
    global embedder
    global task_counter

    print("=== ENTER STAGE1 ===")
    print(f"Train split  : {TRAIN_DATA_DIR}")
    print(f"Test split   : {TEST_DATA_DIR}")
    print(f"Output file  : {PROCESSED_TASKS_PATH}")

    task_counter = 0

    tasks = read_json_file(TASKS_FILE_PATH)
    train_tasks = read_json_file(TRAIN_TASKS_FILE_PATH)

    print(f"Loaded test tasks  : {len(tasks)}")
    print(f"Loaded train tasks : {len(train_tasks)}")

    embedder = EmbeddingClient()

    embeddings, valid_train_tasks = build_or_load_embeddings(
        train_tasks,
        HISTORY_EMBEDDINGS_PATH,
    )

    retriever = TaskRetriever(valid_train_tasks, embeddings, have_self=True)

    processed_tasks = process_tasks(tasks)
    write_json_file(PROCESSED_TASKS_PATH, processed_tasks)

    retriever.print_stats()

    print("Stage1 finished")
    print(f"Processed tasks: {task_counter}")


if __name__ == "__main__":
    main()
