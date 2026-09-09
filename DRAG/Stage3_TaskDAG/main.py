import os
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from file_utils import read_json_file
from dag_generator import DAGGenerator

# ==================== 路径配置区域 ====================
# 基础目录（和之前文件完全统一，修改只需改这里）
BASE_DATA_DIR = "../../Data_no_self/Llama3.1:70b-bge/hug"
# BASE_DATA_DIR = "../../Data_no_self/Llama3.1:8b-bge/ultratool"


# 输入输出文件路径（通过基础目录拼接，跨平台兼容）
# ✅ 新增
TOOL_DESC_PATH = os.path.join(BASE_DATA_DIR, "tool_desc.json")

# TOOL_DESC_PATH = "../../Data/Gemma27b_normal/mul/tool_desc.json"
INPUT_FILE = os.path.join(BASE_DATA_DIR, "stage15_services.json")
OUTPUT_FILE = os.path.join(BASE_DATA_DIR, "stage2_dag.json")
UNFINISHED_FILE = os.path.join(BASE_DATA_DIR, "unfinished_stage22.json")
# ==================== 路径配置结束 ====================

# =============================
# 其他配置
# =============================
SAVE_INTERVAL = 50
MAX_WORKERS = min(6, os.cpu_count())


generator = None

processed_tasks = []
processed_ids = set()

save_lock = threading.Lock()
task_counter = 0


# =============================
# 保存函数（线程安全）
# =============================

def save_progress():

    processed_tasks.sort(key=lambda x: x["annotation_id"])

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(processed_tasks, f, indent=4, ensure_ascii=False)

    print(f"💾 自动保存 {len(processed_tasks)} 条数据")


# =============================
# 单任务处理
# =============================

def process_single_task(task):

    global task_counter

    query = task["confirmed_task"]
    num = task.get("pred_task_num", 3)

    # ✅ 关键改动：读取 stage1.5 的服务
    service_ids = task.get("st1.5_service", [])

    dag = generator.generate_dag(query, num, service_ids)
    if dag is None:
        raise ValueError("Invalid DAG (cycle or disconnected)")
    task["nodes"] = dag.get("nodes", [])
    task["edges"] = dag.get("edges", [])

    need_save = False

    with save_lock:

        processed_tasks.append(task)
        processed_ids.add(task["annotation_id"])

        task_counter += 1

        if task_counter % 5 == 0:
            print(f"已生成 {task_counter} 个 DAG")

        if len(processed_tasks) % SAVE_INTERVAL == 0:
            need_save = True

    if need_save:
        save_progress()

    return task


# =============================
# 批量处理
# =============================

def process_tasks(tasks):

    unfinished_tasks = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:

        futures = {
            executor.submit(process_single_task, task): task
            for task in tasks
        }

        for future in as_completed(futures):

            task = futures[future]

            try:
                future.result(timeout=600)

            except Exception as e:

                print(f"❌ fail: {e}")
                print(f"task: {task['confirmed_task']}")

                unfinished_tasks.append(task)

    return unfinished_tasks


# =============================
# 主函数
# =============================

def main():

    global generator
    global processed_tasks
    global processed_ids
    global task_counter

    tasks = read_json_file(INPUT_FILE)

    generator = DAGGenerator(TOOL_DESC_PATH)

    # =============================
    # 断点恢复
    # =============================

    if os.path.exists(OUTPUT_FILE):

        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            processed_tasks = json.load(f)

        processed_ids = set(
            task["annotation_id"] for task in processed_tasks
        )

        task_counter = len(processed_tasks)

        print(f"🔁 检测到已有 {task_counter} 条结果，继续运行")

    else:

        processed_tasks = []
        processed_ids = set()

    # =============================
    # 过滤剩余任务
    # =============================

    remaining_tasks = [
        task for task in tasks
        if task["annotation_id"] not in processed_ids
    ]

    print(f"📊 总任务数: {len(tasks)}")
    print(f"✅ 已完成: {len(processed_tasks)}")
    print(f"⏳ 剩余任务: {len(remaining_tasks)}")

    if not remaining_tasks:
        print("🎉 所有任务已完成")
        return

    # =============================
    # 开始生成
    # =============================

    unfinished_tasks = process_tasks(remaining_tasks)

    # =============================
    # 最终保存
    # =============================

    save_progress()

    if unfinished_tasks:

        with open(UNFINISHED_FILE, "w", encoding="utf-8") as f:
            json.dump(unfinished_tasks, f, indent=4, ensure_ascii=False)

        print(f"⚠️ 保存失败任务 {len(unfinished_tasks)} 条")

    print("\n🎉 Stage2 finished")
    print(f"总生成 DAG: {len(processed_tasks)}")


# =============================
# 入口
# =============================

if __name__ == "__main__":
    main()