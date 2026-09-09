# =============================
# main.py - Stage7 主入口
# =============================

import os
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from file_utils import read_json_file
from chain_merger import ChainMerger


# =============================
# 全局配置
# =============================

# ==================== 路径配置区域 ====================
# 基础目录，统一管理
# BASE_DATA_DIR = "../../Data/Gemma12b_mxbai_qishishibge123/mul"

BASE_DATA_DIR = "../../Data_no_self/Gemma12b_bge/hug"

# 输入输出文件路径（标准化拼接）
INPUT_FILE = os.path.join(BASE_DATA_DIR, "stage6_selected_weight_sorted.json")
SERVICE_FILE = os.path.join(BASE_DATA_DIR, "tool_desc.json")
OUTPUT_FILE = os.path.join(BASE_DATA_DIR, "stage7_final_weight_sorted.json")
UNFINISHED_FILE = os.path.join(BASE_DATA_DIR, "unfinished_stage7.json")
# ==================== 路径配置结束 ====================

# 打印当前基础路径（方便确认目录）
print(f"📂 当前数据基础目录: {os.path.abspath(BASE_DATA_DIR)}")
SAVE_INTERVAL = 99999
MAX_WORKERS = min(6, os.cpu_count())


merger = None

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

    result = merger.process_task(task)

    need_save = False
    with save_lock:
        processed_tasks.append(result)
        processed_ids.add(task["annotation_id"])
        task_counter += 1
        if task_counter % 10 == 0:
            print(f"已处理 {task_counter} 个任务")
        if len(processed_tasks) % SAVE_INTERVAL == 0:
            need_save = True
    if need_save:
        save_progress()
    return result


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
                future.result(timeout=60)
            except Exception as e:
                print(f"❌ fail: {e}")
                print(f"task: {task['confirmed_task']}")
                unfinished_tasks.append(task)
    return unfinished_tasks


# =============================
# 主函数
# =============================

def main():
    global merger
    global processed_tasks
    global processed_ids
    global task_counter

    tasks = read_json_file(INPUT_FILE)
    services = read_json_file(SERVICE_FILE)

    if not tasks:
        print("No tasks found in input file.")
        return

    # 构建服务字典
    service_dict = {s["action_uid"]: s for s in services}

    # 初始化合并器
    merger = ChainMerger(tasks[0]["edges"], service_dict)

    # 断点恢复
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            processed_tasks = json.load(f)
        processed_ids = set(task["annotation_id"] for task in processed_tasks)
        task_counter = len(processed_tasks)
        print(f"🔁 检测到已有 {task_counter} 条结果，继续运行")
    else:
        processed_tasks = []
        processed_ids = set()

    # 过滤剩余任务
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

    # 开始处理
    unfinished_tasks = process_tasks(remaining_tasks)

    # 最终保存
    save_progress()

    if unfinished_tasks:
        with open(UNFINISHED_FILE, "w", encoding="utf-8") as f:
            json.dump(unfinished_tasks, f, indent=4, ensure_ascii=False)
        print(f"⚠️ 保存失败任务 {len(unfinished_tasks)} 条")

    print("\n🎉 Stage7 finished")
    print(f"总处理任务: {len(processed_tasks)}")


# =============================
# 入口
# =============================

if __name__ == "__main__":
    main()
