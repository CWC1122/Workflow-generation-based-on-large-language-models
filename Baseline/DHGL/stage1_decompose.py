#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from baseline_runtime import (
    baseline_test_path,
    get_baseline_llm_model,
    get_dataset_name,
    test_data_dir,
)
from localLLM.localLLM import localLLM

# ==================== 路径配置区域 ====================
DATASET_NAME = get_dataset_name()
BASE_DATA_DIR = test_data_dir(DATASET_NAME)

# 输入输出文件路径（标准化拼接）
INPUT_FILE = baseline_test_path("data.json", DATASET_NAME)
OUTPUT_FILE = baseline_test_path("stage1_decompose.json", DATASET_NAME)
UNFINISHED_FILE = baseline_test_path("unfinished_tasks.json", DATASET_NAME)
# ==================== 路径配置结束 ====================

# =============================
# 全局配置 (所有路径和参数集中在此)
# =============================
SAVE_INTERVAL = 50          # 自动保存间隔 (每处理N条保存一次)
MAX_WORKERS = min(6, os.cpu_count()) # 线程池大小
LLM_MODEL = get_baseline_llm_model("local-chat-model")

# 示例返回数据格式
example_response_format = """
[
    "Get the weather for a specific city and a specific day",
    "Do a specific operation on a specific stock",
    "Send an email to a specific email address",
    "Buy a specific insurance from a specific insurance company",
    "Play a specific music by a specific title"
]
"""

# =============================
# 全局运行时变量
# =============================
processed_tasks = []
processed_ids = set()
save_lock = threading.Lock()
task_counter = 0
total_tasks = 0

# =============================
# 核心逻辑函数
# =============================

def decompose_task(confirmed_task):
    """
    调用本地LLM分解任务 (静默处理，不打印单条提问)
    """
    prompt = f"请学习以下示例数据格式：\n{example_response_format}\n\n,将任务：{confirmed_task}分解，分解为2-10个步骤，返回与上面相同的数据格式，用英文回答，只能返回一个子任务序列，不要返回任何多余的文字，不要有开头语句，不要有结尾语句"

    messages = [
        {"role": "system", "content": "You are a helpful assistant that decomposes tasks into subtasks."},
        {"role": "user", "content": prompt}
    ]
    response = localLLM(messages, model=LLM_MODEL, temperature=0.0)

    # 分解字符串
    steps = [s.strip('\n    "') for s in response.strip('[]').split('\n')]
    if len(steps)>2: del steps[0]
    if len(steps)>2: del steps[-1]

    return steps

def read_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

# =============================
# 工程化功能函数
# =============================

def save_progress():
    """线程安全的保存函数"""
    with save_lock:
        # 排序方便调试
        processed_tasks.sort(key=lambda x: x.get("annotation_id", x["confirmed_task"]))
        
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(processed_tasks, f, indent=4, ensure_ascii=False)
        
        print(f"\n💾 自动保存: {len(processed_tasks)} / {total_tasks}")

def update_progress():
    """更新控制台进度条显示"""
    global task_counter
    with save_lock:
        task_counter += 1
        # 简单的进度条逻辑
        if task_counter % 3 == 0 or task_counter == total_tasks:
            percent = (task_counter / total_tasks) * 100
            print(f"\r⏳ 进度: {task_counter}/{total_tasks} ({percent:.1f}%)", end="", flush=True)

# =============================
# 任务处理流程
# =============================

def process_single_task(task):
    need_save = False
    try:
        confirmed_task = task["confirmed_task"]
        decomposed_task = decompose_task(confirmed_task)
        task["decomposed_task"] = decomposed_task

        with save_lock:
            processed_tasks.append(task)
            # 使用 annotation_id 或 confirmed_task 作为去重ID
            task_id = task.get("annotation_id", confirmed_task)
            processed_ids.add(task_id)
            
            # 检查是否需要自动保存
            if len(processed_tasks) % SAVE_INTERVAL == 0:
                need_save = True
        
        update_progress()
        
        if need_save:
            save_progress()
            
        return task
    except Exception as e:
        # 异常处理，确保单个任务失败不卡死进度条
        with save_lock:
            pass 
        raise e # 抛出异常给外层捕获

def process_tasks(tasks):
    unfinished_tasks = []
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_single_task, task): task for task in tasks}
        
        for future in as_completed(futures):
            task = futures[future]
            try:
                future.result(timeout=600)
            except Exception as e:
                print(f"\n❌ 任务失败: {e}")
                print(f"   内容: {task['confirmed_task'][:50]}...")
                unfinished_tasks.append(task)
                update_progress() # 即使失败也更新进度
                
    return unfinished_tasks

# =============================
# 主入口
# =============================

def main():
    global processed_tasks, processed_ids, total_tasks

    # 1. 加载数据 (默认加载全部)
    all_tasks = read_json_file(INPUT_FILE)

    # 2. 断点恢复
    if os.path.exists(OUTPUT_FILE):
        print(f"🔁 正在加载历史进度...")
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            processed_tasks = json.load(f)
        
        processed_ids = set(t.get("annotation_id", t["confirmed_task"]) for t in processed_tasks)
        
        # 【关键修复】同步计数器
        global task_counter
        task_counter = len(processed_tasks) 
        
        print(f"✅ 已恢复 {len(processed_tasks)} 条记录")
    else:
        processed_tasks = []
        processed_ids = set()
    # 3. 过滤任务
    remaining_tasks = [t for t in all_tasks if t.get("annotation_id", t["confirmed_task"]) not in processed_ids]
    total_tasks = len(all_tasks)
    
    print(f"📊 总任务: {total_tasks} | 剩余: {len(remaining_tasks)}")

    if not remaining_tasks:
        print("🎉 所有任务已完成！")
        return

    # 4. 开始执行
    print("🚀 开始处理...")
    unfinished_tasks = process_tasks(remaining_tasks)

    # 5. 最终保存
    save_progress()
    
    if unfinished_tasks:
        with open(UNFINISHED_FILE, "w", encoding="utf-8") as f:
            json.dump(unfinished_tasks, f, indent=4, ensure_ascii=False)
        print(f"\n⚠️  保存失败任务 {len(unfinished_tasks)} 条至 {UNFINISHED_FILE}")

    print(f"\n🏁 全部完成! 最终结果保存在: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
