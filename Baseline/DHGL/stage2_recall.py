#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import json
import re
import numpy as np
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from baseline_runtime import (
    baseline_test_path,
    get_baseline_llm_model,
    get_dataset_name,
    get_word2vec_binary_path,
    test_data_dir,
)
from localLLM.localLLM import localLLM
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import KeyedVectors

# ==================== 路径配置区域 ====================
DATASET_NAME = get_dataset_name()
BASE_DATA_DIR = test_data_dir(DATASET_NAME)

# 输入输出文件路径
INPUT_FILE = baseline_test_path("stage1_decompose.json", DATASET_NAME)
OUTPUT_FILE = baseline_test_path("stage2_recall.json", DATASET_NAME)
UNFINISHED_FILE = baseline_test_path("unfinished_stage2.json", DATASET_NAME)
API_FILE = baseline_test_path("tool_desc.json", DATASET_NAME)
# ==================== 路径配置结束 ====================

# =============================
# 其他全局配置
# =============================
SAVE_INTERVAL = 20
MAX_WORKERS = 1

TOP_M = 10
MAX_ITER = 5
LAMBDA_THRESH = 0.05
LLM_MAX_RETRIES = 10  # 大模型调用最大重试次数
LLM_MODEL = get_baseline_llm_model("local-chat-model")
WORD2VEC_MODEL_PATH = get_word2vec_binary_path()

# =============================
# 全局运行时变量
# =============================
processed_tasks = []
processed_ids = set()
save_lock = threading.Lock()
task_counter = 0
total_tasks = 0
word2vec_model = None

# =============================
# 核心逻辑
# =============================

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return ' '.join(text.split())

def string_to_vector(text, model):
    words = preprocess_text(text).split()
    vectors = [model[word] for word in words if word in model]
    return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)

def calculate_similarity(str1, str2, model):
    vec1, vec2 = string_to_vector(str1, model), string_to_vector(str2, model)
    return cosine_similarity([vec1], [vec2])[0][0]

def compute_value_gain(Tk, Rk, Q, model):
    subtask_service_sim = np.mean([max([calculate_similarity(t, s, model) for s in Rk] or [0]) for t in Tk])
    subtask_query_sim = np.mean([calculate_similarity(t, Q, model) for t in Tk])
    return subtask_service_sim + subtask_query_sim

def recall_llm(Tk, Rk, Nk, Q, task_identifier=""):
    """
    调用大模型，带重试机制。
    返回: (result_steps, total_attempts_made)
    """
    prompt = f"""
    你是一名专业的任务规划师与严格的审查者。请使用以下输入信息：仔细分析当前的子任务序列，检查每个子任务是否有必要，以及是否符合用户查询的意图。参考召回池中的服务，评估这些子任务在现有服务下是否可行。利用被移除的服务作为反例，避免生成与这些无效服务相似的新子任务。然后，请重新生成一个精炼的子任务序列，使其更好地满足用户的原始意图，并且能够用召回池中的服务直接执行。请确保子任务简洁、逻辑有序，并避免出现反例中暴露的问题。只以原格式返回新的子任务列表，不要输出任何多余信息
    用户查询: {Q}
    当前子任务: {Tk}
    召回池服务: {Rk}
    已移除服务（反例）: {Nk}
    """
    
    messages = [
        {"role": "system", "content": "You are a professional task planner and a strict reviewer."},
        {"role": "user", "content": prompt}
    ]

    last_exception = None
    for attempt in range(1, LLM_MAX_RETRIES + 1):
        # 【修改1】打印当前是第几次尝试（为了避免多线程刷屏，只打印简短提示）
        # 注意：多线程环境下这里的打印可能会穿插，但能看到实时状态
        # print(f"[{task_identifier}] LLM尝试 {attempt}/{LLM_MAX_RETRIES}...") 
        
        try:
            response = localLLM(
                messages,
                model=LLM_MODEL,
                temperature=0.0,
            )
            steps = [s.strip('\n "') for s in response.strip('[]').split('\n') if s.strip()]
            return steps, attempt # 成功返回：(结果, 尝试次数)
        except Exception as e:
            last_exception = e
            # 打印失败尝试
            print(f"\n⚠️  [{task_identifier}] LLM第{attempt}次调用失败: {str(e)[:50]}...")
    
    # 3次都失败
    print(f"\n❌ [{task_identifier}] LLM连续{LLM_MAX_RETRIES}次调用失败，使用原始子任务兜底")
    return None, LLM_MAX_RETRIES

def process_single_task(task, apis):
    global word2vec_model
    Q = task["confirmed_task"]
    # 生成一个简短的任务ID用于打印
    task_id_short = task.get("annotation_id", Q[:10])
    
    GeneralPool = apis

    CandidatePool = sorted(
        GeneralPool, 
        key=lambda api: calculate_similarity(Q, api['target_action_reprs'], word2vec_model), 
        reverse=True
    )[:2*TOP_M]
    
    Rk = sorted(
        CandidatePool, 
        key=lambda api: calculate_similarity(" ".join(task.get("decomposed_task", [])), api['target_action_reprs'], word2vec_model),
        reverse=True
    )[:TOP_M]
    Rk = [api['target_action_reprs'] for api in Rk]

    Tk = task.get("decomposed_task", [])
    Value_prev = compute_value_gain(Tk, Rk, Q, word2vec_model)
    Nk = []
    iter_count = 0
    recall_results = []
    llm_failed = False
    
    # 【修改2】新增总调用次数计数器
    total_llm_calls_for_task = 0

    while iter_count < MAX_ITER:
        iter_count += 1
        
        # 调用LLM，并获取本次尝试次数
        new_Tk, attempts_this_time = recall_llm(Tk, Rk, Nk, Q, task_identifier=task_id_short)
        
        # 累加总调用次数
        total_llm_calls_for_task += attempts_this_time

        # LLM调用失败处理
        if new_Tk is None:
            llm_failed = True
            break

        Value_curr = compute_value_gain(new_Tk, Rk, Q, word2vec_model)
        Delta = Value_curr - Value_prev

        if Delta > LAMBDA_THRESH:
            Tk = new_Tk
            Nk = list(set(Rk) - set([api['target_action_reprs'] for api in CandidatePool]))
            Rk = sorted(
                CandidatePool, 
                key=lambda api: max([calculate_similarity(t, api['target_action_reprs'], word2vec_model) for t in Tk]),
                reverse=True
            )[:TOP_M]
            Rk = [api['target_action_reprs'] for api in Rk]
            Value_prev = Value_curr
            
            overlap_count = len(set(Tk) & set(Rk))
            recall_results.append({
                "recall_round": iter_count,
                "recalled_task": Tk.copy(),
                "overlap_count": overlap_count,
                "llm_success": True
            })
        else:
            overlap_count = len(set(Tk) & set(Rk))
            recall_results.append({
                "recall_round": iter_count,
                "recalled_task": Tk.copy(),
                "overlap_count": overlap_count,
                "llm_success": True
            })
            break

    # 兜底处理
    if llm_failed:
        overlap_count = len(set(Tk) & set(Rk))
        recall_results.append({
            "recall_round": iter_count,
            "recalled_task": Tk.copy(),
            "overlap_count": overlap_count,
            "llm_success": False,
            "note": "LLM调用失败，使用原始子任务兜底"
        })

    # 【修改3】保存结果并打印总结
    task["recall_results"] = recall_results
    task["llm_total_calls"] = total_llm_calls_for_task # 新增字段保存次数
    
    # 打印该任务的最终调用情况
    status = "成功" if not llm_failed else "失败兜底"
    print(f"\n✅ [{task_id_short}] 任务完成，状态: {status}，总计调用LLM: {total_llm_calls_for_task} 次")

    return task

def read_json_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

# =============================
# 工程化功能
# =============================

def save_progress():
    with save_lock:
        processed_tasks.sort(key=lambda x: x.get("annotation_id", x["confirmed_task"]))
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(processed_tasks, f, indent=4, ensure_ascii=False)
        print(f"\n💾 自动保存: {len(processed_tasks)} / {total_tasks}")

def update_progress():
    global task_counter
    with save_lock:
        task_counter += 1
        # 进度条依然保留
        if task_counter % 1 == 0 or task_counter == total_tasks:
            percent = (task_counter / total_tasks) * 100
            print(f"\r⏳ 总进度: {task_counter}/{total_tasks} ({percent:.1f}%)", end="", flush=True)

# =============================
# 任务处理
# =============================

def process_tasks_wrapper(tasks, apis):
    unfinished_tasks = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_single_task, task, apis): task for task in tasks}
        for future in as_completed(futures):
            task = futures[future]
            need_save = False
            try:
                result_task = future.result(timeout=1800)
                with save_lock:
                    processed_tasks.append(result_task)
                    task_id = result_task.get("annotation_id", result_task["confirmed_task"])
                    processed_ids.add(task_id)
                    if len(processed_tasks) % SAVE_INTERVAL == 0:
                        need_save = True
                update_progress()
                if need_save:
                    save_progress()
            except Exception as e:
                print(f"\n❌ 任务非LLM异常失败: {e}")
                print(f"   内容: {task['confirmed_task'][:50]}...")
                unfinished_tasks.append(task)
                update_progress()
    return unfinished_tasks

# =============================
# 主入口
# =============================

def main():
    global processed_tasks, processed_ids, total_tasks, word2vec_model, task_counter

    print("🔧 正在加载 Word2Vec 模型...")
    word2vec_model = KeyedVectors.load_word2vec_format(WORD2VEC_MODEL_PATH, binary=True)
    print("✅ 模型加载完成")

    print("📂 正在读取数据...")
    all_tasks = read_json_file(INPUT_FILE)
    apis = read_json_file(API_FILE)

    # 断点恢复
    if os.path.exists(OUTPUT_FILE):
        print(f"🔁 正在加载历史进度...")
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            processed_tasks = json.load(f)
        
        processed_ids = set(t.get("annotation_id", t["confirmed_task"]) for t in processed_tasks)
        task_counter = len(processed_tasks)
        print(f"✅ 已恢复 {len(processed_tasks)} 条记录")
    else:
        processed_tasks = []
        processed_ids = set()
        task_counter = 0

    remaining_tasks = [t for t in all_tasks if t.get("annotation_id", t["confirmed_task"]) not in processed_ids]
    total_tasks = len(all_tasks)
    
    print(f"📊 总任务: {total_tasks} | 剩余: {len(remaining_tasks)}")

    if not remaining_tasks:
        print("🎉 所有任务已完成！")
        return

    print("🚀 开始处理 (Stage 2 Recall)...")
    unfinished_tasks = process_tasks_wrapper(remaining_tasks, apis)

    save_progress()
    
    if unfinished_tasks:
        with open(UNFINISHED_FILE, "w", encoding="utf-8") as f:
            json.dump(unfinished_tasks, f, indent=4, ensure_ascii=False)
        print(f"\n⚠️  保存非LLM异常失败任务 {len(unfinished_tasks)} 条至 {UNFINISHED_FILE}")

    print(f"\n🏁 Stage 2 全部完成! 最终结果保存在: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
