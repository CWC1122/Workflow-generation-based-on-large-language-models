#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Greedy + 批处理 + 断点续传 + 自动保存 + unfinished
⚠️ 核心算法完全未修改
"""

import os
import json
import re
import time
import threading
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import KeyedVectors

from baseline_runtime import baseline_test_path, get_dataset_name, get_word2vec_binary_path, test_data_dir


# =============================
# 路径配置（统一放这里）
# =============================
DATASET_NAME = get_dataset_name()
BASE_DATA_DIR = test_data_dir(DATASET_NAME)

INPUT_FILE = baseline_test_path("data.json", DATASET_NAME)
OUTPUT_FILE = baseline_test_path("greedy_result.json", DATASET_NAME)
UNFINISHED_FILE = baseline_test_path("unfinished_greedy.json", DATASET_NAME)

TEST_APIS_PATH = baseline_test_path("tool_desc.json", DATASET_NAME)

WORD2VEC_PATH = get_word2vec_binary_path()

print(f"📂 当前数据目录: {os.path.abspath(BASE_DATA_DIR)}")

# =============================
# 工程配置
# =============================
SAVE_INTERVAL = 50

# =============================
# 全局变量
# =============================
word2vec_model = None
processed_tasks = []
unfinished_tasks = []
processed_ids = set()
save_lock = threading.Lock()
task_counter = 0
total_tasks = 0


# =============================
# 原有工具函数（未改）
# =============================
def read_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


# 使用word2vec将字符串转换为向量
def string_to_vector(text, model):
    words = text.split()
    vectors = [model[word] for word in words if word in model]
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(model.vector_size)


# 计算余弦相似度
def calculate_similarity(str1, str2, model):
    # 确保输入的字符串是有效的
    if not str1 or not str2:
        return 0
    vec1 = string_to_vector(str1, model)
    vec2 = string_to_vector(str2, model)
    return cosine_similarity([vec1], [vec2])[0][0]


# 将 API 信息拼接成一个长字符串
def concatenate_api_info(api):
    info = []
    info.append(api.get('target_action_reprs', ''))
    return ' '.join(info)


# 将一个API组的信息拼接成一个长字符串
def concatenate_api_group_info(api_group, task_desc, model):
    """
    将一组 API 的信息拼接成一个长字符串，优先选择与 Task 描述更相似的字段
    """
    concatenated_info = []
    for api in api_group:
        # 计算 target_action_reprs 和 enhanced_reprs 与 Task 描述的相似度
        target_sim = calculate_api_task_similarity(task_desc,
                                                   {'target_action_reprs': api.get('target_action_reprs', '')}, model)
        enhanced_sim = calculate_api_task_similarity(task_desc, {'enhanced_reprs': api.get('enhanced_reprs', '')},
                                                     model)

        # 选择相似度更高的字段进行拼接
        if target_sim >= enhanced_sim:
            concatenated_info.append(api.get('target_action_reprs', ''))
        else:
            concatenated_info.append(api.get('enhanced_reprs', ''))

    # 将所有 API 的信息拼接成一个长字符串
    return ' '.join(concatenated_info)


def calculate_api_task_similarity(task_desc, api, model):
    similarities = []
    for field in ['target_action_reprs', 'enhanced_reprs']:
        if field in api:
            api_desc = api[field]
            sim = calculate_similarity(task_desc, api_desc, model)
            similarities.append(sim)
    if similarities:
        return max(similarities)
    else:
        return -1


def calculate_group_task_similarity(task_desc, api_group, model):
    """
    计算一组 API 与 Task 描述的相似度
    """
    # 将 API 组的信息拼接成一个长字符串
    api_group_info = concatenate_api_group_info(api_group, task_desc, model)

    # 计算 Task 描述与 API 组信息的相似度
    similarity = calculate_similarity(task_desc, api_group_info, model)

    return similarity


# =============================
# 核心算法（完全未改，仅移除domain筛选）
# =============================
def find_top_matching_apis(task_desc, apis, model):
    # 已按要求去掉 domain 筛选，数据集已提前过滤
    filtered_apis = apis

    # 计算每个API与Task的相似度
    api_similarities = []
    for api in filtered_apis:
        sim = calculate_api_task_similarity(task_desc, api, model)
        api_similarities.append((sim, api))

    # 按相似度从高到低排序
    api_similarities.sort(key=lambda x: x[0], reverse=True)

    recommended_apis = []
    current_similarity = -1

    for sim, api in api_similarities:
        # 暂时加入推荐列表
        temp_group = recommended_apis + [api]
        temp_similarity = calculate_group_task_similarity(task_desc, temp_group, model)

        # 检查是否提高相似度
        if temp_similarity > current_similarity:
            recommended_apis.append(api)
            current_similarity = temp_similarity
        else:
            break  # 终止推荐

    return recommended_apis


def Greedy_process_task(task, apis, model):
    task_desc = task['confirmed_task']
    recommended_apis = find_top_matching_apis(task_desc, apis, model)
    return recommended_apis, []


# =============================
# 工程增强（新增）
# =============================
def clean_data(data):
    if isinstance(data, dict):
        return {k: clean_data(v) for k, v in data.items() if k != "vector"}
    elif isinstance(data, list):
        return [clean_data(i) for i in data]
    elif isinstance(data, np.ndarray):
        return data.tolist()
    return data


def save_progress():
    with save_lock:
        processed_tasks.sort(key=lambda x: x.get("annotation_id", x["confirmed_task"]))

        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(clean_data(processed_tasks), f, indent=4, ensure_ascii=False)

        with open(UNFINISHED_FILE, "w", encoding="utf-8") as f:
            json.dump(clean_data(unfinished_tasks), f, indent=4, ensure_ascii=False)

        print(f"\n💾 保存：成功 {len(processed_tasks)} | 失败 {len(unfinished_tasks)}")


def update_progress():
    global task_counter
    with save_lock:
        task_counter += 1
        percent = (task_counter / total_tasks) * 100
        print(f"\r⏳ 进度: {task_counter}/{total_tasks} ({percent:.1f}%)", end="", flush=True)


def process_single_task_wrapper(task, apis, model):
    apis_copy = [a.copy() for a in apis]

    start_time = time.time()
    recom_result, recom_edges = Greedy_process_task(task, apis_copy, model)
    elapsed = time.time() - start_time

    # 👉 关键：空结果算失败
    if not recom_result:
        return None

    task['recom_result'] = recom_result
    task['recom_edges'] = recom_edges
    task['all_recom_results'] = []
    task['process_time'] = elapsed

    return task


# =============================
# 主流程（增强版）
# =============================
def main():
    global word2vec_model, processed_tasks, processed_ids, task_counter, total_tasks

    print("🔧 加载 Word2Vec...")
    word2vec_model = KeyedVectors.load_word2vec_format(WORD2VEC_PATH, binary=True)

    print("📂 读取数据...")
    all_tasks = read_json(INPUT_FILE)
    apis = read_json(TEST_APIS_PATH)

    # ===== 断点恢复 =====
    if os.path.exists(OUTPUT_FILE):
        processed_tasks = read_json(OUTPUT_FILE)
        processed_ids = set(t.get("annotation_id", t["confirmed_task"]) for t in processed_tasks)
        task_counter = len(processed_tasks)
        print(f"🔁 恢复 {task_counter}")

    if os.path.exists(UNFINISHED_FILE):
        unfinished_tasks.extend(read_json(UNFINISHED_FILE))

    remaining_tasks = [
        t for t in all_tasks
        if t.get("annotation_id", t["confirmed_task"]) not in processed_ids
    ]

    total_tasks = len(all_tasks)

    print(f"📊 总任务: {total_tasks} | 剩余: {len(remaining_tasks)}")

    # ===== 主循环 =====
    for task in remaining_tasks:
        need_save = False

        try:
            result_task = process_single_task_wrapper(task, apis, word2vec_model)

            with save_lock:
                task_id = task.get("annotation_id", task["confirmed_task"])

                if result_task:
                    processed_tasks.append(result_task)
                    processed_ids.add(task_id)
                else:
                    unfinished_tasks.append(task)

                if len(processed_tasks) % SAVE_INTERVAL == 0:
                    need_save = True

            update_progress()

            if need_save:
                save_progress()

        except Exception as e:
            print(f"\n❌ 任务失败: {e}")
            unfinished_tasks.append(task)
            update_progress()

    save_progress()
    print("\n🏁 全部完成")


if __name__ == "__main__":
    main()
