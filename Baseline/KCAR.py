#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
KCAR + 批处理 + 断点续传 + 自动保存 + unfinished
⚠️ 核心算法完全未修改
"""

import os
import json
import re
import time
import threading
import numpy as np
from collections import deque
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import KeyedVectors

from baseline_runtime import baseline_test_path, get_dataset_name, get_word2vec_binary_path, test_data_dir


# =============================
# 路径配置（统一放这里）
# =============================
DATASET_NAME = get_dataset_name()
BASE_DATA_DIR = test_data_dir(DATASET_NAME)

INPUT_FILE = baseline_test_path("stage1_decompose.json", DATASET_NAME)
OUTPUT_FILE = baseline_test_path("stage3_kcar_result111.json", DATASET_NAME)
UNFINISHED_FILE = baseline_test_path("unfinished_stage3_kcar.json", DATASET_NAME)

GRAPH_FILE_PATH = baseline_test_path("graph_desc.json", DATASET_NAME)
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


def extract_keywords(text):
    words = re.findall(r'\b\w+\b', text.lower())
    prepositions = {"in", "on", "at", "by", "for", "with", "to", "of", "from", "about"}
    return set(word for word in words if word not in prepositions)


def string_to_vector(text, model):
    words = text.split()
    vectors = [model[word] for word in words if word in model]
    if vectors:
        return np.mean(vectors, axis=0)
    return np.zeros(model.vector_size)


def calculate_similarity(api, subtask, model):
    vec1 = string_to_vector(api['target_action_reprs'], model)
    vec2 = string_to_vector(subtask, model)
    return cosine_similarity([vec1], [vec2])[0][0]


def select_top_apis(task, apis, subtask, top_k=5):
    similarities = [
        (api, calculate_similarity(api, subtask, word2vec_model))
        for api in apis
    ]

    similarities.sort(key=lambda x: x[1], reverse=True)
    return [api for api, _ in similarities[:top_k]]


# =============================
# 图加载（未改）
# =============================
def load_graph_from_file():
    print("📊 从文件加载图数据...")
    graph_data = read_json(GRAPH_FILE_PATH)

    graph = {}
    for edge in graph_data:
        s, t = edge["source"], edge["target"]

        if s not in graph:
            graph[s] = []
        if t not in graph:
            graph[t] = []

        graph[s].append((t, 1.0))
        graph[t].append((s, 1.0))

    print(f"✅ 图加载完成：{len(graph)} 个节点")
    return graph


# =============================
# 核心算法（完全未改）
# =============================
def min_group_steiner_tree(task, apis, graph):
    subtasks = task["decomposed_task"]

    subtask_apis = {}
    for subtask in subtasks:
        top_apis = select_top_apis(task, apis, subtask)
        subtask_apis[subtask] = set(api["action_uid"] for api in top_apis)

    class Tree:
        def __init__(self, nodes, weight, covered_subtasks):
            self.nodes = nodes
            self.weight = weight
            self.covered_subtasks = covered_subtasks

    queue = deque()

    for subtask in subtasks:
        for api in apis:
            if api["action_uid"] in subtask_apis[subtask]:
                queue.append(Tree({api["action_uid"]}, 0, {subtask}))

    def tree_growth(tree):
        for node in tree.nodes:
            for neighbor, weight in graph.get(node, []):
                new_nodes = tree.nodes | {neighbor}
                new_weight = tree.weight + weight
                new_covered = tree.covered_subtasks.copy()

                for subtask in subtasks:
                    if neighbor in subtask_apis[subtask]:
                        new_covered.add(subtask)

                if len(new_covered) > len(tree.covered_subtasks):
                    queue.append(Tree(new_nodes, new_weight, new_covered))
                    return

        for node in tree.nodes:
            for neighbor, weight in graph.get(node, []):
                queue.append(Tree(tree.nodes | {neighbor}, tree.weight + weight, tree.covered_subtasks))
                return

    def tree_merge(tree):
        for other in list(queue):
            if other != tree:
                merged_nodes = tree.nodes | other.nodes
                merged_weight = tree.weight + other.weight
                merged_subtasks = tree.covered_subtasks | other.covered_subtasks

                if merged_weight > tree.weight and merged_weight > other.weight:
                    queue.remove(other)
                    queue.append(Tree(merged_nodes, merged_weight, merged_subtasks))
                    return True
        return False

    def tree_end(tree):
        if len(tree.covered_subtasks) == len(subtasks):
            return tree.nodes
        return None

    while 0 < len(queue) < 2000:
        current = queue.popleft()

        result = tree_end(current)
        if result:
            return [api for api in apis if api["action_uid"] in result]

        if len(current.nodes) > 25:
            continue

        tree_growth(current)

        if not tree_merge(current):
            queue.append(current)

    print("⚠️ 搜索失败或队列过长")
    return []


def KCAR_process_task(task, apis, graph):
    # print("🚀 处理任务:", task['confirmed_task'])
    return min_group_steiner_tree(task, apis, graph), []


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


def process_single_task_wrapper(task, apis, graph):
    apis_copy = [a.copy() for a in apis]

    start_time = time.time()
    recom_result, recom_edges = KCAR_process_task(task, apis_copy, graph)
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
    graph = load_graph_from_file()

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
            result_task = process_single_task_wrapper(task, apis, graph)

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
