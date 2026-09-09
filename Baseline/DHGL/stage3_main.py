#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import os
import json
import re
import time
import threading
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import KeyedVectors

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from baseline_runtime import (
    baseline_test_path,
    get_dataset_name,
    get_word2vec_binary_path,
    test_data_dir,
)

# ==================== 路径配置区域 ====================
DATASET_NAME = get_dataset_name()
BASE_DATA_DIR = test_data_dir(DATASET_NAME)

# 输入输出文件路径（标准化拼接）
INPUT_FILE = baseline_test_path("stage25_get_num_test_auto1.json", DATASET_NAME)
OUTPUT_FILE = baseline_test_path("stage3_final_result1.json", DATASET_NAME)
UNFINISHED_FILE = baseline_test_path("unfinished_stage3.json", DATASET_NAME)
API_FILE = baseline_test_path("tool_desc.json", DATASET_NAME)
GRAPH_FILE = baseline_test_path("graph_desc.json", DATASET_NAME)
print(f"📂 当前数据基础目录: {os.path.abspath(BASE_DATA_DIR)}")
# ==================== 路径配置结束 ====================


# =============================
# 全局配置 (所有路径和参数集中在此)
# =============================

SAVE_INTERVAL = 50           # 自动保存间隔
MAX_WORKERS = 3               # CPU密集型，建议保持单线程
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
# 第一部分：Stage3 Construct 核心算法
# =============================

def preprocess_text(text):
    return ' '.join(re.sub(r'[^\w\s]', '', text.lower()).split())

def calculate_tfidf(documents):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)
    return tfidf_matrix.toarray(), vectorizer.get_feature_names_out()

def get_word2vec_vector(word):
    try:
        return word2vec_model[word]
    except KeyError:
        return np.zeros(word2vec_model.vector_size)

def get_weighted_word2vec(tfidf_weights, vocab, doc_index, text):
    words = preprocess_text(text).split()
    doc_vec = np.zeros(word2vec_model.vector_size)
    for word in words:
        try:
            idx = list(vocab).index(word)
            weight = tfidf_weights[doc_index][idx]
            doc_vec += get_word2vec_vector(word) * weight
        except (ValueError, KeyError):
            continue
    return doc_vec.reshape(1, -1)

def cosine_sim(vec1, vec2):
    return cosine_similarity(vec1, vec2)[0][0]

def find_core_special_subtasks(subtasks_vectors, task_vector, subtasks_text):
    sims_to_query = [cosine_sim(v, task_vector) for v in subtasks_vectors]
    t_core_index = np.argmax(sims_to_query)

    if len(subtasks_vectors) == 1:
        t_special_index = 0
        return t_core_index, t_special_index
    
    inter_sims = []
    for i, v in enumerate(subtasks_vectors):
        sim_sum = sum(cosine_sim(v, subtasks_vectors[j])
                      for j in range(len(subtasks_vectors)) if j != i)
        avg_sim = sim_sum / (len(subtasks_vectors) - 1)
        inter_sims.append(avg_sim)
    t_special_index = np.argmin(inter_sims)
    return t_core_index, t_special_index

def calculate_score(V_nodes, E_edges, task_vector, subtasks_vectors, graph_data):
    if len(V_nodes) == 0:
        return 0
    task_sim = np.mean([cosine_sim(v['vector'], task_vector) for v in V_nodes])
    subtask_sim = np.mean([max(cosine_sim(v['vector'], s) for s in subtasks_vectors) for v in V_nodes])
    edge_weight = np.mean([e['weight'] for e in E_edges]) if E_edges else 0
    compactness = 1 / np.log(len(V_nodes) + 1)
    return task_sim + subtask_sim + edge_weight + compactness

def expand_agent(agent, all_services, task_vector, subtasks_vectors, graph_data, h):
    V_star = agent['nodes']
    E_star = agent['edges']

    while len(V_star) < h:
        neighbors = []
        for v in V_star:
            for edge in graph_data:
                if edge['source'] == v['id'] and edge['target'] not in [n['id'] for n in V_star]:
                    neighbors.append((edge['target'], edge))
                if edge['target'] == v['id'] and edge['source'] not in [n['id'] for n in V_star]:
                    neighbors.append((edge['source'], edge))

        best_score, best_node, best_edge = -1, None, None
        for nid, edge in neighbors:
            new_node = next((s for s in all_services if s['id'] == nid), None)
            if not new_node:
                continue
            new_V = V_star + [new_node]
            new_E = E_star + [edge]
            score = calculate_score(new_V, new_E, task_vector, subtasks_vectors, graph_data)
            if score > best_score:
                best_score, best_node, best_edge = score, new_node, edge

        if not best_node:
            break
        V_star.append(best_node)
        E_star.append(best_edge)

    agent['nodes'] = V_star
    agent['edges'] = E_star
    return agent

def dual_channel_solve(task, services, graph_data):
    documents = [preprocess_text(task['confirmed_task'])] + \
                [preprocess_text(s) for s in task['recall_results'][0]['recalled_task']] + \
                [preprocess_text(s['target_action_reprs']) for s in services]
    
    if len(documents) < 2:
        return [], []
        
    tfidf_weights, vocab = calculate_tfidf(documents)

    task_vec = get_weighted_word2vec(tfidf_weights, vocab, 0, documents[0])
    subtasks_vecs = [get_weighted_word2vec(tfidf_weights, vocab, i + 1, doc) for i, doc in enumerate(task['recall_results'][0]['recalled_task'])]
    
    for i, service in enumerate(services):
        service['vector'] = get_weighted_word2vec(tfidf_weights, vocab, len(subtasks_vecs) + 1 + i, service['target_action_reprs'])
        service['id'] = service['action_uid']

    core_idx, special_idx = find_core_special_subtasks(subtasks_vecs, task_vec, task['recall_results'][0]['recalled_task'])
    t_core = subtasks_vecs[core_idx]
    t_special = subtasks_vecs[special_idx]

    agents = []
    for kind, t_ref in [('core', t_core), ('special', t_special)]:
        sims = [(cosine_sim(s['vector'], t_ref), s) for s in services]
        sims.sort(reverse=True, key=lambda x: x[0])
        for _, s in sims[:5]:
            agents.append({'type': kind, 'nodes': [s], 'edges': []})

    h = task.get('recom_num', 5)
    for agent in agents:
        expand_agent(agent, services, task_vec, subtasks_vecs, graph_data, h)

    best_agent = max(agents, key=lambda a: calculate_score(a['nodes'], a['edges'], task_vec, subtasks_vecs, graph_data))
    return best_agent['nodes'], best_agent['edges']

# =============================
# 第二部分：指标计算 (Input Output 逻辑)
# =============================

def calculate_task_metrics(recom_result, actual_apis_reprs):
    recom_true_positives = 0
    actual_true_positives = 0
    total_true = 0
    recom_result_reprs = set()

    for api in recom_result:
        if api is not None and 'target_action_reprs' in api:
            recom_result_reprs.add(api['target_action_reprs'])
            if api['target_action_reprs'] in actual_apis_reprs:
                recom_true_positives += 1
    for api_repr in actual_apis_reprs:
        if api_repr in recom_result_reprs:
                actual_true_positives += 1

    precision = recom_true_positives / len(recom_result) if len(recom_result) > 0 else 0
    recall = actual_true_positives / len(actual_apis_reprs) if len(actual_apis_reprs) > 0 else 0
    if precision == 1.0 and recall == 1.0:
        total_true = 1.0
        
    return precision, recall, total_true

def calculate_task_granularity_deviation(recom_result, actual_steps):
    recom_result_count = len(recom_result)
    actual_steps_count = len(actual_steps)
    deviation = abs(recom_result_count - actual_steps_count) / actual_steps_count if actual_steps_count > 0 else 0
    return deviation

def calculate_executability(task, test_api_dict, graph_data):
    action_reprs = [s['target_action_reprs'] for s in task.get('recom_result', [])]
    num_pairs = len(action_reprs) - 1
    if num_pairs <= 0:
        return 0.0

    adjacency_list = {}
    for link in graph_data:
        source = link['source']
        target = link['target']
        if source not in adjacency_list:
            adjacency_list[source] = set()
        adjacency_list[source].add(target)

    executable_pairs = 0
    for i in range(num_pairs):
        current_repr = action_reprs[i]
        next_repr = action_reprs[i + 1]

        current_api = test_api_dict.get(current_repr)
        next_api = test_api_dict.get(next_repr)

        if not current_api or not next_api:
            continue
        current_api_id = current_api.get('action_uid')
        next_api_id = next_api.get('action_uid')

        if not current_api_id or not next_api_id:
            continue

        if current_api_id in adjacency_list and next_api_id in adjacency_list[current_api_id]:
            executable_pairs += 1

    return executable_pairs / num_pairs if num_pairs > 0 else 0.0

def calculate_and_print_average_metrics(completed_tasks, test_api_dict, graph_data):
    """计算并打印最终指标"""
    print("\n" + "="*30)
    print("📊 正在计算最终指标...")
    
    total_precision = 0
    total_recall = 0
    total_granularity_deviation = 0
    total_executability = 0 
    total_whole_precision = 0
    num_tasks = len(completed_tasks)

    for task in completed_tasks:
        recom_result = task.get('recom_result', [])
        actual_apis_reprs = task.get('action_reprs', [])

        precision, recall, whole_precision = calculate_task_metrics(recom_result, actual_apis_reprs)
        total_precision += precision
        total_recall += recall
        total_whole_precision += whole_precision

        granularity_deviation = calculate_task_granularity_deviation(recom_result, actual_apis_reprs)
        total_granularity_deviation += granularity_deviation

        executability = calculate_executability(task, test_api_dict, graph_data)
        total_executability += executability

    average_precision = total_precision / num_tasks if num_tasks > 0 else 0
    average_recall = total_recall / num_tasks if num_tasks > 0 else 0
    average_granularity_deviation = total_granularity_deviation / num_tasks if num_tasks > 0 else 0
    average_executability = total_executability / num_tasks if num_tasks > 0 else 0
    average_total_whole_precision = total_whole_precision / num_tasks if num_tasks > 0 else 0

    print(f"✅ 指标计算完成 (样本数: {num_tasks})")
    print(f"平均精确率:     {average_precision:.4f}")
    print(f"平均召回率:     {average_recall:.4f}")
    print(f"平均粒度偏差:   {average_granularity_deviation:.4f}")
    print(f"平均可执行性:   {average_executability:.4f}")
    print(f"平均整体成功率: {average_total_whole_precision:.4f}")
    print("="*30 + "\n")

# =============================
# 第三部分：工程化辅助函数
# =============================

def read_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def clean_data(data):
    """清理数据，移除vector和numpy类型，方便json保存"""
    if isinstance(data, dict):
        cleaned = {}
        for key, value in data.items():
            if key == "vector":
                continue
            if isinstance(value, np.ndarray):
                cleaned[key] = value.tolist()
            elif isinstance(value, (dict, list)):
                cleaned[key] = clean_data(value)
            elif isinstance(value, (str, int, float, bool)) or value is None:
                cleaned[key] = value
        return cleaned
    elif isinstance(data, list):
        return [clean_data(item) for item in data]
    else:
        return data

def save_progress():
    with save_lock:
        # 排序
        processed_tasks.sort(key=lambda x: x.get("annotation_id", x["confirmed_task"]))
        # 清理数据后保存
        data_to_save = clean_data(processed_tasks)
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(data_to_save, f, indent=4, ensure_ascii=False)
        print(f"\n💾 自动保存: {len(processed_tasks)} / {total_tasks}")

def update_progress():
    global task_counter
    with save_lock:
        task_counter += 1
        if task_counter % 1 == 0 or task_counter == total_tasks:
            percent = (task_counter / total_tasks) * 100
            print(f"\r⏳ 进度: {task_counter}/{total_tasks} ({percent:.1f}%)", end="", flush=True)

# =============================
# 第四部分：主流程
# =============================

def process_single_task_wrapper(task, services, graph_data):
    """封装单个任务处理逻辑"""
    # 复制一份services，防止引用污染
    services_copy = [s.copy() for s in services]
    
    # 调用算法 (静默处理，移除内部print)
    start_time = time.time()
    recom_result, recom_edges = dual_channel_solve(task, services_copy, graph_data)
    elapsed_time = time.time() - start_time

    # 组装结果
    task['recom_result'] = recom_result
    task['recom_edges'] = recom_edges
    task['all_recom_results'] = [] # 保留字段兼容
    task['process_time'] = elapsed_time
    
    return task

def main():
    global processed_tasks, processed_ids, total_tasks, word2vec_model, task_counter

    # 1. 加载模型
    print("🔧 正在加载 Word2Vec 模型...")
    word2vec_model = KeyedVectors.load_word2vec_format(WORD2VEC_MODEL_PATH, binary=True)
    print("✅ 模型加载完成")

    # 2. 加载数据
    print("📂 正在读取数据...")
    all_tasks = read_json_file(INPUT_FILE)
    test_api = read_json_file(API_FILE)
    graph_data = read_json_file(GRAPH_FILE)
    test_api_dict = {api['target_action_reprs']: api for api in test_api}

    # 3. 断点恢复
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

    # 4. 过滤任务
    remaining_tasks = [t for t in all_tasks if t.get("annotation_id", t["confirmed_task"]) not in processed_ids]
    total_tasks = len(all_tasks)
    
    print(f"📊 总任务: {total_tasks} | 剩余: {len(remaining_tasks)}")

    if not remaining_tasks:
        print("🎉 所有任务已处理完成，直接计算指标")
        calculate_and_print_average_metrics(processed_tasks, test_api_dict, graph_data)
        return

    # 5. 开始执行 (单线程循环)
    print("🚀 开始构建 DAG (Stage 3)...")
    
    for i, task in enumerate(remaining_tasks):
        need_save = False
        try:
            result_task = process_single_task_wrapper(task, test_api, graph_data)
            
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
            task_id_short = task.get("annotation_id", task["confirmed_task"][:15])
            print(f"\n❌ 任务 [{task_id_short}] 处理失败: {e}")
            update_progress()

    # 6. 最终保存
    save_progress()

    # 7. 计算最终指标
    calculate_and_print_average_metrics(processed_tasks, test_api_dict, graph_data)
    
    print(f"🏁 全部完成! 结果文件: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
