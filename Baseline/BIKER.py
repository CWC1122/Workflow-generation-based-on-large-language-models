#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import time
import threading
import numpy as np

from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import math
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

from baseline_runtime import (
    baseline_artifact_path,
    baseline_test_path,
    baseline_train_path,
    get_dataset_name,
    test_data_dir,
    train_data_dir,
)

# ==================== 路径配置 ====================
DATASET_NAME = get_dataset_name()
BASE_TEST_DATA_DIR = test_data_dir(DATASET_NAME)

INPUT_FILE = baseline_test_path("data.json", DATASET_NAME)
OUTPUT_FILE = baseline_test_path("biker_result111.json", DATASET_NAME)
UNFINISHED_FILE = baseline_test_path("biker_unfinished.json", DATASET_NAME)

BASE_TRAIN_DATA_DIR = train_data_dir(DATASET_NAME)

TRAIN_TASKS_PATH = baseline_train_path("data.json", DATASET_NAME)
TRAIN_APIS_PATH = baseline_train_path("tool_desc.json", DATASET_NAME)
TEST_APIS_PATH = baseline_test_path("tool_desc.json", DATASET_NAME)

MODEL_PATH = baseline_artifact_path("biker", "word2vec.model", DATASET_NAME)
IDF_PATH = baseline_artifact_path("biker", "idf.pkl", DATASET_NAME)

print(f"📂 数据目录: {os.path.abspath(BASE_TRAIN_DATA_DIR)}")

# ==================== 全局配置 ====================
SAVE_INTERVAL = 100
MAX_WORKERS = 8  # 建议设为 CPU 核心数
ENABLE_TRAIN = False

# ==================== 全局变量 ====================
processed_tasks = []
processed_ids = set()
unfinished_tasks = []

save_lock = threading.Lock()
task_counter = 0
total_tasks = 0
progress_bar = None

# =============================
# 训练进度条
# =============================
class EpochLogger(CallbackAny2Vec):
    def __init__(self):
        self.epoch = 0

    def on_epoch_begin(self, model):
        print(f"🧠 Epoch {self.epoch} 开始")

    def on_epoch_end(self, model):
        print(f"✅ Epoch {self.epoch} 结束")
        self.epoch += 1

# =============================
# 基础工具函数
# =============================
def read_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_idf(idf_file):
    with open(idf_file, 'rb') as f:
        return pickle.load(f)

def harmonic_mean(sim1, sim2):
    if sim1 + sim2 == 0:
        return 0
    return (2 * sim1 * sim2) / (sim1 + sim2)

# =============================
# 核心优化：NumPy 向量化相似度计算
# =============================
def compute_similarity_vectorized(source_vecs, target_vecs, source_idf_weights):
    """
    向量化计算相似度：source 对 target
    """
    if source_vecs.shape[0] == 0 or target_vecs.shape[0] == 0:
        return 0.0

    # L2 归一化
    source_norm = source_vecs / np.linalg.norm(source_vecs, axis=1, keepdims=True)
    target_norm = target_vecs / np.linalg.norm(target_vecs, axis=1, keepdims=True)
    
    # 矩阵乘法：一次性计算所有词对的余弦相似度 (M, N)
    sim_matrix = np.dot(source_norm, target_norm.T)
    
    # 每行取最大值 (Max-Pooling)
    max_sims = np.max(sim_matrix, axis=1)
    
    # IDF 加权平均
    total_weight = np.sum(source_idf_weights)
    if total_weight == 0:
        return 0.0
    return np.sum(max_sims * source_idf_weights) / total_weight

def compute_harmonic_mean_vectorized(source_vecs, target_vecs, source_idf, target_idf):
    sim1 = compute_similarity_vectorized(source_vecs, target_vecs, source_idf)
    sim2 = compute_similarity_vectorized(target_vecs, source_vecs, target_idf)
    return harmonic_mean(sim1, sim2)

# =============================
# 训练模块
# =============================
def build_corpus_and_train_model(train_tasks, train_apis, test_apis):
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(IDF_PATH), exist_ok=True)

    ps = PorterStemmer()
    corpus = []

    for task in train_tasks:
        tokens = [ps.stem(w.lower()) for w in word_tokenize(task.get('confirmed_task', ''))]
        corpus.append(tokens)

    for api in train_apis + test_apis:
        desc = api.get('target_action_reprs', '') + ' ' + ' '.join(api.get('enhanced_repr', '').split("\n"))
        tokens = [ps.stem(w.lower()) for w in word_tokenize(desc)]
        corpus.append(tokens)

    model = Word2Vec(
        sentences=corpus,
        vector_size=100,
        window=5,
        min_count=1,
        workers=4,
        epochs=5,
        callbacks=[EpochLogger()]
    )

    model.save(MODEL_PATH)

    doc_freq = {}
    for doc in corpus:
        for word in set(doc):
            doc_freq[word] = doc_freq.get(word, 0) + 1

    idf = {w: math.log(len(corpus) / f) for w, f in doc_freq.items()}

    with open(IDF_PATH, 'wb') as f:
        pickle.dump(idf, f)

    return model, idf

# =============================
# 候选 API 查找（已去嵌套线程池）
# =============================
def find_candidate_apis(model_api_group, filtered_apis, model, idf, domain, subdomain):
    ps = PorterStemmer()
    
    # 预处理 Token
    model_api_data = {}
    for api in model_api_group:
        tokens = [ps.stem(w.lower()) for w in word_tokenize(api.get('target_action_reprs',''))]
        valid_tokens = [w for w in tokens if w in model.wv]
        vecs = model.wv[valid_tokens] if valid_tokens else np.zeros((0, 100))
        idfs = np.array([idf.get(w, 0) for w in valid_tokens])
        model_api_data[api['action_uid']] = (vecs, idfs)

    filtered_api_data = {}
    for api in filtered_apis:
        tokens = [ps.stem(w.lower()) for w in word_tokenize(api.get('target_action_reprs',''))]
        valid_tokens = [w for w in tokens if w in model.wv]
        vecs = model.wv[valid_tokens] if valid_tokens else np.zeros((0, 100))
        idfs = np.array([idf.get(w, 0) for w in valid_tokens])
        filtered_api_data[api['action_uid']] = (api, vecs, idfs)

    # 朴素循环查找（比嵌套线程池更稳定）
    candidate_apis = []
    for m_api in model_api_group:
        mid = m_api['action_uid']
        m_vecs, m_idfs = model_api_data[mid]
        
        best_score = -1.0
        best_api = None
        
        for aid, (api, a_vecs, a_idfs) in filtered_api_data.items():
            score = compute_harmonic_mean_vectorized(m_vecs, a_vecs, m_idfs, a_idfs)
            if score > best_score:
                best_score = score
                best_api = api
        
        if best_api:
            candidate_apis.append((best_api, best_score))

    candidate_apis.sort(key=lambda x: x[1], reverse=True)
    return [api for api, _ in candidate_apis[:len(model_api_group)]]

# =============================
# BIKER 核心逻辑
# =============================
def BIKER_try_process_task(task_a, apis, train_tasks, train_apis, model, idf, train_task_vecs_list, train_task_idfs_list):
    ps = PorterStemmer()

    # 1. 处理当前 Task
    task_tokens = [ps.stem(w.lower()) for w in word_tokenize(task_a.get('confirmed_task', ''))]
    valid_task_tokens = [w for w in task_tokens if w in model.wv]
    
    if not valid_task_tokens:
        return [], []
        
    task_vecs = model.wv[valid_task_tokens]
    task_idfs = np.array([idf.get(w, 0) for w in valid_task_tokens])

    # 2. 计算与训练任务的相似度（使用预计算结果）
    similarities = []
    for i in range(len(train_tasks)):
        hm = compute_harmonic_mean_vectorized(
            task_vecs, train_task_vecs_list[i],
            task_idfs, train_task_idfs_list[i]
        )
        similarities.append((train_tasks[i], hm))

    # 3. 取 Top 5
    top_5 = sorted(similarities, key=lambda x: x[1], reverse=True)[:5]

    # 4. 构建 Model API Group
    model_api_group = []
    for t, _ in top_5:
        for api in train_apis:
            if api['action_uid'] in t.get('action_id_list', []):
                model_api_group.append(api)

    avg_num = int(len(model_api_group) / 5) if len(model_api_group) >= 5 else 0
    if avg_num == 0:
        return [], []

    # 5. 查找候选 API（已按要求去掉 domain 筛选）
    filtered = apis
    candidates = find_candidate_apis(model_api_group, filtered, model, idf, None, None)

    # 6. 最终排序
    scored = []
    for api in candidates:
        api_tokens = [ps.stem(w.lower()) for w in word_tokenize(api.get('target_action_reprs',''))]
        valid_api_tokens = [w for w in api_tokens if w in model.wv]
        
        if not valid_api_tokens:
            continue
            
        api_vecs = model.wv[valid_api_tokens]
        api_idfs = np.array([idf.get(w, 0) for w in valid_api_tokens])
        
        hm = compute_harmonic_mean_vectorized(task_vecs, api_vecs, task_idfs, api_idfs)
        scored.append((api, hm))

    scored.sort(key=lambda x: x[1], reverse=True)
    return [a for a, _ in scored[:avg_num]], []

# =============================
# 工程化：保存进度
# =============================
def save_progress():
    with save_lock:
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(processed_tasks, f, indent=4, ensure_ascii=False)

        with open(UNFINISHED_FILE, "w", encoding="utf-8") as f:
            json.dump(unfinished_tasks, f, indent=4, ensure_ascii=False)

def update_progress():
    global task_counter, progress_bar
    with save_lock:
        task_counter += 1
        if progress_bar:
            progress_bar.update(1)

# =============================
# 主流程
# =============================
def main():
    global total_tasks, processed_tasks, processed_ids, progress_bar

    print("📂 加载数据...")
    tasks = read_json(INPUT_FILE)
    train_tasks = read_json(TRAIN_TASKS_PATH)
    train_apis = read_json(TRAIN_APIS_PATH)
    test_apis = read_json(TEST_APIS_PATH)

    total_tasks = len(tasks)

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(IDF_PATH), exist_ok=True)

    if ENABLE_TRAIN or not os.path.exists(MODEL_PATH):
        print("🧠 训练 Word2Vec...")
        model, idf = build_corpus_and_train_model(train_tasks, train_apis, test_apis)
    else:
        print("⚡ 加载已有模型...")
        model = Word2Vec.load(MODEL_PATH)
        idf = load_idf(IDF_PATH)

    # =============================
    # 【核心优化】预计算所有训练任务的向量
    # =============================
    print("⚡ 预计算训练任务向量 (仅需运行一次)...")
    ps = PorterStemmer()
    train_task_vecs_list = []
    train_task_idfs_list = []
    
    for t in tqdm(train_tasks, desc="预计算进度"):
        tokens = [ps.stem(w.lower()) for w in word_tokenize(t.get('confirmed_task', ''))]
        valid_tokens = [w for w in tokens if w in model.wv]
        
        if not valid_tokens:
            train_task_vecs_list.append(np.zeros((0, 100)))
            train_task_idfs_list.append(np.zeros(0))
            continue
            
        vecs = model.wv[valid_tokens]
        idfs = np.array([idf.get(w, 0) for w in valid_tokens])
        
        train_task_vecs_list.append(vecs)
        train_task_idfs_list.append(idfs)

    # 断点续存
    if os.path.exists(OUTPUT_FILE):
        processed_tasks = read_json(OUTPUT_FILE)
        processed_ids = set(t.get("annotation_id", i) for i, t in enumerate(processed_tasks))

    remaining = [t for i, t in enumerate(tasks) if i not in processed_ids]
    print(f"📊 总任务: {total_tasks} | 剩余: {len(remaining)}")

    progress_bar = tqdm(total=len(remaining), desc="🚀 处理任务进度")

    # =============================
    # 多进程并行处理
    # =============================
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {}
        for idx, task in enumerate(remaining):
            futures[executor.submit(
                BIKER_try_process_task,
                task,
                test_apis,
                train_tasks,
                train_apis,
                model,
                idf,
                train_task_vecs_list,
                train_task_idfs_list
            )] = (idx, task)

        for future in as_completed(futures):
            idx, task = futures[future]
            try:
                result, edges = future.result()
                task['recom_result'] = result
                task['recom_edges'] = edges
                processed_tasks.append(task)
            except Exception as e:
                task['error'] = str(e)
                unfinished_tasks.append(task)

            update_progress()
            if len(processed_tasks) % SAVE_INTERVAL == 0:
                save_progress()

    progress_bar.close()
    save_progress()
    print("\n🎉 全部完成")

if __name__ == "__main__":
    main()
    
