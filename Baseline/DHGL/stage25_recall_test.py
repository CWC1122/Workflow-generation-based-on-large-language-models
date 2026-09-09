#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import re
import os
import sys
import numpy as np
from gensim.models import KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from baseline_runtime import (
    baseline_test_path,
    baseline_train_path,
    get_baseline_dataset_list,
    get_word2vec_binary_path,
)

# ==================== 路径配置区域 ====================
# 输入输出文件名
INPUT_FILENAME = "stage2_recall.json"
TRAIN_FILENAME = "data.json"
OUTPUT_FILENAME = "stage25_get_num_test_auto1.json"
# ==================== 路径配置结束 ====================

SAVE_INTERVAL = 50
TOP_K_SIMILAR = 5
EXCLUDE_SELF = True
DATASETS = get_baseline_dataset_list(["mul", "hug", "daily", "ultratool"])
WORD2VEC_MODEL_PATH = get_word2vec_binary_path()

word2vec_model = None


# =============================
# 工具函数
# =============================
def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r"[^\w\s]", "", text)
    return " ".join(text.split())


def string_to_vector(text, model):
    words = preprocess_text(text).split()
    vectors = [model[word] for word in words if word in model]
    return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)


def read_json_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json_file(file_path, data):
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def precompute_train_vectors(train_tasks, model):
    """预计算训练集向量，只执行一次"""
    train_texts = [t["confirmed_task"] for t in train_tasks]
    train_vectors = [string_to_vector(text, model) for text in train_texts]
    return np.array(train_vectors)


def process_single_task_fast(task, train_vectors, train_tasks, model):
    """对单条任务计算 recom_num"""
    q_text = task["confirmed_task"]
    q_vec = string_to_vector(q_text, model).reshape(1, -1)

    similarities = cosine_similarity(q_vec, train_vectors)[0]

    # 是否排除自身
    if EXCLUDE_SELF:
        task_id = task.get("annotation_id", None)
        if task_id is not None:
            for i, train_task in enumerate(train_tasks):
                train_id = train_task.get("annotation_id", None)
                if train_id == task_id:
                    similarities[i] = -1
                    break

    # TopK
    top_k_idx = np.argsort(similarities)[::-1][:TOP_K_SIMILAR]

    total_num = 0
    actual_k = len(top_k_idx)
    if actual_k == 0:
        task["recom_num"] = 0
        return task

    for idx in top_k_idx:
        total_num += len(train_tasks[idx].get("action_id_list", []))

    task["recom_num"] = int(total_num / actual_k)
    return task


# =============================
# 单个数据集处理
# =============================
def process_dataset(dataset, model):
    input_file = baseline_test_path(INPUT_FILENAME, dataset)
    train_file = baseline_train_path(TRAIN_FILENAME, dataset)
    output_file = baseline_test_path(OUTPUT_FILENAME, dataset)

    print("\n" + "=" * 80)
    print(f"🚀 开始处理数据集: {dataset}")
    print(f"📥 输入文件: {input_file}")
    print(f"📚 训练文件: {train_file}")
    print(f"💾 输出文件: {output_file}")

    if not os.path.exists(input_file):
        print(f"❌ 跳过 {dataset}：未找到输入文件")
        return

    if not os.path.exists(train_file):
        print(f"❌ 跳过 {dataset}：未找到训练文件")
        return

    all_tasks = read_json_file(input_file)
    train_tasks = read_json_file(train_file)

    if not isinstance(all_tasks, list):
        raise ValueError(f"{input_file} 顶层不是 list")
    if not isinstance(train_tasks, list):
        raise ValueError(f"{train_file} 顶层不是 list")

    # 训练集向量预计算
    print("🔢 预计算训练集向量...")
    train_vectors = precompute_train_vectors(train_tasks, model)
    print(f"✅ 训练集向量预计算完成，共 {len(train_vectors)} 条")

    # 断点续跑：如果输出文件已存在，则从输出文件恢复
    processed_tasks = []
    processed_ids = set()
    task_counter = 0

    if os.path.exists(output_file):
        processed_tasks = read_json_file(output_file)
        if not isinstance(processed_tasks, list):
            raise ValueError(f"{output_file} 顶层不是 list")

        processed_ids = set(
            t.get("annotation_id", t["confirmed_task"])
            for t in processed_tasks
            if isinstance(t, dict)
        )
        task_counter = len(processed_tasks)
        print(f"✅ 已恢复 {task_counter} 条记录")
    else:
        print("🆕 未发现历史输出，将从头开始处理")

    total_tasks = len(all_tasks)
    remaining_tasks = [
        t for t in all_tasks
        if isinstance(t, dict) and t.get("annotation_id", t["confirmed_task"]) not in processed_ids
    ]

    print(f"📊 总任务: {total_tasks} | 剩余待处理: {len(remaining_tasks)}")

    if not remaining_tasks:
        print(f"🎉 {dataset} 所有任务已完成")
        return

    print("🚀 开始计算 recom_num...")
    for task in remaining_tasks:
        try:
            result_task = process_single_task_fast(task, train_vectors, train_tasks, model)
            processed_tasks.append(result_task)
            processed_ids.add(result_task.get("annotation_id", result_task["confirmed_task"]))
            task_counter += 1

            if task_counter % 10 == 0 or task_counter == total_tasks:
                percent = (task_counter / total_tasks) * 100
                print(f"\r⏳ {dataset} 进度: {task_counter}/{total_tasks} ({percent:.1f}%)", end="", flush=True)

            if len(processed_tasks) % SAVE_INTERVAL == 0:
                save_json_file(output_file, processed_tasks)
                print(f"\n💾 中途保存: {output_file}")

        except Exception as e:
            print(f"\n❌ {dataset} 某条任务处理失败: {e}")
            task_counter += 1

    save_json_file(output_file, processed_tasks)
    print(f"\n✅ {dataset} 处理完成，结果已保存到: {output_file}")


# =============================
# 主函数：依次跑四个数据集
# =============================
def main():
    global word2vec_model

    print("🔧 正在加载 Word2Vec 模型...")
    word2vec_model = KeyedVectors.load_word2vec_format(WORD2VEC_MODEL_PATH, binary=True)
    print("✅ 模型加载完成")

    for dataset in DATASETS:
        try:
            process_dataset(dataset, word2vec_model)
        except Exception as e:
            print(f"\n❌ 数据集 {dataset} 处理失败: {e}")

    print("\n🏁 所选数据集已全部跑完！")


if __name__ == "__main__":
    main()
