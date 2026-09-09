#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2026-03-30
# @Author : wenchao
# @Desc : 任务评估指标计算脚本（ER + ER-soft + softOHR）

import json
import os
import numpy as np

# ==================== 路径配置 ====================
# BASE_DATA_DIR = "../../Data_no_self/Llama3.1:8b-bge/mul"
BASE_DATA_DIR = "../../Data/Gemma31b_bge/test/hug"

INPUT_TEST_TASK = os.path.join(BASE_DATA_DIR, "stage7_final_weight_sorted.json")
INPUT_TEST_API = os.path.join(BASE_DATA_DIR, "tool_desc.json")
INPUT_GRAPH = os.path.join(BASE_DATA_DIR, "graph_desc.json")
OUTPUT_RESULT = os.path.join(BASE_DATA_DIR, "evaluate_result_weight_sorted.json")

print(f"📂 数据基础目录: {os.path.abspath(BASE_DATA_DIR)}")
TEST_NUM = 9999  # 这里修改数字，比如50、200、1000

# =============================
# 工具函数
# =============================
def read_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


# =============================
# Precision / Recall
# =============================
def calculate_task_metrics(recom_result, actual_apis_reprs):
    recom_true_positives = 0
    actual_true_positives = 0
    total_true = 0

    recom_result_reprs = set()

    for api in recom_result:
        if api is not None and 'target_action_reprs' in api:
            recom_result_reprs.add(api['target_action_reprs'])

    for api_repr in recom_result_reprs:
        if api_repr in actual_apis_reprs:
            recom_true_positives += 1

    for api_repr in actual_apis_reprs:
        if api_repr in recom_result_reprs:
            actual_true_positives += 1

    precision = recom_true_positives / len(recom_result) if len(recom_result) > 0 else 0
    recall = actual_true_positives / len(actual_apis_reprs) if len(actual_apis_reprs) > 0 else 0

    if precision == 1.0 and recall == 1.0:
        total_true = 1.0

    return precision, recall, total_true


def calculate_highest_metrics(all_recom_results, actual_apis_reprs):
    highest_precision = 0.0
    highest_recall = 0.0

    for api_list in all_recom_results:
        current_precision, current_recall, _ = calculate_task_metrics(api_list, actual_apis_reprs)
        highest_precision = max(highest_precision, current_precision)
        highest_recall = max(highest_recall, current_recall)

    return highest_precision, highest_recall


# =============================
# Granularity
# =============================
def calculate_task_granularity_deviation(recom_result, actual_steps):
    if len(actual_steps) == 0:
        return 0
    return abs(len(recom_result) - len(actual_steps)) / len(actual_steps)


# =============================
# 构建邻接表
# =============================
def build_adjacency(graph_data):
    adjacency_list = {}
    for link in graph_data:
        s = link['source']
        t = link['target']
        if s not in adjacency_list:
            adjacency_list[s] = set()
        adjacency_list[s].add(t)
    return adjacency_list


# =============================
# 核心：计算单任务的严格ER（返回原始得分 + 是否有效）
# =============================
def _calculate_er_core(action_reprs, test_api_dict, adjacency_list):
    """
    内部辅助函数：计算ER的核心逻辑
    返回：(er_score, is_valid_for_real_er)
    """
    num_pairs = len(action_reprs) - 1

    # 情况1：单节点或空
    if num_pairs <= 0:
        # All_ER 得 1.0，但 Real_ER 不算有效样本
        return 1.0, False

    # 情况2：多节点，正常计算
    executable_pairs = 0
    for i in range(num_pairs):
        cur = test_api_dict.get(action_reprs[i])
        nxt = test_api_dict.get(action_reprs[i + 1])

        if not cur or not nxt:
            continue

        cur_id = cur.get('action_uid')
        nxt_id = nxt.get('action_uid')

        if cur_id in adjacency_list and nxt_id in adjacency_list[cur_id]:
            executable_pairs += 1

    score = executable_pairs / num_pairs
    return score, True


# =============================
# ER-soft
# =============================
def calculate_executability_soft(task, test_api_dict, adjacency_list):
    action_reprs = [s['target_action_reprs'] for s in task.get('recom_result', [])]

    n = len(action_reprs)
    if n <= 1:
        return 1.0

    satisfied = 0
    total = n - 1

    for i in range(1, n):
        cur = test_api_dict.get(action_reprs[i])
        if not cur:
            continue

        cur_id = cur.get('action_uid')
        ok = False

        for j in range(i):
            prev = test_api_dict.get(action_reprs[j])
            if not prev:
                continue

            prev_id = prev.get('action_uid')

            if prev_id in adjacency_list and cur_id in adjacency_list[prev_id]:
                ok = True
                break

        if ok:
            satisfied += 1

    return satisfied / total if total > 0 else 0.0


# =============================
# ⭐ softOHR（新增）
# =============================
def calculate_soft_ohr(task, test_api_dict, adjacency_list):
    recom_result = task.get('recom_result', [])
    pred_reprs = [s['target_action_reprs'] for s in recom_result]
    gt_reprs = set(task.get('action_reprs', []))

    # 条件1：pred ⊆ gt
    if not set(pred_reprs).issubset(gt_reprs):
        return 0.0

    # 条件2：满足 ER-soft（必须完全可执行）
    er_soft = calculate_executability_soft(task, test_api_dict, adjacency_list)

    return 1.0 if er_soft == 1.0 else 0.0


# =============================
# 总指标
# =============================
def calculate_average_metrics(test_task, test_api_dict, graph_data):
    adjacency_list = build_adjacency(graph_data)

    total_precision = 0
    total_recall = 0
    total_highest_precision = 0
    total_highest_recall = 0
    total_granularity_deviation = 0
    total_loose_granularity_acc = 0
    total_executability_soft = 0
    total_whole_precision = 0
    total_soft_ohr = 0

    # 🆕 新增：Real_ER 和 All_ER 的专用计数器
    total_all_er = 0.0       # All_ER 累加和
    total_real_er = 0.0      # Real_ER 累加和
    num_real_er_tasks = 0    # Real_ER 的有效分母数（节点数>=2）

    num_tasks = len(test_task)

    for task in test_task:
        recom_result = task.get('recom_result', [])
        all_recom_results = task.get('all_recom_results', [])
        actual_apis_reprs = task.get('action_reprs', [])
        
        # 🆕 先提取 action_reprs 供 ER 使用
        action_reprs = [s['target_action_reprs'] for s in recom_result]

        precision, recall, whole_precision = calculate_task_metrics(recom_result, actual_apis_reprs)

        total_precision += precision
        total_recall += recall
        total_whole_precision += whole_precision

        hp, hr = calculate_highest_metrics(all_recom_results, actual_apis_reprs)
        total_highest_precision += hp
        total_highest_recall += hr

        gd = calculate_task_granularity_deviation(recom_result, actual_apis_reprs)
        total_granularity_deviation += gd

        # ✅ 宽松粒度准确率
        pred_steps = len(recom_result)
        gt_steps = len(actual_apis_reprs)
        if abs(pred_steps - gt_steps) <= 1:
            total_loose_granularity_acc += 1

        # 🆕 核心改动：计算 Real_ER 和 All_ER
        er_score, is_real_valid = _calculate_er_core(action_reprs, test_api_dict, adjacency_list)
        
        # All_ER：所有任务都累加
        total_all_er += er_score
        
        # Real_ER：只有多节点任务才累加且计数
        if is_real_valid:
            total_real_er += er_score
            num_real_er_tasks += 1

        er_soft = calculate_executability_soft(task, test_api_dict, adjacency_list)
        soft_ohr = calculate_soft_ohr(task, test_api_dict, adjacency_list)

        total_executability_soft += er_soft
        total_soft_ohr += soft_ohr

    # 🆕 计算最终平均值
    avg_all_er = total_all_er / num_tasks if num_tasks > 0 else 0.0
    avg_real_er = total_real_er / num_real_er_tasks if num_real_er_tasks > 0 else 0.0

    return (
        total_precision / num_tasks,
        total_recall / num_tasks,
        total_highest_precision / num_tasks,
        total_highest_recall / num_tasks,
        total_granularity_deviation / num_tasks,
        total_loose_granularity_acc / num_tasks,
        avg_real_er,        # 🆕 Real_ER
        avg_all_er,         # 🆕 All_ER
        total_executability_soft / num_tasks,
        total_whole_precision / num_tasks,
        total_soft_ohr / num_tasks,
        num_tasks,
        num_real_er_tasks   # 🆕 顺便返回 Real_ER 的有效样本数，方便打印
    )


# =============================
# 清理数据
# =============================
def clean_data(data):
    if isinstance(data, dict):
        return {k: clean_data(v) for k, v in data.items() if k != "vector"}
    elif isinstance(data, list):
        return [clean_data(i) for i in data]
    elif isinstance(data, np.ndarray):
        return data.tolist()
    else:
        return data


# =============================
# 主函数
# =============================
def main():
    test_task = read_json_file(INPUT_TEST_TASK)
    test_api = read_json_file(INPUT_TEST_API)
    graph_data = read_json_file(INPUT_GRAPH)

    test_api_dict = {api['target_action_reprs']: api for api in test_api}

    (
        avg_p, avg_r,
        avg_hp, avg_hr,
        avg_gd,
        avg_lga,
        avg_real_er,      # 🆕
        avg_all_er,       # 🆕
        avg_er_soft,
        avg_wp,
        avg_soft_ohr,
        total_count,
        real_er_count     # 🆕
    ) = calculate_average_metrics(test_task, test_api_dict, graph_data)

    print(f"📊 数据总条数: {total_count}")
    print(f"平均精确率: {avg_p:.4f}")
    print(f"平均召回率: {avg_r:.4f}")
    print(f"平均分解粒度偏差: {avg_gd:.4f}")
    print(f"宽松粒度准确率（±1步）: {avg_lga:.4f}")
    print("-" * 40)
    print(f"🆕 All_ER (含单节点，单节点算1.0): {avg_all_er:.4f}")
    print(f"🆕 Real_ER (仅节点数≥2, 样本数: {real_er_count}): {avg_real_er:.4f}")
    print("-" * 40)
    print(f"ER-soft: {avg_er_soft:.4f}")
    print(f"整体成功率（严格OHR）: {avg_wp:.4f}")
    print(f"softOHR（图约束+子集匹配）: {avg_soft_ohr:.4f}")

    with open(OUTPUT_RESULT, 'w', encoding='utf-8') as f:
        json.dump(clean_data(test_task), f, indent=4, ensure_ascii=False)

    print(f"结果已保存至 {OUTPUT_RESULT}")


if __name__ == "__main__":
    main()