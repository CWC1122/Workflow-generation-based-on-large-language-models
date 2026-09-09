import json
from tqdm import tqdm


def calculate_dgd_pair(gt, pred):
    """
    通用DGD计算函数
    gt: 真实值（分母）
    pred: 预测值
    """
    if gt == 0:
        return 0 if pred == 0 else 1.0
    return abs(gt - pred) / gt


def compute_stats(data):
    total = len(data)

    # === 严格命中统计 ===
    equal_node_recom = 0
    equal_node_pred = 0
    equal_pred_recom = 0

    # === 宽松命中统计（±1均算正确）【两条都加上】 ===
    loose_equal_node_recom = 0    # node_num <-> recom_num
    loose_equal_pred_recom = 0   # pred_task_num <-> recom_num  👈 新增这条

    # === DGD累计 ===
    total_dgd_node_recom = 0.0
    total_dgd_node_pred = 0.0
    total_dgd_pred_recom = 0.0

    # === 误差统计 ===
    diff_node_recom = []
    diff_node_pred = []
    diff_pred_recom = []

    for item in tqdm(data):
        node_num = len(item.get("nodes", []))
        recom_num = item.get("recom_num", 0)
        pred_task_num = item.get("pred_task_num", 0)

        # === 严格命中 ===
        if node_num == recom_num:
            equal_node_recom += 1

        if node_num == pred_task_num:
            equal_node_pred += 1

        if pred_task_num == recom_num:
            equal_pred_recom += 1

        # === 宽松命中 1：node_num <-> recom_num ===
        if abs(node_num - recom_num) <= 1:
            loose_equal_node_recom += 1

        # === 宽松命中 2：pred_task_num <-> recom_num  👈 新增 ===
        if abs(pred_task_num - recom_num) <= 1:
            loose_equal_pred_recom += 1

        # === DGD ===
        dgd_node_recom = calculate_dgd_pair(recom_num, node_num)
        dgd_node_pred = calculate_dgd_pair(pred_task_num, node_num)
        dgd_pred_recom = calculate_dgd_pair(recom_num, pred_task_num)

        total_dgd_node_recom += dgd_node_recom
        total_dgd_node_pred += dgd_node_pred
        total_dgd_pred_recom += dgd_pred_recom

        # === 误差 ===
        diff_node_recom.append(abs(node_num - recom_num))
        diff_node_pred.append(abs(node_num - pred_task_num))
        diff_pred_recom.append(abs(pred_task_num - recom_num))

    # === 平均 ===
    avg_dgd_node_recom = total_dgd_node_recom / total if total > 0 else 0
    avg_dgd_node_pred = total_dgd_node_pred / total if total > 0 else 0
    avg_dgd_pred_recom = total_dgd_pred_recom / total if total > 0 else 0

    # 严格准确率
    acc_node_recom = equal_node_recom / total if total > 0 else 0
    acc_node_pred = equal_node_pred / total if total > 0 else 0
    acc_pred_recom = equal_pred_recom / total if total > 0 else 0

    # 两条宽松准确率
    loose_acc_node_recom = loose_equal_node_recom / total if total > 0 else 0
    loose_acc_pred_recom = loose_equal_pred_recom / total if total > 0 else 0

    avg_diff_node_recom = sum(diff_node_recom) / total if total > 0 else 0
    avg_diff_node_pred = sum(diff_node_pred) / total if total > 0 else 0
    avg_diff_pred_recom = sum(diff_pred_recom) / total if total > 0 else 0

    result = {
        "total": total,

        # 严格命中数
        "equal_node_recom_count": equal_node_recom,
        "equal_node_pred_count": equal_node_pred,
        "equal_pred_recom_count": equal_pred_recom,

        # 两条宽松命中数
        "loose_equal_node_recom_count": loose_equal_node_recom,
        "loose_equal_pred_recom_count": loose_equal_pred_recom,

        # 严格准确率
        "acc_node_recom": acc_node_recom,
        "acc_node_pred": acc_node_pred,
        "acc_pred_recom": acc_pred_recom,

        # 两条宽松准确率
        "loose_acc_node_recom": loose_acc_node_recom,
        "loose_acc_pred_recom": loose_acc_pred_recom,

        # DGD
        "avg_dgd_node_recom": avg_dgd_node_recom,
        "avg_dgd_node_pred": avg_dgd_node_pred,
        "avg_dgd_pred_recom": avg_dgd_pred_recom,

        # 平均误差
        "avg_diff_node_recom": avg_diff_node_recom,
        "avg_diff_node_pred": avg_diff_node_pred,
        "avg_diff_pred_recom": avg_diff_pred_recom,
    }

    return result


if __name__ == "__main__":
    TEST_NUM = 9999
    
    file_path = "../../Data/Gemma31b_bge/test/mul/stage2_dag.json"

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    data = data[:TEST_NUM]
    stats = compute_stats(data)

    print("\n===== 统计结果 =====")
    print(f"数据总量: {stats['total']}")

    print("\n--- 严格命中情况 ---")
    print(f"node_num == recom_num: {stats['equal_node_recom_count']}")
    print(f"pred_task_num == recom_num: {stats['equal_pred_recom_count']}")

    print("\n--- 宽松命中情况（±1 均算正确） ---")
    print(f"node_num <-> recom_num 宽松命中: {stats['loose_equal_node_recom_count']}")
    print(f"pred_task_num <-> recom_num 宽松命中: {stats['loose_equal_pred_recom_count']}")  # 👈 显示第二条

    print("\n--- 严格准确率 ---")
    print(f"node_num vs recom_num: {stats['acc_node_recom']:.4f}")
    print(f"pred_task_num vs recom_num: {stats['acc_pred_recom']:.4f}")

    print("\n--- 宽松准确率（±1 均算正确） ---")
    print(f"node_num vs recom_num 宽松准确率: {stats['loose_acc_node_recom']:.4f}")
    print(f"pred_task_num vs recom_num 宽松准确率: {stats['loose_acc_pred_recom']:.4f}")  # 👈 显示第二条

    print("\n--- DGD ---")
    print(f"avg_dgd_node_recom: {stats['avg_dgd_node_recom']:.4f}")
    print(f"avg_dgd_pred_recom: {stats['avg_dgd_pred_recom']:.4f}")

    print("\n--- 平均绝对误差 ---")
    print(f"node_num vs recom_num: {stats['avg_diff_node_recom']:.4f}")
    print(f"pred_task_num vs recom_num: {stats['avg_diff_pred_recom']:.4f}")