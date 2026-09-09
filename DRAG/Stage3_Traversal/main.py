import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from topo import (
    extract_chains,
    topo_sort,
    build_dependency_map,
    remove_cycles,
    normalize_edges,
    get_node_text
)
from utils import read_json, write_json


# ===== 路径 =====
INPUT_PATH = "../../Data/mul303/stage2_dag_gema27.json"
OUTPUT_PATH = "../../Data/mul303/topo/stage3_sequence_topo.json"
FAILED_PATH = "../../Data/mul303/topo/unfinished_topo.json"


def process_single_task(task):
    try:
        nodes = task.get("nodes") or []
        edges = task.get("edges") or []

        if not isinstance(nodes, list):
            nodes = []
        if not isinstance(edges, list):
            edges = []
        if not nodes:
            task["chains"] = []
            task["dependency_map"] = {}
            return task

        # ===== 1. 标准化 edges =====
        edges = normalize_edges(edges)

        # ===== 2. 去环 =====
        edges = remove_cycles(nodes, edges)

        # ===== 3. 过滤空 requirement =====
        valid_nodes = []
        valid_ids = set()
        for n in nodes:
            text = get_node_text(n).strip()
            if text:
                valid_nodes.append(n)
                valid_ids.add(n["id"])

        # 同步过滤 edges
        edges = [
            e for e in edges
            if e["from"] in valid_ids and e["to"] in valid_ids
        ]

        if not valid_nodes:
            task["chains"] = []
            task["dependency_map"] = {}
            return task

        # =============================
        # 🔥 【核心修改】直接使用 extract_chains 的返回值
        # 现在 extract_chains 已经返回结构化格式，不需要再转ID了
        # =============================
        chains_structured = extract_chains(valid_nodes, edges)

        # ===== 4. dependency map =====
        dep_map = build_dependency_map(edges)

        # ===== 输出 =====
        task["chains"] = chains_structured  # 直接赋值
        task["dependency_map"] = dep_map

        return task

    except Exception as e:
        raise RuntimeError(f"Stage3 error: {e}")


def process_tasks(tasks):
    processed = []
    failed = []
    max_workers = min(6, os.cpu_count())
    count = 0
    total = len(tasks)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_single_task, t): t
            for t in tasks
        }

        for future in as_completed(futures):
            task = futures[future]
            try:
                result = future.result(timeout=60)
                processed.append(result)
            except Exception as e:
                print(f"fail: {e}, task: {task.get('confirmed_task')}")
                failed.append(task)

            count += 1
            if count % 50 == 0:
                print(f"[Stage3] {count}/{total}")
            if count % 1000 == 0:
                write_json(processed, OUTPUT_PATH)
                print(f"[Checkpoint] saved {count}")

    if failed:
        write_json(failed, FAILED_PATH)
        print(f"saved failed: {len(failed)}")

    return processed


def main():
    tasks = read_json(INPUT_PATH)
    processed = process_tasks(tasks)
    write_json(processed, OUTPUT_PATH)
    print(f"Stage3 finished → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()