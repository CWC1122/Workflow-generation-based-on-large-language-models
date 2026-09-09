from collections import deque


def extract_chains(nodes, edges):
    """
    【修改版】从 DAG 中提取多个 chain
    输出格式：和 chain_builder 完全一致的结构化列表
    返回：
        List[List[dict]] -> [{"id": 1, "requirement": "...", "index": 0}, ...]
    """

    graph, in_degree = build_graph(nodes, edges)

    # 构建 node_id -> node_info 的映射，方便快速查找
    node_map = {n["id"]: n for n in nodes if "id" in n}

    visited = set()
    chains = []

    # 找所有入度为0的节点（chain起点）
    start_nodes = sorted([n for n in in_degree if in_degree[n] == 0])

    for start in start_nodes:

        if start in visited:
            continue

        chain = []
        cur = start

        while True:

            if cur in visited:
                break

            visited.add(cur)
            
            # 🔥 【核心修改】把纯ID转换成结构化字典
            node_info = node_map.get(cur, {})
            chain.append({
                "id": cur,
                "requirement": node_info.get("requirement", ""),
                "index": len(chain)  # 链内索引
            })

            # 如果没有后继，结束
            if cur not in graph or not graph[cur]:
                break

            # 如果有多个分支，只取第一个（保证稳定）
            next_nodes = sorted(graph[cur])
            nxt = next_nodes[0]

            # 如果下一个已经访问过，停止
            if nxt in visited:
                break

            cur = nxt

        if chain:
            chains.append(chain)

    # ===== 处理剩余未访问节点（防 disconnected 或环残留）=====
    for nid in graph:
        if nid not in visited:
            node_info = node_map.get(nid, {})
            chains.append([{
                "id": nid,
                "requirement": node_info.get("requirement", ""),
                "index": 0
            }])

    # ===== 按 chain 起点排序 =====
    chains.sort(key=lambda c: c[0]["id"])

    return chains


def normalize_edges(edges):
    """
    统一 edge 格式
    """
    if not edges:
        return []

    new_edges = []

    for e in edges:

        if not isinstance(e, dict):
            continue

        u = e.get("from") or e.get("source")
        v = e.get("to") or e.get("target")

        if u is None or v is None:
            continue

        if u == v:
            continue

        new_edges.append({
            "from": int(u),
            "to": int(v),
            "type": e.get("type", "text")
        })

    return new_edges


def build_graph(nodes, edges):
    """
    不使用 defaultdict，避免动态修改
    """
    node_ids = [n.get("id") for n in nodes if "id" in n]

    graph = {nid: [] for nid in node_ids}
    in_degree = {nid: 0 for nid in node_ids}

    for e in edges:
        u, v = e["from"], e["to"]

        if u in graph and v in graph:
            graph[u].append(v)
            in_degree[v] += 1

    return graph, in_degree


def is_dag(graph):
    """
    检测是否有环（Kahn）
    """
    in_degree = {k: 0 for k in graph}

    for u in graph:
        for v in graph[u]:
            in_degree[v] += 1

    queue = deque([n for n in in_degree if in_degree[n] == 0])
    count = 0

    while queue:
        cur = queue.popleft()
        count += 1

        for nxt in graph[cur]:
            in_degree[nxt] -= 1
            if in_degree[nxt] == 0:
                queue.append(nxt)

    return count == len(graph)


def remove_cycles(nodes, edges):
    """
    贪心去环（安全版本）
    """
    clean = []

    for e in edges:
        temp = clean + [e]
        graph, _ = build_graph(nodes, temp)

        if is_dag(graph):
            clean.append(e)

    return clean


def topo_sort(nodes, edges):
    """
    稳定拓扑排序
    """
    graph, in_degree = build_graph(nodes, edges)

    queue = deque(sorted([n for n in in_degree if in_degree[n] == 0]))
    order = []

    while queue:
        cur = queue.popleft()
        order.append(cur)

        for nxt in graph.get(cur, []):
            in_degree[nxt] -= 1
            if in_degree[nxt] == 0:
                queue.append(nxt)

    # 防 disconnected
    if len(order) < len(nodes):
        remain = set(in_degree.keys()) - set(order)
        order.extend(sorted(remain))

    return order


def build_dependency_map(edges):
    """
    构建依赖映射
    """
    dep_map = {}

    for e in edges:
        to = str(e["to"])

        if to not in dep_map:
            dep_map[to] = []

        dep_map[to].append({
            "from": e["from"],
            "type": e["type"]
        })

    return dep_map


def get_node_text(node):
    return node.get("requirement") or node.get("task") or ""