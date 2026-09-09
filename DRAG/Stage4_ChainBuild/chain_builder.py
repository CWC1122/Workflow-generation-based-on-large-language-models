"""
语义链构建器模块

该模块实现了基于DAG（有向无环图）的语义链构建系统，用于将复杂的任务图
分解为多条线性执行链，每条链代表一个可以顺序执行的子任务序列。

主要功能：
1. 计算DAG节点的语义权重（基于任务相似度、服务相似度、结构重要性）
2. 使用动态规划算法从DAG中提取最优执行链
3. 对多条链进行拓扑排序，确保链之间的依赖关系正确

核心算法：
- 节点权重计算：综合考虑用户查询相似度、服务匹配度和图结构重要性
- 链提取：基于动态规划的最长路径算法，每次提取权重最大的路径
- 链排序：基于链间依赖关系的拓扑排序

典型使用流程：
    embedder = SomeEmbedder()
    service_dict = {"service_1": {...}, "service_2": {...}}
    builder = SemanticChainBuilder(embedder, service_dict)
    
    # 计算节点权重
    weights = builder.compute_node_weights(nodes, edges, user_query, service_ids)
    
    # 构建链
    chains = builder.build_chains(nodes, edges, weights)
    
    # 对链排序
    sorted_chains = builder.sort_chains(chains, edges)
"""

import numpy as np


def cosine_sim(a, b):
    """
    计算两个向量的余弦相似度
    
    余弦相似度衡量两个向量在方向上的相似程度，值域为 [-1, 1]。
    值越接近1表示两个向量越相似，越接近-1表示越相反。
    
    Args:
        a: 向量a，可以是列表或numpy数组
        b: 向量b，可以是列表或numpy数组
    
    Returns:
        float: 余弦相似度值，范围 [-1, 1]
    
    Note:
        - 分母添加了 1e-8 的小常数，防止除零错误
        - 输入会自动转换为numpy数组进行计算
    
    Example:
        >>> cosine_sim([1, 0, 0], [1, 0, 0])
        1.0
        >>> cosine_sim([1, 0, 0], [0, 1, 0])
        0.0
    """
    a = np.array(a)
    b = np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


class SemanticChainBuilder:
    """
    语义链构建器
    
    该类负责将DAG形式的任务图分解为多条线性执行链。每条链代表一个
    可以顺序执行的子任务序列，链的构建基于节点的语义权重。
    
    权重计算公式：
        weight = alpha * sim_task + beta * sim_service + gamma * structure
    
    其中：
        - sim_task: 节点需求与用户查询的语义相似度
        - sim_service: 节点需求与目标服务的最大语义相似度
        - structure: 节点的结构重要性（归一化的度数）
        - alpha, beta, gamma: 各部分的权重系数
    
    Attributes:
        embedder: 嵌入向量客户端，需实现 get_embedding(text) 方法
        service_dict: 服务字典，服务ID到服务信息的映射
        alpha: 任务相似度权重系数，默认0.7
        beta: 服务相似度权重系数，默认0.2
        gamma: 结构重要性权重系数，默认0.1
    
    Example:
        >>> builder = SemanticChainBuilder(embedder, service_dict, alpha=0.7, beta=0.2, gamma=0.1)
        >>> weights = builder.compute_node_weights(nodes, edges, "查询天气", ["weather_service"])
        >>> chains = builder.build_chains(nodes, edges, weights)
    """

    def __init__(self, embedder, service_dict,
                 alpha=0.7, beta=0.2, gamma=0.1):
        """
        初始化语义链构建器
        
        Args:
            embedder: 嵌入向量客户端，用于获取文本的向量表示
                      需实现 get_embedding(text) -> np.ndarray 方法
            service_dict: 服务字典，键为服务ID，值为服务信息字典
                          服务信息需包含 target_action_reprs 字段（服务描述文本）
            alpha: 任务相似度权重，默认0.7
                   控制节点需求与用户查询相似度对权重的影响
            beta: 服务相似度权重，默认0.2
                  控制节点需求与目标服务相似度对权重的影响
            gamma: 结构权重，默认0.1
                   控制节点在图中结构重要性对权重的影响
        
        Note:
            权重系数应满足 alpha + beta + gamma = 1，以确保权重归一化
        """
        self.embedder = embedder
        self.service_dict = service_dict

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def compute_node_weights(self, nodes, edges, user_query, service_ids):
        """
        计算DAG图中每个节点的权重
        
        节点权重由三部分组成：
        1. 任务相似度：节点需求与用户查询的语义相似度
        2. 服务相似度：节点需求与目标服务的最大语义相似度
        3. 结构重要性：节点在图中的度数（归一化）
        
        Args:
            nodes: 节点列表，每个节点是字典，包含：
                - id: 节点唯一标识
                - requirement: 节点的需求描述文本
            edges: 边列表，每条边是字典，包含：
                - source: 源节点ID
                - target: 目标节点ID
            user_query: 用户查询文本，用于计算任务相似度
            service_ids: 目标服务ID列表，用于计算服务相似度
        
        Returns:
            dict: 节点ID到权重值的映射，权重值范围约为 [0, 1]
        
        Note:
            - 如果节点没有需求文本，权重设为0
            - 结构重要性通过度数归一化计算，避免度数差异过大
            - 服务相似度取与所有目标服务的最大值
        
        Example:
            >>> nodes = [{"id": "n1", "requirement": "获取天气数据"}]
            >>> edges = []
            >>> weights = builder.compute_node_weights(nodes, edges, "查询天气", ["weather_api"])
            >>> # weights = {"n1": 0.85}
        """
        node_map = {n["id"]: n for n in nodes}

        # 计算用户查询的嵌入向量
        user_emb = self.embedder.get_embedding(user_query)

        # 计算所有目标服务的嵌入向量列表
        service_embs = []
        for sid in service_ids:
            s = self.service_dict.get(sid)
            if s:
                service_embs.append(
                    self.embedder.get_embedding(s["target_action_reprs"])
                )

        # 计算每个节点的入度和出度
        in_deg = {n["id"]: 0 for n in nodes}
        out_deg = {n["id"]: 0 for n in nodes}

        for e in edges:
            u = e["source"]
            v = e["target"]
            out_deg[u] += 1
            in_deg[v] += 1

        # 结构权重归一化准备
        # 找到所有节点中的最大度数，用于归一化
        max_degree = 0
        for nid in node_map:
            total_degree = in_deg[nid] + out_deg[nid]
            if total_degree > max_degree:
                max_degree = total_degree

        # 防除零保护：若所有节点都是孤立节点（度数全为0），则将max_degree设为1
        max_degree = max(max_degree, 1)

        weights = {}

        for nid, node in node_map.items():
            text = node.get("requirement", "")

            # 如果节点没有需求文本，权重设为0
            if not text:
                weights[nid] = 0
                continue

            # 获取节点需求文本的嵌入向量
            emb = self.embedder.get_embedding(text)

            # 1. 计算任务相似度：节点需求与用户查询的语义相似度
            sim_task = cosine_sim(emb, user_emb)

            # 2. 计算服务相似度：节点需求与所有目标服务的最大相似度
            sim_service = 0
            if service_embs:
                sim_service = max(cosine_sim(emb, s) for s in service_embs)

            # 3. 计算结构重要性：节点的归一化度数
            # 度数越高表示节点在图中越重要（连接更多节点）
            raw_structure = in_deg[nid] + out_deg[nid]
            structure = raw_structure / max_degree

            # 综合计算节点权重
            w = (
                self.alpha * sim_task +
                self.beta * sim_service +
                self.gamma * structure
            )

            weights[nid] = w

        return weights

    def build_chains(self, nodes, edges, weights):
        """
        【修复BUG版】全局加权最长路径优先 (W-LPF)
        完美支持多分支汇聚DAG，永不空链，保留全局最优
        """
        # 1. 构建图结构
        graph = {n["id"]: [] for n in nodes}
        reverse_graph = {n["id"]: [] for n in nodes}
        
        for e in edges:
            u = e["source"]
            v = e["target"]
            graph[u].append(v)
            reverse_graph[v].append(u)

        remaining_nodes = set(n["id"] for n in nodes)
        chains = []

        while remaining_nodes:
            # =============================
            # 修复1：简化子图拓扑排序（无冗余计算）
            # =============================
            # 构建当前剩余节点的邻接表
            sub_graph = {n: [] for n in remaining_nodes}
            for n in remaining_nodes:
                sub_graph[n] = [v for v in graph[n] if v in remaining_nodes]
            
            # 拓扑排序
            in_degree = {n: 0 for n in remaining_nodes}
            for n in remaining_nodes:
                for neighbor in sub_graph[n]:
                    in_degree[neighbor] += 1
            
            topo_order = []
            queue = [n for n in remaining_nodes if in_degree[n] == 0]
            temp_in = in_degree.copy()
            
            while queue:
                u = queue.pop(0)
                topo_order.append(u)
                for v in sub_graph[u]:
                    temp_in[v] -= 1
                    if temp_in[v] == 0:
                        queue.append(v)

            # 兜底：拓扑为空直接取权重最大节点
            if not topo_order:
                best = max(remaining_nodes, key=lambda x: weights.get(x, 0))
                chains.append([best])
                remaining_nodes.remove(best)
                continue

            # =============================
            # 修复2：DP初始化全节点赋初值（核心修复！）
            # =============================
            # 所有节点初始化为自身权重，不再只给入度0赋值
            dp = {nid: weights.get(nid, 0) for nid in remaining_nodes}
            prev = {nid: None for nid in remaining_nodes}
            max_weight = -float("inf")
            end_node = None

            # =============================
            # 修复3：DP按拓扑序更新（稳定支持汇聚节点）
            # =============================
            for u in topo_order:
                # 更新最大权重终点
                if dp[u] > max_weight:
                    max_weight = dp[u]
                    end_node = u
                # 遍历后继节点
                for v in sub_graph[u]:
                    if dp[v] < dp[u] + weights.get(v, 0):
                        dp[v] = dp[u] + weights.get(v, 0)
                        prev[v] = u

            # 兜底防护
            if end_node is None or end_node not in remaining_nodes:
                best = max(remaining_nodes, key=lambda x: weights.get(x, 0))
                chains.append([best])
                remaining_nodes.remove(best)
                continue

            # =============================
            # 回溯路径
            # =============================
            current_chain = []
            cur = end_node
            while cur is not None:
                current_chain.append(cur)
                cur = prev.get(cur)
            current_chain.reverse()

            # 更新剩余节点
            chains.append(current_chain)
            remaining_nodes -= set(current_chain)

        return chains
    def sort_chains(self, chains, weights):
        """
        按链重要性排序（基于节点权重）
        链权重 = 链中所有节点权重之和
        """
        if not chains:
            return []

        def chain_score(chain):
            return sum(weights.get(n, 0) for n in chain)

        # 按权重降序排序
        sorted_chains = sorted(
            chains,
            key=lambda c: chain_score(c),
            reverse=True
        )

        return sorted_chains        
    # def sort_chains(self, chains, edges):
    #     """
    #     【自动破环版】链拓扑排序（支持环场景，永不空链）
    #     新增逻辑：若所有链入度>0（环），全局入度-1，直到出现入度0的链
    #     """
    #     if not chains:
    #         return []

    #     # 1. 构建节点到链索引的映射
    #     node_to_chain = {}
    #     for chain_idx, chain in enumerate(chains):
    #         for nid in chain:
    #             node_to_chain[nid] = chain_idx

    #     # 2. 构建链依赖图
    #     chain_graph = {i: set() for i in range(len(chains))}
    #     chain_in_deg = {i: 0 for i in range(len(chains))}

    #     for e in edges:
    #         u = e["source"]
    #         v = e["target"]
    #         if u in node_to_chain and v in node_to_chain:
    #             c_u = node_to_chain[u]
    #             c_v = node_to_chain[v]
    #             if c_u != c_v and c_v not in chain_graph[c_u]:
    #                 chain_graph[c_u].add(c_v)
    #                 chain_in_deg[c_v] += 1

    #     # 3. 拓扑排序（核心：新增自动破环逻辑）
    #     sorted_chain_indices = []
    #     # 复制入度，避免修改原始数据
    #     temp_in_deg = chain_in_deg.copy()

    #     while len(sorted_chain_indices) < len(chains):
    #         # 第一步：找当前入度=0的链
    #         queue = [i for i in temp_in_deg if temp_in_deg[i] == 0]

    #         # =============================
    #         # 🛡️ 【你的核心需求】破环逻辑
    #         # 若没有入度0的链（全>0，出现环）
    #         # 所有链入度 -1，直到出现入度0的链
    #         # =============================
    #         while not queue:
    #             # 所有链入度统一减 1
    #             for i in temp_in_deg:
    #                 temp_in_deg[i] -= 1
    #             # 重新查找入度0的链
    #             queue = [i for i in temp_in_deg if temp_in_deg[i] == 0]

    #         # 同层级按长度降序排列（长链优先）
    #         queue.sort(key=lambda x: -len(chains[x]))

    #         # 处理当前入度0的链
    #         for c_idx in queue:
    #             sorted_chain_indices.append(c_idx)
    #             # 标记为已处理（设为-1，避免重复处理）
    #             temp_in_deg[c_idx] = -1
    #             # 更新后继链的入度
    #             for neighbor in chain_graph[c_idx]:
    #                 if temp_in_deg[neighbor] > 0:
    #                     temp_in_deg[neighbor] -= 1

    #     # 4. 重建排序后的链
    #     sorted_chains = [chains[i] for i in sorted_chain_indices]
    #     return sorted_chains