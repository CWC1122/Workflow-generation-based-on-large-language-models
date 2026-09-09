# =============================
# chain_merger.py - Stage7 链合并与插入核心逻辑
# =============================

class ChainMerger:
    """
    链合并器：基于DAG依赖约束，实现主链锚定+小链约束插入
    """

    def __init__(self, edges, service_dict):
        """
        初始化合并器
        Args:
            edges: 原始DAG的边列表
            service_dict: 服务字典，用于输入输出校验
        """
        self.edges = edges
        self.service_dict = service_dict

        # 构建节点的前驱和后继映射
        self.predecessors = {}  # {node_id: [前驱节点列表]}
        self.successors = {}    # {node_id: [后继节点列表]}
        self._build_dependency_map()

    def _build_dependency_map(self):
        """构建节点的前驱后继映射"""
        all_nodes = set()
        for e in self.edges:
            u = e["source"]
            v = e["target"]
            all_nodes.add(u)
            all_nodes.add(v)

            if v not in self.predecessors:
                self.predecessors[v] = []
            self.predecessors[v].append(u)

            if u not in self.successors:
                self.successors[u] = []
            self.successors[u].append(v)

        # 初始化没有前驱/后继的节点
        for n in all_nodes:
            if n not in self.predecessors:
                self.predecessors[n] = []
            if n not in self.successors:
                self.successors[n] = []

    def _get_chain_node_ids(self, chain):
        """从链中提取节点ID列表"""
        return [node["id"] for node in chain]

    def _get_global_predecessors(self, chain_node_ids):
        """获取分支链的全局前驱节点（不在分支链内的前驱）"""
        global_pred = set()
        for nid in chain_node_ids:
            for pred in self.predecessors.get(nid, []):
                if pred not in chain_node_ids:
                    global_pred.add(pred)
        return list(global_pred)

    def _get_global_successors(self, chain_node_ids):
        """获取分支链的全局后继节点（不在分支链内的后继）"""
        global_succ = set()
        for nid in chain_node_ids:
            for succ in self.successors.get(nid, []):
                if succ not in chain_node_ids:
                    global_succ.add(succ)
        return list(global_succ)

    def _calculate_insert_window(self, main_chain_node_ids, global_pred, global_succ):
        """计算合法插入窗口"""
        # 最早插入位置：所有前驱的最晚位置 + 1
        pred_positions = []
        for pred in global_pred:
            if pred in main_chain_node_ids:
                pred_positions.append(main_chain_node_ids.index(pred))
        earliest_pos = max(pred_positions) + 1 if pred_positions else 0

        # 最晚插入位置：所有后继的最早位置
        succ_positions = []
        for succ in global_succ:
            if succ in main_chain_node_ids:
                succ_positions.append(main_chain_node_ids.index(succ))
        latest_pos = min(succ_positions) if succ_positions else len(main_chain_node_ids)

        # 边界保护
        earliest_pos = max(earliest_pos, 0)
        latest_pos = min(latest_pos, len(main_chain_node_ids))

        return earliest_pos, latest_pos

    def _select_best_insert_pos(self, main_chain_node_ids, earliest_pos, latest_pos, global_pred, global_succ):
        """在合法窗口内选择最优插入位置"""
        # 优先输入就近：插在最后一个前驱的紧后面
        if global_pred:
            pred_positions = [main_chain_node_ids.index(pred) for pred in global_pred if pred in main_chain_node_ids]
            if pred_positions:
                last_pred_pos = max(pred_positions)
                if last_pred_pos + 1 >= earliest_pos and last_pred_pos + 1 <= latest_pos:
                    return last_pred_pos + 1

        # 其次输出就近：插在第一个后继的紧前面
        if global_succ:
            succ_positions = [main_chain_node_ids.index(succ) for succ in global_succ if succ in main_chain_node_ids]
            if succ_positions:
                first_succ_pos = min(succ_positions)
                if first_succ_pos >= earliest_pos and first_succ_pos <= latest_pos:
                    return first_succ_pos

        # 兜底：插在窗口的最前面
        return earliest_pos

    def merge_chains(self, chains_selected):
        """
        合并所有链，生成最终的执行序列
        Args:
            chains_selected: stage6输出的chain_selected_services
        Returns:
            final_execution_sequence: 最终的线性执行序列
        """
        if not chains_selected:
            return []

        # 步骤1：确定主链和分支链
        main_chain = chains_selected[0]  # 主链：排序后的第一条
        branch_chains = chains_selected[1:]  # 分支链

        # 分支链排序：节点数从小到大 → 总权重从高到低
        branch_chains.sort(key=lambda x: (len(x), -sum([n["selected_service_score"] for n in x])))

        # 步骤2：迭代插入每个分支链
        for branch_chain in branch_chains:
            main_chain_node_ids = self._get_chain_node_ids(main_chain)
            branch_node_ids = self._get_chain_node_ids(branch_chain)

            # 计算全局前驱和后继
            global_pred = self._get_global_predecessors(branch_node_ids)
            global_succ = self._get_global_successors(branch_node_ids)

            # 计算合法插入窗口
            earliest_pos, latest_pos = self._calculate_insert_window(main_chain_node_ids, global_pred, global_succ)

            # 选择最优插入位置
            best_pos = self._select_best_insert_pos(main_chain_node_ids, earliest_pos, latest_pos, global_pred, global_succ)

            # 执行插入
            main_chain = main_chain[:best_pos] + branch_chain + main_chain[best_pos:]

        # 返回最终执行序列
        return main_chain

    # =============================
    # 🔥 【关键修改】输出格式严格对齐你的 recom_result
    # =============================
    def convert_to_recom_format(self, final_sequence):
        """
        把合并后的序列 转换成 你要求的 recom_result 格式
        """
        recom_result = []
        for node in final_sequence:
            service_id = node["selected_service"]
            service_info = self.service_dict.get(service_id, {})

            recom_item = {
                "action_uid": service_info.get("action_uid", service_id),
                "target_action_reprs": service_info.get("target_action_reprs", ""),
                "input-type": service_info.get("input-type", []),
                "output-type": service_info.get("output-type", []),
                "id": service_id  # 严格和你例子一致
            }
            recom_result.append(recom_item)
        return recom_result

    def process_task(self, task):
        """
        处理单个任务的主入口
        Args:
            task: 任务字典
        Returns:
            task: 新增recom_result字段的任务
        """
        chains_selected = task.get("chain_selected_services", [])
        if not chains_selected:
            task["recom_result"] = []
            return task

        # 合并链
        final_sequence = self.merge_chains(chains_selected)
        
        # 🔥 转换成你要的格式
        final_recom = self.convert_to_recom_format(final_sequence)

        # 🔥 写入任务：新增 recom_result 键值对
        task["recom_result"] = final_recom
        return task