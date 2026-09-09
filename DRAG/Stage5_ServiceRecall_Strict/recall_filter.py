"""
服务召回过滤器模块
"""

import numpy as np


class RecallFilter:
    """
    服务召回过滤器类
    """

    def __init__(self, embedder, services, graph_data):
        self.embedder = embedder
        self.services = services

        self.service_text = {
            s["action_uid"]: s["target_action_reprs"]
            for s in services
        }

        self.service_io = {
            s["action_uid"]: {
                "input": set(s.get("input-type", [])),
                "output": set(s.get("output-type", []))
            }
            for s in services
        }

        self.graph = {}
        for g in graph_data:
            u = g["source"]
            v = g["target"]
            if u not in self.graph:
                self.graph[u] = set()
            self.graph[u].add(v)

        self.service_emb = {}
        for sid, text in self.service_text.items():
            self.service_emb[sid] = self.embedder.get_embedding(text)

    def cosine(self, a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)

    def recall(self, query, topk=4):
        q_emb = self.embedder.get_embedding(query)
        scores = []
        for sid, emb in self.service_emb.items():
            sim = self.cosine(q_emb, emb)
            scores.append((sid, sim))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:topk]

    def filter_by_score(self, candidates):
        filtered = [c for c in candidates if c[1] >= 0.65]
        if not filtered:
            return [candidates[0]]
        return filtered

    # =============================
    # ✅ 修改点：细化 IO 过滤逻辑，针对单条边判断 none
    # =============================
    def filter_by_io(self, current_node_id, prev_node_id, next_node_id, edges, candidates):
        """
        基于输入输出类型兼容性过滤候选服务
        
        核心逻辑：
        1. 头节点（无前驱）：不管输入，只检查输出（如果有后继且边不为none）
        2. 尾节点（无后继）：不管输出，只检查输入（如果有前驱且边不为none）
        3. 中间节点：
           - 若 prev->current 边为 none：当前节点输入不受限
           - 若 current->next 边为 none：当前节点输出不受限
        """
        
        # --- 步骤 1：找到两条关键边的类型 ---
        prev_edge_type = None
        next_edge_type = None

        # 寻找前驱边 (prev -> current)
        if prev_node_id is not None:
            for e in edges:
                if e["source"] == prev_node_id and e["target"] == current_node_id:
                    prev_edge_type = e.get("type", "")
                    break

        # 寻找后继边 (current -> next)
        if next_node_id is not None:
            for e in edges:
                if e["source"] == current_node_id and e["target"] == next_node_id:
                    next_edge_type = e.get("type", "")
                    break

        new_candidates = []

        for sid, score in candidates:
            io = self.service_io.get(sid, {"input": set(), "output": set()})

            # --- 步骤 2：分别判断输入和输出是否通过 ---
            
            # 判断输入是否通过
            in_ok = True
            # 只有当【存在前驱】且【前驱边类型不是 none】时，才检查输入
            if prev_node_id is not None and prev_edge_type != "none":
                # 检查服务的 input 是否包含前驱边的类型
                in_ok = prev_edge_type in io["input"]

            # 判断输出是否通过
            out_ok = True
            # 只有当【存在后继】且【后继边类型不是 none】时，才检查输出
            if next_node_id is not None and next_edge_type != "none":
                # 检查服务的 output 是否包含后继边的类型
                out_ok = next_edge_type in io["output"]

            if in_ok and out_ok:
                new_candidates.append((sid, score))

        return new_candidates if new_candidates else candidates

    def filter_by_graph(self, chains_candidates):
        """
        图过滤保持完全不变
        """
        for chain in chains_candidates:
            for i in range(len(chain) - 1):
                left = chain[i]["candidates"]
                right = chain[i + 1]["candidates"]

                new_left = []
                new_right = []

                for l_sid, l_score in left:
                    ok = False
                    for r_sid, _ in right:
                        if l_sid in self.graph and r_sid in self.graph[l_sid]:
                            ok = True
                            break
                    if ok:
                        new_left.append((l_sid, l_score))

                for r_sid, r_score in right:
                    ok = False
                    for l_sid, _ in left:
                        if l_sid in self.graph and r_sid in self.graph[l_sid]:
                            ok = True
                            break
                    if ok:
                        new_right.append((r_sid, r_score))

                if new_left:
                    chain[i]["candidates"] = new_left
                if new_right:
                    chain[i + 1]["candidates"] = new_right

        return chains_candidates

    def strong_filter(self, candidates, st15_services):
        if not candidates:
            return candidates
        top1 = candidates[0]
        if top1[1] >= 0.999 and top1[0] in st15_services:
            return [top1]
        return candidates

    def process_task(self, task):
        chains = task.get("chains", [])
        edges = task.get("edges", [])
        st15 = set(task.get("st1.5_service", []))

        chains_candidates = []

        for chain in chains:
            node_list = []
            chain_len = len(chain)

            for idx, node in enumerate(chain):
                nid = node["id"]
                text = node["requirement"]

                prev_node_id = chain[idx - 1]["id"] if idx > 0 else None
                next_node_id = chain[idx + 1]["id"] if idx < chain_len - 1 else None

                cands = self.recall(text, topk=4)
                cands = self.filter_by_score(cands)
                cands = self.filter_by_io(nid, prev_node_id, next_node_id, edges, cands)
                cands = self.strong_filter(cands, st15)

                node_list.append({
                    "id": nid,
                    "requirement": text,
                    "candidates": cands
                })

            chains_candidates.append(node_list)

        chains_candidates = self.filter_by_graph(chains_candidates)
        task["chain_candidates"] = chains_candidates

        return task