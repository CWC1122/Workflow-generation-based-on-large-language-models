import json
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gptLLM.gptLLM import localLLM
from utils import extract_json


# ==================== GPT 配置区域 ====================
GPT_MODEL = "gpt-4o-mini"
DEFAULT_GPT_MODEL = os.getenv("OPENAI_PATH_MODEL") or os.getenv("OPENAI_MODEL") or GPT_MODEL
# ==================== GPT 配置结束 ====================

def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))
class ServiceSelector:

    def __init__(self, service_dict, service_graph, embedder, beam_width=3):
        self.service_dict = service_dict
        self.graph = service_graph
        self.embedder = embedder
        self.beam_width = beam_width

        # 🔥 embedding cache（全局）
        self.emb_cache = {}


    
    # =============================
    # embedding cache
    # =============================
    def get_emb(self, text):
        if text not in self.emb_cache:
            emb = self.embedder.get_embedding(text)

            # 🔥 强制转 numpy（关键修复）
            emb = np.array(emb, dtype=np.float32)

            self.emb_cache[text] = emb

        return self.emb_cache[text]

    # =============================
    # Beam Search（终极版）
    # =============================
    def beam_search(self, task, chain, chain_candidates, edges):

        user_query = task.get("confirmed_task", "")
        user_emb = self.get_emb(user_query)

        # edge type map
        edge_type_map = {
            (e["source"], e["target"]): e.get("type", "none")
            for e in edges
        }

        # path = (services, score, emb)
        paths = [([], 0.0, None)]

        for step_idx, node in enumerate(chain_candidates):

            new_paths = []

            for path, score, path_emb in paths:

                for sid, local_score in node["candidates"]:

                    # =============================
                    # 🔥 图过滤（你的规则）
                    # =============================
                    if path:
                        prev_node = chain[step_idx - 1]["id"]
                        curr_node = chain[step_idx]["id"]

                        edge_type = edge_type_map.get((prev_node, curr_node), "none")

                        if edge_type != "none":
                            ok = False
                            for prev_sid in path:
                                if prev_sid in self.graph and sid in self.graph[prev_sid]:
                                    ok = True
                                    break
                            if not ok:
                                continue

                    # =============================
                    # embedding
                    # =============================
                    desc = self.service_dict.get(sid, {}).get("target_action_reprs", "")
                    sid_emb = self.get_emb(desc)

                    # =============================
                    # 路径 embedding（增量平均）
                    # =============================
                    if path_emb is None:
                        new_emb = sid_emb
                    else:
                        new_emb = (path_emb * len(path) + sid_emb) / (len(path) + 1)

                    # =============================
                    # 语义分数
                    # =============================
                    semantic_score = cosine(new_emb, user_emb)

                    # =============================
                    # Hybrid Score
                    # =============================
                    alpha = 0.7
                    hybrid = alpha * semantic_score + (1 - alpha) * (score + local_score)

                    new_paths.append((path + [sid], hybrid, new_emb))

            # 截断 top-k
            new_paths.sort(key=lambda x: -x[1])
            paths = new_paths[:self.beam_width]

            if not paths:
                break

        return [(p, s) for p, s, _ in paths]

    # =============================
    # Prompt（主链）
    # =============================
    def build_main_prompt(self, task, paths):

        user_query = task.get("confirmed_task", "")

        path_str = ""
        for i, (path, score) in enumerate(paths):
            path_str += f"\nPath {i+1} (score={score:.4f}):\n"
            for sid in path:
                desc = self.service_dict.get(sid, {}).get("target_action_reprs", "")
                path_str += f"  - {sid}: {desc}\n"

        return f"""
You are selecting the MAIN workflow.

User Task:
{user_query}

Candidate paths:
{path_str}

CRITICAL THINKING RULES:
1. DO NOT just pick highest score
2. You MUST evaluate:
   - Does this path COMPLETE the task?
   - Does it cover ALL core steps?
3. Prefer FULL solution over partial match
4. Workflow must be EXECUTABLE
5. Avoid missing steps

BAD:
- High score but incomplete
- Missing key steps

GOOD:
- Complete workflow
- Logical step-by-step pipeline

Output JSON:
{{"selected_path_id": 1, "reason": "why this path fully solves the task"}}
"""

    # =============================
    # Prompt（补充链）
    # =============================
    def build_sub_prompt(self, task, paths, main_chain):

        user_query = task.get("confirmed_task", "")

        main_str = ""
        for sid in main_chain:
            desc = self.service_dict.get(sid, {}).get("target_action_reprs", "")
            main_str += f"  - {sid}: {desc}\n"

        path_str = ""
        for i, (path, score) in enumerate(paths):
            path_str += f"\nPath {i+1} (score={score:.4f}):\n"
            for sid in path:
                desc = self.service_dict.get(sid, {}).get("target_action_reprs", "")
                path_str += f"  - {sid}: {desc}\n"

        return f"""
You are selecting a SUPPLEMENTARY workflow.

User Task:
{user_query}

MAIN WORKFLOW:
{main_str}

Candidate paths:
{path_str}

CRITICAL THINKING:
1. DO NOT repeat main workflow
2. MUST fill missing capability
3. MUST add new functionality
4. MUST be executable

Focus:
- What is NOT covered by main chain?
- Which path adds missing step?

Output JSON:
{{"selected_path_id": 1, "reason": "what gap it fills"}}
"""

    # =============================
    # Graph 强修复（提升ER）
    # =============================
    def fix_chain_with_graph(self, chain_selected):

        for i in range(len(chain_selected) - 1):
            s1 = chain_selected[i]["selected_service"]
            s2 = chain_selected[i + 1]["selected_service"]

            if s1 not in self.graph or s2 not in self.graph[s1]:

                # 从候选中找合法
                for candidate in self.graph.get(s1, []):
                    chain_selected[i + 1]["selected_service"] = candidate
                    chain_selected[i + 1]["source"] = "graph_fix"
                    break

        return chain_selected

    # =============================
    # 主函数
    # =============================
    def select_for_task(self, task):

        chains = task.get("chains", [])
        chain_candidates = task.get("chain_candidates", [])
        edges = task.get("edges", [])

        chain_selected_services = []
        main_chain_services = None

        for chain_idx, (chain, candidates) in enumerate(zip(chains, chain_candidates)):

            # ===== Beam =====
            paths = self.beam_search(task, chain, candidates, edges)

            if not paths:
                continue

            is_main = (chain_idx == 0)

            if is_main:
                prompt = self.build_main_prompt(task, paths)
            else:
                prompt = self.build_sub_prompt(task, paths, main_chain_services)

            try:
                messages = [
                    {"role": "system", "content": "Output JSON only"},
                    {"role": "user", "content": prompt}
                ]

                response = localLLM(
                                messages,
                                model=DEFAULT_GPT_MODEL,
                                temperature=0.0)
                result = extract_json(response)
                # print(response)
                # print("-----------------------------------------------------------------------------------------------")
                # print(result)
                idx = result.get("selected_path_id", 1) - 1
                idx = max(0, min(idx, len(paths) - 1))

                selected_path = paths[idx][0]
                reason = result.get("reason", "")

                current_selected = []

                for i, node in enumerate(chain):
                    sid = selected_path[i]

                    score = 0.0
                    for s_id, s_score in candidates[i]["candidates"]:
                        if s_id == sid:
                            score = s_score
                            break

                    current_selected.append({
                        "id": node["id"],
                        "requirement": node["requirement"],
                        "selected_service": sid,
                        "selected_service_score": score,
                        "source": "llm",
                        "reason": reason
                    })

                # current_selected = self.fix_chain_with_graph(current_selected)

                chain_selected_services.append(current_selected)

                if is_main:
                    main_chain_services = selected_path

            except Exception as e:
                print("LLM fail:", e)

        task["chain_selected_services"] = chain_selected_services
        return task
