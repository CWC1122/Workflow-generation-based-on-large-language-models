import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gptLLM.gptLLM import localLLM
from utils import extract_json, load_dependency_types,fix_json_string


# ==================== GPT 配置区域 ====================
GPT_MODEL = "gpt-4o-mini"
DEFAULT_GPT_MODEL = os.getenv("OPENAI_DAG_MODEL") or os.getenv("OPENAI_MODEL") or GPT_MODEL
# ==================== GPT 配置结束 ====================


class DAGGenerator:

    def __init__(self, tool_desc_path):

        # ✅ 从 main 传入路径
        self.dep_types = load_dependency_types(tool_desc_path)


        self.service_dict = self.load_service_dict(tool_desc_path)

    # =============================
    # 加载服务字典
    # =============================
    def load_service_dict(self, path):

        with open(path, "r", encoding="utf-8") as f:
            services = json.load(f)

        return {s["action_uid"]: s for s in services}

    # =============================
    # 构建 node 模板
    # =============================
    def build_node_template(self, task_num):

        return [
            {"id": i + 1, "requirement": ""}
            for i in range(task_num)
        ]

    # =============================
    # 构建 prompt（只做增强，不改结构）
    # =============================
    def build_prompt(self, task, nodes, service_ids):

        node_count = len(nodes)

        # ===== 服务描述 =====
        service_blocks = []
        selected_services = []

        for sid in service_ids or []:
            s = self.service_dict.get(sid)
            if not s:
                continue

            selected_services.append(s)

            input_type = s.get("input-type", [])
            output_type = s.get("output-type", [])

            block = f"""
    Service ID: {s["action_uid"]}
    Description: {s["target_action_reprs"]}
    Input: {input_type}
    Output: {output_type}
    """
            service_blocks.append(block)

        # ===== ✅ 判断是否有 IO =====
        has_io = any(
            ("input-type" in s and "output-type" in s)
            for s in selected_services
        )

        dep_str = ", ".join(self.dep_types) if has_io else ""

        service_text = "\n".join(service_blocks) if service_blocks else "None"

        prompt = f"""
You are a task planner for an service composition system.

Your job is NOT to solve the task.
Your job is to decompose the task into capability requirements
that can be matched to services.

--------------------------------
Decomposition Guidance
--------------------------------

You are given an estimated number of subtasks: {node_count}.

IMPORTANT:
- This number is ONLY a reference derived from similar historical tasks.
- It is NOT a strict requirement.

Rules:
1. If the task can be completed with FEWER subtasks, use fewer nodes.
2. If the task REQUIRES more subtasks, you may exceed this number.
3. DO NOT create unnecessary or unrelated subtasks just to match the number.
4. Every subtask MUST contribute directly to solving the user task.

--------------------------------
Subtask Definition
--------------------------------

A subtask must describe WHAT capability is needed,
NOT the final answer.

Good examples:
- "Analyze the sentiment of the text"
- "Translate the text into English"
- "Detect objects in the image"
- "Transcribe speech from the audio"

Bad examples:
- "The sentiment is positive"
- "The translated result is ..."
- "The detected object is a cat"

--------------------------------
Graph Construction Rules
--------------------------------

You MUST follow the node template EXACTLY for IDs.

Node template:
{json.dumps(nodes, indent=2)}

Rules:
1. You MAY leave some nodes unused if not needed (they will be removed later).
2. Fill only necessary nodes with meaningful requirements.
3. DO NOT invent unrelated capabilities.
4. If the task naturally contains partially independent capabilities, you MAY represent them using DAG branches.

--------------------------------
DAG Constraints (STRICT)
--------------------------------

- The graph MUST be a DAG (no cycles).
- The graph MUST be fully CONNECTED (single component).
- DO NOT create independent disconnected subgraphs.
- DO NOT generate self-loops (e.g., 3 -> 3).
- Edges must only connect existing nodes.

IMPORTANT:
- The workflow does NOT need to be a single chain.
- If multiple subtasks are independent but all contribute to the user task, represent them as parallel branches in the DAG.
- If one subtask depends on outputs from multiple earlier subtasks, it MAY have multiple predecessors.
- Use branching or merging only when it reflects the real task structure.
- DO NOT create artificial branches if a simple chain is sufficient.

Examples:
Valid branching:
A -> B
A -> C

Valid merging:
A -> C
B -> C

Valid connected DAG:
A -> B
A -> C
B -> D
C -> D

--------------------------------
Branching and Merging Guidance
--------------------------------

Use a branch when:
- Two or more subtasks are independent
- They can be completed separately
- They contribute to different parts of the task

Use a merge when:
- A later subtask needs results, information, or artifacts from multiple previous subtasks

Do NOT force truly parallel subtasks into a single linear chain only for simplicity.

--------------------------------
Edge Constraints
--------------------------------

Edge types must be chosen ONLY from:

{dep_str}

IMPORTANT:

If the edge type list above is EMPTY:
-> This means there is NO strong dependency between services
-> You MUST set ALL edge types to "none"

Example:
{{"source":1,"target":2,"type":"none"}}

--------------------------------
Service Awareness
--------------------------------

You are given candidate services:

{service_text}

Rules:
1. Each subtask SHOULD map to a service
2. Prefer service wording
3. Make decomposition easy for service matching

--------------------------------
Special Case
--------------------------------

If there is only 1 node:
- The graph has no edges.

--------------------------------
Output Format (STRICT JSON)
--------------------------------

{{
 "nodes":[
   {{"id":1,"requirement":"..."}}
 ],
 "edges":[
    {{"source":1,"target":2,"type":"text"}}
 ]
}}

--------------------------------
User Task:
{task}
"""
        return prompt

    # =============================
    # 删除自环
    # =============================
    def remove_self_loops(self, edges):
        return [e for e in edges if e["source"] != e["target"]]

    # =============================
    # 判断 DAG（DFS）
    # =============================
    def is_dag(self, nodes, edges):

        graph = {n["id"]: [] for n in nodes}

        for e in edges:
            graph[e["source"]].append(e["target"])

        visited = set()
        stack = set()

        def dfs(v):
            if v in stack:
                return False
            if v in visited:
                return True

            stack.add(v)

            for nxt in graph[v]:
                if not dfs(nxt):
                    return False

            stack.remove(v)
            visited.add(v)
            return True

        return all(dfs(n) for n in graph)


    def force_connect_dag(self, nodes, edges):

        if not nodes:
            return nodes, edges

        # ===== 构图（无向）=====
        graph = {n["id"]: set() for n in nodes}

        for e in edges:
            graph[e["source"]].add(e["target"])
            graph[e["target"]].add(e["source"])

        visited = set()
        components = []

        def dfs(v, comp):
            comp.append(v)
            visited.add(v)
            for nxt in graph[v]:
                if nxt not in visited:
                    dfs(nxt, comp)

        for n in nodes:
            if n["id"] not in visited:
                comp = []
                dfs(n["id"], comp)
                components.append(comp)

        # ===== ✅ 修改点1：按 node id 排序 =====
        components.sort(key=lambda x: min(x))

        # ===== 如果已经连通 =====
        if len(components) <= 1:
            return nodes, edges

        # ===== 强制串联 =====
        new_edges = edges[:]

        for i in range(len(components) - 1):
            src = components[i][-1]
            tgt = components[i + 1][0]

            new_edges.append({
                "source": src,
                "target": tgt,
                # ===== ✅ 修改点2：type 改为 none =====
                "type": "none"
            })

        return nodes, new_edges

    # =============================
    # 删除环（贪心）
    # =============================
    def remove_cycles(self, nodes, edges):

        clean = []

        for e in edges:
            clean.append(e)
            if not self.is_dag(nodes, clean):
                clean.pop()

        return clean

    # =============================
    # 检测连通性（弱连通）
    # =============================
    def is_connected(self, nodes, edges):

        if not nodes:
            return False

        graph = {n["id"]: set() for n in nodes}

        for e in edges:
            graph[e["source"]].add(e["target"])
            graph[e["target"]].add(e["source"])

        visited = set()

        def dfs(v):
            visited.add(v)
            for nxt in graph[v]:
                if nxt not in visited:
                    dfs(nxt)

        dfs(nodes[0]["id"])

        return len(visited) == len(nodes)
    # =============================
    # 删除空节点 + 清理边 + 重排id
    # =============================
    def remove_empty_nodes(self, nodes, edges):

        # ===== Step1: 找到非空节点 =====
        valid_nodes = [
            n for n in nodes
            if n.get("requirement", "").strip() != ""
        ]

        if not valid_nodes:
            return [], []

        valid_ids = set(n["id"] for n in valid_nodes)

        # ===== Step2: 删除相关边 =====
        clean_edges = [
            e for e in edges
            if e["source"] in valid_ids and e["target"] in valid_ids
        ]

        # ===== Step3: 重排 node id（关键！）=====
        id_map = {}
        new_nodes = []

        for new_id, node in enumerate(valid_nodes, start=1):
            old_id = node["id"]
            id_map[old_id] = new_id

            new_nodes.append({
                "id": new_id,
                "requirement": node["requirement"]
            })

        # ===== Step4: 更新 edges =====
        new_edges = []

        for e in clean_edges:
            new_edges.append({
                "source": id_map[e["source"]],
                "target": id_map[e["target"]],
                "type": e["type"]
            })

        return new_nodes, new_edges
    # =============================
    # 主函数
    # =============================
    def generate_dag(self, task, task_num, service_ids=None):

        node_template = self.build_node_template(task_num)

        prompt = self.build_prompt(task, node_template, service_ids)

        messages = [
            {"role": "system", "content": "You are a strict JSON generator. Output valid JSON only. No explanation."},           
            {"role": "user", "content": prompt}
        ]

        response = localLLM(
            messages,
            model=DEFAULT_GPT_MODEL,
            temperature=0.0
        )

        response = fix_json_string(response)
        dag = extract_json(response)
        # print(response)
        # print("-----------------------------------------------------------------------------------------------")
        # print(dag)
        if not dag:
            print("❌ fail: JSON parse error")
            return None

        nodes = dag.get("nodes", [])
        edges = dag.get("edges", [])
        # ✅ Step0: 先清理空节点（必须最前）
        nodes, edges = self.remove_empty_nodes(nodes, edges)
        # ✅ 新增
        nodes, edges = self.force_connect_dag(nodes, edges)
        # ===== Step1: 删除自环 =====
        edges = self.remove_self_loops(edges)

        # ===== Step2: 删除环 =====
        edges = self.remove_cycles(nodes, edges)

        # ===== Step3: 检查 DAG =====
        if not self.is_dag(nodes, edges):
            return None

        # ===== Step4: 检查连通性 =====
        if not self.is_connected(nodes, edges):
            return None

        return {
            "nodes": nodes,
            "edges": edges
        }
