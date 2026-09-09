import json



import re

def fix_json_string(s):
    # 去掉 trailing comma
    s = re.sub(r",\s*([\]}])", r"\1", s)
    return s
def extract_json(text):
    if not text:
        return None

    import json

    # 直接尝试
    try:
        return json.loads(text)
    except:
        pass

    stack = []
    start = None

    for i, ch in enumerate(text):
        if ch == "{":
            if not stack:
                start = i
            stack.append(ch)
        elif ch == "}":
            if stack:
                stack.pop()
                if not stack and start is not None:
                    candidate = text[start:i+1]
                    try:
                        parsed = json.loads(candidate)
                        if "nodes" in parsed:
                            return parsed
                    except:
                        continue

    return None


# import json
# import re
 
# def extract_json(text):
#     """
#     从LLM输出中提取JSON
#     """
    
#     if not text:
#         return None
    
#     # 尝试找到完整的JSON对象
#     # 使用正则表达式匹配最外层的 { ... }
#     pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
    
#     # 查找所有可能的JSON对象
#     matches = re.findall(pattern, text, re.DOTALL)
    
#     # 从后往前尝试，因为通常完整的JSON在后面
#     for match in reversed(matches):
#         try:
#             parsed = json.loads(match)
#             # 检查是否包含nodes字段
#             if "nodes" in parsed:
#                 return parsed
#         except:
#             continue
    
#     # 如果正则表达式方法失败，尝试传统方法
#     start = text.find("{")
#     end = text.rfind("}")
 
#     if start == -1 or end == -1:
#         return None

#     json_str = text[start:end+1]

#     try:
#         return json.loads(json_str)
#     except:
#         return None
    


def load_dependency_types(tool_desc_path):
    """
    从 tool_desc.json 中提取所有 input-type
    """

    with open(tool_desc_path, "r", encoding="utf-8") as f:
        tools = json.load(f)

    dep_types = set()

    for tool in tools:

        for t in tool.get("input-type", []):
            dep_types.add(t)

    return sorted(list(dep_types))