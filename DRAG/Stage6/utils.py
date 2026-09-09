# =============================
# service_selector.py - Stage6 核心服务选择逻辑
# =============================

import json
from gptLLM.gptLLM import localLLM


def extract_json(text):
    """
    【Stage6专用】提取JSON，不检查 "nodes"
    """
    if not text:
        return None

    # 直接尝试
    try:
        return json.loads(text)
    except:
        pass

    # 栈匹配
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
                        return json.loads(candidate)
                    except:
                        continue
    return None

