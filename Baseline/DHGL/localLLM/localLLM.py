import sys
from pathlib import Path

from openai import OpenAI


BASELINE_ROOT = Path(__file__).resolve().parents[2]
if str(BASELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(BASELINE_ROOT))

from baseline_runtime import get_baseline_llm_base_url, get_baseline_llm_model


def localLLM(
    messages,
    model=None,
    temperature=0,
    base_url=None,
    api_key="ollama",
):
    resolved_model = model or get_baseline_llm_model("local-chat-model")
    resolved_base_url = base_url or get_baseline_llm_base_url()
    client = OpenAI(
        base_url=resolved_base_url,
        api_key=api_key,
    )
    response = client.chat.completions.create(
        model=resolved_model,
        messages=messages,
        temperature=temperature,
    )
    return response.choices[0].message.content
