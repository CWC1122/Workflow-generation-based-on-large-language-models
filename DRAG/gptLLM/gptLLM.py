import json
import os
import re
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional


try:
    from openai import OpenAI
except Exception:  # pragma: no cover - fallback path when the OpenAI SDK is unavailable
    OpenAI = None


# ==================== GPT API configuration ====================
# These defaults mirror the working GPT5.5LLM.py caller. Environment variables
# still take precedence when they are set by the runner.
GPT_MODEL = "gpt-5.4-mini"
GPT_BASE_URL = "https://claude.aiapis.help/v1"
GPT_API_KEY = ""
GPT_API_STYLE = "responses"  # "responses" for GPT5.5LLM.py style, "chat" for /chat/completions.
GPT_PROXY_URL = "http://127.0.0.1:7897"
GPT_PROXY_FALLBACK_URLS = [
    "http://127.0.0.1:7897",
    "http://127.0.0.1:7890",
    "http://127.0.0.1:10809",
    "",
]
GPT_TIMEOUT = 600
GPT_MAX_RETRIES = 3
GPT_MAX_OUTPUT_TOKENS = 8192
# ==================== GPT API configuration ends ====================


DEFAULT_MODEL = os.getenv("OPENAI_MODEL") or GPT_MODEL
DEFAULT_BASE_URL = os.getenv("OPENAI_BASE_URL") or GPT_BASE_URL
DEFAULT_API_KEY = os.getenv("OPENAI_API_KEY") or GPT_API_KEY
DEFAULT_API_STYLE = (os.getenv("OPENAI_API_STYLE") or GPT_API_STYLE).lower()
DEFAULT_PROXY_URL = os.getenv("CODEGEN_PROXY_URL") or os.getenv("OPENAI_PROXY_URL") or GPT_PROXY_URL
DEFAULT_PROXY_URLS = (
    os.getenv("CODEGEN_PROXY_URLS")
    or os.getenv("OPENAI_PROXY_URLS")
    or ",".join(GPT_PROXY_FALLBACK_URLS)
)
DEFAULT_TIMEOUT = int(os.getenv("OPENAI_TIMEOUT") or GPT_TIMEOUT)
DEFAULT_MAX_RETRIES = int(os.getenv("OPENAI_MAX_RETRIES") or GPT_MAX_RETRIES)
DEFAULT_MAX_OUTPUT_TOKENS = int(
    os.getenv("CODEGEN_RESPONSES_MAX_OUTPUT_TOKENS")
    or os.getenv("OPENAI_MAX_OUTPUT_TOKENS")
    or GPT_MAX_OUTPUT_TOKENS
)


def _normalize_base_url(base_url: str) -> str:
    endpoint = (base_url or DEFAULT_BASE_URL).rstrip("/")
    return endpoint if endpoint.endswith("/v1") else f"{endpoint}/v1"


def _responses_endpoint(base_url: str) -> str:
    endpoint = (base_url or DEFAULT_BASE_URL).rstrip("/")
    if endpoint.endswith("/responses"):
        return endpoint
    if endpoint.endswith("/v1"):
        return f"{endpoint}/responses"
    return f"{endpoint}/v1/responses"


def _resolve_api_key(api_key: Optional[str]) -> str:
    resolved_key = api_key or DEFAULT_API_KEY
    if not resolved_key:
        raise ValueError(
            "GPT API key is required. Pass api_key=... or set the OPENAI_API_KEY environment variable."
        )
    return resolved_key


def _strip_thinking(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"^\s*```(?:json)?\s*", "", text.strip(), flags=re.IGNORECASE)
    text = re.sub(r"\s*```\s*$", "", text.strip())
    return text.strip()


def _extract_response_text(response: Dict[str, Any]) -> str:
    if response.get("output_text"):
        return str(response["output_text"])

    parts: List[str] = []
    for item in response.get("output", []) or []:
        if item.get("type") != "message":
            continue
        for content in item.get("content", []) or []:
            if content.get("type") in {"output_text", "text"}:
                parts.append(str(content.get("text", "")))
    return "".join(parts)


def _build_opener(proxy_url: Optional[str]) -> urllib.request.OpenerDirector:
    if proxy_url:
        return urllib.request.build_opener(
            urllib.request.ProxyHandler({"http": proxy_url, "https": proxy_url})
        )
    return urllib.request.build_opener()


def _proxy_candidates(proxy_url: Optional[str]) -> List[Optional[str]]:
    if proxy_url is not None:
        raw_values = [proxy_url]
    else:
        raw_values = DEFAULT_PROXY_URLS.split(",")

    candidates: List[Optional[str]] = []
    for raw_value in raw_values:
        value = raw_value.strip()
        candidate = value or None
        if candidate not in candidates:
            candidates.append(candidate)
    return candidates or [None]


def _post_responses_streaming(
    base_url: str,
    api_key: str,
    payload: Dict[str, Any],
    timeout: int,
    max_retries: int,
    proxy_url: Optional[str],
) -> str:
    endpoint = _responses_endpoint(base_url)
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
        "User-Agent": "python-requests/2.32.5",
        # Some third-party gateways reject the OpenAI SDK stainless headers. The
        # working standalone caller sends them as empty strings, so we mirror it.
        "X-Stainless-Lang": "",
        "X-Stainless-Package-Version": "",
        "X-Stainless-OS": "",
        "X-Stainless-Arch": "",
        "X-Stainless-Runtime": "",
        "X-Stainless-Runtime-Version": "",
        "X-Stainless-Async": "",
    }

    proxy_candidates = _proxy_candidates(proxy_url)
    last_error: Optional[BaseException] = None
    for attempt in range(max_retries + 1):
        for current_proxy in proxy_candidates:
            proxy_label = current_proxy or "direct/system proxy"
            chunks: List[str] = []
            final_text = ""
            completed_response: Dict[str, Any] = {}
            try:
                req = urllib.request.Request(endpoint, data=body, headers=headers, method="POST")
                opener = _build_opener(current_proxy)
                with opener.open(req, timeout=timeout) as response:
                    for raw_line in response:
                        line = raw_line.decode("utf-8", errors="replace").strip()
                        if not line.startswith("data:"):
                            continue

                        event_data = line.removeprefix("data:").strip()
                        if event_data == "[DONE]":
                            break

                        try:
                            event = json.loads(event_data)
                        except json.JSONDecodeError:
                            continue

                        event_type = event.get("type")
                        if event_type == "response.output_text.delta":
                            chunks.append(str(event.get("delta", "")))
                        elif event_type == "response.output_text.done":
                            final_text = str(event.get("text", ""))
                        elif event_type == "response.completed":
                            completed_response = event.get("response") or {}
                        elif event_type in {"response.failed", "response.incomplete"}:
                            error = event.get("error") or (event.get("response") or {}).get("error")
                            raise RuntimeError(f"Responses API stream failed via {proxy_label}: {error or event}")

                answer = "".join(chunks) or final_text or _extract_response_text(completed_response)
                if not answer:
                    raise RuntimeError(
                        f"Responses API returned successfully via {proxy_label} but contained no text output."
                    )
                return _strip_thinking(answer)
            except urllib.error.HTTPError as exc:
                details = exc.read().decode("utf-8", errors="replace")
                last_error = RuntimeError(
                    f"Responses API request failed via {proxy_label} (HTTP {exc.code}): {details}"
                )
            except urllib.error.URLError as exc:
                last_error = RuntimeError(f"Responses API connection failed via {proxy_label}: {exc}")
            except Exception as exc:
                last_error = exc

        if attempt >= max_retries:
            raise last_error
        time.sleep(min(2 ** attempt, 30))

    raise RuntimeError("Responses API request failed.") from last_error


def _post_chat_completion(
    base_url: str,
    api_key: str,
    payload: Dict[str, Any],
    timeout: int,
    max_retries: int,
) -> str:
    endpoint = f"{_normalize_base_url(base_url)}/chat/completions"
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    last_error: Optional[BaseException] = None
    for attempt in range(max_retries + 1):
        try:
            req = urllib.request.Request(endpoint, data=body, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            return _strip_thinking(data["choices"][0]["message"]["content"])
        except Exception as exc:
            last_error = exc
            if attempt >= max_retries:
                raise
            time.sleep(min(2 ** attempt, 30))
    raise RuntimeError("Chat Completions request failed.") from last_error


def gptLLM(
    messages: List[Dict[str, str]],
    model: str = DEFAULT_MODEL,
    temperature: float = 0.3,
    base_url: str = DEFAULT_BASE_URL,
    api_key: Optional[str] = None,
    timeout: int = DEFAULT_TIMEOUT,
    max_retries: int = DEFAULT_MAX_RETRIES,
    max_tokens: Optional[int] = None,
    request_timeout: Optional[int] = None,
    api_style: Optional[str] = None,
    proxy_url: Optional[str] = None,
    max_output_tokens: Optional[int] = None,
    **kwargs: Any,
) -> str:
    """Call a GPT-compatible endpoint with the same signature as localLLM.

    By default this uses the Responses API streaming protocol from GPT5.5LLM.py.
    Set OPENAI_API_STYLE=chat if a standard /chat/completions endpoint is needed.
    """
    resolved_key = _resolve_api_key(api_key)
    resolved_timeout = int(request_timeout or timeout)
    resolved_style = (api_style or DEFAULT_API_STYLE).lower()
    resolved_proxy = proxy_url

    if max_output_tokens is None:
        max_output_tokens = max_tokens or DEFAULT_MAX_OUTPUT_TOKENS

    if resolved_style == "responses":
        payload: Dict[str, Any] = {
            "model": model,
            "input": messages,
            "store": False,
            "temperature": temperature,
            "max_output_tokens": int(max_output_tokens),
            "stream": True,
        }
        reasoning_effort = kwargs.pop("reasoning_effort", None) or os.getenv("CODEGEN_REASONING_EFFORT")
        if reasoning_effort:
            payload["reasoning"] = {"effort": reasoning_effort}
        return _post_responses_streaming(
            base_url=base_url,
            api_key=resolved_key,
            payload=payload,
            timeout=resolved_timeout,
            max_retries=max_retries,
            proxy_url=resolved_proxy,
        )

    request_kwargs: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    if max_tokens is not None:
        request_kwargs["max_tokens"] = max_tokens
    request_kwargs.update(kwargs)

    if OpenAI is not None:
        client = OpenAI(
            base_url=_normalize_base_url(base_url),
            api_key=resolved_key,
            timeout=resolved_timeout,
        )
        last_error: Optional[BaseException] = None
        for attempt in range(max_retries + 1):
            try:
                response = client.chat.completions.create(**request_kwargs)
                return _strip_thinking(response.choices[0].message.content or "")
            except Exception as exc:
                last_error = exc
                if attempt >= max_retries:
                    raise
                time.sleep(min(2 ** attempt, 30))
        raise RuntimeError("GPT request failed.") from last_error

    return _post_chat_completion(
        base_url=base_url,
        api_key=resolved_key,
        payload=request_kwargs,
        timeout=resolved_timeout,
        max_retries=max_retries,
    )


# Drop-in alias for files that expect a function named localLLM.
localLLM = gptLLM


if __name__ == "__main__":
    demo_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Reply with one short sentence."},
    ]
    print(gptLLM(demo_messages, temperature=0))
