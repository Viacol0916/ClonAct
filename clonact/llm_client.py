"""Homogeneous System-2 LLM Client supporting `deepseek-flash`, `gpt-6-luna`, and `qwen3.8-flash`.

Preserves strict 1-to-1 protocol parity with `@deepseek-ai/dsh`:
  - `deepseek-flash`: DeepSeek Anthropic-Messages wire protocol (`https://api.deepseek.com/anthropic/v1/messages`)
  - `gpt-6-luna`: OpenAI Responses API (`https://api.openai.com/v1/responses`, `reasoning={"effort": "high"}`)
  - `qwen3.8-flash`: Alibaba Token Plan OpenAI-Completions API (`https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1/chat/completions`, `enable_thinking=True`, `reasoning_effort="xhigh"`)
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx


def load_project_dotenv() -> None:
    """Load `.env` into `os.environ` if not already set (never prints secrets)."""
    env_file = Path(__file__).resolve().parent.parent / ".env"
    if not env_file.exists():
        return
    for raw in env_file.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip().strip("'\"")
        if k and k not in os.environ:
            os.environ[k] = v


def resolve_backbone_model_spec(model: str | None = None) -> dict[str, str]:
    """Return canonical `(model, protocol, dsh_provider, endpoint_url, api_key)` metadata."""
    load_project_dotenv()
    eff_model = (model or os.environ.get("PA_LLM_MODEL") or "deepseek-flash").strip() or "deepseek-flash"
    low = eff_model.lower()
    if low.startswith(("gpt-", "o1", "o3", "o4", "chatgpt-")):
        base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").strip().rstrip("/")
        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        return {
            "model": eff_model,
            "protocol": "openai_responses",
            "dsh_provider": "openai",
            "endpoint_url": f"{base_url}/responses",
            "api_key": api_key,
            "api_key_env": "OPENAI_API_KEY",
        }
    if low.startswith("qwen"):
        base_url = os.environ.get(
            "QWEN_BASE_URL",
            "https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
        ).strip().rstrip("/")
        api_key = (
            os.environ.get("QWEN_TOKEN_PLAN_CN_API_KEY", "").strip()
            or os.environ.get("QWEN_TOKEN_PLAN_API_KEY", "").strip()
            or os.environ.get("DASHSCOPE_API_KEY", "").strip()
        )
        return {
            "model": eff_model,
            "protocol": "qwen_completions",
            "dsh_provider": "qwen-token-plan-cn",
            "endpoint_url": f"{base_url}/chat/completions",
            "api_key": api_key,
            "api_key_env": "QWEN_TOKEN_PLAN_CN_API_KEY",
        }
    api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    return {
        "model": eff_model,
        "protocol": "deepseek_anthropic",
        "dsh_provider": "deepseek-official",
        "endpoint_url": "https://api.deepseek.com/anthropic/v1/messages",
        "api_key": api_key,
        "api_key_env": "DEEPSEEK_API_KEY",
    }


def query_deepseek_balance_cny(api_key: str | None = None) -> float | None:
    """Query live CNY total_balance from `https://api.deepseek.com/user/balance`."""
    load_project_dotenv()
    key = api_key or os.environ.get("DEEPSEEK_API_KEY", "")
    if not key:
        return None
    try:
        resp = httpx.get(
            "https://api.deepseek.com/user/balance",
            headers={"Authorization": f"Bearer {key}"},
            timeout=15.0,
        )
        if resp.status_code != 200:
            return None
        infos = resp.json().get("balance_infos") or []
        for item in infos:
            if item.get("currency") == "CNY":
                return float(item["total_balance"])
    except Exception:
        return None
    return None


def estimate_cost_cny_upper_bound(
    cache_hit_tokens: int,
    cache_miss_tokens: int,
    output_tokens: int,
    *,
    model: str | None = None,
) -> float:
    """Conservative CNY upper-bound estimate per 1M tokens for the active backbone model."""
    eff_model = (model or os.environ.get("PA_LLM_MODEL") or "deepseek-flash").strip().lower()
    if eff_model.startswith(("gpt-6-luna", "gpt-5.4-nano")):
        # OpenAI catalog rate for gpt-6-luna: $0.01/1M cached, $0.10/1M miss, $0.50/1M output (@ 7.25 CNY/USD)
        usd = (
            (cache_hit_tokens / 1_000_000.0) * 0.01
            + (cache_miss_tokens / 1_000_000.0) * 0.10
            + (output_tokens / 1_000_000.0) * 0.50
        )
        return usd * 7.25
    if eff_model.startswith("qwen"):
        return (
            (cache_hit_tokens / 1_000_000.0) * 0.05
            + (cache_miss_tokens / 1_000_000.0) * 0.50
            + (output_tokens / 1_000_000.0) * 2.00
        )
    # Standard DeepSeek ceiling rates (Cache Hit: ¥0.5/1M, Cache Miss: ¥4.0/1M, Output+Thinking: ¥12.0/1M)
    return (
        (cache_hit_tokens / 1_000_000.0) * 0.50
        + (cache_miss_tokens / 1_000_000.0) * 4.00
        + (output_tokens / 1_000_000.0) * 12.00
    )


@dataclass
class LLMCallMetric:
    caller: str
    prompt_tokens: int
    cache_hit_tokens: int
    cache_miss_tokens: int
    completion_tokens: int
    latency_ms: float


class HomogeneousLLMClient:
    """Shared θ System-2 client tracking prefix KV-cache reuse across DeepSeek, OpenAI, and Qwen."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        reasoning_effort: str = "high",
        max_tokens: int = 32768,
    ) -> None:
        spec = resolve_backbone_model_spec(model)
        self.model = spec["model"]
        self.protocol = spec["protocol"]
        self.endpoint_url = spec["endpoint_url"]
        self.api_key_env = spec["api_key_env"]
        self.api_key = api_key or spec["api_key"]
        self.reasoning_effort = reasoning_effort
        self.max_tokens = max_tokens
        self.metrics: list[LLMCallMetric] = []

    @staticmethod
    def _convert_tools_to_anthropic(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
        if not tools:
            return []
        converted: list[dict[str, Any]] = []
        for t in tools:
            fn = t.get("function", t)
            converted.append(
                {
                    "name": fn["name"],
                    "description": fn.get("description", ""),
                    "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
                }
            )
        return converted

    @staticmethod
    def _convert_tools_to_openai_responses(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
        if not tools:
            return []
        converted: list[dict[str, Any]] = []
        for t in tools:
            fn = t.get("function", t)
            converted.append(
                {
                    "type": "function",
                    "name": fn["name"],
                    "description": fn.get("description", ""),
                    "parameters": fn.get("parameters", {"type": "object", "properties": {}}),
                }
            )
        return converted

    @staticmethod
    def _convert_tools_to_openai_completions(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
        if not tools:
            return []
        converted: list[dict[str, Any]] = []
        for t in tools:
            if t.get("type") == "function" and "function" in t:
                converted.append(t)
            else:
                fn = t.get("function", t)
                converted.append(
                    {
                        "type": "function",
                        "function": {
                            "name": fn["name"],
                            "description": fn.get("description", ""),
                            "parameters": fn.get("parameters", {"type": "object", "properties": {}}),
                        },
                    }
                )
        return converted

    @staticmethod
    def _serialize_messages_to_anthropic(
        messages: list[dict[str, Any]],
    ) -> tuple[str, list[dict[str, Any]]]:
        """Convert internal OpenAI-shaped message list into DeepSeek Anthropic Messages wire format."""
        top_system = ""
        wire_msgs: list[dict[str, Any]] = []
        pending_system_notes: list[str] = []

        for idx, msg in enumerate(messages):
            role = msg.get("role", "user")
            if role == "system":
                txt = str(msg.get("content") or "").strip()
                if not txt:
                    continue
                if idx == 0 and not top_system:
                    top_system = txt
                else:
                    pending_system_notes.append(txt)
                continue

            if role == "assistant":
                raw_blocks = msg.get("_raw_anthropic_blocks")
                if raw_blocks:
                    blocks = list(raw_blocks)
                else:
                    blocks = []
                    txt = str(msg.get("content") or "")
                    if txt:
                        blocks.append({"type": "text", "text": txt})
                    for tc in msg.get("tool_calls") or []:
                        fn = tc.get("function", {})
                        try:
                            inp = json.loads(fn.get("arguments") or "{}")
                        except Exception:
                            inp = {}
                        blocks.append(
                            {
                                "type": "tool_use",
                                "id": tc["id"],
                                "name": fn.get("name", ""),
                                "input": inp if isinstance(inp, dict) else {},
                            }
                        )
                if not blocks:
                    continue
                if wire_msgs and wire_msgs[-1]["role"] == "assistant":
                    wire_msgs[-1]["content"].extend(blocks)
                else:
                    wire_msgs.append({"role": "assistant", "content": blocks})
                continue

            if role == "tool":
                tr_block = {
                    "type": "tool_result",
                    "tool_use_id": msg["tool_call_id"],
                    "content": [{"type": "text", "text": str(msg.get("content") or "(empty)")}],
                }
                if wire_msgs and wire_msgs[-1]["role"] == "user":
                    wire_msgs[-1]["content"].append(tr_block)
                else:
                    wire_msgs.append({"role": "user", "content": [tr_block]})
                continue

            # role == "user"
            user_blocks: list[dict[str, Any]] = []
            if pending_system_notes:
                merged_sys = "\n\n".join(pending_system_notes)
                pending_system_notes.clear()
                user_blocks.append(
                    {"type": "text", "text": f"<context_directive>\n{merged_sys}\n</context_directive>"}
                )
            txt = str(msg.get("content") or "").strip()
            if txt:
                user_blocks.append({"type": "text", "text": txt})
            if not user_blocks:
                continue
            if wire_msgs and wire_msgs[-1]["role"] == "user":
                wire_msgs[-1]["content"].extend(user_blocks)
            else:
                wire_msgs.append({"role": "user", "content": user_blocks})

        if pending_system_notes:
            extra = {"type": "text", "text": "\n\n".join(pending_system_notes)}
            if wire_msgs and wire_msgs[-1]["role"] == "user":
                wire_msgs[-1]["content"].append(extra)
            else:
                wire_msgs.append({"role": "user", "content": [extra]})

        # Enforce Anthropic Messages rule: inside any user message, `tool_result` blocks must precede text blocks
        for wm in wire_msgs:
            if wm["role"] == "user":
                trs = [b for b in wm["content"] if b.get("type") == "tool_result"]
                others = [b for b in wm["content"] if b.get("type") != "tool_result"]
                wm["content"] = trs + others

        return top_system, wire_msgs

    @staticmethod
    def _serialize_messages_to_openai_responses(
        messages: list[dict[str, Any]],
    ) -> tuple[str, list[dict[str, Any]]]:
        """Convert internal OpenAI-shaped message list into OpenAI `/v1/responses` `(instructions, input)`."""
        instructions_parts: list[str] = []
        items: list[dict[str, Any]] = []

        for idx, msg in enumerate(messages):
            role = msg.get("role", "user")
            if role == "system":
                txt = str(msg.get("content") or "").strip()
                if not txt:
                    continue
                if idx == 0 and not items:
                    instructions_parts.append(txt)
                else:
                    items.append({"role": "developer", "content": txt})
                continue

            if role == "assistant":
                txt = str(msg.get("content") or "").strip()
                if txt:
                    items.append({"role": "assistant", "content": txt})
                for tc in msg.get("tool_calls") or []:
                    fn = tc.get("function", {})
                    items.append(
                        {
                            "type": "function_call",
                            "call_id": str(tc.get("id") or "call_0"),
                            "name": str(fn.get("name") or ""),
                            "arguments": str(fn.get("arguments") or "{}"),
                        }
                    )
                continue

            if role == "tool":
                items.append(
                    {
                        "type": "function_call_output",
                        "call_id": str(msg.get("tool_call_id") or "call_0"),
                        "output": str(msg.get("content") or "(empty)"),
                    }
                )
                continue

            txt = str(msg.get("content") or "").strip()
            if txt:
                items.append({"role": "user", "content": txt})

        if not items:
            items.append({"role": "user", "content": "Proceed."})
        return "\n\n".join(instructions_parts).strip(), items

    @staticmethod
    def _serialize_messages_to_qwen_completions(
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Normalize internal messages for Qwen OpenAI-Completions (`supportsDeveloperRole=False`)."""
        wire: list[dict[str, Any]] = []
        for idx, msg in enumerate(messages):
            role = msg.get("role", "user")
            if role == "system":
                txt = str(msg.get("content") or "").strip()
                if not txt:
                    continue
                if idx == 0 or not wire:
                    wire.append({"role": "system", "content": txt})
                else:
                    wire.append({"role": "user", "content": f"<context_directive>\n{txt}\n</context_directive>"})
                continue
            if role == "assistant":
                entry: dict[str, Any] = {
                    "role": "assistant",
                    "content": str(msg.get("content") or ""),
                }
                if msg.get("reasoning_content"):
                    entry["reasoning_content"] = str(msg["reasoning_content"])
                if msg.get("tool_calls"):
                    entry["tool_calls"] = list(msg["tool_calls"])
                wire.append(entry)
                continue
            if role == "tool":
                wire.append(
                    {
                        "role": "tool",
                        "tool_call_id": str(msg.get("tool_call_id") or ""),
                        "content": str(msg.get("content") or "(empty)"),
                    }
                )
                continue
            wire.append({"role": "user", "content": str(msg.get("content") or "")})
        return wire

    async def _post_json_with_retries(
        self,
        url: str,
        headers: dict[str, str],
        payload: dict[str, Any],
        caller: str,
    ) -> tuple[dict[str, Any], float]:
        t0 = time.perf_counter()
        last_exc: Exception | None = None
        data: dict[str, Any] = {}
        for attempt in range(6):
            try:
                async with httpx.AsyncClient(timeout=180.0) as client:
                    resp = await client.post(url, headers=headers, json=payload)
                if resp.status_code in (429, 500, 502, 503, 504) and attempt < 5:
                    await asyncio.sleep(min(30.0, 2.5 * (2**attempt)))
                    continue
                if resp.status_code >= 400:
                    raise RuntimeError(
                        f"LLM HTTP {resp.status_code} [{self.model} | {caller}]: {resp.text[:500]}"
                    )
                data = resp.json()
                break
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_exc = exc
                if attempt < 5:
                    await asyncio.sleep(min(30.0, 2.5 * (2**attempt)))
                    continue
                raise
        else:
            if last_exc:
                raise last_exc
        return data, (time.perf_counter() - t0) * 1000.0

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        caller: str = "cell",
        temperature: float = 0.1,
        reasoning_effort_override: str | None = None,
        max_tokens_override: int | None = None,
    ) -> dict[str, Any]:
        if not self.api_key:
            raise RuntimeError(f"{self.api_key_env} is required for System-2 execution ({self.model}).")

        effort = reasoning_effort_override if reasoning_effort_override is not None else self.reasoning_effort
        eff_max_tokens = max_tokens_override if max_tokens_override is not None else self.max_tokens

        if self.protocol == "openai_responses":
            return await self._chat_openai_responses(
                messages=messages,
                tools=tools,
                caller=caller,
                temperature=temperature,
                effort=effort,
                eff_max_tokens=eff_max_tokens,
            )
        if self.protocol == "qwen_completions":
            return await self._chat_qwen_completions(
                messages=messages,
                tools=tools,
                caller=caller,
                temperature=temperature,
                effort=effort,
                eff_max_tokens=eff_max_tokens,
            )
        return await self._chat_deepseek_anthropic(
            messages=messages,
            tools=tools,
            caller=caller,
            temperature=temperature,
            effort=effort,
            eff_max_tokens=eff_max_tokens,
        )

    async def _chat_openai_responses(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        caller: str,
        temperature: float,
        effort: str,
        eff_max_tokens: int,
    ) -> dict[str, Any]:
        instructions, input_items = self._serialize_messages_to_openai_responses(messages)
        resp_tools = self._convert_tools_to_openai_responses(tools)
        oai_effort = "none" if effort == "off" else effort
        payload: dict[str, Any] = {
            "model": self.model,
            "input": input_items,
            "max_output_tokens": eff_max_tokens,
            "reasoning": {"effort": oai_effort},
        }
        if instructions:
            payload["instructions"] = instructions
        if resp_tools:
            payload["tools"] = resp_tools
        if oai_effort == "none":
            payload["temperature"] = temperature

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        data, latency_ms = await self._post_json_with_retries(self.endpoint_url, headers, payload, caller)
        usage = data.get("usage") or {}
        in_details = usage.get("input_tokens_details") or {}
        p_tok = int(usage.get("input_tokens", 0))
        hit_tok = min(p_tok, int(in_details.get("cached_tokens", 0)))
        miss_tok = max(0, p_tok - hit_tok)
        c_tok = int(usage.get("output_tokens", 0))

        self.metrics.append(
            LLMCallMetric(
                caller=caller,
                prompt_tokens=p_tok,
                cache_hit_tokens=hit_tok,
                cache_miss_tokens=miss_tok,
                completion_tokens=c_tok,
                latency_ms=latency_ms,
            )
        )

        text_parts: list[str] = []
        reasoning_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        for item in data.get("output") or []:
            itype = item.get("type")
            if itype == "message":
                for c in item.get("content") or []:
                    if c.get("type") in ("output_text", "text") and c.get("text"):
                        text_parts.append(str(c["text"]))
            elif itype == "reasoning":
                for s in item.get("summary") or []:
                    if s.get("text"):
                        reasoning_parts.append(str(s["text"]))
            elif itype == "function_call":
                tool_calls.append(
                    {
                        "id": str(item.get("call_id") or item.get("id") or "call_0"),
                        "type": "function",
                        "function": {
                            "name": str(item.get("name") or ""),
                            "arguments": str(item.get("arguments") or "{}"),
                        },
                    }
                )

        out_msg: dict[str, Any] = {
            "role": "assistant",
            "content": "\n".join(text_parts).strip(),
            "_prompt_tokens": p_tok,
        }
        if reasoning_parts:
            out_msg["reasoning_content"] = "\n".join(reasoning_parts).strip()
        if tool_calls:
            out_msg["tool_calls"] = tool_calls
        return out_msg

    async def _chat_qwen_completions(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        caller: str,
        temperature: float,
        effort: str,
        eff_max_tokens: int,
    ) -> dict[str, Any]:
        wire_messages = self._serialize_messages_to_qwen_completions(messages)
        comp_tools = self._convert_tools_to_openai_completions(tools)
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": wire_messages,
            "max_tokens": eff_max_tokens,
        }
        if effort == "off":
            payload["enable_thinking"] = False
            payload["temperature"] = temperature
        else:
            payload["enable_thinking"] = True
            payload["reasoning_effort"] = "xhigh" if effort == "high" else effort
        if comp_tools:
            payload["tools"] = comp_tools

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        data, latency_ms = await self._post_json_with_retries(self.endpoint_url, headers, payload, caller)
        usage = data.get("usage") or {}
        p_details = usage.get("prompt_tokens_details") or {}
        p_tok = int(usage.get("prompt_tokens", 0))
        hit_tok = min(p_tok, int(p_details.get("cached_tokens", 0)))
        miss_tok = max(0, p_tok - hit_tok)
        c_tok = int(usage.get("completion_tokens", 0))

        self.metrics.append(
            LLMCallMetric(
                caller=caller,
                prompt_tokens=p_tok,
                cache_hit_tokens=hit_tok,
                cache_miss_tokens=miss_tok,
                completion_tokens=c_tok,
                latency_ms=latency_ms,
            )
        )

        choice = (data.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        out_msg: dict[str, Any] = {
            "role": "assistant",
            "content": str(msg.get("content") or "").strip(),
            "_prompt_tokens": p_tok,
        }
        if msg.get("reasoning_content"):
            out_msg["reasoning_content"] = str(msg["reasoning_content"]).strip()
        if msg.get("tool_calls"):
            out_msg["tool_calls"] = [
                {
                    "id": str(tc.get("id") or "call_0"),
                    "type": "function",
                    "function": {
                        "name": str((tc.get("function") or {}).get("name") or ""),
                        "arguments": str((tc.get("function") or {}).get("arguments") or "{}"),
                    },
                }
                for tc in msg["tool_calls"]
            ]
        return out_msg

    async def _chat_deepseek_anthropic(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        caller: str,
        temperature: float,
        effort: str,
        eff_max_tokens: int,
    ) -> dict[str, Any]:
        top_system, wire_messages = self._serialize_messages_to_anthropic(messages)
        anth_tools = self._convert_tools_to_anthropic(tools)

        payload: dict[str, Any] = {
            "model": self.model,
            "max_tokens": eff_max_tokens,
            "messages": wire_messages,
            "thinking": {"type": "disabled" if effort == "off" else "enabled"},
        }
        if effort != "off":
            payload["output_config"] = {"effort": effort}
        else:
            payload["temperature"] = temperature
        if top_system:
            payload["system"] = top_system
        if anth_tools:
            payload["tools"] = anth_tools

        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        data, latency_ms = await self._post_json_with_retries(self.endpoint_url, headers, payload, caller)
        usage = data.get("usage") or {}
        hit_tok = int(usage.get("cache_read_input_tokens", 0))
        miss_tok = int(usage.get("input_tokens", 0)) + int(usage.get("cache_creation_input_tokens", 0))
        p_tok = hit_tok + miss_tok
        c_tok = int(usage.get("output_tokens", 0))

        self.metrics.append(
            LLMCallMetric(
                caller=caller,
                prompt_tokens=p_tok,
                cache_hit_tokens=hit_tok,
                cache_miss_tokens=miss_tok,
                completion_tokens=c_tok,
                latency_ms=latency_ms,
            )
        )

        blocks = data.get("content") or []
        text_parts: list[str] = []
        reasoning_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        raw_keep: list[dict[str, Any]] = []

        for b in blocks:
            btype = b.get("type")
            if btype == "thinking":
                reasoning_parts.append(b.get("thinking", ""))
                raw_keep.append(b)
            elif btype == "text":
                if b.get("text"):
                    text_parts.append(b["text"])
                    raw_keep.append(b)
            elif btype == "tool_use":
                raw_keep.append(b)
                tool_calls.append(
                    {
                        "id": b["id"],
                        "type": "function",
                        "function": {
                            "name": b["name"],
                            "arguments": json.dumps(b.get("input") or {}, ensure_ascii=False),
                        },
                    }
                )

        out_msg: dict[str, Any] = {
            "role": "assistant",
            "content": "\n".join(text_parts).strip(),
            "_raw_anthropic_blocks": raw_keep,
            "_prompt_tokens": p_tok,
        }
        if reasoning_parts:
            out_msg["reasoning_content"] = "\n".join(reasoning_parts).strip()
        if tool_calls:
            out_msg["tool_calls"] = tool_calls
        return out_msg

    async def chat_fast_utility(
        self,
        messages: list[dict[str, Any]],
        caller: str,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        """Lightweight deterministic utility call (`thinking=disabled`, `temperature=0.0`) on the same model."""
        return await self.chat(
            messages=messages,
            tools=None,
            caller=caller,
            temperature=0.0,
            reasoning_effort_override="off",
            max_tokens_override=max_tokens,
        )

    def summary(self) -> dict[str, Any]:
        total_prompt = sum(m.prompt_tokens for m in self.metrics)
        total_hit = sum(m.cache_hit_tokens for m in self.metrics)
        total_miss = sum(m.cache_miss_tokens for m in self.metrics)
        total_comp = sum(m.completion_tokens for m in self.metrics)
        return {
            "model": self.model,
            "protocol": self.protocol,
            "reasoning_effort": self.reasoning_effort,
            "system2_calls": len(self.metrics),
            "total_prompt_tokens": total_prompt,
            "total_cache_hit_tokens": total_hit,
            "total_cache_miss_tokens": total_miss,
            "prefix_cache_hit_rate": round(total_hit / total_prompt, 4) if total_prompt else 0.0,
            "total_completion_tokens": total_comp,
            "estimated_cost_cny_upper": round(
                estimate_cost_cny_upper_bound(total_hit, total_miss, total_comp, model=self.model), 4
            ),
            "total_latency_ms": round(sum(m.latency_ms for m in self.metrics), 1),
        }
