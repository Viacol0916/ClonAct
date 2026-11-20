"""System-1 Primitive Judgment Engine (`System1Judge`).

Primary Backend: Official TypeSafe Python SDK (`AsyncTypeSafeClient`, `jev-latest`)
hitting `POST https://api.typesafe.ai/v1/systemone` with parallel `Noul` and `Choice` primitives.

Verified Fallback Backend: Parallel single-token logprob probe (`max_tokens=1, logprobs=True`)
on DeepSeek API when `TYPESAFE_API_KEY` fails authentication or is offline—always recording
`backend_used` transparently so fallback runs are never misreported as live Jev runs.
"""
from __future__ import annotations

import asyncio
import json
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
from typesafe_sdk import AsyncTypeSafeClient, Choice, Noul, NoulCriteria
from typesafe_sdk._core.errors import TypeSafeError


@dataclass
class NoulSpec:
    instructions: str
    true_desc: str | None = None
    false_desc: str | None = None


@dataclass
class ChoiceSpec:
    instructions: str
    criteria: dict[str, str]


QuestionSpec = NoulSpec | ChoiceSpec


@dataclass
class JudgmentResult:
    answers: dict[str, dict[str, Any]]
    latency_ms: float
    backend_used: str  # "typesafe_jev" | "deepseek_logprob_fallback"
    input_tokens: int = 0
    output_tokens: int = 0
    fallback_reason: str | None = None
    cache_hit_tokens: int = 0
    cache_miss_tokens: int = 0
    resolved_model: str | None = None


def load_project_env(env_path: Path | None = None) -> dict[str, str]:
    if env_path is None:
        env_path = Path(__file__).resolve().parent.parent / ".env"
    loaded: dict[str, str] = {}
    if env_path.exists():
        for raw in env_path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            loaded[k.strip()] = v.strip()
            os.environ.setdefault(k.strip(), v.strip())
    return loaded


class System1Judge:
    """Unified System-1 non-generative primitive evaluator."""

    def __init__(
        self,
        typesafe_api_key: str | None = None,
        deepseek_api_key: str | None = None,
        jev_model: str = "jev-latest",
    ) -> None:
        load_project_env()
        self.typesafe_api_key = typesafe_api_key or os.environ.get("TYPESAFE_API_KEY", "")
        self.deepseek_api_key = deepseek_api_key or os.environ.get("DEEPSEEK_API_KEY", "")
        self.jev_model = jev_model
        self.force_logprob_mmu: bool = (
            os.environ.get("CLONACT_FORCE_LOGPROB_MMU", "").strip().lower()
            in ("1", "true", "yes", "on")
        )
        self.logprob_model: str = (
            os.environ.get("CLONACT_LOGPROB_MODEL", "").strip() or "deepseek-chat"
        )
        self._jev_disabled_reason: str | None = None
        self._ts_client: AsyncTypeSafeClient | None = None
        self.call_history: list[dict[str, Any]] = []

    def _get_ts_client(self) -> AsyncTypeSafeClient:
        if self._ts_client is None:
            self._ts_client = AsyncTypeSafeClient(api_key=self.typesafe_api_key, model=self.jev_model)
        return self._ts_client

    async def aclose(self) -> None:
        if self._ts_client is not None:
            await self._ts_client.aclose()
            self._ts_client = None

    async def decide(
        self,
        state: dict[str, Any] | str,
        questions: dict[str, QuestionSpec],
        tag: str = "system1",
    ) -> JudgmentResult:
        """Evaluate multiple independent Noul/Choice questions in parallel over `state`."""
        t0 = time.perf_counter()

        force_logprob = self.force_logprob_mmu or (
            os.environ.get("CLONACT_FORCE_LOGPROB_MMU", "").strip().lower()
            in ("1", "true", "yes", "on")
        )
        if force_logprob:
            res = await self._call_deepseek_logprob_fallback(
                state,
                questions,
                fallback_reason="forced_same_weight_logprob_mmu",
            )
            self._record(tag, state, questions, res)
            return res

        if self.typesafe_api_key and self._jev_disabled_reason is None:
            try:
                res = await self._call_typesafe_jev(state, questions, t0)
                self._record(tag, state, questions, res)
                return res
            except TypeSafeError as exc:
                # Cache 401 auth failure so subsequent turns don't pay redundant failed RTTs
                err_msg = str(exc)
                if "401" in err_msg or "Cannot authenticate" in err_msg:
                    self._jev_disabled_reason = f"TypeSafe 401 Auth Error: {err_msg}"
                fallback_reason = err_msg
            except Exception as exc:
                fallback_reason = f"{type(exc).__name__}: {exc}"
        else:
            fallback_reason = self._jev_disabled_reason or "Missing TYPESAFE_API_KEY"

        res = await self._call_deepseek_logprob_fallback(state, questions, fallback_reason)
        self._record(tag, state, questions, res)
        return res

    async def _call_typesafe_jev(
        self,
        state: dict[str, Any] | str,
        questions: dict[str, QuestionSpec],
        t0: float,
    ) -> JudgmentResult:
        sdk_questions: dict[str, Any] = {}
        for qid, spec in questions.items():
            if isinstance(spec, NoulSpec):
                crit = (
                    NoulCriteria(true=spec.true_desc, false=spec.false_desc)
                    if (spec.true_desc and spec.false_desc)
                    else None
                )
                sdk_questions[qid] = (
                    Noul(instructions=spec.instructions, criteria=crit)
                    if crit
                    else Noul(instructions=spec.instructions)
                )
            elif isinstance(spec, ChoiceSpec):
                sdk_questions[qid] = Choice(instructions=spec.instructions, criteria=spec.criteria)

        client = self._get_ts_client()
        resp = await client.system_one(state=state, questions=sdk_questions)

        latency_ms = (time.perf_counter() - t0) * 1000.0
        parsed: dict[str, dict[str, Any]] = {}
        for qid, ans in resp.answers.items():
            if getattr(ans, "type", None) == "noul" or hasattr(ans, "noul"):
                parsed[qid] = {"type": "noul", "noul": float(ans.noul)}
            else:
                parsed[qid] = {
                    "type": "choice",
                    "choice": str(ans.choice),
                    "confidence": float(ans.confidence or 0.0),
                    "probabilities": dict(ans.probabilities or {}),
                }

        in_tok = getattr(resp.usage, "input_tokens", 0) if resp.usage else 0
        out_tok = getattr(resp.usage, "output_tokens", 0) if resp.usage else 0
        return JudgmentResult(
            answers=parsed,
            latency_ms=latency_ms,
            backend_used=f"typesafe_jev ({resp.model})",
            input_tokens=in_tok,
            output_tokens=out_tok,
        )

    async def _call_deepseek_logprob_fallback(
        self,
        state: dict[str, Any] | str,
        questions: dict[str, QuestionSpec],
        fallback_reason: str,
    ) -> JudgmentResult:
        """Parallel single-token (`max_tokens=1, logprobs=True`) System-1 probability probe."""
        if not self.deepseek_api_key:
            raise RuntimeError(f"Neither TypeSafe Jev ({fallback_reason}) nor DEEPSEEK_API_KEY is available.")

        t0 = time.perf_counter()
        state_json = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False, indent=2)

        async with httpx.AsyncClient(timeout=30.0) as client:
            tasks = {
                qid: self._probe_single_question(client, state_json, spec)
                for qid, spec in questions.items()
            }
            keys = list(tasks.keys())
            results = await asyncio.gather(*(tasks[k] for k in keys))

        answers: dict[str, dict[str, Any]] = {}
        in_tok = 0
        out_tok = 0
        hit_tok = 0
        miss_tok = 0
        resolved_model: str | None = None
        for k, (ans, u_in, u_out, u_hit, u_miss, r_model) in zip(keys, results):
            answers[k] = ans
            in_tok += u_in
            out_tok += u_out
            hit_tok += u_hit
            miss_tok += u_miss
            if r_model and resolved_model is None:
                resolved_model = r_model

        return JudgmentResult(
            answers=answers,
            latency_ms=(time.perf_counter() - t0) * 1000.0,
            backend_used="deepseek_logprob_system1_fallback",
            input_tokens=in_tok,
            output_tokens=out_tok,
            fallback_reason=fallback_reason,
            cache_hit_tokens=hit_tok,
            cache_miss_tokens=miss_tok,
            resolved_model=resolved_model,
        )

    async def _post_logprob_with_retry(
        self,
        client: httpx.AsyncClient,
        headers: dict[str, str],
        payload: dict[str, Any],
        max_attempts: int = 5,
    ) -> dict[str, Any]:
        last_exc: Exception | None = None
        for attempt in range(max_attempts):
            try:
                resp = await client.post(
                    "https://api.deepseek.com/chat/completions",
                    headers=headers,
                    json=payload,
                )
                if resp.status_code in (429, 500, 502, 503, 504):
                    last_exc = httpx.HTTPStatusError(
                        f"Transient HTTP {resp.status_code} from DeepSeek logprob endpoint",
                        request=resp.request,
                        response=resp,
                    )
                    if attempt + 1 < max_attempts:
                        await asyncio.sleep(min(16.0, 0.75 * (2 ** attempt)))
                        continue
                    resp.raise_for_status()
                resp.raise_for_status()
                return resp.json()
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_exc = exc
                if attempt + 1 < max_attempts:
                    await asyncio.sleep(min(16.0, 0.75 * (2 ** attempt)))
                    continue
                raise
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("DeepSeek logprob probe failed without response")

    @staticmethod
    def _extract_usage_tokens(data: dict[str, Any]) -> tuple[int, int, int, int]:
        usage = data.get("usage") or {}
        prompt_tok = int(usage.get("prompt_tokens", 0) or 0)
        comp_tok = int(usage.get("completion_tokens", 0) or 0)
        raw_hit = usage.get("prompt_cache_hit_tokens")
        if raw_hit is None:
            raw_hit = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
        hit_tok = int(raw_hit or 0)
        raw_miss = usage.get("prompt_cache_miss_tokens")
        if raw_miss is None:
            miss_tok = max(0, prompt_tok - hit_tok)
        else:
            miss_tok = int(raw_miss or 0)
        return prompt_tok, comp_tok, hit_tok, miss_tok

    async def _probe_single_question(
        self,
        client: httpx.AsyncClient,
        state_json: str,
        spec: QuestionSpec,
    ) -> tuple[dict[str, Any], int, int, int, int, str]:
        headers = {
            "Authorization": f"Bearer {self.deepseek_api_key}",
            "Content-Type": "application/json",
        }
        model_name = os.environ.get("CLONACT_LOGPROB_MODEL", "").strip() or self.logprob_model or "deepseek-chat"
        if isinstance(spec, NoulSpec):
            crit_block = ""
            if spec.true_desc or spec.false_desc:
                crit_block = f"\nYES means: {spec.true_desc or ''}\nNO means: {spec.false_desc or ''}"
            prompt = (
                f"STATE:\n{state_json}\n\n"
                f"QUESTION: {spec.instructions}{crit_block}\n"
                "Answer with a single character ONLY: Y (yes/true) or N (no/false)."
            )
            payload = {
                "model": model_name,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 1,
                "temperature": 0.0,
                "logprobs": True,
                "top_logprobs": 5,
            }
            data = await self._post_logprob_with_retry(client, headers, payload)
            p_yes = self._extract_yes_prob(data["choices"][0])
            u_in, u_out, u_hit, u_miss = self._extract_usage_tokens(data)
            return (
                {"type": "noul", "noul": round(p_yes, 4)},
                u_in,
                u_out,
                u_hit,
                u_miss,
                str(data.get("model") or model_name),
            )

        # ChoiceSpec: map options to single uppercase letters A, B, C... for exact 1-token logprobs
        opt_keys = list(spec.criteria.keys())
        letters = [chr(ord("A") + i) for i in range(len(opt_keys))]
        letter_to_key = dict(zip(letters, opt_keys))
        options_str = "\n".join(
            f"{ltr}: ({key}) {spec.criteria[key]}" for ltr, key in zip(letters, opt_keys)
        )
        prompt = (
            f"STATE:\n{state_json}\n\n"
            f"QUESTION: {spec.instructions}\nOPTIONS:\n{options_str}\n"
            f"Reply with ONLY the single option letter ({'/'.join(letters)})."
        )
        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1,
            "temperature": 0.0,
            "logprobs": True,
            "top_logprobs": min(20, max(5, len(letters) + 2)),
        }
        data = await self._post_logprob_with_retry(client, headers, payload)
        probs_by_key, winner, confidence = self._extract_choice_probs(data["choices"][0], letter_to_key)
        u_in, u_out, u_hit, u_miss = self._extract_usage_tokens(data)
        return (
            {
                "type": "choice",
                "choice": winner,
                "confidence": round(confidence, 4),
                "probabilities": {k: round(v, 4) for k, v in probs_by_key.items()},
            },
            u_in,
            u_out,
            u_hit,
            u_miss,
            str(data.get("model") or model_name),
        )

    @staticmethod
    def _extract_yes_prob(choice_obj: dict[str, Any]) -> float:
        lp_items = (
            choice_obj.get("logprobs", {}).get("content", [{}])[0].get("top_logprobs", [])
            if choice_obj.get("logprobs") and choice_obj["logprobs"].get("content")
            else []
        )
        yes_mass = 0.0
        no_mass = 0.0
        for item in lp_items:
            tok = item.get("token", "").strip().upper()
            p = math.exp(float(item.get("logprob", -20.0)))
            if tok in ("Y", "YES", "TRUE", "是"):
                yes_mass += p
            elif tok in ("N", "NO", "FALSE", "否"):
                no_mass += p
        if yes_mass + no_mass > 1e-9:
            return yes_mass / (yes_mass + no_mass)
        raw = choice_obj.get("message", {}).get("content", "").strip().upper()
        return 0.92 if raw.startswith("Y") else 0.08

    @staticmethod
    def _extract_choice_probs(
        choice_obj: dict[str, Any],
        letter_to_key: dict[str, str],
    ) -> tuple[dict[str, float], str, float]:
        lp_items = (
            choice_obj.get("logprobs", {}).get("content", [{}])[0].get("top_logprobs", [])
            if choice_obj.get("logprobs") and choice_obj["logprobs"].get("content")
            else []
        )
        mass: dict[str, float] = {k: 1e-6 for k in letter_to_key.values()}
        for item in lp_items:
            tok = item.get("token", "").strip().upper()
            if tok in letter_to_key:
                mass[letter_to_key[tok]] += math.exp(float(item.get("logprob", -20.0)))
        total = sum(mass.values())
        probs = {k: v / total for k, v in mass.items()}
        winner = max(probs, key=lambda k: probs[k])
        n = len(probs)
        if n <= 1:
            conf = 1.0
        else:
            # Normalized entropy concentration: 1.0 = point mass, 0.0 = uniform split
            entropy = -sum(p * math.log(max(p, 1e-12)) for p in probs.values())
            conf = max(0.0, min(1.0, 1.0 - entropy / math.log(n)))
        return probs, winner, conf

    def _record(
        self,
        tag: str,
        state: dict[str, Any] | str,
        questions: dict[str, QuestionSpec],
        res: JudgmentResult,
    ) -> None:
        in_tok = int(res.input_tokens or 0)
        out_tok = int(res.output_tokens or 0)
        if in_tok == 0:
            raw_state = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
            raw_q = json.dumps(
                {k: getattr(v, "instructions", "") for k, v in questions.items()},
                ensure_ascii=False,
            )
            in_tok = max(1, (len(raw_state) + len(raw_q)) // 4)
        if out_tok == 0 and questions:
            out_tok = len(questions)
        entry: dict[str, Any] = {
            "tag": tag,
            "latency_ms": round(res.latency_ms, 1),
            "backend": res.backend_used,
            "input_tokens": in_tok,
            "output_tokens": out_tok,
            "cache_hit_tokens": int(res.cache_hit_tokens or 0),
            "cache_miss_tokens": int(
                res.cache_miss_tokens
                if (res.cache_miss_tokens or res.cache_hit_tokens)
                else max(0, in_tok - int(res.cache_hit_tokens or 0))
            ),
            "question_ids": list(questions.keys()),
            "answers": res.answers,
            "fallback_reason": res.fallback_reason,
        }
        if res.resolved_model:
            entry["model"] = res.resolved_model
        self.call_history.append(entry)


class GenerativeLLMJudge(System1Judge):
    """Single-variable ablation backend (`ClonAct w/ Generative LLM MMU`).

    Replaces System-1 (`Jev`) single-pass primitive scoring inside `ContextMMU` with a homogeneous
    structured JSON generation call (`deepseek-flash`, `effort="off"`, `temperature=0.0`), while
    returning the exact `JudgmentResult` schema (`noul` float / `choice` key + `confidence`).
    """

    def __init__(
        self,
        *,
        model: str | None = None,
        api_key: str | None = None,
    ) -> None:
        super().__init__(deepseek_api_key=api_key)
        from clonact.llm_client import HomogeneousLLMClient

        resolved_model = (
            model
            or os.environ.get("PA_LLM_MODEL")
            or os.environ.get("DSH_EXECUTOR_MODEL")
            or "deepseek-flash"
        ).strip() or "deepseek-flash"
        self.llm = HomogeneousLLMClient(
            api_key=api_key or self.deepseek_api_key,
            model=resolved_model,
            reasoning_effort="off",
            max_tokens=1024,
        )
        self.total_cache_hit_tokens: int = 0
        self.total_cache_miss_tokens: int = 0

    @staticmethod
    def _extract_json_object(raw_text: str) -> dict[str, Any]:
        txt = (raw_text or "").strip()
        if txt.startswith("```"):
            lines = txt.splitlines()
            if lines:
                lines = lines[1:]
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            txt = "\n".join(lines).strip()
        try:
            obj = json.loads(txt)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
        start = txt.find("{")
        end = txt.rfind("}")
        if start != -1 and end > start:
            try:
                obj = json.loads(txt[start : end + 1])
                if isinstance(obj, dict):
                    return obj
            except Exception:
                pass
        return {}

    @staticmethod
    def _coerce_answers(
        parsed: dict[str, Any],
        questions: dict[str, QuestionSpec],
    ) -> dict[str, dict[str, Any]]:
        answers: dict[str, dict[str, Any]] = {}
        for qid, spec in questions.items():
            raw_val = parsed.get(qid)
            if isinstance(spec, NoulSpec):
                prob = 0.5
                if isinstance(raw_val, bool):
                    prob = 0.95 if raw_val else 0.05
                elif isinstance(raw_val, (int, float)):
                    prob = float(raw_val)
                elif isinstance(raw_val, dict):
                    if isinstance(raw_val.get("noul"), (int, float)):
                        prob = float(raw_val["noul"])
                    elif isinstance(raw_val.get("probability"), (int, float)):
                        prob = float(raw_val["probability"])
                    elif isinstance(raw_val.get("answer"), bool):
                        prob = 0.95 if raw_val["answer"] else 0.05
                elif isinstance(raw_val, str):
                    s_low = raw_val.strip().lower()
                    if s_low in ("true", "yes", "y", "1"):
                        prob = 0.95
                    elif s_low in ("false", "no", "n", "0"):
                        prob = 0.05
                    else:
                        try:
                            prob = float(s_low)
                        except ValueError:
                            prob = 0.5
                prob = max(0.0, min(1.0, prob))
                answers[qid] = {"type": "noul", "noul": round(prob, 4)}
            else:
                opt_keys = list(spec.criteria.keys())
                chosen = opt_keys[0] if opt_keys else ""
                conf = 0.75
                cand_str = ""
                if isinstance(raw_val, str):
                    cand_str = raw_val.strip()
                elif isinstance(raw_val, dict):
                    cand_str = str(raw_val.get("choice") or raw_val.get("option") or "").strip()
                    if isinstance(raw_val.get("confidence"), (int, float)):
                        conf = max(0.0, min(1.0, float(raw_val["confidence"])))
                if cand_str in spec.criteria:
                    chosen = cand_str
                elif cand_str:
                    lower_map = {k.lower(): k for k in opt_keys}
                    if cand_str.lower() in lower_map:
                        chosen = lower_map[cand_str.lower()]
                rem = max(1, len(opt_keys) - 1)
                probs = {
                    k: round(conf if k == chosen else (1.0 - conf) / rem, 4)
                    for k in opt_keys
                }
                answers[qid] = {
                    "type": "choice",
                    "choice": chosen,
                    "confidence": round(conf, 4),
                    "probabilities": probs,
                }
        return answers

    async def decide(
        self,
        state: dict[str, Any] | str,
        questions: dict[str, QuestionSpec],
        tag: str = "unnamed",
    ) -> JudgmentResult:
        if not questions:
            return JudgmentResult(
                answers={},
                latency_ms=0.0,
                backend_used=f"generative_llm_mmu ({self.llm.model})",
            )

        state_str = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False, indent=2)
        q_schema: dict[str, Any] = {}
        for qid, spec in questions.items():
            if isinstance(spec, NoulSpec):
                q_schema[qid] = {
                    "kind": "boolean_probability",
                    "instructions": spec.instructions,
                    "true_meaning": spec.true_desc or "true / yes",
                    "false_meaning": spec.false_desc or "false / no",
                    "output_format": {"noul": "<float in [0.0, 1.0]>"},
                }
            else:
                q_schema[qid] = {
                    "kind": "categorical_choice",
                    "instructions": spec.instructions,
                    "valid_options": spec.criteria,
                    "output_format": {
                        "choice": f"<exact key from {list(spec.criteria.keys())}>",
                        "confidence": "<float in [0.0, 1.0]>",
                    },
                }

        sys_prompt = (
            "You are a deterministic structured routing and context-management classifier.\n"
            "Evaluate EVERY question in `QUESTIONS` against `STATE` and output ONLY a single valid JSON object "
            "mapping each question key to its requested `output_format` (no prose, no markdown commentary)."
        )
        user_prompt = (
            f"STATE:\n{state_str}\n\n"
            f"QUESTIONS:\n{json.dumps(q_schema, ensure_ascii=False, indent=2)}\n\n"
            "Return ONLY the JSON object."
        )

        t0 = time.perf_counter()
        prev_metric_len = len(self.llm.metrics)
        resp = await self.llm.chat(
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt},
            ],
            tools=None,
            caller=f"llm_mmu:{tag}",
            temperature=0.0,
            reasoning_effort_override="off",
            max_tokens_override=1024,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        in_tok = 0
        out_tok = 0
        for m in self.llm.metrics[prev_metric_len:]:
            in_tok += m.prompt_tokens
            out_tok += m.completion_tokens
            self.total_cache_hit_tokens += m.cache_hit_tokens
            self.total_cache_miss_tokens += m.cache_miss_tokens

        parsed = self._extract_json_object(str(resp.get("content") or ""))
        answers = self._coerce_answers(parsed, questions)
        res = JudgmentResult(
            answers=answers,
            latency_ms=elapsed_ms,
            backend_used=f"generative_llm_mmu ({self.llm.model})",
            input_tokens=in_tok,
            output_tokens=out_tok,
        )
        self._record(tag, state, questions, res)
        return res
