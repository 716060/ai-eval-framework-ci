"""
    统一 LLM 调用层：兼容 DeepSeek / Qwen（OpenAI 协议）

    修复点:
    1、只对瞬时错误重试（限流/连接超时/5xx），4xx参数错误立即失败；
    2、尊重 Retry-After，退避带抖动，避免并发场景同步重试放大；
    3、返回 token / 延迟 / 重试次数，共成本与延迟指标适用；
    4、支持请求级缓存与异步调用（避免 Judge 阻塞事件循环）；
"""

from __future__ import annotations

import time

import openai
from openai import AsyncOpenAI, OpenAI

from config.settings import MAX_RETRIES, REQUEST_TIMEOUT, SUPPORTED_ENDPOINTS
from core.cache import DiskCache
from core.contract import InferenceError, LLMClientError, LLMResponse
from utils.logger import get_logger

logger = get_logger("llm_client")

# 仅这些异常值得重试，其余 4xx 属于配置或参数错误，重试无意义。
TRANSIENT_ERRORS = (
    openai.RateLimitError,
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.InternalServerError,
)

class LLMClient:
    def __init__(self, provider: str, cache: DiskCache | None = None,
                 seed: int | None = None):
        if provider not in SUPPORTED_ENDPOINTS:
            raise LLMClientError(f"不支持的模型 provider: {provider}")
        ep = SUPPORTED_ENDPOINTS[provider]
        if not ep.api_key:
            raise LLMClientError(f"缺少 {provider} 的 API key，请检查 .env 配置")

        self.provider = provider
        self.model_name = ep.model
        self.seed = seed
        self._cache = cache
        self._sync = OpenAI(api_key=ep.api_key, base_url=ep.base_url, timeout=REQUEST_TIMEOUT)
        self._async = AsyncOpenAI(api_key=ep.api_key, base_url=ep.base_url, timeout=REQUEST_TIMEOUT)

    # 缓存键必须包含所有影响输出的因素
    def _fingerprint(self, prompt: str, system_prompt: str | None,
                     json_model: bool, temperature: float) -> str:
        return DiskCache.fingerprint(provider=self.provider, model=self.model_name,
                                     prompt=prompt, system=system_prompt,
                                     json_model=json_model, temperature=temperature,
                                     seed=self.seed)

    def _kwargs(self, prompt: str, system_prompt: str,
                json_model: bool, temperature: float, max_tokens: int | None) -> dict:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        kwargs: dict = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
        }
        if json_model:
            # OpenAI 兼容协议要求 prompt 中出现 "json" 字样，否则部分网关直接报错。
            kwargs["response_format"] = {"type": "json_object"}
        if max_tokens:
            kwargs["max_tokens"] = max_tokens
        if self.seed is not None:
            kwargs["seed"] = self.seed
        return kwargs

    def chat(self, prompt: str, system_prompt: str | None = None, json_model: bool = False,
             temperature: float = 0.0, max_tokens: int | None = None) -> LLMResponse:
        key = self._fingerprint(prompt, system_prompt, json_model, temperature)
        if self._cache is not None and not self._cache.refresh:
            hit = self._cache.get(key)
            if hit is not None:
                return LLMResponse(**hit, cached=True)

        started = time.perf_counter()
        last_err: Exception | None = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = self._sync.chat.completions.create(
                    **self._kwargs(prompt, system_prompt, json_model, temperature, max_tokens))
                usage = getattr(resp, "usage", None)
                out = LLMResponse(
                    text=resp.choices[0].message.content or "",
                    prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                    completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
                    latency_s=round(time.perf_counter() - started, 3),
                    attempts=attempt,
                )
                if self._cache is not None:
                    self._cache.put(key, {"text": out.text,
                                          "prompt_tokens": out.prompt_tokens,
                                          "completion_tokens": out.completion_tokens,
                                          "latency_s": out.latency_s})
                return out
            except openai.BadRequestError as exc:
                # 模型名/参数错误重试 3 次只会浪费配额，直接失败并暴露真是原因
                raise LLMClientError(f"请求参数或模型配置错误（重试无意义）：{exc}") from exc
            except openai.AuthenticationError as exc:
                raise LLMClientError(f"鉴权失败，请检查 API Key：{exc}") from exc
            except TRANSIENT_ERRORS as exc:
                last_err = exc
                delay = self._backoff(attempt, exc)
                logger.warning("[%s] 瞬时错误 %d%d，%.1fs 后重试：%s",
                               self.provider, attempt, MAX_RETRIES, delay, exc)
                if attempt < MAX_RETRIES:
                    time.sleep(delay)
            except Exception as exc:  # noqa: BLE001 未知异常不重试，避免掩盖 bug
                raise  LLMClientError(f"未预期的调用错误：{exc}") from exc
        raise InferenceError(f"{self.provider} 连续 {MAX_RETRIES} 次瞬时失败：{last_err}")

    @staticmethod
    def _backoff(attempt: int, exc: Exception) -> float:
        """指数退避 + 抖动；优先尊重服务器 Retry-After。"""
        import random
        resp = getattr(exc, "response", None)
        retry_after = None
        if resp is not None:
            raw = resp.headers.get("retry-after") if hasattr(resp, "headers") else None
            if raw:
                try:
                    retry_after = float(raw)
                except ValueError:
                    retry_after = None
        base = retry_after if retry_after is not None else min(2 ** attempt, 8)
        return base + random.uniform(0, 0.5)

    async def achat(self, prompt: str, system_prompt: str | None = None,
                    json_model: bool = False, temperature: float = 0.0) -> LLMResponse:
        """异步版本，供 DeepEval 的a_generate 使用，避免在事件循环里阻塞 HTTP。"""
        started = time.perf_counter()
        resp = await self._async.chat.completions.create(
            **self._kwargs(prompt, system_prompt, json_model, temperature, None))
        usage = getattr(resp, "usage", None)
        return LLMResponse(
            text=resp.choices[0].message.content or "",
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            latency_s=round(time.perf_counter() - started, 3),
        )

    # 兼容既有调用方
    def generate(self, prompt: str, system_prompt: str | None = None, json_model: bool = False,
                 temperature: float = 0.0, seed: int | None = None) -> str:
        return self.chat(prompt, system_prompt, json_model, temperature).text

def build_client(provider: str, **kwargs) -> LLMClient:
    return LLMClient(provider, **kwargs)







