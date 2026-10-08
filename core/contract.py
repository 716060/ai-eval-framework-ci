"""模型调用契约：让 evaluator 依赖抽象而非具体实现，从而可以注入假模型做离线测试。"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


class LLMClientError(RuntimeError):
    """模型调用最终失败"""

class InferenceError(LLMClientError):
    """单条用例推理失败（基础设施层面），不计入模型质量。"""


@dataclass
class LLMResponse:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0
    attempts: int = 1
    cached: bool = False
    meta: dict = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

@runtime_checkable
class ChatModel(Protocol):
    provider: str
    model_name: str

    def chat(self, prompt: str, system_prompt: str | None = None,
             json_model: bool = False, temperature: float = 0.0,
             max_tokens: int | None  = None) -> LLMResponse: ...

    def generate(self, prompt: str, system_prompt: str | None = None,
                 json_model: bool = False, temperature: float = 0.0,
                 seed: int | None = None) -> str: ...