"""把 DeepSeek/Qwen 适配为 DeepEval 的 Judge 模型（LLM-as-a-Judge 走自有API，零额外成本）"""

import json
import re
from typing import List

from core.llm_client import LLMClient
from core.contract import ChatModel
from utils.logger import get_logger

logger = get_logger(__name__)

try:
    from deepeval.models.base_model import DeepEvalBaseLLM
    DEEPEVAL_AVAILABLE = True
except ImportError:
    DEEPEVAL_AVAILABLE = False
    DeepEvalBaseLLM = object    #未安装 deepeval 时退化未普通类，保证导入不炸

class JudgeFormatError(RuntimeError):
    """Judge 输出无法解析为要求的 JSON。暴露给覆盖率门禁，不静默降级。"""

class OwnLLMJudgeModel(DeepEvalBaseLLM):
    """DeepEval 自定义模型契约：generate / a_generate / get_model_name"""

    def __init__(self, provider: str = "qwen", client: ChatModel | None = None):
        self._client = client or LLMClient(provider)
        self.provider = provider
        if DEEPEVAL_AVAILABLE:
            super().__init__(model=f"{provider}:{self._client.model_name}")

    def load_model(self, **kwargs):
        return self._client

    def generate(self, prompt: str, **kwargs) -> str:
        schema = kwargs.get("schema")
        if schema is not None:
            prompt = prompt + self._schema_instruction(schema)
        raw = self._client.generate(prompt, temperature=0.0)
        return self._extract_json(raw) if schema is not None else raw

    async def a_generate(self, prompt: str, **kwargs) -> str:
        """真正的异步：避免在 DeepEval 的事件循环里做阻塞 HTTP 调用。"""
        schema = kwargs.get("schema")
        if schema is not None:
            prompt = prompt + self._schema_instruction(schema)
        if hasattr(self._client, "achat"):
            resp = await self._client.achat(prompt, temperature=0.0)
            raw = resp.text
        else:
            raw = self._client.generate(prompt, temperature=0.0)
        return self._extract_json(raw) if schema is not None else raw

    def batch_generate(self, prompts, **kwargs):
        """DeepEval 部分指标会走批量接口，原实现直接 NotImplementedError。"""
        return [self.generate(p, **kwargs) for p in prompts]


    def get_model_name(self) -> str:
        return f"own-judge/{self._client.provider}/{self._client.model_name}"

    @staticmethod
    def _schema_instruction(schema) -> str:
        """DeepEval 内部会传 pydantic schema，这里转成强制 JSON 指令"""
        try:
            if hasattr(schema, "model_json_schema"):
                json_schema = schema.model_json_schema()
            elif hasattr(schema, "schema"):
                json_schema = schema.schema()
            else:
                json_schema = {"description": str(schema)}
        except Exception:   # noqa: BLE001
            json_schema = {"description": str(schema)}

        return ("\n\n# 输出格式强制要求\n仅输出一个合法 JSON 对象(不要markdown代码块、不要解释),"
                "必须满足以下 JSON Schema:\n" + json.dumps(json_schema, ensure_ascii=False))

    @staticmethod
    def _extract_json(raw: str) -> str:
        """从 Judge 输出中稳健抽取 JSON 片段"""
        raw = raw.strip()
        try:
            json.loads(raw)
            return raw
        except Exception:   #noqa: BLE001
            pass
        fence = re.search(r"```(?:json)?\s*(.*?)```", raw, re.S)
        if fence:
            raw = fence.group(1).strip()

        for lo, hi in (("{", "}"), ("[", "]")):
            start, end = raw.find(lo), raw.find(hi)
            if start != -1 and end > start:
                candidate = raw[start:end + 1]
                try:
                    json.loads(candidate)
                    return candidate
                except Exception: # noqa: BLE001
                    continue
        raise JudgeFormatError(f"Judge 输出无法解析为 JSON：{raw[:200]}")