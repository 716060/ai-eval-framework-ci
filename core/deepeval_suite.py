"""通用指标套件（DeepEval）：幻觉 / 答案相关性 / 重忠实度。单条失败不影响整体评测。

修复点：
1、缺 context 时不再拿 expected_output 冒充事实源（会把"表述不用"误判为幻觉）；
2、无 context 的用例标记 not_applicable，而不是给个假分；
3、judge_client 可注入，便于离线测试；async_mode=False 配合同步客户端
"""

from __future__ import annotations

import json

from core.contract import ChatModel
from core.metrics import MetricResult
from utils.logger import get_logger

from core.judge_model import OwnLLMJudgeModel, DEEPEVAL_AVAILABLE

logger = get_logger("deepeval_suite")

if DEEPEVAL_AVAILABLE:
    from deepeval.metrics import (HallucinationMetric, AnswerRelevancyMetric,
                                  FaithfulnessMetric)
    from deepeval.test_case import LLMTestCase



SEMANTIC_METRICS = ("hallucination_rate", "answer_relevancy", "faithfulness")
# 需要 context 才能算的指标；AnswerRelevancy 只看 input / actual_output
CONTEXT_DEPENDENT = ("hallucination_rate", "faithfulness")

class DeepEvalSuite:
    def __init__(self, judge_provider: str = "qwen", threshold: float = 0.7,
                 judge_client: ChatModel | None = None,
                 judge_model_name: str | None = None,
                 async_mode: bool = False):
        if not DEEPEVAL_AVAILABLE:
            raise RuntimeError("deepeval 未安装，无法启用通用指标套件")
        self.judge = OwnLLMJudgeModel(provider=judge_provider, client=judge_client)
        self.judge_name = judge_model_name or self.judge.get_model_name()
        self.threshold = threshold
        self.async_mode = async_mode

    @staticmethod
    def _expected_text(record) -> str:
        return (record.expected_output if isinstance(record.expected_output, str)
                else json.dumps(record.expected_output, ensure_ascii=False))

    @classmethod
    def _build_case(cls, record, actual_output: str) -> "LLMTestCase":
        # context = record.context or [json.dumps(record.expected_output, ensure_ascii=False)]
        # expected = (record.expected_output if isinstance(record.expected_output, str)
        #             else json.dumps(record.expected_output, ensure_ascii=False))
        # return LLMTestCase(
        #     input=record.input,
        #     actual_output=actual_output,
        #     expected_output=expected,
        #     context=context,                    # HallucinationMetric 的事实源
        #     retrieval_context=context,          # FaithfulnessMetric 的检索上下文
        # )
        return LLMTestCase(
            input=record.input,
            actual_output=actual_output,
            expected_output=cls._expected_text(record),
            context=list(record.context) or None,       # 事实源之来自数据集声明
            retrieval_context=list(record.context) or None,
            tags=list(record.tags) or None,
            name=record.id,
        )

    def evaluate(self, record, actual_output: str) -> dict[str, MetricResult]:
        if not actual_output.strip():
            return {name: self._not_applicable(name, "输出为空，语义指标不适用")
                    for name in SEMANTIC_METRICS}

        has_context = bool(record.context)
        case = self._build_case(record, actual_output)
        out: dict[str, MetricResult] = {}

        if has_context:
            out["hallucination_rate"] = self._run(
                "hallucination_rate", HallucinationMetric, case, higher_is_better=False)
            out["faithfulness"] = self._run(
                "faithfulness", FaithfulnessMetric, case)
        else:
            # 明确标记不适用，聚合层会把它排除在分母外，而不是塞个假分进来
            for name in CONTEXT_DEPENDENT:
                out[name] = self._not_applicable(name, "用例未声明 context，该指标不适用")

        out["answer_relevancy"] = self._run(
            "answer_relevancy",AnswerRelevancyMetric, case)

        return out

    def _run(self, name: str, metric_cls, case, higher_is_better: bool = True) -> MetricResult:
        try:
            metric = metric_cls(model=self.judge, threshold=self.threshold,
                                include_reason=True, async_mode=self.async_mode)
            metric.measure(case)
            return MetricResult(
                name=name, score=round(float(metric.score), 4), threshold=self.threshold,
                higher_is_better=higher_is_better,reason=getattr(metric, "reason", "") or "",
                detail={"judge": self.judge_name})
        except Exception as exc:        # noqa: BLE001  单指标失败降级为 error，由覆盖率门禁兜底
            logger.warning("DeepEval 指标 [%s] 计算失败", name, exc)
            return MetricResult(name=name, score=None, threshold=self.threshold,
                                higher_is_better=higher_is_better, status="error",
                                reason=f"metric_error: {exc}")

    @staticmethod
    def _not_applicable(name: str, reason: str) -> MetricResult:
        return MetricResult(name=name, score=None, threshold=0.0, status="not_applicable",
                            reason=reason)
