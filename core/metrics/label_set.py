"""
手写指标：分类/枚举任务的标签合法性。
与 ExactMatchMetric 的分工：
- label_validity：默契维度 —— 输出是否落在允许标签集合内（抓”写了一整段解释“）
- exact_match：正确性维度 —— 输出的标签是否正确
两者同时为 0 指向 "没按格式做答"，只有 exact_match 为 0 指向"答错了标签"
"""

from __future__ import annotations

from .base import BaseHandWrittenMetric, MetricResult
from .exact_match import normalize_text

class LabelSetMetric(BaseHandWrittenMetric):
    name = "label_validity"

    def __init__(self, threshold: float = 1.0):
        self.threshold = threshold

    def measure(self, record, actual_output: str | None, parsed_output) -> MetricResult:
        labels = list(getattr(record, "allowed_labels", None) or [])
        if not labels:
            return MetricResult(self.name, None, self.threshold, True,
                                status="not_applicable",
                                reason="未声明 allowed_labels，标签合法性不适用")
        if actual_output is None:
            return MetricResult(self.name, None, self.threshold, True,
                                status="error", reason="推理失败，无输出可评")

        norm = normalize_text(actual_output)
        matched = [x for x in labels if normalize_text(x) == norm]
        if matched:
            return MetricResult(self.name, 1.0, self.threshold, True,
                                detail={"label": matched[0]},
                                reason=f"命中标签：{matched[0]}")

        contained = [x for x in labels if normalize_text(x) in norm]
        reason = (f"未按要求只输出标签（输出包含 {contained}，但夹带了解释）"
                  if contained else "输出不在允许标签集合内")
        return MetricResult(self.name, 0.0, self.threshold, True,
                            detail={"labels": labels, "pred": actual_output[:200]},
                            reason=reason)