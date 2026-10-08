"""手写指标：准确率（Exact Match / Accuracy），带文本归一化"""

import re
import unicodedata

from .base import BaseHandWrittenMetric, MetricResult

def normalize_text(text: str) -> str:
    """归一化：NFKC -> 小写 -> 压缩空白 -> 去除中英文标点。"""

    text = unicodedata.normalize('NFKC', str(text)).lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r"[，。、；：！？,.!?;:\"’‘”“()\[\]{}<>《》]", "", text)
    return text.strip()

class ExactMatchMetric(BaseHandWrittenMetric):
    name = "exact_match"

    def __init__(self, threshold: float = 1.0):
        self.threshold = threshold

    def measure(self, record, actual_output: str | None, parsed_output) -> MetricResult:
        if not isinstance(record.expected_output, str):
            return MetricResult(self.name, None, self.threshold, True, status="not_applicable",
                                reason="expected_output 非字符串，该任务不适合EM（已从均值排除）")
        if actual_output is None:
            return MetricResult(self.name, None, self.threshold, True,
                                status="error", reason="推理失败，无输出可评")
        score = 1.0 if normalize_text(actual_output) == normalize_text(record.expected_output) else 0.0
        return MetricResult(self.name, score, self.threshold, True,
                            detail={"pred": actual_output[:200], "gold": record.expected_output[:200]})
