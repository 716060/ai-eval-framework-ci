"""手写指标：集合级 精确率（Precision） / 召回率（Recall） / F1，用于抽取与多标签任务。"""

import json
import re
from typing import Any

from .base import BaseHandWrittenMetric, MetricResult
from .exact_match import normalize_text

ENVELOP_KEYS = ("items",)

class SetPrecisionRecallMetric(BaseHandWrittenMetric):
    name = "set_pr"

    def __init__(self, threshold: float = 0.8):    # 门禁作用于 f1
        self.threshold = threshold

    @staticmethod
    def _to_set(value) -> set:
        """
        把 列表/JSON字符串/逗号分割字符串/包裹对象 统一为归一化集合。
        注意：抽取任务的 {"item": [...]} 在这里被解包，
        信封本身的合法性由 FormatMatchMetric 的required_keys 负责。
        """
        if isinstance(value, list):
            items = value
        elif isinstance(value, dict):
            # 原实现取第一个list值，多键时结果不确定
            # lists = [v for v in value.values() if isinstance(v, list)]
            # items = lists[0] if lists else list(value.values())
            picked = next((value[k] for k in ENVELOP_KEYS
                          if isinstance(value.get(k), list)), None)
            if picked is None:
                lists = [v for v in value.values() if isinstance(v, list)]
                picked = lists[0] if lists else list(value.values())
            items = picked
        elif isinstance(value, str):
            try:
                loaded = json.loads(value)
                items = loaded if isinstance(loaded, list) else [loaded]
            except Exception:  #noqa: BLE001
                items = [s for s in re.split(r"[,，;；、]", value) if s.strip()]
        else:
            items = [value]
        return {normalize_text(i) for i in items if str(i).strip()}

    def measure(self, record: Any, actual_output: str | None, parsed_output) -> MetricResult:
        if actual_output is None:
            return MetricResult(self.name, None, self.threshold, True,
                                status="error", reason="推理失败，无输出可评")
        pred = self._to_set(parsed_output if isinstance(parsed_output, (list, dict))
                            else actual_output)
        gold = self._to_set(record.expected_output)

        if not gold and not pred:
            precision = recall = f1 = 1.0
        elif not pred:
            precision = recall = f1 = 0.0
        else:
            tp = len(pred & gold)
            precision = tp / len(pred)
            recall = tp / len(gold) if gold else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

        return MetricResult(
            self.name, round(f1, 4), self.threshold, True,
            detail={"precision": round(precision, 4), "recall": round(recall, 4),
                    "f1": round(f1, 4), "missing": sorted(gold - pred), "extra": sorted(pred - gold)},
        )









