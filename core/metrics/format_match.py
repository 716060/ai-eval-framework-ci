"""手写指标：格式匹配率（JSON必须键 / 正则 / 非空文本）"""

import json
import re

from .base import BaseHandWrittenMetric, MetricResult

class FormatMatchMetric(BaseHandWrittenMetric):
    name = "format_match"

    def __init__(self, threshold: float = 1.0):
        self.threshold = threshold

    def measure(self, record, actual_output: str, parsed_output) -> MetricResult:
        spec = record.format_spec or {"type": "text"}
        ftype = spec.get("type", "text")

        if actual_output is None:
            return MetricResult(self.name, None, self.threshold, True,
                                status="error", reason="推理失败，无输出可评")

        if ftype == "text":
            score, reason = (1.0, "文本非空") if actual_output.strip() else (0.0, "输出为空")
        elif ftype == "json":
            ok, reason = self._check_json(actual_output, parsed_output, spec.get("required_keys", []))
            score = 1.0 if ok else 0.0
        else:   # regex
            pattern = spec.get("pattern", "")
            if not pattern:
                return MetricResult(self.name, None, self.threshold, True,
                                    status="error", reason="regex 类型缺少 pattern")
            score = 1.0 if re.search(pattern, actual_output) else 0.0
            reason = f"regex={pattern}"

        return MetricResult(self.name, score, self.threshold, True,
                            detail={"spec": spec}, reason=reason)

    @staticmethod
    def _check_json(raw: str, parsed, required_keys: list) -> tuple[bool, str]:
        if parsed is None:
            try:
                parsed = json.loads(raw)
            except Exception as exc:    #noqa: BLE001
                return False, f"JSON 解析失败：{exc}"
        if not isinstance(parsed, (dict, list)):
            return False, "JSON 顶层类型不符（dict/list）"
        if required_keys:
            # 声明了必须键 = 契约要求顶层是对象；裸数组即违约。
            if not isinstance(parsed, dict):
                return False, (f"声明了必须键 {required_keys}，顶层必须是 JSON 对象，"
                               f"实际为 {type(parsed).__name__}")
            missing = [k for k in required_keys if k not in parsed]
            if missing:
                return False, f"缺少必须键：{missing}"

        return True, "JSON 结构与必需键校验通过"

