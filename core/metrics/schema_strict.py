"""结构化输出的严格 JSON Schema 校验"""
from __future__ import annotations

import json
from typing import Any

from .base import BaseHandWrittenMetric, MetricResult

class SchemaStrictMetric(BaseHandWrittenMetric):
    name = "schema_strict"

    def measure(self, record, actual_output: str | None , parsed_output) -> MetricResult:
        schema = (record.format_spec or {}).get("json_schema")
        if not schema:
            return MetricResult(self.name, None, 1.0, status="not_applicable",
                                reason="未声明 json_schema")
        try:
            import jsonschema
        except ImportError:
            # 不静默降级：依赖缺失必须显示暴露，由覆盖率门禁拦截
            return MetricResult(self.name, None, 1.0, status="error",
                                reason="未安装 jsonschema，无法执行严格校验")
        if parsed_output is None:
            return MetricResult(self.name, 0.0, 1.0, True, reason="输出不是合法 JSON")
        try:
            jsonschema.validate(instance=parsed_output, schema=schema)
        except Exception as exc:   #noqa: BLE001
            return MetricResult(self.name, 0.0, 1.0, True, reason=f"schema 校验失败：{exc}")
        return MetricResult(self.name, 1.0, 1.0, True, reason="schema 校验通过")