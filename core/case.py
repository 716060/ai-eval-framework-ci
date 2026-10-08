"""
单用例结果对象。

约定：内存里始终持有 MetricResult 对象，序列化只发生在报告边界。
原实现边构造边 asdict()，导致 passed / applicable 这类派生属性丢失，
下游又用 .get("passed", True) 读回来 —— 真实失败被静默吞掉。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from core.metrics.base import MetricResult

@dataclass
class CaseResult:
    record_id: str
    task_type: str
    input: str
    actual_output: str | None
    expected_output: Any
    run_key: str = ""       # 新增：repeats 场景下的唯一身份
    tags: list = field(default_factory=list)
    severity: str = "normal"
    handwritten: dict[str, MetricResult] = field(default_factory=dict)
    semantic: dict[str, MetricResult] = field(default_factory=dict)
    error: dict | None = None
    usage: dict = field(default_factory=dict)

    # --- 判定：全局唯一实现，evaluator 与 reporter 共用 ---
    def _merged(self) -> dict[str, MetricResult]:
        return {**self.handwritten, **self.semantic}

    def failed_metrics(self) -> list[str]:
        """算出来了但没通过。status=error 不算在内（那是覆盖率门禁的职责）"""
        return [k for k, r in self._merged().items()
                if r.status == "ok" and r.score is not None and not r.passed]

    def errored_metrics(self) -> list[str]:
        return [k for k, r in self._merged().items() if r.status == "error"]

    # --- 序列化边界 ---
    def to_payload(self) -> dict:
        return {
            "record_id": self.record_id,
            "task_type": self.task_type,
            "input": self.input,
            "actual_output": self.actual_output,
            "expected_output": self.expected_output,
            "run_key": self.run_key,
            "tags": list(self.tags),
            "severity": self.severity,
            "error": self.error,
            "usage": self.usage,
            "handwritten": {k: v.to_payload() for k, v in self.handwritten.items()},
            "semantic": {k: v.to_payload() for k, v in self.semantic.items()},
        }

    @classmethod
    def from_payload(cls, payload: dict) -> CaseResult:
        return cls(
            record_id=payload["record_id"],
            task_type=payload["task_type"],
            input=payload["input"],
            actual_output=payload.get("actual_output"),
            expected_output=payload.get("expected_output"),
            run_key=payload.get("run_key", ""),
            tags=payload.get("tags", []),
            severity=payload.get("severity", "normal"),
            handwritten={k: MetricResult.from_payload(v)
                         for k, v in (payload.get("handwritten") or {}).items()},
            semantic={k: MetricResult.from_payload(v)
                      for k, v in (payload.get("semantic") or {}).items()},
            error=payload.get("error"),
            usage=payload.get("usage", {}),
        )
