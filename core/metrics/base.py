"""手写指标基类与结果对象"""

from dataclasses import dataclass, field
from typing import Any, Literal

MetricStatus = Literal["ok", "not_applicable", "error"]

# 事实字段：只有这些会被序列化。passed / applicable 是派生属性，不落盘。
_FACT_FIELDS = ("name", "score", "threshold", "higher_is_better",
                "detail", "reason", "status")

@dataclass
class MetricResult:
    name: str
    score: float | None         # None = 不适合或计算失败，绝不参与均值
    threshold: float
    higher_is_better: bool = True
    detail: dict = field(default_factory=dict)
    reason: str = ""
    status: MetricStatus = "ok"


    @property
    def applicable(self) -> bool:
        return is_applicable(self.to_payload())

    @property
    def passed(self) -> bool:
        return is_passed(self.to_payload())

    # ---- 序列化边界 ----
    def to_payload(self) -> dict:
        return {k: getattr(self, k) for k in _FACT_FIELDS}

    @classmethod
    def from_payload(cls, payload: dict) -> "MetricResult":
        """
        严格还原：关键字段缺失直接报错。

        原：MetricResult(**{k: v for k, v in payload.items() if k in Known}) 宽松重建 + 默认值填充。
        载荷缺 status 时会被补成 "ok"，配上 score=None 就变成 [确定答案]，方向正好反了。
        """
        missing = {"name", "score", "threshold", "status"} - set(payload)
        if missing:
            raise ValueError(f"MetricResult 载荷缺少字段：{sorted(missing)}")
        return cls(
            name=payload["name"],
            score=payload["score"],
            threshold=payload["threshold"],
            higher_is_better=payload.get("higher_is_better", True),
            detail=payload.get("detail") or {},
            reason=payload.get("reason", ""),
            status=payload["status"],
        )



class BaseHandWrittenMetric:
    """所有手写指标的基类，统一 measure 签名"""

    name = "base"
    higher_is_better = True

    def measure(self, record: Any, actual_output: str | None, parsed_output: Any) -> MetricResult:
        raise NotImplementedError

def is_passed(payload: dict) -> bool:
    """
    供消费报告 JSON 的外部程序使用。
    passed / applicable 是 @property，不会出现在 asdict() 结果里。
    任何读落盘 JSON 的代码都必须走这个函数；直接读 payload["passed"]
    会永远拿到默认值 True，导致报告层把所有失败用例都显示为通过。
    """

    score = payload.get("score")
    if score is None:
        return False
    if payload.get("higher_is_better", True):
        return score >= payload["threshold"]
    return score <= payload["threshold"]

def is_applicable(payload: dict) -> bool:
    return payload.get("status", "ok") != "not_applicable"