"""基线与回归对比：大厂评测的核心价值不是绝对分，而是 '这次比上次差了吗'。 """
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

@dataclass
class MetricDelta:
    metric: str
    current: float
    baseline: float
    delta: float
    tolerance: float
    higher_is_better: bool

    @property
    def is_regression(self) -> bool:
        return (self.delta < -self.tolerance if self.higher_is_better
                else self.delta > self.tolerance)


@dataclass
class ComparisonResult:
    deltas: list[MetricDelta] = field(default_factory=list)
    coverage_drops: list[str] = field(default_factory=list)
    dropped_metrics: list[str] = field(default_factory=list)        # 基线有、本次没有 = 观测退化
    new_metrics: list[str] = field(default_factory=list)            # 本次新增 = 仅提示，不阻断

    @property
    def regressions(self) -> list[MetricDelta]:
        return [d for d in self.deltas if d.is_regression]

    @property
    def has_regression(self) -> bool:
        # 原：把[本次新增的指标] 也当成回退 -> 刚加 pii_leak 就让 CI 变红
        return bool(self.regressions or self.coverage_drops or self.dropped_metrics)

    def summary_line(self) -> str:
        if not self.has_regression:
            extra = f"，新增指标{self.new_metrics}" if self.new_metrics else ""
            return f"无回退（对比 {len(self.deltas)} 项{extra}）"
        parts = [f"{d.metric} {d.baseline:.4f}->{d.current:.4f}" for d in self.regressions]
        if self.coverage_drops:
            parts.append(f"覆盖率下降：{self.coverage_drops}")
        if self.dropped_metrics:
            parts.append(f"基线存在但本次缺失：{self.dropped_metrics}")
        return "；".join(parts)

def compare_with_baseline(current: dict, gates: dict, tolerance: float = 0.02,
                          baseline_path: str | None = None) -> ComparisonResult:
    path = Path(baseline_path or "reports/baseline.json")
    if not path.exists():
        return ComparisonResult()
    baseline = json.loads(path.read_text(encoding="utf-8"))
    base_summary = baseline.get("summary", {})
    base_metrics = base_summary.get("metrics", {})
    base_stats = base_summary.get("metric_stats", {})
    cur_metrics, cur_stats = current["metrics"], current["metric_stats"]

    result = ComparisonResult()
    for key in set(base_metrics) - set(cur_metrics):
        result.dropped_metrics.append(key)

    for key, value in cur_metrics.items():
        if key.startswith("__") or value is None:
            continue
        old = base_metrics.get(key)
        if old is None:
            result.new_metrics.append(key)
            continue
        spec = gates.get(key)
        # hib = spec["op"] == ">=" if spec else True
        hib = spec.higher_is_better if spec else True
        result.deltas.append(MetricDelta(key, value, old, round(value - old, 4), tolerance, hib))
        # 覆盖率下降意味着数字不可比，必须单独告警
        old_cov = (base_stats.get(key) or {}).get("coverage")
        new_cov = (cur_stats.get(key) or {}).get("coverage")
        if old_cov is not None and new_cov is not None and new_cov < old_cov - 1e-9:
            result.coverage_drops.append(f"{key} {old_cov:.2%}->{new_cov:.2%}")
    return result