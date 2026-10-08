"""质量门禁：严格语义 ———— 指标缺失即失败，覆盖率不足即失败，全空拒绝放行。"""

from __future__ import annotations

from dataclasses import dataclass, field

@dataclass(frozen=True)
class GateSpec:
    metric: str
    op: str                         # ">=" | "<="
    threshold: float
    required: bool = True           # 本次评测是否必须有该指标的适用样本
    min_coverage: float = 1.0       # 适用样本中必须成功算出的最低比例

    @property
    def higher_is_better(self) -> bool:
        return self.op == ">="

@dataclass
class GateViolation:
    metric: str
    value: float | None
    spec: GateSpec | None
    reason: str
    kind: str = "threshold"         # threshold | coverage | missing | blocker | empty

@dataclass
class GateReport:
    passed: bool
    violations: list[GateViolation] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    blocker_failures: list[str] = field(default_factory=list)   # 新增：一票否决清单
    evaluated: int = 0

    def as_dict(self) -> dict:
        return {
            "passed": self.passed,
            "evaluated": self.evaluated,
            "skipped": self.skipped,
            "blocker_failures": self.blocker_failures,
            "violations": [
                {"metric": v.metric, "value": v.value, "kind": v.kind,
                 "reason": v.reason,
                 "op": v.spec.op if v.spec else None,
                 "threshold": v.spec.threshold if v.spec else None}
                for v in self.violations
            ]
        }

def evaluate_gates(gates: dict, summary: dict,
                   blocker_failures: list[str] | None = None) -> GateReport:
    """
    三层判断：阈值 + 覆盖率 + blocker 一票否决，任一个不过即整体不过。

    blocker_failures 必须是显式入参。原实现试图把它做成 metrics["__blocker__"]，
    但 evaluate_gates 只遍历 gates 的键，哪个值永远不会读到 —— 死代码。
    """

    metrics = summary.get("metrics", {})
    stats = summary.get("metric_stats", {})
    report = GateReport(passed=True)

    for key, spec in gates.items():
        stat = stats.get(key)
        value = metrics.get(key)

        # 1、完全没有适用的样本：按 required 判定，绝不静默跳过
        if not stat or not stat.get("applicable"):
            if spec.required:
                report.passed = False
                report.violations.append(GateViolation(
                    key, None, spec,
                    "无有效值（指标缺失或全部不可用），required=True 判定为失败", kind="missing"))
            else:
                report.skipped.append(key)
            continue

        # 2、覆盖率不足：结论不可信，直接失败（堵住 judge 故障变绿灯）
        coverage = stat.get("coverage") or 0.0
        if coverage < spec.min_coverage:
            report.passed = False
            report.violations.append(GateViolation(
                key, value, spec,
                f"覆盖率 {coverage:.2%} 低于要求 {spec.min_coverage:.2%}"
                f"（{stat.get('errors', 2)} 条计算失败）", kind="coverage"))
            continue

        if value is None:
            report.passed = False
            report.violations.append(GateViolation(key, None, spec, "指标无有效取值",
                                                   kind="missing"))
            continue

        # 3、阈值比较
        ok = value >= spec.threshold if spec.higher_is_better else value <= spec.threshold
        if not ok:
            report.passed = False
            report.violations.append(GateViolation(key, value, spec,
                                                   f"要求 {spec.op} {spec.threshold}，实际{value}", kind="threshold"))
        else:
            report.evaluated += 1

    # 4、blocker 一票否决权，与任何指标无关。
    for item in blocker_failures or []:
        report.blocker_failures.append(item)
        report.passed = False
        report.violations.append(GateViolation(
            "__blocker__", None, None,
            f"blocker 用例未通过（一票否决）：{item}", kind="blocker"))

    # 5、没有任何指标真正参与判定 -> 拒绝放行（例如把 gates 传成空 dict）
    if report.evaluated == 0:
        report.passed = False
        report.violations.append(GateViolation(
            "__all__", None, None, "没有任何指标参与门禁判定，拒绝放行", kind="empty"))

    return report









