"""
指标聚合：区分 有效值 / 不适用 / 计算失败，并暴露覆盖率。
原实现 _mean 简单过滤 None，导致“judge 全挂”与“模型满分产出同一份报告。”
"""

from __future__ import annotations
from dataclasses import dataclass, field

from core.metrics import MetricResult, export_suite_metrics


@dataclass
class MetricBucket:
    values: list[float] = field(default_factory=list)
    not_applicable: int = 0
    errors: int = 0

    def add(self, res) -> None:
        if res.status == "not_applicable":
            self.not_applicable += 1
        elif res.status == "error" or res.score is None:
            self.errors += 1
        else:
            self.values.append(float(res.score))

    @property
    def n(self) -> int:
        return len(self.values)

    @property
    def applicable(self) -> int:
        """产于该指标的样本数 = 成功 + 失败（不适合不计入）"""
        return self.n + self.errors

    @property
    def mean(self) -> float | None:
        return round(sum(self.values) / self.n, 4) if self.values else None

    @property
    def coverage(self) -> float | None:
        """覆盖率 = 成功算出 / 应算出。覆盖率低意味着结论不可信。"""
        return round(self.n / self.applicable, 4) if self.applicable else None

    def as_dict(self) -> dict:
        return {
            "value": self.mean,
            "n": self.n,
            "not_applicable": self.not_applicable,
            "errors": self.errors,
            "applicable": self.applicable,
            "coverage": self.coverage,
        }

class MetricBundle:
    """按 全局 / 任务类型 / tag 三个维度同时累计指标。"""
    def __init__(self):
        self.buckets: dict[str, MetricBucket] = {}
        self.by_task_type: dict[str, dict[str, MetricBucket]] = {}
        self.by_tag: dict[str, dict[str, MetricBucket]] = {}
        self.blocker_failures: list[str] = []

    @staticmethod
    def _add(view: dict[str, MetricBucket], name: str, res: MetricResult) -> None:
        view.setdefault(name, MetricBucket()).add(res)

    def add_case(self, handwritten: dict[str, MetricResult], semantic: dict[str, MetricResult],
                 task_type: str, tags: list[str]) -> dict[str, MetricResult]:
        """展开为套件级指标并累计，同时把展开结果返还给调用方做 blocker 判定"""
        suite = export_suite_metrics({**handwritten, **semantic})
        task_view = self.by_task_type.setdefault(task_type, {})
        for name, res in suite.items():
            self._add(self.buckets, name, res)
            self._add(task_view, name, res)
            for tag in tags:
                self._add(self.by_tag.setdefault(tag, {}), name, res)
        return suite

    def mark_blocker_failure(self, record_id: str) -> None:
        self.blocker_failures.append(record_id)


    @property
    def metrics(self) -> dict[str, float | None]:
        return {k: b.mean for k, b in self.buckets.items()}

    @property
    def stats(self) -> dict[str, dict]:
        return {k: b.as_dict() for k, b in self.buckets.items()}


# def consistency_rate(records_outputs: dict[str, list[str]]) -> float | None:
#     """同一用例重复采样的输出一致率；温度>0 时应显著低于 1，否则说明采样无效。"""
#     from core.metrics.exact_match import normalize_text
#     from collections import Counter
#     rates = []
#     for outputs in records_outputs.values():
#         if len(outputs) < 2:
#             continue
#         normalized = [normalize_text(o) for o in outputs]
#         counts = Counter(normalized)
#         most_common_count = counts.most_common(1)[0][1]
#         rates.append(most_common_count / len(normalized))
#     return round(sum(rates) / len(rates), 4) if rates else None

