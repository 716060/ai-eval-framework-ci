"""
指标路由 + 用例级 -> 套件级指标名导出契约。

原实现只有METRIC_ROUTER，用例级名字(exact_match / set_pr/ format_match)
与门禁键名（accuracy / f1 / format_match_rate）靠隐式约定对应，
聚合层一旦重写就断链，门禁键永远取不到值。
"""
from .base import (BaseHandWrittenMetric, MetricResult, is_applicable, is_passed)
from .exact_match import ExactMatchMetric, normalize_text
from .set_pr import SetPrecisionRecallMetric
from .format_match import FormatMatchMetric
from .pii import PIILeakMetric
from .schema_strict import SchemaStrictMetric
from .refusal import RefusalMetric
from .label_set import LabelSetMetric

METRIC_ROUTER:dict[str, list[type[BaseHandWrittenMetric]]] = {
    "qa":               [ExactMatchMetric, FormatMatchMetric, PIILeakMetric, RefusalMetric],
    "classification":   [ExactMatchMetric, FormatMatchMetric, LabelSetMetric],
    "extraction":       [SetPrecisionRecallMetric, FormatMatchMetric, SchemaStrictMetric],
    "generation":       [FormatMatchMetric, RefusalMetric, PIILeakMetric]     #开放生成只卡格式，语义交给 DeepEval
}

# 用例级指标名 -> 套件级指标名（门禁只认套件级名字）
SUITE_EXPORTERS: dict[str, callable] = {
    "exact_match": lambda r: {"exact_match_rate": r.score},
    "format_match": lambda r: {"format_match_rate": r.score},
    "set_pr": lambda r:{
        "precision": r.detail.get("precision"),
        "recall": r.detail.get("recall"),
        "f1": r.score
    },
    "pii_leak": lambda r: {"pii_leak_rate": r.score},
    "refusal": lambda r: {"refusal_rate": r.score},
}


def build_metric_for(task_type: str) -> list[BaseHandWrittenMetric]:
    classes = METRIC_ROUTER.get(task_type, [FormatMatchMetric])
    return [cls() for cls in classes]

def export_suite_metrics(case_level: dict[str, MetricResult]) -> dict[str, MetricResult]:
    """把单条用例的指标展开成套件级指标，状态和原因完整传递。"""
    out: dict[str, MetricResult] = {}
    for name, res in case_level.items():
        export = SUITE_EXPORTERS.get(name)
        if export is None:
            out[name] = res
            continue
        for suite_name, value in export(res).items():
            out[suite_name] = MetricResult(
                name=suite_name,
                score=value,
                threshold=res.threshold,
                higher_is_better=res.higher_is_better,
                detail=res.detail,
                reason=res.reason,
                status=res.status,
            )
    return out



__all__ = ["BaseHandWrittenMetric", "MetricResult", "ExactMatchMetric",
           "SetPrecisionRecallMetric", "FormatMatchMetric",
           "build_metric_for", "normalize_text", "SchemaStrictMetric",
           "RefusalMetric", "PIILeakMetric", "METRIC_ROUTER", "SUITE_EXPORTERS",
           "export_suite_metrics", "is_applicable", "is_passed", "LabelSetMetric"]