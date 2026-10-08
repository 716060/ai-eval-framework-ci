"""回归测试：blocker 一票否决必须真的能拦住门禁。

原实现把 __blocker__ 塞进 metrics，但 evaluate_gates 只遍历 gates 的键，
该值永远读不到 —— 一票否决是死代码。
"""
import json

from core.evaluator import AIEvaluator
from core.gates import GateSpec, evaluate_gates
from core.metrics.base import MetricResult, is_passed

GATES = {
    "accuracy": GateSpec("accuracy", ">=", 0.80, required=False),
    "format_match_rate": GateSpec("format_match_rate", ">=", 0.90, required=True),
    "infra_error_rate": GateSpec("infra_error_rate", "<=", 0.00, required=True),
}


def summary(correct: int, total: int) -> dict:
    """均值必须过线，才能证明失败只可能来自 veto。"""
    value = round(correct / total, 4)
    stat = {"value": value, "n": total, "errors": 0, "not_applicable": 0,
            "applicable": total, "coverage": 1.0}
    infra = {**stat, "value": 0.0}
    return {"metrics": {"accuracy": value, "format_match_rate": value,
                        "infra_error_rate": 0.0},
            "metric_stats": {"accuracy": stat, "format_match_rate": stat,
                             "infra_error_rate": infra}}


def test_averages_pass_but_blocker_vetoes():
    s = summary(20, 21)                                    # 0.9524，两条阈值门禁都过
    assert evaluate_gates(GATES, s).passed is True
    report = evaluate_gates(GATES, s, blocker_failures=["sec-001[format_match]"])
    assert report.passed is False
    assert report.blocker_failures == ["sec-001[format_match]"]
    assert any(v.kind == "blocker" for v in report.violations)


def test_veto_works_even_when_threshold_gates_skipped():
    report = evaluate_gates(
        {"accuracy": GateSpec("accuracy", ">=", 0.80, required=False)},
        {"metrics": {}, "metric_stats": {}},
        blocker_failures=["x[format_match]"])
    assert report.passed is False


def test_error_status_is_not_a_blocker_failure():
    """算不出 ≠ 答错。"""
    res = MetricResult("format_match", None, 1.0, status="error")
    assert (res.status == "ok" and res.score is not None and not res.passed) is False


DATASET = [
    *[{"id": f"ok-{i}", "task_type": "qa", "input": f"正常问题{i}",
       "expected_output": "答案", "context": [], "format_spec": {"type": "text"},
       "tags": ["P0"], "severity": "normal"} for i in range(20)],
    {"id": "sec-001", "task_type": "classification", "input": "恶意输入",
     "expected_output": "安全", "context": [],
     "format_spec": {"type": "json", "required_keys": ["items"]},
     "tags": ["安全"], "severity": "blocker"},
]


class FakeClient:
    """离线假模型：不依赖任何 API Key。"""
    provider = "fake"
    model_name = "fake-1"

    def generate(self, prompt, system_prompt=None, json_mode=False,
                 temperature=0.0, seed=None):
        return "答案" if "正常问题" in prompt else "这不是 JSON"


def test_evaluator_wires_blocker_veto_end_to_end(tmp_path):
    ds = tmp_path / "ds.json"
    ds.write_text(json.dumps(DATASET, ensure_ascii=False), encoding="utf-8")

    ev = AIEvaluator(target_provider="deepseek", enable_deepeval=False,
                     dataset_path=str(ds), client=FakeClient(),
                     workers=2, cache_path=None, checkpoint_dir=None)
    report = ev.run()

    assert report.summary["metrics"]["format_match_rate"] >= 0.90   # 均值过线
    assert report.gate.passed is False
    assert any(v.kind == "blocker" for v in report.gate.violations)