"""门禁单测：指标缺失、覆盖率不足、全空，都必须判失败"""
from core.gates import GateSpec, evaluate_gates

GATES = {"accuracy": GateSpec("accuracy", ">=", 0.80, required=True, min_coverage=0.98)}

def summary(value=None, errors=0, applicable=10):
    if value is None and applicable == 0:
        return {"metrics": {}, "metric_stats": {}}
    return {
        "metrics": {"accuracy": value},
        "metric_stats": {"accuracy": {"value": value, "n": applicable - errors,
                                      "errors": errors, "not_applicable": 0,
                                      "applicable": applicable,
                                      "coverage": (applicable - errors)/applicable}}}


def test_pass():
    assert evaluate_gates(GATES, summary(0.9)).passed is True

def test_low_value_fails():
    assert evaluate_gates(GATES, summary(0.5)).passed is False

def test_missing_required_metric_fails():
    """原实现里指标为 None 直接 continue，门禁边绿灯。"""
    report = evaluate_gates(GATES, summary(applicable=0))
    assert report.passed == False
    assert report.violations[0].kind == "missing"

def test_low_coverage_fails():
    report = evaluate_gates(GATES, summary(0.95, errors=5, applicable=10))
    assert report.passed is False
    assert report.violations[0].kind == "coverage"

def test_empty_gate_set_refuses_to_pass():
    assert evaluate_gates({}, summary(0.9)).passed is False