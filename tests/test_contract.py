"""序列化契约守卫：派生字段跨边界必须重算。"""
import json
from dataclasses import asdict

import pytest

from core.case import CaseResult
from core.metrics.base import MetricResult, is_passed


def test_derived_fields_are_not_serialized():
    """passed / applicable 是 @property，不会出现在落盘字段里。"""
    payload = MetricResult("format_match", 0.0, 1.0).to_payload()
    assert "passed" not in payload and "applicable" not in payload
    assert is_passed(payload) is False
    assert is_passed({**payload, "score": 1.0}) is True


def test_from_payload_rejects_missing_status():
    """未知不能退化成 ok：缺 status 必须报错。"""
    with pytest.raises(ValueError):
        MetricResult.from_payload({"name": "x", "score": 0.0, "threshold": 1.0})


def test_case_result_keeps_metric_objects():
    case = CaseResult(record_id="r1", task_type="qa", input="i", actual_output="o",
                      expected_output="o",
                      handwritten={"format_match": MetricResult("format_match", 0.0, 1.0)})
    assert isinstance(case.handwritten["format_match"], MetricResult)
    assert case.failed_metrics() == ["format_match"]


def test_case_roundtrip_preserves_verdict():
    """断点续跑读回的结果，判定必须与首轮一致。"""
    case = CaseResult(record_id="r1", task_type="qa", input="i", actual_output="o",
                      expected_output="o", run_key="r1#1",
                      handwritten={"format_match": MetricResult("format_match", 0.0, 1.0)})
    restored = CaseResult.from_payload(json.loads(json.dumps(case.to_payload())))
    assert restored.run_key == "r1#1"
    assert restored.failed_metrics() == ["format_match"]