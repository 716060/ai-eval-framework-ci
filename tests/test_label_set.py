"""标签集约束的回归测试。"""
import pytest

from core.golden_dataset import GoldenDatasetValidator, GoldenRecord
from core.metrics.exact_match import ExactMatchMetric
from core.metrics.label_set import LabelSetMetric

LABELS = ["资费咨询", "账单查询", "网络故障", "其他"]


def rec(**kw) -> GoldenRecord:
    base = dict(id="cls-001", task_type="classification", input="这个月流量费怎么突然涨了？",
                expected_output="资费咨询", allowed_labels=LABELS)
    base.update(kw)
    return GoldenRecord(**base)


@pytest.mark.parametrize("output", ["资费咨询", "资费咨询。", "“资费咨询”", " 资费咨询 "])
def test_punctuation_and_quotes_do_not_break_matching(output):
    assert LabelSetMetric().measure(rec(), output, None).score == 1.0


def test_wrong_label_is_a_correctness_issue_not_a_contract_issue():
    """答错标签：label_validity 通过，exact_match 失败。"""
    assert LabelSetMetric().measure(rec(), "网络故障", None).score == 1.0
    assert ExactMatchMetric().measure(rec(), "网络故障", None).score == 0.0


def test_explanation_breaks_the_contract():
    out = "这个月流量费突然涨，常见原因有：套餐外流量用超了、优惠包到期……"
    assert LabelSetMetric().measure(rec(), out, None).score == 0.0


def test_label_with_explanation_reports_contract_breach():
    res = LabelSetMetric().measure(rec(), "我认为是资费咨询，因为……", None)
    assert res.score == 0.0
    assert "夹带" in res.reason


def test_not_applicable_without_labels():
    res = LabelSetMetric().measure(rec(allowed_labels=[]), "任意输出", None)
    assert res.status == "not_applicable" and res.score is None


def test_prompt_lists_all_labels():
    from core.evaluator import AIEvaluator
    prompt, json_model = AIEvaluator._build_prompt(rec())
    assert json_model is False
    for label in LABELS:
        assert label in prompt
    assert "直接输出标签本身" in prompt


@pytest.mark.parametrize("item,expect_error", [
    ({"id": "c1", "task_type": "classification", "input": "问", "expected_output": "资费咨询"}, True),
    ({"id": "c2", "task_type": "classification", "input": "问", "expected_output": "资费咨询",
      "allowed_labels": ["账单查询"]}, True),
    ({"id": "c3", "task_type": "classification", "input": "问", "expected_output": "资费咨询",
      "allowed_labels": LABELS}, False),
])
def test_validator_enforces_label_set(item, expect_error):
    report = GoldenDatasetValidator().validate([item])
    assert (not report.valid) is expect_error