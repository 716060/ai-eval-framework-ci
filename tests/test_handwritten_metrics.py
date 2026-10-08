"""手写指标单侧：纯函数、零外部依赖，必须先绿"""
import pytest

from core.golden_dataset import GoldenRecord
from core.metrics.exact_match import ExactMatchMetric, normalize_text
from core.metrics.format_match import FormatMatchMetric
from core.metrics.set_pr import SetPrecisionRecallMetric

def rec(**kw) -> GoldenRecord:
    base = dict(id="t1", task_type="qa", input="问", expected_output="答")
    base.update(kw)
    return GoldenRecord(**base)

@pytest.mark.parametrize("left, right, expected", [
    ("你好，世界！", "你好世界", True),
    ("A B C", "abc", True),         # 全角 NFKC 归一化
    ("  a   b  ", "a b", True),     # 空白压缩
    ("答案一", "答案二", False),
])
def test_normalize_text(left, right, expected):
    assert (normalize_text(left) == normalize_text(right)) is expected

def test_exact_match_hit_and_miss():
    m = ExactMatchMetric()
    assert m.measure(rec(expected_output="答案是7天"), "答案是7天", None).score == 1.0
    assert m.measure(rec(), "别的答案", None).score == 0.0

def test_exact_match_not_applicable_when_gold_is_not_str():
    """非字符串标准答案必须返回 None + not_application，绝不能用 0 分污染 accuracy 均值。"""
    res = ExactMatchMetric().measure(rec(expected_output=["a", "b"]), "任意输出", None)
    assert res.score is None
    assert res.status == "not_applicable"
    assert res.applicable is False

def test_set_pr_boundaries():
    m = SetPrecisionRecallMetric()
    both_empty = m.measure(rec(expected_output=[]), "", None)
    assert (both_empty.detail["precision"], both_empty.detail["recall"]) == (1.0, 1.0)

    pred_empty = m.measure(rec(expected_output=["a"]), "", None)
    assert pred_empty.detail["recall"] == 0.0

    partial = m.measure(rec(expected_output=["a", "b"]), '["a", "c"]', ["a", "c"])
    assert partial.detail["recall"] == 0.5
    assert partial.detail["precision"] == 0.5
    assert partial.detail["extra"] == ["c"]
    assert partial.detail["missing"] == ["b"]

def test_set_pr_accepts_wrapper_object():
    """extraction 统一输出 {"items": [...]}，指标层必须能解包"""
    m = SetPrecisionRecallMetric()
    res = m.measure(rec(expected_output=["a", "b"]), "", {"items": ["a", "b"]})
    assert res.score == 1.0




@pytest.mark.parametrize("spec,output,expected", [
    ({"type": "text"}, " ", 0.0),
    ({"type": "text"}, "有内容", 1.0),
    ({"type": "json", "required_keys": ["a"]}, '{"b":1}', 0.0),
    ({"type": "json", "required_keys": ["a"]}, '{"a":1}', 1.0),
    ({"type": "regex", "pattern": r"\d{3}$"}, "123", 1.0),
    ({"type": "regex", "pattern": r"\d{3}$"}, "12", 0.0),
])
def test_format_match(spec, output, expected):
    res = FormatMatchMetric().measure(rec(format_spec=spec), output, None)
    assert res.score == expected


def test_empty_regex_pattern_is_error_not_pass():
    """空 pattern 用 re.search("") 会匹配一切 -> 恒为 1.0。"""
    res = FormatMatchMetric().measure(rec(format_spec={"type": "regex"}), "任意", None)
    assert res.status == "error" and res.score is None


















