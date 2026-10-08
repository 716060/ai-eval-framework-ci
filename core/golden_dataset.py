"""黄金数据集：Recode 定义 + 纯手写规则校验器（Fail-Fast，校验不过禁止开跑）。"""

import json
import re

from dataclasses import dataclass, field
from typing import Any
from utils.logger import get_logger

logger = get_logger(__name__)

ALLOWED_TASK_TYPES = {"qa", "classification", "extraction", "generation"}
ALLOWED_FORMAT_TYPES = {"text", "json", "regex"}
ALLOWED_SEVERITY = {"blocker", "normal"}          # 新增



class GoldenDatasetValidationError(ValueError):
    """黄金数据集不合法"""

@dataclass(frozen=True)
class GoldenRecord:
    id: str
    task_type: str
    input: str
    expected_output: Any
    context: list = field(default_factory=list)    # 事实清单，供幻觉/忠诚度使用
    format_spec: dict = field(default_factory=lambda: {"type": "text"})
    tags: list = field(default_factory=list)
    severity: str = "normal"                # 新增：blocker 用例不通过即整体失败
    allowed_labels: list = field(default_factory=list)  # 新增：闭集标签

@dataclass
class ValidationReport:
    valid: bool
    errors: list
    warnings: list
    stats: dict

class GoldenDatasetValidator:
    """手写校验规则：必填/类型/唯一性/枚举/业务一致性/格式声明合法性"""

    def validate(self, raw: Any) -> ValidationReport:
        errors, warnings = [], []
        if not isinstance(raw, list) or not raw:
            return ValidationReport(False, ["数据集必须是非空 JSON 数组"], [], {})

        seen_ids, type_counter = set(), {}
        for idx, item in enumerate(raw):
            loc = f"#[{idx}]"
            if not isinstance(item, dict):
                errors.append(f"{loc} 元素必须是 JSON 对象")
                continue
            rid = item.get("id")
            loc = f"#[{idx}](id={rid})"

            # 规则1：必填字段与类型
            for fname, ftype in (("id", str), ("task_type", str),
                                 ("input", str), ("expected_output", (str, list, dict))):
                if fname not in item:
                    errors.append(f"{loc} 缺少必填字段 `{fname}`")
                elif not isinstance(item[fname], ftype):
                    errors.append(f"{loc} 字段 `{fname}` 类型错误")
            if not isinstance(rid, str):
                continue

            # 规则2：id 唯一
            if rid in seen_ids:
                errors.append(f"{loc} id 重复")
            seen_ids.add(rid)

            # 规则3：input 非空
            if not item.get("input", "").split():
                errors.append(f"{loc} input 为空")

            # 规则4：task_type 枚举 + 分布统计
            task_type = item.get("task_type")
            if task_type not in ALLOWED_TASK_TYPES:
                errors.append(f"{loc} task_type 非法：{task_type}")
            else:
                type_counter[task_type] = type_counter.get(task_type, 0) + 1

            # 规则5：业务一致性 —— extraction 任务的标注必须是列表
            expected = item.get("expected_output")
            if expected in ("", [], {}) or expected is None:
                warnings.append(f"{loc} expected_output 为空")
            if task_type == "extraction" and not isinstance(expected, list):
                errors.append(f"{loc} extraction 任务的 expected_output 必须是列表")

            # 规则6：context 必须为非空字符串列表（幻觉/忠实度指标依赖）
            context = item.get("context", [])
            if not isinstance(context, list) or not all(
                isinstance(c, str) and c.strip() for c in context
            ):
                errors.append(f"{loc} context 必须为非空字符串列表")
            elif not context:
                warnings.append(f"{loc} 缺少 context，幻觉/忠诚度指标将退化为已标注为事实源")

            # 规则7：format_spec 声明合法性（正则必须可编译）
            spec = item.get("format_spec", {"type": "text"})
            if not isinstance(spec, dict) or spec.get("type") not in ALLOWED_FORMAT_TYPES:
                errors.append(f"{loc} format_spec.type 非法：{spec}")
            elif spec["type"] == "json" and not isinstance(spec.get("required_keys", []), list):
                errors.append(f"{loc} json 格式声明需要 required_keys 列表")
            elif spec["type"] == "regex":
                try:
                    re.compile(spec.get("pattern", ""))
                except re.error:
                    errors.append(f"{loc} regex pattern 无法编译")
            # 规则8：severity字段类型只能是{"blocker", "normal"}
            if item.get("severity", "normal") not in ALLOWED_SEVERITY:
                errors.append(f"{loc} severity 非法：{item.get('severity')}")

            # 规则9：tags字段非空且类型必须是list
            if not isinstance(item.get("tags", []), list):
                errors.append(f"{loc} tags 必须是列表")

            # 规则10：blocker 用例必须带 context 或明确说明豁免，否则语义指标无从校验
            if item.get("severity") == "blocker" and not item.get("context"):
                warnings.append(f"{loc} blocker 用例缺少 context，幻觉/忠实度将不可评估")
            # 规则11：闭集任务必须声明标签集，且标准答案必须在集合内
            if task_type == "classification":
                labels = item.get("allowed_labels")
                if not isinstance(labels, list) or not labels:
                    errors.append(f"{loc} classification 任务必须声明非空 allowed_labels")
                elif not all(isinstance(x, str) and x.strip() for x in labels):
                    errors.append(f"{loc} allowed_labels 必须是非空字符串列表")
                elif item.get("expected_output") not in labels:
                    errors.append(f"{loc} expected_output {item.get('expected_output')!r} "
                                  f"不在 allowed_labels 内")
            elif "allowed_labels" in item and not isinstance(item["allowed_labels"], list):
                errors.append(f"{loc} allowed_labels 必须是列表")


        return ValidationReport(
            valid=not errors,
            errors=errors,
            warnings=warnings,
            stats={"total": len(raw), "by_task_type": type_counter},
        )

def load_golden_dataset(path: str) -> tuple[list[GoldenRecord], ValidationReport]:
    """加载并校验黄金数据集，校验失败直接抛异常阻断评测"""
    with open(path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)

    report = GoldenDatasetValidator().validate(raw)
    for warn in report.warnings:
        logger.warning("数据集警告：%s", warn)
    if not report.valid:
        for err in report.errors:
            logger.error("数据集错误：%s", err)
        raise GoldenDatasetValidationError(
            f"黄金数据集校验失败，共 {len(report.errors)} 个错误，已阻断评测")

    record = [
        GoldenRecord(
            id=item["id"],
            task_type=item["task_type"],
            input=item["input"],
            expected_output=item["expected_output"],
            context=item.get("context", []),
            format_spec=item.get("format_spec", {"type": "text"}),
            tags=item.get("tags", []),
            severity=item.get("severity", "normal"),
            allowed_labels=item.get("allowed_labels", []),
        ) for item in raw
    ]

    logger.info("黄金数据集加载成功：%d 条，分布=%s", report.stats["total"], report.stats["by_task_type"])
    return record, report












