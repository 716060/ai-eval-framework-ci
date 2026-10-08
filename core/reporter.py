"""
报告输出: 控制台摘要 + JSON 明细 + Markdown 报告(CI artifact) + Junit XML。

原实现两种表示各写一遍判断逻辑，且读 asdict 产物里的 passed 字段（永远是默认值）。
这里统一在对象上判定、只在写文件时序列化。
"""
from __future__ import annotations

import json
import os
import time
from xml.sax.saxutils import escape, quoteattr

from core.gates import GateSpec
from utils.logger import get_logger

logger = get_logger("reporter")


class Reporter:
    def __init__(self, output_dir: str = "reports"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def dump(self, report) -> dict:
        ts = time.strftime("%Y%m%d_%H%M%S")
        base = f"eval_{ts}_{report.manifest['run_id'][:8]}"
        payload = {
            "manifest": report.manifest,
            "summary": report.summary,
            "gate": report.gate.as_dict(),
            "by_task_type": report.by_task_type,
            "by_tag": report.by_tag,
            "cases": [c.to_payload() for c in report.cases],
        }
        paths = {
            "json": self._write(os.path.join(self.output_dir, f"{base}.json"),
                                json.dumps(payload, ensure_ascii=False, indent=2)),
            "markdown": self._write(os.path.join(self.output_dir, f"{base}.md"),
                                    self._markdown(report)),
            "junit": self._write(os.path.join(self.output_dir, f"{base}.xml"),
                                 self._junit(report)),
        }
        self._console(report)
        return paths

    def _write(self, path: str, content: str) -> str:
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return path

    def _junit(self, report) -> str:
        """JUnit XML：让 Jenkins / GitLab / GitHub Actions 直接展示逐用例结果。"""
        cases = report.cases
        failures = sum(1 for c in cases if c.error or c.failed_metrics())
        lines = [f'<testsuites name="ai-eval" tests="{len(cases)}">',
                 f'<testsuite name="golden-dataset" tests="{len(cases)}" '
                 f'failures="{failures}" '
                 f'timestamp="{escape(report.manifest["started_at"])}">']
        for case in cases:
            name = case.run_key or case.record_id
            lines.append(f'<testcase classname={quoteattr(case.task_type)} '
                         f'name={quoteattr(name)}>')
            if case.error:
                lines.append(f'<error message={quoteattr(str(case.error.get("message", ""))[:400])} '
                             f'type={quoteattr(str(case.error.get("type", "Error")))}/>')
            else:
                failed = case.failed_metrics()
                if failed:
                    detail = "; ".join(
                        f"{k}={case.handwritten.get(k, case.semantic.get(k)).score}"
                        f"({case.handwritten.get(k, case.semantic.get(k)).reason})"
                        for k in failed)
                    lines.append(f'<failure message={quoteattr(detail[:800])}/>')
            lines.append("</testcase>")
        lines += ["</testsuite>", "</testsuites>"]
        return "\n".join(lines)

    @staticmethod
    def _fmt(v) -> str:
        return "N/A" if v is None else f"{v:.2%}"

    @staticmethod
    def _gate_cell(stat: dict, spec: "GateSpec | None") -> str:
        if not spec:
            return "-"
        mark = "-"
        if stat["value"] is not None:
            ok = (stat["value"] >= spec.threshold if spec.higher_is_better
                  else stat["value"] <= spec.threshold)
            mark = "√" if ok else "×"
        return f"{spec.op} {spec.threshold} {mark}"


    def _markdown(self, report) -> str:
        """在原报告基础上补：指标覆盖率表、tag 切片、blocker 失败清单、成本延迟、run 指纹。"""
        m, s, g = report.manifest, report.summary, report.gate
        lines = ["# AI 黄金数据集评测报告", "",
                 f"- run_id: `{m['run_id']}`  schema: `{m['schema_version']}`",
                 f"- 被测模型: `{m['target']['provider']}/{m['target']['model']}`",
                 f"- Judge模型: `{m['judge']['provider']}/{m['judge']['model']}`"
                 if m["judge"] else "- Judge模型: 未启用",
                 f"- 数据集: `{m['dataset']['path']}` sha256 `{m['dataset']['sha256'][:12]}`",
                 f"- 门禁配置: `{m['gates']['path']}` sha256 `{m['gates']['sha256'][:12]}`",
                 f"- 代码版本: `{(m['git_sha'] or 'N/A')[:8]}` 脏工作区=`{m['git_dirty']}`",
                 f"- 解码参数: temperature={m['decoding']['temperature']} "
                 f"seed={m['decoding']['seed']} repeats={m['decoding']['repeats']}",
                 f"- 耗时 {s['infra']['duration_s']}s  平均延迟 {s['infra']['avg_latency_s']}s "
                 f"缓存命中 {s['infra']['cache_hits']}"
                 f"tokens {s['infra']['prompt_tokens']}+{s['infra']['completion_tokens']} "
                 f"预估成本 ${s['infra']['cost_estimate_usd']}",
                 "", "## 指标（含覆盖率）", "",
                 "| 指标 | 得分 | 有效样本 | 不适用 | 计算失败 | 覆盖率 | 门禁 |", "|---|---|---|---|---|---|---|"]
        for key, stat in s["metric_stats"].items():
            lines.append(
                f"| {key} | {self._fmt(stat['value'])} | {stat['n']} | "
                f"{stat['not_applicable']} | {stat['errors']} | "
                f"{self._fmt(stat['coverage'])} | "
                f"{self._gate_cell(stat, report.gates.get(key))} |")

        if g.blocker_failures:
            lines += ["", "## ⛔ Blocker 用例失败（一票否决）", ""]
            lines += [f"- `{x}`" for x in g.blocker_failures]

        lines += ["", "## 门禁结论", "", "✅ 通过" if g.passed else "❌ 未通过", ""]
        lines += [f"- `{v.metric}`（{v.kind}）：{v.reason}" for v in g.violations]
        if g.skipped:
            lines += ["", f"跳过（无适用样本且非必需）：{', '.join(g.skipped)}"]

        if report.by_tag:
            lines += ["", "## 分 tag 指标", "", "| tag | 指标 | 值 |", "|---|---|"]
            for tag, view in report.by_tag.items():
                for name, value in view["metrics"].items():
                    lines.append(f"| {tag} | {name} | {self._fmt(value)} |")
        return "\n".join(lines) + "\n"


    def _console(self, report) -> None:
        s, g = report.summary, report.gate
        print("\n================ 评测摘要 ================")
        for key, stat in s["metric_stats"].items():
            print(f"  {key:<20}: {self._fmt(stat['value']):>8}  "
                  f"覆盖率 {self._fmt(stat['coverage'])}")
        verdict = "PASS ✅" if g.passed else "FAIL ❌"
        print(f"  门禁: {verdict}")
        for v in g.violations:
            print(f"    - [{v.kind}] {v.metric}: {v.reason}")
        print("==========================================\n")