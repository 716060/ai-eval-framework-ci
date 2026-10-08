"""评测编排器：黄金数据集 -> 模型推演 -> 手写指标 + DeepEval指标 -> 聚合 -> 质量门禁 -> 基线。"""
from __future__ import annotations

import time
import json

from dataclasses import dataclass, field
from utils.logger import get_logger

from core.runner import run_cases
from config.settings import load_gates
from core.aggregate import MetricBundle
from core.cache import DiskCache
from core.case import CaseResult
from core.contract import ChatModel, InferenceError, LLMResponse
from core.gates import GateReport, evaluate_gates
from core.golden_dataset import GoldenRecord, load_golden_dataset
from core.llm_client import build_client
from core.manifest import build_manifest, dataset_sha256
from core.metrics import build_metric_for


logger = get_logger("evaluator")

SYSTEM_PROMPT = "你是一个严谨的助手, 严格按用户要求的格式作答, 不输出与任务无关的内容。"

EXTRACTION_ENVELOPE_KEY = "items"
TEMPERATURE = 0.0

# 各模型每百万 token 价格（示例值，接入前请按实际合同价替换）
PRICE_TABLE = {"deepseek": (0.27, 1.10), "qwen": (0.40, 1.20)}

import inspect

def _assert_chat_model(model: object, role: str) -> None:
    """runtime_checkable 只校验方法名，这里显示校验真实调用契约。"""
    chat = getattr(model, "chat", None)
    if not callable(chat):
        raise TypeError(
            f"{role} 必须实现 chat(prompt, system_prompt=None, json_model=False, temperature=0.0, max_tokens=None) -> LLMResponse;"
            f"实际类型 {type(model).__name__} 只提供了"
            f"{[m for m in ('chat', 'generate') if hasattr(model, m)]}")
    if "prompt" not in inspect.signature(chat).parameters:
        raise TypeError(f"{role}.chat 缺少 prompt 参数")

@dataclass
class EvalReport:
    manifest: dict
    summary: dict
    by_task_type: dict
    by_tag: dict
    cases: list[CaseResult] = field(default_factory=list)
    gate: GateReport | None = None
    gates: dict = field(default_factory=dict)

    @property
    def gate_passed(self) -> bool:
        return bool(self.gate and self.gate.passed)

    @property
    def gate_violations(self) -> list:
        return self.gate.violations if self.gate else []

    @property
    def infra_error_rate(self) -> float:
        return self.summary["infra"]["infra_error_rate"]


class AIEvaluator:
    def __init__(self, target_provider: str = "deepseek", judge_provider: str = "qwen",
                 dataset_path: str = "data/golden_dataset.json", enable_deepeval: bool = True,
                 gates_path: str | None = None, client: ChatModel | None = None, judge_client: ChatModel | None = None,
                 workers: int = 8, cache_path: str | None = ".cache/llm_cache.jsonl",
                 refresh_cache: bool = False, checkpoint_dir: str | None = None,
                 repeats: int = 1, seed: int | None = None, tags: list[str] | None = None,
                 limit: int | None = None, deepeval_threshold: float = 0.7,
                 allow_self_judge: bool = False) -> None:

        self.dataset_path = dataset_path
        self.gates_path = gates_path
        self.gates, self.gates_sha, self.gates_source = load_gates(self.gates_path)
        self.workers = workers
        self.checkpoint_dir = checkpoint_dir
        self.repeats = repeats
        self.seed = seed
        self.tag_filter = set(tags or [])
        self.limit = limit
        self.allow_self_judge = allow_self_judge

        cache = DiskCache(cache_path, refresh=refresh_cache)
        self.target = client or build_client(target_provider, cache=cache, seed=seed)
        _assert_chat_model(self.target, "target")
        self.judge_client: ChatModel | None = None
        self.target_provider = target_provider
        self.judge_provider = judge_provider
        self.suite = None

        if enable_deepeval:
            from core.deepeval_suite import DEEPEVAL_AVAILABLE, DeepEvalSuite
            if DEEPEVAL_AVAILABLE:
                # 原方案：judge 客户端无条件构建
                # ⬆ --no-deepeval 也必须配 QWEN_API_KEY，纯手写指标跑不起来、无法离线测试
                self.judge_client = judge_client or build_client(judge_provider, cache=cache)
                _assert_chat_model(self.judge_client, "judge_client")
                self.suite = DeepEvalSuite(judge_provider, deepeval_threshold,
                                           judge_client=self.judge_client)
            else:
                logger.warning("未安装 deepeval，通用指标跳过")
        self._assert_judge_separation()

    def _assert_judge_separation(self) -> None:
        """被测与判官同源会引入自评偏好，企业评测必须显示豁免才允许"""
        if self.allow_self_judge or self.suite is None:
            return
        target_model = getattr(self.target, "model_name", "")
        judge_model = getattr(self.judge_client, "model_name", "")
        if self.target_provider == self.judge_provider and target_model == judge_model:
            raise ValueError(
                f"judge 与被测模型同源（{self.target_provider}/{target_model}），"
                f"会引入自评偏差，请换 judge，或显示传 --allow-self-judge 豁免。")

    # ------ 单用例执行：任何异常都必须被归类，不能变成 0 分 ------
    def _evaluate_case(self, record: GoldenRecord) -> CaseResult:
        base = dict(record_id=record.id, task_type=record.task_type, input=record.input,
                    expected_output=record.expected_output, tags=list(record.tags),
                    severity=record.severity)
        try:
            resp = self._infer(record)
        except InferenceError as exc:
            logger.error("用例 %s 推理失败（计入基础设施错误）：%s", record.id, exc)
            return CaseResult(actual_output=None, error={
                "kind": "inference", "type": type(exc).__name__, "message": str(exc)}, **base)
        except Exception as exc:   #noqa: BLE001 框架 bug 也要可见，而不是静默 0 分
            logger.exception("用例 %s 出现未知错误", record.id)
            return CaseResult(actual_output=None, error={
                "kind": "unexpected", "type": type(exc).__name__, "message": str(exc)}, **base)

        parsed = self._try_parse(resp.text)

        # 原：handwritten[res.name] = asdict(res)   <-   对象被降级成dict，派生属性丢失
        handwritten = {m.name: m.measure(record, resp.text, parsed)
                       for m in build_metric_for(record.task_type)}

        semantic = self.suite.evaluate(record, resp.text) if self.suite else {}

        return CaseResult(
            actual_output=resp.text, handwritten=handwritten, semantic=semantic,
            usage={"prompt_tokens": resp.prompt_tokens, "completion_tokens": resp.completion_tokens,
                   "latency_s": resp.latency_s, "attempts": resp.attempts, "cached": resp.cached},
            **base)

    @staticmethod
    def _build_prompt(record) -> tuple[str, bool]:
        prompt, json_model = record.input, record.format_spec.get("type") == "json"
        if record.task_type == "extraction":
            prompt += (f"\n【格式要求】仅输出一个 JSON 对象，形如 "
                       f'{{"{EXTRACTION_ENVELOPE_KEY}": ["a", "b"]}},'
                       f"不要 markdown 代码块，不要任何解释。")
            json_model = True
        elif json_model:
            prompt += "\n【格式要求】仅输出合法 JSON，不要代码块或解释。"

        # 新增：分类是闭集任务，必须把候选标签显示写进 prompt，
        # 否则模型会按自己的理解输出解释性长文
        if record.task_type == "classification" and record.allowed_labels:
            labels = "、".join(record.allowed_labels)
            prompt += (f"\n【格式要求】只能从一下标签中选择一个，直接输出标签本身，"
                       f"不要任何解释、前缀、编号或标点：{labels}")
        return prompt, json_model



    # 推理
    def _infer(self, record) -> LLMResponse:
        prompt, json_model = self._build_prompt(record)
        return self.target.chat(prompt, system_prompt=SYSTEM_PROMPT, json_model=json_model,
                                temperature=TEMPERATURE)
    @staticmethod
    def _try_parse(raw: str):
        """稳健解析：容忍代码块包裹与前后缀文本。

        注意：返回「信封层」原始结构，不做 items 解包。
        信封合法性由 FormatMatchMetric 的 required_keys 校验，
        业务值由 SetPrecisionRecallMetric 自行解包 dict。
        """
        clean = raw.strip()
        if clean.startswith("```"):
            clean = clean.strip("`")
            if clean.lower().startswith("json"):
                clean = clean[4:]
        try:
            return json.loads(clean)
        except json.JSONDecodeError:
            for lo, hi in (("{", "}"), ("[", "]")):
                start, end = clean.find(lo), clean.rfind(hi)
                if start != -1 and end > start:
                    try:
                        return json.loads(clean[start:end + 1])
                    except json.JSONDecodeError:
                        continue
        return None


    # 主流程
    def run(self) -> EvalReport:
        started = time.time()
        records, vreport = load_golden_dataset(self.dataset_path)
        records = self._filter(records)
        if not records:
            raise ValueError("过滤后没有可执行用例，请检查 --tag / --limit")
        if self.repeats > 1:
            logger.warning(
                "repeats=%d：确定性解码（temperature=0）+ 缓存命中时重复采样必然一致，"
                "只有提高 temperature 或关闭缓存才能测出稳定性", self.repeats)

        results = run_cases(records, self._evaluate_case, workers=self.workers,
                            checkpoint_dir=self.checkpoint_dir, repeats=self.repeats)
        summary = self._summarize(results, vreport, time.time() - started)
        gate = evaluate_gates(self.gates, summary,
                              blocker_failures=self._block_failures(results))
        manifest = build_manifest(
            dataset_path=self.dataset_path, dataset_sha=dataset_sha256(self.dataset_path),
            gates_path=self.gates_source, gates_sha=self.gates_sha,
            target=self.target, judge=self.judge_client if self.suite else None,
            deepeval_enabled=self.suite is not None, seed=self.seed,
            repeats=self.repeats, workers=self.workers, temperature=TEMPERATURE,
            started_at=started, finished_at=time.time()
        )
        bundle = self._last_bundle
        return EvalReport(
            manifest=manifest, summary=summary,
            by_task_type=self._group_view(bundle.by_task_type),
            by_tag=self._group_view(bundle.by_tag),
            cases=results,
            gate=gate, gates=self.gates,
        )

    def _filter(self, records: list[GoldenRecord]) -> list[GoldenRecord]:
        out = records
        if self.tag_filter:
            out = [r for r in out if self.tag_filter & set(r.tags)]
        if self.limit:
            out = out[:self.limit]
        return out

    @staticmethod
    def _block_failures(results: list[CaseResult]) -> list[str]:
        """
        blocker 一票否决：只有 [算出来但没通过] 才算作答失败。
        status="error"（指标算不出来）单独标注，不与答错混为一谈。
        """
        labels = []
        for case in results:
            if case.error or case.severity != "blocker":
                continue
            failed = case.failed_metrics()
            if failed:
                labels.append(f"{case.record_id}[{'/'.join(sorted(failed))}]")
            elif case.errored_metrics():
                labels.append(f"{case.record_id}[指标计算失败：{'/'.join(sorted(case.errored_metrics()))}]")
        return labels

    def _summarize(self, results: list[CaseResult], vreport, duration:float):
        bundle = MetricBundle()
        infra_errors = 0
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "latency_s": 0.0, "cached": 0}

        for case in results:
            if case.error:
                infra_errors += 1
                continue
            bundle.add_case(case.handwritten, case.semantic, case.task_type, case.tags)

            for key in usage:
                usage[key] += case.usage.get(key, 0) or 0

        total = len(results)
        infra_rate = round(infra_errors / total, 4) if total else 1.0
        metrics = bundle.metrics
        stats = bundle.stats

        # 基础设施错误率作为一个受门禁约束的指标
        metrics["infra_error_rate"] = infra_rate
        stats["infra_error_rate"] = {
            "value": infra_rate,
            "n": total,
            "errors": 0,
            "not_applicable": 0,
            "applicable": total,
            "coverage": 1.0
        }

        self._last_bundle = bundle

        # 幻觉用例占比：从个案分数派生
        hall = [c.semantic.get("hallucination_rate", {}) for c in results
                if c.semantic.get("hallucination_rate") and c.semantic["hallucination_rate"].score is not None]
        hall_rate = (round(sum(1 for h in hall if h.score > 0) / len(hall), 4) \
                    if hall else None)

        p_price, c_price = PRICE_TABLE.get(self.target_provider, (0.0, 0.0))
        cost = (usage["prompt_tokens"] * p_price + usage["completion_tokens"] * c_price) / 1_000_000

        summary = {
            "total_cases": total,
            "metrics": metrics,
            "metric_stats": stats,
            "hallucination_case_rate": hall_rate,
            "blocker_failures": bundle.blocker_failures,
            "dataset_stats": vreport.stats,
            "infra": {
                "inference_errors": infra_errors,
                "infra_error_rate": infra_rate,
                "duration_s": round(duration, 2),
                "avg_latency_s": round(usage["latency_s"] / max(total - infra_errors, 1), 3),
                "cache_hits": usage["cached"],
                "prompt_tokens": usage["prompt_tokens"],
                "completion_tokens": usage["completion_tokens"],
                "cost_estimate_usd": round(cost, 4),
            }
        }

        return summary

    @staticmethod
    def _group_view(group: dict) -> dict:
        return {
            name: {"metrics": {k: b.mean for k, b in buckets.items()},
                   "metric_stats": {k: b.as_dict() for k, b in buckets.items()}}
            for name, buckets in group.items()

        }











