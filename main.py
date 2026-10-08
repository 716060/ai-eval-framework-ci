"""CLI 入口。退出码：0 通过 / 1 质量门禁失败 / 2 基础设施错误 / 3 数据集非法 / 4 配置错误。"""
from __future__ import annotations

import argparse
import sys


from core.baseline import compare_with_baseline
from core.evaluator import AIEvaluator
from core.exit_codes import (EXIT_CONFIG_ERROR, EXIT_DATASET_INVALID,
                             EXIT_GATE_FAILED, EXIT_INFRA_ERROR, EXIT_OK)
from core.golden_dataset import GoldenDatasetValidationError
from core.contract import LLMClientError
from core.reporter import Reporter
from utils.logger import get_logger

logger = get_logger("main")

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="AI 黄金数据集自动化评测框架")
    p.add_argument("--target-model", choices=["deepseek", "qwen"], default="deepseek")
    p.add_argument("--judge-model", choices=["deepseek", "qwen"], default="qwen")
    p.add_argument("--dataset", default="data/golden_dataset.json")
    p.add_argument("--gates", default=None)
    p.add_argument("--no-deepeval", action="store_true")
    p.add_argument("--output-dir", default="reports")
    p.add_argument("--tag", action="append", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--refresh-cache", action="store_true")
    p.add_argument("--checkpoint-dir", default=None)
    p.add_argument("--baseline", default=None)
    p.add_argument("--max-regression", type=float, default=0.02)
    p.add_argument("--allow-self-judge", action="store_true")
    p.add_argument("--smoke", action="store_true", help="冒烟：不跑 DeepEval、不落基线")
    return p.parse_args(argv)

def main():
    args = parse_args()
    if args.smoke:
        args.no_deepeval = True

    logger.info("启动测评：target=%s judge=%s dataset=%s workers=%d",
                args.target_model, args.judge_model, args.dataset, args.workers)

    try:
        evaluator = AIEvaluator(
            target_provider=args.target_model,
            judge_provider=args.judge_model,
            dataset_path=args.dataset,
            gates_path=args.gates,
            enable_deepeval=not args.no_deepeval,
            workers=args.workers,
            repeats=args.repeats,
            seed=args.seed,
            tags=args.tag,
            limit=args.limit,
            cache_path=None if args.no_cache else ".cache/llm_cache.jsonl",
            refresh_cache=args.refresh_cache,
            checkpoint_dir=args.checkpoint_dir,
            allow_self_judge=args.allow_self_judge,
        )
    except LLMClientError as exc:
        logger.error("基础设施配置错误：%s", exc)
        return EXIT_INFRA_ERROR
    except (ValueError, FileNotFoundError, PermissionError) as exc:
        # 自评护栏，门禁文件违法等
        logger.error("配置错误：%s", exc)
        return EXIT_CONFIG_ERROR
    except ImportError as exc:
        logger.error("依赖缺失：%s", exc)
        return EXIT_CONFIG_ERROR

    try:
        report = evaluator.run()
    except GoldenDatasetValidationError as exc:     # 注意：它是 ValueError 子类，必须先捕获
        logger.error("数据集不合法，已阻断：%s", exc)
        return EXIT_DATASET_INVALID
    except (FileNotFoundError, PermissionError, ValueError) as exc:
        logger.error("配置或数据文件错误：%s", exc)
        return EXIT_CONFIG_ERROR
    except LLMClientError as exc:
        logger.error("模型调用失败：%s", exc)
        return EXIT_INFRA_ERROR

    paths = Reporter(args.output_dir).dump(report)
    logger.info("报告产物：%s", paths)

    regression_failed = False
    if args.baseline:
        reg = compare_with_baseline(report.summary, report.gates,
                                    tolerance=args.max_regression,
                                    baseline_path=args.baseline)
        logger.info("基线对比：%s", reg.summary_line())
        regression_failed = reg.has_regression

    if report.infra_error_rate > 0:
        logger.error("存在 %.2f%% 的基础设施错误，结果不完整，请重跑",
                     report.infra_error_rate * 100)
        return EXIT_INFRA_ERROR
    if not report.gate.passed:
        for v in report.gate.violations:
            logger.error("门禁失败 [%s] %s：%s", v.kind, v.metric, v.reason)
        return EXIT_GATE_FAILED
    if regression_failed:
        logger.error("阈值门禁通过，但存在超出容差的指标回退")
        return EXIT_GATE_FAILED

    logger.info("质量门禁全部通过 ✅（判定 %d 项，跳过 %s）",
                report.gate.evaluated, report.gate.skipped or "无")
    return EXIT_OK


if __name__ == '__main__':
     sys.exit(main())