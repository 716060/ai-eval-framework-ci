"""
用例执行器：并发、断点续跑、重复跑稳定性度量

原方案在续跑时把 cases.jsonl 读回来的裸 dict 和新构造的
CaseResult 混进同一个列表，_summarize 的属性访问会直接炸。
这里统一走 from_payload 还原，类型始终唯一。
"""
from __future__ import annotations

import json
import threading

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

from utils.logger import get_logger

from core.case import CaseResult
from core.golden_dataset import GoldenRecord

logger = get_logger("runner")

def run_key_of(record: GoldenRecord, index: int, repeats: int) -> str:
    return record.id if repeats == 1 else f"{record.id}#{index}"

def load_checkpoint(path: Path) -> dict[str, CaseResult]:
    done: dict[str, CaseResult] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        case = CaseResult.from_payload(json.loads(line))
        done[case.run_key or case.record_id] = case
    return done

def run_cases(records: list[GoldenRecord], fn: Callable[[Any], CaseResult],
              workers: int = 8, checkpoint_dir: str | None = None,
              resume: bool = True, repeats: int = 1, progress_every: int = 10) -> list[CaseResult]:
    ckpt = Path(checkpoint_dir) / "cases.jsonl" if checkpoint_dir else None
    done: dict[str, CaseResult] = {}
    if ckpt and resume and ckpt.exists():
        done = load_checkpoint(ckpt)
        logger.info("断点续跑：复用 %d, %d 条已完成用例", len(done), len(records))

    todo: list[tuple[GoldenRecord, str]] = []
    for rec in records:
        for index in range(repeats):
            key = run_key_of(rec, index, repeats)
            if key not in done:
                todo.append((rec, key))

    if ckpt:
        ckpt.parent.mkdir(parents=True, exist_ok=True)
    logger.info("待执行 %d 条（总用例 %d，重复 %d 次）", len(todo), len(records), repeats)

    results: list[CaseResult] = list(done.values())
    lock = threading.Lock()

    def _run_one(rec: GoldenRecord, key: str) -> CaseResult:
        case = fn(rec)
        case.run_key = key
        return case

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_run_one, rec, key): key for rec, key in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            case = fut.result()
            with lock:
                results.append(case)
                if ckpt:
                    with ckpt.open("a", encoding="utf-8") as fh:
                        fh.write(json.dumps(case.to_payload(), ensure_ascii=False) + "\n")
            if i % progress_every == 0 or i == len(todo):
                logger.info("评测进度 %d, %d", i, len(todo))
    return results
