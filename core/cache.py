"""请求级磁盘缓存：重跑时复用已有输出，省掉重复的模型与 judge 费用。"""
from __future__ import annotations

import hashlib
import json
import threading

from pathlib import Path

class DiskCache:
    def __init__(self, path: str | None = None, refresh: bool = False):
        self.refresh = refresh
        self._path = Path(path) if path else None
        self._lock = threading.Lock()
        self._data: dict[str, dict] = {}

        if self._path and self._path.exists():
            for line in self._path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    key, value = json.loads(line)
                    self._data[key] = value

    @staticmethod
    def fingerprint(**parts) -> str:
        blob = json.dumps(parts, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def get(self, key: str) -> dict | None:
        return self._data.get(key)

    def put(self, key:str, value: dict) -> None:
        with self._lock:
            self._data[key] = value
            if self._path:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                with self._path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps([key, value], ensure_ascii=False) + "\n")


    def __len__(self) -> int:
        return len(self._data)