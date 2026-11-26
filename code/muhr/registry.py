"""Mualliflar reestri: 24 bitli ID <-> muallif ma'lumotlari (JSON faylda)."""
from __future__ import annotations

import json
import secrets
from datetime import datetime
from pathlib import Path


class AuthorRegistry:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.data: dict[str, dict] = {}
        if self.path.exists():
            self.data = json.loads(self.path.read_text(encoding="utf-8"))

    def _save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, ensure_ascii=False, indent=2), encoding="utf-8")

    def add(self, name: str, note: str = "") -> int:
        name = name.strip()
        if not name:
            raise ValueError("Muallif ismini kiriting")
        while True:
            aid = secrets.randbelow(2 ** 24 - 1) + 1
            if str(aid) not in self.data:
                break
        self.data[str(aid)] = {"name": name, "note": note.strip(),
                               "created": datetime.now().strftime("%Y-%m-%d %H:%M")}
        self._save()
        return aid

    def name(self, aid: int) -> str | None:
        rec = self.data.get(str(aid))
        return rec["name"] if rec else None

    def choices(self) -> list[tuple[str, int]]:
        return [(f"{v['name']} (ID {k})", int(k)) for k, v in self.data.items()]

    def rows(self) -> list[list]:
        return [[int(k), v["name"], v.get("note", ""), v.get("created", "")] for k, v in self.data.items()]
