from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .records import UserRecord


def load_user_records(path: str | Path) -> list[UserRecord]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "records" in payload:
        payload = payload["records"]
    if not isinstance(payload, list):
        raise ValueError("Expected a JSON list of user records or {'records': [...]} payload.")
    return [UserRecord.from_dict(item) for item in payload]


def save_json(path: str | Path, payload: Any) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
