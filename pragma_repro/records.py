from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


Scalar = str | int | float | bool
FieldValue = Scalar | list[Scalar] | tuple[Scalar, ...]


def parse_timestamp(value: datetime | str) -> datetime:
    if isinstance(value, datetime):
        return value
    candidate = value.replace("Z", "+00:00")
    return datetime.fromisoformat(candidate)


def ensure_list(value: FieldValue) -> list[Scalar]:
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


@dataclass(slots=True)
class LifelongEvent:
    key: str
    value: FieldValue
    timestamp: datetime | str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LifelongEvent":
        return cls(
            key=str(data["key"]),
            value=data["value"],
            timestamp=data["timestamp"],
        )


@dataclass(slots=True)
class EventRecord:
    timestamp: datetime | str
    fields: dict[str, FieldValue] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EventRecord":
        return cls(
            timestamp=data["timestamp"],
            fields=dict(data.get("fields", {})),
        )


@dataclass(slots=True)
class UserRecord:
    user_id: str
    evaluation_ts: datetime | str
    profile: dict[str, FieldValue] = field(default_factory=dict)
    lifelong: list[LifelongEvent] = field(default_factory=list)
    events: list[EventRecord] = field(default_factory=list)
    label: Any = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "UserRecord":
        return cls(
            user_id=str(data["user_id"]),
            evaluation_ts=data["evaluation_ts"],
            profile=dict(data.get("profile", {})),
            lifelong=[LifelongEvent.from_dict(item) for item in data.get("lifelong", [])],
            events=[EventRecord.from_dict(item) for item in data.get("events", [])],
            label=data.get("label"),
        )

