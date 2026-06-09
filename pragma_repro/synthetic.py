from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone

from .records import EventRecord, LifelongEvent, UserRecord


PLANS = ["standard", "plus", "premium", "metal"]
REGIONS = ["uk", "eu", "sg", "us"]
CURRENCIES = ["eur", "gbp", "usd"]
MERCHANTS = [
    ("metro", "metro commute"),
    ("netstream", "video subscription"),
    ("grocero", "fresh groceries"),
    ("caffix", "coffee stop"),
    ("fitloop", "fitness pass"),
    ("cloudbox", "cloud storage"),
    ("airrail", "business travel"),
]
MCCS = ["4111", "4899", "5411", "5814", "7997", "5734", "4511"]
SCREENS = ["home", "transfer", "cards", "vaults", "stocks", "crypto", "loans"]
CHANNELS = ["email", "push", "sms"]
SUBSCRIPTIONS = [
    ("netstream", "monthly video plan"),
    ("fitloop", "gym membership"),
    ("cloudbox", "cloud backup"),
]


def generate_synthetic_records(
    count: int,
    *,
    seed: int = 0,
    min_events: int = 16,
    max_events: int = 72,
) -> list[UserRecord]:
    rng = random.Random(seed)
    records: list[UserRecord] = []
    base_time = datetime(2025, 1, 1, tzinfo=timezone.utc)

    for index in range(count):
        evaluation_ts = base_time + timedelta(days=rng.randint(60, 420), hours=rng.randint(0, 23))
        created_ts = evaluation_ts - timedelta(days=rng.randint(120, 1800))
        plan = rng.choices(PLANS, weights=[0.45, 0.2, 0.2, 0.15])[0]
        region = rng.choice(REGIONS)
        currency = rng.choice(CURRENCIES)
        balance = max(0.0, rng.gauss(2200.0, 1700.0))
        balance_quantile = max(0, min(99, int(balance / 60.0)))
        has_recurring = rng.random() < 0.35
        recurring_pair = rng.choice(SUBSCRIPTIONS) if has_recurring else None

        profile = {
            "plan": plan,
            "service_region": region,
            "home_currency": currency,
            "balance_quantile": balance_quantile,
        }
        lifelong = [
            LifelongEvent(
                key="first_topup",
                value="seen",
                timestamp=created_ts + timedelta(days=rng.randint(0, 14)),
            ),
            LifelongEvent(
                key="first_card_payment",
                value="seen",
                timestamp=created_ts + timedelta(days=rng.randint(7, 60)),
            ),
        ]
        if rng.random() < 0.45:
            lifelong.append(
                LifelongEvent(
                    key="first_trade",
                    value="stocks",
                    timestamp=created_ts + timedelta(days=rng.randint(30, 300)),
                )
            )

        events: list[EventRecord] = []
        for _ in range(rng.randint(min_events, max_events)):
            kind = rng.choices(
                ["card_payment", "bank_transfer", "app_nav", "communication", "trade"],
                weights=[0.45, 0.18, 0.18, 0.12, 0.07],
            )[0]
            timestamp = evaluation_ts - timedelta(
                days=rng.randint(1, 365),
                hours=rng.randint(0, 23),
                minutes=rng.randint(0, 59),
            )
            if kind == "card_payment":
                merchant, description = rng.choice(MERCHANTS)
                amount = round(max(1.0, rng.gauss(38.0, 22.0)), 2)
                mcc = MCCS[MERCHANTS.index((merchant, description))]
                fields = {
                    "type": kind,
                    "direction": "out",
                    "merchant": merchant,
                    "amount": amount,
                    "currency": currency,
                    "mcc": mcc,
                    "description": f"{description} {rng.choice(['city', 'london', 'weekly', 'family'])}",
                }
            elif kind == "bank_transfer":
                fields = {
                    "type": kind,
                    "direction": rng.choice(["in", "out"]),
                    "counterparty": rng.choice(["salary", "friend", "landlord", "broker"]),
                    "amount": round(max(5.0, rng.gauss(160.0, 80.0)), 2),
                    "currency": currency,
                    "description": rng.choice(
                        ["salary payment", "shared rent", "expense split", "broker settlement"]
                    ),
                }
            elif kind == "app_nav":
                fields = {
                    "type": kind,
                    "screen": rng.choice(SCREENS),
                    "channel": "app",
                    "description": rng.choice(
                        ["opened screen", "reviewed limits", "checked vault", "edited card settings"]
                    ),
                }
            elif kind == "communication":
                fields = {
                    "type": kind,
                    "channel": rng.choice(CHANNELS),
                    "campaign": rng.choice(["loan_offer", "cashback", "travel", "savings"]),
                    "description": rng.choice(
                        ["opened reminder", "dismissed offer", "clicked notification", "read update"]
                    ),
                }
            else:
                fields = {
                    "type": kind,
                    "symbol": rng.choice(["AAPL", "MSFT", "NVDA", "SPY"]),
                    "direction": rng.choice(["buy", "sell"]),
                    "amount": round(max(20.0, rng.gauss(240.0, 120.0)), 2),
                    "currency": currency,
                    "description": rng.choice(
                        ["fractional stock order", "weekly investment", "sold position", "portfolio rebalance"]
                    ),
                }
            events.append(EventRecord(timestamp=timestamp, fields=fields))

        if recurring_pair is not None:
            recurring_merchant, recurring_description = recurring_pair
            base_amount = round(max(6.0, rng.gauss(14.0, 4.0)), 2)
            for month_offset in range(1, rng.randint(3, 6)):
                recurring_ts = evaluation_ts - timedelta(days=30 * month_offset + rng.randint(-2, 2))
                events.append(
                    EventRecord(
                        timestamp=recurring_ts,
                        fields={
                            "type": "card_payment",
                            "direction": "out",
                            "merchant": recurring_merchant,
                            "amount": base_amount,
                            "currency": currency,
                            "mcc": rng.choice(MCCS),
                            "description": recurring_description,
                        },
                    )
                )

        events.sort(key=lambda item: item.timestamp)

        recurring_hits = 0
        lookback_start = evaluation_ts - timedelta(days=120)
        for event in events:
            if event.timestamp < lookback_start:
                continue
            if event.fields.get("merchant") == (recurring_pair[0] if recurring_pair else None):
                recurring_hits += 1
        label = 1 if recurring_hits >= 2 else 0

        records.append(
            UserRecord(
                user_id=f"user-{index:05d}",
                evaluation_ts=evaluation_ts,
                profile=profile,
                lifelong=lifelong,
                events=events,
                label=label,
            )
        )

    return records


def split_records(records: list[UserRecord], *, train_fraction: float = 0.8) -> tuple[list[UserRecord], list[UserRecord]]:
    pivot = int(len(records) * train_fraction)
    return records[:pivot], records[pivot:]

