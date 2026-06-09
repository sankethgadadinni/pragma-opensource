from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field


WORD_RE = re.compile(r"[A-Za-z0-9_]+|[^\sA-Za-z0-9_]")
END_OF_WORD = "</w>"


def basic_word_tokenize(text: str) -> list[str]:
    return WORD_RE.findall(text.lower())


def _merge_pair_in_word(symbols: tuple[str, ...], pair: tuple[str, str]) -> tuple[str, ...]:
    merged: list[str] = []
    i = 0
    while i < len(symbols):
        if i + 1 < len(symbols) and (symbols[i], symbols[i + 1]) == pair:
            merged.append(symbols[i] + symbols[i + 1])
            i += 2
        else:
            merged.append(symbols[i])
            i += 1
    return tuple(merged)


@dataclass(slots=True)
class BPETokenizer:
    vocab_size: int = 4096
    min_frequency: int = 2
    merges: list[tuple[str, str]] = field(default_factory=list)
    tokens: list[str] = field(default_factory=list)
    token_to_id: dict[str, int] = field(default_factory=dict)

    def fit(self, texts: list[str]) -> None:
        words = Counter()
        for text in texts:
            for word in basic_word_tokenize(text):
                symbols = tuple(list(word) + [END_OF_WORD])
                words[symbols] += 1

        if not words:
            self.tokens = []
            self.token_to_id = {}
            self.merges = []
            return

        symbols = set()
        for word in words:
            symbols.update(word)

        merges: list[tuple[str, str]] = []
        while len(symbols) < self.vocab_size:
            pair_counts: Counter[tuple[str, str]] = Counter()
            for word, count in words.items():
                for pair in zip(word, word[1:]):
                    pair_counts[pair] += count
            if not pair_counts:
                break
            best_pair, best_count = pair_counts.most_common(1)[0]
            if best_count < self.min_frequency:
                break
            words = Counter(
                {_merge_pair_in_word(word, best_pair): count for word, count in words.items()}
            )
            merges.append(best_pair)
            symbols = set()
            for word in words:
                symbols.update(word)

        ordered_tokens = sorted(symbols)
        self.merges = merges
        self.tokens = ordered_tokens
        self.token_to_id = {token: idx for idx, token in enumerate(ordered_tokens)}

    def encode(self, text: str) -> list[str]:
        output: list[str] = []
        for word in basic_word_tokenize(text):
            symbols = tuple(list(word) + [END_OF_WORD])
            for pair in self.merges:
                symbols = _merge_pair_in_word(symbols, pair)
            pieces = [piece for piece in symbols if piece != END_OF_WORD]
            if not pieces:
                continue
            output.extend(pieces)
        return output

    def to_dict(self) -> dict[str, object]:
        return {
            "vocab_size": self.vocab_size,
            "min_frequency": self.min_frequency,
            "merges": [list(pair) for pair in self.merges],
            "tokens": self.tokens,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "BPETokenizer":
        tokenizer = cls(
            vocab_size=int(data["vocab_size"]),
            min_frequency=int(data["min_frequency"]),
        )
        tokenizer.merges = [tuple(item) for item in data.get("merges", [])]  # type: ignore[arg-type]
        tokenizer.tokens = list(data.get("tokens", []))  # type: ignore[arg-type]
        tokenizer.token_to_id = {token: idx for idx, token in enumerate(tokenizer.tokens)}
        return tokenizer

