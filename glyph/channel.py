"""The Speaker→Builder channel — English (baseline) and Native (forged).

Density is ALWAYS scored in bytes, never tokens (PLAN non-negotiable): a native
symbol costs SYMBOL_BYTES on the wire regardless of how its demo glyph happens to
UTF-8 encode. The glyph is a font; the symbol-ID is the language.
"""
from abc import ABC, abstractmethod

SYMBOL_BYTES = 2          # ~2-4k inventory → 2 bytes / symbol-ID
INVENTORY_SIZE = 2048
_GLYPH_BASE = 0x4E00      # cosmetic render base (PLAN Open Decision #3); CJK = alien to a Western audience, monospace


def glyph(sym: int) -> str:
    """Render a symbol-ID as its demo glyph. Cosmetic only."""
    if not 0 <= sym < INVENTORY_SIZE:
        raise ValueError(f"symbol {sym} outside 0..{INVENTORY_SIZE - 1}")
    return chr(_GLYPH_BASE + sym)


def channel_bytes(symbols) -> int:
    """Wire bytes for a sequence of symbol-IDs (the `bytes` term in Phase 1 reward)."""
    return len(symbols) * SYMBOL_BYTES


class Channel(ABC):
    name: str

    @abstractmethod
    def speaker_hint(self) -> str:
        """Appended to the Speaker prompt — how to phrase the message."""

    @abstractmethod
    def builder_text(self, message: str) -> str:
        """What the Builder actually reads (it never sees the task)."""

    @abstractmethod
    def bytes(self, message: str) -> int:
        """Honest wire size of the message."""


class English(Channel):
    name = "english"

    def speaker_hint(self):
        return "Write the message in plain English. Do not write code."

    def builder_text(self, message):
        return message

    def bytes(self, message):
        return len(message.encode())


class Native(Channel):
    name = "native"

    def __init__(self, size: int = INVENTORY_SIZE):
        self.size = size

    @property
    def glyphs(self):
        return [glyph(i) for i in range(self.size)]

    def is_symbol(self, ch: str) -> bool:
        return _GLYPH_BASE <= ord(ch) < _GLYPH_BASE + self.size

    def speaker_hint(self):
        return "Reply ONLY with channel symbols — no English, no code."

    def builder_text(self, message):
        return message  # glyph string; the real Builder reads the underlying symbol-IDs

    def bytes(self, message):
        return channel_bytes([c for c in message if self.is_symbol(c)])

    def allowed_token_ids(self, tokenize) -> set:
        """The hard decode-time mask (PLAN §9): the set of model token-IDs that
        encode channel glyphs. Speaker generation is restricted to these — zero
        existing-vocab leakage. `tokenize(str) -> list[int]` is the model's
        encoder (injected; kept here so the mask LOGIC is GPU-free testable).
        Only single-token glyphs are admissible.
        """
        ids = set()
        for g in self.glyphs:
            t = tokenize(g)
            if len(t) == 1:
                ids.add(t[0])
        return ids
