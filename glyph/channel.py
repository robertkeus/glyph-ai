SYMBOL_BYTES = 2  # ~2-4k inventory → 2 bytes/symbol-ID


def channel_bytes(symbol_ids) -> int:
    """Bytes a native message occupies on the Speaker→Builder channel.

    Density is ALWAYS scored in bytes, never tokens — the tokenizer can't be
    gamed this way. This is the `bytes` term in the Phase 1 reward.
    """
    return len(symbol_ids) * SYMBOL_BYTES
