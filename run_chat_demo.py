"""Boot test: base-only English chat (intent + reference decode, NO adapters) +
verify the Gradio UI builds. Proves the ~3-min instant demo works."""
import chat_app
from glyph.channel import Native
from glyph.policy import LoraPolicy

chat_app.P = LoraPolicy("Qwen/Qwen2.5-Coder-3B-Instruct", channel=Native())  # base only
for msg in ["biggest value after sorting big to small",
            "flip signs then put it backwards",
            "dedupe then add up everything",
            "keep positives, square them, then sum"]:
    print(f"\n>>> {msg}\n{chat_app._respond(msg, None)}")

import gradio as gr  # noqa
chat_app.build_ui()  # constructs the Blocks UI (no launch) — catches version errors
print("\nUI BUILD OK")
