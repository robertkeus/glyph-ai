"""Final proof: full chat pipeline on FREE-typed English → glyphs → working code."""
import chat_app
from glyph.channel import Native
from glyph.policy import LoraPolicy
p = LoraPolicy("Qwen/Qwen2.5-Coder-3B-Instruct", channel=Native())
p.load(chat_app._resolve_adapters("/kaggle/input"))
chat_app.P = p
for msg in ["biggest value after sorting big to small",
            "flip signs then put it backwards",
            "dedupe then add up everything",
            "keep positives, square them, then sum"]:
    print(f"\n>>> {msg}\n{chat_app._respond(msg, None)}")
