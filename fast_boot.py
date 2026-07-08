"""Fast-load boot test: load pre-trained adapters from the dataset, NO warmup.

Proves the production instant-start path: chat responds in ~1-2 min instead of 15.
"""
import glob

import chat_app
from glyph.channel import Native
from glyph.policy import LoraPolicy

MODEL = "Qwen/Qwen2.5-Coder-3B-Instruct"
ROOT = "/kaggle/input"

print("input tree:", glob.glob(f"{ROOT}/**/speaker/adapter_config.json", recursive=True))
p = LoraPolicy(MODEL, channel=Native())
p.load(chat_app._resolve_adapters(ROOT))
chat_app.P = p
print("=== FAST BOOT (loaded from dataset, no warmup) ===")
for m in ["keep the even numbers, then double each",
          "sort descending, then return the maximum", "三下"]:
    print(f"\nUSER: {m}\n{chat_app._respond(m, None)}")
print("\nFAST BOOT OK")
