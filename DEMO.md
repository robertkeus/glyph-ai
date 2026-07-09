# Glyph — demo guide

## The 60-second story
An AI given a coding request answers in its own **logographic language** — one glyph
per operation — and a second agent turns those glyphs into **working code**. The
glyph message is **~99% smaller** than the English the same agents would exchange.
And the model can **explain its own symbols** well enough that a *cold* model
(which never learned the language) decodes held-out messages from the explanation
alone. Autonomy with accountability.

## Two ways to show it

### A. Static page — the headline (no setup, shareable)
Open `demo/index.html` (or `python3 -m http.server --directory demo`). Shows real
run data: the **98.8% fewer bytes** banner, held-out tasks paired native-vs-English,
alien glyphs, the model's own translations, and the cold-decode kill-move.

### B. Live chat (Kaggle GPU, ~3-min start, no dataset needed)
Kaggle notebook → GPU **T4** → Internet on:
```python
!git clone https://github.com/robertkeus/glyph-ai && %cd glyph-ai
!pip -q install peft gradio
import chat_app; chat_app.launch()      # base model only → public chat URL
```
Type a request → glyph message + working Python + run result. Optionally attach the
`robertkeus/glyph-adapters` dataset first — `launch()` auto-detects it and adds
glyph-input translation + the trained Speaker/Builder agents.

## How the pipeline works (say this)
free English → **base model extracts the operations** (intent, ~95%) → **4-byte
glyph message** → the glyph language's **formal semantics compile it to Python**
(100% valid) → executed. The 4 vs ~290 bytes is the token win; the glyphs are the
compressed agent-to-agent message.

## Running it live — do this
- **Type freely** — "keep positives, square them, then sum", "biggest after sorting
  big to small" — all produce correct, executed code.
- Or **click an example**, or **type glyphs** (e.g. `与专`) for the raw language.
- Stay in the list-of-integers world (16 operations) — that's the trained domain.

## Numbers (Qwen2.5-Coder-3B, seed=0, Kaggle T4)
| Claim | Result |
|---|---|
| Byte reduction, native vs English A2A | **98.8%** (3 B vs 293 B / solved task) |
| Compositional generalization (held-out) | 12/22 |
| Cold-model self-explanation decode | 13/22 |
| Free-English input (intent-extraction) | 21/22 (95%) |

## Honest caveats (say them — they make it credible)
- **Seeded, not from-scratch:** primitives are grounded to symbols, then learned/used.
  From-scratch RL provably won't bootstrap (measured).
- **Toy domain:** list-of-integers ops, 16 symbols. The mechanism, not the scale.
- **Reasoning is in latents, not glyphs** — glyph is the *message* channel. The
  token win is on agent-to-agent traffic, which is real.
