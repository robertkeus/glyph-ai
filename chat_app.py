"""Browser chat with Glyph — Gradio app, public share link (Kaggle GPU).

Kaggle notebook (GPU T4, Internet ON):
    !git clone https://github.com/robertkeus/glyph-ai && %cd glyph-ai
    !pip -q install peft gradio
    import chat_app; chat_app.launch()      # ~15 min warmup, then prints a public URL

Open the URL and type a list-of-integers request, e.g.:
    "keep the even numbers, then double each"      (English → glyphs → code)
    "sort descending, then return the maximum"
    or a glyph string directly.
Scope: it only speaks list operations (16 symbols). English works best near the
example phrasings; glyph input is the most reliable.
"""
from glyph.agents import _extract_code, builder_prompt, speaker_prompt
from glyph.channel import Native
from glyph.paraphrase import HELDOUT_VARIANT, V, english
from glyph.seed import PRIM_ORDER
from glyph.policy import LoraPolicy
from glyph.tasks import load_tasks
from glyph.verifier import run_tests

MODEL = "Qwen/Qwen2.5-Coder-3B-Instruct"
CH = Native()
DEMO_INPUT = [3, -1, 2, 2, -5]
P = None


_OPS = "\n".join(f"- {V[p][0]}" for p in PRIM_ORDER)


def _normalize(msg):
    """Free English → canonical task phrasing, via the base model's comprehension
    (the Speaker is only reliable on canonical-style phrasing)."""
    q = ("Rewrite the request as a pipeline of these list operations, in the exact "
         "form 'Take a list of integers; OP, then OP.' — use ONLY these operations:\n"
         + _OPS + f"\n\nRequest: {msg}\nRewrite:")
    out = P.ask_base(q)
    return out if "list of integers" in out else f"Take a list of integers; {msg}."


def _respond(msg, _history):
    if P is None:
        return "Model still warming up — try again in a moment."
    msg = (msg or "").strip()[:200]
    if not msg:
        return "Ask for a list-of-integers operation, e.g. *keep the even numbers, then double each*."
    try:
        if all(CH.is_symbol(c) for c in msg):            # user typed glyphs
            glyphs, translation = msg, P.translate(msg)
        else:                                            # user typed English
            task = {"prompt": _normalize(msg)}            # base model normalizes arbitrary phrasing
            glyphs = P.sample(speaker_prompt(task, CH), 1, greedy=True)[0]
            translation = None
        if not glyphs:
            return "The Speaker couldn't encode that — try phrasing it like the examples."
        code = _extract_code(P.build(builder_prompt(CH.builder_text(glyphs))))
        r = run_tests(code, f"print(solve({DEMO_INPUT}))")
        ran = r["stdout"] if r["passed"] else "error: " + r["detail"][:80]
    except Exception as e:  # never crash the chat
        return f"Error: {type(e).__name__}: {e}"
    out = [f"**glyph** `{glyphs}`  ·  {CH.bytes(glyphs)} bytes"]
    if translation:
        out.append(f"**model reads it as:** {translation}")
    out.append(f"```python\n{code}\n```")
    out.append(f"`solve({DEMO_INPUT})` → `{ran}`")
    return "\n\n".join(out)


def _resolve_adapters(root):
    """Find the dir containing speaker/ builder/ translator/ under root (robust to
    Kaggle dataset mount nesting)."""
    import glob, os
    hits = glob.glob(os.path.join(root, "**", "speaker", "adapter_config.json"),
                     recursive=True)
    if not hits:
        raise FileNotFoundError(f"no adapters under {root}")
    return os.path.dirname(os.path.dirname(hits[0]))


def launch(adapters="auto", rounds=10, share=True):
    """adapters=<dir> loads a trained checkpoint (seconds); "auto" uses the attached
    glyph-adapters dataset if present; None forces a fresh warmup (~15 min)."""
    import glob
    global P
    if adapters == "auto":
        adapters = "/kaggle/input" if glob.glob(
            "/kaggle/input/**/speaker/adapter_config.json", recursive=True) else None
    P = LoraPolicy(MODEL, channel=CH)
    if adapters:
        P.load(_resolve_adapters(adapters))
    else:
        P.warmup_seeded(load_tasks(split="train"), rounds=rounds,
                        english=lambda t, rd: english(t, rd % HELDOUT_VARIANT), translate=True)
    import gradio as gr
    ex = [t["prompt"].split("; ", 1)[1] for t in load_tasks(split="heldout")[:6]]
    gr.ChatInterface(
        _respond, title="Glyph — talk to a model that answers in its own language",
        description="Ask for a list-of-integers operation. It replies in glyphs, "
                    "then working Python. (16-symbol vocabulary; phrase like the examples.)",
        examples=ex,
    ).launch(share=share)
