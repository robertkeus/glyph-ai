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
from glyph.channel import Native
from glyph.decode import decode
import re

from glyph.paraphrase import HELDOUT_VARIANT, V, english
from glyph.seed import PRIM_ORDER, prim_symbol
from glyph.policy import LoraPolicy
from glyph.tasks import load_tasks
from glyph.verifier import run_tests

MODEL = "Qwen/Qwen2.5-Coder-3B-Instruct"
CH = Native()
DEMO_INPUT = [3, -1, 2, 2, -5]
P = None


_KEYS = "\n".join(f"{p} = {V[p][0]}" for p in PRIM_ORDER)
_PSET = set(PRIM_ORDER)


def _intent(msg):
    """Free English → ordered primitive keys, via the base model's comprehension.
    The glyph message is then built DETERMINISTICALLY (no brittle Speaker step)."""
    q = ("List the operations the request performs, IN ORDER, as comma-separated "
         "keys from this menu (keys only, e.g. `evens, double`):\n" + _KEYS +
         f"\n\nRequest: {msg}\nKeys:")
    out = P.ask_base(q, max_new=32)
    return [k for tok in re.split(r"[,\s]+", out) if (k := tok.strip()) in _PSET]


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
            prims = _intent(msg)                          # base extracts ordered ops
            glyphs = "".join(prim_symbol(p) for p in prims)
            translation = None
        if not glyphs:
            return "Couldn't map that to the 16 operations — try phrasing like the examples."
        code = decode(glyphs)                             # language semantics → code (100% valid)
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
    ex = [t["prompt"].split("; ", 1)[1].rstrip(".") for t in load_tasks(split="heldout")[:6]]
    css = """
    .gradio-container {background:#0b0d12 !important; color:#e6e9ef !important;}
    #hdr h1 {font-size:1.6rem; margin:.2rem 0;}
    #hdr p {color:#8b94a7; margin:.2rem 0;}
    footer {display:none !important;}
    """

    def submit(msg, hist):
        return hist + [(msg, _respond(msg, hist))], ""

    with gr.Blocks(css=css, title="Glyph", theme=gr.themes.Base()) as ui:
        gr.Markdown("# Glyph\nAsk for a list-of-integers operation. Two agents talk in "
                    "a 4-byte glyph language (~99% fewer bytes than English) and out "
                    "comes working, executed Python.", elem_id="hdr")
        chat = gr.Chatbot(height=430)
        box = gr.Textbox(placeholder="e.g. keep positives, square them, then sum  —  or type glyphs",
                         show_label=False)
        gr.Examples(ex, box, label="Try one (held-out — never trained as a combo)")
        box.submit(submit, [box, chat], [chat, box])
    ui.launch(share=share)
