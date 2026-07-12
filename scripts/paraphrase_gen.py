"""LLM paraphrase generation + round-trip validation (TRAINING_DATA_PLAN §3).

Per primitive: generate ~15 phrasings with Claude, then keep only those a judge
maps back to the exact primitive key (round-trip check) — no unverified data.
Writes glyph/paraphrases.json {key: [phrasings...]}. Resumable (skips done keys).

    ANTHROPIC_API_KEY must be set in the environment.
    .venv/bin/python scripts/paraphrase_gen.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import anthropic

from glyph.lang import BY_KEY
from glyph.tasks import ROOT

MODEL = "claude-haiku-4-5-20251001"
OUT = ROOT / "glyph" / "paraphrases.json"
N = 15

client = anthropic.Anthropic()
MENU = "\n".join(f"{k} = {p.en[0]}" for k, p in BY_KEY.items())


def ask(prompt, max_tokens=1000):
    r = client.messages.create(model=MODEL, max_tokens=max_tokens,
                               messages=[{"role": "user", "content": prompt}])
    return r.content[0].text.strip()


def generate(key):
    p = BY_KEY[key]
    need = ["{a}", "{b}"][:p.slots]
    slot = (f" Use the literal placeholder(s) {' and '.join(need)} for the operand(s), "
            f"in every phrasing (write them exactly as given)." if p.slots else "")
    q = (f"Operation on {p.tin}: \"{p.en[0]}\" (code: `{p.py}`).\n"
         f"Write {N} short, varied English phrasings a real user might type for exactly "
         f"this operation — mix formal and casual, include one terse and one wordy. "
         f"One per line, no numbering, no quotes.{slot}")
    lines = [l.strip("-• ").strip() for l in ask(q).splitlines() if l.strip()]
    if p.slots:  # must survive .format(): only the required placeholder braces allowed
        lines = [l for l in lines
                 if all(l.count(n) >= 1 for n in need)
                 and l.count("{") == l.count("}") == sum(l.count(n) for n in need)]
    return [l for l in lines if 3 <= len(l) <= 90][:N]


def validate(key, phrasings):
    """Round-trip: judge must map each phrasing back to exactly this key."""
    kept = []
    for batch_start in range(0, len(phrasings), 5):
        batch = phrasings[batch_start:batch_start + 5]
        numbered = "\n".join(f"{i+1}. {s}" for i, s in enumerate(batch))
        q = (f"Menu of operation keys:\n{MENU}\n\nFor each phrase below, answer with the "
             f"single best-matching key (or NONE), one per line, keys only.\n{numbered}")
        answers = [a.strip().lower() for a in ask(q, 200).splitlines() if a.strip()]
        for s, a in zip(batch, answers):
            if a.split()[-1].strip(".") == key:
                kept.append(s)
    return kept


def main():
    done = json.loads(OUT.read_text()) if OUT.exists() else {}
    keys = [k for k in BY_KEY if k not in done]
    print(f"{len(done)} done, {len(keys)} to go")
    for i, key in enumerate(keys):
        try:
            cands = generate(key)
            kept = validate(key, cands)
            # originals always included; LLM ones only if round-trip-validated
            done[key] = list(dict.fromkeys(list(BY_KEY[key].en) + kept))
            OUT.write_text(json.dumps(done, indent=1, ensure_ascii=False))
            print(f"[{i+1}/{len(keys)}] {key}: {len(cands)} generated, {len(kept)} kept "
                  f"-> {len(done[key])} total")
        except anthropic.APIError as e:
            print(f"[{i+1}/{len(keys)}] {key}: API error {e}; stopping (resumable)")
            break
    total = sum(len(v) for v in done.values())
    print(f"paraphrases.json: {len(done)} primitives, {total} phrasings")


if __name__ == "__main__":
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY not set — export it in your shell first")
    main()
