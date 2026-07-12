# Results

## Phase 1 (seeded) — the three tests (PLAN), single seed=0, Qwen2.5-Coder-3B

Run via `forge_kaggle.py` on a Kaggle T4. Reproducible (seed fixed). These are
positive signals at small scale, not a polished paper — see caveats.

| Test | Result | What it shows |
|---|---|---|
| **Grounding** | builder 14/16, speaker 14/16 | Two agents communicate through the native symbol channel and produce correct code. Builder gets a NEUTRAL function name (`solve`) — no task leak via the name. |
| **Test 2 — compositional generalization** | **10/22 (45%)** | The trained Builder decodes held-out tasks — novel symbol *combinations* it never trained on — via per-symbol meaning. Above the 0/22 lookup-table floor → language, not phrasebook. |
| **Test 3 — load-bearing self-explanation (HEADLINE)** | **13/22 (59%)** | The model explains each symbol itself (Builder decodes each single-symbol message → its operation). A COLD model (same base, `disable_adapter()`, never learned the language) decodes held-out messages from that explanation alone. |
| Model decode (glyphs → operation keys, "reader") | **17/22 (77%)** | Builder as classifier beats code-gen (15/22); residual = specific symbol confusions. The DEMO uses the deterministic reference decoder (100%) for reliable code; this is the reported model-decode number. |

Test 3 > test 2: an explicit self-explanation + one-shot lets a cold model
out-compose the trained Builder.

### What this is NOT (honest scope)
- **Seeded, not from-scratch.** From-scratch RL would not bootstrap (random Speaker →
  Builder can't decode → no gradient; measured). We ground each primitive to a symbol
  and SFT both agents, then it's used/evaluated. Claim: "seeded symbol language with a
  load-bearing self-explanation," not "forged from nothing."
- **Explanation is code, not English.** Test 3 explanation = the model's per-symbol
  *operation* (code). The PLAN ideal is an English explanation (needs a Phase-2
  metalinguistic step). Code-explanation is a valid but weaker form.
- **Small + single-seed.** 16 primitive symbols, 22 held-out tasks, one seed. Report
  variance over seeds before any strong claim.
- **No compression yet (test 1).** The forge has no gradient once messages are minimal;
  compression is a separate, unfinished phase.

## English baseline (Phase 0 smoke, 1.5B, CPU)
pass_rate 0.833, bytes_per_solved 676 — harness proof only, superseded by the above.

## Phase 2 — full finetune v2 (2026-07-10, seed 0)

Qwen2.5-Coder-3B + one multi-task LoRA (r=32, all-linear), 40,592 pairs
(sft2.jsonl: 68 primitives, 1,135 validated paraphrases), 3 epochs, Kaggle T4.

| metric | e0 | e1 | e2 |
|---|---|---|---|
| builder: glyphs→code on HELD-OUT compositions (tests executed) | 96.5% | 98.5% | **99.5%** |
| speaker: UNSEEN phrasings → exact glyph message | 84.5% | 85.5% | **86.5%** |
| zero-shot symbols (never trained) | 1% | 2% | 2% |

Gates: builder ≥90% PASSED (99.5). Speaker 86.5 vs 90 target — close; product
covers the gap with the intent front-end (95%) or paraphrase-pool patches.
Adapter: huggingface.co/robertkeus/glyph-adapters/tree/main/v2. Single seed so
far; seeds 1-2 pending for variance.

## Phase 2 — v3 finetune: argument slots + string operands (2026-07-12, seed 0)

Language v3: 86 primitives — 12 int-slotted (operand = digit glyphs 200-209,
multi-digit works: addn:100, minagen:18) + 6 string-slotted (operand = one
vocab glyph, 210-219: # - _ ! , ; a e o x). 50,448 pairs (sft3.jsonl, bank3:
9,408 tasks, 6,287 slotted; 1,427 validated paraphrases incl. {a} templates).
Same recipe: Qwen2.5-Coder-3B, one multi-task LoRA r=32, 3 epochs, T4.

| metric | e0 | e1 | e2 |
|---|---|---|---|
| builder: glyphs→code on HELD-OUT compositions (tests executed) | 98% | 98% | **99.5%** |
| speaker: UNSEEN phrasings → exact glyph message (incl. operand transcription) | 85.5% | 83.5% | **84.5%** |
| zero-shot symbols (never trained, incl. slotted ltn) | 5% | 4% | 2% |

Operand slots cost nothing: builder held-out identical to v2 (99.5%) even
though 2/3 of held-out compositions now carry operands the builder must place
into code, and the speaker must transcribe free-text numbers/separators into
digit/vocab glyphs (84.5% vs 86.5% — within noise of the added difficulty).
Zero-shot unchanged: never-trained opcodes (incl. slotted ltn) don't decode —
known limitation, same as v2. Adapter:
huggingface.co/robertkeus/glyph-adapters/tree/main/v3.
