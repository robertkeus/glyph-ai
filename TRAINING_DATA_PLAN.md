# Training-data plan — scaling the glyph language

Goal: 67 tasks → **5k–20k verified tasks**, 16 → **~120 primitives**, 6 → **15–20
paraphrases each**. This is the binding constraint on everything (free-English
robustness, compositional generalization, domain usefulness).

## 0. Read first — related work (verified on arXiv, 2026-07)

**Name collision (important):** *Glyph: Scaling Context Windows via Visual-Text
Compression* ([arXiv:2510.17800](https://arxiv.org/abs/2510.17800)) — an unrelated,
well-known Zhipu paper that owns the name "Glyph" in LLM circles (renders long text
as images for VLMs). **Consider renaming the project** (or subtitle it clearly) to
avoid being mistaken for it.

Directly on-thesis:
- *Why do AI agents communicate in human language?*
  ([arXiv:2506.02739](https://arxiv.org/abs/2506.02739)) — argues natural language
  is a suboptimal A2A medium; cite as motivation.
- *Optima: Optimizing Effectiveness and Efficiency for LLM-Based Multi-Agent
  System* — trains LLM agents for communication efficiency (the ~90% token-
  reduction figure PLAN.md referenced). Our closest quantitative baseline.
- *Beyond tokens: a unified framework for latent communication in LLM-based
  multi-agent systems* — the competing approach (continuous vectors, not discrete
  symbols). Our differentiator: discrete + human-inspectable + self-explainable.
- *AI Mother Tongue: Self-Emergent Communication in MARL via Endogenous Symbol
  Systems* — from-scratch symbol emergence in MARL; compare with our measured
  cold-start failure.
- *AgentDropout* — orthogonal token-saving lever (drop agents/turns, not compress
  messages); good "alternatives" section citation.

Emergent-language classics (for eval methodology, not implementation):
- *Compositionality and Generalization in Emergent Languages* (Chaabouni et al.)
- *Capacity, Bandwidth, and Compositionality in Emergent Language Learning*
- *Anti-efficient encoding in emergent communication* — why compression pressure
  must be explicit (our λ term).
- Surveys: *Beyond Self-Talk: A Communication-Centric Survey of LLM-Based
  Multi-Agent Systems*; *Searching for Structure: Investigating Emergent
  Communication with LLMs* / *Shaping Shared Languages* (LLM inductive biases).

Takeaways applied below: (1) explicit pressure or emergent codes don't compress;
(2) compositionality must be *measured*, it doesn't come free; (3) discrete
inspectable symbols are our niche vs latent-vector work; (4) rename.

## 1. Primitive families (16 → ~120)

Extend `taskgen.py` with new families, same contract as today: each primitive =
(key, one code line/snippet per target language, English description, N paraphrase
phrasings). Solutions and hidden tests are **derived by execution** — never
hand-written — so the reward stays ungameable.

| Family | Examples | ~Count |
|---|---|---|
| int-list (today) | evens, double, sortd, sum… | 16 |
| string ops | lower, strip, split, join, prefix-filter, replace | ~20 |
| string-list | dedupe-ci, sort-by-length, longest, concat | ~12 |
| dict/record ops | pluck field, filter-by-field, group-by, count-by | ~15 |
| numeric scalar | clamp, round, abs-diff, mean, median | ~12 |
| predicates | any/all/none matching, contains, is-sorted | ~10 |
| control-flow shapes | take-while, drop-while, first-match, split-at | ~12 |
| error paths | default-on-empty, raise-on-negative, try-parse | ~8 |
| composition helpers | pipe two named ops, conditional op | ~8 |
| I/O-ish (pure-sim) | parse CSV line, format report string | ~8 |

Each family gets **both Python and JS renderings** (multi-target decode is now the
"not a cipher" proof — keep it as an invariant + test).

## 2. Task composition (67 → 5k–20k)

- Chains of 1–4 primitives (today: ≤3), typed: a string op can't follow a
  reducer; dict ops need record inputs. Add a light **type system** to `taskgen`
  (each primitive declares in-type/out-type; composer only builds well-typed
  chains). This is the main new engineering.
- Inputs per type (int lists, strings, records…) with edge cases (empty, dupes,
  negatives, unicode).
- **Held-out compositional split carved before training** (existing
  `check_heldout_novelty` generalizes as-is): hold out (a) unseen pairs, (b) all
  chains of length 4, (c) 10% of primitives entirely (zero-shot symbols — new,
  stronger test).
- Verify every generated task with the existing smoke gate (solution passes,
  stub fails) — scales unchanged.

## 3. Paraphrases (6 → 15–20 per task, LLM-generated)

- Generate per-primitive phrase variants with an LLM (Claude API), not per-task:
  ~20 phrasings × 120 primitives ≈ 2.4k generations, cheap. Compose task
  paraphrases from primitive variants (today's mechanism, unchanged).
- Plus ~2k whole-task free-form paraphrases (colloquial: "get rid of the odd
  ones") for realism the compositional templates can't produce.
- **Auto-validate**: a paraphrase is kept only if an LLM judge maps it back to
  the exact primitive sequence (round-trip check) — no unverified data enters
  training. Hold out 20% of phrasings per primitive for the robustness eval.

## 4. Vocabulary (symbols) for ~120 primitives

- Inventory already has 256 single-Qwen-token glyphs (`channel.py`) — enough.
- Assign symbols to new primitives by the same seeding rule; keep the alien
  display remap (codepoint math already covers the block).
- Multi-seed symbol assignments (3 seeds) so "results don't depend on a lucky
  mapping" is answerable.

## 5. Pipeline + budget

```
extend taskgen (typed families)  → CPU, local, ~2-3 sessions of work
LLM paraphrase gen + round-trip  → ~$5-20 API, hours
verify (smoke gate, all tasks)   → CPU minutes
train (LoRA r=32-64, 3B, epochs) → Kaggle T4: ~10-20h (within free 30h/wk)
eval gates                       → free-English ≥90%, held-out decode ≥90%,
                                   zero-shot symbols reported, multi-seed
```

Order of execution: typed taskgen → generate+verify → paraphrases → train →
eval. Steps 1–3 are model-free and fully autonomous; training reuses the
existing Kaggle loop.
