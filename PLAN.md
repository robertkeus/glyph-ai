# PLAN.md — Glyph

**Working title:** Glyph — *a model that forges its own native language, then explains it.*

## One-line claim

An entity whose **native tongue is a from-scratch, logographic agent-language**: it ingests English, reasons natively in its own evolved symbols, emits English / native / code on demand, and can **explain its own language well enough that a cold model can decode it.** Autonomy (it invented the language) with accountability (it can teach it).

The headline result is not "an AI made a language" — that's cheap and unfalsifiable. It's **"the AI's own explanation of its self-forged language is accurate enough to be load-bearing,"** proven by a third party decoding held-out messages from that explanation alone.

**Long-term aim:** because the language forges *on coding tasks*, its primitives should map onto coding concepts — making it, later, an especially good *communication medium* for code-domain agent work (handoffs, plans, reviews). It stays a communication language, not a programming one. Scoped as a stretch phase below — a measurement on the forged language, not a new research bet.

---

## Why this is the right object (and the traps it avoids)

- **Forged, not designed.** Meaning comes from *use under pressure*, not introspection. The language precipitates out of agents solving tasks through a constrained channel; symbols acquire meaning because they reliably helped pass tasks. A language an LLM "designs" in a vacuum is a hollow conlang — pretty, ungrounded, and un-explainable in the load-bearing sense.
- **"Finished" = task-performance plateau, never self-report.** Self-termination optimizes for looking done, not being done. The environment grades; the agent never holds the gradebook.
- **Logographic, scored in bytes.** "Not an alphabet" only matters if units are *meaning-units* (one symbol ≈ one concept), not glyph cosmetics. Score in **bytes** so density is honest and the tokenizer can't be gamed.
- **Self-explanation is the falsification test, not a feature.** Most emergent-language work can't refute "it's just a cipher." Glyph can be interrogated, and the explanation is checked by a fresh decoder. That converts the field's central weakness into the headline.

---

## Architecture

One base model (small open coder: **Qwen2.5-Coder-1.5B or 3B**), three competencies, trained in phases so they don't fight:

| Competency | What | Phase |
|---|---|---|
| **Native fluency** | reason + act in the emergent logographic channel | 1 (forge) |
| **Bidirectional translation** | native ⇄ English, faithful | 2 (anchor) |
| **Metalinguistic** | describe / define / justify / judge the native language *in English* | 2 (anchor) |

**Roles in the forge loop:**
- **Speaker** — English task in → native message out.
- **Builder** — native message in → code out.
- Same base weights, separate LoRA adapters; co-trained. (Shared weights keep the "one entity, one native tongue" framing; separate adapters let the two ends specialize.)

**The channel:** a fixed inventory of **~2–4k initially-meaningless discrete symbol-IDs**. No seed meanings — from scratch. RL assigns meaning. IDs render as arbitrary glyphs for the demo; the glyph is a font, the ID is the language.

**Channel is hard-masked.** Speaker output is restricted to the symbol-ID inventory at decode time — zero existing-vocab leakage. This is a constraint, not a reward term: a pretrained coder will route meaning through English/code tokens if a soft penalty lets it, and then nothing alien forges. Mask, don't penalize.

---

## The three failure modes and the constraint that kills each

1. **Collapse to a trivial cipher** (a 12-symbol lookup that passes the train tasks, generalizes to nothing).
   → **Held-out compositional task split.** Scoring tasks recombine primitives seen only separately in training. A phrasebook can't generalize; a language can. *This is the single most important design element.* (But see Training mechanics §G — compositionality must be measured in the emergent *symbol* space, not assumed from task labels.)
2. **Win by getting dumber** (offload meaning into noise that correlates with train success).
   → **Frozen-Builder success on held-out tasks** gates the *evaluation*. Can't game an unseen-task partner without genuinely transmitting information. (Frozen-Builder is an eval/gate condition, NOT the early-training condition — see §B.)
3. **Confabulated explanation** (model writes a plausible grammar unrelated to actual usage).
   → **Fresh-translator test.** The explanation must let a *cold* model decode held-out native messages. Predictive, not pretty. (The cold model must never appear in training — see §F.)

---

## Training mechanics (the underspecified half — added v2)

The evaluation design was always strong; the training design is where Glyph fails quietly. These are method-level decisions, not tuning.

**§A — Name the algorithm; design the easy end around reward *variance*, not magnitude.**
Reward is sparse and terminal (Builder passes hidden tests). Under GRPO, when every sampled Speaker message fails, all group rewards are equal → advantage is exactly **zero** → no gradient. Cold-start is not "weak signal," it is *no* signal. The curriculum's precise job: at the easy end, guarantee that *some* sampled messages pass, so each group has reward variance. Tag easy tasks as "easy enough that near-random messages sometimes succeed." PPO with a learned value head tolerates sparsity slightly better at the cost of its own instability. **Decision deferred to Phase 0 measurement (GRPO vs PPO), but the variance requirement holds either way.**

**§B — Separate "frozen Builder as eval gate" from "frozen Builder in training."**
The eval gate (frozen Builder must succeed on held-out tasks) is correct — keep it. A Builder frozen *during early Speaker RL* cannot interpret symbols it never co-adapted to, so reward ≈ 0 for reasons unrelated to language quality (compounds §A). **Sequence: joint warmup where Speaker+Builder bootstrap a shared code together (both adapters update) → freeze Builder only for the held-out gate and final eval.**

**§C — Stabilize co-training; the channel is otherwise a moving target.**
If both adapters update freely, the Speaker optimizes against a shifting Builder (classic emergent-comms oscillation / protocol drift / mode collapse). Mechanism required: Builder on a slower LR with a **lagged target** (freeze-refresh every K steps), or a small self-play population. "Co-trained" alone underspecifies this.

**§D — Keep a λ floor or hard length cap from step one.**
λ=0 early lets the Speaker emit maximally long, English-shadowing messages → a verbose transliterator that never condenses into reused logographic units → no language to explain. Byte pressure is what *creates* the object. Keep a small fixed λ floor (or hard length cap) from the start — enough to force meaning to condense, not enough to collapse to a cipher. **Anneal toward higher λ after competence; never start at zero.**

**§E — Plateau stop is a Pareto criterion, not a ratio.**
"Bytes-per-solved-task stops improving" is a scalar over two moving quantities (success ↑, bytes ↓); it can flatten while success still climbs. **Stop on the 2-D Pareto front:** fix success rate and watch bytes (or vice versa). This is the principled definition of "till it's finished."

**§F — Do not train the explanation against the cold model. It contaminates the headline.**
Rewarding explanation accuracy via the fresh-translator test (a) is prohibitively expensive per RL step, and (b) makes the cold model no longer cold — the falsifier is leaked into the optimizer and the headline result is dead to any reviewer. **Train explanation against a rotated *proxy* decoder pool; reserve exactly one truly-never-touched model + held-out messages for the single final reported test.** This is a correctness bug in the method if violated.

**§G — Measure compositionality in symbol-space, not task-space.**
You control recombination of *hand-labeled* primitives, but the model segments meaning its own way — its latent symbols won't map 1:1 to your labels. A held-out *task* recombination does not guarantee a novel *language*-level composition. **Test 2 must check the emergent inventory directly:** do unseen symbol-combinations appear and succeed on held-out tasks? Otherwise test 2 can pass while the language is still a lookup table indexed differently than assumed.

---

## Data plan

There is **no seed language corpus** — that's what "from scratch" means. You assemble the *environment*; the loop manufactures the language, translations, and explanations.

**Data need 1 — the task bank (the real seed data):**
- Source: MBPP, HumanEval-style, CodeContests, Stack-derived single-function problems (permissive licenses). Keep artifacts **single-function** — the *language* is the variable, not build difficulty.
- **Synthetic generation (where coding agents legitimately help):** point Claude Code / Codex at "generate N single-function tasks parameterized over primitives {P}, recombined in held-out ways, each with executable hidden tests."
- **Verification = run the tests.** Reward signal is code-pass-rate, not a judge model. Clean, ungameable RL signal.

**Data need 2 — difficulty curriculum (decides whether emergence happens in week 2 or never):**
- Pure from-scratch RL on hard tasks → near-zero early reward → no signal → no emergence. The fix is in the data: an explicit **easy→hard gradient** so some symbol-meaning gets reinforced early, then ramp. Build the task bank as a curriculum, not a flat pile.
- **Curriculum's hard requirement (per §A):** the easiest stage must produce *reward variance* under near-random messages, not merely "low difficulty." Variance is the thing that makes the gradient exist.

**Data need 3 — held-out compositional split:**
- Carved out *before* training touches anything. Tasks recombining train primitives in absent combinations. Agent proposes recombinations; you verify genuine novelty.
- Pair with the symbol-space compositionality probe (§G) at eval time.

**Data need 4 — translation + explanation pairs:**
- *Generated after Phase 1* by the trained model itself (its native messages paired with the English task context you already have). Does not exist up front; do not try to gather it now.

> **Can the Gemma-challenge agents help?** No. They're inference-serving optimizers on a fixed model/leaderboard you don't own — orthogonal to this. But the *pattern* (agents generate data by playing the task) is exactly what you copy, with **your** agents, for synthetic task generation.

---

## Training phases

**Phase 0 — Harness (the explicit gate; decides from-scratch vs seeded).**
Build before any model:
1. Task-bank loader + **run-the-tests verifier** (sandboxed exec → pass-rate).
2. **Curriculum tagger** (primitives per task) + held-out compositional split generator with a *novelty check*.
3. **Byte-counted channel** (~2–4k symbol-ID inventory, hard-masked, Speaker→Builder only).
4. **English-baseline smoke test:** can two agents pass held-out tasks through *English* in this harness at all? If not, the harness is broken, not the language — find out before training.
5. **Cold-start probe:** run the actual RL loop on the easiest curriculum stage; measure whether early reward variance is nonzero (§A).

**Phase 0 is the decision point for from-scratch vs seeded vocabulary.** If the cold-start probe shows persistent zero-variance after the easy-end is steepened to its limit, commit to a lightly-seeded primitive vocabulary; otherwise run pure from-scratch. Do not pre-commit — measure.

**Phase 1 — Forge.**
- RL (algorithm fixed in Phase 0). Reward = `held_out_task_success − λ·bytes_on_channel`, with λ at its **floor from step one** (§D).
- **Joint warmup** (Speaker+Builder co-adapt) → **freeze Builder** for the held-out gate (§B).
- **Stabilize co-training** via lagged Builder target / population (§C).
- **Anneal λ upward** *after* competence emerges — compression pressure rises, never starts at zero.
- **Stop on the Pareto front** (§E): when held-out bytes stop improving at fixed success, the language has stabilized.

**Phase 2 — Anchor (translation + metalinguistic).**
- Freeze the forged language. Train translation (native ⇄ English) and metalinguistic competence (describe/define/justify/judge).
- Explanation training rewards **verifiable accuracy against a rotated proxy decoder pool** (§F) — never against the reserved cold model.
- Post-hoc by design: a genuine *description of an independently-evolved language* is a stronger claim than co-evolving language and explanation.

---

## Evaluation — the three tests that ARE the paper

1. **Capability under compression.** Two agents solve **held-out** tasks through the native channel; beat single-agent English baseline on **bytes-per-solved-task at equal success rate.** Report *your own* bytes; do not borrow OPTIMA's ~90% *token*-reduction figure as a target — it's a different setup and not apples-to-apples. Either reproduce their bytes or drop the number. Proves the channel carries real task information.
2. **Compositional generalization.** Solve tasks requiring recombination of separately-seen primitives — **and verify the recombination occurs in the emergent symbol inventory** (§G), not just at the task-label level. Separates *language* from *lookup table*. The test reviewers will demand.
3. **Load-bearing self-explanation (headline).** Hand the model's *own* English explanation of its language to a **cold model** (held-out, never saw the language *and never appeared in explanation training* — §F). Cold model must decode held-out native messages using only that explanation.
   - High decode accuracy → the self-description is true → real, structured, **self-documented** language.
   - Plus a **decodability probe:** train a decoder to recover English from native messages. High recovery → "learned compression of English" (still real). Low recovery → "genuinely divergent representation" (the flashier claim). Either is publishable; **not running it is the only failure.**

**Headline claim is pre-committed:** "denser self-documented protocol, fully measured." Likely outcome for a 3B on code tasks is high English-recovery (substantially learned compression) — which would also make the explanation test work, inviting "of course, it's just English." Lead with the defensible claim; let divergence *upgrade* the result only if the probe surprises you. (Resolves former Open Decision #1.)

---

## Demo (keynote artifact)

Interactions-view (clone the Gemma-challenge Space pattern — static page reading `events.json`, grouped/categorized, message-linked):
- **Left:** English task in.
- **Middle:** the two agents exchanging alien logographic messages — visibly not English.
- **Right:** working code falls out.
- **Live kill-move:** ask "explain that rule," feed the explanation to a cold model on stage, watch it correctly decode a fresh native message.

*An AI teaches its private language to another AI well enough to be understood.* The autonomy-with-accountability thesis in 90 seconds.

---

## Phase 3 (stretch) — the native language as an *ideal medium for coding work*

Deliberately **after** the core result. The language stays a **communication** medium — it is *not* made executable. The claim is narrower and far safer: because it forged on coding tasks, its primitives segment the coding-concept space, so it's an especially good wire format for **code-related agent-to-agent work** (handoffs, plans, diffs-as-intent, reviews) — not a programming language itself.

This keeps Phase 3 a **measurement on the language you already have**, not a separate research bet. Nothing about the channel changes; you're testing whether it's *good at the coding domain* specifically.

**What to measure (no new training beyond Phases 1–2):**
- **Domain density.** Bytes-per-coding-task through the native channel vs (a) English baseline, (b) the same channel on a *non-coding* task set. If native beats English by a wider margin on coding than on neutral tasks, the language is *coding-specialized*. That delta is the whole Phase 3 result.
- **Primitive→concept alignment.** Probe whether stable native symbols correspond to recurring coding concepts (control flow, types, I/O, error paths). Evidence the segmentation is conceptual, not arbitrary. (Reuses the symbol-space machinery from §G.)
- **Handoff fidelity.** Speaker sends a native coding-intent message; frozen Builder produces code; measure success vs the same handoff in English at equal bytes. Tests the language as a *coordination* medium for code.

**Don't start until** Phase 2 passes the load-bearing self-explanation test — an unexplainable language isn't worth specializing claims about. **No new objective conflict:** the language stays communicative (not executable), so there's no run-must-pass pressure fighting compression.

---

## Risks & kill criteria

| Risk | Signal | Kill / mitigate |
|---|---|---|
| **Cold-start zero-variance** (the main risk) | reward variance ≈ 0 on easiest curriculum stage in Phase 0 probe (§A) | steepen easy end until near-random messages sometimes pass; if still zero after the Phase 0 compute budget, commit to seeded primitive vocabulary |
| Frozen-Builder unreachable reward | reward stuck at ~0 despite competence elsewhere | verify the warmup→freeze sequence (§B); Builder must co-adapt before it is frozen |
| Co-training instability | success oscillates / protocol drifts across steps | slower Builder LR + lagged target, or population self-play (§C) |
| No divergence (verbose transliterator) | messages stay long and English-shaped; bytes never fall | λ floor / length cap from step one (§D); raise λ sooner |
| Cipher collapse | passes train, fails held-out compositional in *symbol* space | raise compositional fraction; delay λ anneal; check §G probe |
| Explanation-test contamination | cold model appeared (even indirectly) in explanation training | enforce rotated proxy pool; quarantine the reserved cold model (§F) |
| Confabulated explanation | cold-model decode ≈ chance | explanation is theater; report honestly — a *negative* result here is a real finding about LLM metalinguistic limits |
| Steganographic English | decodability probe ≈ full recovery | acceptable; reframes claim to "learned compression" — headline already pre-committed to this |
| Result fragility | seeds diverge wildly | fix seeds, report variance, never single-run |

**Non-negotiables (name on the line):** score in **bytes** not tokens · train explanation against a **proxy pool**, never the reserved cold model · keep **frozen-Builder as eval gate, co-adapted Builder in early training** · evaluation behind a **held-out compositional set the loop never touches**, checked in **symbol space** · success verified by **running code**, not a judge model.

---

## Open decisions before build

1. **RL algorithm (GRPO vs PPO):** decide in Phase 0 from the cold-start probe; both must satisfy the reward-variance requirement (§A).
2. **Compute budget for Phase 1** — from-scratch RL is the cost sink and sets the cold-start kill threshold. Put a concrete GPU-time/step-count number on it *before* Phase 1, or the kill rule can't fire. (×N seeds; never single-run.)
3. **Glyph set for the demo** — purely cosmetic, but it's what the audience sees. Pick something legibly alien.

*(Former Open Decision #1 — headline claim — is now resolved above: "denser self-documented protocol, fully measured.")*
