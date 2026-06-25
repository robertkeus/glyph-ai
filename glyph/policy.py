"""Two-adapter LoRA policy for the forge loop (PLAN §B).

One base model, two LoRA adapters on shared weights:
  - `speaker`: generation hard-masked to symbol tokens (the channel glyphs are
    added as single tokens; PLAN §9 — zero existing-vocab leakage).
  - `builder`: reads the glyph message, writes code.

Implements the forge.py policy protocol — sample / build / learn (speaker GRPO) —
plus warmup_builder: SFT the builder to decode message→reference code so the
frozen-Builder reward is reachable (PLAN §B: co-adapt before freeze).

RUNNABLE SCAFFOLD. Smoke-tested end-to-end on CPU (scripts/smoke_policy.py): the
training code executes (forward, mask, backward, optimizer step). Convergence and
the §A cold-start / §D λ tuning are the open research — needs a GPU to train.
"""
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor

from glyph.agents import builder_prompt, speaker_prompt
from glyph.channel import Native


class _SymbolMask(LogitsProcessor):
    """Restrict generation to the allowed token-IDs (symbols + eos)."""
    def __init__(self, allowed_ids, vocab_size, device):
        self.mask = torch.full((vocab_size,), float("-inf"), device=device)
        self.mask[torch.tensor(sorted(allowed_ids), device=device)] = 0.0

    def __call__(self, input_ids, scores):
        return scores + self.mask


def _adapter_params(model, name):
    return [p for n, p in model.named_parameters() if name in n and "lora" in n]


class LoraPolicy:
    def __init__(self, model_name, channel=None, device=None, lr=1e-4,
                 max_msg=32, min_msg=2, max_code=256, lora_r=8):
        self.channel = channel or Native()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_msg, self.min_msg, self.max_code = max_msg, min_msg, max_code

        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.tok.add_tokens(self.channel.glyphs)            # glyphs → single tokens
        self.sym_ids = self.tok.convert_tokens_to_ids(self.channel.glyphs)
        self.sym_set = set(self.sym_ids)
        self.eos = self.tok.eos_token_id

        base = AutoModelForCausalLM.from_pretrained(model_name)
        base.resize_token_embeddings(len(self.tok))
        base.to(self.device)
        cfg = LoraConfig(r=lora_r, lora_alpha=2 * lora_r, lora_dropout=0.0,
                         target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM")
        self.model = get_peft_model(base, cfg, adapter_name="speaker")
        self.model.add_adapter("builder", cfg)
        for n, p in self.model.named_parameters():  # add_adapter leaves builder frozen
            if "lora" in n:
                p.requires_grad_(True)
        self.mask = _SymbolMask(self.sym_set | {self.eos}, len(self.tok), self.device)
        self.opt_speaker = torch.optim.AdamW(_adapter_params(self.model, "speaker"), lr=lr)
        self.opt_builder = torch.optim.AdamW(_adapter_params(self.model, "builder"), lr=lr)

    # --- forge.py protocol -------------------------------------------------
    @torch.no_grad()
    def sample(self, prompt, n):
        self.model.set_adapter("speaker")
        self.model.eval()
        enc = self.tok(prompt, return_tensors="pt").to(self.device)
        out = self.model.generate(
            **enc, max_new_tokens=self.max_msg, min_new_tokens=self.min_msg,
            do_sample=True, temperature=1.0, top_k=0, num_return_sequences=n,
            logits_processor=[self.mask], pad_token_id=self.eos)
        plen = enc.input_ids.shape[1]
        return [self.tok.decode([t for t in row[plen:].tolist() if t in self.sym_set])
                for row in out]

    @torch.no_grad()
    def build(self, prompt):
        self.model.set_adapter("builder")
        self.model.eval()
        enc = self.tok(prompt, return_tensors="pt").to(self.device)
        out = self.model.generate(**enc, max_new_tokens=self.max_code,
                                  do_sample=False, pad_token_id=self.eos)
        return self.tok.decode(out[0][enc.input_ids.shape[1]:].cpu(),
                               skip_special_tokens=True)

    def learn(self, prompt, messages, advantages):
        """One GRPO update on the speaker: loss = −Σ advantage·logπ(message)."""
        self.model.set_adapter("speaker")
        self.model.train()
        p_ids = self.tok(prompt, return_tensors="pt").input_ids.to(self.device)
        self.opt_speaker.zero_grad()
        used = 0
        for msg, adv in zip(messages, advantages):
            m_ids = self.tok(msg, add_special_tokens=False,
                             return_tensors="pt").input_ids.to(self.device)
            if m_ids.shape[1] == 0:
                continue
            seq = torch.cat([p_ids, m_ids], dim=1)
            logp = torch.log_softmax(self.model(seq).logits[:, :-1], dim=-1)
            tok_lp = logp.gather(-1, seq[:, 1:].unsqueeze(-1)).squeeze(-1)
            msg_lp = tok_lp[:, p_ids.shape[1] - 1:].sum()
            (-adv * msg_lp).backward()
            used += 1
        if used:
            self.opt_speaker.step()
        self.model.eval()
        return used

    # --- PLAN §B: co-adapt the builder before freezing --------------------
    def warmup_builder(self, tasks, rounds=1):
        """SFT the builder to map a sampled message → reference code, so it can
        interpret the speaker's (initially random) symbols. Run before forging."""
        for _ in range(rounds):
            for t in tasks:
                msg = self.sample(speaker_prompt(t, self.channel), 1)[0]
                bp = builder_prompt(self.channel.builder_text(msg), t["entry_point"])
                target = f"```python\n{t['solution']}\n```{self.tok.eos_token}"
                self._sft_builder(bp, target)

    def _sft_builder(self, prompt, target):
        self.model.set_adapter("builder")
        self.model.train()
        p = self.tok(prompt, return_tensors="pt").input_ids.to(self.device)
        full = self.tok(prompt + target, return_tensors="pt").input_ids.to(self.device)
        labels = full.clone()
        labels[:, :p.shape[1]] = -100  # supervise only the code continuation
        self.opt_builder.zero_grad()
        loss = self.model(full, labels=labels).loss
        loss.backward()
        self.opt_builder.step()
        self.model.eval()
        return loss.item()

    # --- checkpoint (free tiers wipe disk + cap 12h) ----------------------
    def save(self, path):
        self.model.save_pretrained(path)

    def load(self, path):
        self.model.load_adapter(path, adapter_name="speaker")
        self.model.load_adapter(path, adapter_name="builder")
