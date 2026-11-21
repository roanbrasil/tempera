"""`tempera-worker finetune`: LoRA / QLoRA fine-tuning of a small chat LLM on agent traces (#47).

Defaults follow docs/research/agents-and-m5-literature.md: LoRA on every linear layer, r=16, alpha=32
(Dettmers et al. 2023, QLoRA; Biderman et al. 2024); 4-bit NF4 with double quantization when a CUDA
GPU and bitsandbytes are present; loss on assistant tokens only.

The run always measures the untouched base model first (perplexity and tool-call accuracy on the same
held-out conversations) so the adapter is judged against a baseline, never alone.

Manifest (JSON or YAML), see docs/wiki/Contracts.md section 9:

    {"base_model": "Qwen/Qwen2.5-0.5B-Instruct", "train": "train.jsonl", "valid": "valid.jsonl",
     "test": "test.jsonl", "work_dir": "...", "model_dir": "...", "seed": 42,
     "params": {"r": 16, "alpha": 32, "epochs": 1, ...}}
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tempera_worker.context import RunContext
from tempera_worker.events import EventSink
from tempera_worker.llm.data import (
    IGNORE_INDEX,
    Chat,
    DataError,
    Example,
    read_chats,
    render,
    score_tool_calls,
    tokenize,
    tool_call_cases,
)
from tempera_worker.manifest import ManifestError

EXIT_OK, EXIT_STOPPED = 0, 3

DEFAULTS: dict[str, Any] = {
    "r": 16,
    "alpha": 32,
    "dropout": 0.05,
    "target_modules": "all-linear",
    "learning_rate": 2e-4,
    "weight_decay": 0.0,
    "warmup_ratio": 0.03,
    "epochs": 1.0,
    "max_steps": 0,  # > 0 caps the run (smoke tests, budgets)
    "batch_size": 1,
    "grad_accum": 8,
    "max_len": 2048,
    "quantization": "auto",  # auto | nf4 | none
    "gradient_checkpointing": True,
    "eval_every": 50,  # optimizer steps between validation passes; 0 = only at the end
    "eval_samples": 64,  # tool-call cases generated per model (base and tuned)
    "max_new_tokens": 160,
}

QUANTIZATIONS = ("auto", "nf4", "none")


@dataclass
class FinetuneManifest:
    base_model: str
    train: Path
    work_dir: Path
    model_dir: Path
    valid: Path | None = None
    test: Path | None = None
    seed: int = 0
    resume_from: Path | None = None
    params: dict[str, Any] = field(default_factory=dict)


def load_manifest(path: Path) -> FinetuneManifest:
    try:
        text = path.read_text()
    except OSError as e:
        raise ManifestError(f"cannot read manifest {path}: {e}") from None
    try:
        raw = json.loads(text)
    except json.JSONDecodeError:
        import yaml

        raw = yaml.safe_load(text)
    if not isinstance(raw, dict):
        raise ManifestError("manifest must be an object")
    for key in ("base_model", "train", "work_dir", "model_dir"):
        if not raw.get(key):
            raise ManifestError(f"manifest is missing `{key}`")
    params = {**DEFAULTS, **(raw.get("params") or {})}
    unknown = sorted(set(params) - set(DEFAULTS))
    if unknown:
        raise ManifestError(f"unknown params: {', '.join(unknown)} (known: {', '.join(sorted(DEFAULTS))})")
    if params["quantization"] not in QUANTIZATIONS:
        raise ManifestError(f"`quantization` must be one of {', '.join(QUANTIZATIONS)}")
    for key in ("r", "batch_size", "grad_accum", "max_len"):
        if not isinstance(params[key], int) or params[key] < 1:
            raise ManifestError(f"`{key}` must be a positive integer")
    if not (0 < float(params["learning_rate"]) < 1):
        raise ManifestError("`learning_rate` must be in (0, 1)")

    def opt(key: str) -> Path | None:
        return Path(raw[key]) if raw.get(key) else None

    m = FinetuneManifest(
        base_model=str(raw["base_model"]),
        train=Path(raw["train"]),
        work_dir=Path(raw["work_dir"]),
        model_dir=Path(raw["model_dir"]),
        valid=opt("valid"),
        test=opt("test"),
        seed=int(raw.get("seed", 0)),
        resume_from=opt("resume_from"),
        params=params,
    )
    for p in (m.train, m.valid, m.test):
        if p is not None and not p.is_file():
            raise ManifestError(f"data file not found: {p}")
    return m


# ---------------------------------------------------------------- model loading


def _device_and_quant(requested: str) -> tuple[str, bool, str]:
    """`(device, use_4bit, why)` for the requested quantization."""
    import torch

    cuda = torch.cuda.is_available()
    try:
        import bitsandbytes  # noqa: F401

        bnb = True
    except Exception:  # missing, or installed without a usable CUDA runtime
        bnb = False
    if requested == "nf4":
        if not (cuda and bnb):
            raise ManifestError("quantization `nf4` needs a CUDA GPU and bitsandbytes (the `llm` extra)")
        return "cuda", True, "4-bit NF4 (QLoRA)"
    if requested == "auto" and cuda and bnb:
        return "cuda", True, "4-bit NF4 (QLoRA)"
    if cuda:
        return "cuda", False, "16-bit LoRA" + ("" if requested == "none" else " (bitsandbytes unavailable)")
    return "cpu", False, "32-bit LoRA on CPU"


def base_model_fingerprint(name_or_path: str) -> dict[str, str]:
    """What identifies the base weights: the hub revision when cached, else a hash of config files."""
    fp: dict[str, str] = {"name": name_or_path}
    p = Path(name_or_path)
    if p.is_dir():
        h = hashlib.sha256()
        for f in sorted(p.glob("*.json")):
            h.update(f.name.encode())
            h.update(f.read_bytes())
        fp["config_sha256"] = h.hexdigest()
        return fp
    try:
        from huggingface_hub import try_to_load_from_cache

        cfg = try_to_load_from_cache(name_or_path, "config.json")
        if isinstance(cfg, str):
            fp["revision"] = Path(cfg).parent.name  # snapshots/<commit>/config.json
    except Exception:
        pass
    return fp


def load_model(m: FinetuneManifest, ctx: RunContext) -> tuple[Any, Any, str, str]:
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer

    p = m.params
    device, four_bit, mode = _device_and_quant(p["quantization"])
    tok = AutoTokenizer.from_pretrained(m.base_model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    if not getattr(tok, "chat_template", None):
        raise ManifestError(f"{m.base_model} has no chat template; use an instruct/chat model")

    bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if bf16 else (torch.float16 if device == "cuda" else torch.float32)
    kwargs: dict[str, Any] = {"dtype": dtype}
    if four_bit:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )
        kwargs["device_map"] = {"": 0}
    model = AutoModelForCausalLM.from_pretrained(m.base_model, **kwargs)
    if not four_bit:
        model.to(device)  # pyright: ignore[reportArgumentType]
    if four_bit:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=bool(p["gradient_checkpointing"])
        )
    elif p["gradient_checkpointing"] and device == "cuda":
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    if m.resume_from:
        model = PeftModel.from_pretrained(model, str(m.resume_from), is_trainable=True)
        ctx.log(f"resumed adapter from {m.resume_from}")
    else:
        cfg = LoraConfig(
            r=p["r"],
            lora_alpha=p["alpha"],
            lora_dropout=p["dropout"],
            target_modules=p["target_modules"],
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, cfg)
    if not four_bit and device == "cuda":
        # LoRA weights train in fp32 even when the frozen base is 16-bit.
        for n, prm in model.named_parameters():
            if prm.requires_grad and "lora_" in n:
                prm.data = prm.data.float()
    return model, tok, device, mode


# ---------------------------------------------------------------- batching and evaluation


def _batches(examples: list[Example], size: int, pad: int, device: str) -> Any:
    import torch

    for i in range(0, len(examples), size):
        chunk = examples[i : i + size]
        n = max(len(e.input_ids) for e in chunk)
        ids = torch.full((len(chunk), n), pad, dtype=torch.long)
        labels = torch.full((len(chunk), n), IGNORE_INDEX, dtype=torch.long)
        mask = torch.zeros((len(chunk), n), dtype=torch.long)
        for k, e in enumerate(chunk):
            L = len(e.input_ids)
            ids[k, :L] = torch.tensor(e.input_ids)
            labels[k, :L] = torch.tensor(e.labels)
            mask[k, :L] = 1
        yield ids.to(device), labels.to(device), mask.to(device)


def _token_loss(model: Any, ids: Any, labels: Any, mask: Any) -> tuple[Any, int]:
    """Summed next-token loss over labelled tokens, and how many there were."""
    import torch.nn.functional as F

    logits = model(input_ids=ids, attention_mask=mask).logits[:, :-1].float()
    target = labels[:, 1:]
    loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)), target.reshape(-1), ignore_index=IGNORE_INDEX, reduction="sum"
    )
    return loss, int((target != IGNORE_INDEX).sum())


def evaluate_loss(model: Any, examples: list[Example], batch: int, pad: int, device: str) -> float:
    import torch

    if not examples:
        return float("nan")
    was_training = model.training
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for ids, labels, mask in _batches(examples, batch, pad, device):
            loss, n = _token_loss(model, ids, labels, mask)
            total += float(loss)
            count += n
    if was_training:
        model.train()
    return total / max(count, 1)


def generate_calls(
    model: Any, tok: Any, chats: list[Chat], limit: int, max_new: int, device: str
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    """Greedy generation at each held-out tool-calling turn; exact-match metrics and samples."""
    import torch

    cases = tool_call_cases(chats, limit)
    if not cases:
        return {}, []
    model.eval()
    outs: list[str] = []
    with torch.no_grad():
        for c in cases:
            prompt = render(tok, c.prefix, c.tools, gen_prompt=True)
            enc = tok(prompt, return_tensors="pt", add_special_tokens=False).to(device)
            # Gradient checkpointing turns the KV cache off for training; generation needs it back.
            gen = model.generate(
                **enc, max_new_tokens=max_new, do_sample=False, use_cache=True, pad_token_id=tok.pad_token_id
            )
            outs.append(tok.decode(gen[0, enc["input_ids"].shape[1] :], skip_special_tokens=False))
    model.train()
    metrics = score_tool_calls([c.expected for c in cases], outs)
    samples = [{"id": c.id, "expected": c.expected, "generated": o} for c, o in zip(cases, outs, strict=True)]
    return metrics, samples


def _ppl(loss: float) -> float:
    return math.exp(min(loss, 50.0)) if math.isfinite(loss) else float("nan")


# ---------------------------------------------------------------- run


def _tokenized(
    path: Path | None, tok: Any, max_len: int, events: EventSink, what: str
) -> tuple[list[Chat], list[Example]]:
    if path is None:
        return [], []
    chats, skipped = read_chats(path)
    for line, why in skipped[:5]:
        events.log(f"{what}: skipped line {line}: {why}", level="warn")
    if len(skipped) > 5:
        events.log(f"{what}: skipped {len(skipped)} lines in total", level="warn")
    examples: list[Example] = []
    too_long = 0
    for c in chats:
        try:
            e = tokenize(tok, c, max_len)
        except DataError as err:
            raise ManifestError(str(err)) from None
        if e is None:
            too_long += 1
        elif e.trained_tokens:
            examples.append(e)
    if too_long:
        events.log(
            f"{what}: dropped {too_long} conversations longer than max_len={max_len} tokens "
            "(truncating would cut tool calls)",
            level="warn",
        )
    return chats, examples


def run(m: FinetuneManifest, events: EventSink) -> int:
    import torch

    p = m.params
    random.seed(m.seed)
    torch.manual_seed(m.seed)
    m.work_dir.mkdir(parents=True, exist_ok=True)
    ctx = RunContext(
        events=events, work_dir=m.work_dir, params=dict(p), seed=m.seed, resume_from=m.resume_from
    )
    ctx.install_signal_handlers()

    model, tok, device, mode = load_model(m, ctx)
    trainable = sum(x.numel() for x in model.parameters() if x.requires_grad)
    total = sum(x.numel() for x in model.parameters())
    events.log(
        f"base {m.base_model} on {device}: {mode}; LoRA r={p['r']} alpha={p['alpha']} on "
        f"{p['target_modules']}: {trainable:,} trainable of {total:,} parameters"
    )

    _, train = _tokenized(m.train, tok, p["max_len"], events, "train")
    valid_chats, valid = _tokenized(m.valid, tok, p["max_len"], events, "valid")
    test_chats, test = _tokenized(m.test, tok, p["max_len"], events, "test")
    if not train:
        raise ManifestError("no trainable conversations in the train file")
    held_chats, held = (test_chats, test) if test else (valid_chats, valid)
    events.log(
        f"{len(train)} train · {len(valid)} valid · {len(test)} test conversations; "
        f"{sum(e.trained_tokens for e in train):,} assistant tokens to learn"
    )
    pad = tok.pad_token_id
    bs, accum = p["batch_size"], p["grad_accum"]

    # Baseline: the base model on the same held-out data, before any update.
    summary: dict[str, float] = {}
    base_metrics: dict[str, float] = {}
    base_samples: list[dict[str, Any]] = []
    if held:
        with model.disable_adapter():
            base_loss = evaluate_loss(model, held, bs, pad, device)
            base_metrics, base_samples = generate_calls(
                model, tok, held_chats, p["eval_samples"], p["max_new_tokens"], device
            )
        summary["base_perplexity"] = _ppl(base_loss)
        events.metric("eval/base_perplexity", summary["base_perplexity"])
        for k, v in base_metrics.items():
            summary[f"base_{k}"] = v
            events.metric(f"eval/base_{k}", v)

    steps_per_epoch = math.ceil(len(train) / (bs * accum))
    total_steps = math.ceil(steps_per_epoch * float(p["epochs"]))
    if p["max_steps"]:
        total_steps = min(total_steps, int(p["max_steps"]))
    warmup = max(1, int(total_steps * float(p["warmup_ratio"])))
    params_to_train = [x for x in model.parameters() if x.requires_grad]
    opt = torch.optim.AdamW(
        params_to_train, lr=float(p["learning_rate"]), weight_decay=float(p["weight_decay"])
    )
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lambda s: (
            (s + 1) / warmup
            if s < warmup
            else 0.5 * (1 + math.cos(math.pi * (s - warmup) / max(1, total_steps - warmup)))
        ),
    )
    autocast = (
        torch.autocast("cuda", dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
        if device == "cuda"
        else nullcontext()
    )

    model.train()
    step, started = 0, time.monotonic()
    order = list(range(len(train)))
    stopped = False
    while step < total_steps and not stopped:
        random.shuffle(order)
        micro = [train[i] for i in order]
        batches = list(_batches(micro, bs, pad, device))
        for start in range(0, len(batches), accum):
            group = batches[start : start + accum]
            n_tokens = sum(int((lab[:, 1:] != IGNORE_INDEX).sum()) for _, lab, _ in group)
            loss_sum = 0.0
            for ids, labels, mask in group:
                with autocast:
                    loss, _ = _token_loss(model, ids, labels, mask)
                (loss / max(n_tokens, 1)).backward()
                loss_sum += float(loss.detach())
            torch.nn.utils.clip_grad_norm_(params_to_train, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            events.metric("train/loss", loss_sum / max(n_tokens, 1), step=step)
            events.metric("train/lr", float(sched.get_last_lr()[0]), step=step)
            ctx.progress(step, total_steps)
            if valid and p["eval_every"] and step % int(p["eval_every"]) == 0 and step < total_steps:
                vl = evaluate_loss(model, valid, bs, pad, device)
                events.metric("valid/loss", vl, step=step)
                events.metric("valid/perplexity", _ppl(vl), step=step)
            if ctx.checkpoint_due():
                ck = m.work_dir / "checkpoint"
                model.save_pretrained(str(ck))
                ctx.saved_checkpoint(ck, step)
            if ctx.should_stop():
                stopped = True
                break
            if step >= total_steps:
                break
    if stopped:
        events.error("stopped before completion; adapter checkpoint saved", retryable=True)
        return EXIT_STOPPED
    events.log(f"trained {step} optimizer steps in {time.monotonic() - started:.0f}s")

    if valid:
        vl = evaluate_loss(model, valid, bs, pad, device)
        events.metric("valid/loss", vl, step=step)
        events.metric("valid/perplexity", _ppl(vl), step=step)
    samples: list[dict[str, Any]] = []
    if held:
        tuned_loss = evaluate_loss(model, held, bs, pad, device)
        summary["perplexity"] = _ppl(tuned_loss)
        events.metric("eval/perplexity", summary["perplexity"])
        tuned_metrics, samples = generate_calls(
            model, tok, held_chats, p["eval_samples"], p["max_new_tokens"], device
        )
        for k, v in tuned_metrics.items():
            summary[k] = v
            events.metric(f"eval/{k}", v)
        for k, v in tuned_metrics.items():
            events.log(f"{k}: base {base_metrics.get(k, float('nan')):.3f} → tuned {v:.3f}")
        # Same cases in the same order: show what the base model said next to the adapter.
        for s, b in zip(samples, base_samples, strict=False):
            s["base_generated"] = b["generated"]

    save(m, model, tok, mode, step, summary, samples, events)
    events.done(summary)
    return EXIT_OK


def save(
    m: FinetuneManifest,
    model: Any,
    tok: Any,
    mode: str,
    steps: int,
    summary: dict[str, float],
    samples: list[dict[str, Any]],
    events: EventSink,
) -> None:
    adapter = m.model_dir / "adapter"
    m.model_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(adapter), safe_serialization=True)
    tok.save_pretrained(str(adapter))
    descriptor = {
        "algorithm": "llm_lora",
        "task": "llm_fine_tune",
        "base_model": base_model_fingerprint(m.base_model),
        "mode": mode,
        "steps": steps,
        "params": m.params,
        "seed": m.seed,
        "metrics": {k: v for k, v in summary.items() if math.isfinite(v)},
        "adapter": "adapter",
    }
    (m.model_dir / "model.json").write_text(json.dumps(descriptor, indent=2))
    if samples:
        with (m.model_dir / "samples.jsonl").open("w") as f:
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        events.artifact("report", str(m.model_dir / "samples.jsonl"))
    events.artifact("model", str(adapter), "safetensors")
