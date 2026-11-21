"""LLM fine-tuning (#47): data conversion, loss masks, tool-call scoring, and an end-to-end CLI run.

The end-to-end tests build a tiny random Qwen2 model and a byte-level BPE tokenizer on the fly, so they
run offline on CPU in seconds. A real QLoRA run on a cached instruct model is marked `gpu`.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest
from helpers import run_manifest

from tempera_worker.llm.data import (
    IGNORE_INDEX,
    Chat,
    DataError,
    parse_chat,
    parse_generated_call,
    read_chats,
    score_tool_calls,
    tool_call_cases,
)

TOOLS_OPENAI = [
    {
        "type": "function",
        "function": {
            "name": "get_order_status",
            "description": "Status of an order.",
            "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}}},
        },
    }
]


def openai_chat(i: int) -> dict[str, Any]:
    oid = f"A{i:03d}"
    return {
        "id": f"c{i}",
        "messages": [
            {"role": "system", "content": "You are a support agent."},
            {"role": "user", "content": f"where is order {oid}?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_order_status", "arguments": json.dumps({"order_id": oid})},
                    }
                ],
            },
            {"role": "tool", "content": '{"status": "shipped"}'},
            {"role": "assistant", "content": f"Order {oid} has shipped."},
        ],
        "tools": TOOLS_OPENAI,
    }


def normalized_chat(i: int) -> dict[str, Any]:
    """The shape `tempera traces finetune` writes (Rust `Conversation` serialized)."""
    oid = f"B{i:03d}"
    return {
        "id": f"n{i}",
        "messages": [
            {"role": "user", "content": f"status of {oid}"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "name": "get_order_status",
                        "arguments": {"order_id": oid},
                        "raw": f'{{"order_id":"{oid}"}}',
                    }
                ],
            },
        ],
        "tools": [{"name": "get_order_status", "parameters": {"type": "object"}}],
    }


# ---------------------------------------------------------------- unit: parsing (stdlib only)


def test_openai_arguments_become_a_dict_for_chat_templates() -> None:
    c = parse_chat(openai_chat(1), "x")
    call = c.messages[2]["tool_calls"][0]
    assert call == {
        "type": "function",
        "function": {"name": "get_order_status", "arguments": {"order_id": "A001"}},
    }
    assert c.messages[2]["content"] == ""
    assert c.tools[0]["function"]["name"] == "get_order_status"
    assert c.tools[0]["function"]["description"] == "Status of an order."


def test_normalized_traces_are_accepted() -> None:
    c = parse_chat(normalized_chat(2), "x")
    assert c.id == "n2"
    assert c.messages[1]["tool_calls"][0]["function"]["arguments"] == {"order_id": "B002"}
    assert c.tools == [
        {"type": "function", "function": {"name": "get_order_status", "parameters": {"type": "object"}}}
    ]


@pytest.mark.parametrize(
    ("obj", "reason"),
    [
        ({"messages": []}, "no `messages`"),
        ({"messages": [{"role": "robot", "content": "x"}]}, "unknown role"),
        ({"messages": [{"role": "user", "content": "hi"}]}, "no assistant turn"),
        (
            {"messages": [{"role": "user", "content": "x", "tool_calls": [{"name": "f"}]}]},
            "on a user message",
        ),
        (
            {"messages": [{"role": "assistant", "content": "", "tool_calls": [{"arguments": "{}"}]}]},
            "without a name",
        ),
    ],
)
def test_bad_conversations_are_rejected(obj: dict[str, Any], reason: str) -> None:
    with pytest.raises(DataError, match=reason):
        parse_chat(obj, "x")


def test_read_chats_skips_bad_lines_and_reports_them(tmp_path: Path) -> None:
    f = tmp_path / "t.jsonl"
    f.write_text(json.dumps(openai_chat(1)) + "\n\n{not json\n[1]\n" + json.dumps(normalized_chat(2)) + "\n")
    chats, skipped = read_chats(f)
    assert [c.id for c in chats] == ["c1", "n2"]
    assert [line for line, _ in skipped] == [3, 4]


@pytest.mark.parametrize(
    ("text", "tried", "call"),
    [
        (
            '<tool_call>\n{"name": "f", "arguments": {"a": 1}}\n</tool_call>',
            True,
            {"name": "f", "arguments": {"a": 1}},
        ),
        ('Sure. {"name": "f", "arguments": "{\\"a\\": 1}"}', True, {"name": "f", "arguments": {"a": 1}}),
        # Cut by max_new_tokens after the JSON: still a call.
        ('<tool_call>{"name": "f", "arguments": {"a": 1}}', True, {"name": "f", "arguments": {"a": 1}}),
        # Cut inside the JSON: tried, but broken.
        ('<tool_call>{"name": "f", "arguments": {"a": 1}', True, None),
        ("<tool_call>{name: f}</tool_call>", True, None),
        ('{"answer": 42}', True, None),
        ("Your order has shipped.", False, None),
    ],
)
def test_generated_tool_calls_are_parsed(text: str, tried: bool, call: dict[str, Any] | None) -> None:
    assert parse_generated_call(text) == (tried, call)


def test_tool_call_scores_are_exact_match() -> None:
    expected = [{"name": "f", "arguments": {"a": 1}}] * 4
    generated = [
        '<tool_call>{"name": "f", "arguments": {"a": 1}}</tool_call>',  # exact
        '<tool_call>{"name": "f", "arguments": {"a": 2}}</tool_call>',  # right tool, wrong args
        '<tool_call>{"name": "g", "arguments": {"a": 1}}</tool_call>',  # wrong tool
        "<tool_call>oops</tool_call>",  # broken
    ]
    assert score_tool_calls(expected, generated) == {
        "tool_call_rate": 1.0,
        "valid_call_rate": 0.75,
        "tool_name_accuracy": 0.5,
        "exact_match": 0.25,
    }
    assert score_tool_calls([], []) == {}


def test_tool_call_cases_stop_before_the_first_call() -> None:
    chats = [parse_chat(openai_chat(i), "x") for i in range(3)]
    chats.append(
        Chat("no-call", [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hey"}])
    )
    cases = tool_call_cases(chats, limit=2)
    assert [c.id for c in cases] == ["c0", "c1"]
    assert [m["role"] for m in cases[0].prefix] == ["system", "user"]
    assert cases[0].expected == {"name": "get_order_status", "arguments": {"order_id": "A000"}}


def test_finetune_manifest_validation(tmp_path: Path) -> None:
    from tempera_worker.llm.finetune import DEFAULTS, load_manifest
    from tempera_worker.manifest import ManifestError

    data = tmp_path / "train.jsonl"
    data.write_text(json.dumps(openai_chat(1)) + "\n")
    base = {
        "base_model": "m",
        "train": str(data),
        "work_dir": str(tmp_path / "w"),
        "model_dir": str(tmp_path / "o"),
    }

    def write(extra: dict[str, Any]) -> Path:
        p = tmp_path / "m.json"
        p.write_text(json.dumps({**base, **extra}))
        return p

    m = load_manifest(write({"params": {"r": 8}}))
    assert m.params["r"] == 8 and m.params["alpha"] == DEFAULTS["alpha"] == 32
    assert m.params["target_modules"] == "all-linear"
    for extra, msg in [
        ({"params": {"rank": 8}}, "unknown params: rank"),
        ({"params": {"quantization": "int3"}}, "quantization"),
        ({"params": {"r": 0}}, "`r` must be"),
        ({"params": {"learning_rate": 2}}, "learning_rate"),
        ({"valid": str(tmp_path / "missing.jsonl")}, "not found"),
        ({"base_model": ""}, "missing `base_model`"),
    ]:
        with pytest.raises(ManifestError, match=msg):
            load_manifest(write(extra))


# ---------------------------------------------------------------- tokenizer + model fixtures (llm extra)

CHAT_TEMPLATE = (
    "{%- if tools %}<|system|>tools:{{ tools | tojson }}\n{%- endif %}"
    "{%- for m in messages %}"
    "{%- if m.role == 'assistant' %}<|assistant|>{{ m.content }}"
    "{%- for tc in m.tool_calls or [] %}<tool_call>{{ tc.function | tojson }}</tool_call>{%- endfor %}"
    "<|end|>\n"
    "{%- else %}<|{{ m.role }}|>{{ m.content }}<|end|>\n{%- endif %}"
    "{%- endfor %}"
    "{%- if add_generation_prompt %}<|assistant|>{%- endif %}"
)


def tiny_tokenizer(corpus: list[str]) -> Any:
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast

    special = ["<|pad|>", "<|end|>", "<|system|>", "<|user|>", "<|assistant|>", "<|tool|>"]
    t = Tokenizer(models.BPE())
    t.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    t.decoder = decoders.ByteLevel()
    t.train_from_iterator(
        corpus,
        trainers.BpeTrainer(
            vocab_size=400, special_tokens=special, initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
        ),
    )
    tok = PreTrainedTokenizerFast(tokenizer_object=t, pad_token="<|pad|>", eos_token="<|end|>")
    tok.chat_template = CHAT_TEMPLATE
    return tok


@pytest.fixture(scope="module")
def tiny_model(tmp_path_factory: pytest.TempPathFactory) -> Path:
    pytest.importorskip("peft")
    pytest.importorskip("torch")
    from transformers import Qwen2Config, Qwen2ForCausalLM

    chats = [openai_chat(i) for i in range(20)]
    corpus = [json.dumps(c) for c in chats] + ["<tool_call></tool_call> tools: where is order status shipped"]
    tok = tiny_tokenizer(corpus)
    cfg = Qwen2Config(
        vocab_size=len(tok),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=1024,
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id,
        tie_word_embeddings=True,
    )
    import torch

    torch.manual_seed(0)
    path = tmp_path_factory.mktemp("tiny-qwen")
    Qwen2ForCausalLM(cfg).save_pretrained(path)
    tok.save_pretrained(path)
    return path


@pytest.mark.llm
def test_loss_mask_covers_assistant_turns_only(tiny_model: Path) -> None:
    from transformers import AutoTokenizer

    from tempera_worker.llm.data import tokenize

    tok = AutoTokenizer.from_pretrained(tiny_model)
    ex = tokenize(tok, parse_chat(openai_chat(7), "x"), max_len=4096)
    assert ex is not None
    learned = tok.decode([t for t in ex.labels if t != IGNORE_INDEX])
    ignored = tok.decode([i for i, lab in zip(ex.input_ids, ex.labels, strict=True) if lab == IGNORE_INDEX])
    # The tool call and the final answer are learned; prompts, tool output and tool specs are not.
    assert '"order_id": "A007"' in learned and "Order A007 has shipped." in learned
    assert "where is order A007" in ignored and 'shipped"}' in ignored and "tools:" in ignored
    assert "where is order" not in learned and "tools:" not in learned
    assert tokenize(tok, parse_chat(openai_chat(7), "x"), max_len=10) is None


@pytest.mark.llm
def test_template_without_prefix_property_is_refused(tiny_model: Path) -> None:
    from transformers import AutoTokenizer

    from tempera_worker.llm.data import tokenize

    tok = AutoTokenizer.from_pretrained(tiny_model)
    # A template that rewrites earlier turns once later ones exist (e.g. strips past reasoning).
    tok.chat_template = "{{ messages | length }}{% for m in messages %}{{ m.content }}{% endfor %}"
    with pytest.raises(DataError, match="growing prefixes"):
        tokenize(tok, parse_chat(openai_chat(1), "x"), max_len=4096)


def _write_split(dir: Path, name: str, chats: list[dict[str, Any]]) -> str:
    p = dir / f"{name}.jsonl"
    p.write_text("".join(json.dumps(c) + "\n" for c in chats))
    return str(p)


@pytest.mark.llm
def test_finetune_cli_end_to_end_on_cpu(tiny_model: Path, tmp_path: Path) -> None:
    chats = [openai_chat(i) for i in range(30)] + [normalized_chat(i) for i in range(6)]
    manifest = {
        "base_model": str(tiny_model),
        "train": _write_split(tmp_path, "train", chats[:24]),
        "valid": _write_split(tmp_path, "valid", chats[24:30]),
        "test": _write_split(tmp_path, "test", chats[30:]),
        "work_dir": str(tmp_path / "work"),
        "model_dir": str(tmp_path / "model"),
        "seed": 7,
        "params": {
            "r": 4,
            "alpha": 8,
            "epochs": 3,
            "grad_accum": 2,
            "batch_size": 2,
            "eval_every": 5,
            "learning_rate": 5e-3,
            "eval_samples": 3,
            "max_new_tokens": 12,
            "quantization": "none",
        },
    }
    proc, events = run_manifest("finetune", manifest, tmp_path / "m.json")
    assert proc.returncode == 0, proc.stderr[-3000:]
    kinds = [e["t"] for e in events]
    assert kinds[-1] == "done"
    metrics: dict[str, list[float]] = {}
    for e in events:
        if e["t"] == "metric":
            metrics.setdefault(e["name"], []).append(e["value"])
    # Baseline first, then training, then the tuned model on the same held-out data.
    assert kinds.index("metric") == next(
        i for i, e in enumerate(events) if e.get("name") == "eval/base_perplexity"
    )
    loss = metrics["train/loss"]
    assert len(loss) == 18  # ceil(24 / (2*2)) * 3 epochs
    assert sum(loss[-3:]) < sum(loss[:3]), loss
    assert metrics["eval/perplexity"][0] < metrics["eval/base_perplexity"][0]
    assert {"eval/tool_call_rate", "eval/base_exact_match", "valid/perplexity"} <= set(metrics)
    done = events[-1]["summary"]
    assert done["perplexity"] < done["base_perplexity"]

    model_dir = tmp_path / "model"
    assert (model_dir / "adapter" / "adapter_model.safetensors").is_file()
    assert (model_dir / "adapter" / "adapter_config.json").is_file()
    desc = json.loads((model_dir / "model.json").read_text())
    assert desc["task"] == "llm_fine_tune" and desc["params"]["r"] == 4
    assert desc["base_model"]["config_sha256"]
    import torch

    # quantization: none → plain LoRA, 16-bit on a GPU, 32-bit on CPU; never 4-bit.
    assert desc["mode"] == ("16-bit LoRA" if torch.cuda.is_available() else "32-bit LoRA on CPU")
    samples = [json.loads(x) for x in (model_dir / "samples.jsonl").read_text().splitlines()]
    assert len(samples) == 3
    assert {"id", "expected", "generated", "base_generated"} <= set(samples[0])
    arts = [(e["kind"], e.get("format")) for e in events if e["t"] == "artifact"]
    assert ("model", "safetensors") in arts

    # The adapter loads back on the base model and changes its output.
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    base = AutoModelForCausalLM.from_pretrained(tiny_model)
    tuned = PeftModel.from_pretrained(base, str(model_dir / "adapter"))
    assert sum(p.numel() for n, p in tuned.named_parameters() if "lora_" in n) > 0


@pytest.mark.llm
def test_finetune_cli_rejects_bad_manifest_and_template(tiny_model: Path, tmp_path: Path) -> None:
    train = _write_split(tmp_path, "train", [openai_chat(1)])
    base = {
        "base_model": str(tiny_model),
        "train": train,
        "work_dir": str(tmp_path / "w"),
        "model_dir": str(tmp_path / "o"),
        "params": {"quantization": "none"},
    }
    proc, events = run_manifest("finetune", {**base, "params": {"lora_rank": 3}}, tmp_path / "a.json")
    assert proc.returncode == 2 and "unknown params" in events[-1]["msg"]
    # Only user turns: nothing to learn.
    _write_split(tmp_path, "train", [{"messages": [{"role": "user", "content": "x"}]}])
    proc, events = run_manifest("finetune", base, tmp_path / "b.json")
    assert proc.returncode == 2 and "no trainable conversations" in events[-1]["msg"]


# ---------------------------------------------------------------- real QLoRA on a GPU (local only)

GPU_MODEL = os.environ.get("TEMPERA_TEST_LLM", "Qwen/Qwen2.5-0.5B-Instruct")


def _gpu_ready() -> bool:
    try:
        import bitsandbytes  # noqa: F401
        import torch
        from huggingface_hub import try_to_load_from_cache
    except Exception:
        return False
    return torch.cuda.is_available() and isinstance(try_to_load_from_cache(GPU_MODEL, "config.json"), str)


@pytest.mark.llm
@pytest.mark.gpu
@pytest.mark.skipif(not _gpu_ready(), reason=f"needs CUDA, bitsandbytes and {GPU_MODEL} in the HF cache")
def test_qlora_on_gpu_beats_the_base_model(tmp_path: Path) -> None:
    examples = Path(__file__).parents[3] / "examples" / "data" / "agent_traces_clean.jsonl"
    # OpenAI-format lines only: Hermes ones are normalized by `tempera traces finetune`, not the worker.
    lines = [x for x in examples.read_text().splitlines() if '"messages"' in x]
    manifest = {
        "base_model": GPU_MODEL,
        "train": _write_split(tmp_path, "train", [json.loads(x) for x in lines[:160]]),
        "test": _write_split(tmp_path, "test", [json.loads(x) for x in lines[-60:]]),
        "work_dir": str(tmp_path / "work"),
        "model_dir": str(tmp_path / "model"),
        "params": {"max_steps": 20, "grad_accum": 4, "batch_size": 2, "eval_samples": 12, "max_len": 1024},
    }
    proc, events = run_manifest("finetune", manifest, tmp_path / "m.json", timeout=1200)
    assert proc.returncode == 0, proc.stderr[-3000:]
    done = events[-1]["summary"]
    assert "QLoRA" in json.loads((tmp_path / "model" / "model.json").read_text())["mode"]
    assert done["perplexity"] < done["base_perplexity"]
    assert done["valid_call_rate"] >= done["base_valid_call_rate"]
    assert done["exact_match"] > done["base_exact_match"]
