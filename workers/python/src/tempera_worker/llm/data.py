"""Chat data for LLM fine-tuning (#47): conversations in, token ids and loss masks out.

Accepted lines (one conversation per JSONL line):

* Têmpera's normalized traces, as written by `tempera traces finetune`: `messages[].tool_calls[]`
  carries `name` and `arguments` (an object) directly; `tools[]` carries `name` and `parameters`.
* OpenAI chat: `tool_calls[].function.{name, arguments}` with `arguments` as a JSON string, and
  `tools[].function.{name, description, parameters}`.

Everything is converted to the shape Hugging Face chat templates expect: tool-call arguments as a
dict (templates `tojson` them, so a string would be encoded twice) and tools as OpenAI function specs.

The loss is computed on assistant turns only. The mask comes from character offsets: the text of
the conversation up to (and including the generation prompt before) assistant turn *i* is a prefix
of the full text, so the assistant tokens are those that start between the two prefixes.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

IGNORE_INDEX = -100

Message = dict[str, Any]


@dataclass
class Chat:
    """One conversation, ready for `tokenizer.apply_chat_template`."""

    id: str
    messages: list[Message]
    tools: list[dict[str, Any]] = field(default_factory=list)


class DataError(ValueError):
    pass


def _arguments(raw: Any) -> Any:
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return raw
    return raw if raw is not None else {}


def _tool_call(tc: dict[str, Any]) -> dict[str, Any]:
    fn: dict[str, Any] = tc["function"] if isinstance(tc.get("function"), dict) else tc
    name = fn.get("name")
    if not isinstance(name, str) or not name:
        raise DataError("tool call without a name")
    return {"type": "function", "function": {"name": name, "arguments": _arguments(fn.get("arguments"))}}


def _tool_def(t: dict[str, Any]) -> dict[str, Any]:
    fn: dict[str, Any] = t["function"] if isinstance(t.get("function"), dict) else t
    name = fn.get("name")
    if not isinstance(name, str) or not name:
        raise DataError("tool definition without a name")
    out: dict[str, Any] = {"name": name, "parameters": fn.get("parameters") or {"type": "object"}}
    if fn.get("description"):
        out["description"] = fn["description"]
    return {"type": "function", "function": out}


def parse_chat(obj: dict[str, Any], fallback_id: str) -> Chat:
    raw = obj.get("messages")
    if not isinstance(raw, list) or not raw:
        raise DataError("no `messages` list")
    messages: list[Message] = []
    for m in raw:
        role = m.get("role")
        if role not in ("system", "user", "assistant", "tool"):
            raise DataError(f"unknown role {role!r}")
        msg: Message = {"role": role, "content": m.get("content") or ""}
        calls = m.get("tool_calls") or []
        if calls:
            if role != "assistant":
                raise DataError(f"`tool_calls` on a {role} message")
            msg["tool_calls"] = [_tool_call(tc) for tc in calls]
        if role == "tool" and m.get("name"):
            msg["name"] = m["name"]
        messages.append(msg)
    if not any(m["role"] == "assistant" for m in messages):
        raise DataError("no assistant turn to learn from")
    tools = [_tool_def(t) for t in obj.get("tools") or []]
    return Chat(id=str(obj.get("id") or fallback_id), messages=messages, tools=tools)


def read_chats(path: Path) -> tuple[list[Chat], list[tuple[int, str]]]:
    """Conversations of a JSONL file, plus `(line, reason)` for each line that was skipped."""
    chats: list[Chat] = []
    skipped: list[tuple[int, str]] = []
    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                if not isinstance(obj, dict):
                    raise DataError("line is not a JSON object")
                chats.append(parse_chat(obj, f"{path.stem}:{i}"))
            except (json.JSONDecodeError, DataError) as e:
                skipped.append((i, str(e)))
    return chats, skipped


def render(tokenizer: Any, messages: list[Message], tools: list[dict[str, Any]], gen_prompt: bool) -> str:
    kwargs: dict[str, Any] = {"tokenize": False, "add_generation_prompt": gen_prompt}
    if tools:
        kwargs["tools"] = tools
    return tokenizer.apply_chat_template(messages, **kwargs)


@dataclass
class Example:
    id: str
    input_ids: list[int]
    labels: list[int]

    @property
    def trained_tokens(self) -> int:
        return sum(1 for t in self.labels if t != IGNORE_INDEX)


def assistant_spans(tokenizer: Any, chat: Chat) -> tuple[str, list[tuple[int, int]]]:
    """Full text and the character spans of every assistant turn in it."""
    full = render(tokenizer, chat.messages, chat.tools, gen_prompt=False)
    spans: list[tuple[int, int]] = []
    for i, m in enumerate(chat.messages):
        if m["role"] != "assistant":
            continue
        before = render(tokenizer, chat.messages[:i], chat.tools, gen_prompt=True) if i else ""
        through = render(tokenizer, chat.messages[: i + 1], chat.tools, gen_prompt=False)
        if not (full.startswith(through) and through.startswith(before)):
            raise DataError(
                "the chat template does not render conversations as growing prefixes; "
                "cannot isolate assistant turns"
            )
        spans.append((len(before), len(through)))
    return full, spans


def tokenize(tokenizer: Any, chat: Chat, max_len: int) -> Example | None:
    """Token ids with labels on assistant tokens only; `None` when longer than `max_len`."""
    text, spans = assistant_spans(tokenizer, chat)
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids: list[int] = enc["input_ids"]
    if len(ids) > max_len:
        return None
    labels = [IGNORE_INDEX] * len(ids)
    for k, (start, _end) in enumerate(enc["offset_mapping"]):
        if any(a <= start < b for a, b in spans):
            labels[k] = ids[k]
    return Example(chat.id, ids, labels)


# ---------------------------------------------------------------- tool-call parsing (agent eval)

_TOOL_CALL = re.compile(r"<tool_call>\s*(.*?)\s*(?:</tool_call>|$)", re.S)


def _first_json_object(text: str) -> Any:
    dec = json.JSONDecoder()
    for m in re.finditer(r"\{", text):
        try:
            obj, _ = dec.raw_decode(text, m.start())
            return obj
        except json.JSONDecodeError:
            continue
    return None


def parse_generated_call(text: str) -> tuple[bool, dict[str, Any] | None]:
    """`(emitted_a_call, call)` for model output: Hermes `<tool_call>` blocks, else bare JSON.

    `call` is `{"name", "arguments"}` when the call is well-formed, `None` when the model tried to
    call a tool (or produced JSON) but it does not parse.
    """
    m = _TOOL_CALL.search(text)
    body = m.group(1) if m else None
    candidate = _first_json_object(body if body is not None else text)
    tried = body is not None or candidate is not None
    if isinstance(candidate, dict) and isinstance(candidate.get("name"), str):
        args = _arguments(candidate.get("arguments", candidate.get("parameters", {})))
        return True, {"name": candidate["name"], "arguments": args}
    return tried, None


@dataclass
class ToolCallCase:
    """Prefix of a conversation that ends right before an assistant turn with a tool call."""

    id: str
    prefix: list[Message]
    tools: list[dict[str, Any]]
    expected: dict[str, Any]


def tool_call_cases(chats: list[Chat], limit: int) -> list[ToolCallCase]:
    """The first tool-calling assistant turn of each conversation, up to `limit` cases."""
    cases: list[ToolCallCase] = []
    for c in chats:
        for i, m in enumerate(c.messages):
            if m["role"] == "assistant" and m.get("tool_calls"):
                fn = m["tool_calls"][0]["function"]
                cases.append(ToolCallCase(c.id, c.messages[:i], c.tools, dict(fn)))
                break
        if len(cases) >= limit:
            break
    return cases


def score_tool_calls(expected: list[dict[str, Any]], generated: list[str]) -> dict[str, float]:
    """Exact-match tool-use metrics (BFCL-style AST match on name and arguments)."""
    n = len(expected)
    if n == 0:
        return {}
    emitted = parsed = name_ok = args_ok = 0
    for exp, out in zip(expected, generated, strict=True):
        tried, call = parse_generated_call(out)
        emitted += tried
        if call is None:
            continue
        parsed += 1
        if call["name"] == exp["name"]:
            name_ok += 1
            if call["arguments"] == exp["arguments"]:
                args_ok += 1
    return {
        "tool_call_rate": emitted / n,
        "valid_call_rate": parsed / n,
        "tool_name_accuracy": name_ok / n,
        "exact_match": args_ok / n,
    }
