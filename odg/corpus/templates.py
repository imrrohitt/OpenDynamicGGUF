"""Chat-template renderers for calibration documents."""

from __future__ import annotations

from typing import Any


def render_chat(
    user: str,
    assistant: str | None = None,
    *,
    template: str | None = "gemma3",
) -> str:
    """Render a user(/assistant) turn with the model's chat template."""
    t = (template or "plain").lower()
    if t in {"gemma", "gemma2", "gemma3", "functiongemma"}:
        return _gemma(user, assistant)
    if t in {"chatml", "qwen", "qwen2"}:
        return _chatml(user, assistant)
    if t in {"llama3", "llama-3"}:
        return _llama3(user, assistant)
    return _plain(user, assistant)


def render_document(doc: dict[str, Any], *, template: str | None) -> str:
    """
    doc keys:
      user / assistant  — single turn
      turns             — list of {role, content}
      text              — already-formatted raw text (no template)
    """
    if "text" in doc and doc["text"]:
        return str(doc["text"]).strip() + "\n"

    turns = doc.get("turns")
    if turns:
        return render_turns(turns, template=template)

    user = str(doc.get("user") or "").strip()
    assistant = doc.get("assistant")
    assistant_s = str(assistant).strip() if assistant is not None else None
    return render_chat(user, assistant_s, template=template)


def render_turns(turns: list[dict[str, str]], *, template: str | None) -> str:
    t = (template or "plain").lower()
    if t in {"gemma", "gemma2", "gemma3", "functiongemma"}:
        parts: list[str] = []
        for turn in turns:
            role = turn.get("role", "user")
            content = (turn.get("content") or "").strip()
            if role == "system":
                # Gemma has no system turn; prepend to first user
                continue
            gemma_role = "model" if role in {"assistant", "model"} else "user"
            parts.append(f"<start_of_turn>{gemma_role}\n{content}<end_of_turn>")
        # If system present, prepend to first user content
        sys_msgs = [x.get("content", "") for x in turns if x.get("role") == "system"]
        if sys_msgs and parts:
            sys_text = "\n".join(s.strip() for s in sys_msgs if s).strip()
            if sys_text and parts[0].startswith("<start_of_turn>user\n"):
                parts[0] = parts[0].replace(
                    "<start_of_turn>user\n",
                    f"<start_of_turn>user\n{sys_text}\n\n",
                    1,
                )
        return "\n".join(parts) + "\n"

    # Fallback: concatenate plain turns
    lines = []
    for turn in turns:
        role = turn.get("role", "user").capitalize()
        lines.append(f"{role}: {(turn.get('content') or '').strip()}")
    return "\n".join(lines) + "\n"


def _gemma(user: str, assistant: str | None) -> str:
    parts = [f"<start_of_turn>user\n{user.strip()}<end_of_turn>"]
    if assistant:
        parts.append(f"<start_of_turn>model\n{assistant.strip()}<end_of_turn>")
    return "\n".join(parts) + "\n"


def _chatml(user: str, assistant: str | None) -> str:
    parts = [
        "<|im_start|>user\n" + user.strip() + "<|im_end|>",
    ]
    if assistant:
        parts.append("<|im_start|>assistant\n" + assistant.strip() + "<|im_end|>")
    return "\n".join(parts) + "\n"


def _llama3(user: str, assistant: str | None) -> str:
    parts = [
        "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
        + user.strip()
        + "<|eot_id|>"
    ]
    if assistant:
        parts.append(
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
            + assistant.strip()
            + "<|eot_id|>"
        )
    return "".join(parts) + "\n"


def _plain(user: str, assistant: str | None) -> str:
    if assistant:
        return f"User: {user.strip()}\nAssistant: {assistant.strip()}\n"
    return f"User: {user.strip()}\n"
