"""Stable instruction templates for RAV's captioning and grounded responses."""

CAPTION_PROMPT = (
    "Describe the visible scene accurately and concisely."
)
GROUNDING_INSTRUCTION = (
    "Answer the user's question directly using visible evidence. Be concise and mention the "
    "objects and scene details you can see."
)


def caption_messages(
    caption: str | None = None, prompt: str = CAPTION_PROMPT
) -> list[dict[str, object]]:
    messages: list[dict[str, object]] = [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}
    ]
    if caption is not None:
        messages.append({"role": "assistant", "content": [{"type": "text", "text": caption}]})
    return messages
