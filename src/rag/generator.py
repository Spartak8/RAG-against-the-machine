"""Grounded answer generation using Qwen/Qwen3-0.6B."""

from pathlib import Path
import re
from typing import Any

from ..models import MinimalSource

DEFAULT_MODEL_NAME = "Qwen/Qwen3-0.6B"
_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
_SYSTEM_PROMPT = (
    "You are a technical assistant answering questions about the vLLM "
    "codebase. Answer concisely and accurately using ONLY the provided "
    "context snippets. Do not invent facts outside the context."
)


def read_source_snippet(source: MinimalSource) -> str:
    """Read the exact character span for a MinimalSource from disk."""
    path = Path(source.file_path)
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            content = handle.read()
    except OSError:
        return ""
    return content[source.first_character_index:source.last_character_index]


def build_context(
    sources: list[MinimalSource],
    max_sources: int = 5,
    max_chars: int = 6000,
) -> str:
    """Assemble retrieved source snippets within a character budget."""
    parts: list[str] = []
    total_chars = 0

    for idx, src in enumerate(sources[:max_sources], start=1):
        snippet = read_source_snippet(src).strip()
        if not snippet:
            continue
        header = f"[Source {idx}: {src.file_path}]\n"
        remaining = max_chars - total_chars - len(header)
        if remaining <= 0:
            break
        block = header + snippet[:remaining]
        parts.append(block)
        total_chars += len(block) + 2

    return "\n\n".join(parts)


class AnswerGenerator:
    """Local LLM wrapper for grounded RAG answer generation."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        max_new_tokens: int = 256,
    ) -> None:
        """Load tokenizer and causal LM weights for model_name."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self._torch: Any = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer: Any = AutoTokenizer.from_pretrained(model_name)
        self.model: Any = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=(
                torch.float16 if self.device == "cuda" else torch.float32
            ),
        )
        self.model.to(self.device)
        self.model.eval()

    def generate(
        self,
        question: str,
        sources: list[MinimalSource],
        max_sources: int = 5,
    ) -> str:
        """Generate an answer grounded in the given retrieved sources."""
        if not question or not question.strip():
            return "No question provided."

        context = build_context(sources, max_sources=max_sources)
        if not context:
            return (
                "No relevant context was retrieved from the codebase to "
                "answer this question."
            )

        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}",
            },
        ]

        try:
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_len = inputs["input_ids"].shape[1]

        with self._torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generated = output_ids[0][input_len:]
        raw_text = str(
            self.tokenizer.decode(generated, skip_special_tokens=True)
        )
        cleaned = _THINK_RE.sub("", raw_text).strip()
        return cleaned if cleaned else "Unable to generate an answer."
