from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class GenerationConfig:
	max_new_tokens: int = 512
	temperature: float = 0.2
	top_p: float = 0.9
	do_sample: bool = True


class LlamaClient:
	def __init__(self, *, model: Any, tokenizer: Any, config: GenerationConfig | None = None):
		self._model = model
		self._tokenizer = tokenizer
		self._config = config or GenerationConfig()

	def generate(self, prompt: str) -> str:
		if not isinstance(prompt, str) or not prompt.strip():
			raise ValueError("Prompt must be a non-empty string")

		inputs = self._tokenizer(prompt, return_tensors="pt")

		# Move tensors to the model's device when applicable.
		try:
			device = getattr(self._model, "device", None)
			if device is not None:
				inputs = {k: v.to(device) for k, v in inputs.items()}
		except Exception:
			pass

		gen = self._model.generate(
			**inputs,
			max_new_tokens=self._config.max_new_tokens,
			temperature=self._config.temperature,
			top_p=self._config.top_p,
			do_sample=self._config.do_sample,
			pad_token_id=getattr(self._tokenizer, "eos_token_id", None),
		)

		text = self._tokenizer.decode(gen[0], skip_special_tokens=True)
		# Many instruction models echo the prompt; try to strip it.
		if text.startswith(prompt):
			return text[len(prompt) :].lstrip()
		return text

