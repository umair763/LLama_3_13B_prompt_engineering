from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class LlamaResources:
	model: Any
	tokenizer: Any


def load_llama(*, model_id: str, device_map: str = "auto", torch_dtype: str | None = None) -> LlamaResources:
	"""Load a LLaMA(-compatible) instruct model via HuggingFace Transformers.

	This repo is designed to run on GPU (e.g., Kaggle). The exact model weights are
	not shipped here; provide a valid `model_id` (local path or HF model name).
	"""

	try:
		from transformers import AutoModelForCausalLM, AutoTokenizer
	except Exception as exc:  # pragma: no cover
		raise RuntimeError(
			"Transformers is required. Install: pip install transformers accelerate sentencepiece"
		) from exc

	model_kwargs: dict[str, Any] = {"device_map": device_map}
	if torch_dtype:
		try:
			import torch

			model_kwargs["torch_dtype"] = getattr(torch, torch_dtype)
		except Exception:
			# If torch not available or dtype invalid, ignore and let HF decide.
			pass

	tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
	model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)

	return LlamaResources(model=model, tokenizer=tokenizer)

