from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import os


@dataclass(frozen=True)
class LlamaResources:
	model: Any
	tokenizer: Any


def load_llama(
	*,
	model_id: str,
	device_map: str = "auto",
	torch_dtype: str | None = None,
	local_files_only: bool | None = None,
) -> LlamaResources:
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

	if local_files_only is None:
		# Respect TRANSFORMERS_OFFLINE env var automatically
		local_files_only = bool(os.environ.get("TRANSFORMERS_OFFLINE"))

	tokenizer = AutoTokenizer.from_pretrained(
		model_id, use_fast=True, local_files_only=local_files_only
	)
	model = AutoModelForCausalLM.from_pretrained(
		model_id, local_files_only=local_files_only, **model_kwargs
	)

	return LlamaResources(model=model, tokenizer=tokenizer)

