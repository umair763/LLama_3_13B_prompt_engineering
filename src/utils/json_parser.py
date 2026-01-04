from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any


class JSONParseError(ValueError):
	pass


_FENCED_JSON_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)


def extract_first_json_object(text: str) -> dict[str, Any]:
	"""Extract the first valid JSON object from arbitrary text.

	Handles:
	- raw JSON
	- ```json fenced blocks
	- leading/trailing model chatter
	"""

	if not isinstance(text, str) or not text.strip():
		raise JSONParseError("Empty LLM output; cannot parse JSON.")

	fenced_match = _FENCED_JSON_RE.search(text)
	if fenced_match:
		candidate = fenced_match.group(1)
		return _loads_object(candidate)

	decoder = json.JSONDecoder()
	# Try raw-decode at each '{' occurrence.
	for match in re.finditer(r"\{", text):
		idx = match.start()
		try:
			obj, end = decoder.raw_decode(text[idx:])
		except json.JSONDecodeError:
			continue
		if isinstance(obj, dict):
			return obj
	raise JSONParseError("Could not find a valid JSON object in LLM output.")


def _loads_object(candidate: str) -> dict[str, Any]:
	try:
		obj = json.loads(candidate)
	except json.JSONDecodeError as exc:
		raise JSONParseError(f"Invalid JSON: {exc}") from exc
	if not isinstance(obj, dict):
		raise JSONParseError("Parsed JSON was not an object.")
	return obj


def coerce_int(value: Any, *, default: int | None = None) -> int | None:
	if value is None:
		return default
	if isinstance(value, bool):
		return default
	if isinstance(value, int):
		return value
	if isinstance(value, float):
		return int(round(value))
	if isinstance(value, str):
		s = value.strip()
		if not s:
			return default
		try:
			return int(float(s))
		except ValueError:
			return default
	return default


def coerce_float(value: Any, *, default: float | None = None) -> float | None:
	if value is None:
		return default
	if isinstance(value, bool):
		return default
	if isinstance(value, (int, float)):
		return float(value)
	if isinstance(value, str):
		s = value.strip()
		if not s:
			return default
		try:
			return float(s)
		except ValueError:
			return default
	return default


def clamp(value: float, lo: float, hi: float) -> float:
	return max(lo, min(hi, value))


@dataclass(frozen=True)
class NormalizedSubtask:
	title: str
	estimatedMinutes: int
	difficulty: int
	frogScore: float | None = None


def normalize_subtasks_payload(payload: dict[str, Any]) -> list[NormalizedSubtask]:
	"""Validate + normalize a {"subtasks": [...]} payload."""

	subtasks_raw = payload.get("subtasks")
	if not isinstance(subtasks_raw, list) or not subtasks_raw:
		raise JSONParseError("Expected a non-empty 'subtasks' array.")

	normalized: list[NormalizedSubtask] = []
	for item in subtasks_raw:
		if not isinstance(item, dict):
			continue
		title = str(item.get("title", "")).strip()
		if not title:
			continue

		minutes = coerce_int(item.get("estimatedMinutes"), default=60) or 60
		minutes = max(5, minutes)

		difficulty = coerce_int(item.get("difficulty"), default=3) or 3
		difficulty = int(clamp(float(difficulty), 1, 5))

		frog = item.get("frogScore")
		frog_score = None
		if frog is not None:
			frog_score = coerce_float(frog, default=0.5)
			if frog_score is None:
				frog_score = 0.5
			frog_score = clamp(float(frog_score), 0.0, 1.0)

		normalized.append(
			NormalizedSubtask(
				title=title,
				estimatedMinutes=int(minutes),
				difficulty=difficulty,
				frogScore=frog_score,
			)
		)

	if not normalized:
		raise JSONParseError("No valid subtasks found after normalization.")
	return normalized

