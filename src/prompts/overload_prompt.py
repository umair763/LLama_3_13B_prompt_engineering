from __future__ import annotations

import json


def build_overload_prompt(
	*,
	window_start: str,
	window_end: str,
	items: list[dict],
) -> str:
	"""Prompt that suggests splitting/reordering to reduce overload in next 7–15 days."""

	items_json = json.dumps(items, ensure_ascii=False)

	return (
		"SYSTEM:\n"
		"You are a productivity assistant.\n"
		"If the upcoming window contains too many urgent/high-effort tasks, propose ways to reduce overload.\n"
		"You may suggest splitting tasks into smaller subtasks (30–90 minutes) and reordering.\n"
		"Output JSON only. No prose.\n\n"
		"USER:\n"
		f"Window: {window_start} to {window_end}\n"
		f"Items (tasks/subtasks) JSON: {items_json}\n\n"
		"Return JSON with this exact shape:\n"
		"{\n"
		"  \"suggestions\": [\n"
		"    {\n"
		"      \"itemTitle\": \"string\",\n"
		"      \"action\": \"split|reorder|defer\",\n"
		"      \"reason\": \"string\",\n"
		"      \"proposedSubtasks\": [\n"
		"        {\"title\": \"string\", \"estimatedMinutes\": 60, \"difficulty\": 3}\n"
		"      ]\n"
		"    }\n"
		"  ]\n"
		"}\n"
	)

