from __future__ import annotations

import json


def build_frog_scoring_prompt(*, task: str, subtasks: list[dict], deadline: str | None = None) -> str:
	"""Prompt to assign frogScore (0-1) and difficulty (1-5) per subtask as JSON only."""

	deadline_text = deadline or ""
	subtasks_json = json.dumps(subtasks, ensure_ascii=False)

	return (
		"SYSTEM:\n"
		"You are a productivity assistant using the 'eat the frog' method.\n"
		"For each subtask, assign:\n"
		"- difficulty: integer 1–5 (5 = hardest)\n"
		"- frogScore: number 0–1 (1 = most important/urgent/high impact)\n"
		"Rules:\n"
		"- Keep the same titles and estimatedMinutes.\n"
		"- If a subtask is a prerequisite for others, it should have a higher frogScore.\n"
		"- Output JSON only. No prose.\n\n"
		"USER:\n"
		f"Parent task: \"{task}\"\n"
		f"Deadline: \"{deadline_text}\"\n"
		f"Subtasks JSON: {subtasks_json}\n\n"
		"Return JSON with this exact shape:\n"
		"{\n"
		"  \"subtasks\": [\n"
		"    {\"title\": \"string\", \"estimatedMinutes\": 30, \"difficulty\": 1, \"frogScore\": 0.5}\n"
		"  ]\n"
		"}\n"
	)

