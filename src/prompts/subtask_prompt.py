from __future__ import annotations


def build_subtask_generation_prompt(*, task: str, deadline: str | None = None, description: str | None = None) -> str:
	"""Prompt for generating actionable subtasks (30–90 minutes) as JSON only."""

	deadline_text = deadline or ""
	description_text = description or ""

	return (
		"SYSTEM:\n"
		"You are an expert task planner.\n"
		"Break the given task into actionable subtasks.\n"
		"Constraints:\n"
		"- Each subtask must be executable on its own.\n"
		"- Each subtask should take 30–90 minutes.\n"
		"- Avoid vague steps (e.g., 'think about', 'research').\n"
		"- Order subtasks hardest/most important first (eat the frog).\n"
		"- Output JSON only. No prose.\n\n"
		"USER:\n"
		f"Task: \"{task}\"\n"
		f"Deadline: \"{deadline_text}\"\n"
		f"Description: \"{description_text}\"\n\n"
		"Return JSON with this exact shape:\n"
		"{\n"
		"  \"subtasks\": [\n"
		"    {\"title\": \"string\", \"estimatedMinutes\": 30, \"difficulty\": 1}\n"
		"  ]\n"
		"}\n"
	)

