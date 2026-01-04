from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Iterable


def parse_date(value: str | None) -> date | None:
	if not value:
		return None
	value = value.strip()
	if not value:
		return None
	# Accept ISO-8601 date or datetime.
	for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
		try:
			dt = datetime.strptime(value, fmt)
			return dt.date()
		except ValueError:
			continue
	try:
		return datetime.fromisoformat(value).date()
	except ValueError:
		return None


def urgency_score(*, deadline: date | None, today: date, horizon_days: int = 30) -> float:
	"""0..1 urgency based on deadline proximity within a horizon."""

	if deadline is None:
		return 0.1
	days = (deadline - today).days
	if days <= 0:
		return 1.0
	if days >= horizon_days:
		return 0.0
	return max(0.0, 1.0 - (days / float(horizon_days)))


@dataclass(frozen=True)
class ScoredItem:
	taskTitle: str
	subtaskTitle: str
	estimatedMinutes: int
	difficulty: int
	frogScore: float
	urgencyScore: float
	priorityScore: float
	deadline: str | None = None


def compute_priority_score(*, frog_score: float, difficulty: int, urgency: float) -> float:
	# As specified: priorityScore = frogScore*0.4 + difficulty*0.3 + urgencyScore*0.3
	return (frog_score * 0.4) + (difficulty * 0.3) + (urgency * 0.3)


def score_subtasks(
	*,
	task_title: str,
	deadline: str | None,
	subtasks: Iterable[dict],
	today: date,
) -> list[ScoredItem]:
	dd = parse_date(deadline)
	urgency = urgency_score(deadline=dd, today=today)

	scored: list[ScoredItem] = []
	for st in subtasks:
		title = str(st.get("title", "")).strip()
		if not title:
			continue
		minutes = int(st.get("estimatedMinutes", 60) or 60)
		difficulty = int(st.get("difficulty", 3) or 3)
		frog = float(st.get("frogScore", 0.5) or 0.5)

		ps = compute_priority_score(frog_score=frog, difficulty=difficulty, urgency=urgency)
		scored.append(
			ScoredItem(
				taskTitle=task_title,
				subtaskTitle=title,
				estimatedMinutes=minutes,
				difficulty=difficulty,
				frogScore=frog,
				urgencyScore=urgency,
				priorityScore=ps,
				deadline=deadline,
			)
		)

	scored.sort(key=lambda x: x.priorityScore, reverse=True)
	return scored

