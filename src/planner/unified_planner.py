from __future__ import annotations

import json
import datetime
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any

from src.llm.llama_inference import GenerationConfig, LlamaClient
from src.llm.llama_loader import load_llama
from src.prompts.frog_priority_prompt import build_frog_scoring_prompt
from src.prompts.subtask_prompt import build_subtask_generation_prompt
from src.rules.overload_detection import detect_overload_windows
from src.scoring.priority_scoring import parse_date, score_subtasks, urgency_score
from src.utils.json_parser import (
    JSONParseError,
    extract_first_json_object,
    normalize_subtasks_payload,
)


@dataclass(frozen=True)
class TaskInput:
    title: str
    deadline: str | None = None
    description: str | None = None


def _today() -> date:
    return datetime.now().date()


def _ensure_0_1(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def plan_month(
    *,
    tasks: list[TaskInput],
    model_id: str,
    today: date | None = None,
) -> dict[str, Any]:
    """End-to-end monthly planner.

    Flow:
    - For each task: LLM generates subtasks JSON
    - LLM assigns frogScore+difficulty JSON
    - Rule engine computes urgency/priority, sorts
    - Detect overload in next 7–15 days
    - Produce a simple day-by-day schedule
    """

    today = today or _today()

    resources = load_llama(model_id=model_id)
    llm = LlamaClient(model=resources.model, tokenizer=resources.tokenizer, config=GenerationConfig())

    all_scored: list[dict[str, Any]] = []
    generation_errors: list[dict[str, str]] = []

    for task in tasks:
        try:
            # 1) Subtask generation
            gen_prompt = build_subtask_generation_prompt(
                task=task.title, deadline=task.deadline, description=task.description
            )
            raw = llm.generate(gen_prompt)
            payload = extract_first_json_object(raw)
            subtasks_norm = normalize_subtasks_payload(payload)
            subtasks_for_scoring = [
                {
                    "title": st.title,
                    "estimatedMinutes": st.estimatedMinutes,
                    "difficulty": st.difficulty,
                }
                for st in subtasks_norm
            ]

            # 2) Frog scoring (difficulty may be adjusted)
            frog_prompt = build_frog_scoring_prompt(
                task=task.title, deadline=task.deadline, subtasks=subtasks_for_scoring
            )
            raw2 = llm.generate(frog_prompt)
            payload2 = extract_first_json_object(raw2)
            subtasks_norm2 = normalize_subtasks_payload(payload2)

            scored = score_subtasks(
                task_title=task.title,
                deadline=task.deadline,
                subtasks=[
                    {
                        "title": st.title,
                        "estimatedMinutes": st.estimatedMinutes,
                        "difficulty": st.difficulty,
                        "frogScore": _ensure_0_1(st.frogScore if st.frogScore is not None else 0.5),
                    }
                    for st in subtasks_norm2
                ],
                today=today,
            )

            for item in scored:
                dd = parse_date(item.deadline)
                all_scored.append(
                    {
                        "taskTitle": item.taskTitle,
                        "subtaskTitle": item.subtaskTitle,
                        "estimatedMinutes": item.estimatedMinutes,
                        "difficulty": item.difficulty,
                        "frogScore": item.frogScore,
                        "urgencyScore": item.urgencyScore,
                        "priorityScore": item.priorityScore,
                        "deadline": item.deadline,
                        "deadlineDate": dd,
                    }
                )
        except JSONParseError as exc:
            generation_errors.append(
                {
                    "taskTitle": task.title,
                    "error": f"JSONParseError: {exc}",
                }
            )
        except Exception as exc:  # pragma: no cover
            generation_errors.append(
                {
                    "taskTitle": task.title,
                    "error": f"UnexpectedError: {type(exc).__name__}: {exc}",
                }
            )

    # Global sort: frog-first priority score already includes urgency.
    all_scored.sort(key=lambda x: x.get("priorityScore", 0.0), reverse=True)

    schedule = build_schedule(items=all_scored, today=today)

    overload_items = []
    for it in all_scored:
        dd = it.get("deadlineDate")
        if dd is None:
            continue
        overload_items.append(it)

    overloads = detect_overload_windows(items=overload_items, today=today)
    overload_payload = [
        {
            "start": w.start.isoformat(),
            "end": w.end.isoformat(),
            "totalMinutes": w.totalMinutes,
            "capacityMinutes": w.capacityMinutes,
            "urgentCount": w.urgentCount,
            "highEffortCount": w.highEffortCount,
            "reason": w.reason,
            "suggestions": w.suggestions,
        }
        for w in overloads
    ]

    # Drop non-serializable deadlineDate from output.
    serialized_scored = [
        {k: v for k, v in it.items() if k != "deadlineDate"}
        for it in all_scored
    ]

    result = {
        "generatedAt": today.isoformat(),
        "tasksCount": len(tasks),
        "subtasksCount": len(serialized_scored),
        "prioritizedSubtasks": serialized_scored,
        "schedule": schedule,
        "overload": overload_payload,
        "errors": generation_errors,
    }
    return _ensure_json_safe(result)


def build_schedule(*, items: list[dict[str, Any]], today: date) -> dict[str, Any]:
    """Greedy schedule builder.

    Capacity assumption: 360 minutes per workday (Mon–Fri).
    Places each item on the earliest day with remaining capacity before its deadline.
    """

    def is_workday(d: date) -> bool:
        return d.weekday() < 5

    def next_day(d: date) -> date:
        return d + timedelta(days=1)

    def day_capacity_minutes(d: date) -> int:
        return 360 if is_workday(d) else 0

    # Precompute a reasonable planning horizon.
    deadlines = [parse_date(it.get("deadline")) for it in items]
    max_deadline = max([d for d in deadlines if d is not None], default=today + timedelta(days=30))
    horizon_end = max(today + timedelta(days=30), max_deadline)

    day_buckets: dict[date, list[dict[str, Any]]] = {}
    remaining: dict[date, int] = {}

    d = today
    while d <= horizon_end:
        remaining[d] = day_capacity_minutes(d)
        day_buckets[d] = []
        d = next_day(d)

    # Sort by priority (already sorted), break ties by earliest deadline.
    def sort_key(it: dict[str, Any]) -> tuple[float, int]:
        ps = float(it.get("priorityScore", 0.0) or 0.0)
        dd = parse_date(it.get("deadline"))
        days = (dd - today).days if dd is not None else 10_000
        return (-ps, days)

    for it in sorted(items, key=sort_key):
        minutes = int(it.get("estimatedMinutes", 60) or 60)
        dd = parse_date(it.get("deadline")) or horizon_end
        target_end = min(dd, horizon_end)

        placed = False
        cursor = today
        while cursor <= target_end:
            if remaining.get(cursor, 0) >= minutes:
                day_buckets[cursor].append({**it})
                remaining[cursor] -= minutes
                placed = True
                break
            cursor = next_day(cursor)

        if not placed:
            # Put on deadline day even if overbooked.
            day_buckets[target_end].append({**it, "overbooked": True})

    days_out: list[dict[str, Any]] = []
    for d in sorted(day_buckets.keys()):
        if not day_buckets[d]:
            continue
        total = sum(int(x.get("estimatedMinutes", 0) or 0) for x in day_buckets[d])
        days_out.append({"date": d.isoformat(), "totalMinutes": total, "items": day_buckets[d]})

    weeks: dict[str, int] = {}
    for day in days_out:
        d = parse_date(day["date"]) or today
        year, week, _ = d.isocalendar()
        key = f"{year}-W{week:02d}"
        weeks[key] = weeks.get(key, 0) + int(day.get("totalMinutes", 0) or 0)

    weeks_out = [{"week": k, "totalMinutes": v} for k, v in sorted(weeks.items())]

    return {"days": days_out, "weeks": weeks_out}


def _ensure_json_safe(obj: Any) -> Any:
    """Recursively convert non-JSON types (date/datetime, sets) to serializable forms."""
    if isinstance(obj, dict):
        return {k: _ensure_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_ensure_json_safe(v) for v in obj]
    if isinstance(obj, (set, tuple)):
        return [_ensure_json_safe(v) for v in obj]
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    return obj


def load_tasks_from_json(path: str) -> list[TaskInput]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Input JSON must be a list of tasks")

    tasks: list[TaskInput] = []
    for it in data:
        if not isinstance(it, dict):
            continue
        title = str(it.get("title", "")).strip()
        if not title:
            continue
        tasks.append(
            TaskInput(
                title=title,
                deadline=(str(it.get("deadline")).strip() if it.get("deadline") else None),
                description=(
                    str(it.get("description")).strip() if it.get("description") else None
                ),
            )
        )
    if not tasks:
        raise ValueError("No valid tasks found in input JSON")
    return tasks
