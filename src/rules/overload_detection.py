from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Iterable


@dataclass(frozen=True)
class OverloadWindow:
	start: date
	end: date
	totalMinutes: int
	capacityMinutes: int
	urgentCount: int
	highEffortCount: int
	reason: str
	suggestions: list[str]


def _window_capacity_minutes(window_days: int) -> int:
	# Assume 6 focused hours per workday, 5/7 workdays.
	workdays = max(1, round(window_days * 5 / 7))
	return int(workdays * 360)


def detect_overload_windows(
	*,
	items: Iterable[dict],
	today: date,
	min_window_days: int = 7,
	max_window_days: int = 15,
	urgent_urgency_threshold: float = 0.75,
	high_effort_minutes_threshold: int = 90,
	overload_ratio_threshold: float = 0.90,
	min_urgent_count: int = 6,
) -> list[OverloadWindow]:
	"""Detect overload windows in the next 7–15 days.

	Expected item fields:
	- estimatedMinutes (int)
	- urgencyScore (float 0..1)
	- deadlineDate (date)
	- difficulty (int)
	- priorityScore (float)
	"""

	items_list: list[dict] = []
	for it in items:
		if not isinstance(it, dict):
			continue
		if not isinstance(it.get("deadlineDate"), date):
			continue
		items_list.append(it)

	overloads: list[OverloadWindow] = []
	for window_days in range(min_window_days, max_window_days + 1):
		start = today
		end = today + timedelta(days=window_days)
		in_window = [
			it
			for it in items_list
			if start <= it["deadlineDate"] < end
		]
		if not in_window:
			continue

		total_minutes = int(sum(int(it.get("estimatedMinutes", 0) or 0) for it in in_window))
		capacity = _window_capacity_minutes(window_days)

		urgent_count = sum(1 for it in in_window if float(it.get("urgencyScore", 0.0) or 0.0) >= urgent_urgency_threshold)
		high_effort_count = sum(1 for it in in_window if int(it.get("estimatedMinutes", 0) or 0) >= high_effort_minutes_threshold)

		overloaded = (total_minutes >= int(capacity * overload_ratio_threshold)) or (urgent_count >= min_urgent_count)
		if not overloaded:
			continue

		suggestions: list[str] = []
		suggestions.append("Split or simplify the highest-effort items in this window into 30–90 minute chunks.")
		suggestions.append("Pull forward one or two medium-urgency items earlier in the month to reduce deadline clustering.")
		suggestions.append("Defer or downscope any low-urgency items that are not deadline-critical.")

		reason = (
			f"Total load {total_minutes} min vs capacity {capacity} min; "
			f"urgent={urgent_count}, highEffort={high_effort_count}."
		)

		overloads.append(
			OverloadWindow(
				start=start,
				end=end,
				totalMinutes=total_minutes,
				capacityMinutes=capacity,
				urgentCount=urgent_count,
				highEffortCount=high_effort_count,
				reason=reason,
				suggestions=suggestions,
			)
		)

	# Keep only the strongest overload signal per (start,end) and avoid spam: prefer larger ratios.
	overloads.sort(key=lambda w: (w.totalMinutes / max(1, w.capacityMinutes)), reverse=True)
	return overloads[:3]

