from __future__ import annotations

import argparse
import json
from datetime import datetime
from typing import Any

import matplotlib.pyplot as plt


def load_json(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def plot_weeks(plan: dict[str, Any], outdir: str) -> str:
    weeks = plan.get("schedule", {}).get("weeks", [])
    if not weeks:
        return ""
    labels = [w.get("week", "?") for w in weeks]
    values = [int(w.get("totalMinutes", 0) or 0) for w in weeks]

    plt.figure(figsize=(8, 4))
    plt.bar(labels, values, color="#4C78A8")
    plt.ylabel("Total minutes")
    plt.xlabel("ISO Week")
    plt.title("Weekly Planned Load")
    plt.xticks(rotation=45)
    plt.tight_layout()
    path = f"{outdir}/weekly_load.png"
    plt.savefig(path)
    plt.close()
    return path


essential_cols = [
    "taskTitle",
    "subtaskTitle",
    "estimatedMinutes",
    "priorityScore",
]


def plot_top_priorities(plan: dict[str, Any], outdir: str, top_k: int = 10) -> str:
    items = plan.get("prioritizedSubtasks", [])[:top_k]
    if not items:
        return ""
    labels = [x.get("subtaskTitle", "?")[:40] for x in items]
    scores = [float(x.get("priorityScore", 0)) for x in items]

    plt.figure(figsize=(8, 4))
    plt.barh(range(len(labels)), scores, color="#72B7B2")
    plt.yticks(range(len(labels)), labels)
    plt.xlabel("Priority Score")
    plt.title("Top Priorities (Eat the Frog)")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    path = f"{outdir}/top_priorities.png"
    plt.savefig(path)
    plt.close()
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description="Visualize planner output")
    parser.add_argument("--plan", required=True, help="Path to plan.json")
    parser.add_argument("--outdir", required=True, help="Directory to write figures")
    parser.add_argument("--top-k", type=int, default=10, help="Top-K priorities to show")
    args = parser.parse_args()

    plan = load_json(args.plan)

    weeks_path = plot_weeks(plan, args.outdir)
    top_path = plot_top_priorities(plan, args.outdir, args.top_k)

    print(json.dumps({"weeks": weeks_path, "top_priorities": top_path}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
