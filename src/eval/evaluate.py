from __future__ import annotations

import argparse
import json
from typing import Any, Dict, List, Tuple

try:
    from sklearn.metrics import precision_score, recall_score, f1_score
except Exception:
    precision_score = recall_score = f1_score = None

try:
    from scipy.stats import spearmanr
except Exception:
    spearmanr = None


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_title(s: str) -> str:
    return (s or "").strip().lower()


def eval_subtask_generation(gt: Dict[str, Any], pred: Dict[str, Any]) -> Dict[str, Any]:
    """Compute precision/recall/F1 of subtask titles vs ground truth.

    Ground truth shape expectation:
    { "expectedSubtasks": [ {"title": "..."}, ... ] }
    Prediction shape uses planner output: { "prioritizedSubtasks": [ {"subtaskTitle": "..."}, ... ] }
    """
    gt_items = gt.get("expectedSubtasks", [])
    pred_items = pred.get("prioritizedSubtasks", [])

    gt_titles = {normalize_title(x.get("title", "")) for x in gt_items}
    pred_titles = {normalize_title(x.get("subtaskTitle", "")) for x in pred_items}

    gt_titles.discard("")
    pred_titles.discard("")

    tp = len(gt_titles & pred_titles)
    fp = len(pred_titles - gt_titles)
    fn = len(gt_titles - pred_titles)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def eval_priority_ranking(gt: Dict[str, Any], pred: Dict[str, Any]) -> Dict[str, Any]:
    """Compute Spearman correlation between ground-truth ranks and predicted order.

    Ground truth expectation:
    { "rankedSubtasks": [ {"title": "...", "rank": 1}, ... ] }
    Prediction: planner output list order defines rank (1..N).
    """
    gt_rank_list = gt.get("rankedSubtasks", [])
    pred_list = pred.get("prioritizedSubtasks", [])

    # Map titles -> gt rank
    gt_rank: Dict[str, int] = {}
    for x in gt_rank_list:
        t = normalize_title(x.get("title", ""))
        r = x.get("rank")
        if t and isinstance(r, (int, float)):
            gt_rank[t] = int(r)

    # Build predicted rank sequence for titles present in gt
    pred_order: Dict[str, int] = {}
    for i, x in enumerate(pred_list, start=1):
        t = normalize_title(x.get("subtaskTitle", ""))
        if t:
            pred_order[t] = i

    # Intersect by titles
    common = [t for t in gt_rank.keys() if t in pred_order]
    if not common or spearmanr is None:
        return {"spearman": None, "count": 0}

    gt_vals = [gt_rank[t] for t in common]
    pred_vals = [pred_order[t] for t in common]

    corr, _ = spearmanr(gt_vals, pred_vals)
    return {"spearman": float(corr), "count": len(common)}


def eval_overload_detection(gt: Dict[str, Any], pred: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate overload window detection by comparing weeks flagged.

    Ground truth expectation:
    { "overloadWeeks": ["2026-W02", "2026-W03"] }
    Prediction: planner output contains {"overload": [ {"start": "...", "end": "..."}, ... ]}
    We'll convert start/end to week keys and compare sets.
    """
    from datetime import date
    from src.scoring.priority_scoring import parse_date

    gt_weeks = set(gt.get("overloadWeeks", []))

    pred_weeks: set[str] = set()
    for w in pred.get("overload", []):
        start = parse_date(w.get("start"))
        end = parse_date(w.get("end"))
        if start:
            y, wk, _ = start.isocalendar()
            pred_weeks.add(f"{y}-W{wk:02d}")
        if end:
            y, wk, _ = end.isocalendar()
            pred_weeks.add(f"{y}-W{wk:02d}")

    tp = len(gt_weeks & pred_weeks)
    fp = len(pred_weeks - gt_weeks)
    fn = len(gt_weeks - pred_weeks)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate planner outputs vs ground truth")
    parser.add_argument("--ground-truth", required=True, help="Path to ground truth JSON")
    parser.add_argument("--pred", required=True, help="Path to planner output JSON (plan.json)")
    args = parser.parse_args()

    gt = load_json(args.ground_truth)
    pred = load_json(args.pred)

    subtask_metrics = eval_subtask_generation(gt, pred)
    ranking_metrics = eval_priority_ranking(gt, pred)
    overload_metrics = eval_overload_detection(gt, pred)

    report = {
        "subtasks": subtask_metrics,
        "ranking": ranking_metrics,
        "overload": overload_metrics,
    }

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
