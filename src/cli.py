from __future__ import annotations

import argparse
import json
import os

from src.planner.unified_planner import load_tasks_from_json, plan_month


def main() -> int:
    parser = argparse.ArgumentParser(description="Unified AI task planner (LLaMA + rules)")
    parser.add_argument(
        "--input",
        required=True,
        help="Path to tasks JSON (list of {title, deadline, description})",
    )
    parser.add_argument(
        "--model-id",
        default=os.environ.get("MODEL_ID", ""),
        help="HF model id or local path (or set MODEL_ID env var)",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional output path (defaults to stdout)",
    )

    args = parser.parse_args()
    if not args.model_id:
        raise SystemExit(
            "Missing --model-id (or MODEL_ID env var). Example: --model-id meta-llama/Meta-Llama-3-13B-Instruct"
        )

    tasks = load_tasks_from_json(args.input)
    result = plan_month(tasks=tasks, model_id=args.model_id)

    out = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(out)
    else:
        print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
