# LLama_3_13B_prompt_engineering

AI-powered task planning pipeline:

- LLaMA 3 Instruct generates 30–90 minute subtasks (JSON)
- LLaMA assigns `difficulty` (1–5) + `frogScore` (0–1)
- Rule engine computes `priorityScore = frogScore*0.4 + difficulty*0.3 + urgencyScore*0.3`
- Scheduler produces a simple day/week plan
- Overload detector flags the next 7–15 days if too clustered

## Input

Provide a JSON file containing a list of tasks:

```json
[{ "title": "Prepare project presentation", "deadline": "2026-01-12", "description": "Q1 status + risks" }]
```

## Run

Set your model id (HF name or local path), then run the CLI:

- PowerShell:

```powershell
$env:MODEL_ID = "meta-llama/Meta-Llama-3-13B-Instruct"
python -m src.cli --input example_tasks.json --output plan.json
```

## Output

The CLI prints JSON containing:

- `prioritizedSubtasks`: globally sorted subtasks (eat-the-frog first)
- `schedule.days`: day-by-day plan with estimated minutes
- `schedule.weeks`: weekly total minutes
- `overload`: top overload windows in the next 7–15 days + suggestions
- `errors`: tasks that failed JSON parsing

## Evaluate

Provide a small ground-truth file with expected subtasks, ranked order, and overloaded weeks (optional):

```json
{
  "expectedSubtasks": [{ "title": "Outline key points" }, { "title": "Draft slide content" }],
  "rankedSubtasks": [
    { "title": "Outline key points", "rank": 1 },
    { "title": "Draft slide content", "rank": 2 }
  ],
  "overloadWeeks": ["2026-W02"]
}
```

Run evaluation:

```bash
python -m src.eval.evaluate --ground-truth ground_truth.json --pred plan.json
```

## Visualize

Generate simple charts (weekly load and top priorities):

```bash
python -m src.eval.visualize --plan plan.json --outdir figs
```

Outputs `figs/weekly_load.png` and `figs/top_priorities.png`.

## Offline Use

- If you already have the model locally (e.g., as a Kaggle Dataset), set `MODEL_ID` to the local path and set offline mode:

```bash
export TRANSFORMERS_OFFLINE=1  # PowerShell: $env:TRANSFORMERS_OFFLINE=1
python -m src.cli --input example_tasks.json --output plan.json --model-id /kaggle/input/your-model-folder
```

The loader respects `TRANSFORMERS_OFFLINE` and will not hit the network.

## Fine-tune with LoRA

Prepare training data as JSON/JSONL with records that include task context and the expected JSON output (subtasks). Example JSONL record:

```json
{
  "task": "Prepare project presentation",
  "deadline": "2026-01-12",
  "description": "Q1 status",
  "expectedJson": { "subtasks": [{ "title": "Outline key points", "estimatedMinutes": 60, "difficulty": 3 }] }
}
```

Install extra deps:

```bash
pip install peft bitsandbytes datasets
```

Run LoRA fine-tuning (saves small adapters suitable for offline use):

```bash
python -m src.train.lora_finetune \
  --base-model mistralai/Mistral-7B-Instruct-v0.2 \
  --train-data data/train.jsonl \
  --output-dir lora_adapters \
  --epochs 1 --batch-size 1
```

To use adapters later, load the base model and then apply the adapters with PEFT, or merge during export if needed.
