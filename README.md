# VLM Pipeline — Manufacturing-Drawing Part Classification

Three-stage pipeline: ingest one single-part manufacturing PDF drawing and return
part class (sheet / tube), bend count (for sheets), and localized bounding boxes
for each view / title-block region.

Assigned task: AAI Labs, 2026.

## Pipeline stages

| Stage | Purpose | Status |
|---|---|---|
| 1. Localize | Grounding VLM finds flat_pattern / orthographic_view / isometric_view / section_view / title_block boxes | not started |
| 2. Classify | sheet vs. tube, using title block + view crops | **in progress** |
| 3. Count bends | count folds in flat pattern, corroborated by side/section/isometric views | not started |

## Repo structure

```
src/            pipeline code (one module per stage + a batch runner)
configs/        model configs, prompts versioned separately from code
prompts/        prompt templates for each stage
reference_samples/   the 4 ground-truth drawings (gitignored — keep PDFs local)
outputs/crops/  cropped region images (gitignored)
outputs/json/   per-drawing prediction JSON (gitignored)
docs/           model-choice rationale, inference-time & hardware analysis
```

## Setup

```bash
python -m venv venv
source venv/bin/activate        # or venv\Scripts\activate on Windows
pip install -r requirements.txt
cp .env.example .env            # then fill in your API key(s)
```

## Running Stage 2 (classification) on the reference set

```bash
python src/classify.py --input reference_samples/ --out outputs/json/
```

## Ground truth (reference set)

| Drawing | Class | Bends | Discriminating signal |
|---|---|---|---|
| 001 BRACKET | sheet | 2 | Abwicklung present; U-channel side profile = 2 folds |
| 002 PLATE | sheet | 0 | Flat plate; corner chamfers 2×45° are NOT bends |
| 003 SQUARE TUBE | tube | n/a | Hollow constant section + 2790mm length |
| 004 ADAPTER PLATE | sheet | 0 | Flat bar; stepped outline is a cut, NOT a fold |

## Model-choice rationale

See `docs/model_rationale.md` — benchmark of open-weight (Qwen2.5-VL) vs.
API (Claude / GPT-4o) across accuracy, latency, and cost per stage.

## Deterministic runs

All API calls use `temperature=0` and fixed seeds where supported, so results
are reproducible across runs.
