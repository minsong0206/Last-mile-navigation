# Project Scripts

This directory contains project-specific utilities that were added around the
FrodoBots OSM and OmniVLA fine-tuning workflow.

## Layout

```text
scripts/
├── analysis/   # dataset analysis, map ablations, attention checks, visualizations
├── data/       # frame extraction, GNM/odom-map preparation, MBRA reannotation
└── omnivla/    # OmniVLA checkpoint conversion and rides_11 fine-tuning
```

Run scripts from the repository root so config paths and generated output
directories stay consistent.

Examples:

```bash
conda run -n mbra python scripts/omnivla/finetune_omnivla_edge.py \
  --config config/rides11_finetune_odom.yaml

python3 scripts/analysis/analyze_dataset_distribution.py

conda run -n mbra python scripts/data/extract_frames.py --resume
```

Large generated outputs such as checkpoints, extracted frames, W&B runs, and OSM
map images should remain local and are excluded by `.gitignore`.
