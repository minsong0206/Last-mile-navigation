# Project Execution Scripts

This directory contains the project-specific utilities added for the FrodoBots
OSM and OmniVLA fine-tuning workflow. Run commands from the repository root
unless noted otherwise.

Large outputs such as checkpoints, extracted frames, W&B runs, and generated OSM
map images are intentionally ignored by Git.

## Layout

```text
scripts/
├── analysis/   # dataset analysis, map ablations, attention checks, visualizations
├── data/       # frame extraction, GNM/odom-map preparation, MBRA reannotation
└── omnivla/    # OmniVLA checkpoint conversion and rides_11 fine-tuning
```

## Typical Workflow

1. Prepare OSRM servers and OSM maps with `osm_pipeline/`.
2. Extract camera frames if a visualization script needs frame JPEGs.
3. Convert the OmniVLA checkpoint when using the 3-channel OSM-map model.
4. Fine-tune OmniVLA on rides_11.
5. Run analysis/visualization scripts against the saved checkpoint.

## OmniVLA Fine-Tuning

| Script | Purpose | Main output |
| --- | --- | --- |
| `scripts/omnivla/convert_goal_encoder_9ch_to_3ch.py` | Convert the original OmniVLA-Edge 9-channel map encoder checkpoint to the 3-channel OSM-map variant. | `omnivla-edge-odom3ch.pth` |
| `scripts/omnivla/finetune_omnivla_edge.py` | Fine-tune OmniVLA-Edge/Odom on FrodoBots rides_11 with OSM map inputs. | `checkpoints/omnivla_edge_rides11_odom/` |

Convert checkpoint:

```bash
conda run -n mbra python scripts/omnivla/convert_goal_encoder_9ch_to_3ch.py \
  --input /path/to/omnivla-edge.pth \
  --output /path/to/omnivla-edge-odom3ch.pth
```

Train:

```bash
conda run -n mbra python scripts/omnivla/finetune_omnivla_edge.py \
  --config config/rides11_finetune_odom.yaml
```

Resume:

```bash
conda run -n mbra python scripts/omnivla/finetune_omnivla_edge.py \
  --config config/rides11_finetune_odom.yaml \
  --resume_ckpt checkpoints/omnivla_edge_rides11_odom/best.pth \
  --start_epoch 4 \
  --resume_val_loss 0.2624
```

Evaluate only:

```bash
conda run -n mbra python scripts/omnivla/finetune_omnivla_edge.py \
  --config config/rides11_finetune_odom.yaml \
  --eval_only \
  --eval_ckpt checkpoints/omnivla_edge_rides11_odom/best.pth
```

## Data Preparation

| Script | Purpose | Main output |
| --- | --- | --- |
| `scripts/data/extract_frames.py` | Extract frame JPEGs from rides_11 MP4 files using Arrow timestamps. | `FrodoBots-2K/processed/output_rides_11/frames/` |
| `scripts/data/make_odom_maps.py` | Build odom map images from existing `traj_data.pkl` in GNM-format rides. | `FrodoBots-2K/processed_gnm/` |
| `scripts/data/reannotate_gnm.py` | Run MBRA reannotation on GNM-format rides and generate odom maps. | `FrodoBots-2K/processed_gnm/` |

Extract frames:

```bash
conda run -n mbra python scripts/data/extract_frames.py --resume --workers 8
```

Extract a small episode range:

```bash
conda run -n mbra python scripts/data/extract_frames.py --ep 0 5
```

Build odom maps from existing trajectories:

```bash
conda run -n mbra python scripts/data/make_odom_maps.py
```

Run MBRA reannotation:

```bash
conda run -n mbra python scripts/data/reannotate_gnm.py \
  --ckpt train/logs/frodobot-gnm/frodobot-gnm_2026_04_27_10_31_23/mbra.pth \
  --device cuda:0
```

## Analysis and Visualization

| Script | Purpose | Main output |
| --- | --- | --- |
| `scripts/analysis/analyze_dataset_distribution.py` | Classify selected rides_11 segments by curvature/scenario and produce plots. | `dataset_analysis/` |
| `scripts/analysis/check_map_ablation.py` | Quick held-out map ablation check for the fine-tuned Odom model. | `attention_analysis/ablation_result.png` |
| `scripts/analysis/check_map_attention.py` | Run ablation, Grad-CAM, and transformer attention checks. | `attention_analysis/` |
| `scripts/analysis/test_map_causality.py` | Test whether predictions change under held-out, map-swap, matched, and sequence settings. | `attention_analysis/` |
| `scripts/analysis/vis_seg02_epochs.py` | Visualize non-Odom checkpoint predictions across epochs for ep0405/seg02. | `checkpoints/omnivla_edge_rides11/vis/` |
| `scripts/analysis/vis_seg02_epochs_odom.py` | Visualize Odom checkpoint predictions across epochs for ep0405/seg02. | `checkpoints/omnivla_edge_rides11_odom/vis/` |

Dataset distribution:

```bash
python3 scripts/analysis/analyze_dataset_distribution.py
```

Attention and ablation suite:

```bash
conda run -n mbra python scripts/analysis/check_map_attention.py \
  --ckpt checkpoints/omnivla_edge_rides11_odom/best.pth \
  --method all \
  --n_samples 200
```

Map causality tests:

```bash
conda run -n mbra python scripts/analysis/test_map_causality.py \
  --ckpt checkpoints/omnivla_edge_rides11_odom/best.pth \
  --method all
```

Epoch visualization:

```bash
conda run -n mbra python scripts/analysis/vis_seg02_epochs_odom.py \
  --device cuda:1 \
  --batch 16 \
  --overwrite
```

Quick visualization smoke test:

```bash
conda run -n mbra python scripts/analysis/vis_seg02_epochs_odom.py --test
```

## Notes

- Most scripts assume rides_11 data exists under `FrodoBots-2K/processed/`.
- OSM map images are expected under
  `osm_pipeline/osm_data/output_rides_11/osm_maps_arrow/`.
- OmniVLA inference code is vendored under `third_party/omnivla/inference/`.
- Generated outputs are local artifacts and should not be added with `git add .`.
