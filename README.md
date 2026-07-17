# Last Mile Navigation

> **Fork / research extension**
>
> This repository is based on the original
> [NHirose/Learning-to-Drive-Anywhere-with-MBRA](https://github.com/NHirose/Learning-to-Drive-Anywhere-with-MBRA)
> codebase. It keeps the MBRA/LogoNav components and adds a FrodoBots
> OSM/OSRM preprocessing pipeline, vendored OmniVLA inference modules,
> OmniVLA-Edge-Odom fine-tuning scripts, and map analysis tools.

## 한국어 파이프라인 문서

현재 fork에서 추가한 OSM Docker, FrodoBots rides_11, OmniVLA-Edge-Odom
fine-tuning, attention/ablation 분석 흐름은 아래 문서를 기준으로 보면 됩니다.

- [README_KO.md](README_KO.md): 전체 한국어 파이프라인 설명과 실행 순서
- [osm_pipeline/README.md](osm_pipeline/README.md): OSRM Docker와 OSM map 생성
- [scripts/README.md](scripts/README.md): 실행 스크립트별 목적, 옵션, 출력 위치

## Quick Commands

```bash
# OSRM 서버 시작
cd osm_pipeline
bash scripts/start_osrm.sh rides11

# OSM segment selection + map generation
bash scripts/run_pipeline.sh

# OmniVLA-Edge-Odom fine-tuning
cd ..
conda run -n mbra python scripts/omnivla/finetune_omnivla_edge.py \
  --config config/rides11_finetune_odom.yaml

# Map attention / ablation analysis
conda run -n mbra python scripts/analysis/check_map_attention.py \
  --ckpt checkpoints/omnivla_edge_rides11_odom/best.pth \
  --method all
```

## Upstream

For the original MBRA paper, installation notes, and baseline training code,
refer to the upstream repository:

https://github.com/NHirose/Learning-to-Drive-Anywhere-with-MBRA
