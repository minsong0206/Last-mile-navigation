# Last-mile Navigation OSM/OmniVLA Pipeline

> 이 저장소는 원본
> [NHirose/Learning-to-Drive-Anywhere-with-MBRA](https://github.com/NHirose/Learning-to-Drive-Anywhere-with-MBRA)
> 코드를 기반으로 한 연구용 fork입니다. 원본 MBRA/LogoNav 코드는 유지하면서,
> FrodoBots rides_11 데이터셋을 OSM 지도와 결합하고 OmniVLA-Edge-Odom을
> fine-tuning하기 위한 파이프라인, 실행 스크립트, 분석 도구를 추가했습니다.

## 개요

이 repo의 추가 파이프라인은 FrodoBots 실주행 데이터에서 OSM 보행자 경로와
잘 맞는 구간을 선별하고, 각 프레임마다 heading-up OSM 지도 이미지를 생성한 뒤,
카메라 관측과 OSM map을 함께 사용하는 OmniVLA-Edge-Odom 모델을 fine-tuning합니다.

주요 목표:

- FrodoBots rides_11 데이터셋에서 유효한 이동 세그먼트 선별
- 로컬 OSRM Docker 서버로 보행자 경로 기반 OSM routing 수행
- 프레임별 ego-centric OSM map 생성
- OmniVLA-Edge checkpoint를 3채널 OSM map 입력용으로 변환
- rides_11 데이터로 OmniVLA-Edge-Odom fine-tuning
- map ablation, attention, causality 분석으로 OSM map 사용 여부 검증

## 전체 파이프라인

```text
FrodoBots rides_11
  ├─ Arrow + mp4 + GPS/EKF
  │
  ├─ Step 1. OSRM Docker 준비
  │   ├─ .osm.pbf 다운로드
  │   ├─ osrm-extract -p foot.lua
  │   ├─ osrm-partition
  │   └─ osrm-customize
  │
  ├─ Step 2. Episode/segment selection
  │   ├─ 정지 구간 제거
  │   ├─ 이동 세그먼트 분리
  │   ├─ OSRM nearest로 sidewalk snap 검사
  │   └─ Frechet/Chamfer/heading/length ratio 필터링
  │
  ├─ Step 3. OSM map generation
  │   ├─ OSM tile 다운로드 및 캐시
  │   ├─ OSRM/실제 GPS 기반 경로 오버레이
  │   └─ 224x224 heading-up map PNG 생성
  │
  ├─ Step 4. OmniVLA checkpoint conversion
  │   └─ goal_encoder 9ch → 3ch OSM map 입력용 변환
  │
  ├─ Step 5. OmniVLA-Edge-Odom fine-tuning
  │   ├─ obs image context
  │   ├─ OSM map image
  │   └─ relative waypoint prediction
  │
  └─ Step 6. Analysis
      ├─ map ablation
      ├─ Grad-CAM / attention
      ├─ map swap / matched map test
      └─ epoch visualization
```

## 모델 입력/출력

현재 fine-tuning 스크립트는 `OmniVLA_edge_odom`을 사용합니다.

입력:

| 입력 | Shape | 설명 |
| --- | --- | --- |
| `obs_img` | `(B, 18, 96, 96)` | 과거 5프레임 + 현재 1프레임, RGB concat |
| `map_images` | `(B, 3, 96, 96)` | OSM map 1장, 3채널 |
| `goal_img` | `(B, 3, 96, 96)` | 현재 obs 이미지 |
| `goal_pose` | `(B, 4)` | rides_11에서는 zero 입력 |
| `goal_mask` | `(B,)` | modality id |
| `feat_text` | `(B, 512)` | rides_11에서는 zero 입력 |
| `current_img` | `(B, 3, 224, 224)` | FiLM conditioning용 현재 이미지 |

출력:

| 출력 | Shape | 설명 |
| --- | --- | --- |
| `action_pred` | `(B, 8, 4)` | 8-step waypoint `[x, y, cos_h, sin_h]` |
| `dist_pred` | `(B, 1)` | 거리 예측, 현재 학습 loss에는 사용하지 않음 |

주요 loss:

```text
L_total = smooth_l1(pred_xy, gt_waypoints) + 0.1 * acceleration_smoothness
```

## 디렉터리 구조

```text
.
├── README.md                         # 원본 MBRA README + fork 안내 링크
├── README_KO.md                      # 현재 fork 파이프라인 한국어 설명
├── config/
│   ├── rides11_finetune.yaml
│   └── rides11_finetune_odom.yaml
├── osm_pipeline/
│   ├── README.md                     # OSRM Docker / OSM map generation 설명
│   ├── scripts/
│   │   ├── preprocess_osrm.sh
│   │   ├── start_osrm.sh
│   │   └── run_pipeline.sh
│   └── py/
│       ├── episode_selector.py
│       ├── osm_map_generator.py
│       └── rides11_dataset.py
├── scripts/
│   ├── README.md                     # 실행 스크립트 전체 가이드
│   ├── omnivla/
│   │   ├── convert_goal_encoder_9ch_to_3ch.py
│   │   └── finetune_omnivla_edge.py
│   ├── data/
│   │   ├── extract_frames.py
│   │   ├── make_odom_maps.py
│   │   └── reannotate_gnm.py
│   └── analysis/
│       ├── analyze_dataset_distribution.py
│       ├── check_map_attention.py
│       ├── test_map_causality.py
│       ├── vis_seg02_epochs.py
│       └── vis_seg02_epochs_odom.py
├── third_party/
│   └── omnivla/inference/             # vendored OmniVLA inference modules
└── docs/
    ├── 0531.md
    └── pipeline_overview.md
```

## Git에 포함하지 않는 산출물

아래 항목은 용량이 크거나 재생성 가능한 로컬 산출물이므로 Git에 올리지 않습니다.

```text
FrodoBots-2K/
checkpoints/
wandb/
osm_pipeline/osrm/
osm_pipeline/tile_cache/
osm_pipeline/osm_data/output_*/osm_maps/
osm_pipeline/osm_data/output_*/osm_maps_arrow/
*.pth
*.pt
*.arrow
*.mp4
```

## 설치

원본 MBRA 환경을 기준으로 사용합니다.

```bash
conda env create -f train/environment_mbra.yml
conda activate mbra
pip install -e train/
```

추가로 OSM/분석 파이프라인에서 주로 사용하는 패키지:

```bash
pip install requests pillow pyarrow av opencv-python tqdm matplotlib wandb
```

OmniVLA inference 모듈은 `third_party/omnivla/inference/`에 포함되어 있으므로
별도 submodule은 필요하지 않습니다.

## 1. OSRM Docker 준비

OSM routing은 Docker의 `osrm/osrm-backend` 이미지를 사용합니다.

rides_11 지역별 포트:

| Region | Port | PBF |
| --- | ---: | --- |
| Wuhan | 5004 | `hubei-latest.osm.pbf` |
| Manila | 5005 | `philippines-latest.osm.pbf` |
| Rome | 5006 | `italy-latest.osm.pbf` |
| Wellington | 5007 | `new-zealand-latest.osm.pbf` |
| Florida | 5008 | `florida-latest.osm.pbf` |
| Brighton | 5009 | `great-britain-latest.osm.pbf` |
| Madrid | 5010 | `spain-latest.osm.pbf` |

상태 확인:

```bash
cd osm_pipeline
bash scripts/preprocess_osrm.sh status
```

전처리:

```bash
bash scripts/preprocess_osrm.sh rides11
```

서버 시작:

```bash
bash scripts/start_osrm.sh rides11
```

서버 상태 확인/종료:

```bash
bash scripts/start_osrm.sh status
bash scripts/start_osrm.sh stop
```

OSRM 데이터를 repo 밖에 둘 경우:

```bash
OSRM_DIR=/path/to/osrm_data bash osm_pipeline/scripts/start_osrm.sh rides11
```

자세한 내용은 [osm_pipeline/README.md](osm_pipeline/README.md)를 참고합니다.

## 2. OSM segment selection / map generation

전체 OSM pipeline:

```bash
cd osm_pipeline
bash scripts/run_pipeline.sh
```

이미 segment selection이 끝났고 특정 episode/segment만 다시 만들 때:

```bash
cd osm_pipeline
bash scripts/run_pipeline.sh --skip-selection --ep 405 --seg 2
```

주요 출력:

```text
osm_pipeline/osm_data/output_rides_11/episode_scores.json
osm_pipeline/osm_data/output_rides_11/osm_maps_arrow/
```

## 3. 프레임 추출

일부 visualization script는 MP4에서 추출된 frame JPEG를 사용합니다.

```bash
conda run -n mbra python scripts/data/extract_frames.py --resume --workers 8
```

일부 episode만 테스트:

```bash
conda run -n mbra python scripts/data/extract_frames.py --ep 0 5
```

출력:

```text
FrodoBots-2K/processed/output_rides_11/frames/
```

## 4. OmniVLA checkpoint 변환

원본 OmniVLA-Edge checkpoint의 `goal_encoder`는 9채널 map 입력을 기대합니다.
현재 OSM-Odom fine-tuning은 3채널 OSM map 1장을 사용하므로 checkpoint를 변환합니다.

```bash
conda run -n mbra python scripts/omnivla/convert_goal_encoder_9ch_to_3ch.py \
  --input /path/to/omnivla-edge.pth \
  --output /path/to/omnivla-edge-odom3ch.pth
```

## 5. OmniVLA-Edge-Odom fine-tuning

기본 학습:

```bash
conda run -n mbra python scripts/omnivla/finetune_omnivla_edge.py \
  --config config/rides11_finetune_odom.yaml
```

resume:

```bash
conda run -n mbra python scripts/omnivla/finetune_omnivla_edge.py \
  --config config/rides11_finetune_odom.yaml \
  --resume_ckpt checkpoints/omnivla_edge_rides11_odom/best.pth \
  --start_epoch 4 \
  --resume_val_loss 0.2624
```

평가만 실행:

```bash
conda run -n mbra python scripts/omnivla/finetune_omnivla_edge.py \
  --config config/rides11_finetune_odom.yaml \
  --eval_only \
  --eval_ckpt checkpoints/omnivla_edge_rides11_odom/best.pth
```

주요 출력:

```text
checkpoints/omnivla_edge_rides11_odom/
wandb/
```

## 6. 분석 / 검증

데이터셋 분포 분석:

```bash
python3 scripts/analysis/analyze_dataset_distribution.py
```

출력:

```text
dataset_analysis/
```

Map ablation / Grad-CAM / attention:

```bash
conda run -n mbra python scripts/analysis/check_map_attention.py \
  --ckpt checkpoints/omnivla_edge_rides11_odom/best.pth \
  --method all \
  --n_samples 200
```

Map causality test:

```bash
conda run -n mbra python scripts/analysis/test_map_causality.py \
  --ckpt checkpoints/omnivla_edge_rides11_odom/best.pth \
  --method all
```

Epoch별 trajectory visualization:

```bash
conda run -n mbra python scripts/analysis/vis_seg02_epochs_odom.py \
  --device cuda:1 \
  --batch 16 \
  --overwrite
```

출력:

```text
attention_analysis/
checkpoints/omnivla_edge_rides11_odom/vis/
```

## 참고 문서

- [osm_pipeline/README.md](osm_pipeline/README.md): OSRM Docker와 OSM map 생성
- [scripts/README.md](scripts/README.md): 실행 스크립트 상세 명령
- [docs/0531.md](docs/0531.md): 진행 로그
- [docs/pipeline_overview.md](docs/pipeline_overview.md): 기존 pipeline overview

## 원본 프로젝트

이 fork는 MBRA/LogoNav 원본 프로젝트를 기반으로 합니다. 원본 논문, 설치,
기본 학습/추론 설명은 [README.md](README.md)의 upstream 설명을 참고하세요.
