# OmniVLA-Edge Fine-tuning Pipeline

FrodoBots rides_11 데이터셋을 이용해 OmniVLA-Edge 모델을 파인튜닝하는 전체 파이프라인 문서.

---

## 전체 파이프라인 개요

```
[FrodoBots-2K Arrow 데이터]
        │
        ▼
Step 1. Episode Selection          episode_selector.py
        (11,365 segments → 538 selected)
        │
        ▼
Step 2. OSM Map Generation         osm_map_generator.py
        (선별된 세그먼트별 224×224 ego-centric PNG)
        │
        ▼
Step 3. Dataset 구성               rides11_dataset.py
        (Arrow + OSM 맵 + 비디오 프레임 → PyTorch Dataset)
        │
        ▼
Step 4. Fine-tuning                finetune_omnivla_edge.py
        (OmniVLA-Edge, 20 epochs, partial freeze)
        │
        ▼
Step 5. 결과 시각화                vis_seg02_epochs.py
        (ep0405_seg02, epoch 5/10/15/20 궤적 오버레이)
```

---

## 경로 구조

```
/media/ms/WD_BLACK_4TB/
├── OmniVLA/
│   ├── OmniVLA/inference/
│   │   └── model_omnivla_edge.py          # OmniVLA_edge 모델 클래스
│   └── omnivla-edge/
│       └── omnivla-edge.pth               # 저자 사전학습 체크포인트
│
└── Learning-to-Drive-Anywhere-with-MBRA/
    ├── FrodoBots-2K/processed/output_rides_11/
    │   ├── train/data-00000-of-00001.arrow # Arrow IPC (GPS/IMU/비디오)
    │   └── frames/episode_{ep:04d}/        # 추출된 JPEG 프레임
    │       └── {fi:06d}.jpg
    │
    ├── osm_pipeline/
    │   ├── py/
    │   │   ├── episode_selector.py         # Step 1: 세그먼트 선별
    │   │   ├── osm_map_generator.py        # Step 2: OSM 맵 생성
    │   │   └── rides11_dataset.py          # Step 3: PyTorch Dataset
    │   └── osm_data/output_rides_11/
    │       ├── episode_scores.json         # 선별 결과 (11,365 → 538)
    │       └── osm_maps_arrow/
    │           └── episode_{ep:04d}_seg{seg:02d}/
    │               ├── osm_map_{i:06d}.png # 224×224 ego-centric OSM 맵
    │               ├── gps.csv
    │               └── bev_overview.png
    │
    ├── config/
    │   └── rides11_finetune.yaml           # 학습 하이퍼파라미터
    ├── scripts/omnivla/finetune_omnivla_edge.py            # Step 4: 학습 스크립트
    ├── scripts/analysis/vis_seg02_epochs.py                 # Step 5: 결과 시각화
    │
    └── checkpoints/omnivla_edge_rides11/
        ├── epoch_005.pth
        ├── epoch_010.pth
        ├── epoch_015.pth
        ├── epoch_020.pth
        ├── best.pth
        └── vis/
            ├── epoch_001.png ~ epoch_020.png  # 학습 중 validation 시각화
            └── ep0405_seg02/
                └── frame_{fi:06d}.png          # 결과 시각화 (693장)
```

---

## Step 1. Episode Selection

**스크립트:** `osm_pipeline/py/episode_selector.py`

**목적:** Arrow 데이터의 전체 에피소드를 정지 구간 기준으로 세그먼트 단위로 분할하고, OSRM 보행자 경로와의 궤적 정합성을 기준으로 양질의 세그먼트를 선별한다.

**선별 기준:**
| 지표 | 의미 | 기준 |
|---|---|---|
| `max_snap_m` | GPS→OSRM 도로 최대 오차 | ≤ 10m |
| `length_ratio` | EKF 이동거리 / OSRM 경로 길이 | 0.5 ~ 2.0 |
| `frechet_norm` | 정규화 Fréchet 거리 | ≤ 0.5 |
| `chamfer_norm` | 정규화 Chamfer 거리 | ≤ 0.3 |
| `heading_err_deg` | 진행 방향 오차 | ≤ 30° |

**결과:**
- 입력: 11,365 세그먼트
- 출력: `episode_scores.json` (selected=True 538개)

**실행:**
```bash
cd /media/ms/WD_BLACK_4TB/Learning-to-Drive-Anywhere-with-MBRA
conda run -n mbra python osm_pipeline/py/episode_selector.py
```

---

## Step 2. OSM Map Generation

**스크립트:** `osm_pipeline/py/osm_map_generator.py`

**목적:** 선별된 각 세그먼트의 모든 프레임에 대해 ego-centric, heading-up 224×224 OSM 맵 PNG를 생성한다.

**핵심 설정:**
```python
MAP_SIZE_PX  = 224       # 출력 이미지 크기
MAP_RANGE_M  = 25.0      # 맵 반폭 (m) — 전체 50m × 50m
ZOOM         = 18        # OSM 타일 줌 레벨
PX_PER_M     = 224 / 50  # = 4.48 px/m
```

**맵 구성:**
- 배경: OSM 타일 스티칭 (zoom=18)
- 회색 선: 과거 경로 (GPS 기반)
- 빨간 선: 미래 경로 (GPS 기반)
- 초록 점: Ego 위치
- 회전: `rot_deg = 90 - heading_deg` → 진행 방향이 항상 위를 향함

**좌표 변환 (ego-frame → 픽셀):**
```python
# 이미지 중심 = ego 위치
cx, cy = 112, 112
pixel_x = cx - y_ego_m * PX_PER_M   # left → image left
pixel_y = cy - x_ego_m * PX_PER_M   # forward → image up
```

**OSRM 서버 필요 (Docker):**
| 도시 | 포트 |
|---|---|
| Perth | 5001 |
| Taipei | 5002 |
| Tokyo | 5003 |
| Wuhan | 5004 |
| Manila | 5005 |
| Rome | 5006 |
| Wellington | 5007 |
| Florida | 5008 |
| Brighton | 5009 |
| Madrid | 5010 |

**실행:**
```bash
# 선별된 세그먼트만 생성
conda run -n mbra python osm_pipeline/py/osm_map_generator.py

# 특정 에피소드/세그먼트
conda run -n mbra python osm_pipeline/py/osm_map_generator.py --ep 405 --seg 2

# 전체
conda run -n mbra python osm_pipeline/py/osm_map_generator.py --all_episodes
```

---

## Step 3. Dataset 구성

**스크립트:** `osm_pipeline/py/rides11_dataset.py`

**클래스:** `Rides11Dataset(Dataset)`

**1 샘플 구성:**
```
ep, seg, fi (현재 프레임 인덱스)
  │
  ├── obs_stack    (18, 96, 96)  — 6프레임 RGB 스택
  │     [fi-15, fi-12, fi-9, fi-6, fi-3, fi] 각 3채널
  │
  ├── map_images   (9, 96, 96)   — OSM 맵 × 2 + obs_cur × 1
  │     [osm_map(3) + osm_map(3) + obs_cur(3)]
  │
  └── gt_waypoints (8, 2)        — ego-frame waypoints / 0.125
        fi+3, fi+6, ..., fi+24 의 미래 위치
```

**Valid frame 조건:**
```python
PAST_MARGIN   = 15  # CTX_STRIDE(3) × N_CTX(5)
FUTURE_MARGIN = 24  # WAYPOINT_STRIDE(3) × N_WAYPOINTS(8)

fi >= seg_fi_start + 15   # 과거 5프레임 확보
fi <= seg_fi_end   - 24   # 미래 8 waypoint 확보
```

**Waypoint 좌표계:**
```python
# 전역 (E, N) → ego frame (forward, left)
x_ego =  dx * cos_h + dy * sin_h   # forward
y_ego = -dx * sin_h + dy * cos_h   # left

# 정규화
METRIC_WAYPOINT_SPACING = 0.125  # m
wp_normalized = [x_ego / 0.125, y_ego / 0.125]
```

**이미지 로드 우선순위:**
1. `frames/episode_{ep:04d}/{fi:06d}.jpg` (추출된 JPEG, 빠름)
2. 없으면 mp4 직접 디코딩 (PyAV, fallback)

---

## Step 4. Fine-tuning

**스크립트:** `scripts/omnivla/finetune_omnivla_edge.py`  
**설정 파일:** `config/rides11_finetune.yaml`

### 모델 입력/출력

**입력 (OmniVLA_edge.forward):**
| 텐서 | Shape | 설명 |
|---|---|---|
| `obs_img` | (B, 18, 96, 96) | 과거 5 + 현재 1 프레임 스택 |
| `goal_pose` | (B, 4) | [x, y, cos, sin] — rides_11에서는 zeros |
| `map_images` | (B, 9, 96, 96) | OSM 현재(3) + OSM 현재(3) + obs_cur(3) |
| `goal_img` | (B, 3, 96, 96) | goal 이미지 — obs_cur 반복 |
| `goal_mask` | (B,) | modality_id=0 ("satellite only") |
| `feat_text` | (B, 512) | CLIP 텍스트 피처 — zeros |
| `current_img` | (B, 3, 224, 224) | FiLM 컨디셔닝용 고해상도 이미지 |

**출력:**
| 텐서 | Shape | 설명 |
|---|---|---|
| `action_pred` | (B, 8, 4) | 8-step waypoints [x, y, cos_h, sin_h] |
| `dist_pred` | (B, 1) | 거리 예측 (학습에 미사용) |

### Loss 함수

```python
# Waypoint loss (smooth L1)
L_wp = smooth_l1_loss(pred_xy, gt_waypoints)

# Smoothness penalty (가속도 억제)
velocity = pred_xy[:, 1:] - pred_xy[:, :-1]   # (B, 7, 2)
accel    = velocity[:, 1:] - velocity[:, :-1]  # (B, 6, 2)
L_smooth = accel.pow(2).mean()

L_total = L_wp + 0.1 * L_smooth
```

### 평가 지표

```python
METRIC_WAYPOINT_SPACING = 0.125  # m

# 실제 미터로 환산
pred_m = pred_xy * 0.125
gt_m   = gt_xy   * 0.125

ADE = mean(||pred_m - gt_m||)        # 8 waypoint 평균 L2
FDE = mean(||pred_m[:,-1] - gt_m[:,-1]||)  # 마지막 waypoint L2
```

### Freeze 전략

| 전략 | 학습 파라미터 | 용도 |
|---|---|---|
| `map_encoder_only` | goal_encoder + compress_obs_enc_map | Stage 1 (빠른 초기 적응) |
| `partial` | transformer + action_predictor | Stage 2 (권장, 현재 사용) |
| `none` | 전체 | Stage 3 (과적합 위험) |

### 하이퍼파라미터 (rides11_finetune.yaml)

```yaml
epochs:        20
batch_size:    16
lr:            1.0e-4
weight_decay:  1.0e-4
val_ratio:     0.1
smooth_weight: 0.1
save_freq:     5       # epoch_005/010/015/020.pth
freeze:        partial
```

### 실행

```bash
cd /media/ms/WD_BLACK_4TB/Learning-to-Drive-Anywhere-with-MBRA
conda run -n mbra python scripts/omnivla/finetune_omnivla_edge.py \
    --config config/rides11_finetune.yaml
```

### 저장 체크포인트

```
checkpoints/omnivla_edge_rides11/
├── epoch_005.pth   # {epoch, model_state_dict, optimizer_state_dict, ...}
├── epoch_010.pth
├── epoch_015.pth
├── epoch_020.pth
└── best.pth        # model_state_dict only (val_loss 기준)
```

---

## Step 5. 결과 시각화

**스크립트:** `scripts/analysis/vis_seg02_epochs.py`

**대상:** `ep0405_seg02` (valid frames: 693장, fi=67~759)

**출력:** `checkpoints/omnivla_edge_rides11/vis/ep0405_seg02/frame_{fi:06d}.png`

### 그래프 레이아웃 (figsize=28×10)

| 패널 | 내용 | width_ratio |
|---|---|---|
| Col 1 | 카메라 이미지 (현재 프레임) | 1 |
| Col 2 | Trajectory plot (GT + Epoch 5/10/15/20) | 3 |
| Col 3 | OSM 맵 오버레이 (224×224 원본) | 1 |

### Trajectory Plot 설계

- **X축:** Left / Right (m), 좌우 대칭
- **Y축:** Forward (m)
- **색상:** GT=녹색점선, Ep5=하늘, Ep10=주황, Ep15=빨강, Ep20=보라
- **스케일:** 전체 693장의 전역 min/max 기준 고정 (프레임별 크기 일관성 보장)
- **aspect:** `equal` (1m = 1m, 물리적 비율 정확)

```
Global limits (ep0405_seg02):
  x: ±0.95 m
  y: -0.73 ~ 3.18 m
```

### OSM 맵 오버레이 좌표 변환

```python
PX_PER_M = 224 / (25 * 2)  # = 4.48 px/m
cx, cy   = 112, 112          # 이미지 중심 = ego 위치

pixel_x = cx - wp_m[:, 1] * PX_PER_M   # left → 왼쪽
pixel_y = cy - wp_m[:, 0] * PX_PER_M   # forward → 위쪽
```

### 실행

```bash
# 테스트 (1장, vis/ 루트에 저장)
conda run -n mbra python scripts/analysis/vis_seg02_epochs.py --test

# 전체 693장 생성
conda run -n mbra python scripts/analysis/vis_seg02_epochs.py --overwrite

# 옵션
#   --device cuda:0   GPU 지정
#   --batch 16        추론 배치 크기
#   --overwrite       기존 파일 덮어쓰기
#   --test            첫 프레임 1장만 → vis/test_*.png
```

### 실행 순서 (내부)

```
1단계: 데이터 사전 수집
  - 전체 valid_frames 순회
  - gt_wp_m, obs_stack_t, map_images_t 캐싱

2단계: 모델 추론 (checkpoint별)
  - epoch_005 → preds_all["Epoch 5"]  (693 × (8,2))
  - epoch_010 → preds_all["Epoch 10"]
  - epoch_015 → preds_all["Epoch 15"]
  - epoch_020 → preds_all["Epoch 20"]

3단계: 전역 trajectory limits 계산
  - 전체 GT + 4 epoch 예측의 min/max → fixed_xlim, fixed_ylim

4단계: 시각화 & 저장
  - 각 프레임: raw_img + raw_osm 로드 → make_figure() → PNG 저장
```

---

## 환경

```bash
conda activate mbra
# 주요 패키지: torch, torchvision, clip, pyarrow, av, matplotlib, tqdm
```

**CLIP 모델:** `ViT-B/32` (자동 다운로드, `~/.cache/clip/`)

---

## 주요 상수 요약

| 상수 | 값 | 의미 |
|---|---|---|
| `METRIC_WAYPOINT_SPACING` | 0.125 m | waypoint 정규화 단위 |
| `N_WAYPOINTS` | 8 | 예측 waypoint 수 |
| `WAYPOINT_STRIDE` | 3 frames | waypoint 간격 |
| `N_CTX` | 5 | context 프레임 수 |
| `CTX_STRIDE` | 3 frames | context 간격 |
| `MAP_RANGE_M` | 25.0 m | OSM 맵 반폭 |
| `MAP_SIZE_PX` | 224 | OSM 맵 픽셀 크기 |
| `PX_PER_M` | 4.48 | 픽셀/미터 (224÷50) |
| `MODALITY_ID` | 0 | "satellite only" (GPS/언어 없음) |
