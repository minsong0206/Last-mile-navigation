# OmniVLA-Edge FrodoBot Mini 배포 파이프라인

FrodoBots rides_11 데이터로 파인튜닝한 OmniVLA-Edge-Odom 체크포인트를 실제 FrodoBot Mini
로봇에 배포하기 위한 실시간 추론 파이프라인입니다. 카메라 + OSM 지도(OSRM 실시간 라우팅)를
입력으로 받아 웨이포인트를 예측하고, FrodoBot SDK로 제어 명령을 전송합니다.

## 구성 파일

| 파일 | 역할 |
|---|---|
| `build_live_map.py` | 출발/목적지 GPS로 OSRM 경로를 1회 계산·캐싱하고, 현재 위치 기준으로 ego-centric 지도 이미지를 실시간 생성 |
| `omnivla_edge_deploy.py` | FrodoBot SDK(REST API) 폴링 → 카메라 6프레임 컨텍스트 구성 → 지도 생성 → 모델 추론 → 제어 명령 전송, 전체 루프 |
| `debug_web.py` | 배포 중 카메라/지도/GPS/예측/에러를 실시간으로 보여주는 모니터링 웹 대시보드 (외부 의존성 없음, 표준 라이브러리만 사용) |

두 파일 모두 `osm_pipeline/py/osm_map_generator.py`와 `third_party/omnivla/inference/model_omnivla_edge_odom.py`를 import하므로, 이 두 경로도 함께 있어야 합니다.

## 요구 사항

- **GPU**: 로컬(로봇과 같은 네트워크)에서 실시간(3Hz) 추론이 가능한 CUDA GPU. 최신 GPU(RTX 50xx 등)는 PyTorch/CUDA 버전 호환 확인 필요.
- **Docker**: OSRM 라우팅 서버 실행용
- **FrodoBot Mini SDK**: `earth-rovers-sdk` 서버가 로컬(`127.0.0.1:8000`)에서 실행 중이어야 함 (`/control`, `/v2/front`, `/data` 엔드포인트 제공)
- Python 3.10+, PyTorch, torchvision, requests, pillow, opencv-python, numpy

## 1. 설치

```bash
git clone <이 저장소>
cd <이 저장소>

python3 -m venv venv && source venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124  # GPU/CUDA 버전에 맞게 조정
pip install requests pillow opencv-python numpy
```

## 2. 체크포인트 준비

파인튜닝된 체크포인트(`best.pth`)가 필요합니다. 용량이 커서(약 434MB) git에는 포함하지 않고 Hugging Face에 올려뒀습니다.

```bash
hf auth login   # 최초 1회 (private repo이므로 로그인 필요)

# 12m 체크포인트 (지금까지 가장 좋은 성능 — 아래 표 참고, 우선 추천)
mkdir -p checkpoints/omnivla_edge_rides11_odom_12m
hf download minsonganingee/omnivla-edge-rides11-odom-12m best.pth \
    --local-dir checkpoints/omnivla_edge_rides11_odom_12m/

# 20m 체크포인트
mkdir -p checkpoints/omnivla_edge_rides11_odom_20m
hf download minsonganingee/omnivla-edge-rides11-odom-20m best.pth \
    --local-dir checkpoints/omnivla_edge_rides11_odom_20m/

# 25m 체크포인트 (baseline)
mkdir -p checkpoints/omnivla_edge_rides11_odom
hf download minsonganingee/omnivla-edge-rides11-odom-25m best.pth \
    --local-dir checkpoints/omnivla_edge_rides11_odom/
```

**체크포인트별로 `--map_range` 값이 다릅니다 — 반드시 맞는 값과 함께 사용하세요.**
`--map_range`는 중심(로봇)에서 가장자리까지의 반경(half-width)이므로, 로봇이 실제로 보는 지도 전체 범위는 이 값의 2배입니다.

| 체크포인트 | HF repo | `--map_range` | 실제 지도 범위 | val_loss (best epoch) | ADE |
|---|---|---|---|---|---|
| `omnivla_edge_rides11_odom/best.pth` (baseline) | `omnivla-edge-rides11-odom-25m` | 25 | 50m × 50m | - | - |
| `omnivla_edge_rides11_odom_12m/best.pth` | `omnivla-edge-rides11-odom-12m` | 12 | 24m × 24m | 0.8143 (epoch5) | **0.234m** ← 지금까지 가장 좋음 |
| `omnivla_edge_rides11_odom_20m/best.pth` | `omnivla-edge-rides11-odom-20m` | 20 | 40m × 40m | 0.8815 (epoch1) | 0.250m |

## 3. OSRM 라우팅 서버 준비 (배포 지역: 서울)

경로 계산용 로컬 OSRM 서버가 필요합니다. 전처리는 시간이 걸리므로(수십 분), 가능하면 더 강력한 머신에서 미리 만들어서 파일만 옮기는 걸 권장합니다.

### 옵션 A — 이미 전처리된 파일 받기 (권장)

```bash
hf auth login   # 최초 1회 (private repo이므로 로그인 필요)
mkdir -p osm_pipeline/osrm/seoul
hf download minsonganingee/seoul-osrm-foot seoul_osrm_files.tar.gz \
    --repo-type dataset --local-dir .
tar -xzf seoul_osrm_files.tar.gz -C osm_pipeline/osrm/seoul/
```

### 옵션 B — 직접 전처리 (Docker 필요, 서울 기준 약 285MB pbf, 몇 분~수십 분)

```bash
mkdir -p osm_pipeline/osrm
wget -O osm_pipeline/osrm/south-korea-latest.osm.pbf \
  https://download.geofabrik.de/asia/south-korea-latest.osm.pbf

bash osm_pipeline/scripts/preprocess_osrm.sh seoul
```

### 서버 실행 (매번, 배포 시작 전)

```bash
bash osm_pipeline/scripts/start_osrm.sh seoul     # localhost:5011 에서 서빙 시작

# 확인
curl "http://localhost:5011/route/v1/foot/127.0778,37.6317;127.0800,37.6330?overview=false"
# {"code":"Ok", "routes":[{"distance":...}]} 형태 응답이 오면 정상
```

다른 지역에서 배포하려면 `osm_pipeline/py/osm_map_generator.py::osrm_port()`에 위경도 범위와 포트를 새로 추가하고, `preprocess_osrm.sh`/`start_osrm.sh`에도 같은 이름으로 항목을 추가하세요.

## 4. FrodoBot SDK 서버 실행

```bash
conda activate frodobot   # 또는 해당 환경
cd ~/earth-rovers-sdk
hypercorn main:app --bind 127.0.0.1:8000
```

브라우저로 `http://127.0.0.1:8000/sdk` 접속해서 카메라 영상이 뜨는지 먼저 확인하세요.

## 5. 배포 실행

```bash
python3 deployment/omnivla_edge_deploy.py \
    --ckpt checkpoints/omnivla_edge_rides11_odom_12m/best.pth \
    --map_range 12 \
    --goal_lat 37.6330 --goal_lon 127.0800
```

- 시작하면 첫 프레임에서 출발 위치 기준으로 OSRM 경로를 1회 계산(`set_goal`)하고, 이후 매 루프(3Hz)는 캐싱된 경로만 사용합니다 (매번 OSRM에 재쿼리하지 않음).
- 로봇이 경로에서 15m 이상 벗어나면 자동으로 재라우팅합니다.
- `Ctrl+C`로 정지하면 `linear=0, angular=0` 정지 명령을 자동 전송합니다.

### 실시간 모니터링 대시보드

FrodoBot SDK의 `/sdk` 페이지는 카메라 영상만 보여주고, 우리 파이프라인이 만드는 지도/GPS/예측/에러는
안 보여주므로 별도 대시보드(`deployment/debug_web.py`)를 자동으로 같이 띄웁니다 — 배포 스크립트
실행하면 기본적으로 **8080 포트**에서 뜹니다:

```
http://127.0.0.1:8080          # 같은 머신(노트북)에서
http://<노트북 IP>:8080        # 같은 네트워크의 다른 기기(폰 등)에서
```

- 현재 카메라 프레임, 실시간 생성 중인 지도(경로선 포함) 이미지가 1초마다 갱신됩니다
- **예측 궤적 확대 패널(`/traj.jpg`)**: 지도 축척(예: 20m 체크포인트=전체 40m)이 예측 horizon(8-step≈2.4초,
  실거리 약 2m)보다 훨씬 넓어서 지도 위 예측 궤적이 전체 폭의 5%(≈11px)밖에 안 되어 육안으로 방향 판단이
  거의 불가능함 — 그래서 지도와 별개로 예측 궤적만 좁은 고정 범위(±1.2m 좌우, 0~2.2m 전방, 0.5m 격자)로
  확대해서 보여주는 패널을 추가함. 이걸로 지금 우회전/좌회전/직진 중 뭘 예측하는지 바로 확인 가능.
- GPS(fix 없으면 빨간색 경고), heading(원본 orientation 값도 같이 표시), 현재 제어 명령(linear/angular), 실제 루프 주기(Hz), 누적 에러 수 표시
- 최근 로그 30줄(경로 재계산, GPS fix 없음, 에러 등)

### ⚠ 알려진 두 가지 실배포 이슈와 대응 (2026-08 실기기 테스트에서 발견)

1. **제어 명령이 항상 작게(안전 캡에 붙어서) 나오는 문제** — `waypoint_to_control()`이 목표 waypoint의
   실제 시점(target_step=2 → 0.9초 뒤)이 아니라 제어 루프 주기(DT=0.333초)로 나누고 있어서 필요 속도가
   ~2.7배 부풀려져 항상 `MAX_V`/`MAX_W` 캡에 걸리던 버그를 수정함(시점을 정확히 맞춤). 그래도 여전히
   느리게 느껴지면 `MAX_V`, `MAX_W` 상수(기본 0.3) 자체를 올려서 재시도할 것 — 처음엔 안전을 위해
   일부러 보수적으로 잡아둔 값입니다. (참고로 이날 로봇이 실제로 안 움직였던 진짜 원인은 별개로,
   `send_control()`이 `data=`로 보내서 명령 자체가 깨지던 버그였음 — `json=`으로 수정됨.)
2. **대시보드 지도가 실제 heading과 안 맞게 회전하는 문제** — `poll_frodobot()`의 컴퍼스 기반
   `orientation` 변환에 90도 보정이 빠져 있을 수 있다는 가설이 있었지만, 추측으로 고치는 대신
   **`step()`에 IMU 컴퍼스 heading vs GPS 궤적 기반 heading(`estimate_heading_from_track` — 학습
   데이터의 heading을 만든 것과 동일한 방식) 비교 로그를 추가**해서 실측으로 진단하는 쪽을 택함.
   콘솔에 매 tick `[heading] IMU컴퍼스=... GPS궤적=... 차이=...`가 출력되고, `deployment/logs/`의
   JSONL에도 `heading_diff_deg`로 기록됨. **이 차이가 계속 크게 벌어지면 컴퍼스 heading 자체가
   학습 때 heading과 안 맞는다는 뜻**이므로, 그땐 `render_frame()`에 GPS 궤적 기반 heading을
   대신 넣는 방향으로 고쳐야 함 (아직 미적용 — 로그 데이터로 먼저 확인 필요).
- 포트를 바꾸거나 끄고 싶으면 `--debug_port 8081` 또는 `--debug_port 0`

⚠ 추론/네트워크 등 어떤 에러가 나든 로봇을 먼저 정지시키고 대시보드에 에러를 기록한 뒤 스크립트가
중단되도록 되어 있습니다 (에러를 무시하고 계속 움직이지 않음).

## ⚠ 배포 전 안전 체크리스트

1. **저속 테스트부터**: `omnivla_edge_deploy.py`의 `MAX_V`, `MAX_W` 값을 기본(0.3)보다 낮춰서(예: 0.1) 첫 실행
2. **개활지**에서, 사람이 즉시 개입 가능한 상태로 시작
3. **GPS fix 확인**: `/data` 응답의 `latitude`/`longitude`가 `1000`(fix 없음 sentinel)이 아닌지 첫 실행 로그에서 확인 — 실내/음영지역에서는 지도 자체가 만들어지지 않습니다
4. `--ckpt`와 `--map_range`가 서로 맞는 조합인지 재확인 (표 참고)
5. e-stop 또는 SDK 긴급정지 방법을 미리 숙지

## 알려진 이슈 / 트러블슈팅

| 증상 | 원인 | 해결 |
|---|---|---|
| `RuntimeError: Attempting to deserialize object on CUDA device 1 but torch.cuda.device_count() is 1` | `CUDA_VISIBLE_DEVICES`로 GPU를 제한했는데 코드에 `cuda:1`이 하드코딩된 경우 충돌 | `CUDA_VISIBLE_DEVICES` 환경변수 없이 실행하거나, 코드의 device 지정을 `cuda:0`으로 통일 |
| 지도 위 예측 궤적이 이상하게 작게/크게 겹쳐 보임 | `--map_range`가 체크포인트 학습 시 값과 다름 | 위 표에서 맞는 값 확인 |
| OSRM 쿼리 실패 로그 후 직선 경로로 폴백 | OSRM 서버가 안 켜져 있거나 포트가 안 맞음 | `start_osrm.sh seoul` 실행 확인, `osrm_port()`의 위경도 범위 확인 |
| RTX 50xx 등 최신 GPU에서 `torch.cuda.is_available()`은 True인데 연산 시 에러 | PyTorch 빌드가 해당 GPU 아키텍처(sm_120 등) 미지원 | 최신 PyTorch nightly 또는 맞는 CUDA 버전 빌드 설치 필요 |

## 학습-추론 일치 확인 항목 (참고)

`omnivla_edge_deploy.py`는 `scripts/omnivla/finetune_omnivla_edge.py::prepare_batch()`와 다음 항목이 반드시 일치해야 합니다:
- `MAP_RANGE_M` (위 표)
- 컨텍스트: 과거 5프레임 + 현재 1프레임, 0.3초 간격
- `modality_id`/`goal_mask` = 0 ("map only")
- `METRIC_WAYPOINT_SPACING` = 0.125 (원본 OmniVLA-Edge 학습 코드 `train_omnivla.py`의 `0.25*0.5`와 동일 — 저자 추론 데모 스크립트의 0.1과는 다름, 0.125가 맞는 값)
