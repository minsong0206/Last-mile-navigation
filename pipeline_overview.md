# 학습 데이터 전처리 파이프라인 전체 정리

> **프로젝트**: Learning to Drive Anywhere with MBRA  
> **작성일**: 2026-05-17  
> **목적**: FrodoBots-2K 데이터셋 기반 자율주행 내비게이션 모델 학습 데이터 구축

---

## 1. 프로젝트 목표

- **최종 목표**: 다양한 실외 환경(도심, 공원, 보행로 등)에서 로봇이 OSM 지도를 참고하여 목적지까지 자율 주행할 수 있는 모델 학습
- **활용 모델**: NoMaD (Navigation with Map-less Diffusion) 및 유사 토폴로지 내비게이션 모델
- **핵심 아이디어**: GPS 궤적 + OSM 타일 지도 → ego-heading-up 로컬 맵 이미지를 모델 입력으로 사용

```
[카메라 이미지] + [OSM 로컬 맵 (224×224)] → 모델 → [이동 명령 (속도, 조향)]
```

---

## 2. 데이터셋: FrodoBots-2K 개요

### 2.1 원시 데이터 구조

```
FrodoBots-2K/extracted/
  output_rides_0/          # ride_00 원시 데이터 (Tokyo 등)
  output_rides_11/         # 신규 데이터 (10개 지역, 981개 라이드 원본)
  output_rides_3/          # Florida-A 등 (학습 제외)

각 ride 폴더 내부:
  gps_data_XXXXX.csv             # GPS 궤적 (1Hz, lat/lon/timestamp)
  front_camera_timestamps_XXXXX.csv  # 카메라 타임스탬프 (20Hz)
  imu_data_XXXXX.csv             # IMU (가속도계, 자이로, 지자기)
  control_data_XXXXX.csv         # 제어 입력 (선속도, 각속도)
  recordings/                    # 비디오 클립 (.ts, m3u8 playlist)
```

### 2.2 센서 특성

| 센서 | 샘플링 레이트 | 비고 |
|---|---|---|
| GPS | 1 Hz | 소수점 5~6자리, 약 1m 정밀도 |
| 카메라 (front/rear) | 20 Hz | .ts 형식 클립 → mp4 변환 |
| IMU | ~100 Hz | EKF 입력 |
| 제어 신호 | ~20 Hz | linear/angular velocity |

---

## 3. OSM 데이터 활용 방법

### 3.1 OSM이란?

**OpenStreetMap(OSM)** 은 전 세계 자원봉사자들이 구축한 오픈소스 지도 데이터베이스.  
우리 파이프라인에서는 두 가지 방식으로 OSM을 활용한다:

| 활용 목적 | OSM 구성요소 | 사용 시점 |
|---|---|---|
| 지도 배경 이미지 렌더링 | **OSM 래스터 타일** (tile.openstreetmap.org) | OSM 맵 이미지 생성 시 |
| 도로/보행로 경로 스냅 및 검증 | **OSRM** (로컬 Docker 서버) | episode_selector.py 필터링 시 |

---

### 3.2 OSM 래스터 타일 기반 맵 이미지 생성

#### (1) GPS 좌표 → 타일 좌표 변환 (Web Mercator)

FrodoBots 데이터에서 추출한 GPS 궤적(위도·경도)을 **Zoom 18 Web Mercator 타일** 좌표로 변환한다.

```python
def latlon_to_tile(lat, lon, zoom):
    n = 2 ** zoom
    tx = int((lon + 180.0) / 360.0 * n)           # 타일 X
    lat_r = math.radians(lat)
    ty = int((1.0 - math.log(math.tan(lat_r) +
              1.0 / math.cos(lat_r)) / math.pi) / 2.0 * n)  # 타일 Y
    return tx, ty
```

- **Zoom 18**: 1픽셀 ≈ 0.6m (가장 상세한 보행자 수준 해상도)
- 타일 1장 = 256×256픽셀

#### (2) 타일 다운로드 및 캐시

```
URL: https://tile.openstreetmap.org/{zoom}/{tx}/{ty}.png

- 세션당 1회 요청 (재시도 3회)
- 로컬 캐시 저장: osm_pipeline/tile_cache/{zoom}_{tx}_{ty}.png
- 재실행 시 캐시 재사용 → 네트워크 요청 최소화
- User-Agent 명시: "MBRA-Research/1.0"
```

#### (3) 타일 스티칭 (build_canvas)

GPS 궤적 전체를 덮는 타일을 모두 다운로드하여 **하나의 대형 캔버스**로 합친다.

```
GPS 바운딩 박스 계산
  → 필요한 타일 범위 산출 (여백 ±1타일)
  → 각 타일을 256px 단위로 배치
  → 최종 캔버스: (n_tx × 256) × (n_ty × 256) 픽셀
  → 전역 픽셀 좌표 원점(gx0, gy0) 기록
```

---

### 3.3 GPS 궤적 → 맵 위 경로 오버레이

#### (1) 경로 표현 방식

OSRM이 구축되지 않은 경우(또는 현재 파이프라인), **실제 GPS 궤적을 고정 경로**로 사용한다.

```python
# 1단계: 중복 GPS 좌표 제거 (1Hz GPS의 해상도 한계)
mask_unique = np.concatenate([[True],
    np.any(np.diff(gps_raw, axis=0) != 0, axis=1)])
gps_unique = gps_raw[mask_unique]

# 2단계: 1m 간격 선형 보간 (densify)
route_latlon = _densify_route(gps_unique, step_m=1.0)
# 결과: [[lat0, lon0], [lat1, lon1], ...] (1m 간격)
```

#### (2) 현재 프레임 기준 과거/미래 분할

```python
# EKF 누적 거리 비율로 현재 프레임의 route 위치 추정
frame_route_idx = map_frames_to_route(fp, route_latlon, lats, lons)

# 과거: 현재 위치 이전 경로
past_route   = route_latlon[:closest_idx + 1]

# 미래: 현재 위치 이후 경로
future_route = route_latlon[closest_idx + 1:]
```

#### (3) 캔버스 위 경로 그리기

```python
# 과거 경로 → 회색 선
cv2.line(img, pts[k-1], pts[k], (160, 160, 160), 4, cv2.LINE_AA)

# 미래 경로 → 빨간 선 (BGR: 0, 0, 255)
cv2.line(img, pts[k-1], pts[k], (0, 0, 255), 4, cv2.LINE_AA)

# Ego 마커 → 초록 원 (현재 위치, 이미지 중앙)
cv2.circle(img, (ego_cx, ego_cy), 7, (0, 200, 0), -1)
```

---

### 3.4 ego-heading-up 크롭 및 회전

```
① Ego 위치 주변 반경 25m를 캔버스에서 크롭
   (√2 × crop_r로 크게 잘라 회전 시 코너 잘림 방지)

② 진행 방향이 이미지 위를 향하도록 회전:
   rot_deg = 90° - heading_deg  (OpenCV CCW 기준)

③ 중앙 크롭 → 224×224 리사이즈
```

```python
heading_deg = math.degrees(heading_rad)  # EKF heading (East=0, North=+90°)
rot_deg = 90.0 - heading_deg
M = cv2.getRotationMatrix2D((sq/2, sq/2), rot_deg, 1.0)
rotated = cv2.warpAffine(sq_img, M, (sq, sq), borderValue=(255,255,255))
final = cv2.resize(rotated[c-r:c+r, c-r:c+r], (224, 224))
```

**최종 이미지 의미**:
- **이미지 위쪽** = 로봇 진행 방향
- **이미지 중앙** = 현재 Ego 위치 (초록 원)
- **빨간 선** = 앞으로 갈 경로 (위쪽으로 뻗음)
- **회색 선** = 지나온 경로 (아래쪽에서 올라옴)

---

### 3.5 OSRM을 이용한 경로 검증 (episode_selector)

`episode_selector.py`에서는 OSRM 로컬 서버를 통해 **GPS 궤적이 실제 보행로/도로 위에 있는지 검증**한다.

```
① GPS waypoint 샘플링 (EKF 누적 거리 기준, 최대 25개)
② OSRM foot 프로파일로 경로 스냅:
   GET http://localhost:{port}/route/v1/foot/{coords}
       ?overview=full&geometries=geojson
③ 반환된 OSRM 경로 vs 실제 GPS 궤적 비교:
   - Fréchet 거리 (형상 유사도) ≤ 0.5
   - Chamfer 거리 (밀착도) ≤ 0.15
   - OSRM 스냅 거리 ≤ 15m (도로에서 너무 멀면 제외)
   - 헤딩 오차 ≤ 45°
```

| OSRM 서버 | 대상 지역 | 포트 |
|---|---|---|
| Perth 보행로 데이터 | Perth, 호주 | 5001 |
| Taipei 보행로 데이터 | Taipei, 대만 | 5002 |
| Tokyo 보행로 데이터 | Tokyo, 일본 | 5003 |
| **신규 지역 (Wuhan 등)** | **10개 지역** | **미구축** |

---

### 3.6 OSM 시각화를 통한 데이터 선별 판단

지역 검토 시, 각 지역 대표 라이드의 OSM 조감도(`bev_overview.png`)를 생성하여 **육안으로 환경 적합성을 판단**했다.

**BEV 조감도 생성 과정**:
```
① GPS 궤적 전체를 OSM 타일 위에 오버레이 (파란 선)
② 시작점 (초록 원) / 끝점 (빨간 원) 표시
③ 이동 거리 기반 자동 zoom 조정:
   - 이동 < 30m  → zoom 18 (상세)
   - 30~100m     → zoom 17
   - > 100m      → zoom 16 (광역)
④ North-up 고정 (회전 없음)
⑤ 출력: bev_overview.png
```

**판단 기준 예시**:

| 지역 | BEV에서 확인된 내용 | 판단 |
|---|---|---|
| 선전, 중국 | 296개 rides가 30m×30m에 밀집, 공장 사유지 | ❌ 삭제 |
| 플로리다-A | OSM 지도에 도로 없음, 녹지 위 주행 | ❌ 삭제 |
| 우한, 중국 | 대학 캠퍼스 내 보행로, OSM 경로 명확 | ✅ 사용 |
| 마닐라, 필리핀 | 도심 도로 패턴, 지도 정합성 양호 | ✅ 사용 |
| 웰링턴, 뉴질랜드 | 공원 산책로, OSM 등록 경로 존재 | ✅ 사용 |

---

## 4. 데이터 선별 과정 (output_rides_11)

### 4.1 원본 981개 라이드 지역 분류

GPS 중심 좌표를 **0.01도(약 1km) 격자**로 군집화하여 지역별로 분류.

| 지역 | 원본 Rides | 환경 | 선별 결과 |
|---|---|---|---|
| 선전(深圳), 중국 | 296 | 공장 사유지 (~30m×30m) | ❌ 삭제 |
| 우한(武漢), 중국 | 127 | 대학 캠퍼스 보행로 | ✅ 사용 |
| 마닐라-A, 필리핀 | 61 | 도심 이면도로 | ✅ 사용 |
| 마닐라-B, 필리핀 | 53 | 주택가 도로 | ✅ 사용 |
| 로마, 이탈리아 | 53 | 공원 산책로 | ✅ 사용 |
| 웰링턴, 뉴질랜드 | 43 | 도심 공원 보행로 | ✅ 사용 |
| 플로리다-A, 미국 | 114 | 잔디밭 비포장 | ❌ 삭제 |
| 타이베이, 대만 | 39 | 공원 호수 보행로 | ✅ 사용 |
| 플로리다-B, 미국 | 33 | 도심 격자 도로 | ✅ 사용 |
| 브라이턴-A, 영국 | 20 | 주택가+공원 보행로 | ✅ 사용 |
| 마드리드-B, 스페인 | 7 | 공원 보행로 | ✅ 사용 |
| 마드리드-C, 스페인 | 6 | 공원 인근 보도 | ✅ 사용 |
| Perth, Vienna, Indiana 등 | ~130 | 비포장·소량·사유지 | ❌ 제외 |

### 4.2 삭제 기준

1. **선전 296개**: 30m×30m 사유 공장 단지 — 환경 다양성 없음
2. **플로리다-A 114개**: OSM 미등록 비포장 잔디밭, 지도 정합성 낮음
3. **소량·저품질 지역**: Perth(40), Vienna(17), Indiana(16), Madrid-A(12), Brighton-B(5), Florida-C(1)
4. **GPS 오류 데이터 22개**: lat=1000, lon=1000 등 좌표 범위 초과

### 4.3 최종 선별 결과

| 항목 | 수치 |
|---|---|
| 원본 라이드 수 | 981 |
| 최종 선별 라이드 수 | **442개** |
| 선별 지역 수 | **10개 지역** |
| 총 GPS 포인트 | **223,174개** |
| 총 주행 시간 | **약 62.5시간** |
| 총 주행 거리 | **약 115.4km** |

---

## 5. 원시 데이터 정리

### 5.1 삭제 과정

```bash
# GPS 기반 지역 분류 후 불필요 rides 삭제
python3 filter_rides.py  # 91개 rides (6.07 GB) 삭제
                          # 22개 GPS 오류 rides (684.7 MB) 삭제
# 총 113개 rides, 약 6.75 GB 정리
```

### 5.2 정리 후 폴더 구조

```
FrodoBots-2K/extracted/output_rides_11/
  ride_28153_20240406072220/   # Wuhan
  ride_28154_20240406073823/   # Wuhan
  ...
  (442개 rides만 존재)
```

---

## 6. EKF 전처리: convert_to_hf.py

### 6.1 왜 EKF가 필요한가?

원시 GPS 데이터는 두 가지 근본적인 한계가 있다:

| 문제 | 원인 | 결과 |
|---|---|---|
| **낮은 샘플링 레이트** | GPS = 1Hz (초당 1포인트) | 카메라(20Hz)와 직접 매칭 불가 |
| **낮은 정밀도** | 약 1~5m 오차, 소수점 5자리 해상도 | 연속된 두 GPS 좌표가 동일하게 찍히는 현상 빈번 |
| **헤딩 없음** | GPS는 위치만 제공, 방향 미포함 | 로봇이 어느 방향을 보고 있는지 알 수 없음 |

**EKF(Extended Kalman Filter)** 는 GPS와 IMU를 융합하여 이 문제를 해결한다:

```
GPS (1Hz, 낮은 정밀도) ──┐
                          ├─→ EKF ─→ filtered_position (10Hz, ~cm 정밀도 UTM XY)
IMU (100Hz, 누적 오차)  ──┘       └→ filtered_heading  (10Hz, 진행 방향 rad)
```

---

### 6.2 원시 데이터 → EKF 출력까지 단계별 변환

#### 단계 1: 원시 CSV 로드

각 ride 폴더에서 4개의 CSV 파일을 읽는다:

```
gps_data_XXXXX.csv:
  latitude, longitude, timestamp(ms)    ← 1Hz, 약 1m 정밀도

imu_data_XXXXX.csv:
  accelerometer: [x, y, z, t]           ← ~100Hz
  gyroscope:     [x, y, z, t]
  compass(magnetometer): [x, y, z, t]

control_data_XXXXX.csv:
  linear, angular, rpm_1~4, timestamp   ← ~20Hz

front_camera_timestamps_XXXXX.csv:
  timestamp(ms)                          ← 20Hz, 프레임별 타임스탬프
```

#### 단계 2: 비디오 인코딩 (ffmpeg)

`.ts` 형식 클립들을 하나의 `.mp4`로 합친다:

```
recordings/
  *1000__uid_e_video.ts  (클립 1)      front 카메라
  *1000__uid_e_video.ts  (클립 2)  ──→ ride_XXXXX_front_camera.mp4 (540×360)
  ...

  *1001__uid_e_video.ts  (클립 1)      rear 카메라
  *1001__uid_e_video.ts  (클립 2)  ──→ ride_XXXXX_rear_camera.mp4  (192×128)
```

- m3u8 플레이리스트를 파싱해 클립 순서를 결정
- ffmpeg `concat:` 프로토콜로 무손실 이어붙이기 후 H.264 재인코딩 (CRF=30)

#### 단계 3: 타임스탬프 정렬 및 10Hz 보간

각 센서는 서로 다른 레이트로 기록되어 있으므로, **10Hz 공통 타임라인**으로 보간한다:

```
GPS 1Hz:    ●         ●         ●         ●
IMU 100Hz:  |||||||||||||||||||||||||||||||||||
Camera 20Hz: | | | | | | | | | | | | | | | |
                ↓ cut_interp_clips()
10Hz grid:  ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ●
            (각 시점에서 가장 가까운 값으로 보간)
```

데이터 공백(센서 끊김, 5초 이상)이 있으면 그 지점에서 **에피소드를 분할**한다.

#### 단계 4: EKF 적용 (`do_ekf_filter()`)

`frodo_dataset/filtering.py`의 `do_ekf_filter()`가 핵심:

```
상태 벡터: [x(UTM 동), y(UTM 북), heading(rad), v(속도)]

예측 단계 (IMU 기반, 100Hz):
  x_{t+1} = x_t + v_t * cos(heading_t) * dt
  y_{t+1} = y_t + v_t * sin(heading_t) * dt
  heading → 자이로스코프 각속도로 업데이트

보정 단계 (GPS 도착 시, 1Hz):
  GPS (lat, lon) → UTM 변환 → 칼만 이득으로 상태 보정
  지자기(compass) → 헤딩 보정

출력 (10Hz):
  filtered_position: [easting, northing]  (UTM 미터 단위)
  filtered_heading:  heading (rad, East=0, North=+π/2)
```

**왜 UTM인가?**: GPS의 위도/경도는 구면 좌표계 → 거리 계산이 복잡. UTM은 평면 직교 좌표계라 `√(dx²+dy²)`로 바로 거리 계산 가능.

#### 단계 5: Arrow 형식으로 저장

HuggingFace `datasets` 라이브러리의 Arrow 포맷으로 저장:

```
모든 ride의 데이터가 하나의 Arrow 파일로 합쳐짐
  data-00000-of-00001.arrow

episode_index로 ride 구분:
  episode 0 → ride_28153 (Wuhan ride 1)
  episode 1 → ride_28154 (Wuhan ride 2)
  ...
  episode 441 → ride_XXXXX (Madrid-C ride 6)

frame_index: 에피소드 내 프레임 번호 (0, 1, 2, ...)
timestamp:   에피소드 시작 기준 경과 시간 (초)
```

---

### 6.3 EKF 전후 데이터 비교

| 항목 | 원시 GPS | EKF 출력 |
|---|---|---|
| 샘플링 레이트 | 1 Hz | **10 Hz** |
| 위치 형식 | 위도/경도 (도) | **UTM XY (미터)** |
| 위치 정밀도 | ~1~5m | **~10~30cm** |
| 헤딩 정보 | 없음 | **있음 (rad)** |
| 연속 중복 좌표 | 빈번 | 거의 없음 |
| 카메라 매칭 | 타임스탬프 불일치 | **10Hz 정렬 완료** |

---

### 6.4 실행 방법

```bash
# mbra conda 환경에서 실행 (host)
conda activate mbra
cd /media/ms/WD_BLACK_4TB/Learning-to-Drive-Anywhere-with-MBRA/frodo_dataset/

# output_rides_11 전용
python convert_to_hf.py \
  --rides_group output_rides_11 \
  --dataset_id  output_rides_11
```

**CLI 인자 (argparse 추가)**:

| 인자 | 기본값 | 설명 |
|---|---|---|
| `--raw_dir` | `.../FrodoBots-2K/extracted` | 원시 데이터 루트 |
| `--out_dir` | `.../FrodoBots-2K/processed` | 출력 루트 |
| `--dataset_id` | `frodobots_dataset` | 출력 서브폴더명 |
| `--rides_group` | None (전체 글로브) | 특정 `output_rides_*` 만 처리 |

---

### 6.5 출력 구조

```
FrodoBots-2K/processed/output_rides_11/
  train/
    data-00000-of-00001.arrow   # 전체 프레임 (10Hz, 442 rides)
  videos/
    ride_XXXXX_front_camera.mp4
    ride_XXXXX_rear_camera.mp4
  meta_data/
    info.json                   # fps, encoding 정보
    stats.json                  # 각 컬럼 평균/표준편차
    episode_data_index.json     # episode별 시작/끝 frame index
  dataset_cache.zarr             # 중간 캐시 (재실행 시 생략 가능)
```

---

### 6.6 Arrow 주요 컬럼

| 컬럼 | 타입 | 설명 |
|---|---|---|
| `observation.filtered_position` | float32[2] | EKF 위치 (UTM easting, northing, m) |
| `observation.filtered_heading` | float32 | EKF 헤딩 (rad, East=0, North=+π/2) |
| `observation.latitude` | float64 | 원시 GPS 위도 |
| `observation.longitude` | float64 | 원시 GPS 경도 |
| `observation.utm_position` | float64[2] | GPS→UTM 직접 변환 (EKF 보정 전) |
| `observation.accelerometer` | float32[3] | IMU 가속도 (x, y, z) |
| `observation.gyroscope` | float32[3] | IMU 각속도 (x, y, z) |
| `observation.images.front` | VideoFrame | 전방 카메라 (경로+타임스탬프) |
| `action` | float32[2] | 제어 입력 [linear_vel, angular_vel] |
| `episode_index` | int64 | ride 번호 (0-based) |
| `frame_index` | int64 | 에피소드 내 프레임 번호 |
| `timestamp` | float32 | 에피소드 시작 기준 경과 시간 (s) |

---

## 7. 세그먼트 분할: episode_selector.py

### 7.1 목적

EKF 처리된 Arrow 데이터에서 **학습에 적합한 유효 구간만 추출**

### 7.2 처리 기준 (필터링 조건)

| 조건 | 임계값 | 설명 |
|---|---|---|
| 최소 프레임 수 | 100 frames | 너무 짧은 구간 제거 |
| 최소 이동 거리 | 30.0 m | 정지 상태 구간 제거 |
| 정지 판정 거리 | 0.05 m/frame | 이 이하면 정지로 판단 |
| 장기 정지 구간 | 50 frames 이상 | 구간 분할 기준 |
| 최대 OSRM 스냅 거리 | 15.0 m | GPS 궤적 ↔ OSM 도로 최대 허용 오차 |
| Fréchet 거리 (정규화) | ≤ 0.5 | 궤적-도로 형상 일치도 |
| Chamfer 거리 (정규화) | ≤ 0.15 | 궤적-도로 밀착도 |
| 최대 헤딩 오차 | 45.0° | 진행 방향 ↔ 도로 방향 오차 |

### 7.3 OSRM 서버 의존성

```
ride_00 (기존):
  Perth  → OSRM 포트 5001
  Taipei → OSRM 포트 5002
  Tokyo  → OSRM 포트 5003

output_rides_11 (신규):
  ⚠️ Wuhan, Manila, Rome, Wellington 등
  → OSRM 서버 미구축 → 추후 구축 예정
```

### 7.4 출력 구조

```
osm_pipeline/osm_maps/
  episode_0001_seg01/
  episode_0001_seg02/
  episode_0002_seg01/
  ...
```

---

## 8. OSM 맵 이미지 생성: osm_map_generator.py

### 8.1 개요

각 에피소드/세그먼트의 프레임마다 **ego-heading-up 로컬 OSM 맵** 1장 생성  
→ 카메라 이미지와 1:1 대응하는 지도 이미지

### 8.2 파라미터

| 파라미터 | 값 |
|---|---|
| 출력 이미지 크기 | **224 × 224 px** |
| OSM 타일 Zoom 레벨 | **18** (~0.6m/px) |
| 표현 반경 (map_range_m) | **25m** |
| 과거 경로 색상 | 회색 `(160, 160, 160)` |
| 미래 경로 색상 | 빨간색 `(0, 0, 255)` BGR |
| Ego 마커 | 초록 원 `(0, 200, 0)` |
| 좌표계 | Web Mercator (EPSG:3857) |

### 8.3 ego-heading-up 회전 공식

**핵심 공식** (버그 수정 후):

```python
# filtered_heading: East=0, North=+90°, South=-90° (수학적 CCW)
# OSM 타일: 위=북쪽, 오른쪽=동쪽 (north-up)
# 목표: 로봇 진행 방향이 이미지 위를 향하도록 회전

heading_deg = math.degrees(heading_rad)
rot_deg = 90.0 - heading_deg   # OpenCV CCW 기준

# 검증:
# heading=0° (동쪽) → rot=90° CCW → 동쪽이 위 ✅
# heading=90° (북쪽) → rot=0° → 그대로 (북쪽이 위) ✅
# heading=-90° (남쪽) → rot=180° → 남쪽이 위 ✅
```

**수정 전 버그**: `rot_deg = 270.0 - heading_deg` → 방향이 반대로 나왔음

### 8.4 경로 시각화

```
OSM 배경 타일 (북쪽-위 기준 렌더링)
  + 과거 경로 (회색 선): 지나온 궤적
  + 미래 경로 (빨간 선): 앞으로 갈 궤적
  + Ego 마커 (초록 원): 현재 위치 (이미지 중앙)
  → 전체 이미지를 rot_deg만큼 CCW 회전
  → 224×224 크롭 (ego 중심)
```

### 8.5 GPS 궤적 기반 고정 경로 처리

```python
# 1. 연속 중복 GPS 제거 (GPS 해상도 한계 대응)
gps_unique = gps_raw[mask_unique]

# 2. 1m 간격 densify (선형 보간)
route_latlon = _densify_route(gps_unique, step_m=1.0)

# 3. 각 프레임 → route 인덱스 매핑
#    EKF 누적 거리 비율 기반, 단조 증가 보장
frame_route_idx = map_frames_to_route(fp, route_latlon, lats, lons)
```

### 8.6 BEV 조감도 생성

각 에피소드/세그먼트마다 **전체 궤적을 한눈에 볼 수 있는 개요 이미지** 1장 생성.

```
- 방향: 북쪽-위 고정 (회전 없음)
- 중심: 세그먼트 시작점
- Zoom 자동 결정:
    이동 거리 < 30m  → zoom=18
    30m ~ 100m       → zoom=17
    > 100m           → zoom=16
- 출력: bev_overview.png
```

---

## 9. 테스트 케이스: Episode 6

### 9.1 초기 테스트 (Episode 6, Segment 1)

- **지역**: Perth, 호주 (ride_00 기반)
- **목적**: ego-heading-up 회전 공식 검증
- **헤딩**: 약 -82° (거의 정남 방향)
- **기대 결과**: 빨간 미래 경로가 이미지 위쪽을 향해야 함

### 9.2 발견된 버그 및 수정

| 항목 | 수정 전 | 수정 후 |
|---|---|---|
| 회전 공식 | `rot_deg = 270 - heading_deg` | `rot_deg = 90 - heading_deg` |
| 결과 | 북쪽-위와 유사, 방향 무관 | 로봇 진행 방향이 항상 위를 향함 |

### 9.3 Episode 8로 교차 검증

- **헤딩**: 약 -155° (SSW, 남남서)
- `rot_deg = 90 - (-155) = 245°` CCW 회전
- 결과: 빨간 선이 이미지 위로 향함 ✅

### 9.4 전체 재생성

수정 후 167개 세그먼트 전체 재생성:
- **총 125,306장** OSM 맵 이미지 생성
- 프레임 수 불일치: **0건** (Arrow 프레임 수와 완전 일치)

---

## 10. ride_00 전처리 파이프라인 전체 흐름

```
[원시 데이터]
FrodoBots-2K/extracted/output_rides_0/
  (Tokyo 기반 231개 rides)

        ↓ convert_to_hf.py (mbra 환경)

[Arrow 데이터셋]
FrodoBots-2K/processed/frodobots_dataset/
  train/data-00000-of-00001.arrow
    - 252 episodes, 392,745 frames (10Hz)
    - filtered_position, filtered_heading 포함
    - Perth: 153,830 / Taipei: 207,711 / Tokyo: 31,204 frames

        ↓ episode_selector.py (OSRM 필요)

[유효 세그먼트 목록]
  - 정지 구간 제거, 최소 거리 30m 이상
  - OSRM 스냅 검증 (보행로/도로 위 주행 확인)

        ↓ osm_map_generator.py

[OSM 맵 이미지]
osm_pipeline/osm_maps/
  episode_XXXX_segYY/
    osm_map_000000.png  (224×224, ego-heading-up)
    ...
    bev_overview.png
    gps.csv
```

---

## 11. output_rides_11 전처리 파이프라인 (진행 중)

```
[원시 데이터 정리 완료]
FrodoBots-2K/extracted/output_rides_11/
  442개 rides (10개 지역, 223,174 GPS 포인트)

        ↓ convert_to_hf.py --rides_group output_rides_11 [실행 중]

[Arrow 데이터셋] (생성 예정)
FrodoBots-2K/processed/output_rides_11/
  train/, videos/, meta_data/

        ↓ episode_selector.py [OSRM 서버 구축 후 실행 예정]

        ↓ osm_map_generator.py

[OSM 맵 이미지] (생성 예정)
osm_pipeline/output_rides_11/osm_maps/
```

---

## 12. 모델 입력 형식

### 12.1 학습 데이터 구성

```
각 프레임마다:
  입력:
    - front camera 이미지 (540×360 → 모델별 리사이즈)
    - OSM 로컬 맵 이미지 (224×224, ego-heading-up)
  출력 (레이블):
    - action: [linear_velocity, angular_velocity]
```

### 12.2 타임스탬프 기반 매칭

- Arrow 데이터: 10Hz
- OSM 맵: 프레임 1:1 대응 (동일 10Hz)
- output_rides_11 GPS 1Hz → nearest-neighbor 보간으로 카메라 프레임과 매칭

### 12.3 NoMaD 호환 형식

```python
# 모델 입력 예시
{
  "observation.images.front": Tensor[B, T, C, H, W],
  "observation.osm_map":      Tensor[B, T, 3, 224, 224],
  "action":                   Tensor[B, T, 2],
}
```

---

## 13. 현재 상태 및 미해결 과제

### 13.1 완료된 작업

- [x] ride_00 EKF 전처리 (Arrow 생성, 392,745 frames)
- [x] ego-heading-up 회전 버그 수정 (`270→90`)
- [x] ride_00 OSM 맵 전체 재생성 (167 세그먼트, 125,306장)
- [x] output_rides_11 지역 분류 및 불필요 rides 삭제 (113개, 6.75GB)
- [x] `convert_to_hf.py` argparse 개선 (재사용 가능)
- [x] output_rides_11 EKF 전처리 실행 중 (442 rides)

### 13.2 진행 중

- [ ] output_rides_11 `convert_to_hf.py` 완료 대기 중
- [ ] 새 지역(Wuhan, Manila, Rome 등) OSRM 서버 구축
- [ ] output_rides_11 `episode_selector.py` 실행
- [ ] output_rides_11 OSM 맵 이미지 생성

### 13.3 알려진 한계 및 개선 방향

| 항목 | 현재 한계 | 개선 방향 |
|---|---|---|
| OSRM 서버 | Perth/Taipei/Tokyo 3개 지역만 구축 | 10개 신규 지역 PBF 다운로드 후 구축 |
| GPS 헤딩 추정 | output_rides_11은 EKF 대신 GPS 궤적 기반 추정 | EKF 완료 후 `filtered_heading` 사용 |
| 비디오 매칭 | 일부 rides에서 프레임 수 ↔ 타임스탬프 불일치 | 에러 rides 개별 확인 필요 |
| 데이터 불균형 | Wuhan 127 rides vs Madrid-C 6 rides | 학습 시 지역별 샘플링 가중치 적용 |
| OSM 맵 품질 | 일부 지역 타일 미비 (예: 캠퍼스 내부) | 줌 레벨 조정 또는 타일 캐시 보완 |

---

## 14. 파일 구조 전체 요약

```
Learning-to-Drive-Anywhere-with-MBRA/
│
├── FrodoBots-2K/
│   ├── extracted/
│   │   ├── output_rides_0/          # ride_00 원시 (231 rides, Tokyo 등)
│   │   ├── output_rides_11/         # 신규 원시 (442 rides, 10개 지역) ← 정리 완료
│   │   └── output_rides_3/          # Florida-A 등 (미사용)
│   └── processed/
│       ├── frodobots_dataset/        # ride_00 Arrow (392,745 frames)
│       │   ├── train/*.arrow
│       │   ├── videos/*.mp4
│       │   └── meta_data/
│       └── output_rides_11/          # 신규 Arrow (생성 중)
│           ├── train/*.arrow
│           ├── videos/*.mp4
│           └── meta_data/
│
├── frodo_dataset/
│   ├── convert_to_hf.py              # EKF 전처리 (argparse 개선)
│   ├── filtering.py                  # do_ekf_filter() 구현
│   └── interpolation_utils.py        # 10Hz 보간, 구간 분할
│
├── osm_pipeline/
│   ├── py/
│   │   ├── osm_map_generator.py      # ride_00 OSM 맵 생성기
│   │   └── episode_selector.py       # 유효 세그먼트 선별 (OSRM 필요)
│   ├── osm_maps/                     # ride_00 OSM 맵 이미지 (125,306장)
│   │   └── episode_XXXX_segYY/
│   │       ├── osm_map_NNNNNN.png
│   │       ├── bev_overview.png
│   │       └── gps.csv
│   └── output_rides_11/
│       ├── osm_map_generator_rides11.py   # GPS 기반 임시 생성기
│       ├── rides_11_prepro.md             # 전처리 문서 (한국어)
│       └── osm_maps/                      # 생성 예정
│
└── pipeline_overview.md              # 이 문서
```

---

## 15. 핵심 스크립트 실행 순서 정리

```bash
# Step 1: EKF 전처리 (mbra 환경)
conda activate mbra
cd frodo_dataset/
python convert_to_hf.py \
  --rides_group output_rides_11 \
  --dataset_id output_rides_11

# Step 2: OSRM 서버 구축 (각 지역 PBF 필요)
# docker run -p 5004:5000 osrm/osrm-backend ...

# Step 3: 유효 세그먼트 선별
cd osm_pipeline/py/
python episode_selector.py \
  --arrow_path ../../FrodoBots-2K/processed/output_rides_11/train/ \
  --output_dir ../output_rides_11/segments/

# Step 4: OSM 맵 이미지 생성
python osm_map_generator.py \
  --segments_dir ../output_rides_11/segments/ \
  --output_dir ../output_rides_11/osm_maps/
```

---

*© OpenStreetMap contributors | FrodoBots-2K Dataset*
