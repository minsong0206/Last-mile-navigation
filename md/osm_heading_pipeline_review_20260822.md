# OSM/OSRM 맵 생성 & 헤딩 정렬 파이프라인 검증 (2026-08-22)

> 목적: FrodoBot Mini 실배포 전, OSM 지도 생성 파이프라인과 헤딩 정렬 로직을 실제 코드 기준으로
> 검증하고, "배포 웹사이트에서 지도가 로봇 헤딩 방향을 제대로 못 가리키는" 버그의 원인을 특정한다.
> 이 문서는 지도교수 발표/설명용으로 작성되었으며, 모든 결론은 추측이 아니라 실제 코드
> (`Last-mile-navigation/` 하위)를 읽고 확인한 내용이다.

---

## 0. 조사한 파일

- `deployment/build_live_map.py` — 실배포용 실시간 맵 빌더
- `deployment/omnivla_edge_deploy.py` — 실배포 추론/제어 루프
- `deployment/debug_web.py` — 모니터링 대시보드
- `deployment/LogoNav_frodobot.py` — 구(舊) LogoNav 모델용 레거시 배포 스크립트 (현재 미사용,
  헤딩 부호 관례를 그대로 복사해온 원본)
- `osm_pipeline/py/osm_map_generator.py` — 핵심 맵 렌더링 함수(`render_frame`, `build_canvas` 등),
  ride_00 학습 데이터 생성기
- `osm_pipeline/osm_data/output_rides_11/osm_map_generator_rides11.py` — **실제 배포 체크포인트가
  학습된 rides_11 데이터셋**의 맵/헤딩 생성기
- `osm_pipeline/py/rides11_dataset.py` — 학습 시 Dataset/ego-frame 변환
- `examples/basics/15_waypoint_navigation.py`, `README.md` (상위 `earth-rovers-sdk` 리포) — SDK
  `orientation` 필드의 관례 확인용
- `docs/pipeline_overview.md`, `docs/deployment_verification_checklist.md`,
  `Last-mile-navigation/CLAUDE.md` — 이전 세션에서 이미 검증된 내용 확인용

---

## 1. OSM/OSRM 배포 파이프라인 (실측)

```
[출발 GPS(첫 poll_frodobot 결과), 목표 GPS(--goal_lat/--goal_lon)]
        ↓  query_osrm_route()  build_live_map.py:79-102
   OSRM 서버(localhost:{port}, foot 프로파일)에 1회만 GET 요청
   → osrm_port(lat, lon) osm_map_generator.py:52-64 로 지역별 포트 결정 (서울=5011)
   → 반환된 [[lon,lat],...] geometry → [[lat,lon],...]로 변환
   → _densify_route(route, 1.0m) osm_map_generator.py:384-397 로 1m 간격 보간
        ↓
[route_latlon: (M,2) 배열, 전체 경로, 1m 간격] — LiveMapBuilder._route_latlon 에 캐싱
        ↓  build_canvas() osm_map_generator.py:117-141
   route bbox를 덮는 OSM 래스터 타일(zoom=18, tile.openstreetmap.org)을 다운로드/스티칭
   → 캔버스 전체 1장 (gx0,gy0 = 캔버스 원점의 global pixel 좌표) — 이것도 set_goal() 1회만 실행
        ↓  (매 제어 루프 3Hz마다 반복되는 부분)
   현재 (lat,lon) → _closest_route_idx() build_live_map.py:140-145 로 route 상 가장 가까운 인덱스 탐색
   → future_route = route_latlon[idx:]  (남은 전체 경로, 고정 lookahead 거리 제한 없음)
   → past_track (로봇이 실제 지나온 (lat,lon), deque maxlen=200) 사용
        ↓  render_frame() osm_map_generator.py:152-237
   1) 캔버스 위에 past(회색)/future(빨강) 선 + ego 마커(초록) 그리기 — 위경도를
      latlon_to_pixel_global() (Web Mercator, EPSG:3857) 으로 전역 픽셀좌표 변환 후
      global_to_canvas() 로 캔버스 로컬좌표로 이동
   2) ego 중심으로 sqrt(2)*crop_r 반경 만큼 여유있게 pre-crop (osm_map_generator.py:192-215)
   3) rot_deg = 90 - heading_deg 로 회전 (osm_map_generator.py:223-224)
   4) 회전 후 중앙을 map_range_m 반경만큼 crop (osm_map_generator.py:230-233)
   5) out_size(기본 224)로 resize, BGR→RGB (osm_map_generator.py:236-237)
        ↓
[224×224 RGB numpy] → LiveMapBuilder.transform: Resize(96,96)+ToTensor+Normalize
        (build_live_map.py:116-120, IMG_MEAN/STD)
        ↓
[(1,3,96,96) 텐서] → OmniVLA-Edge-Odom 모델 입력 (omnivla_edge_deploy.py:213-226)
```

### 핵심 파라미터 (실측값)

| 항목 | 값 | 근거 |
|---|---|---|
| OSM 타일 zoom | 18 | `osm_map_generator.py:70` `ZOOM=18` |
| 지도 표시 반경 (map_range_m) | 25/20/12m (체크포인트별) | `omnivla_edge_deploy.py:19-21`, `--map_range` 필수 인자 |
| 렌더 해상도 | 224×224 | `MAP_SIZE_PX=224`, `osm_map_generator.py:67` |
| 모델 입력 해상도 | 96×96 | `build_live_map.py:107`, `rides11_dataset.py:45` |
| 경로 재쿼리 주기 | 배포 시작 1회 + 이탈(15m) 시만 | `omnivla_edge_deploy.py:191-204` |
| 경로 색상 | 미래=빨강(0,0,255) BGR, 과거=회색(160,160,160) | `osm_map_generator.py:72-73` |
| 로봇 위치 | 항상 이미지 정중앙 | `render_frame`에서 crop이 ego 중심으로 고정 |

OSRM은 "실시간으로 매 프레임 재쿼리"하는 게 아니라 **목표 설정 시 딱 1번**만 경로를 계산하고
이후엔 그 경로에서 현재 위치와 가장 가까운 점만 찾는 구조다 (`build_live_map.py` 1-19줄 주석에
이 설계 배경이 직접 쓰여 있음 — 예전엔 매 프레임 재쿼리하는 버그가 있었고 고쳤다고 명시).

---

## 2. 헤딩 버그 조사 — 원인을 코드에서 특정함

### 학습 시 heading 소스 (실제 배포 체크포인트가 학습된 rides_11 데이터셋 기준)

```python
# osm_pipeline/osm_data/output_rides_11/osm_map_generator_rides11.py:65-77
def estimate_headings(lats, lons):
    """GPS 궤적 기반 헤딩 추정 (East=0, North=+π/2, rad)."""
    ...
    headings[i] = math.atan2(dlat, dlon)   # 73행
```

`atan2(북쪽성분, 동쪽성분)` → **East=0°, North=+90°, 수학적 CCW** 컨벤션. `render_frame()`의
`rot_deg = 90 - heading_deg` 공식은 정확히 이 컨벤션을 전제로 검증되었다
(`docs/deployment_verification_checklist.md` 실험 0-2, `scripts/analysis/verify_map_coordinate_axes.py`로
나침반/좌우회전 합성 테스트 완료 — **이 회전 공식 자체는 버그 없음**, 이미 검증됨).

### 배포 시 heading 소스

```python
# deployment/omnivla_edge_deploy.py:165-174
def poll_frodobot(self):
    ...
    gps = requests.get(f"{FRODOBOT_BASE}/data", timeout=5.0).json()
    lat, lon = gps["latitude"], gps["longitude"]
    # LogoNav_frodobot.py와 동일한 부호 규약: orientation(시계방향, deg) → CCW radian
    heading_rad = -float(gps["orientation"]) / 180.0 * math.pi   # 171행
```

이 공식은 `deployment/LogoNav_frodobot.py:95`의
`cur_compass = -float(gpsdata["orientation"])/180.0*3.141592`을 그대로 복사해온 것이다.

### [Issue]
`orientation` → heading 변환식이 **East=0° 기준으로의 이동(+90°)을 빠뜨리고 단순 부호반전만 함**.
그 결과 배포 시 `render_frame()`에 들어가는 heading이 학습 때 컨벤션과 상시 90° 어긋난다.

### [Evidence]
- `README.md:105-153`(상위 `earth-rovers-sdk` 리포)의 `/data` 응답 예시: `"orientation": 128`
  (0-360 범위의 각도값)
- `examples/basics/15_waypoint_navigation.py:77-95`의 `calculate_bearing()`이 계산하는 값은
  **표준 지리방위각**(0°=North, 시계방향 증가: `x=sin(Δlon)cos(lat2)`,
  `y=cos(lat1)sin(lat2)-...`, `atan2(x,y)`)이고, 같은 파일 165행에서
  `heading_error = normalize_angle(target_bearing - current_heading)`으로 이 `target_bearing`을
  SDK의 `orientation`(`current_heading`, 44행)과 **직접 뺄셈**한다. 두 값이 같은 물리적 정의
  (진북=0°, 시계방향 증가)가 아니면 이 뺄셈 자체가 의미가 없으므로, `orientation`은
  **"진북=0°, 시계방향(CW) 증가" 나침반 방위각**으로 봐야 한다.
- 반면 학습 heading은 East=0°, CCW 증가 (수학적 극좌표) — **기준축(East vs North)도 다르고
  회전방향(CCW vs CW)도 다름**.
- CW 나침반 방위각 θ를 East=0/CCW 수학각으로 정확히 바꾸는 공식은 `θ_math = 90° - θ_compass`
  (즉 `-θ` 가 아니라 `90 - θ`) — 회전방향 반전은 맞았지만 **원점 이동(+90°)이 빠짐**.
  - 검증: θ=0°(진북) → 올바른 값은 90°(북쪽=math 90°)인데 코드값은 `-0=0°`(East로 해석됨)
    → **90° 오차**
  - θ=90°(동쪽) → 올바른 값 0°인데 코드값 `-90°`(South로 해석) → 90° 오차
  - θ=270°(서쪽) → 올바른 값 180°인데 코드값 `+90°`(North로 해석) → 90° 오차
  - **모든 각도에서 일정하게 90° 어긋남**.
- 이 버그는 `omnivla_edge_deploy.py` 고유 버그가 아니라 `LogoNav_frodobot.py:95`에서 그대로
  물려받은 것이며, `rides11_dataset.py:292-294`의 ego-frame 변환
  (`x_ego = dx*cos_h + dy*sin_h`, East=0/CCW 전제)과도 어긋나므로 **LogoNav 계열 코드베이스
  전체에 잠재된 레거시 버그**로 보인다.

### [Why it matters]
`render_frame()`의 `rot_deg = 90 - heading_deg`에 이 90°만큼 어긋난 heading이 들어가면, 캔버스가
항상 실제와 90° 다른 방향으로 회전되어 "위=로봇 진행방향"이 성립하지 않는다.
`Last-mile-navigation/CLAUDE.md`에 기록된 "실로봇 좌회전 편향(2026-08-10)" 문제와도 방향이 일치할
수 있는, 컴퍼스 캘리브레이션과 무관한 **소프트웨어 컨벤션 버그**다. (기존 메모는 "자북 편각/마운트
오프셋"처럼 하드웨어 원인을 의심했는데, 이번 조사로 하드웨어 캘리브레이션과 무관하게 **코드 공식
자체에 90° 오차가 있음**이 확인됨 — 실제 나침반이 완벽히 캘리브레이션되어 있어도 이 버그는 그대로
발생한다.)

### [Recommended fix] (아직 미적용 — 확인 요청 사항)
```python
# 현재 (omnivla_edge_deploy.py:171)
heading_rad = -float(gps["orientation"]) / 180.0 * math.pi
# 제안
heading_rad = math.radians(90.0 - float(gps["orientation"]))
```
`LogoNav_frodobot.py:95`도 원리상 동일 버그를 갖고 있지만, 현재 실배포 스크립트가 아니므로
우선순위는 낮음.

### [Verification]
1. 로봇을 진북(나침반 앱 등으로 확인)으로 세운 뒤 `/data`의 `orientation` 값을 읽는다.
2. 수정 전/후 각각 `render_frame()` 결과 지도를 저장해 "위쪽"이 실제 진행 방향(여기선 북쪽)과
   일치하는지 육안 비교.
3. 이미 만들어져 있는 진단 로그(`omnivla_edge_deploy.py:273-288`의 `heading_diff_deg`: IMU
   orientation 기반 heading vs GPS 궤적 기반 heading 비교)를 활용 — 수정 후 이 diff가 이전보다
   안정적으로 0 근처에 모이는지 확인 (참고: 이 비교 로직 자체는 두 heading 소스가 다르다는 것만
   보여줄 뿐 90° 오프셋 여부를 직접 알려주진 않으므로, 1번의 직접 나침반 테스트가 확정적 검증
   방법).

---

## 3. 통제된 헤딩 테스트 설계 (0°/90°/180°/270°)

**코드 변경 없이 지금 바로 가능**하다 — `build_live_map.py:229-246`에 이미 이 용도의 `__main__`
블록이 있다:

```python
if __name__ == "__main__":
    ...
    p.add_argument("--heading_deg", type=float, default=0.0)
    ...
    img = builder.get_map_image(args.lat, args.lon, math.radians(args.heading_deg))
    Image.fromarray(img).save(args.out)
```

이 스크립트는 **route 투영(OSRM+타일)과 회전(rotate)을 완전히 분리**해서 볼 수 있게 해준다 —
`lat/lon/goal`은 고정한 채 `--heading_deg`만 바꾸면 route/캔버스는 그대로이고 회전만 달라지므로:

```bash
for h in 0 90 180 270; do
  python3 deployment/build_live_map.py \
    --lat <출발위도> --lon <출발경도> \
    --goal_lat <목표위도> --goal_lon <목표경도> \
    --heading_deg $h --map_range 20 \
    --out /tmp/heading_test_${h}.png
done
```

**분리 판정 기준**:
- 4장 모두에서 route(빨강)/ego(초록) 상대 배치가 회전만 다르고 형태는 동일
  → **route 투영은 정상**, 문제는 회전/heading 소스 쪽
- 4장을 90°씩 서로 돌려서 겹쳐봤을 때 정확히 일치 → **`render_frame()` 회전 공식은 정상**
  (이미 `verify_map_coordinate_axes.py`로 검증된 결과와 동일해야 함)
- 실제 로봇 IMU를 연결한 실배포 화면(대시보드 `/map.jpg`)에서 위 4장과 다른 패턴으로 어긋난다면
  → **IMU/orientation 변환(2번 항목의 버그)이 원인** (route/회전 로직 자체는 결백)

OSRM 서버(5011번 포트, 서울 지역)와 인터넷(타일 다운로드) 접근이 필요하다.

---

## 4. 큰 캔버스 → 회전 → 크롭 순서 검증

**결론: 이미 원하는 순서로 구현되어 있음.** `render_frame()` (`osm_map_generator.py:152-237`)
안에서:

```
① build_canvas() — route 전체 bbox를 덮는 대형 OSM 타일 캔버스 (route보다 훨씬 큼)
② 그 큰 캔버스 위에 route(빨강/회색)+ego 마커를 직접 투영 (192줄 이전)
③ ego 중심으로 sqrt(2)*crop_r + 여유(4~10px) 만큼 "회전 시 코너 잘림 방지용" pre-crop
   (osm_map_generator.py:196-198, 204-206)
④ 정사각형으로 패딩 (211-215)
⑤ cv2.warpAffine으로 rot_deg만큼 회전 (225-228)
⑥ 회전 후에야 최종 map_range_m 반경으로 중앙 crop (230-233)
⑦ out_size로 resize (236)
```

`sqrt(2)`배 pre-crop이 정확히 "회전 대각선 길이"를 커버하므로 회전 중 코너 정보 손실이 없다 —
**정보 손실 이슈는 이미 해결되어 있고, 추가 수정 불필요**. (다만 pad를 `canvas_bgr` 전체에 대해
적용해서 큰 캔버스일수록 메모리를 조금 더 쓰긴 하지만, 정확성 문제는 아니고 사소한 성능 이슈.)

---

## 5. 궤적 누적(trajectory accumulation) 검증

| | 학습 시 | 배포 시 |
|---|---|---|
| 저장 방식 | `route_latlon` (OSRM/GPS 궤적을 1m 간격 densify) 중 `[:closest_idx+1]` (`osm_map_generator.py:479`) | `self.past_track = deque(maxlen=200)` (`omnivla_edge_deploy.py:147`), 매 tick `(lat,lon)` append (271줄) |
| 누적 기준 | **거리 기준(1m 간격 보간된 전체 세그먼트 시작점까지)** — 사실상 무제한 | **프레임(tick) 개수 기준** — GPS fix가 유효한 tick마다 1개, 최대 200개 |
| 부드러움 | 1m 간격으로 densify된 매끈한 선 | **raw GPS 샘플 그대로** (densify 안 함) → 저속 주행 시 GPS jitter로 지그재그로 보일 수 있음 |
| 실질적 커버 범위 | 세그먼트 시작점까지 전부 (map_range보다 훨씬 김, 어차피 crop되어 map_range_m 밖은 안 보임) | 3Hz 기준 200 tick ≈ 최대 66초. 로봇 최고속도 0.3m/s 가정 시 이론상 최대 ~20m — `map_range_m`(12/20/25m)에 근접하거나 못 미칠 수 있음 |

**누적 범위를 늘리면 바뀌는 것**: `deque(maxlen=200)`을 늘리면 특히 **저속/정지-출발 반복 구간**에서
회색 past-line이 지도 반경 끝까지 채워짐 — 현재도 일반 주행 속도라면 200개면 충분히 map_range를
커버하지만, 로봇이 자주 멈췄다 가는 상황에서는 부족할 수 있다. 다만 이건 heading 버그와는 독립적인
별개 파라미터 튜닝 이슈.

**학습-배포 불일치로 짚을 점**: 배포의 past-line은 `_densify_route()`를 거치지 않은 raw 좌표라서,
학습 때 본 매끈한 회색선과 시각적 분포가 다르다(→ OOD 가능성). 필요시 `past_track`도
`_densify_route()`로 보간해서 넘기는 게 학습 분포와 더 가깝다.

---

## 6. GPS/IMU 신호 현황 (Kalman Filter는 아직 미통합 — 현황 확인만)

`/data` 응답(상위 `earth-rovers-sdk` 리포 `README.md:105-153`)에서 실제로 제공되는 신호:

| 신호 | 필드 | 현재 배포 코드에서 사용 여부 |
|---|---|---|
| GPS 위경도 | `latitude`, `longitude` | ✅ 사용 (`poll_frodobot()`) |
| GPS 신호세기 | `gps_signal` | ❌ 로깅만 (`record["frodobot_raw"]`), 헤딩/제어엔 미사용 |
| 융합 헤딩(추정) | `orientation` | ✅ 사용 — 단, 위 2번 항목의 변환 버그 있음 |
| 가속도계 | `accels[[x,y,z,t],...]` (~버스트 단위) | ❌ 로깅만, 실시간 융합 없음 |
| 자이로 | `gyros[[x,y,z,t],...]` | ❌ 로깅만 |
| 지자기 | `mags[[x,y,z,t]]` | ❌ 로깅만 |
| 속도 | `speed` | ❌ 미사용 |

**결론**: raw accel/gyro/mag이 이미 `/data`로 제공되고 `deployment/logs/deploy_*.jsonl`에 전부
로깅되고 있어 (`omnivla_edge_deploy.py:258-260`) 향후 GPS+IMU 칼만 필터 통합의 재료는 이미
갖춰져 있다. 다만 현재는 **`orientation`(로봇 내부에서 어떤 형태로든 융합된 값으로 추정) 하나만
그대로 heading으로 쓰고 있고, raw IMU 융합은 전혀 하지 않는 상태**다. 2번 항목의 컨벤션 버그부터
확정/수정하고, 그 후에 raw accel/gyro를 실제로 융합하는 EKF를 붙이는 순서가 맞다 — 지금 칼만필터를
넣으면 "컨벤션 버그"와 "필터 자체 오차"가 뒤섞여 디버깅이 더 어려워진다.

---

## 7. 최종 요약

1. **현재 배포 파이프라인**: OSRM 1회 라우팅(`build_live_map.py:79-102`) → 타일 스티칭
   (`osm_map_generator.py:117-141`) → route/ego 투영 → 여유있게 pre-crop → heading 기반 회전 →
   최종 crop → resize(224→96) → 모델. (섹션 1)
2. **관련 파일/함수**: `build_live_map.py::LiveMapBuilder`,
   `omnivla_edge_deploy.py::poll_frodobot/build_inputs`,
   `osm_map_generator.py::render_frame/build_canvas/osrm_port`. (섹션 1, 2)
3. **현재 좌표 컨벤션**: 학습(East=0°, North=+90°, CCW, `atan2(dlat,dlon)`) vs 배포 heading 변환
   (부호만 반전, 원점 이동 없음) — **서로 다름**. (섹션 2)
4. **현재 맵 파라미터**: zoom=18, map_range_m=12/20/25(체크포인트별), 렌더 224px→모델입력 96px.
   (섹션 1)
5. **현재 IMU/heading 컨벤션**: `orientation`은 진북=0°, 시계방향 증가 나침반 방위각으로 추정
   (다른 예제 스크립트의 사용 방식으로 추론, SDK 공식 문서에 명시적 정의는 없음 — 확실성 중간).
   (섹션 2)
6. **헤딩 버그의 가장 유력한 원인**: `omnivla_edge_deploy.py:171`
   (`heading_rad = -orientation/180*pi`)에서 **+90° 원점 이동이 누락**됨. 회전 공식
   (`render_frame`의 `90-heading`)과 route 투영 로직은 별도 검증(0-2 실험)에서 이미 문제없음으로
   확인됨. (섹션 2)
7. **0/90/180/270 테스트**: `build_live_map.py`의 기존 `__main__` CLI로 코드 수정 없이 즉시 실행
   가능. (섹션 3)
8. **큰 캔버스→회전→크롭 필요 여부**: **이미 그렇게 구현되어 있음** (sqrt(2) pre-crop), 추가 수정
   불필요. (섹션 4)
9. **학습-배포 불일치**: (a) heading 소스 자체가 다름(GPS-atan2 vs IMU orientation, 90° 오프셋),
   (b) past-track이 배포에선 densify 안 된 raw GPS라 학습 때보다 지그재그일 수 있음. (섹션 2, 5)
10. **우선순위별 변경 제안** (아직 미적용, 확인 후 적용 권장):

| 우선순위 | 내용 | 근거 |
|---|---|---|
| 1 (최우선) | `omnivla_edge_deploy.py:171`의 heading 변환식을 `math.radians(90 - orientation_deg)`로 수정 | 90° 오프셋의 근본 원인으로 특정됨 |
| 2 | 수정 후 반드시 3번 항목의 0/90/180/270 테스트 + 실제 나침반 대조로 검증 | 가설 확정 |
| 3 | `past_track`도 `_densify_route()`로 보간해서 학습 때 past-line 분포와 맞추기 | train/deploy 분포 불일치 완화 |
| 4 (낮음) | `past_track` `maxlen=200`을 상황에 맞게 조정 검토 (저속 구간 대응) | 섹션 5 |
| 5 (낮음) | `LogoNav_frodobot.py:95`도 동일 버그 있으나 현재 미사용 스크립트라 후순위 | 참고용 |
