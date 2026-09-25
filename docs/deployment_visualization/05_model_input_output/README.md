# 05. Model Input → Output (실제 checkpoint, 실제 HuggingFace 주행 데이터)

`osm_pipeline/py/osm_map_generator.py`(맵 회전/렌더링), `deployment/build_live_map.py`
(`LiveMapBuilder`), `deployment/omnivla_edge_deploy.py`(heading 추정, 상수)의 **실제
프로덕션 함수를 그대로 import**해서, HuggingFace의 실제 raw 주행 세션 데이터로
"OSM heading-up 변환 → 실제 6프레임 카메라+지도 모델 입력 → 실제 checkpoint 추론"
전 과정을 offline으로 재현한다. 새로 지어낸 map-rendering 코드나 임의의 수치는
없다 — 아래 "재사용한 실제 코드" 절 참고.

## 사용한 데이터 / checkpoint

- HuggingFace dataset: `minsonganingee/frodobot-drive-2026-08-09_07-14-06`
  (raw ride recording: `control.jsonl` 1076개 카메라 프레임 @ ~8.4Hz,
  `gps_imu.jsonl` 109개 GPS/IMU 샘플 @ ~1Hz, 128초 분량. FrodoBots-2K/rides_11
  아카이브가 아니라 **학습에 전혀 쓰이지 않은 새 raw 세션** — 이 프로젝트가 요구한
  "실제 주행 데이터"에 해당)
- checkpoint: `checkpoints/omnivla_edge_rides11_odom_20m_20260910/best.pth`
  (`MAP_RANGE_M=20`, `MODEL_PARAMS`는 `deployment/omnivla_edge_deploy.py` 그대로)
- 선택한 3개 시나리오 tick (원본 `local_timestamp`, 아래 "시나리오 선정 방법" 참고):

  | 시나리오 | t (local_timestamp) | lat, lon | heading_deg | ADE (m) | FDE (m) |
  |---|---:|---|---:|---:|---:|
  | Straight | 1786194922.35 | 37.629620, 127.076546 | −62.2 | 1.006 | 0.955 |
  | Curve    | 1786194897.35 | 37.629700, 127.076492 | −90.0 | 1.458 | 2.539 |
  | Turn     | 1786194956.35 | 37.629471, 127.076660 | −32.3 | 0.736 | 1.067 |

  (전체 수치는 `scripts/results.json`에 그대로 저장 — heading_deg, route_bearing_deg,
  map_rotation_deg, heading_route_diff_deg, ctx_timestamps, pred_xy_m 8개, gt_xy_m 8개
  전부 포함. Curve 시나리오가 item A/B의 "detailed sample"로도 재사용됨.)

## 파일

- `heading_up_process_curve.png` — item A: North-up → ego+heading → rotation → 최종
  heading-up → 96×96 모델 입력, 5단계
- `model_input_curve.png` — item B: 실제 6프레임 카메라 context + 최종 heading-up 지도
- `trajectory_comparison_{straight,curve,turn}.png` — item C/D: 6프레임+지도 →
  실제 checkpoint 예측 vs GT, 3개 시나리오
- `scripts/build_viz.py` — 공용 함수(지도 렌더링/추론/GT 계산), 전부 실제 코드 import
- `scripts/run_all.py` — 3개 시나리오에 대해 다운로드+렌더링+추론 실행, `results.json` 저장
- `scripts/compose_figures.py` — 위 PNG 5개를 조합
- `scripts/analyze_track.py` — 시나리오(straight/curve/turn) 후보 탐색에 쓴 스크립트
- `scripts/results.json` — 이번 생성에 실제로 사용된 모든 수치 (재검증용)

## 재사용한 실제 코드 (레딩: "재구현 아님")

| 기능 | 실제 함수 (직접 import) |
|---|---|
| 타일 캔버스 빌드 | `osm_map_generator.build_canvas()` |
| heading-up 회전/렌더링 | `osm_map_generator.render_frame()` (rot_deg=90−heading_deg, 단일 warpAffine) |
| 미래 경로 densify | `osm_map_generator._densify_route()` |
| 지도 생성/오버레이 | `LiveMapBuilder.get_map_image()`, `.draw_predicted_trajectory()`, `.route_bearing_rad()`, `._closest_route_idx()` — `set_goal()`(OSRM 네트워크 호출)만 건너뛰고 `_route_latlon`/`_canvas`를 직접 주입(아래 참고), 나머지 메서드는 전부 무수정 |
| heading 추정 | `omnivla_edge_deploy.estimate_heading_from_track()` |
| 모델 상수 | `omnivla_edge_deploy.MODEL_PARAMS/IMG_MEAN/IMG_STD/METRIC_WAYPOINT_SPACING/WAYPOINT_STRIDE_SEC/N_CTX/CTX_STRIDE_SEC` |
| 모델 클래스 | `model_omnivla_edge_odom.OmniVLA_edge_odom` (실제 checkpoint 로드) |

실제로 **새로 작성한 코드**는 다음 두 가지뿐이고, 둘 다 "판단 로직"이 아니라
"기존 렌더링을 다른 시점에서 다시 보여주는 시각화 전용" 코드다:
1. `prewarp_northup_crop()` — item A의 "회전 전" 패널(A,B)을 보여주기 위해,
   `render_frame()`이 내부적으로 warp 전에 그리는 것과 **동일한** past/future 선
   드로잉 코드(15줄 내외)를 캔버스에 다시 적용하고 자른 것. **rotation/rescale/anchor
   수학 자체는 여기 없음** — 그건 항상 `render_frame()`을 직접 호출해서 얻음(패널 D).
   ego 마커 + heading 화살표(패널 B)는 프로덕션에 없는 디버그 주석용 오버레이.
2. `compass_panel()` — 패널 C(회전각 수치 표시)는 프로덕션에 없는 개념도. 실제
   회전은 `render_frame()` 내부에서 rescale/anchor와 함께 단일 `warpAffine`으로
   처리되어 "회전만 적용된" 중간 이미지가 존재하지 않기 때문에, 텍스트+간단한 나침반
   그림으로 `rot_deg` 값만 표시함 (README/그림 캡션에 그렇게 명시).

## Ground truth 산출 방법 (중요 caveat)

이 raw 세션에는 학습 때 쓰는 EKF-filtered `filtered_position`/`filtered_heading`이
없다. GT future waypoint는 `gps_imu.jsonl`(~1Hz)을 시간 기준 **선형보간**해서
`WAYPOINT_STRIDE_SEC=0.7`초 간격 미래 위치를 구하고, `rides11_dataset.py::_get_waypoints()`와
동일한 회전 공식(`x=dE·cosθ+dN·sinθ, y=−dE·sinθ+dN·cosθ`)으로 ego-frame 변환한 것이다.
학습 시 실제 GT(EKF-filtered, 고정밀)보다 노이즈가 크므로, ADE/FDE 수치는
"체크포인트가 절대적으로 얼마나 정확한가"의 엄밀한 벤치마크가 아니라 "예측이 실제
경로 방향과 정성적으로 맞는가"를 보여주는 용도로 해석해야 한다.

## 시나리오 선정 방법

`scripts/analyze_track.py`가 `estimate_heading_from_track()`(실제 배포 함수)을 1초
간격으로 128초 전체에 적용해 heading을 구하고, 3초 앞선 시점과의 heading 차이(Δθ)로
`|Δθ|<10°→STRAIGHT`, `10–35°→CURVE`, `>35°→TURN`로 분류했다. 각 클래스에서 여러 틱이
연속으로 같은 분류를 유지하는("애매하지 않은") 구간의 중간 지점을 대표 샘플로 선택함
(예: TURN은 1786194954~958 사이 5개 틱이 연속으로 TURN이었던 구간의 중앙).
이 128초 세션 하나에서는 세 클래스가 실제로 명확히 구분되는 구간이 존재했음
(억지로 만든 것 아님 — 원본 로그: `analyze_track.py` 실행 결과 전체 참고).

## 검증: forward 방향이 이미지 위쪽과 정렬되는가

`render_frame()`의 최종 heading-up 출력에서, ego 앵커(anchor_px_y≈165.9/224,
`map_range_m/(map_range_m·(1+REAR_RATIO))` 비율) 근처(60px 이내) 빨간 미래-경로
픽셀의 **행(row) 위치가 앵커보다 작은(=이미지 위쪽) 비율**을 계산:

```
straight: n_red_near=337  frac_above_anchor=1.00
curve:    n_red_near=370  frac_above_anchor=1.00
turn:     n_red_near=349  frac_above_anchor=1.00
```

3개 시나리오 전부 100% — forward 방향이 이미지 상단과 정렬됨을 코드(rot_deg 공식)와
실제 렌더링 결과 양쪽에서 확인.

## 추가 검증: training vs deployment preprocessing 비교

(요청받은 대로 시각화 제작 **전에** 실제 코드를 대조한 결과 — 세션 대화 로그 참고)

| 항목 | 결과 |
|---|---|
| map rotation 공식 | `build_live_map.py`가 `osm_map_generator.render_frame`을 **직접 import** — 코드 중복 없이 완전 동일 |
| map range 의미/해상도 | 동일 함수이므로 자동 동일 (224px 렌더 → 96×96) |
| camera context 순서/개수 | 학습 `[fi-15..fi-3,fi]`, 배포 `frame_buffer` deque append — 둘 다 과거→현재, N_CTX=5+1=6 |
| waypoint 좌표계/공식 | 학습 `_get_waypoints()`와 배포 `waypoint_to_control()`이 같은 회전 공식(x=forward,y=left) 사용 |
| waypoint stride | `rides11_dataset.py WAYPOINT_STRIDE=7` ↔ 배포 `WAYPOINT_STRIDE_SEC=0.7` — 값은 일치하나, **checkpoint 파일(.pth) 자체엔 학습 당시 stride가 기록돼 있지 않아 checkpoint 하나만으로는 기계적으로 재검증 불가**(디렉토리명 날짜로 정황 추정 — 별도 트레이싱 갭, 이번 라운드에서 고치지 않음) |
| **heading 소스** | **불일치**: 학습(rides_11 arrow)은 FrodoBots 자체 EKF가 만든 `filtered_heading`(이 repo에 재현 코드 없음)을 사용. 배포는 `estimate_heading_from_track()`(raw GPS 누적-경로장 기반 robust atan2)을 사용. **축 정의(East=0/North=+90 CCW)는 동일하지만 추정 알고리즘 자체가 다름.** 이 문서의 모든 heading 값은 배포가 실제로 쓰는 `estimate_heading_from_track()`으로 계산한 것이므로, "배포 파이프라인이 실제로 무엇을 모델에 넣는지"의 정확한 재현이지 "학습 때 정확히 이 값을 봤는지"의 보장은 아님. |

## 재생성

```bash
conda activate frodobot
export CARTO_API_KEY="cb1_2yj0_1_ffcc69174af013089c6c5da6"
cd ~/Last-mile-navigation
python3 docs/deployment_visualization/05_model_input_output/scripts/run_all.py
python3 docs/deployment_visualization/05_model_input_output/scripts/compose_figures.py
```

`_cache/`(다운로드한 raw 데이터셋/타일/렌더링 중간산출물)는 `.gitignore` 처리됨 —
재실행 시 HuggingFace에서 다시 받아옴(카메라 프레임 18장 + gps/control jsonl만,
용량 작음).

**실제 FrodoBot 연결/제어 명령 없음 — 전부 offline (저장된 데이터셋 + 저장된
checkpoint + 로컬 CARTO 타일 렌더링).**
