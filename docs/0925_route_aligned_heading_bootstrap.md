# 2026-09-25 (2): Route-aligned initial heading bootstrap + ARM/GO LIVE 재설계

지난 실외 배포 실패의 근본 원인을 재분석하고, 코드 레벨로 고친 세션. 실제 FrodoBot
연결/이동 테스트는 하지 않았고, 전부 offline/mock(`deployment/offline_smoke_test.py`)
으로 검증했다.

## 1. 근본 원인 재확인

```
deployment 시작 → frame_buffer가 ~1.5초 만에 준비(카메라만 있으면 됨, 이동 불필요)
  → 그 시점 GPS 궤적(past_track)은 아직 부족(로봇이 안 움직였으므로)
  → estimate_heading_from_track()이 None → imu_fallback 사용
  → 불안정한 IMU heading으로 heading-up 지도 회전
  → 모델에 잘못된 지도 입력 → 잘못된 trajectory 예측
  → (그 세션은 LIVE였으므로) 즉시 non-zero 명령 전송
```

코드로 재확인한 핵심 사실 두 가지:
- **frame_buffer 게이트는 heading 품질과 무관**하다 — 카메라만 있으면 이동 없이도 찬다.
- **GPS course heading은 실제 이동(누적 1.5m)이 있어야만 생긴다** — 가만히 서서
  GPS fix를 아무리 오래 기다려도 생기지 않는 진짜 bootstrap deadlock이 있었다.
- (이전 세션에 이미 발견) `--dry_run`→재시작으로 LIVE 전환하면 `frame_buffer`/
  `past_track`/heading EMA/route가 전부 초기화되어, 힘들게 dry-run으로 검증해도
  LIVE는 매번 새로 콜드스타트하는 구조였다.

## 2. 설계: Route-aligned initial heading (사용자가 물리적으로 정렬한다는 전제)

이번 실험 조건(출발 전 로봇을 실제 route 방향으로 물리적으로 정렬해둠)에 한해서만
쓰는 부트스트랩. 일반적으로 `route_bearing == robot_heading`을 자동 가정하지 않고,
**대시보드에서 사용자가 명시적으로 확인해야만** 값이 확정된다.

핵심 결정 — **`_heading_ema_vec`는 절대 이 값으로 시드하지 않는다.** 대신 heading
분기의 우선순위를 다음처럼 확장했다(`omnivla_edge_deploy.py::step()`):

```
1. 실측 GPS course heading 있음        → gps_track  (EMA 하드셋/블렌드, 기존 로직 그대로)
2. 없지만 EMA가 이미 실측으로 확립됨    → gps_ema_hold / gps_ema_gyro (기존 로직 그대로)
3. 없고 EMA도 없지만 route-align 확정됨 → route_aligned (NEW — 고정값, EMA 안 건드림)
4. 전부 없음                           → imu_fallback (최후 폴백, 코드에 남겨둠)
```

EMA를 순수하게 유지하는 이유: 실측이 처음 들어오는 순간에도 `_heading_ema_vec`가
여전히 `None`이므로 기존 "최초엔 하드셋(스무딩 없음)" 로직이 오염 없이 그대로
작동한다 — 사람이 정렬을 잘못했어도 그 오차가 실측 평균에 섞여 남지 않는다. 대가는
`route_aligned → gps_track` 전환 순간 스무딩 없는 순간적 값 변화가 있을 수 있다는
것 — 이번 라운드에서는 별도로 완화하지 않고, 대신 그 전환을 **정확히 1회, 명시적으로
로그**한다(`event: "heading_source_transition"`, 이전값/새값/차이/tick_id 포함).

### Lookahead 분리

기존 `route_bearing_rad(lookahead_m=5.0)`은 디버그용(`heading_route_diff_deg`)으로
그대로 두고, initial heading 전용으로 `INITIAL_HEADING_LOOKAHEAD_M = 2.0`을 신설했다.
근거: route는 1m 간격으로 densify됨(`_densify_route`) — 1m(세그먼트 1개)는 OSRM
폴리라인 정점 단위 잡음에 취약하고, 5m는 출발부가 바로 꺾이는 경로에서 "지금 서야 할
방향"이 아니라 "5m 뒤 방향"을 대표해버린다. 2m(세그먼트 2개 평균)로 절충.

## 3. ARM / GO LIVE 2단계 + 같은 프로세스에서 DRY→LIVE 전환

`self.dry_run`(불리언) 대신 `self.control_stage ∈ {"DRY_RUN", "ARMED", "LIVE"}` 상태
머신을 도입. `--dry_run` CLI 플래그는 의미가 바뀌어 **"이 프로세스에서 GO LIVE를
영구 잠금"**이 되었다(순수 검증 세션용, 대시보드에서 눌러도 거부됨). 플래그 없이
실행해도 항상 `DRY_RUN`으로 시작하고, 대시보드의 세 버튼(정렬 확인 → ARM → GO LIVE,
GO LIVE는 confirm 다이얼로그 포함)을 순서대로 눌러야만 `LIVE`가 된다.

**같은 프로세스, 재시작 없음** — `frame_buffer`/`past_track`/heading EMA/route 캐시가
전환 내내 그대로 유지된다. 구현: HTTP 핸들러 스레드는 명령 문자열(`confirm_alignment`/
`arm`/`go_live`/`abort`)을 `queue.Queue`에 넣기만 하고, 실제 상태 변경은 전부 제어
루프 스레드가 `step()` 맨 앞(`_apply_pending_commands()`)에서 큐를 비우며 순차 처리한다
— 복잡한 객체(`map_builder`, route 캐시 등)에 락을 걸 필요 없이 단일 소비자로 동시성
문제를 원천 차단.

부가 안전장치: `ARMED` 상태로 30초(`ARM_TIMEOUT_S`) 안에 GO LIVE를 안 누르면 자동으로
`DRY_RUN`으로 복귀(재-ARM 필요). `dry_run_lock`이면 `go_live` 명령 자체가 거부됨.
route-align이 확정 안 된 채로는 `arm`조차 거부됨(1차 방어선). LIVE 상태여도
`map_heading_source == "imu_fallback"`이면 `run()`의 전송 직전 게이트가 강제로
`(0,0)`을 보낸다(2차 방어선 — 정상 흐름에서는 걸릴 일이 없어야 함).

## 4. 대시보드 확장

- North-up 경로 미리보기 패널 신설(`/map_northup.jpg`, `LiveMapBuilder.get_northup_preview_image()`)
  — heading-up 회전 **전** 상태를 보여줘서 "route geometry 자체 문제"(A)와
  "heading-up 회전 문제"(B)를 분리 확인 가능. 회전 수학은 재구현하지 않고, 항상
  `render_frame()`이 만든 최종 결과(B)와 별개로 캐시된 캔버스에 past/future 선만
  다시 그림.
- `map_heading_source`를 색상 구분(초록=gps_track/hold, 주황=route_aligned, 빨강=imu_fallback)
  해서 노출 — 이전엔 로그에만 있고 대시보드에는 전혀 안 보이던 값.
- GPS heading 준비도(`accumulated_path_m` / `net_displacement_m`, 기준 1.5m/0.3m 같이 표시).
- OSRM 정상/fallback 여부, 시작/목표 snap 거리 — 이전엔 콘솔 print에서만 보이고
  사라지던 값.
- ARM/GO LIVE 버튼은 `control_stage`에 따라 자동으로 활성/비활성화(JS).

## 5. Offline/mock 검증 결과 (`deployment/offline_smoke_test.py`)

세 시나리오 전부 통과:
1. **dry_run 잠금**: confirm/arm/go_live 다 눌러도 `control_stage`가 절대 LIVE 안 됨, 전송값 항상 (0,0).
2. **전체 워크플로**: fix 없음 → route 초기화(OSRM 없어서 fallback=True로 정상 동작,
   frame_buffer 준비와 무관하게 첫 유효 fix 직후 초기화됨 확인) → 정렬 확인
   (heading=-127.5°) → ARM → GO LIVE, **같은 프로세스에서** 전환. GO LIVE 전엔 GPS가
   바뀌어도 전송 항상 (0,0). GO LIVE 직후 `route_aligned`로 이동 시작(imu_fallback
   아님, non-zero 명령 정상 전송 확인) → 실측 이동 누적 후 `gps_track`으로 정확히
   1회 전환(diff=+15.9°, 전환 로그의 route_aligned 값이 정렬 확인 시점 값과 완전히
   일치 — **EMA 오염 없음** 확인) → 이후 `gps_track`/`gps_ema_hold`만 유지,
   `imu_fallback` 재등장 없음. North-up 미리보기 생성 확인.
3. **정렬 없이 ARM 시도**: 거부되고 `DRY_RUN` 유지 확인.

대시보드 HTTP 계층(페이지 렌더링/`status.json`/North-up 이미지/POST 명령 큐잉/잘못된
명령 400 거부)도 별도로 기동해서 확인함.

## 6. 여전히 남은 것 (이번 라운드 범위 밖)

- `step()`의 GPS fix 게이트는 여전히 `lat==1000`만 확인 — `fix_quality` 자체를 하드
  게이트로 쓰지는 않음(대시보드에 표시만 됨, 판단은 사람이 함).
- `route_aligned → gps_track` 전환 순간의 순간적 heading 점프에 대한 스무딩은 넣지
  않음(의도적 — 사용자 요청).
- 정지 GPS 전용 진단 스크립트(로봇을 움직이지 않고 20~30초 `/data`만 polling)는
  설계만 하고 별도 스크립트로 구현하지는 않음 — 필요시 다음 세션에서.

## 실행 방법 (요약, 상세 체크리스트는 `docs/0925_outdoor_test_procedure.md`)

```bash
python3 deployment/omnivla_edge_deploy.py \
    --ckpt checkpoints/omnivla_edge_rides11_odom_20m_20260910/best.pth \
    --map_range 20 --goal_lat <목표위도> --goal_lon <목표경도>
# 재시작 불필요 — 대시보드에서 정렬 확인 → ARM → GO LIVE
```

재검증(offline): `python3 deployment/offline_smoke_test.py`
