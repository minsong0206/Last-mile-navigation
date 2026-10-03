# 2026-09-26 실외 배포 세션 — unity/semantic-nav 작업용 인수인계 문서

이 문서는 `main` 레포(`~/Last-mile-navigation`)에서 오늘(2026-09-26) 진행한 실외 배포
테스트 세션의 전체 내용을 정리한 것이다. **목적**: `unity/Last-mile-navigation_merge`
클론에서 semantic navigation(`forodobot_semantic_nav`) 작업을 이어가는 Claude가
이 문서만 읽고도 오늘 무엇이 바뀌었고, 왜 바뀌었고, 아직 뭐가 안 고쳐졌는지 파악할
수 있게 하는 것. 관련 커밋: **`a4006ff`** (`main` 브랜치 HEAD, "Add goal-reached
latch, heading_mode=route_aligned_fixed, and near-goal reroute suppression").

같은 GitHub remote(`origin/minsong0206/Last-mile-navigation`)를 보고 있으므로,
`unity/Last-mile-navigation_merge`에서 `git fetch && git pull`(또는 `git merge
origin/main`)로 이 커밋을 그대로 받아올 수 있다 — 코드 자체를 복붙할 필요 없음.

## 1. 오늘 세션에서 실제로 바뀐 것 (커밋 `a4006ff`)

변경 파일: `deployment/omnivla_edge_deploy.py`, `deployment/debug_web.py`,
`deployment/offline_smoke_test.py`

### 1-1. `--heading_mode {auto, route_aligned_fixed}` (신규)

- **`auto`**(기본, 기존 동작): route-aligned heading은 실측 GPS course heading이
  확보되기 전까지의 임시값. 확보되는 즉시 `gps_track`/`gps_ema_hold`로 자동 전환됨.
- **`route_aligned_fixed`**(신규): 정렬 확인으로 고정한 heading을 **런 내내 그대로
  사용**, 이후 GPS 궤적이 얼마나 잡히든 절대 안 넘어감. `_heading_ema_vec`는 이
  모드에서 `route_aligned_fixed`가 활성일 때 아예 안 건드림(`step()`의 분기
  우선순위 최상단에서 체크).

**도입 이유**: 실외 테스트 중 `auto` 모드에서 로봇이 저속/정지 상태로 오래
머무르면, 실제 이동이 아니라 **GPS 수신 잡음(제자리 지그재그)만으로**
`estimate_heading_from_track()`의 최소 이동거리(1.5m) 임계값을 우연히 넘겨서
엉뚱한 방향이 "실측 heading"으로 채택되는 문제가 실측됨(heading이 tick마다
30~40° 이상 흔들림 — `gps_accumulated_path_m`은 쌓이는데 `gps_net_displacement_m`은
그보다 훨씬 작은 지그재그 패턴으로 구분 가능).

### 1-2. `--goal_reach_threshold_m`(기본 2.0m) — 목표 도착 시 영구 정지 (신규)

- 매 tick `latlon_distance_m(lat, lon, goal_lat, goal_lon)`로 직선거리 계산 →
  임계값 이내면 `self._goal_reached = True`로 **래치**(한 번 켜지면 이후 거리가
  다시 늘어나도 절대 안 풀림).
- `run()`의 전송 게이트: `is_live and heading_trustworthy and not self._goal_reached`
  — 세 조건 다 만족해야 non-zero 전송. 계산(`computed_linear/angular`)은 도착 후에도
  계속 로그/대시보드에 남음(전송만 막힘, 관찰은 계속 가능).
- **도입 이유**: 이전엔 목표에 도달해도 정지 로직이 아예 없어서 계속 명령이
  나갔음(실제 로그로 확인된 갭).
- **실외 재현 검증됨**: `deploy_20260926_...` 로그에서 `dist=1.84m`에 도착 감지 →
  이후 전송 (0,0) 유지 확인.

### 1-3. `REROUTE_DISABLE_NEAR_GOAL_M = 10.0` — 목표 근처 재라우팅 비활성화 (신규)

- 목표까지 직선거리 10m 이내에서는 `is_off_route()`가 True여도 재라우팅 안 함
  (최초 `route_init`은 예외 없이 항상 수행됨).
- **도입 이유**: 목표 ~5.5m 앞에서 발생한 reroute가 OSRM 보행로망을 따라
  **헤어핀(왔다갔다) 형태의 새 경로**를 만들어서, 로봇이 이미 가까운데도 계획
  경로상으로는 더 멀어지는 구간이 생겨 "도착 못 하고 계속 도는" 것처럼 보이는
  문제가 실측됨(`deploy_20260926_132829.jsonl`의 `reroute` 이벤트, 그 안의
  `route_latlon` 좌표를 순서대로 따라가 보면 실제로 한번 멀어졌다가 되돌아오는
  형태였음 — 대시보드 North-up 패널 스크린샷으로도 육안 확인).

### 1-4. `--initial_heading_lookahead_m`(기본 2.0m) — 정렬 확인 lookahead 거리 조절 가능 (신규)

- `_confirm_route_alignment()`가 route tangent를 계산할 때 쓰는 lookahead 거리를
  CLI로 조절 가능하게 노출.
- **도입 이유 (⚠ 아직 완전히 안 고쳐진 이슈, 아래 3-1 참고)**: 목표 근처(예:
  `dist_to_goal≈6.9m`)에서 정렬 확인을 눌렀을 때, 짧은 2m lookahead가 마침 경로가
  꺾이는 지점 근처를 잡아서 **실제 진행 방향(`route_bearing_deg`)과 56~58°나
  어긋난 heading이 그대로 고정**되는 사고가 실측됨(`deploy_20260926_135303.jsonl`,
  `smoothed_heading=40.3°` vs `route_bearing=-15.7°~-17.5°` — `route_aligned_fixed`라
  전체 주행 내내 이 큰 오차가 유지되며 모델이 계속 큰 보정 명령(`angular` -0.12~-0.15,
  안전 한계 이내지만 상당히 큼)을 냄). 지금은 이 값을 5~10m로 늘려서 **회피만**
  가능하고, 근본적인 자동 검증은 아직 없음.

### 1-5. `_confirm_route_alignment()`의 GPS heading EMA/이력 오염 방지 (버그 수정)

- "정렬 확인"을 누르는 순간 `self._heading_ema_vec = None` + `self.past_track.clear()`.
- **버그였던 이유**: 원래는 `_heading_ema_vec`를 절대 안 건드리는 설계였는데,
  `auto` 모드에서 DRY_RUN으로 오래 관찰하는 동안 위 1-1의 GPS 잡음 문제로
  `_heading_ema_vec`가 이미 "실측(gps_track)"으로 (잘못) 오염된 채 확정돼 있는
  경우가 있었음 — 이러면 heading 분기 우선순위(`gps_track`/`gps_ema_hold`가
  `route_aligned`보다 항상 우선)때문에 정렬 확인을 몇 번을 다시 눌러도 실제로는
  절대 반영이 안 됐음(로그엔 `route_alignment_confirmed` 이벤트가 매번 남는데
  실제 `map_heading_source`는 계속 `gps_track`). `past_track`도 같이 비워야
  하는 이유: 안 비우면 다음 tick에 남아있는 잡음 이력으로 즉시 재오염됨.

### 1-6. 대시보드 이미지 자동 새로고침 버그 수정 (`debug_web.py`)

- 이전 대시보드 재설계(지난 세션) 때 `<meta http-equiv="refresh">`를 지우고 JS
  폴링으로 바꾸면서, **텍스트 상태는 매초 갱신되는데 `<img>` 태그 `src`는
  페이지 첫 로드 시점에 고정된 채 방치**되던 회귀 버그. `poll()`이 이제 매초
  `#imgFrame/#imgNorthup/#imgMap/#imgTraj`의 `src`에 새 타임스탬프를 강제로
  붙여서 재요청하게 함.

### 1-7. 대시보드 신규 표시 필드 (`debug_web.py`)

`heading_mode` 배지, `dist_to_goal_m` + 🏁 도착 표시(상태—Route 표),
`goal_reached` 상태 — 전부 `status.json`/`snapshot_status()`에도 노출됨.

### 1-8. `offline_smoke_test.py` — 6개 테스트로 확장

`test_a_dry_run_lock`, `test_b_full_workflow`, `test_c_arm_without_alignment`,
`test_d_route_aligned_fixed_mode`, `test_e_goal_reached_stop`,
`test_f_initial_heading_lookahead_configurable` — 전부 실제 SDK 연결 없이
mock으로 검증됨(`python3 deployment/offline_smoke_test.py`).

## 2. 오늘 실외 테스트에서 발견한 핵심 현상 — **모델 자체의 좌회전 편향 (미해결, 중요)**

`heading_mode=route_aligned_fixed`로 **heading을 완전히 고정**(대시보드
"heading 변동폭=0.0" 확인됨)한 상태에서도, 전송된 `angular`가 **지속적으로 한쪽
(좌측, 양수)으로 쏠리는 패턴이 두 번의 독립된 실주행에서 재현됨**:

- 1차: 최근 20개 tick 중 좌=18, 우=0, 평균 +0.023 → +0.05까지 증가
- 2차: 최근 15개 tick 중 좌=11 우=2 → 이후 좌=15 우=0, 평균 +0.057

결정적 증거: 한 시점에서 `heading_route_diff_deg ≈ 0.00001°`(로봇 heading이 경로
방향과 **완벽히 일치**하는 순간)였는데도 `pred_xy_m`의 y(왼쪽) 성분이
`[0.08, 0.23, 0.40, 0.57, 0.73, 0.86, 0.97, 1.04]`로 **모델이 스스로 점점 왼쪽으로
휘는 궤적을 예측**하고 있었음. 즉:
- heading 불안정(지도 회전 wobble) 때문이 **아님** (heading 고정된 채로도 재현됨)
- 실제 경로를 따라가려는 정당한 보정도 **아님** (heading-route 차이가 거의 0인
  순간에도 재현됨)
- 대조군: `auto` 모드(heading이 47.6° 흔들리던 구간)에서는 오히려 좌=3 우=6으로
  방향이 뒤섞였음 — heading이 안정적일 때만 이 좌편향이 깨끗하게 드러남.

이건 CLAUDE.md에 이미 기록된 미해결 이슈("실로봇 배포에서 좌회전 편향 관찰됨,
2026-08-10")와 같은 현상이 **heading 소스 불안정성을 제거한 상태에서도 재현된
것** — heading 소스 문제(2026-09-05에 고쳤다고 여겨졌던 것)가 근본 원인이
아니었거나, 원인이 여러 개라는 뜻. **모델 자체의 학습 편향 가능성이 매우 높아짐.**

## 3. 아직 안 고쳐진 것 / 다음에 볼 것

### 3-1. 정렬 확인 시 lookahead 자체 검증(cross-check) — 설계만 하고 미구현

위 1-4의 56° 사고를 근본적으로 막으려면, "정렬 확인" 시점에 짧은 lookahead와
긴(5m) lookahead 값을 **같이 계산해서 비교**하고, 많이 어긋나면(예: 25° 이상)
**그 자리에서 확정을 거부**하는 로직이 필요함(대시보드 `heading_route_diff_deg`가
이미 30° 넘으면 빨간색으로 표시되긴 하지만, 확정 자체를 막지는 않음). 사용자가
"잠시 꺼두고 진행"하기로 해서 1-4의 CLI 옵션만 넣고 이 부분은 보류됨.

### 3-2. 프로세스 크래시 시 정지 명령 미보장 (세션 초반 발견, 미해결)

`poll_frodobot()`의 `requests.get()` 호출들(및 `run()` 워밍업의 재시도 루프)이
transient `ReadTimeout`/연결 에러에 대한 재시도/보호 없이 그대로 예외를 던짐.
실외 테스트 중 **최소 3회** 이 패턴으로 프로세스가 죽었음(SDK 자체는 매번
직후 확인 시 정상이었음 — 일시적 지연으로 추정). 매번 SDK 텔레메트리로 로봇
정지 상태(`speed: 0`)는 확인했지만, **이건 운이 좋았던 것에 가깝고 코드가
보장하는 게 아님** — 크래시 시 `run()`의 `except` 블록이 `send_control(0,0)`을
시도는 하지만, 그 `except` 자체에 안 걸리는 상위 레벨(워밍업 단계)이나 그
`send_control()` 자체가 실패하는 경우는 보호가 안 됨. **제안했던 수정**(일시적
네트워크 에러만 별도로 잡아서 크래시 대신 재시도)은 사용자가 다른 이슈를
우선하느라 아직 미적용.

### 3-3. 대시보드 표시 버그 2건 (발견됨, 미수정)

- `DeploymentState.heading_deg`가 `poll_frodobot()`에서 IMU raw 값으로만 설정되고
  이후 절대 갱신 안 됨 — 대시보드 "최종 사용 heading" 행이 실제 map_heading_rad가
  아니라 계속 IMU 원시값만 보여주는 표시 버그(기능 자체엔 영향 없음, 진짜 값은
  `90 - map_rotation_deg`로 암산 가능).
- `record["gps_heading_deg"]`가 JSONL 로그에는 남지만 `self.state.update(...)`로
  대시보드에 실제로 반영된 적이 없음(`grep`으로 확인) — "GPS 궤적 heading" 행이
  항상 `None`.

### 3-4. 기존부터 있던 미해결 갭 (CLAUDE.md 참고)

- `fix_quality` 하드 게이트 없음(`lat==1000`만 체크).
- OSRM 실패 시 조용히 직선 폴백 후에도 제어 명령이 계속 나가는 구조.

## 4. unity/semantic-nav 작업과의 접점 제안

`unity/Last-mile-navigation_merge`의 `MAP_NAV → STOPPING → SEMANTIC_INIT →
SEMANTIC_NAV` 상태머신은 이 문서의 1-2(`goal_reach_threshold_m`)와 개념적으로
겹치는 부분이 있어 보임 — `SEMANTIC_SWITCH_DISTANCE_M`(그쪽 기본 8.0m)과
`goal_reach_threshold_m`(이쪽 기본 2.0m)이 서로 다른 목적(하나는 "semantic
nav로 전환", 하나는 "완전 정지")이지만 같은 `dist_to_goal_m` 계산 로직을 쓸 수
있으니, `main`을 pull받으면 같이 딸려오는 `latlon_distance_m()` 헬퍼(이 세션에서
신규 추가, `omnivla_edge_deploy.py` 상단)를 재사용하면 될 것 같음. 또한 2번(좌편향)과
3-1/3-2/3-3은 semantic nav 전환 이후에도 MAP_NAV 구간에서는 여전히 유효한
문제이니 참고할 것.
