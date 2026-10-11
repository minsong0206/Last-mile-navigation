# 2026-10-11 — GPS 동결 / dead reckoning / 경로 모드 (Harness 1)

`experiment-dead-reckoning` 브랜치 전용 문서. 같은 날 조향편향(angular 양자화·좌우
비대칭·트림), override 전면폐기, SDK/인프라 실패모드, 전체 CLI 목록은
**`docs/1011.md`(Harness 2)** 참고 — 여기선 중복 없이 GPS 동결 대응과 경로/
waypoint 관련만 다룬다. 코드: 커밋 `9d23fce`(스캐폴딩)·`8512edb`(DR 적분 버그수정).

## 1. GPS 위치 동결 — 전 구간 최대 82%

- 로봇 `speed`는 0.3~1.1인데 GPS `lat/lon`이 여러 틱 동안 안 바뀌는 "동결"이
  이 지역에서 지배적(실측 최대 82% tick). GPS 수신 모듈 갱신주기 < 제어루프 3Hz,
  또는 건물 사이 홀드. 소프트웨어로 GPS 칩을 더 빨리 돌릴 순 없음.
- 동결 중 멈춘 좌표를 그대로 쓰면 `dist_to_goal`(도착판정)/`cross_track`/
  `route_bearing`/지도렌더링이 전부 같이 얼어붙고, 멈춘 cross_track에 적분이 쌓여
  조향이 폭주함(`deploy_20261011_132138.jsonl`: 와인드업으로 우회전 -0.27).
  → 동결 중엔 **다른 신호로 위치를 메꾸는 것**이 유일한 방향 = dead reckoning(§3).

## 2. `--route_mode {osrm,direct}`

- OSRM이 보도망을 따라가느라 직선 90m를 185m 꺾인 경로로 반환(그 지역 보도가
  OSM에 부정확/누락)하는 문제 때문에, start→goal **완벽 직선**을 쓰는 옵션 추가.
  라우팅 자체는 실제 공개 OSRM + `south-korea-latest.osm.pbf`(Geofabrik 실 OSM)
  기반 — 알고리즘이 가짜가 아니라 **데이터 공백**이 원인(OSM 직접 편집이 정석이나
  Geofabrik 미러 갱신에 ~하루라 데모 타임라인엔 안 맞음).
- `direct` 검증: 사용자 좌표로 `max_bend=0.0000m`, route_bearing 전 구간 일정 확인.
- ⚠ 학습 데이터는 항상 실제 보행경로 모양 → 직선이 건물 가로지르면 모델엔 OOD
  입력. "직선이 실제 걸을 수 있는 길과 일치"를 눈으로 확인한 경우에만 사용.

## 3. dead reckoning (`--enable_route_dead_reckoning`, 기본 OFF)

- 동결 중 로봇 `speed`로 "캐싱된 route 위 추정 위치"를 메꿈
  (`LiveMapBuilder.point_ahead_on_route`). 추정 위치(eff_lat/eff_lon)를
  `dist_to_goal`/`cross_track`/`route_bearing`/`predict_waypoints`(지도렌더링)에
  통과시켜, 동결 중에도 거리가 줄고 지도가 경로를 따라 진행 → 멈춘 좌표로 조향이
  폭주하던 문제 완화. raw `lat/lon`은 로깅/past_track/동결감지/최초 route 생성에는
  그대로 유지(동결 감지는 raw 기준, 학습 heading도 raw 궤적 기반이어야 하므로).
- route_mode=direct면 직선, osrm이면 꺾인 경로를 따라 전진. GPS 동결 가드(정지)와
  독립 — DR은 "동결 중에도 계속 간다"는 쪽이라 보통 `--disable_gps_freeze_guard`와
  같이 씀.
- 한계: 동결 중엔 "로봇이 경로 위에 있다"고 가정 → 그 사이 실제 횡방향 이탈은
  못 잡음(GPS가 멈춰서 어차피 감지 불가). 하드웨어 편향은 GPS 살아있는 구간/트림에서.

### 3-1. 적분 버그수정 (커밋 `8512edb`, `deploy_20261011_152028.jsonl`)
- 처음엔 전진거리를 `현재_speed × 총_경과시간`으로 계산 → speed가 매 tick
  출렁이면(0.2↔0.77) 추정위치가 경로 앞뒤로 널뛰고(eff_dist 5.2↔11.9↔2.4) 지도
  빨간경로가 짧아졌다 길어졌다 함(사용자가 "경로 남았는데 짧게 쿼리" 보고한 증상).
  OSRM 재쿼리 아님(route_init 1번뿐).
- 수정: **`speed × dt`를 매 tick 적분(단조증가) + 실제 GPS 갱신 시 0 리셋 + dt
  2초 클램프**. 같은 run speed/ts로 재생해 단조증가(0.1→0.37→…→1.87) 확인. **아직
  실기기 재검증 전.**

## 4. `--waypoint_target_step` (기본 2, 0~7)

- "모델이 급커브를 못 돈다"는 이전 결론 정정(`docs/experiment_log.md` §1-19):
  raw `pred_xy_m`을 보면 모델은 우회전을 예측하고 있었음(wp2=-1.2° vs wp7(4.0m앞)=
  -21.6°). 문제는 `waypoint_to_control`의 `target_step=2`가 아직 거의 안 꺾인
  가장 가까운 지점만 씀. → 더 먼 step(6~7)으로 모델 예측 신호를 끌어쓸 수 있게 노출.

## 5. 브랜치 구조

- `experiment-heading-dev` (`3e9940d`, origin push됨) — **데모 경로** (DR 기본 OFF).
  Harness 2 담당. unity 클론과 머지 이슈 있어 가능하면 이 상태로 고정.
- `experiment-dead-reckoning` (이 브랜치) — **DR 작업 전용.** 앞으로 DR 커밋은 여기만.

## 6. 재검증 커맨드 (DR 포함, 직진 데모)

```bash
python3 deployment/omnivla_edge_deploy.py \
  --ckpt checkpoints/omnivla_edge_rides11_odom_20m_20260910/best.pth --map_range 27 \
  --goal_lat <위도> --goal_lon <경도> \
  --heading_mode route_bearing_anchored --heading_anchor_pull_threshold_deg 999 \
  --waypoint_target_step 6 --near_goal_override_dist_m 0 \
  --enable_straight_segment_override --straight_near_lookahead_m 2.0 \
  --straight_far_lookahead_m 12.0 --straight_angle_threshold_deg 20.0 \
  --cross_track_gain 0 --cross_track_gain_i 0 --cross_track_blend_gain 0 --cross_track_blend_gain_i 0 \
  --angular_trim -0.06 --goal_reach_threshold_m 5.0 \
  --disable_gps_freeze_guard --route_mode direct --enable_route_dead_reckoning
```
확인 포인트: `dr_advance_m` 단조증가 / 지도 빨간경로 안정 / `dist_to_goal_m` 매끄럽게 감소.

(SDK 서버 `/control` 타임아웃 복구법·로봇 오프라인 판별 3징후는 `docs/1011.md` §6 참고.)

## 7. 데이터 아카이브

오늘 로그/영상/raw프레임 전부 HF private 데이터셋:
`minsonganingee/frodobot-deploy-logs-20261011`
(`dashboard/` 대시보드영상, `frames/<run_id>/` 카메라·지도 영상+raw jpg/png,
`logs/` deploy jsonl — 총 9,856 파일).
