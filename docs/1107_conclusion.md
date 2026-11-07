# 2026-11-07 결론 — "되는 곳에서 직진 데모로 고정"

브랜치 `field-test-1107-direct`. 11/06~11/07 실기기 배포 테스트(실주행 20개 run) 전수 분석 후
내린 결론과 그 근거. 상세 run 분석/영상은 아티팩트("FrodoBot 실주행 로그")와
`deployment/logs/1011/`, run별 수치는 세션 대화 참고.

## 한 줄 결론

**모델·로봇·배터리는 정상이고 직진은 잘 된다. 실패의 공통 원인은 "지도 heading/위치 추정이
로봇 현실과 어긋나는 것"이며, 이는 휴리스틱으로 못 고쳤다. → 캡스톤 데모는 되는 조건(보도가
OSM에 있고 GPS가 안정적인 직선 구간)에서 직진 주행으로 고정한다: 닭도리로 → 왕돈까스로.**

---

## 사용자가 내린 결론 (원문 요약)

1. 명령이 들어가면 `|조향| ≤ 0.05`인 작은 조향값은 펌웨어가 무시 → 실외 gain test로 처리 완료.
2. 직진 주행은 잘 하는데, last mile(목적지 근처)로 가면 heading 불안정 → map 불안정 →
   로봇이 양옆으로 흔들림.
3. 차라리 map을 고정하고 고정 trajectory만 생성하거나, 그냥 직진만 하게 하는 게 나을 듯.
4. OSM에 없는 보도(경춘선 숲길)에서 테스트하면, map이 "최대한 가까운 보도"로 trajectory를
   뽑아내서 경로가 이상하게 생성됨.
5. GPS 값이 불안정해서 localization에 문제가 생김.
6. 휴리스틱을 범벅해도 문제가 해결되지 않는다.
7. 그래서 test 자체를 잘 되는 곳에서만 진행하는 게 맞다. **닭도리로 → 왕돈까스로 직진주행으로 fix.**

---

## 근거 검증 — 이 결론이 맞는가? (대체로 맞음)

### ① 작은 조향(≤0.05) 무시 — ✅ 맞음
- 실내 테스트 A(`docs/1101_report.md`)에서 확인: 펌웨어는 `|angular| ≤ 0.05`를 무시(직진),
  0.06부터 회전속도 ≈ 1.66 rad/s × 명령으로 점프. → 작은 조향은 물리적으로 안 먹힌다(사실).
- 실주행 로그에서도 반복 확인: 155723/153516에서 명령 평균 각속도 −0.02~−0.04(데드밴드 안)라
  로봇이 그냥 직진함.
- 실외 gain 보정이 끝났다는 건 **사용자 실측 결과**(이 레포 로그로는 재확인 안 함). 보정 로직은
  `--enable_actuation_calibration`에 구현돼 있음.

### ② 직진 OK / last mile heading·map 불안정 → 좌우 흔들림 — ✅ 맞음
- 직진 성공 근거: 142414(goal 도달), 151546(goal 도달), 145950(6m 접근), 153516(21m 방향 정확),
  155723(55m 방향 정확) — direct 경로 + 목표 정면 정렬이면 완주/정확.
- 흔들림 근거(heading 폭주): 155723은 55m 직진 후 **막판 자이로 스파이크(±100~138°/s)**로
  heading이 20°→134° 폭주 → 우회전(−0.33) 9회. 150032는 heading 61~164° 폭주, 153059는
  −180~180° 요동. gyro 모드엔 GPS 교정이 없어 한 번 틀어지면 복구 불가.
- "last mile(목적지 근처)"에서 특히 나쁜 이유: 목표 근처에서 속도가 줄면 GPS 양자화 잡음이
  heading 추정을 지배 → 매 틱 지도 회전 → 예측도 흔들림. 추가로 27m 체크포인트의
  **near-goal 좌편향**(화면 속 경로 선분 픽셀 길이 bbox_h가 짧아지면 좌회전, `experiment_log.md`
  §1-8)도 겹침. → 두 효과가 last mile에서 합쳐짐(사실).

### ③ map 고정 / 고정 trajectory / 직진만 — ✅ 합리적 (데모 전략으로)
- 직선 구간에서는 `--route_mode direct --heading_mode route_bearing`이 heading을 매 틱
  "현재 위치→경로 앞점"으로 새로 계산해 **자이로 폭주가 구조적으로 불가능**하고 목표 방향으로
  안정 유지됨(155723 분석의 결론). "map 고정"의 실질적 구현에 가장 가깝다.
- 단, 이것은 **문제 해결이 아니라 실패 요인 회피**다. 회전·원거리·보도 끊김은 여전히 못 함.

### ④ OSM에 없는 보도(경춘선 숲길) → 가까운 보도로 스냅 → 이상한 경로 — ✅ 맞음 (직접 확인)
- 이번 세션에서 직접 재현: 로봇이 보도 위(snap 0.6m)여도 이 지역 OSM footway 망이 조각나
  있어서, 남동쪽 목표를 찍으면 OSRM이 **913m 뒤로 도는 루프 경로**를 뽑음. 끊긴 보도/누락
  보도에서는 nearest-snap이 엉뚱한 세그먼트로 붙어 경로가 깨진다(사실).
- 경춘선 숲길처럼 OSM에 보도 자체가 없으면 더 심함 → 그 구역은 테스트 부적합.

### ⑤ GPS 불안정 → localization 문제 — ✅ 맞음
- 11/01 센서 감사: GPS 동결이 주행 틱의 중앙값 75%, 출발 직후 GPS가 50~84m 튀는 run도 있었음
  (`102751`,`135025`). 위치 자체가 불안정하니 그 위에서 계산하는 heading/경로도 불안정.
- 상보필터(`pose_filter.py`)가 오프라인 평가에선 best였지만 **실기기 미검증**이고, 이것도
  근본 GPS 품질 문제를 완전히 없애진 못함.

### ⑥ 휴리스틱 범벅해도 해결 안 됨 — ✅ 맞음
- 지금까지 쌓은 휴리스틱: 트림(−0.051)/디더링/`angular_right_gain`, near-goal override,
  straight-segment override, off-route safety, GPS freeze guard, heading rate-limit,
  route_bearing anchored pull… 전부 증상 완화일 뿐, 회전/heading 근본 문제는 10/04·10/11·11/01
  내내 재현됨. 실내 테스트 A 이후 **트림·디더링·right_gain은 원리를 잘못 안 값이라 폐기**가 결론.

### ⑦ 되는 곳에서만 / 닭도리로→왕돈까스로 직진 fix — ✅ 데모 결정으로 타당
- 캡스톤 데모 목적(로봇이 경로 따라 목적지 도달을 보여주기)에는, 실패 요인(회전·보도 끊김·
  GPS 튐)이 없는 **OSM에 보도가 있는 직선 구간 + 목표 정면 정렬**이 성공 재현성이 가장 높다.
- **단, 이것은 "last-mile navigation을 풀었다"가 아니라 "데모 범위를 되는 조건으로 좁혔다"**임을
  명확히 할 것. 일반 주행(회전 포함)은 재학습 트랙의 과제로 남는다.

---

## 데모 확정 — 닭도리로 → 왕돈까스로 직진

### 출발 전 반드시 확인(경로가 깨끗한지 검증)
```bash
# 로봇을 출발점에 세운 뒤, 목표(왕돈까스로 쪽) 좌표로:
python3 deployment/analysis/check_robot_placement.py --goal <목표위도> <목표경도>
# 기준: "보도까지 ≤3m(보도 위)" + "총 회전 거의 직선" + "반대-스텁 경고 없음"
# 브라우저 실시간 확인:
python3 deployment/analysis/robot_live_map.py --goal <목표위도> <목표경도>  # http://127.0.0.1:8090
```

### 주행 명령(직선·heading 폭주 방지·대시보드 지연 제거)
```bash
python3 deployment/omnivla_edge_deploy.py \
  --ckpt checkpoints/omnivla_edge_rides11_odom_20m_20260910/best.pth \
  --map_range 27 --goal_lat <목표위도> --goal_lon <목표경도> \
  --route_mode direct --heading_mode route_bearing --waypoint_target_step 5 \
  --near_goal_override_dist_m 0 --disable_gps_freeze_guard \
  --cross_track_gain 0 --cross_track_gain_i 0 \
  --cross_track_blend_gain 0 --cross_track_blend_gain_i 0 \
  --enable_actuation_calibration --actuation_mode offset \
  --goal_reach_threshold_m 5.0 \
  --no-record-dashboard
```
- `--heading_mode route_bearing`: 자이로 폭주(155723) 회피, 직선 구간에서 목표 방향 안정 유지.
- `--no-record-dashboard`: 대시보드 캡처가 SDK와 CPU 경합해 제어 송신을 최대 4.3초(153516),
  최대 13.7초(155723) 지연시키던 문제 제거(라이브 대시보드 8080은 유지).
- 로봇을 **경로 첫 세그먼트 방향**으로 정면 정렬한 뒤 GO LIVE.

---

## 데모로 "덮은" 미해결 문제 = 재학습/후속 트랙 (별도)

데모를 되는 조건으로 좁힌 것이지 아래가 풀린 건 아니다. 상세는 `docs/1101_retraining_plan.md`.

1. **heading/localization**: GPS 품질(동결·튐)과 heading 추정이 근본 문제. 자이로는 스파이크로
   폭주, GPS-track은 양자화 잡음. 상보필터(`pose_filter.py`) 실기기 검증 필요. 배포에 새 heading
   모드로 넣을지 결정.
2. **회전 구간**: 모델이 급커브를 못 도는 문제(10/04 재현). 재학습 시 회전 구간 가중치 +
   좌우반전 증강 + 직진/회전 분리 평가.
3. **지도 입력 일치**: 학습(GT 미래경로)과 배포(OSRM)의 경로가 5~10m 어긋남 → 경로 섭동/OSRM
   섞기 증강. OSM 보도 누락·끊김(경춘선 숲길 등) 지역은 애초에 데이터·배포 모두 부적합.
4. **near-goal 좌편향**: bbox_h 기반 결정론적 좌회전(27m 체크포인트). map_range/geometry 재확정.

## 다음 세션 할 일

1. 닭도리로→왕돈까스로 **출발/목표 좌표 확정** + `check_robot_placement.py`로 경로 검증.
2. 위 명령으로 실외 직진 데모 1~2회, `route_bearing` heading이 끝까지 안정한지 확인.
3. (데모 성공 후) 재학습 트랙 착수 — `docs/1101_retraining_plan.md`.
