# CLAUDE.md

FrodoBots rides_11 데이터로 OmniVLA-Edge(odom3ch 변형)를 파인튜닝해서 FrodoBot Mini 로봇에
배포하는 캡스톤 프로젝트. 이 문서는 이후 세션(또는 협업자)이 빠르게 컨텍스트를 잡기 위한 요약이다.

## ⚠ 무조건 지켜야 하는 규칙

- **RunPod에서 데이터 추출/파인튜닝이 진행 중인 동안, 최소 30분마다 사용자에게 진행상황을 보고할 것.**
  (SSH로 pod 접속해서 `orchestrate_remote.log` / `train.log` / `RUN_STATUS.txt` 확인 후 요약)
  대화가 계속 이어지고 있어서 idle 타이밍이 안 나면 cron 기반 자동 보고가 안 될 수 있으니, 그 경우엔
  사용자가 진행상황을 물어보지 않아도 자연스러운 타이밍에 먼저 보고할 것.
- Pod 과금 방지가 최우선: 크래시/에러 발생 시 스스로 진단·수정 시도하되, 정말 안 풀리면 pod을 stop해서
  과금을 막을 것 (remove는 하지 말 것 — 데이터는 volume에 남겨둘 것). 볼륨 삭제, pod 삭제처럼 되돌리기
  어렵거나 비용이 드는 결정은 사용자에게 먼저 확인받을 것 (2026-09-07 사고 사례 참고 — 아래 이력 섹션).

## 전체 구조 (5단계)

```
Step 1. Episode Selection      osm_pipeline/py/episode_selector.py   → episode_scores.json (538 selected)
Step 2. OSM Map Generation     osm_pipeline/py/osm_map_generator.py  → ego-centric PNG per frame
Step 3. Dataset                osm_pipeline/py/rides11_dataset.py    → PyTorch Dataset
Step 4. Fine-tuning             scripts/omnivla/finetune_omnivla_edge.py → checkpoints/omnivla_edge_rides11_odom{,_12m,_20m}/
Step 5. 배포                    deployment/                          → FrodoBot Mini 실로봇 (README_omnivla_edge.md 참고)
```

시작 체크포인트: `omnivla-edge-odom3ch.pth` (원저자 9채널 체크포인트를 3채널로 평균 변환한 것).
`rides11_finetune.yaml`(9ch 원본)은 현재 3채널 데이터셋과 구조가 안 맞는 stale config — 쓰지 말 것.

## 실행 중요 사항

- **파인튜닝은 GPU1에 하드코딩**되어 있음 (`finetune_omnivla_edge.py`의 `device = torch.device("cuda:1")`).
  `CUDA_VISIBLE_DEVICES` 환경변수와 같이 쓰면 충돌해서 크래시남 (`device_count()=1`인데 `cuda:1` 요청) —
  환경변수 없이 그냥 실행할 것.
- 체크포인트별로 **`MAP_RANGE_M`이 다름**(25m/20m/12m) — 학습에 쓴 값과 추론/배포 시 값이 반드시
  일치해야 함 (안 그러면 학습-추론 스케일 불일치). `deployment/omnivla_edge_deploy.py --map_range`는
  기본값 없이 필수 인자로 만들어져 있음 — 실수 방지용.
  **⚠ 2026-09-05부터 `MAP_RANGE_M`의 의미 자체가 바뀜**: 기존 25/20/12m 체크포인트는 "정중앙 대칭
  half-width"였고, 그 이후 새로 학습하는 체크포인트는 "전방 reach"(로봇이 화면 아래쪽 앵커에 위치,
  후방=전방×`REAR_RATIO`)로 기하가 다름 — 두 세대 체크포인트를 절대 같은 것으로 착각하지 말 것.
  자세한 배경은 `docs/0905.md` 참고.
- **`.gitignore`에 예전에 구멍이 있었음** — `osm_maps_arrow_12m/20m/25m`(각 11~13GB), 루트 `wandb/`,
  `checkpoints/` 등이 안 걸러지고 있었음. push 전에 `git status`로 대용량 미추적 디렉토리 없는지 항상
  확인할 것. 체크포인트(`*.pth`)와 대용량 데이터는 git이 아니라 Hugging Face에 올림
  (계정 `minsonganingee`, private repo).

## ⚠ 2026-10-04 세션 요약 — 다음 세션(다른 컴퓨터 포함) 최우선으로 읽을 것

상세는 반드시 `docs/experiment_log.md` §1-8~§1-18 먼저 읽을 것(이 섹션은 결론만).
이 세션은 Harness 1(학습/모델 분석)과 Harness 2(실배포, 이 세션)가 cross-session
메시지로 계속 교차검증하며 진행함. **git 브랜치 3개로 정리됨**:
- `main` / `stable-gps-freeze-20261004`(`origin`에 둘 다 push됨) — 오늘 아침 커밋
  (`b7d8c9e`, 4개 메커니즘 + GPS 동결 가드 **초기 버전**, 아래 재무장 버그 있는 채로)
  에 고정. 사용자가 "이 지점부터 다시 실험하고 싶다"고 명시적으로 돌아가서 쓴 브랜치.
- `experiment-heading-dev`(`origin`에 push됨, `80c230e`) — 오늘 오후 추가 수정분
  (아래 2번) 전부 포함, 계속 developing 중인 브랜치. **다음 세션은 기본적으로 여기서
  이어갈 것** (단, `stable-gps-freeze-20261004`로 되돌아가 비교 실험을 더 하고 싶다고
  할 수도 있음 — 사용자 지시 확인).

**1) 실기기에서 반복 확인된 결론 — 우회전 실패는 heading_mode/override 설정과
무관하게 재현됨, 모델 자체의 한계로 결론 내림(아직 재학습 전까지는 디플로이 코드로
못 고침):**
- 같은 실제 지점(route_bearing이 `-105°`대에서 `164~178°`대까지 거의 90도 우회전
  요구)에서 **세 가지 완전히 다른 설정**(pull 기본 켬 / `straight_segment_override`만
  끔 / pull까지 완전히 끔(`heading_anchor_pull_threshold_deg=999`) + 위치기반
  `off_route_safety`로 대체)으로 각각 테스트했는데 **전부 우회전 실패**(`143423`,
  `144650`, `145526`(연석 충돌 발생 — 실제 물리적 접촉, 로봇 이상 없음 확인됨),
  `145951`).
- 특히 `145951`은 지도가 100% 깨끗(`route_bearing_anchor`, `diff=0.0`, pull 전혀
  없음)했는데도 모델 raw angular가 ±0.02 수준(필요한 각속도의 1/15)뿐이었음 —
  **지도/heading 노이즈와 무관하게 모델이 이 급커브를 못 돈다는 게 직접 확인됨.**
- 반대로 `130300`(같은 코드, `stable-gps-freeze-20261004`)은 같은 지점에서 pull이
  껴도 몇 틱 만에 `route_bearing_anchor`로 복귀해서 결국 목표 2.3m까지 도달 —
  **같은 코드인데 그날 GPS 환경이 얼마나 오래/심하게 pull을 붙잡아두는지에 따라
  성공/실패가 갈림**(`151207`에서는 pull이 사용자가 멈출 때까지 20초+ 동안 **계속
  다른** 격자각도 사이를 오가며 한 번도 안 풀림 — 재현성이 보장 안 되는 실시간
  환경 변수임을 확인).
- **직진 구간 좌편향(B)도 run마다 방향/크기가 다름**(어떤 run은 +0.02~0.03 좌측,
  `151819`는 오히려 -0.062 우측 쪽) — **고정된 상수 보정(예: unity 클론에 있었던
  `map_angular_bias_rad_s=-0.03`, 메인 레포엔 원래 없었음)은 위험**할 수 있음(run마다
  방향이 바뀌므로). 사용자가 "소프트웨어 명령은 거의 중립인데 실제로는 계속 좌로
  샌다"고 느낀 지점(`151819`)도 있어서, **하드웨어(휠 캘리브레이션) 쪽 가설**도
  Harness 1이 제기함 — 로봇을 평평한 곳에서 `angular=0`만 계속 보내는 순수 하드웨어
  테스트로 확인 필요(아직 안 함).

**2) 오늘 오후 추가 수정 4건(`experiment-heading-dev`, `80c230e`) — 전부 오프라인
스모크 테스트(10개) 통과, 위 1번의 "모델이 급커브 못 도는" 문제는 못 고침(원래
목표가 그게 아니었음 — heading 노이즈/크래시 안정성 목적):**
- GPS 동결 플래그가 10틱마다 재무장(re-arm)되던 버그 수정 — speed 조건을 "매 틱
  AND"에서 "동결 구간 안에서 한 번이라도 OR"로 바꿔서, 한 번 동결 확정되면 실제
  위치 변화 전까지 계속 유지되게 함.
- `route_bearing_anchored`의 pull에 `gps_accumulated_path_m >= 1.5m` readiness
  게이트 추가 — GO LIVE 직후처럼 이동량이 적을 때 짧은창 추정의 잡음성 pull 방지.
  **단, "매번 다른 격자각도로 계속 바뀌며 pull 상태 자체가 안 풀리는" 변종
  (`151207`에서 재확인)은 이 두 수정으로도 못 잡음** — pull을 "절대 지속시간"으로
  도 한계 두는 방향이 다음 후보(Harness 1에 전달 예정).
- `straight_segment_override`/`near_goal_override`/`off_route_safety`(steer)가
  조향 계산 시 그 틱의 (오염 가능한) `map_heading_rad` 대신 매번 새로 구하는
  `route_bearing_rad_now`를 기준으로 쓰도록 수정 — 1틱짜리 노이즈로 풀파워(0.3)
  조향이 튀던 버그 해결.
- `poll_frodobot()`/`send_control()`에 네트워크 재시도(2회, 0.3초 간격) 추가 —
  SDK 서버 1~2초 일시적 hiccup으로 전체 프로세스가 죽던 문제 완화(통신 완전
  불능이면 여전히 정지 후 종료, 안전 원칙 유지).

**3) 별개로 발견한 미해결 이슈 (코드 안 건드림)**: `dashboard_capture.py`가
`-movflags +faststart`로 mp4를 쓰는데, 중간에 pyppeteer/Chrome이 죽으면(오늘 2번,
`144650`/`151819`) **그 시점까지 찍힌 영상 전체가 통째로 복구 불가능해짐**(moov
박스가 끝에 한 번에 써져서). `+frag_keyframe+empty_moov`(fragmented mp4)로 바꾸면
죽어도 그때까지는 재생 가능 — 아직 적용 안 함, 다음에 할 일.

**4) 재학습 시 최우선순위 (기존과 동일, 변경 없음)**: §1-3(raw Arrow 데이터 좌편향
직접 확인, blocked), map_range_m 12m 전후로 새 production geometry로 재학습,
재학습 후 `deployment/analysis/*.py`로 오프라인 선검증. **오늘 추가로 확인된 것**:
이 급커브 실패가 재학습으로도 고쳐지는지 별도로 확인 필요(§1-3과 무관하게 "급커브
자체를 못 돈다"는 더 일반적인 문제일 수 있음 — bbox_h 가설과 같은 원인인지 다른
원인인지는 미확정).

**다음 세션 할 일(우선순위)**: (1) 하드웨어 캘리브레이션 가설 확인(순수 teleop 직진
테스트), (2) pull "절대 지속시간" 상한 추가, (3) dashboard mp4 fragmented 포맷 전환,
(4) 재학습 트랙은 §1-3 블로킹 해소 대기.

---

## ⚠ 2026-10-03 세션 요약 (2026-10-04에 의해 일부 갱신됨 — 위 최신 요약 먼저 읽을 것)

**좌편향 조사가 크게 진전됨 — 메커니즘을 구체적으로 규명하고 완화책까지 구현했으나 아직 실기기 미검증.**
상세는 반드시 `docs/experiment_log.md` 먼저 읽을 것(이 섹션은 요약만).

- **다른 터미널의 별도 Claude Code 세션("Harness 1", 학습/모델 분석 전담)과 이 세션("Harness 2",
  실배포/로봇 담당)을 동시에 띄워서 작업하는 방식을 도입함** — `docs/experiment_log.md`가 둘이
  공유하는 실험 로그/역할분담 문서. 다음 세션도 이 구조를 이어가려면 그 문서 §0 역할분담부터 볼 것.
- **좌편향의 진짜 트리거를 규명함**: "goal까지 실제 거리"가 아니라 **"화면에 그려진 future-route
  빨간 선분의 픽셀 길이(bbox_h)"**였음 — 완전 직선 경로(177틱, 곡률 0)에서도 bbox_h가 줄면 좌편향이
  정비례해서 커짐. 27m 체크포인트 기준 bbox_h<25px(≈dist_to_goal 7-8m) 이하면 실제 경로 방향과
  무관하게 결정론적으로 좌회전(직선 경로 177/177, 실제 우회전 경로에서도 26:2·22:2). map_range_m을
  줄이면 같은 실제 거리에서 bbox_h가 커져서(트리거 구간에 늦게 도달) 편향이 약해 보였던 것 —
  "곡률을 더 잘 보여줘서"가 아니었음. 상세 수치: `docs/experiment_log.md` §1-8.
- **완화책 구현 완료, 실기기 미검증**: `omnivla_edge_deploy.py --near_goal_override_dist_m`(기본
  8.0m) — 이 거리 이내에서는 모델 raw 예측 대신 `route_bearing_rad()` 기반 조향으로 강제 대체.
- **별개로 발견·수정한 문제(heading 추정 불안정성)**: `auto` 모드의 GPS-track heading이 실제 턴
  구간에서 GPS 자체는 깨끗한데도 틱마다 수십~100도씩 흔들림 확인(지도 회전이 매 틱 달라져서 예측도
  덩달아 불안정) → 틱당 변화량 rate-limit + route_bearing과 45°+ 어긋난 채 5틱 이상 지속되면
  강제 재동기화(`map_heading_source="route_corrected"`) 추가.
- **그 외 추가**: goal 도착 시 프로세스 자동 종료(이전엔 매번 수동 Ctrl+C 필요했음), 대시보드/
  frames 녹화를 영상(mp4)으로 저장하되 Notion에서 안 열리던 `cv2.VideoWriter`(mp4v) 대신
  ffmpeg+libx264 직접 호출로 교체.
- **실기기 테스트는 로봇 배터리 방전으로 중단됨** — 충전 후 재개 필요. 커밋 `afde77f`에 전부
  들어있고 push 완료함.
- **다음 세션에 할 일(우선순위 순)**: (1) `--near_goal_override_dist_m 0`으로 override 끄고
  heading rate-limit/route_bearing 재동기화/goal 자동종료 3가지부터 깨끗하게 검증, (2) 그다음
  override 켜고(기본값 8.0m) 별도로 검증, (3) `deployment/dashboard_capture.py`가 비정상 종료 시
  헤드리스 Chrome 프로세스를 못 지우고 좀비로 남기는 버그 발견함(오늘 89개 누적되어 SDK 서버
  행(hang)의 원인이 됨, `pgrep -f pyppeteer`로 확인 가능) — 아직 안 고침, 다음 세션에서 수정 필요.

## 알려진 이슈 / 아직 안 풀린 문제 (과거 기록 — 위 2026-10-03 요약이 최신 상태, 아래는 그 이전 경과)

**"map-zero(직진 상황)에서 우회전을 예측하는" 편향**을 지도교수 피드백으로 조사 중 (오프라인 평가 기준).
- 좌표축/회전 공식, OSM 배경-GPS 정렬, 입력 무관 고정 편향 — **전부 검증 결과 문제없음**.
- 유력 후보: 데이터 클래스 불균형(직진 10.6%, 우:좌=1.53:1, 아직 리밸런싱 미적용), 맵 스케일(25m)이
  실제 예측 horizon(~2m)에 비해 과하게 넓어서 근거리 곡률이 픽셀 몇 개로 뭉개짐.
- 12m로 줄였더니 지표(ADE 0.234m, val_loss 0.8143)가 지금까지 가장 좋음. 20m(근거리 디테일과 회전
  예고 범위의 절충안)는 오히려 더 이른 epoch부터 과적합, 지표도 12m보다 나쁨.
- 상세 조사 경과는 Claude 메모리(`project-map-zero-bias-investigation`)에 있음 — 이 문서에는 결론만.

**실로봇 배포(20m 체크포인트)에서는 반대로 좌회전 편향 관찰됨 (2026-08-10) — 위 오프라인 편향과 방향이 정반대라 원인이 다를 가능성.**
- 서울과학기술대 프론티어관→정문 실주행 로그(372 tick) 통계: `angular` 양수(좌회전) 67.2%, 최대값(+0.300)
  포화 109회 vs 최대 음수(우회전) 포화 35회 (3.1배). 평균 +0.0875 rad/s로 지속적 좌측 편향.
- 같은 구간을 OSRM으로 직접 계산해보면 실제 경로는 **우회전 위주**(우회전 합 461° vs 좌회전 합 89°) —
  "지도가 왼쪽 경로라서 그렇다"는 가설은 반박됨. 경로와 반대 방향으로, 그것도 강하게 치우침.
- 유력 가설이었던 **heading 소스 불일치**(학습=GPS궤적 `atan2`, 배포=IMU 컴퍼스)는 2026-08-25 실배포
  로그 분석에서 `heading_diff_deg` 평균 +97°로 실측 확인됨 (정황 증거, 완전 확정은 아니었음).
  **2026-09-05 세션에서 배포 heading 소스를 GPS궤적 기반으로 전환 완료** — 상세는 `docs/0905.md` 및
  아래 "2026-09-05 세션 변경사항" 참고. **다음 실주행에서 `heading_diff_deg`/좌회전 편향이 실제로
  해소됐는지 재검증 필요 — 아직 실기기 검증 전.**

## 배포 파이프라인

`deployment/README_omnivla_edge.md`에 clone-and-deploy 전체 가이드가 있음 (설치 → 체크포인트/OSRM
데이터 HF 다운로드 → FrodoBot SDK 서버 실행 → 배포 스크립트 실행 → 모니터링 대시보드). 요약:

- 추론은 로봇과 같은 네트워크의 GPU 머신(현재: RTX 5080 Laptop GPU)에서 실행, 이 학습 서버와는 별도 머신.
- 맵 입력은 학습 때(GT 미래경로 재활용)와 다르게, 배포 시에는 **OSRM 실시간 라우팅**으로 대체
  (`deployment/build_live_map.py`). 경로는 목적지 설정 시 1번만 계산해서 캐싱하고, 매 제어 루프(3Hz)마다
  재쿼리하지 않음 — `github.com/hmmdyn/osmnav` 구조를 참고해서 이렇게 고침 (처음엔 매 프레임
  재쿼리하는 버그가 있었음).
- FrodoBot SDK는 ROS가 아니라 **REST API** (`/control`, `/v2/front`, `/data`, 로컬 `127.0.0.1:8000`).
- 실시간 모니터링용 웹 대시보드(`deployment/debug_web.py`, 포트 8080)가 배포 스크립트 실행 시 자동으로
  같이 뜸 — SDK 자체 `/sdk` 페이지는 카메라만 보여주고 우리 지도/GPS/예측/에러는 안 보여주기 때문에
  별도로 만든 것.
- 배포 지역은 서울(서울과학기술대 근처) — OSRM 서버가 `osm_pipeline/py/osm_map_generator.py::osrm_port()`에
  포트 5011로 등록되어 있음.
- **추론 머신 환경**: 별도 venv 대신 기존 conda env `frodobot`(Python 3.10) 재사용 — `earth-rovers-sdk`
  서버 구동에 필요한 fastapi/hypercorn/opencv/requests가 이미 있어서, 여기에 torch/torchvision/pillow만
  추가 설치하면 됨. **RTX 50xx(Blackwell, sm_120) GPU는 cu124가 아니라 `cu128` 휠 필요**
  (`pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128`).
- `deployment/README_omnivla_edge.md`의 실행 예시는 `--goal_lat`/`--goal_lon`만 받고 **출발 좌표 인자는
  없음** — 매 배포 시작 시 로봇의 그 순간 실시간 GPS(`/data`)를 자동으로 출발점으로 씀. 원하는 출발
  위치에 로봇을 세워두고 스크립트만 실행하면 됨.

## 실로봇 첫 배포 테스트에서 고친 버그 (2026-08-10)

1. **`send_control()`이 `requests.post(..., data=...)`로 보내서 명령이 전혀 전달 안 됨** — nested dict를
   form-urlencode하면 `requests`가 값을 깨뜨려서(`command=linear&command=angular`, 숫자값 소실) 서버의
   `request.json()` 파싱이 실패. 콘솔엔 계산된 `linear`/`angular`가 정상 출력되는데 로봇은 안 움직이는
   증상으로 나타남. → `json=`으로 수정 (`examples/basics/*.py` 전부 이 방식 씀 — 앞으로 새 스크립트도
   반드시 `json=` 사용).
2. **헤드리스 브라우저(pyppeteer) 콜드스타트가 1초 HTTP 타임아웃보다 오래 걸려 `ReadTimeout` 크래시** —
   `browser_service.py`가 첫 요청에서 Chrome 실행+`/sdk` 접속+RTM join까지 하는데 수 초~십수 초 소요.
   → `omnivla_edge_deploy.py::run()` 시작 시 `/data`를 30초 타임아웃으로 미리 한 번 호출해서 워밍업,
   이후 루프 타임아웃도 1.0→5.0초로 완화.
3. **`osm_map_generator.py`의 `TILE_CACHE`가 원저자 컴퓨터 절대경로(`/media/ms/...`)로 하드코딩** —
   다른 머신에서 `os.makedirs` 권한 오류로 즉시 크래시. → 리포 상대경로(`osm_pipeline/tile_cache`)로 수정.

**진단 도구 추가** (좌회전 편향 조사용, 위 "알려진 이슈" 참고): IMU vs GPS궤적 heading 비교 로그,
`deployment/logs/deploy_<시각>.jsonl`(매 tick GPS/heading/예측/명령/HTTP상태/원본 텔레메트리/OSRM 경로
전체 기록), 대시보드 지도 위 예측 궤적 오버레이(실제 축척 px/m 일치 + 스케일바).

## 2026-09-07: RunPod 인프라 전부 삭제함 (비용 문제) — 재개 시 `docs/0906_runpod_setup.md` 참고

9/6에 RunPod network volume + GPU pod 만들어서 데이터 업로드/압축해제까지 다 끝냈었는데,
pod을 켜둔 채 34시간 방치해서 과금(~$25)되는 걸 발견 → pod과 volume 둘 다 삭제해서 정리함
(현재 RunPod엔 아무 리소스도 없음, `runpodctl pod list`/`network-volume list` 확인함).
**파인튜닝을 다시 시작하려면 volume 생성부터 처음부터 다시 해야 함** — 절차와 이번에 배운
gotcha(파트 크기, MSYS 경로변환, pod 방치 주의 등)는 `docs/0906_runpod_setup.md`에 전부
기록해둠.

## ⚠ 2026-09-06 진행 중 이슈 (상세: `docs/0906.md`) — 다음 세션 최우선으로 읽을 것

538개 세그먼트 전체 재생성 도중 **cartocdn 타일 서버가 API 키를 요구**하기 시작해서 데이터에
"API KEY REQUIRED" 워터마크가 섞여 들어감 (건물을 가로지르는 것처럼 보이는 원인). 대체로 쓰려던
`tile.openstreetmap.org`도 2026-03 TOTP 스크래핑 방지 정책으로 막힘. CARTO 무료 API 키 발급
대기 중 — 받으면 `fetch_tile()` URL에 반영 후 재생성 재시작. 상세 절차는 `docs/0906.md` 참고.

## 2026-09-05 세션 변경사항 (상세: `docs/0905.md`)

waypoint horizon 확장(2m→5m) + map scale 재확정 + 배포 heading 소스 수정을 같이 진행. 아직 **실제
데이터 재생성/재학습/실기기 검증 전** — 코드만 변경된 상태.

- `osm_pipeline/py/osm_map_generator.py`: 스케일 확정(전방20m/후방7m, `REAR_RATIO=0.35`), 줌 18→19,
  타일 provider→cartocdn voyager_nolabels(캐시 디렉터리 분리), 회전+리스케일+앵커배치를 단일
  `warpAffine`(bicubic)로 통합, robot 마커를 warp 이후에 그리도록 변경, 지도 밖 채움색 흰색→회색,
  goal 마커 추가(기존엔 정의만 있고 렌더링 안 되던 버그).
- `deployment/build_live_map.py`: 새 앵커/스케일에 맞춰 예측궤적 오버레이 좌표 수정.
- `deployment/omnivla_edge_deploy.py`: heading 소스 IMU컴퍼스→GPS궤적 기반(`estimate_heading_from_track`)
  전환, 그 함수의 GPS 양자화 잡음 취약점(직전 2점 비교)도 누적 1.5m 구간 방식으로 수정,
  `WAYPOINT_STRIDE_SEC`을 `CTX_STRIDE_SEC`과 분리된 상수로 신설(둘이 우연히 같은 값이라 혼동 위험 있었음).
- `osm_pipeline/py/rides11_dataset.py`: `WAYPOINT_STRIDE` 3→7 (실측 평균속도 0.90m/s 기준 ~5m horizon,
  기존은 ~2.17m).

**다음에 할 일**: (1) 에피소드 1개만 실제 타일로 재생성해서 눈으로 확인 → (2) 전체 재처리(OSRM 서버
필요, 몇 시간) → (3) 재학습 → (4) 실주행에서 `heading_diff_deg`/좌회전 편향/waypoint 예측거리 재검증.

## 자주 쓰는 명령

```bash
# 파인튜닝
/home/ms/uv-envs/mbra/venv/bin/python scripts/omnivla/finetune_omnivla_edge.py --config config/rides11_finetune_odom_12m.yaml

# OSM 맵 재생성
python3 osm_pipeline/py/osm_map_generator.py --map_range 12 --out_root osm_pipeline/osm_data/output_rides_11/osm_maps_arrow_12m

# test set 영상 생성 (fps=10 권장, 실제 촬영 속도와 동일)
python3 scripts/omnivla/make_test_video.py --ckpt checkpoints/omnivla_edge_rides11_odom_12m/best.pth --config config/rides11_finetune_odom_12m.yaml --fps 10

# 실배포 (README_omnivla_edge.md 참고)
python3 deployment/omnivla_edge_deploy.py --ckpt checkpoints/omnivla_edge_rides11_odom_12m/best.pth --map_range 12 --goal_lat <위도> --goal_lon <경도>

# 2026-10-03 추가 옵션(좌편향 완화책) 끄고 테스트하려면: --near_goal_override_dist_m 0
# (기본 8.0m는 27m 체크포인트 기준 실측 보정값 -- 다른 체크포인트면 docs/experiment_log.md §1-8 참고해 재측정)
```
