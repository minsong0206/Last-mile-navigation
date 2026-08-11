# CLAUDE.md

FrodoBots rides_11 데이터로 OmniVLA-Edge(odom3ch 변형)를 파인튜닝해서 FrodoBot Mini 로봇에
배포하는 캡스톤 프로젝트. 이 문서는 이후 세션(또는 협업자)이 빠르게 컨텍스트를 잡기 위한 요약이다.

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
- **`.gitignore`에 예전에 구멍이 있었음** — `osm_maps_arrow_12m/20m/25m`(각 11~13GB), 루트 `wandb/`,
  `checkpoints/` 등이 안 걸러지고 있었음. push 전에 `git status`로 대용량 미추적 디렉토리 없는지 항상
  확인할 것. 체크포인트(`*.pth`)와 대용량 데이터는 git이 아니라 Hugging Face에 올림
  (계정 `minsonganingee`, private repo).

## 알려진 이슈 / 아직 안 풀린 문제

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
- 현재 유력 가설: **학습 때와 배포 때 heading 소스가 다름**. `output_rides_11` 학습 데이터는
  `osm_map_generator_rides11.py::estimate_headings()`(GPS 궤적 기반 `atan2`, 순수 이동방향)로 heading을
  만들었는데, 배포(`omnivla_edge_deploy.py`)는 로봇 IMU 컴퍼스(`orientation` 필드)를 씀. `render_frame()`의
  heading-up 회전 공식(`rot_deg = 90 - heading_deg`) 자체는 수학적으로 검증 완료(문제 없음) — 그러니 값
  자체가 부정확하면(자북 편각/마운트 오프셋 등) 지도가 잘못된 방향으로 회전해서 들어갈 수 있음.
- **아직 확정 아님 — 검증 중.** 이번 세션에 진단 도구를 깔아뒀으니 다음 실주행에서 확인:
  - 콘솔/로그의 `heading_diff_deg`(IMU vs GPS궤적 heading 차이)가 일정한 오프셋으로 나오면 컴퍼스
    캘리브레이션 문제로 확정, 들쭉날쭉하면 다른 원인 재조사 필요.
  - 대시보드 지도에서 빨강(계획 경로) vs 청록(모델 예측 궤적)이 얼마나 어긋나는지 시각 확인.
  - 확정되면 배포 시 heading을 IMU 대신 `past_track` GPS 궤적 기반으로 바꾸는 게 후보 수정안
    (`estimate_heading_from_track()`가 이미 진단용으로 구현돼 있음 — 검증되면 실제 제어에도 전환).

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
```
