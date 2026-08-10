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

**"map-zero(직진 상황)에서 우회전을 예측하는" 편향**을 지도교수 피드백으로 조사 중.
- 좌표축/회전 공식, OSM 배경-GPS 정렬, 입력 무관 고정 편향 — **전부 검증 결과 문제없음**.
- 유력 후보: 데이터 클래스 불균형(직진 10.6%, 우:좌=1.53:1, 아직 리밸런싱 미적용), 맵 스케일(25m)이
  실제 예측 horizon(~2m)에 비해 과하게 넓어서 근거리 곡률이 픽셀 몇 개로 뭉개짐.
- 12m로 줄였더니 지표(ADE 0.234m, val_loss 0.8143)가 지금까지 가장 좋음. 20m(근거리 디테일과 회전
  예고 범위의 절충안)는 오히려 더 이른 epoch부터 과적합, 지표도 12m보다 나쁨.
- 상세 조사 경과는 Claude 메모리(`project-map-zero-bias-investigation`)에 있음 — 이 문서에는 결론만.

## 배포 파이프라인

`deployment/README_omnivla_edge.md`에 clone-and-deploy 전체 가이드가 있음 (설치 → 체크포인트/OSRM
데이터 HF 다운로드 → FrodoBot SDK 서버 실행 → 배포 스크립트 실행 → 모니터링 대시보드). 요약:

- 추론은 로봇과 같은 네트워크의 GPU 머신(현재: RTX5090 노트북)에서 실행, 이 학습 서버와는 별도 머신.
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
