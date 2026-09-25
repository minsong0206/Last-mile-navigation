# Deployment Visualization (캡스톤 발표용)

`Last-mile-navigation` 배포 코드/로그와 섞이지 않도록 발표 자료만 따로 모아둔 폴더.
각 하위 폴더는 PNG(발표에 바로 사용)와 그 PNG를 재생성하는 Python 스크립트를 같이 둔다.

실제 로그/데이터를 사용하는 시각화는 각 폴더의 README에 **어떤 run_id/tick_id를
썼는지** 명시한다 — 측정 안 한 값이나 가상의 결과를 임의로 만들어 넣지 않는다.
전부 2026-09-25 offline_smoke_test.py(실제 모델 forward pass 포함, 로봇 연결
없음) 또는 실제 과거 배포 로그(2026-09-18)에서 나온 값이다.

## 폴더

| 폴더 | 내용 | 상태 |
|---|---|---|
| `01_pipeline/` | 전체 배포 파이프라인 아키텍처 다이어그램 (Debug Panel/Dry-run Gate/Replay Logger 위치 포함) | 완료 |
| `02_debug_panel/` | 실제 `debug_web.py` 대시보드 헤드리스 캡처 (주행 전 검증 화면) | 완료 |
| `03_heading_route/` | heading vs route bearing 정렬/불일치 비교 (실제 로그값 2건) | 완료 |
| `04_replay_logging/` | tick_id 하나로 model input→control을 역추적하는 실제 예시 | 완료 |

## 재생성 순서 (필요시)

```bash
conda activate frodobot
export CARTO_API_KEY="cb1_2yj0_1_ffcc69174af013089c6c5da6"
cd ~/Last-mile-navigation

python3 deployment/offline_smoke_test.py   # 02/04가 참조하는 run_id를 새로 만듦
                                            # (재생성하면 run_id가 바뀌므로, 02/04
                                            # 스크립트 상단의 RUN_ID 상수도 같이 갱신할 것)

python3 docs/deployment_visualization/01_pipeline/generate_pipeline_diagram.py
python3 docs/deployment_visualization/02_debug_panel/generate_debug_panel_capture.py
python3 docs/deployment_visualization/03_heading_route/generate_heading_route_diagram.py
python3 docs/deployment_visualization/04_replay_logging/generate_replay_logging_diagram.py
```
