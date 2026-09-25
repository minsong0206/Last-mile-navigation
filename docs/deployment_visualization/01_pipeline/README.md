# 01. Deployment Pipeline

`pipeline_diagram.png` — 전체 배포 아키텍처를 한 장으로 보여주는 다이어그램.
`generate_pipeline_diagram.py`로 재생성 (실제 로그/데이터 불필요 — 구조만 그리는
다이어그램이라 실측값을 쓰지 않음).

```
python3 docs/deployment_visualization/01_pipeline/generate_pipeline_diagram.py
```

## 블록 설명

**메인 파이프라인 (좌측, 기존 구조를 그대로 보여줌)**
- **Camera / GPS / IMU** — 센서 입력 (`/v2/front`, `/data`)
- **Frame Buffer** — 최근 6프레임(과거5+현재1), 0.3초 간격으로 채움
- **GPS Heading** — GPS 궤적(누적 이동거리 기반)으로 추정한 진행방향
- **IMU Heading** — 로봇 컴퍼스 센서값 변환
- **OSRM Route** — 출발 시 1회만 계산해서 캐싱하는 경로(매 프레임 재쿼리 안 함)
- **Final Heading** — GPS Heading을 우선 사용하되 EMA로 스무딩, 못 구하면 마지막 값
  유지("관성") 또는 IMU 폴백
- **Heading-up Map** — Final Heading + Route를 이용해 ego-centric으로 회전시킨 지도
- **OmniVLA-Edge** — 카메라 6프레임 + Heading-up Map을 입력으로 받는 모델
- **Predicted Waypoints → Target Waypoint → Controller → Calculated linear/angular**
  — 8-step 예측 궤적에서 목표 지점을 골라 속도/각속도로 변환(안전 클리핑 포함)

**이번에 추가한 구조 (주황색, 점선 테두리)**
- **Dry-run Gate** — calculated linear/angular와 실제 로봇 전송 사이에 위치. LIVE
  모드면 계산값 그대로, DRY RUN 모드면 항상 `(0,0)`을 전송(계산값은 로그에 별도 보존)
- **Replay Logger** — 단순히 control 뒤에 붙는 게 아니라, **같은 tick_id**로
  Camera Context/Model-input Map/Prediction/Heading & Route/Control을 각각 저장하는
  관측 구조. tick_id 하나로 그 순간 모델 입력을 그대로 재구성할 수 있음(04번에서
  실제 예시로 보여줄 예정)
- **Debug Panel** — 파이프라인 전체를 실시간으로 관측하는 모니터링 레이어. 주행
  시작(GO) 전에 fix_quality, heading 정렬, 예측 궤적, 제어값 등을 사람이 직접
  확인하는 용도(02번에서 실제 화면으로 보여줄 예정)

## 발표 핵심 메시지

- 기존: `Sensor → Model → Control → Robot`
- 현재: `Sensor → Localization/Map → Model → Control` + 그 전체를 **Debug Panel**로
  주행 전 확인 + **Dry-run Gate**로 실제 actuator 출력 분리 + **Replay Logger**로
  tick 단위 재현 가능
