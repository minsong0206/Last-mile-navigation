# 04. Exact Model-input Replay

`replay_logging_diagram.png` — `tick_id` 하나로 그 순간 모델 입력 전체(카메라
6프레임 + 지도 + 예측 + target waypoint + 계산된 control)를 재현할 수 있음을
실제 파일로 보여주는 다이어그램. `generate_replay_logging_diagram.py`로 재생성.

## 사용한 데이터

- `run_id = 20260925_165933` (2026-09-25 offline_smoke_test.py, dry_run=True)
- `tick_id = 11` (02_debug_panel과 동일 tick — 같은 순간을 다른 각도에서 보여줌)
- 카메라 6프레임: `deployment/logs/frames/20260925_165933/ctx_000004.jpg` ~ `ctx_000009.jpg`
  (실제 파일, 2026-09-18 배포 스크린캐스트에서 추출한 진짜 카메라 프레임)
- 지도: `map_000011.png` (실제 저장 파일) + 그 tick의 실제 `pred_xy_m`으로
  `LiveMapBuilder.draw_predicted_trajectory()`를 다시 호출해 궤적 오버레이 재현
  (저장 당시와 동일한 함수 — 새로 지어낸 값 아님)
- `target_waypoint_xy`, 계산된 `linear`/`angular`: 해당 tick의 JSONL 레코드 값 그대로

## 핵심 메시지 (0918 vs 지금)

- **2026-09-18**: 대시보드 카메라가 배포 루프의 `frame_buffer`와 독립적으로 갱신돼서,
  나중에 이상 행동(angular=+0.3 포화)을 화면 캡처로 재구성하려 했지만 그날 실제
  기록된 `pred_xy_m`과 전혀 다른 예측이 나옴 → 원인 분리 불가 → "재현 불가"로 결론
  (2026-09-25 세션에서 직접 검증, `docs/` 대화 기록 참고)
- **지금**: `predict_waypoints()`가 실제로 텐서를 만드는 데 쓴 원본 이미지를
  `tick_id`로 그대로 저장 — 이 다이어그램 자체가 "tick_id 하나로 전부 재현 가능"의
  실제 증거임 (모든 값이 `offline_smoke_test.py`로 검증된 실제 파일/레코드)
