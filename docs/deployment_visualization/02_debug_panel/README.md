# 02. Debug / Pre-drive Verification Panel

`debug_panel_capture.png` — `debug_web.py`가 실제로 렌더링하는 대시보드를
헤드리스 브라우저(pyppeteer)로 그대로 스크린샷한 것 (그림을 새로 그린 게 아니라
**실제 서버를 띄우고 캡처**했다 — `debug_web.py` 자체가 발표 캡처에 쓸 만한
레이아웃인지 확인하는 목적도 겸함).

## 사용한 데이터

- `run_id = 20260925_165933` (2026-09-25 offline_smoke_test.py, **dry_run=True**
  — pre-drive 체크 시나리오를 보여주기 위해 dry-run 실행을 선택)
- `tick_id = 11`
- 카메라: 그 tick의 실제 컨텍스트 프레임(2026-09-18 배포 스크린캐스트에서 추출한
  진짜 카메라 프레임)
- 지도: 실제 저장된 model-input map + 실제 `pred_xy_m`으로 다시 그린 예측 궤적
- heading/route/control 등 모든 상태값: 그 tick의 JSONL 레코드 값 그대로

## 캡처 과정에서 발견하고 고친 것

이 캡처를 준비하다가 실제 결함 2개를 발견해서 코드에 반영했습니다 (발표 자료
만들기 전에 먼저 고쳤습니다):

1. **dry-run 중엔 "계산된 명령"을 확인할 방법이 없었음** — 대시보드가 "실제 전송된
   값"(dry-run이면 항상 0,0)만 보여줘서, pre-drive 체크리스트의 "제어 명령이
   합리적인가"를 dry-run 중엔 검증할 수 없었음. `computed_linear`/`computed_angular`를
   `DeploymentState`에 추가해서 "계산된 명령"과 "실제 전송된 명령"을 모두 보여주도록 수정.
2. **GPS 데이터 나이가 비정상적으로 크게 표시됨** — 오프라인 테스트의 GPS mock이
   타임스탬프를 `"0.0"`(플레이스홀더)으로 고정해뒀던 게 원인. 실제 SDK는 항상
   "방금" 시각을 주므로, 테스트 mock도 매 호출 시점의 실제 시각을 넣도록 수정.

## `debug_web.py` 레이아웃 자체는 캡처하기 적합한가

네 — 다크 배경 + 3열 이미지 카드(카메라/지도/확대 예측) + 2개 상태 테이블
(Localization / Model·Control) + 로그 패널 구조가 1500px 뷰포트에서 스크롤 없이
한 화면에 다 들어오고, 색상 코드(초록=정상/빨강=주의)도 스크린샷에서 잘 구분됨.
DRY RUN/LIVE 배지도 상단에 크게 보여서 발표 중 "지금 어느 모드인지" 바로 전달 가능.
