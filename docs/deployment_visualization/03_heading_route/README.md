# 03. Heading vs Route Bearing

`heading_route_diagram.png` — 왜 `heading_route_diff_deg`를 디버그 지표로 추가했는지
두 실제 사례로 비교. `generate_heading_route_diagram.py`로 재생성(계산된 수치는
스크립트 상단 docstring에 출처와 함께 상수로 박아둠 — 원본 재검증하려면 해당
JSONL을 다시 열면 됨).

## 사용한 데이터

| Case | 출처 | heading | route bearing | Δθ |
|---|---|---:|---:|---:|
| A (정렬) | `deploy_20260925_165933.jsonl`(오프라인 스모크 테스트), tick_id=11 | GPS-track −111.6° | −132.4° | **+20.8°** |
| B (심한 불일치) | `deploy_20260918_181503.jsonl`, ts=1789722955.50 (0918 실배포 중 angular=+0.3 포화 순간) | IMU 폴백 −192.0° | −105.7°(오늘 추가한 `route_bearing_rad()`로 재계산) | **−86.3°** |

Case B는 참고용으로, 같은 순간 GPS-track heading(−128.4°, 2026-09-18 세션에서
사후 수정된 `estimate_heading_from_track()`으로 재계산한 값)을 썼다면 차이가
−22.6°까지 줄었을 것이라는 점도 같이 표기 — **IMU 폴백에 갇혀있던 것 자체가 그날
오차의 핵심 원인이었다는 근거.**

## 왜 이 지표가 필요한가

- **직선 구간**: heading과 route bearing이 거의 같아야 정상 — 차이가 크면 heading
  추정 자체가 잘못됐다는 신호(Case B가 정확히 이 경우)
- **커브 직전**: `route_bearing_rad(lookahead_m=5.0)`은 5m 앞까지의 코드(직선) 방향이라,
  실제로 경로가 그 안에서 꺾이면 로봇 heading이 맞아도 차이가 커질 수 있음 — 그래서
  이 값은 **안전 게이트가 아니라 디버그 지표**로만 사용(대시보드에서 사람이 지도를
  같이 보고 판단)
