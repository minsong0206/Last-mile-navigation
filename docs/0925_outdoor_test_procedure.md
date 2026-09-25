# 실외 테스트 절차 (2026-09-25 작성, 2026-09-25 route-aligned bootstrap 반영해 갱신)

실내에서 GPS-fix_quality 안전장치 구멍을 발견해 로봇이 의도치 않게 움직인 사고가
있었음 — 그 교훈으로 확장된 디버그 대시보드 + deterministic replay logging을
도입했고, 이후 세션에서 GPS heading bootstrap deadlock 문제를 route-aligned
initial heading + ARM/GO LIVE 2단계 워크플로로 해결했다(상세 설계/검증 결과는
`docs/0925_route_aligned_heading_bootstrap.md` 참고). **아래 절차는 그 갱신을
반영한 최신 버전 — 재시작 기반 GO는 더 이상 쓰지 않는다.**

## 환경 (전부 고정, 세션 간 바꾸지 말 것)

- 노트북: 이 머신, `frodobot` conda env
- SDK: `conda activate frodobot && cd ~/earth-rovers-sdk && hypercorn main:app --bind 127.0.0.1:8000`
  (다른 env/다른 디렉토리에서 뜨면 `.env` 인증정보가 없어서 `Authorization header not configured`
  에러 남 — 2026-09-25에 실제로 겪음, `~/earth-rovers-sdk`인지 꼭 확인)
- OSRM: `bash osm_pipeline/scripts/start_osrm.sh seoul` (꺼져있으면 조용히 직선 경로로
  폴백되고 그래도 제어 명령은 나감 — 이 세션에서 발견한 미해결 이슈, 반드시 켜져있는지
  직접 확인할 것)
- 체크포인트: `checkpoints/omnivla_edge_rides11_odom_20m_20260910/best.pth`, `--map_range 20`
- 경로: 프론티어관 → 창업보육센터 중 일부 **직진 구간만** 먼저

## 절차

1. 로봇을 실제 보행로 위(OSRM `/nearest`로 스냅 거리 0m에 가까운 지점)에, **실제
   route 진행 방향을 향해 물리적으로 정렬**해서 세움
2. 배포 스크립트 시작(플래그 없이) — 항상 `DRY_RUN` 상태로 시작, 재시작 없이 이후
   전부 진행됨
3. 대시보드(`http://<노트북IP>:8080`)에서 GPS/route/North-up 미리보기 확인
4. 로봇이 실제로 route 방향을 향해 있는 것을 육안 재확인한 뒤, 대시보드 **"정렬 확인"**
   버튼 클릭(그 방향이 initial heading으로 고정됨, 대시보드에 `route_aligned` 값 표시)
5. 최종 heading-up 모델 입력 지도 / camera context / 예측 궤적 / 계산된 control까지
   전부 아래 체크리스트로 확인 — 몇 분간 지켜볼 것
6. 문제없으면 **"ARM"** → **"GO LIVE"** (둘 다 confirm 다이얼로그) — 이 순간부터만
   실제 non-zero 명령 전송. **재시작 안 함**, frame_buffer/GPS 궤적 상태 그대로 유지됨
7. `MAX_V`(현재 0.3)를 낮춘 값으로 짧은 구간만 먼저
8. 초반 몇 초는 `map_heading_source=route_aligned`로 움직이다가, 실제 이동이
   누적되면 `gps_track`으로 자동 전환됨(대시보드 heading source 색상 확인, 로그에도
   `heading_source_transition` 이벤트로 남음) — 전환 시점에 heading이 다소 튈 수
   있음을 감안하고 지켜볼 것
9. 문제 생기면 대시보드 **"ABORT"**로 즉시 `DRY_RUN` 복귀(실제 정지는 e-stop/SDK
   긴급정지 우선) — 또는 Ctrl+C
10. 사후 분석은 `deployment/logs/frames/<run_id>/`의 저장된 카메라/지도로 offline
    replay 가능 (`offline_smoke_test.py`처럼 `build_inputs()`를 직접 호출)

## GO 전 체크리스트 (대시보드에서 확인)

- [ ] 올바른 체크포인트 로드됨 (콘솔의 `[deploy] Loaded checkpoint: ...` 확인)
- [ ] GPS `fix_quality` ≥ 2 (대시보드 초록색)
- [ ] GPS 데이터 나이 < 1.5s (대시보드 초록색)
- [ ] OSRM 정상 (대시보드 "상태 — Route" 카드에 `osrm_fallback: 정상` — fallback이면
  경고, 직선 경로로 대체된 상태이므로 route geometry가 실제 보행로와 다를 수 있음)
- [ ] North-up 경로 미리보기가 실제 보행로 위에 정상적으로 그려짐 (건물 위로 지나가지
  않음) — route geometry 자체 문제와 heading 회전 문제를 여기서 먼저 분리 확인
- [ ] **로봇이 실제로 route 방향을 향해 정렬돼 있음을 육안 확인 후에만 "정렬 확인" 클릭**
- [ ] `route_aligned 확정값`이 표시됨, North-up 미리보기의 진행 방향과 대략 일치
- [ ] 최종 heading-up 모델 입력 지도가 시각적으로 합리적(건물 오버레이/꺾임 없음),
  진행 방향이 이미지 위쪽
- [ ] 예측 궤적(청록색)이 계획 경로(빨간색)와 크게 안 어긋남
- [ ] 계산된 control이 합리적 (직진 구간에서 angular가 크지 않음)
- [ ] `/control` 왕복시간 정상 범위 (대시보드 초록색, <500ms)
- [ ] **"ARM" 버튼이 활성화됨을 확인**(정렬 확인 전에는 비활성) → ARM → **"GO LIVE"**
  둘 다 confirm 다이얼로그에서 내용 재확인 후 클릭

## 실행 명령

```bash
# 터미널 1 — SDK
conda activate frodobot && cd ~/earth-rovers-sdk && hypercorn main:app --bind 127.0.0.1:8000

# 터미널 2 — OSRM
cd ~/Last-mile-navigation && bash osm_pipeline/scripts/start_osrm.sh status
# 안 떠있으면: bash osm_pipeline/scripts/start_osrm.sh seoul

# 터미널 3 — 1회만 실행, 재시작 불필요
conda activate frodobot
export PYTHONNOUSERSITE=1
export CARTO_API_KEY="cb1_2yj0_1_ffcc69174af013089c6c5da6"
cd ~/Last-mile-navigation
python3 deployment/omnivla_edge_deploy.py \
    --ckpt checkpoints/omnivla_edge_rides11_odom_20m_20260910/best.pth \
    --map_range 20 \
    --goal_lat <목표위도> --goal_lon <목표경도>
# 대시보드(http://<IP>:8080)에서 정렬 확인 → ARM → GO LIVE

# 순수 검증만 하고 싶다면(이 프로세스에서 GO LIVE 영구 잠금):
python3 deployment/omnivla_edge_deploy.py \
    --ckpt checkpoints/omnivla_edge_rides11_odom_20m_20260910/best.pth \
    --map_range 20 \
    --goal_lat <목표위도> --goal_lon <목표경도> \
    --dry_run
```

## 알려진 미해결 이슈 (참고만)

- GPS `fix_quality` 게이트 자체는 하드 게이트가 아님 (지금 코드는 `lat==1000`만
  차단) — 대시보드에 값은 표시되지만 판단은 여전히 사람이 함.
- OSRM 실패 시 조용히 직선 경로로 폴백하는 구조 자체는 안 고침 — 단, 이제
  `osrm_fallback` 상태가 대시보드/로그에 노출되므로 폴백 여부를 사람이 즉시 알 수는
  있음(전에는 콘솔 print만 있고 사라졌음). 여전히 OSRM이 켜져 있는지 절차 시작 전
  직접 확인 권장.
- `route_aligned → gps_track` 전환 순간의 스무딩은 의도적으로 넣지 않음 — 전환
  시점에 heading-up 지도가 한 번 튈 수 있음을 감안하고 지켜볼 것.
