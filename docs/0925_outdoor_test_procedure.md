# 2026-09-26 고정 실외 테스트 절차

전날(0925) 실내에서 GPS-fix_quality 안전장치 구멍을 발견해 로봇이 의도치 않게
움직인 사고가 있었음 — 그 교훈으로 `--dry_run` + 확장된 디버그 대시보드 +
deterministic replay logging을 도입한 뒤 실행하는 절차.

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

1. 로봇을 실제 보행로 위(OSRM `/nearest`로 스냅 거리 0m에 가까운 지점)에 세움
2. `--dry_run`으로 배포 스크립트 시작
3. 대시보드(`http://<노트북IP>:8080`)에서 아래 체크리스트 전부 확인 — 몇 분간 지켜볼 것
4. 문제없으면 Ctrl+C → `--dry_run` 빼고 재시작 = 명시적 GO
5. `MAX_V`(현재 0.3)를 낮춘 값으로 짧은 구간만 먼저
6. 문제 생기면 `deployment/logs/frames/<run_id>/`의 저장된 카메라/지도로 그 자리에서 바로
   offline replay 가능 (`offline_smoke_test.py`처럼 `build_inputs()`를 직접 호출)

## GO 전 체크리스트 (대시보드에서 확인)

- [ ] 올바른 체크포인트 로드됨 (콘솔의 `[deploy] Loaded checkpoint: ...` 확인)
- [ ] GPS `fix_quality` ≥ 2 (대시보드 초록색)
- [ ] GPS 데이터 나이 < 1.5s (대시보드 초록색)
- [ ] OSRM 정상 (콘솔에 `OSRM 쿼리 실패` 로그 없음)
- [ ] 경로 정상 생성됨 (지도에 빨간 계획 경로 보임)
- [ ] heading이 눈으로 봤을 때 합리적 (지도가 실제 진행방향 기준 위쪽을 향함)
- [ ] heading − route bearing 차이가 크지 않음 (대시보드 초록색, 대략 ±30° 이내)
- [ ] 모델 입력 지도가 시각적으로 합리적(건물 오버레이/꺾임 없음)
- [ ] 예측 궤적(청록색)이 계획 경로(빨간색)와 크게 안 어긋남
- [ ] 제어 명령이 합리적 (직진 구간에서 angular가 크지 않음)
- [ ] `/control` 왕복시간 정상 범위 (대시보드 초록색, <500ms)

## 내일 실행 명령

```bash
# 터미널 1 — SDK
conda activate frodobot && cd ~/earth-rovers-sdk && hypercorn main:app --bind 127.0.0.1:8000

# 터미널 2 — OSRM
cd ~/Last-mile-navigation && bash osm_pipeline/scripts/start_osrm.sh status
# 안 떠있으면: bash osm_pipeline/scripts/start_osrm.sh seoul

# 터미널 3 — 1) dry-run 먼저
conda activate frodobot
export PYTHONNOUSERSITE=1
export CARTO_API_KEY="cb1_2yj0_1_ffcc69174af013089c6c5da6"
cd ~/Last-mile-navigation
python3 deployment/omnivla_edge_deploy.py \
    --ckpt checkpoints/omnivla_edge_rides11_odom_20m_20260910/best.pth \
    --map_range 20 \
    --goal_lat <목표위도> --goal_lon <목표경도> \
    --dry_run

# 체크리스트 통과 확인 후 Ctrl+C, 그 다음 --dry_run 빼고 재시작 = GO
python3 deployment/omnivla_edge_deploy.py \
    --ckpt checkpoints/omnivla_edge_rides11_odom_20m_20260910/best.pth \
    --map_range 20 \
    --goal_lat <목표위도> --goal_lon <목표경도>
```

## 알려진 미해결 이슈 (오늘 범위 밖 — 참고만)

- GPS `fix_quality` 게이트 자체는 아직 수정 안 함 (지금 코드는 `lat==1000`만 봄) —
  `--dry_run`과 위 체크리스트로 우회 대응 중. 코드 레벨 수정은 다음 세션 후보.
- OSRM 실패 시 조용히 직선 경로 폴백 후에도 제어 명령이 나가는 구조도 아직 안 고침 —
  그래서 OSRM이 꺼져있지 않은지 절차 1단계에서 직접 확인 필수.
