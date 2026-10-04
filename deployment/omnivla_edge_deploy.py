"""
omnivla_edge_deploy.py

FrodoBot Mini 실로봇 배포용 OmniVLA-Edge-Odom 추론 루프.

deployment/LogoNav_frodobot.py의 FrodoBot SDK 연동 패턴(REST API: /v2/front 카메라,
/data GPS, /control 제어명령)을 그대로 재사용하되, 모델을 OmniVLA-Edge-Odom(rides_11
파인튜닝 체크포인트)으로 교체하고, 맵 입력은 build_live_map.py의 LiveMapBuilder로
실시간 생성한다 (학습 때처럼 GT 미래 GPS가 없으므로 OSRM 실시간 라우팅으로 대체).

⚠ 중요 — pre-drive 워크플로 (2026-09-25 재설계, 재시작 없이 같은 프로세스에서 진행):
  0. (권장) 출발 전 로봇을 정지시킨 채 20~30초 GPS만 진단(별도 도구/수동 확인) —
     fix_quality/누적 이동거리가 노이즈 바닥 수준인지 먼저 확인.
  1. 프로세스 시작 → 항상 DRY_RUN 상태로 시작 (실제 명령 전송 안 함).
  2. GPS fix 확보 → route(OSRM) 자동 생성 → **사용자가 로봇을 실제 route 방향으로
     물리적으로 정렬**하고, 대시보드에서 "정렬 확인" 버튼으로 그 방향을
     initial heading(route_aligned)으로 명시적으로 확정.
  3. 대시보드에서 North-up 경로 미리보기 / 최종 heading-up 모델 입력 지도 / camera
     context / 예측 궤적 / 계산된 control을 확인.
  4. "ARM" → "GO LIVE"(둘 다 확인 다이얼로그) — 이 순간부터만 실제 non-zero 명령 전송.
     frame_buffer/GPS 궤적/heading 상태는 전부 그대로 유지된 채 전환됨(재시작 없음).
  5. 저속(MAX_V 작게)으로 개활지에서 첫 테스트, e-stop/SDK 긴급정지 항상 준비.

Route-aligned initial heading — GPS course heading(estimate_heading_from_track)은
실제 이동이 있어야만 생기는데 로봇은 non-zero 명령이 있어야 이동하는 bootstrap
deadlock이 있음. 사용자가 물리적으로 정렬한 route 방향을 "실측 GPS heading이
확보되기 전까지"의 임시 heading으로 쓰고(map_heading_source="route_aligned"),
_heading_ema_vec는 절대 이 값으로 시드하지 않음 — 실측이 들어오면 오염 없이
하드셋됨(전환 이벤트는 로그/대시보드에 명시적으로 표시, 별도 스무딩 없음).

--dry_run 플래그(2026-09-25 추가, 2026-09-25 의미 변경) — 지금은 "이 프로세스 전체에서
GO LIVE를 영구 잠금"을 의미함(순수 검증 세션용, 대시보드에서 눌러도 거부됨). 플래그가
없어도 프로세스는 항상 DRY_RUN으로 시작하고, 대시보드의 명시적 확인 절차를 거쳐야만
LIVE로 전환됨. 센서→지도→추론→궤적→제어 계산은 DRY_RUN/ARMED에서도 전부 그대로
실행하되 실제 actuator 명령은 항상 (0,0)만 보낸다(계산된 값은 로그/대시보드에 별도
표시). "명령을 아예 안 보낸다"가 아니라 "명시적으로 0을 보낸다" 방식을 택한 이유:
로봇/SDK 쪽에 "일정 시간 새 명령이 없으면 자동 정지"하는 watchdog이 있는지
earth-rovers-sdk 전체를 확인해봤지만 문서/코드 어디에도 없었음 — 이 가정이 틀렸을
때의 위험(로봇이 마지막 non-zero 명령을 계속 유지)이 "명시적으로 0을 계속 보내는"
쪽보다 훨씬 크므로, 확인 안 된 가정에 의존하지 않는 쪽을 택함.

deterministic replay logging(2026-09-25 추가) — predict_waypoints()에 실제로
들어간 카메라 6프레임(dedup 저장)과 최종 지도 이미지를 tick_id로 묶어
deployment/logs/frames/<run_id>/에 저장한다. 자세한 설계 근거는
replay_logger.py 모듈 docstring 참고 (2026-09-18 세션에서 대시보드 카메라로
사후 재구성했다가 실제 모델 출력과 안 맞아서 실패했던 사례 때문에 도입).

학습-추론 일치 확인 필수 항목 (finetune_omnivla_edge.py::prepare_batch와 반드시 동일):
  - MAP_RANGE_M: 체크포인트를 학습시킨 맵 반경과 정확히 같은 값을 --map_range로 넘길 것.
    체크포인트별 정답 값:
      checkpoints/omnivla_edge_rides11_odom/best.pth       → 25 (기본, baseline)
      checkpoints/omnivla_edge_rides11_odom_12m/best.pth   → 12
      checkpoints/omnivla_edge_rides11_odom_20m/best.pth   → 20 (교수님 피드백 반영, 얇은 경로선 ROUTE_WIDTH=2)
  - N_CTX=5, CTX_STRIDE=3프레임(~0.3초 간격), 카메라 6장(과거5+현재1)
  - modality_id / goal_mask = 0 ("map only")
  - METRIC_WAYPOINT_SPACING = 0.125 (출력 waypoint를 미터로 변환할 때)
  - WAYPOINT_STRIDE_SEC: 체크포인트가 학습된 WAYPOINT_STRIDE와 정확히 일치시킬 것.
      위 25/12/20m 체크포인트는 전부 WAYPOINT_STRIDE=3(≈2m horizon) → 0.3으로 설정.
      2026-09 이후 5m-horizon(WAYPOINT_STRIDE=7)으로 재학습한 체크포인트는 0.7 사용.

실행 예 (20m-20260910 체크포인트 기준, frodobot conda env):
  # 1회만 실행 — 재시작 불필요. 항상 DRY_RUN으로 시작하고, 대시보드에서
  # 정렬 확인 → ARM → GO LIVE를 눌러야만 실제 명령이 나감.
  python3 deployment/omnivla_edge_deploy.py \
      --ckpt checkpoints/omnivla_edge_rides11_odom_20m_20260910/best.pth \
      --map_range 20 --goal_lat 37.5010 --goal_lon 127.0010

  # 순수 검증(LIVE 절대 금지) 세션이 필요하면 --dry_run 추가 — 이 프로세스는
  # 대시보드에서 GO LIVE를 눌러도 항상 거부됨.
  python3 deployment/omnivla_edge_deploy.py \
      --ckpt checkpoints/omnivla_edge_rides11_odom_20m_20260910/best.pth \
      --map_range 20 --goal_lat 37.5010 --goal_lon 127.0010 --dry_run
"""

import sys
import time
import base64
import io
import json
import queue
import argparse
import math
from pathlib import Path
from collections import deque
from datetime import datetime

import numpy as np
import requests
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "third_party" / "omnivla" / "inference"))
sys.path.insert(0, str(REPO_ROOT / "deployment"))
from model_omnivla_edge_odom import OmniVLA_edge_odom
from build_live_map import LiveMapBuilder
from debug_web import DeploymentState, start_debug_server
from replay_logger import ReplayLogger
from dashboard_capture import DashboardRecorder

FRODOBOT_BASE = "http://127.0.0.1:8000"

# 2026-10-03 추가: GO LIVE 시점에 보이는 웹 대시보드를 자동으로 스크린샷해서 남기는
# 폴더 — 발표/기록용(디버그 재현용은 아님, 그건 replay_logger.py가 이미 더 정확하게 함).
DASHBOARD_CAPTURE_DIR = REPO_ROOT / "1003"

# rides11_dataset.py와 동일한 학습 시 상수
N_CTX = 5
CTX_STRIDE_SEC = 0.3          # 컨텍스트 이미지 캡처 간격 (CTX_STRIDE=3 frames @ ~10Hz)
                               # ⚠ WAYPOINT_STRIDE_SEC과는 별개 상수임 — 우연히 값이
                               # 같았을 뿐(둘 다 예전엔 3프레임), 바꿀 때 혼동하지 말 것.
WAYPOINT_STRIDE_SEC = 0.7      # waypoint 간 시간 간격 (rides11_dataset.py WAYPOINT_STRIDE=7
                               # frames @ ~10Hz). 체크포인트가 어느 WAYPOINT_STRIDE로
                               # 학습됐는지에 정확히 맞춰야 함 (예전 25/12/20m 체크포인트는
                               # WAYPOINT_STRIDE=3 → 이 값 0.3으로 되돌려서 추론할 것).
METRIC_WAYPOINT_SPACING = 0.125
IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD  = [0.229, 0.224, 0.225]

MODEL_PARAMS = dict(
    context_size=5, len_traj_pred=8, learn_angle=True,
    obs_encoder="efficientnet-b0", obs_encoding_size=1024,
    late_fusion=False, mha_num_attention_heads=4,
    mha_num_attention_layers=4, mha_ff_dim_factor=4,
)

# ── 제어 안전 한계 (LogoNav_frodobot.py와 동일한 안전장치 재사용) ──
MAX_V = 0.3   # m/s
MAX_W = 0.3   # rad/s
DT    = 1.0 / 3.0  # 제어 루프 주기 (3Hz) — waypoint_to_control()에서는 안 씀(2026-08 버그 수정, 아래 참고)

# 2026-09-18: 추론 시작 전 GPS 궤적부터 확보하는 직진 웜업 단계를 넣었다가 제거함 —
# 로봇이 그 자리에서 실제로 순이동을 못 하는 지점(장애물/노면 등)에서는 웜업이 영원히
# 안 끝나 추론 자체가 시작조차 안 되는 문제가 실측됨(deploy_20260918_1823/1832.jsonl,
# 8초 넘게 GPS 위치가 거의 그대로였음). 대신 아래 EMA 유지 방식으로, 추론은 즉시
# 시작하되(초반 몇 틱은 IMU 폴백) GPS-heading이 한 번이라도 잡히면 그때부터 안정화.

# 2026-09-18: GPS-heading이 imu_fallback과 번갈아 전환되면서 지도 회전이 틱마다
# 수십~백여도씩 튀는 문제 실측 확인(cmp_imu_-214 vs cmp_gps_-147.7 렌더링 비교).
# 그래서 GPS-heading을 "즉시 대체"가 아니라 EMA로 완만하게만 반영하고, GPS 이동량이
# 잠깐 부족해도(제자리 회전/정지 등) 마지막 EMA 값을 그대로 유지("관성")하도록 바꿈 —
# 순간적인 IMU 폴백으로 스냅되지 않게. ALPHA가 작을수록 더 안정적이지만 실제 방향
# 전환에 더 느리게 반응함.
HEADING_EMA_ALPHA = 0.3

# 2026-10-03 추가: 틱당 heading 변화량 상한(rate limit). 실측(deploy_20261003_143500.jsonl,
# 10~20m "턴 구간")에서 route_bearing_deg는 164.83°로 고정인데 smoothed_heading_deg는
# -133°→-111°→-158°→-114°→-175°→+172°→+141°→... 식으로 틱마다 수십 도씩 흔들리는 걸
# 확인함 — 그 결과 지도가 매번 다른 각도로 회전되어 그려지고, 모델 예측도 따라서
# 좌/우로 계속 튐(사용자 관찰). 로봇이 실제로 한 틱(dt) 사이에 물리적으로 돌 수 있는
# 최대 각도는 로봇 자신의 MAX_W로 이미 정해져 있으므로, 그보다 큰 변화는 추정 잡음으로
# 보고 EMA에 입력하기 전에 clamp한다. HEADING_RATE_LIMIT_MARGIN은 실제 물리적 동역학/
# 센서 지연 여유분(2배) — 이 마진 적용해도 10~20m 구간에서 관측된 수십~100도대 틱당
# 점프는 전부 걸러짐.
HEADING_RATE_LIMIT_MARGIN = 2.0

# 2026-10-03 추가: heading이 route_bearing과 지속적으로 크게 어긋나면(=GPS 추정이 계속
# 틀린 채로 "자신있게" 주행) route_bearing 쪽으로 강제 재동기화한다. 기존엔
# heading_route_diff_deg가 대시보드/로그용 진단 수치일 뿐 실제 heading엔 전혀 반영이
# 안 됐음 — 그래서 한 번 틀어지면 스스로 못 돌아왔음(사용자 관찰: "틀어진 방향을
# 직진이라 믿고 주행해서 목표에 제대로 도달 못 함"). 단발성 노이즈가 아니라 "지속적"
# 어긋남만 보정 대상으로 삼기 위해 PERSIST_TICKS 연속으로 넘어야 발동 — 위 rate limit과
# 합쳐서, 짧은 노이즈는 rate limit이, 오래가는 진짜 오차는 이 재동기화가 맡는 구조.
# route_aligned_fixed 모드/부트스트랩 단계(아직 gps_track 실측 전)에는 적용 안 함 —
# 그쪽은 의도적으로 고정/보류된 값이라 건드리면 그 모드의 존재 이유 자체가 없어짐.
HEADING_ROUTE_CORRECTION_THRESHOLD_DEG = 45.0
HEADING_ROUTE_CORRECTION_PERSIST_TICKS = 5

# 2026-10-04 추가 (Harness 1 제안, docs/experiment_log.md 참고): heading_mode
# "route_bearing_anchored" 전용 상수. 기존 auto 모드는 gps_track이 기본값이고
# route_bearing은 "크게 어긋났을 때만" 끌어오는 구조였는데(위
# HEADING_ROUTE_CORRECTION_*), 이걸 뒤집음 — route_bearing을 기본 anchor로 쓰고,
# gps_track은 "충분히 길고 일관되게" 반박할 때만 끌어당긴다. Harness 1이 오프라인
# 재생(heading_estimator_replay.py)으로 "min_disp_m만 올리는 건 깨끗한 승리가
# 아님(최악 스파이크는 줄지만 중간 드리프트 노출시간은 오히려 늘어남)"을 확인해서
# 나온 결론 — 노이즈 억제와 드리프트 감지를 같은 다이얼로 절충하지 않기 위함.
DEFAULT_HEADING_ANCHOR_PULL_THRESHOLD_DEG = 20.0
DEFAULT_HEADING_ANCHOR_PULL_PERSIST_TICKS = 5

# 2026-10-04 추가 (실측 사고 deploy_20261004_133317.jsonl): "지속적으로 어긋남 →
# gps_track을 믿는다"는 pull 조건의 맹점 — gps_track 추정값이 양자화 격자각도(예:
# -90.0°)에 고정된 채 그대로 유지되면, 그 자체가 "오래 지속되는 반박"처럼 보여서
# 오히려 더 신뢰받아버림(진짜 드리프트와 구분 불가). 이번 run에서 위치는 계속
# 바뀌는데(§1-14의 위치-동결 가드로는 안 잡힘) gps_heading_rad 값만 -90.0°에
# 60초+(수십 틱) 고정된 채 route_bearing(실제로는 178.6°까지 진행)과 계속
# "지속 반박" 조건을 만족시켜서 끝까지 pull된 채 안 풀리는 사고가 실측됨. 추정값
# 자체가 이 틱수 이상 거의 안 바뀌면(진짜 이동 중이라면 GPS 노이즈로 자연스럽게
# 조금씩은 바뀌어야 함) "고장난 추정"으로 보고 pull을 보류한다.
DEFAULT_GPS_HEADING_FREEZE_PERSIST_TICKS = 10
GPS_HEADING_FREEZE_EPS_RAD = math.radians(0.5)

# 2026-10-04 추가 (실측 사고 deploy_20261004_143423.jsonl): GO LIVE 직후처럼
# past_track 누적이 적을 때 estimate_heading_from_track()의 "짧은 창"(fast_disp_m)
# 경로가 매 tick 다른 양자화 격자각도를 내놓을 수 있음(147.7°→162.5°→-162.5°→180°→
# -147.7° 식으로, 같은 값에 고정되는 게 아니라 계속 "다른 값으로 틀림" —
# DEFAULT_GPS_HEADING_FREEZE_PERSIST_TICKS 체크로는 못 잡는 변종). gps_heading_readiness()
# 의 기본 min_disp_m(아래)과 동일한 값 이상 누적돼야(=긴 창 평균화가 실제로 적용되는
# 지점) pull을 허용 — 그 전엔 route_bearing_anchor를 유지.
DEFAULT_HEADING_ANCHOR_PULL_MIN_ACCUM_PATH_M = 1.5

# 2026-10-04 추가: is_off_route() 기반 상시 이탈 감지(안전망) 관련 상수.
# 기존엔 is_off_route() 호출 자체가 allow_reroute(기본 False) 뒤에 숨어있어서
# "감지"와 "재라우팅 실행"이 같이 꺼져 있었음 — 그래서 heading_mode=route_bearing
# 테스트(deploy_20261004_111807.jsonl, 실측 4.85m 이동·목표쪽 진행 0.38m=약 85도
# 옆으로 샘)에서 아무도 못 알아챘음. 감지는 allow_reroute와 완전히 분리해서 항상
# 켜고(DEFAULT_ENABLE_OFF_ROUTE_SAFETY), 재라우팅(비용 크고 예전에 폭주 버그 있었음, 위
# REROUTE_COOLDOWN_S 참고) 대신 가벼운 조치만 취한다.
DEFAULT_ENABLE_OFF_ROUTE_SAFETY = False  # 기본 꺼짐 — 아직 실기기 미검증, 명시적으로 켜야 함
DEFAULT_OFF_ROUTE_SAFETY_THRESHOLD_M = 3.0  # is_off_route()와 동일 기준값 재사용
DEFAULT_OFF_ROUTE_SAFETY_ACTION = "steer"  # "steer"(route_bearing_to_control로 직접 조향) | "stop"(정지)

# 2026-10-04 추가 (사용자 요청): "OSM map이 직진 경로를 보여주면 로봇도 반드시
# 직진하게" — 모델 자체의 미세 좌편향(heading_route_diff_deg=0인 완벽한 지도에서도
# 재현됨, deploy_20261004_104301.jsonl tick 37)은 지도를 아무리 정확하게 넣어도
# 안 없어지는 걸 직접 확인했음(§1-8/§1-9/§1-10) — 그래서 "직진 구간에서는 모델
# 예측 자체를 안 쓰고 route_bearing으로 바로 조향"하는 게 유일하게 그 요구사항을
# 구조적으로 보장하는 방법. 장애물 회피는 이 프로젝트 범위 밖(vanilla 환경 가정,
# 사용자 확인)이라 모델의 판단력을 여기서 포기하는 트레이드오프를 감수하기로 함.
# 짧은 lookahead(바로 앞)와 긴 lookahead(더 멀리)의 route_bearing 차이가 작으면
# "당분간 직진"으로 판단 — 턴이 다가오면 두 값이 벌어지므로 자동으로 모델 예측이
# 다시 쓰임(이 구간은 지금까지 보니 모델이 지도 내용을 비교적 잘 따라감).
DEFAULT_ENABLE_STRAIGHT_SEGMENT_OVERRIDE = False  # 기본 꺼짐 — 명시적으로 켜야 함
DEFAULT_STRAIGHT_NEAR_LOOKAHEAD_M = 2.0
DEFAULT_STRAIGHT_FAR_LOOKAHEAD_M = 12.0
DEFAULT_STRAIGHT_ANGLE_THRESHOLD_DEG = 20.0

# 2026-10-04 추가 (실측 사고, deploy_20261004_123749.jsonl): GPS 위치(lat/lon)가
# frodobot_raw["speed"]는 계속 0보다 크게 보고되는데도 22초+ 동안 한 비트도 안 바뀌는
# "GPS 동결"이 실측됨 — fix_quality=2/gps_fix_ok=True로 계속 "정상"처럼 보여서 기존
# 체크(위 GPS fix 없음/품질 체크)로는 전혀 못 잡음. 동결 동안 route_bearing_rad()와
# gps_track heading 둘 다 똑같이 멈춘 값에 묶여서, 모델이 더 이상 현실과 안 맞는 지도를
# 계속 "충실히" 따라가며 같은 방향(예: 우회전)을 끝없이 반복하게 됨. 게다가
# heading_mode=route_bearing_anchored의 pull 조건("gps_track이 N틱 연속 반박하면
# 신뢰")이 "진짜 반박"과 "동결로 생기는 가짜 반박"을 구분 못 해서, 동결된 gps_track
# 값을 그대로 믿고 끝까지 안 풀리는 2차 버그로 이어짐(Harness 1 cross-session 분석,
# docs/experiment_log.md §1-12). 위치가 실제로 min_disp_m 이상 못 움직인 채
# PERSIST_TICKS 이상 지속되고 그동안 로봇 자신이 보고하는 속도는 SPEED_THRESHOLD보다
# 크면(=움직이고 있다고 자기보고하는데 위치만 안 바뀜) "동결"로 판정한다. 다른
# 메커니즘들과 달리 이건 트레이드오프가 없는 순수 안전장치라 기본 ON(옵트아웃 방식) —
# near_goal_override_dist_m(기본 8.0, 0으로 끄는 방식)과 동일한 선례.
DEFAULT_ENABLE_GPS_FREEZE_GUARD = True
DEFAULT_GPS_FREEZE_MIN_DISP_M = 0.03
DEFAULT_GPS_FREEZE_PERSIST_TICKS = 10
DEFAULT_GPS_FREEZE_SPEED_THRESHOLD_MPS = 0.15

# 2026-10-04 추가: SDK 서버 쪽 헤드리스 브라우저가 1~2초 내로 저절로 회복되는 일시적
# hiccup(ReadTimeout, 빈 JSON 응답)을 오늘 여러 번 실측(deploy_20261004_123354/
# 125728/131744.jsonl 등) — 지금까진 이런 일시적 실패도 전체 프로세스를 바로
# 죽였음(poll_frodobot()/send_control()이 try/except 없이 바로 예외를 올림).
# 짧게 재시도해서 이런 hiccup을 흡수하되, 재시도까지 다 실패하면(=진짜 통신 불능)
# 여전히 예외를 그대로 올려서 run()의 "정지 시도 후 프로세스 종료" 안전 원칙은
# 그대로 유지한다 — 여기서 에러를 삼키고 계속 진행하는 일은 절대 없음(2026-09-26
# 주석의 안전 원칙과 동일).
NETWORK_RETRY_COUNT = 2
NETWORK_RETRY_BACKOFF_S = 0.3

# 2026-09-18: is_off_route() 임계값을 15m→3m로 낮췄더니, OSRM이 자체적으로 요청 좌표를
# 가장 가까운 매핑된 길(way)로 "스냅"하는 거리가 그보다 큰 지점(예: 매핑된 보행로가 없는
# 개활지, 실측 3.38m)에서는 재라우팅을 해도 새 경로 시작점이 여전히 3m 넘게 떨어져 있어
# is_off_route()가 계속 True → 재라우팅을 무한 반복하는 폭주가 실측됨
# (deploy_20260918_184622.jsonl, 0.3~0.5초마다 reroute 이벤트). 원인(매핑 안 된 지점에서
# 출발) 자체는 임계값 튜닝으로 못 고치지만, 적어도 매 틱 재쿼리하는 폭주는 쿨다운으로 막는다.
REROUTE_COOLDOWN_S = 3.0

# 2026-09-26: 목표 이 거리(m) 이내에서는 재라우팅을 하지 않음 — 실제 주행에서 목표
# ~5.5m 앞 reroute가 OSRM 보행로망을 따라 헤어핀(왔다갔다)형 새 경로를 만들어 로봇이
# 계속 도는 것처럼 보이는 문제가 실측됨(deploy_20260926_132829.jsonl). 이 거리 안에서는
# dist_to_goal_m 기반 도착 정지(DEFAULT_GOAL_REACH_THRESHOLD_M)가 곧 작동하므로 경로
# 재계산이 불필요.
REROUTE_DISABLE_NEAR_GOAL_M = 10.0

# ── GPS/IMU 자이로 융합 (실험적, 기본 꺼짐) ──────────────────────────────────
# GPS-heading을 못 구하는 구간(gps_ema_hold)에서 지금은 마지막 EMA 값을 그냥
# 고정해서 쓰는데, 그 사이에 로봇이 실제로 방향을 틀면 반영이 안 됨. 대신
# frodobot_raw["gyros"](수직축 각속도)를 적분해서 "그동안 얼마나 돌았는지"만큼
# EMA heading을 보정하는 방식 — IMU 컴퍼스의 "절대값"은 신뢰 안 하고(자기장 간섭으로
# 세션마다 오차가 +11°~+169°까지 들쭉날쭉했음, 고정 오프셋이 아님) "상대 회전량"만
# 신뢰하는 접근. 축/부호/단위(deg/s 가정)를 실측으로 아직 검증 안 했으므로 기본은
# OFF — 다음 실배포에서 켜보고 로그의 gps_ema_gyro 구간 궤적이 실제와 맞는지 확인
# 후 계속 켤지 결정할 것.
USE_GYRO_FUSION = False
GYRO_YAW_AXIS_SIGN = 1.0   # 부호가 반대로 나오면 -1.0으로 뒤집을 것
GYRO_UNIT_IS_DEG = True    # frodobot_raw["gyros"] 값이 deg/s라고 가정 (rad/s면 False로)

# ── Route-aligned initial heading bootstrap (2026-09-25 추가) ────────────────
# 배경: GPS course heading(estimate_heading_from_track)은 실제 이동(누적 1.5m)이
# 있어야만 생기는데, 로봇은 non-zero 명령을 받아야 이동한다 — 가만히 서서
# GPS fix만 오래 기다려도 이 값은 생기지 않는 bootstrap deadlock이 있음(세션에서
# 코드로 재확인). 이번 실험에서는 사용자가 출발 전 로봇을 실제 route 방향으로
# 물리적으로 정렬해두고, 그 방향(OSRM route의 "현재 위치 바로 앞" tangent)을
# 대시보드에서 명시적으로 확인한 뒤 initial heading으로 쓴다.
#
# 중요: 이 값은 _heading_ema_vec에 절대 섞지 않는다(seed하지 않음) — EMA는
# "실측 GPS course heading으로만" 초기화되도록 순수하게 유지해서, 사람이 정렬을
# 잘못했더라도 그 오차가 이후 실측 평균에 오염되어 남지 않게 한다. 대신 EMA가
# 아직 없을 때(gps_heading 없음 + _heading_ema_vec None)의 "최후 대안"을
# imu_fallback에서 route_aligned로 한 단계 올린다 — imu_fallback은 여전히
# route_aligned조차 없을 때만 쓰이는 최후 폴백으로 코드에 남겨둠(제거하지 않음).
#
# lookahead 값 선택 근거: route는 1m 간격으로 densify됨(_densify_route). 1m(=세그먼트
# 1개)는 OSRM 폴리라인 자체의 정점 단위 잡음에 취약하고, 기존 디버그용 5m lookahead는
# 출발부가 바로 꺾이는 경로에서는 "지금 서야 할 방향"이 아니라 "5m 뒤 방향"을 대표해버림.
# 2m(세그먼트 2개 평균)로 국소 잡음은 어느 정도 평균화하면서, 5m보다 훨씬 로컬한
# tangent를 대표하도록 절충함.
INITIAL_HEADING_LOOKAHEAD_M = 2.0

# ARMED 상태에서 이 시간(초) 안에 GO LIVE를 안 누르면 자동으로 DRY_RUN으로 되돌림
# (실수로 ARM해두고 잊어버린 채 다른 걸 하다가 뒤늦게 GO LIVE를 누르는 상황 방지).
ARM_TIMEOUT_S = 30.0


def estimate_yaw_delta_from_gyro(raw_data, dt_s):
    """자이로 수직축(z, gravity와 같은 축 — accels z≈1g로 확인됨) 평균 각속도로
    dt_s 동안의 heading 변화량(rad)을 추정. USE_GYRO_FUSION 실험 전용, 검증 전."""
    gyros = raw_data.get("gyros")
    if not gyros or dt_s <= 0:
        return 0.0
    mean_z = sum(g[2] for g in gyros) / len(gyros)
    if GYRO_UNIT_IS_DEG:
        mean_z = math.radians(mean_z)
    return GYRO_YAW_AXIS_SIGN * mean_z * dt_s

# heading 변환: 원래 -orientation/180*pi 하나만 썼는데(90도 보정이 빠져있어 지도가
# 어긋나 보일 수 있다는 가설이 있었음), 실기기 테스트 결과 "얼마나 어긋났는지"를
# 추측으로 고치기보다 GPS 궤적 기반 heading(estimate_heading_from_track, 학습 데이터
# heading과 동일한 산출 방식)과 직접 비교해서 진단하는 쪽으로 방향을 잡음 — step()의
# heading_diff_deg 로그 참고. 여기서 고정 오프셋으로 성급하게 "고치지" 않는다.


def decode_frame(b64_str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(b64_str))).convert("RGB")


LAT_M = 111320.0  # 위도 1도당 미터 (근거리 근사)
MIN_NET_DISP_M = 0.3  # "거의 정지"만 걸러내는 순변위 하한 (min_disp_m보다 훨씬 작음)
DEFAULT_GOAL_REACH_THRESHOLD_M = 2.0  # 이 거리 이내면 도착으로 간주하고 영구 정지

# 2026-10-03 추가 (Harness 1의 §1-8 분석, experiment_log.md 참고): future-route가
# 화면상 bbox_h<~25px(27m 체크포인트 기준)로 짧아지면, 실제 지도 내용(곡률 유무·
# 방향)과 완전히 무관하게 모델이 결정론적으로 좌회전함(실측: 직선 경로 177/177,
# 실제 우회전 경로에서도 26:2·22:2). 곡률 압축이 아니라 "짧은 future-route 선분"
# 자체가 트리거인 OOD 현상으로 보임(근본원인은 §1-3 학습데이터 분석이 Arrow 접근
# 풀려야 확정 — 이 상수는 재학습 전까지의 완화책).
# 27m 체크포인트에서 bbox_h<25px가 실측 dist_to_goal_m 대략 7-8m에 대응(직접 측정,
# bbox_h-거리 관계가 근거리에서 비선형이라 공식 환산이 아니라 실측값 사용) — 8.0m로
# 약간 보수적으로 잡음(25-40px 구간은 아직 실제 내용을 어느 정도 반영하는 것으로
# 측정됨, §1-8 표 참고). **다른 map_range_m/체크포인트로 바꾸면 이 값도 재측정 필요**
# — 공식(px/m 비율)으로 자동 환산하지 않는 이유는 위와 동일(비선형).
DEFAULT_NEAR_GOAL_OVERRIDE_DIST_M = 8.0


def latlon_distance_m(lat1, lon1, lat2, lon2):
    """근거리 등적원통 근사 — 이 파일 전체(estimate_heading_from_track 등)와 동일한
    LAT_M 기반 근사를 그대로 재사용(별도 haversine 등 다른 근사식을 새로 쓰지 않음)."""
    dlat = (lat2 - lat1) * LAT_M
    dlon = (lon2 - lon1) * LAT_M * math.cos(math.radians(lat1))
    return math.hypot(dlat, dlon)


def estimate_heading_from_track(past_track, min_disp_m=1.5, fast_disp_m=0.5, fast_disagree_deg=35.0):
    """로봇이 실제로 지나온 GPS 궤적(past_track)에서 진행방향을 추정.
    osm_map_generator_rides11.py::estimate_headings()와 동일한 공식(atan2(북쪽성분, 동쪽성분),
    East=0/North=+90 CCW) — 학습 데이터의 heading이 바로 이 방식으로 만들어졌음.

    직전 두 점만 비교하지 않는다 — GPS가 ~0.42m 단위로만 갱신되는 양자화 잡음 때문에,
    인접한 두 fix만 보면 위도/경도 중 어느 쪽이 먼저 양자화 경계를 넘었는지에 따라
    heading이 순간적으로 남↔서로 튀는 문제가 실측으로 확인됨(docs/0825.md 1-1, 원래는
    시각화 스크립트에서 발견·수정된 문제). 그 수정과 동일하게, 누적 이동거리가
    min_disp_m 이상 되는 지점까지 거슬러 올라가 그 구간 전체의 변위로 추정해서
    양자화 스텝 하나짜리 잡음을 평균화한다. 이동량이 부족하면(정지/막 시작) None.

    2026-09-18: 0.8m로 낮췄다가 원복함 — 순이동이 작고 지그재그인 구간에서 0.8m는 노이즈를
    못 걸러내 GPS-heading이 엉뚱한 방향(최대 169° 차이)으로 튀는 부작용이 실측 확인됨
    (deploy_20260918_180335.jsonl). 1.5m가 더 안정적이라 원래 값으로 복귀.

    2026-09-18(2차): 마지막 체크가 "시작-끝 순변위(직선거리) >= min_disp_m"였는데, 로봇이
    완전 직진이 아니라 약간 지그재그(GPS 위경도 축별 비동기 양자화 + 실제 약간의 굴곡)로
    움직이면 누적 경로 길이(cum_m)는 min_disp_m을 넘겨도 순변위는 계속 그보다 작게 남아
    30초 넘게 계속 None만 반환하는 문제가 실측됨 — 그 사이 IMU가 30°+ 드리프트해서 지도가
    계속 잘못된 방향으로 그려짐. cum_m으로 이미 "양자화 잡음 평균화" 조건(min_disp_m 이상
    경로를 거슬러 올라감)을 충족했다면, 마지막 체크는 "순변위가 거의 0인 진짜 정지 상태"만
    걸러내면 되므로 훨씬 작은 MIN_NET_DISP_M로 완화. cum_m이 min_disp_m에 못 미친(이력
    자체가 부족한) 경우엔 기존처럼 min_disp_m 그대로 요구.

    2026-10-03 추가 (fast_disp_m/fast_disagree_deg): 위 1.5m 고정 창은 실제로 막
    꺾은 직후에도 "꺾기 전" 방향을 새로 1.5m 이동이 쌓일 때까지(저속 구간에서는
    수 초~십수 초) 계속 돌려주는 지연(lag)이 있음을 실측으로 확인
    (deploy_20261003_135403.jsonl — GPS 자체는 깨끗한데(net==accumulated)
    heading_route_diff_deg가 75°→105°로 29틱 연속 벌어진 구간, 직진 후 우회전
    테스트 경로의 실제 턴 구간과 일치). 자이로 적분으로 고치려 했으나
    SDK가 주는 자이로 버스트가 tick 간격의 16~25%만 커버해서 부호조차 IMU 델타와
    무상관(51~58% 일치, 동전던지기 수준) — 데이터 커버리지 문제라 포기함
    (estimate_yaw_delta_from_gyro()는 남겨두되 USE_GYRO_FUSION=False 유지).

    대신 뒤로 거슬러 올라가는 같은 루프 안에서 "짧은 창"(fast_disp_m, 기본 0.5m —
    양자화 잡음 한 스텝(~0.42m)보다는 커야 함)의 방향도 같이 계산해둔다. 긴 창
    결과와 fast_disagree_deg(기본 35°) 이상 벌어지고, 짧은 창 자체도 순변위가
    MIN_NET_DISP_M을 넘어 신뢰할 만하면 짧은 창 쪽(최근 방향)을 대신 채택 — 그 외
    (방향이 일치하거나 짧은 창이 아직 불충분)에는 기존 긴 창 그대로 사용해서
    직진 구간의 잡음 평균화 효과는 그대로 유지한다."""
    if len(past_track) < 2:
        return None
    lat_end, lon_end = past_track[-1]
    lat_start, lon_start = past_track[-2]
    cum_m = 0.0
    reached_min_path = False
    fast_fix = None  # (lat, lon) at the point where cum_m first reaches fast_disp_m
    for i in range(len(past_track) - 2, -1, -1):
        lat_a, lon_a = past_track[i]
        lat_b, lon_b = past_track[i + 1]
        dlat = (lat_b - lat_a) * LAT_M
        dlon = (lon_b - lon_a) * LAT_M * math.cos(math.radians(lat_a))
        cum_m += math.hypot(dlat, dlon)
        lat_start, lon_start = lat_a, lon_a
        if fast_fix is None and cum_m >= fast_disp_m:
            fast_fix = (lat_a, lon_a)
        if cum_m >= min_disp_m:
            reached_min_path = True
            break
    dlat = (lat_end - lat_start) * LAT_M
    dlon = (lon_end - lon_start) * LAT_M * math.cos(math.radians(lat_start))
    net_disp_m = math.hypot(dlat, dlon)
    required_m = MIN_NET_DISP_M if reached_min_path else min_disp_m
    if net_disp_m < required_m:
        return None
    long_heading = math.atan2(dlat, dlon)

    if fast_fix is not None:
        flat, flon = fast_fix
        fdlat = (lat_end - flat) * LAT_M
        fdlon = (lon_end - flon) * LAT_M * math.cos(math.radians(flat))
        fast_net_disp_m = math.hypot(fdlat, fdlon)
        if fast_net_disp_m >= MIN_NET_DISP_M:
            fast_heading = math.atan2(fdlat, fdlon)
            diff_deg = (math.degrees(fast_heading - long_heading) + 180) % 360 - 180
            if abs(diff_deg) >= fast_disagree_deg:
                return fast_heading
    return long_heading


def gps_heading_readiness(past_track, min_disp_m=1.5):
    """진단 전용 — estimate_heading_from_track()과 동일한 누적 방식으로
    accumulated path length / net displacement '숫자'만 반환(판정 없음).
    estimate_heading_from_track() 자체의 판정 로직은 건드리지 않고, 대시보드에서
    "왜 아직 GPS heading이 없는지"를 사람이 수치로 볼 수 있게 하려고 별도로 둠.
    반환: (accumulated_path_m, net_displacement_m)."""
    if len(past_track) < 2:
        return 0.0, 0.0
    lat_end, lon_end = past_track[-1]
    lat_start, lon_start = past_track[-2]
    cum_m = 0.0
    for i in range(len(past_track) - 2, -1, -1):
        lat_a, lon_a = past_track[i]
        lat_b, lon_b = past_track[i + 1]
        dlat = (lat_b - lat_a) * LAT_M
        dlon = (lon_b - lon_a) * LAT_M * math.cos(math.radians(lat_a))
        cum_m += math.hypot(dlat, dlon)
        lat_start, lon_start = lat_a, lon_a
        if cum_m >= min_disp_m:
            break
    dlat = (lat_end - lat_start) * LAT_M
    dlon = (lon_end - lon_start) * LAT_M * math.cos(math.radians(lat_start))
    net_disp_m = math.hypot(dlat, dlon)
    return cum_m, net_disp_m


def clip_control(linear_vel, angular_vel, maxv=MAX_V, maxw=MAX_W):
    """LogoNav_frodobot.py::policy_calc()의 클리핑 로직 재사용 —
    linear/angular 비율(rd)을 유지한 채 한계 안으로 스케일링."""
    if abs(linear_vel) <= maxv and abs(angular_vel) <= maxw:
        return linear_vel, angular_vel
    if abs(angular_vel) <= 1e-6:
        return maxv * np.sign(linear_vel), 0.0
    rd = linear_vel / angular_vel
    if abs(rd) >= maxv / maxw:
        return maxv * np.sign(linear_vel), maxv * np.sign(angular_vel) / abs(rd)
    return maxw * np.sign(linear_vel) * abs(rd), maxw * np.sign(angular_vel)


def route_bearing_to_control(route_bearing_rad, map_heading_rad, target_time_s):
    """2026-10-03 추가(Harness 1 §1-8/§3-1 제안) — near-goal 구간(future-route가
    화면상 너무 짧아져서 모델 raw 출력을 못 믿는 구간, DEFAULT_NEAR_GOAL_OVERRIDE_DIST_M
    참고)에서 waypoint_to_control()의 모델 예측 대신 쓰는 대체 조향. "route_bearing
    방향으로 target_time_s 뒤에 도착한다"는 가상 목표를 가정하고 같은 atan2/시간분모
    방식(waypoint_to_control()과 동일한 target_time_s, 보통 target_step=2의 2.1초)으로
    변환 — 모델 추론 없이 route_bearing_rad()(GPS 궤적 기반 목표 방향, 이미 매 틱
    계산됨)만으로 산출 가능."""
    diff = (route_bearing_rad - map_heading_rad + math.pi) % (2 * math.pi) - math.pi
    angular = diff / target_time_s
    return float(np.clip(MAX_V, 0, MAX_V * 2)), float(np.clip(angular, -MAX_W * 2, MAX_W * 2))


class OmniVLAEdgeDeployment:
    def __init__(self, ckpt_path, map_range_m, goal_lat, goal_lon, device=None,
                 debug_port=8080, dry_run=False, heading_mode="auto",
                 goal_reach_threshold_m=DEFAULT_GOAL_REACH_THRESHOLD_M,
                 initial_heading_lookahead_m=INITIAL_HEADING_LOOKAHEAD_M,
                 allow_reroute=False,
                 near_goal_override_dist_m=DEFAULT_NEAR_GOAL_OVERRIDE_DIST_M,
                 heading_anchor_pull_threshold_deg=DEFAULT_HEADING_ANCHOR_PULL_THRESHOLD_DEG,
                 heading_anchor_pull_persist_ticks=DEFAULT_HEADING_ANCHOR_PULL_PERSIST_TICKS,
                 enable_off_route_safety=DEFAULT_ENABLE_OFF_ROUTE_SAFETY,
                 off_route_safety_threshold_m=DEFAULT_OFF_ROUTE_SAFETY_THRESHOLD_M,
                 off_route_safety_action=DEFAULT_OFF_ROUTE_SAFETY_ACTION,
                 enable_straight_segment_override=DEFAULT_ENABLE_STRAIGHT_SEGMENT_OVERRIDE,
                 straight_near_lookahead_m=DEFAULT_STRAIGHT_NEAR_LOOKAHEAD_M,
                 straight_far_lookahead_m=DEFAULT_STRAIGHT_FAR_LOOKAHEAD_M,
                 straight_angle_threshold_deg=DEFAULT_STRAIGHT_ANGLE_THRESHOLD_DEG,
                 enable_gps_freeze_guard=DEFAULT_ENABLE_GPS_FREEZE_GUARD,
                 gps_freeze_min_disp_m=DEFAULT_GPS_FREEZE_MIN_DISP_M,
                 gps_freeze_persist_ticks=DEFAULT_GPS_FREEZE_PERSIST_TICKS,
                 gps_freeze_speed_threshold_mps=DEFAULT_GPS_FREEZE_SPEED_THRESHOLD_MPS):
        # 2026-09-25: --dry_run의 의미가 "이 프로세스는 GO LIVE 자체를 영구히
        # 거부하는 하드 락"으로 바뀜(순수 검증 세션용). 기본(플래그 없음)은
        # DRY_RUN 상태로 시작하되, 대시보드에서 정렬확인→ARM→GO LIVE를 거치면
        # "같은 프로세스 안에서" LIVE로 전환 가능 — frame_buffer/past_track/
        # heading EMA/route가 전부 그대로 유지됨(재시작 시 전부 날아가던
        # 이전 구조의 핵심 문제를 해결하기 위함).
        self.dry_run_lock = dry_run
        self.control_stage = "DRY_RUN"  # "DRY_RUN" / "ARMED" / "LIVE"
        self._armed_ts = None
        self._pending_commands = queue.Queue()

        # 2026-09-26 추가: heading_mode
        #   "auto"(기본, 기존 동작) — route_aligned는 실측 GPS heading이 확보되기
        #     전까지의 임시 부트스트랩일 뿐, 확보되는 즉시 gps_track/gps_ema_hold로
        #     자동 전환됨.
        #   "route_aligned_fixed" — 정렬 확인으로 고정한 값을 런 내내 그대로 사용,
        #     실측 GPS heading이 이후 얼마나 잡히든 절대 넘기지 않음. 실외 테스트에서
        #     저속/근거리 구간의 GPS 잡음이 gps_track을 계속 흔드는 문제(순변위가
        #     누적경로보다 훨씬 작은 지그재그, heading 100°+ 튐)가 실측되어 대안으로
        #     추가함 — 실제 턴이 있는 경로에서는 그 턴을 아예 못 따라가는 단점이 있음
        #     (2026-10-04 확인, 턴 있는 경로 테스트에는 안 맞음).
        #   "route_bearing"(2026-10-04 추가) — 매 tick route_bearing_rad()를 그대로
        #     heading으로 씀(gps_track/EMA 완전히 안 씀). route_aligned_fixed처럼
        #     GPS 양자화 노이즈에 영향을 안 받으면서도, route_bearing은 route 지오메트리
        #     기반이라 실제 턴이 있으면 자연스럽게 따라감(2026-10-04 실측: 같은 run에서
        #     route_bearing이 54틱 연속 완전히 고정됐던 반면 gps_track 기반 heading은
        #     같은 구간에서 -90°~180°까지 요동침 — docs/experiment_log.md §1-9/1-10).
        #     단점(사용자 확인 후 채택, 2026-10-04): 로봇이 실제로 경로를 벗어나도
        #     이 모드는 그걸 전혀 못 알아챔(지도가 항상 "경로 위에 있다"고 가정하고
        #     그려짐) — 지금처럼 짧고 통제된 테스트 구간에서만 쓰기로 함.
        assert heading_mode in ("auto", "route_aligned_fixed", "route_bearing", "route_bearing_anchored"), \
            f"알 수 없는 heading_mode: {heading_mode!r}"
        self.heading_mode = heading_mode

        # run_id를 여기서 미리 만들어서 JSONL 로그/ReplayLogger(아래)와 대시보드
        # 녹화 파일(dashboard_capture.py)이 같은 이름을 공유하게 함.
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.state = DeploymentState()
        self.debug_port = debug_port
        self._dashboard_recorder = None
        if debug_port:
            start_debug_server(self.state, self._pending_commands, port=debug_port)
            # 2026-10-03: GO LIVE 순간 PNG 한 장 대신, 프로세스 시작(대시보드가 뜨는
            # 즉시)부터 run() 종료(finally)까지 전체를 동영상으로 녹화 — 주행 전체
            # 과정을 나중에 다시 볼 수 있게. 실패해도(Chrome 없음 등) best-effort라
            # 제어 루프에 영향 없음(DashboardRecorder 자체 안전설계 참고).
            self._dashboard_recorder = DashboardRecorder(
                debug_port, DASHBOARD_CAPTURE_DIR, run_id,
                on_done=lambda p, n: self.state.log(f"대시보드 녹화 저장됨: {p} ({n}프레임)"),
                on_error=lambda e: self.state.log_error(f"대시보드 녹화 실패(무시하고 계속): {e!r}"),
            ).start()

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = OmniVLA_edge_odom(**MODEL_PARAMS)
        ckpt = torch.load(ckpt_path, map_location="cpu")
        self.model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=True)
        self.model.to(self.device).eval()
        print(f"[deploy] Loaded checkpoint: {ckpt_path}  (device={self.device})")

        self.map_builder = LiveMapBuilder(map_range_m=map_range_m)
        self.goal_lat, self.goal_lon = goal_lat, goal_lon
        # 경로는 배포 시작 시 1번만 계산해서 캐싱 (osmnav 구조 참고 — 매 프레임 재쿼리 안 함).
        # 시작 위치를 아직 모르므로, 첫 poll_frodobot() 이후 run()에서 set_goal() 호출.
        self._route_initialized = False
        self._is_off_route_now = False  # 2026-10-04 추가: _ensure_route_initialized()가 매 tick 갱신

        # 2026-09-26 추가: goal 도착 감지 — 이전엔 목표 근처에 도달해도 계속 명령이
        # 나갔음(실제 로그로 확인된 갭). 한 번 도착(_goal_reached=True)하면 이후
        # 절대 다시 풀리지 않음(latched) — run()의 전송 게이트에서 heading_trustworthy와
        # 같은 자리에서 AND 조건으로 확인.
        self.goal_reach_threshold_m = goal_reach_threshold_m
        self._goal_reached = False

        # 2026-10-03 추가 (Harness 1 §1-8/§3-1): 이 거리 이내에서는 모델 raw 출력
        # 대신 route_bearing_to_control()로 대체 — 위 DEFAULT_NEAR_GOAL_OVERRIDE_DIST_M
        # 상수 설명 참고. 0 이하로 주면 완전히 비활성화(순수 모델 출력만 사용,
        # 기존 동작) — 비교 테스트용으로 남겨둠.
        self.near_goal_override_dist_m = near_goal_override_dist_m

        # 2026-10-04 추가 — 전부 docs/experiment_log.md의 Harness 1/사용자 논의 결과를
        # config/CLI로 독립 실험 가능하게 분리한 것 (기본은 전부 꺼짐/기존 동작 유지,
        # 하나씩 켜서 비교 가능).
        self.heading_anchor_pull_threshold_deg = heading_anchor_pull_threshold_deg
        self.heading_anchor_pull_persist_ticks = heading_anchor_pull_persist_ticks
        self.enable_off_route_safety = enable_off_route_safety
        self.off_route_safety_threshold_m = off_route_safety_threshold_m
        assert off_route_safety_action in ("stop", "steer"), \
            f"알 수 없는 off_route_safety_action: {off_route_safety_action!r}"
        self.off_route_safety_action = off_route_safety_action
        self.enable_straight_segment_override = enable_straight_segment_override
        self.straight_near_lookahead_m = straight_near_lookahead_m
        self.straight_far_lookahead_m = straight_far_lookahead_m
        self.straight_angle_threshold_deg = straight_angle_threshold_deg

        # 2026-10-04 추가 (실측 사고, deploy_20261004_123749.jsonl — 위 DEFAULT_ENABLE_GPS_FREEZE_GUARD
        # 상수 설명 참고): GPS 위치 동결 감지. past_track과 별개로 독립 추적(정렬확인/보정
        # 이벤트로 past_track이 clear돼도 동결 감지는 끊기지 않게).
        self.enable_gps_freeze_guard = enable_gps_freeze_guard
        self.gps_freeze_min_disp_m = gps_freeze_min_disp_m
        self.gps_freeze_persist_ticks = gps_freeze_persist_ticks
        self.gps_freeze_speed_threshold_mps = gps_freeze_speed_threshold_mps
        self._gps_freeze_ticks = 0
        self._gps_freeze_last_latlon = None
        self._gps_freeze_any_speed_seen = False
        self._gps_frozen_now = False

        # 2026-09-26 추가: INITIAL_HEADING_LOOKAHEAD_M을 CLI로 조절 가능하게 함 —
        # 실제 주행에서 목표 근처(dist_to_goal≈6.9m)에서 정렬 확인을 눌렀더니,
        # 짧은 2m lookahead가 마침 경로가 꺾이는 지점 근처를 잡아서 실제 경로 방향
        # (route_bearing_deg)과 56~58°나 어긋난 heading이 그대로 고정되는 사고가
        # 실측됨(deploy_20260926_135303.jsonl). 근본적인 자체 검증(cross-check) 로직은
        # 아직 안 넣었고(다음 라운드), 우선 이 값을 늘려서(예: 5.0m 이상) 국소적인
        # 꺾임에 덜 민감하게 만들 수 있도록 임시로 조절 가능하게만 함.
        self.initial_heading_lookahead_m = initial_heading_lookahead_m

        # 2026-09-26 추가: 재라우팅 기본 비활성화 — 목표에서 16m 넘게 떨어진 거의
        # 정지 상태에서도 is_off_route()가 쿨다운마다 계속 True를 반환해 reroute가
        # 9번 연속 발생하고, 그때마다 OSRM이 다른 경로를 반환해 route_bearing_deg가
        # 요동치는 문제가 실측됨(deploy_20260926_163849.jsonl). 최초 route_init만
        # 쓰고 런 내내 고정하는 쪽을 기본값으로 바꿈 — 상세 근거는
        # _ensure_route_initialized() 참고.
        self.allow_reroute = allow_reroute

        self.obs_transform = transforms.Compose([
            transforms.Resize((96, 96)),
            transforms.ToTensor(),
            transforms.Normalize(IMG_MEAN, IMG_STD),
        ])

        # 최근 카메라 프레임 (0.3초 간격으로 채워짐, N_CTX+1개 유지)
        self.frame_buffer = deque(maxlen=N_CTX + 1)
        # frame_buffer와 1:1로 나란히 유지되는 replay_logger frame_id (deterministic
        # replay용 — 어느 저장된 프레임 파일들이 지금 컨텍스트를 이루는지 추적)
        self.frame_buffer_ids = deque(maxlen=N_CTX + 1)
        self.last_frame_time = 0.0

        # 로봇이 실제로 지나온 GPS 기록 (odom map의 회색 past 선용, 최근 것만 유지)
        self.past_track = deque(maxlen=200)

        # GPS-heading EMA 상태 (단위원 위 복소수로 유지 — 각도는 선형평균하면 안 되고
        # wraparound(-180/+180 경계)를 다뤄야 하므로 벡터로 평균낸 뒤 각도를 복원함)
        self._heading_ema_vec = None

        # 2026-10-03 추가: heading_route_diff_deg가 HEADING_ROUTE_CORRECTION_THRESHOLD_DEG를
        # 넘은 채 연속으로 몇 틱째인지 — HEADING_ROUTE_CORRECTION_PERSIST_TICKS 도달 시
        # route_bearing으로 강제 재동기화.
        self._heading_divergence_ticks = 0

        # 2026-10-04 추가: heading_mode="route_bearing_anchored" 전용 — gps_track이
        # route_bearing과 DEFAULT_HEADING_ANCHOR_PULL_THRESHOLD_DEG 이상 연속으로 몇 틱째
        # 어긋나는지. DEFAULT_HEADING_ANCHOR_PULL_PERSIST_TICKS 도달 시에만 그 tick 한정으로
        # gps_track 쪽을 신뢰(위 _heading_divergence_ticks와 반대 방향 로직 — 기본은
        # route_bearing, gps_track은 지속적 반증이 쌓여야만 예외적으로 끌어당김).
        self._heading_anchor_pull_ticks = 0

        # 2026-10-04 추가 (실측 사고 deploy_20261004_133317.jsonl): gps_track 추정값
        # 자체가 양자화 격자각도에 고정된 채 수십 틱 유지되는 현상 감지용 — 위
        # _heading_anchor_pull_ticks(위치는 바뀌는데 heading 추정만 고장난 경우, 아래
        # DEFAULT_GPS_HEADING_FREEZE_PERSIST_TICKS 설명 참고)의 "진짜 반박 vs 고장난
        # 추정" 구분을 보완.
        self._gps_heading_freeze_ticks = 0
        self._gps_heading_freeze_last_rad = None

        # 마지막 재라우팅 시각 (REROUTE_COOLDOWN_S 참고 — 무한 재라우팅 스팸 방지)
        self._last_reroute_ts = 0.0

        # 직전 step()의 record ts (USE_GYRO_FUSION 적분용 dt 계산)
        self._prev_step_ts = None

        # 최근 계산한 route bearing(rad) — build_inputs()가 채우고 step()이 읽어서
        # heading_route_diff_deg를 계산(대시보드/로그 디버그 필드, 2026-09-25 추가)
        self._last_route_bearing_rad = None
        self._last_map_replay_path = None

        # 2026-09-25 추가: route-aligned initial heading bootstrap 상태.
        # _heading_ema_vec는 절대 이 값으로 시드하지 않음(위 상수 설명 참고) —
        # 실측 GPS course heading이 없을 때의 "최후 대안"으로만 쓰임.
        self._route_aligned_heading_rad = None
        self._route_aligned_confirmed_ts = None
        self._last_lat, self._last_lon = None, None   # poll_frodobot()이 매 틱 갱신
        self._last_map_heading_source = None            # step()이 매 틱 갱신, run()의 heading-readiness 게이트가 읽음

        # ── 데이터분석용 로그 (JSONL, 실행마다 날짜시간별 파일) ──
        # run_id는 __init__ 맨 위에서 이미 만들어둠(대시보드 녹화 파일명과 공유).
        log_dir = REPO_ROOT / "deployment" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = log_dir / f"deploy_{run_id}.jsonl"
        self._log_fp = open(self.log_path, "a", buffering=1, encoding="utf-8")
        print(f"[deploy] 로그 저장 경로: {self.log_path}")

        # ── Deterministic replay logging (2026-09-25 추가) ──
        # 실제 predict_waypoints()에 들어간 카메라 6프레임 + 최종 지도 이미지를
        # tick_id로 묶어서 저장 — 2026-09-18 세션에서 대시보드 카메라로 사후
        # 재구성을 시도했다가 실제 모델 출력과 안 맞아서 실패한 사례(docs 참고) 때문에
        # 도입. 자세한 설계 근거는 replay_logger.py 모듈 docstring 참고.
        self.replay_logger = ReplayLogger(run_id, log_dir)

        self._log_jsonl({
            "type": "run_start", "ts": time.time(), "run_id": run_id,
            "ckpt_path": str(ckpt_path), "map_range_m": map_range_m,
            "goal_lat": goal_lat, "goal_lon": goal_lon, "dry_run_lock": self.dry_run_lock,
            "initial_heading_lookahead_m": self.initial_heading_lookahead_m,
            "allow_reroute": self.allow_reroute,
            "heading_mode": self.heading_mode,
            "goal_reach_threshold_m": self.goal_reach_threshold_m,
            "near_goal_override_dist_m": self.near_goal_override_dist_m,
            "heading_anchor_pull_threshold_deg": self.heading_anchor_pull_threshold_deg,
            "heading_anchor_pull_persist_ticks": self.heading_anchor_pull_persist_ticks,
            "enable_off_route_safety": self.enable_off_route_safety,
            "off_route_safety_threshold_m": self.off_route_safety_threshold_m,
            "off_route_safety_action": self.off_route_safety_action,
            "enable_straight_segment_override": self.enable_straight_segment_override,
            "straight_near_lookahead_m": self.straight_near_lookahead_m,
            "straight_far_lookahead_m": self.straight_far_lookahead_m,
            "straight_angle_threshold_deg": self.straight_angle_threshold_deg,
            "enable_gps_freeze_guard": self.enable_gps_freeze_guard,
            "gps_freeze_min_disp_m": self.gps_freeze_min_disp_m,
            "gps_freeze_persist_ticks": self.gps_freeze_persist_ticks,
            "gps_freeze_speed_threshold_mps": self.gps_freeze_speed_threshold_mps,
            # 2026-09-25: 각 step 레코드의 context_frame_paths/map_replay_path를
            # 어떻게 실제 파일로 바꾸는지 — 별도 코드를 몰라도 이 JSONL 파일 하나만
            # 보고 알 수 있도록 규칙 자체를 데이터로 남겨둔다.
            "replay_path_convention": "step 레코드의 context_frame_paths/map_replay_path는 "
                                       "이 .jsonl 파일이 들어있는 디렉토리 기준 상대경로",
        })
        if self.dry_run_lock:
            print("[deploy] ⚠ --dry_run 잠금 모드 — 이 프로세스에서는 GO LIVE를 눌러도 "
                  "거부되고 실제 로봇에는 항상 (0,0)만 전송됩니다 (순수 검증 세션용)")
        else:
            print("[deploy] DRY_RUN 상태로 시작합니다. 대시보드에서 "
                  "'정렬 확인' → 'ARM' → 'GO LIVE' 순서로 명시적으로 진행해야 "
                  "실제 로봇에 non-zero 명령이 전송됩니다.")
        self.state.update(dry_run=True, control_stage=self.control_stage, heading_mode=self.heading_mode)

    def _log_jsonl(self, record: dict):
        self._log_fp.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _request_with_retry(self, request_fn, retries=NETWORK_RETRY_COUNT,
                             backoff_s=NETWORK_RETRY_BACKOFF_S):
        # 위 NETWORK_RETRY_COUNT 설명 참고 — request_fn은 "요청 전송 + 바로 필요한
        # 파싱까지"(예: .json() 호출) 전부 포함해서 넘겨야 함, 그래야 빈 응답으로
        # 인한 JSONDecodeError도 재시도 대상이 됨. 재시도를 다 썼는데도 실패하면
        # 마지막 예외를 그대로 올림 — 호출부가 그 예외로 "정지 시도 후 종료"하는
        # 기존 안전 경로를 그대로 타게 함.
        last_exc = None
        for attempt in range(retries + 1):
            try:
                return request_fn()
            except Exception as e:
                last_exc = e
                if attempt < retries:
                    self.state.log(f"⚠ 네트워크 요청 일시 실패(재시도 {attempt + 1}/{retries}): {e!r}")
                    time.sleep(backoff_s)
        raise last_exc

    def poll_frodobot(self):
        cam = self._request_with_retry(
            lambda: requests.get(f"{FRODOBOT_BASE}/v2/front", timeout=5.0).json())
        gps = self._request_with_retry(
            lambda: requests.get(f"{FRODOBOT_BASE}/data", timeout=5.0).json())
        img = decode_frame(cam["front_frame"])
        lat, lon = gps["latitude"], gps["longitude"]
        orientation_deg_raw = float(gps["orientation"])
        # LogoNav_frodobot.py와 동일한 부호 규약: orientation(시계방향, deg) → CCW radian.
        # 90도 보정이 빠져 있다는 가설이 있었으나, step()의 IMU-vs-GPS heading 비교
        # 로그로 먼저 진단하기로 함 — 여기서 성급하게 상수를 바꾸지 않음.
        heading_rad = -float(orientation_deg_raw) / 180.0 * math.pi
        heading_deg = math.degrees(heading_rad)
        self._last_lat, self._last_lon = lat, lon  # 대시보드 명령(정렬 확인 등)이 최신 GPS를 읽을 수 있게
        self.state.update(camera_img=img, lat=lat, lon=lon,
                           heading_deg=heading_deg, orientation_deg_raw=orientation_deg_raw,
                           fix_quality=gps.get("fix_quality"), gps_data_ts=gps.get("timestamp"))
        return img, lat, lon, heading_rad, gps

    def send_control(self, linear, angular):
        # 2026-10-04: 짧은 재시도(_request_with_retry)로 1~2초 내 회복되는 일시적
        # hiccup은 흡수함 — 그래도 실패하면 예외를 그대로 올려서 run()의 루프가
        # 멈추고 정지 명령이 강제되도록 함(에러를 삼키고 계속 움직이는 건 안전상
        # 절대 금지 — step()과 동일한 원칙). linear/angular는 절대 속도 명령이라
        # 재시도로 같은 값이 중복 전송돼도 안전함(상대/누적 명령이 아님).
        r = self._request_with_retry(lambda: requests.post(
            f"{FRODOBOT_BASE}/control",
            json={"command": {"linear": linear, "angular": angular}}, timeout=5.0))
        return r.status_code

    def maybe_update_frame_buffer(self, img):
        now = time.time()
        is_new = now - self.last_frame_time >= CTX_STRIDE_SEC or len(self.frame_buffer) == 0
        if is_new:
            self.frame_buffer.append(self.obs_transform(img))
            self.last_frame_time = now
        # 2026-09-25: deterministic replay logging — 새 프레임일 때만 디스크에 저장(dedup),
        # frame_buffer_ids도 frame_buffer와 정확히 같은 시점에만 append해서 두 deque가
        # 항상 1:1로 대응하게 유지 (frame_buffer_ids[i]가 frame_buffer[i]의 원본 파일 ID).
        frame_id = self.replay_logger.record_context_frame(img, is_new)
        if is_new:
            self.frame_buffer_ids.append(frame_id)
        return is_new

    def _ensure_route_initialized(self, lat, lon, dist_to_goal_m=None):
        """2026-09-25: build_inputs()에서 분리 — 예전엔 frame_buffer가 다 찬 뒤에야
        (즉 predict_waypoints()가 처음 호출될 때) route가 생겼는데, 그러면 route
        생성이 model 파이프라인 준비 상태에 우연히 종속되어서 "GPS position →
        OSRM route 생성"을 그 자체로 독립된 초기 단계로 다룰 수 없었다(route-aligned
        heading 확인은 route가 있어야 가능한데, frame_buffer가 찰 때까지(~1.5초)
        기다릴 이유가 없음). 지금은 GPS fix가 유효해지는 즉시(step()에서 frame_buffer
        조작보다 먼저) 호출됨.

        2026-09-26 추가(1차): 목표 근처(REROUTE_DISABLE_NEAR_GOAL_M 이내)에서는
        재라우팅을 하지 않는다 — 실제 주행에서 목표 ~5.5m 앞 reroute가 OSRM
        보행로망을 따라 헤어핀(왔다갔다) 형태의 새 경로를 만들어서, 이미 가까운데도
        계획 경로상으로는 더 멀어지는 구간이 생겨 "도착 못 하고 계속 도는" 것처럼
        보이는 문제가 실측됨(deploy_20260926_132829.jsonl reroute 이벤트,
        route_latlon 궤적으로 직접 확인).

        2026-09-26 추가(2차, self.allow_reroute): 목표에서 16m 넘게 떨어진, 로봇이
        거의 정지해 있던 구간에서도 is_off_route()가 REROUTE_COOLDOWN_S(3초)마다
        계속 True를 반환해서 reroute가 9번 연속 발생하고, 그때마다 OSRM이 서로
        다른 경로를 반환해 route_bearing_deg가 로봇은 안 움직였는데 -14.6°→-105.3°→
        -92.8°로 요동치는 문제가 실측됨(deploy_20260926_163849.jsonl). 즉 목표 근처만의
        문제가 아니라 이 배포 환경에서 재라우팅 자체가 전반적으로 불안정함(2026-09-18에
        기록된 "무한 재라우팅 폭주"와 같은 계열). 그래서 기본값을 "재라우팅 완전
        비활성화"(최초 route_init만 사용, 런 내내 고정)로 바꾸고, 정말 필요한
        경우에만 --allow_reroute로 기존 동작(이탈 감지 시 재계산)을 켤 수 있게 함.
        REROUTE_DISABLE_NEAR_GOAL_M 로직은 --allow_reroute를 켠 경우에 한해 그대로
        적용됨(목표 근처 보호는 여전히 유효). 최초 route_init은 이 옵션과 무관하게
        항상 수행됨(그게 없으면 애초에 정렬 확인/주행 자체가 불가)."""
        if not self._route_initialized:
            self.map_builder.set_goal(lat, lon, self.goal_lat, self.goal_lon)
            self._route_initialized = True
            self._is_off_route_now = False
            self._log_jsonl({"type": "event", "ts": time.time(), "event": "route_init",
                              "lat": lat, "lon": lon,
                              "route_latlon": self.map_builder.get_route_latlon(),
                              "osrm_fallback": self.map_builder.last_osrm_fallback,
                              "start_snap_m": self.map_builder.last_start_snap_m,
                              "goal_snap_m": self.map_builder.last_goal_snap_m})
        else:
            # 2026-10-04 추가: 감지(is_off_route 호출)를 allow_reroute와 완전히 분리함.
            # 기존엔 아래 elif 조건 자체(allow_reroute and is_off_route(...))에 묶여있어서
            # allow_reroute=False(기본값)일 때 "감지" 자체가 전혀 안 됐음 — 그래서
            # heading_mode=route_bearing 테스트(deploy_20261004_111807.jsonl, 실측
            # 4.85m 이동·목표쪽 진행 0.38m=약 85도 옆으로 샘)를 아무도 못 알아챘음.
            # self._is_off_route_now는 self.enable_off_route_safety가 켜져 있으면 step()
            # 뒷부분에서 가벼운 조치(정지/직접 조향)에 쓰임 — 여기선 감지만 한다.
            self._is_off_route_now = self.map_builder.is_off_route(
                lat, lon, threshold_m=self.off_route_safety_threshold_m)
            if (self.allow_reroute and self._is_off_route_now
                    and time.time() - self._last_reroute_ts >= REROUTE_COOLDOWN_S):
                if dist_to_goal_m is not None and dist_to_goal_m <= REROUTE_DISABLE_NEAR_GOAL_M:
                    self.state.log(f"경로 이탈 감지했지만 목표 근처(dist={dist_to_goal_m:.1f}m ≤ "
                                    f"{REROUTE_DISABLE_NEAR_GOAL_M:.0f}m)라 재라우팅 생략")
                else:
                    self.state.log("경로 이탈 감지 → 재라우팅")
                    self.map_builder.set_goal(lat, lon, self.goal_lat, self.goal_lon)
                    self._last_reroute_ts = time.time()
                    self._log_jsonl({"type": "event", "ts": time.time(), "event": "reroute",
                                      "lat": lat, "lon": lon,
                                      "route_latlon": self.map_builder.get_route_latlon(),
                                      "osrm_fallback": self.map_builder.last_osrm_fallback,
                                      "start_snap_m": self.map_builder.last_start_snap_m,
                                      "goal_snap_m": self.map_builder.last_goal_snap_m})
        self.state.update(osrm_fallback=self.map_builder.last_osrm_fallback,
                           start_snap_m=self.map_builder.last_start_snap_m,
                           goal_snap_m=self.map_builder.last_goal_snap_m)

    def _confirm_route_alignment(self):
        """대시보드 '정렬 확인' 버튼 → 여기로 옴 (_apply_pending_commands()가 호출).
        route의 "현재 위치 바로 앞" tangent(INITIAL_HEADING_LOOKAHEAD_M)를 계산해서
        _route_aligned_heading_rad로 고정.

        2026-09-26 실외 테스트 중 발견한 문제로 추가: 원래는 "_heading_ema_vec는
        절대 안 건드림"이었는데, DRY_RUN으로 오래 관찰하는 동안 로봇이 실제로는
        정지해 있어도 GPS 수신 잡음(drift)만으로 estimate_heading_from_track()의
        최소 이동거리(1.5m) 임계값을 우연히 넘겨서 _heading_ema_vec가 이미
        "실측(gps_track)"으로 오염된 채 확정돼 있는 경우가 실측 확인됨 — 이러면
        heading 분기 우선순위(gps_track/gps_ema_hold가 route_aligned보다 항상 우선)
        때문에 몇 번을 다시 정렬 확인해도 실제로는 절대 반영되지 않는 문제였음.
        그래서 정렬 확인 시점에 _heading_ema_vec와 past_track을 함께 리셋한다 —
        past_track도 같이 비워야 하는 이유: 안 비우면 다음 tick에 남아있는 그
        "잡음 이력"으로 estimate_heading_from_track()이 즉시 다시 값을 반환해서
        EMA가 바로 재오염됨(리셋이 사실상 무의미해짐). 확정 시점 이후의 "진짜 새"
        이동만 다시 GPS heading을 확립할 수 있게 된다."""
        if not self._route_initialized:
            self.state.log_error("정렬 확인 거부됨 — route가 아직 초기화되지 않음 "
                                  "(유효한 GPS fix를 먼저 확보해야 함)")
            return
        if self._last_lat is None or self._last_lat == 1000 or self._last_lon == 1000:
            self.state.log_error("정렬 확인 거부됨 — 유효한 GPS 위치가 없음")
            return
        bearing = self.map_builder.route_bearing_rad(
            self._last_lat, self._last_lon, lookahead_m=self.initial_heading_lookahead_m)
        if bearing is None:
            self.state.log_error("정렬 확인 거부됨 — route_bearing_rad()가 None 반환 "
                                  "(현재 위치가 route 끝 근처일 수 있음)")
            return
        had_stale_ema = self._heading_ema_vec is not None
        self._heading_ema_vec = None
        self._heading_divergence_ticks = 0
        self._heading_anchor_pull_ticks = 0
        self._gps_heading_freeze_ticks = 0
        self._gps_heading_freeze_last_rad = None
        self.past_track.clear()
        self._route_aligned_heading_rad = bearing
        self._route_aligned_confirmed_ts = time.time()
        deg = math.degrees(bearing)
        self.state.log(f"Route-aligned initial heading 확정: {deg:+.1f}° "
                        f"(lookahead={self.initial_heading_lookahead_m}m)"
                        + (" — 기존 GPS heading EMA/이력 초기화함(오염된 값이었을 수 있음)"
                           if had_stale_ema else ""))
        self._log_jsonl({"type": "event", "ts": time.time(), "event": "route_alignment_confirmed",
                          "route_aligned_heading_deg": deg,
                          "lookahead_m": self.initial_heading_lookahead_m,
                          "lat": self._last_lat, "lon": self._last_lon,
                          "reset_stale_ema": had_stale_ema})
        self.state.update(route_aligned_heading_deg=deg)

    def _handle_command(self, cmd):
        """대시보드 버튼 → 큐에 쌓인 명령을 제어 루프 스레드에서 순차 처리
        (모든 상태 변경이 이 스레드 하나에서만 일어나므로 락이 필요 없음)."""
        if cmd == "confirm_alignment":
            self._confirm_route_alignment()
        elif cmd == "arm":
            if self.control_stage == "DRY_RUN":
                if self._route_aligned_heading_rad is None:
                    self.state.log_error("ARM 거부됨 — route-aligned heading이 아직 "
                                          "확정되지 않음 (먼저 '정렬 확인'을 누를 것)")
                else:
                    self.control_stage = "ARMED"
                    self._armed_ts = time.time()
                    self.state.log(f"ARMED — {ARM_TIMEOUT_S:.0f}초 안에 GO LIVE 필요 "
                                    "(아직 실제 명령 전송 안 함)")
                    self._log_jsonl({"type": "event", "ts": time.time(), "event": "armed"})
            else:
                self.state.log_error(f"ARM 무시됨 (현재 상태={self.control_stage})")
        elif cmd == "go_live":
            if self.dry_run_lock:
                self.state.log_error("GO LIVE 거부됨 — --dry_run 잠금 모드로 실행 중 "
                                      "(이 프로세스에서는 LIVE 전환 불가)")
            elif self.control_stage == "ARMED":
                self.control_stage = "LIVE"
                self.state.log("⚠ LIVE 전환됨 — 실제 로봇에 명령 전송을 시작합니다")
                self._log_jsonl({
                    "type": "event", "ts": time.time(), "event": "go_live",
                    "route_aligned_heading_deg": (math.degrees(self._route_aligned_heading_rad)
                                                   if self._route_aligned_heading_rad is not None else None),
                    "map_heading_source_at_go": self._last_map_heading_source,
                })
                # 대시보드 녹화는 __init__에서 프로세스 시작 시점부터 이미 돌고 있음
                # (dashboard_capture.py::DashboardRecorder) — GO LIVE에서 따로 할 일 없음.
            else:
                self.state.log_error(f"GO LIVE 거부됨 (ARMED 상태가 아님: {self.control_stage})")
        elif cmd == "abort":
            if self.control_stage != "DRY_RUN":
                self.state.log(f"ABORT — {self.control_stage} → DRY_RUN")
                self._log_jsonl({"type": "event", "ts": time.time(), "event": "abort",
                                  "from_stage": self.control_stage})
            self.control_stage = "DRY_RUN"
            self._armed_ts = None
        self.state.update(control_stage=self.control_stage, dry_run=(self.control_stage != "LIVE"))

    def _apply_pending_commands(self):
        while True:
            try:
                cmd = self._pending_commands.get_nowait()
            except queue.Empty:
                break
            self._handle_command(cmd)
        # ARMED 상태로 너무 오래 방치되면 자동으로 DRY_RUN으로 되돌림 (실수 방지)
        if self.control_stage == "ARMED" and self._armed_ts is not None \
                and time.time() - self._armed_ts > ARM_TIMEOUT_S:
            self.state.log(f"ARM 시간 초과({ARM_TIMEOUT_S:.0f}초) — 자동으로 DRY_RUN으로 복귀. "
                            "다시 ARM해야 GO LIVE 가능")
            self._log_jsonl({"type": "event", "ts": time.time(), "event": "arm_timeout"})
            self.control_stage = "DRY_RUN"
            self._armed_ts = None
            self.state.update(control_stage=self.control_stage, dry_run=True)

    def build_inputs(self, lat, lon, heading_rad, tick_id=None):
        # 대시보드/로그용 디버그 필드 (2026-09-25 추가) — 0918 replay에서 heading과
        # route bearing 차이가 ~41°였던 것과 비교할 수 있게 매 틱 계산해둠.
        self._last_route_bearing_rad = self.map_builder.route_bearing_rad(lat, lon)

        # obs_stack: 컨텍스트가 아직 안 찼으면 가장 오래된 프레임으로 패딩
        frames = list(self.frame_buffer)
        while len(frames) < N_CTX + 1:
            frames = [frames[0]] + frames if frames else frames
        obs_stack = torch.cat(frames[-(N_CTX + 1):], dim=0).unsqueeze(0).to(self.device)  # (1,18,96,96)
        obs_cur = obs_stack[:, -3:]

        map_np = self.map_builder.get_map_image(
            lat, lon, heading_rad,
            past_track=list(self.past_track) if self.past_track else None,
        )
        self._last_map_img = map_np  # predict_waypoints()에서 예측 궤적을 겹쳐 그리는 데 사용
        self.state.update(map_img=map_np)
        map_tensor = self.map_builder.transform(Image.fromarray(map_np)).unsqueeze(0).to(self.device)  # (1,3,96,96)

        # 2026-09-25: deterministic replay logging — transform 적용 "전"의 map_np(모델이
        # 실제로 본 것과 동일한 224x224 RGB)를 tick_id로 저장. ego 위치/heading이 매 틱
        # 달라서 dedup 없이 항상 저장(ReplayLogger.save_map 참고).
        if tick_id is not None:
            self._last_map_replay_path = self.replay_logger.save_map(map_np, tick_id)

        goal_pose = torch.zeros(1, 4, device=self.device)
        goal_mask = torch.zeros(1, dtype=torch.long, device=self.device)
        feat_text = torch.zeros(1, 512, device=self.device)
        cur_img = F.interpolate(obs_cur, (224, 224), mode="bilinear", align_corners=False)

        return obs_stack, goal_pose, map_tensor, obs_cur, goal_mask, feat_text, cur_img

    def predict_waypoints(self, lat, lon, heading_rad, tick_id=None):
        inputs = self.build_inputs(lat, lon, heading_rad, tick_id=tick_id)
        with torch.no_grad():
            pred, _, _ = self.model(*inputs)
        pred_xy_m = pred[0, :, :2].detach().cpu().numpy() * METRIC_WAYPOINT_SPACING  # (8,2) ego x=fwd,y=left
        self.state.update(pred_xy_m=pred_xy_m)
        # 예측 궤적(청록색)을 방금 만든 ego-centric 지도(빨강=계획 경로, 회색=지나온 길) 위에
        # 같은 축척으로 겹쳐서 대시보드에 표시 — 모델이 실제 경로 대비 어디를 보고 있는지 한눈에 확인용.
        overlay_img = self.map_builder.draw_predicted_trajectory(self._last_map_img, pred_xy_m)
        self.state.update(map_img=overlay_img)
        return pred_xy_m

    def waypoint_to_control(self, pred_xy_m, target_step=2):
        """LogoNav_frodobot.py와 동일한 기하학적 변환(목표 waypoint까지의 각도/거리로
        linear/angular 속도 산출)이되, 시간 분모는 반드시 그 waypoint의 실제 시점과
        일치시켜야 함. LogoNav는 DT=1/4를 쓰는데 이건 "NoMaD 모델 자체의 웨이포인트
        간격이 0.25초"이기 때문에 맞는 값이었음 — 우리 모델의 웨이포인트 간격은
        0.7초(WAYPOINT_STRIDE_SEC)이고, index i(0-based)는 (i+1)*0.7초 뒤를 의미함
        (rides11_dataset.py의 k=range(1, N_WAYPOINTS+1) 인덱싱과 동일).
        이걸 제어 루프 주기 DT(=1/3)로 나누면 시점이 안 맞아서 속도가 실제보다
        부풀려지고, 그 결과 항상 MAX_V/MAX_W 안전 캡에 걸려 모델 예측의 크기 정보가
        사라지는 문제가 있었음 (2026-08 실배포 테스트에서 "명령이 항상 작다"로 발견됨)."""
        target_time_s = (target_step + 1) * WAYPOINT_STRIDE_SEC  # 예: target_step=2 → 2.1초
        x, y = pred_xy_m[target_step]  # x=forward(m), y=left(m)
        EPS = 1e-8
        if abs(x) < EPS and abs(y) < EPS:
            return 0.0, 0.0
        if abs(x) < EPS:
            return 0.0, np.sign(y) * math.pi / (2 * target_time_s)
        linear = x / target_time_s
        angular = math.atan2(y, x) / target_time_s  # y=left이므로 양수 angular=왼쪽 회전(CCW)이 되도록
        return float(np.clip(linear, 0, MAX_V * 2)), float(np.clip(angular, -MAX_W * 2, MAX_W * 2))

    def step(self):
        tick_id = self.replay_logger.next_tick_id()
        img, lat, lon, heading_rad, raw_data = self.poll_frodobot()
        self._apply_pending_commands()  # 대시보드 ARM/GO LIVE/정렬확인/ABORT 처리 (2026-09-25)
        imu_deg = math.degrees(heading_rad)
        # frodobot_raw: FrodoBot Mini가 /data로 내보내는 원본 텔레메트리 그대로 보존
        # (battery, signal_level, speed, gps_signal, vibration, accels/gyros/mags/rpms 등).
        record = {"type": "step", "ts": time.time(), "tick_id": tick_id,
                  "lat": lat, "lon": lon, "imu_heading_deg": imu_deg,
                  "frodobot_raw": raw_data, "control_stage": self.control_stage}
        self.state.update(tick_id=tick_id, control_stage=self.control_stage)

        # GPS fix 없음(sentinel 1000) — 지도 자체를 만들 수 없으므로 정지 유지
        if lat == 1000 or lon == 1000:
            self.state.log_error("GPS fix 없음 (lat/lon=1000) — 정지 유지")
            record.update(gps_fix_ok=False, linear=0.0, angular=0.0, note="gps_fix_missing")
            self._log_jsonl(record)
            return 0.0, 0.0
        record["gps_fix_ok"] = True

        # 2026-09-26 추가: goal 도착 감지 — route/heading/모델 계산과 무관하게
        # GPS만으로 바로 확인 가능해서 여기서 가장 먼저 체크한다. 한 번 도착하면
        # 절대 다시 안 풀림(latched) — 도착 후 GPS 잡음으로 threshold를 들락날락
        # 해도 다시 움직이기 시작하지 않도록.
        dist_to_goal_m = latlon_distance_m(lat, lon, self.goal_lat, self.goal_lon)
        record["dist_to_goal_m"] = dist_to_goal_m
        self.state.update(dist_to_goal_m=dist_to_goal_m)
        if not self._goal_reached and dist_to_goal_m <= self.goal_reach_threshold_m:
            self._goal_reached = True
            self.state.log(f"🏁 목표 도착 감지 (거리={dist_to_goal_m:.2f}m ≤ "
                            f"임계값={self.goal_reach_threshold_m:.2f}m) — 이후 영구 정지")
            self._log_jsonl({"type": "event", "ts": time.time(), "tick_id": tick_id,
                              "event": "goal_reached", "dist_to_goal_m": dist_to_goal_m,
                              "lat": lat, "lon": lon,
                              "goal_lat": self.goal_lat, "goal_lon": self.goal_lon})
            self.state.update(goal_reached=True)

        # 2026-09-25: route(OSRM) 생성을 frame_buffer 준비 상태와 분리 — GPS fix가
        # 유효해지는 즉시 route가 생겨야 대시보드에서 route-aligned 정렬 확인이
        # frame_buffer(~1.5초)를 기다리지 않고 바로 가능함.
        self._ensure_route_initialized(lat, lon, dist_to_goal_m=dist_to_goal_m)

        self.maybe_update_frame_buffer(img)
        # 2026-09-25: context_frame_ids(정수)만으로는 파일 경로 규칙(replay_logger.py의
        # ctx_{id:06d}.jpg 네이밍)을 코드로 따로 알아야만 역추적이 가능했음(offline
        # smoke test로 실측 확인). context_frame_paths를 같이 저장해서 이 JSONL
        # 레코드 하나만으로(코드 몰라도) 실제 파일을 찾을 수 있게 함. map_replay_path와
        # 동일한 규칙(이 .jsonl 파일이 있는 디렉토리 기준 상대경로, 즉 "frames/<run_id>/...")
        # — repo를 통째로 옮기거나 log 폴더만 따로 복사해도 그대로 유효함.
        context_frame_ids = list(self.frame_buffer_ids)
        record["context_frame_ids"] = context_frame_ids
        record["context_frame_paths"] = [self.replay_logger.context_frame_path(fid)
                                          for fid in context_frame_ids]
        self.past_track.append((lat, lon))

        # 2026-10-04 추가 (실측 사고, deploy_20261004_123749.jsonl, 2026-10-04 Harness 1
        # 재분석으로 143423.jsonl에서 세분화): GPS 동결 감지 — 위치가 gps_freeze_min_disp_m
        # 이상 못 움직인 채 지속되고, 그 구간 안에서 speed가 gps_freeze_speed_threshold_mps를
        # 한 번이라도 넘은 적 있으면(=로봇이 실제로 움직인 증거가 있는데 위치만 안 바뀜)
        # gps_freeze_persist_ticks 이상 지속 시 "동결"로 판정한다.
        # ⚠ 143423.jsonl에서 발견된 버그: 원래는 "매 틱 speed>=threshold"를 AND 조건으로
        # 요구해서, 같은 동결 구간 안에서도 speed가 순간적으로 threshold 밑으로 떨어지면
        # 카운터가 0으로 리셋되고 플래그가 False→True를 반복(10틱마다 재무장) — 하나의
        # 긴 동결(tick82~145, 20~40초+)인데도 "멈췄다 주행 반복"이 생기고, 그 False인
        # 틈마다 override/모델예측이 "정상"이라 믿고 계속 명령을 냄. 위치가 실제로
        # 바뀌기 전까지는 리셋 안 하고(freeze_ticks는 위치 변화만으로 리셋), speed 조건은
        # "이 동결 구간 안에서 한 번이라도 움직인 적 있었는지"(OR, 위치 변화 시 재시작)로
        # 바꿔서 한 번 동결로 확정되면 실제로 위치가 바뀔 때까지 계속 True로 유지되게 함.
        # past_track과 별개로 독립 추적 — 정렬확인/route_corrected가 past_track을
        # clear해도 이 감지는 끊기지 않아야 함.
        speed_mps = float((raw_data or {}).get("speed") or 0.0)
        position_unchanged = (self._gps_freeze_last_latlon is not None
                               and latlon_distance_m(lat, lon, *self._gps_freeze_last_latlon)
                               < self.gps_freeze_min_disp_m)
        if position_unchanged:
            self._gps_freeze_ticks += 1
            if speed_mps >= self.gps_freeze_speed_threshold_mps:
                self._gps_freeze_any_speed_seen = True
        else:
            self._gps_freeze_ticks = 0
            self._gps_freeze_any_speed_seen = speed_mps >= self.gps_freeze_speed_threshold_mps
        self._gps_freeze_last_latlon = (lat, lon)
        was_frozen = self._gps_frozen_now
        self._gps_frozen_now = (self.enable_gps_freeze_guard
                                 and self._gps_freeze_ticks >= self.gps_freeze_persist_ticks
                                 and self._gps_freeze_any_speed_seen)
        record["gps_frozen"] = self._gps_frozen_now
        if self._gps_frozen_now and not was_frozen:
            self.state.log(f"⚠ GPS 위치 동결 감지(speed={speed_mps:.2f}m/s인데 위치가 "
                            f"{self.gps_freeze_persist_ticks}틱 이상 {self.gps_freeze_min_disp_m}m "
                            f"이상 안 움직임) — 정지 + heading anchor pull 보류")

        # heading 소스: 2026-08-25 실배포 로그 분석(docs/0825.md 2-2)에서 IMU 컴퍼스와
        # GPS궤적 기반 heading이 평균 +97° 어긋남을 확인 — 학습 데이터의 heading은
        # GPS궤적 기반(atan2, osm_map_generator_rides11.py::estimate_headings()와 동일
        # 공식)으로 만들어지므로, render_frame()에 넣는 heading도 같은 소스로 맞춰서
        # 학습/배포 간 heading 정의 자체가 갈라지는 것을 원천 차단한다.
        #   imu_heading  = -orientation/180*pi (로봇 컴퍼스 센서, 진단/폴백 전용)
        #   gps_heading  = 방금 지나온 GPS 두 점 사이 방향 (학습 heading과 동일 산출 방식)
        # GPS 이동량이 부족(정지/막 시작)해서 gps_heading을 못 구할 때만 IMU로 폴백.
        gps_heading_rad = estimate_heading_from_track(list(self.past_track))
        record["gps_heading_deg"] = math.degrees(gps_heading_rad) if gps_heading_rad is not None else None
        gps_accumulated_path_m, gps_net_displacement_m = gps_heading_readiness(list(self.past_track))
        record["gps_accumulated_path_m"] = gps_accumulated_path_m
        record["gps_net_displacement_m"] = gps_net_displacement_m
        self.state.update(gps_accumulated_path_m=gps_accumulated_path_m,
                           gps_net_displacement_m=gps_net_displacement_m)

        # 2026-10-03 추가: 아래 route-bearing 재동기화에 쓰려고 미리 계산 — build_inputs()도
        # 똑같은 걸(lookahead 기본값 5.0m) 다시 계산해서 self._last_route_bearing_rad에
        # 저장하지만(디버그 필드용), 그건 heading 선택 이후에 호출되는 거라 여기서 따로
        # 한 번 더 구해야 이번 tick의 heading 보정에 쓸 수 있음. route_bearing_rad() 자체는
        # heading과 무관하게 (lat, lon)+캐싱된 route만 쓰는 순수 조회라 중복 호출 비용 작음.
        route_bearing_rad_now = self.map_builder.route_bearing_rad(lat, lon)

        if self.heading_mode == "route_bearing_anchored" and route_bearing_rad_now is not None:
            # 2026-10-04 추가 (Harness 1 제안) — route_bearing을 기본값으로 매 tick
            # 그대로 쓰되(route_bearing 자체는 route 지오메트리 기반이라 안정적임이
            # 실측으로 확인됨, §1-9/1-10: 54틱 연속 완전히 고정됐었음), gps_track이
            # DEFAULT_HEADING_ANCHOR_PULL_THRESHOLD_DEG 이상 DEFAULT_HEADING_ANCHOR_PULL_PERSIST_TICKS
            # 연속으로 반박할 때만 그 tick 한정으로 gps_track을 대신 신뢰함. 위
            # route_corrected(기본=gps_track, route_bearing은 예외적 보정)와 정반대
            # 우선순위 — Harness 1의 오프라인 재생 검증(min_disp_m만 올리는 건 최악
            # 스파이크는 줄지만 중간 드리프트 노출시간이 오히려 늘어남)에 근거함.
            map_heading_rad = route_bearing_rad_now
            source_label = "route_bearing_anchor"
            if gps_heading_rad is not None:
                # 2026-10-04 추가 (실측 사고 deploy_20261004_133317.jsonl, 위
                # DEFAULT_GPS_HEADING_FREEZE_PERSIST_TICKS 설명 참고): gps_heading_rad
                # 값 자체가 거의 안 바뀐 채 여러 틱 유지되는지 추적 — 위치 동결과는
                # 독립적으로, "추정값이 고장나서 고정된 것"을 잡기 위함.
                if (self._gps_heading_freeze_last_rad is not None
                        and abs((gps_heading_rad - self._gps_heading_freeze_last_rad + math.pi)
                                % (2 * math.pi) - math.pi) < GPS_HEADING_FREEZE_EPS_RAD):
                    self._gps_heading_freeze_ticks += 1
                else:
                    self._gps_heading_freeze_ticks = 0
                self._gps_heading_freeze_last_rad = gps_heading_rad
                gps_heading_value_frozen = (self._gps_heading_freeze_ticks
                                             >= DEFAULT_GPS_HEADING_FREEZE_PERSIST_TICKS)

                pull_diff_deg = (math.degrees(gps_heading_rad - route_bearing_rad_now) + 180) % 360 - 180
                if abs(pull_diff_deg) >= self.heading_anchor_pull_threshold_deg:
                    self._heading_anchor_pull_ticks += 1
                else:
                    self._heading_anchor_pull_ticks = 0
                # 2026-10-04 추가 (Harness 1 cross-session 분석, 실측 사고
                # deploy_20261004_123749.jsonl): GPS가 동결되면 gps_track도 route_bearing과
                # 똑같이 멈춘 값에서 계산되므로 "지속 반박" 조건을 영원히 만족시켜버려서
                # (진짜 반박과 구분 불가) 끝까지 안 풀리는 버그가 있었음 — 동결 중에는
                # pull 자체를 보류하고 route_bearing_anchor를 유지(아래 override 체인에서
                # gps_freeze_safety가 최우선으로 정지시킴, 여기선 지도 내용만 안전하게 유지).
                # gps_heading_value_frozen도 같은 이유로 보류(133317 사고: 위치는 계속
                # 바뀌어서 _gps_frozen_now는 안 걸렸는데 추정값만 -90.0°에 60초+ 고정).
                # 2026-10-04 추가 (실측 사고 deploy_20261004_143423.jsonl, Harness 1 교차분석):
                # GO LIVE 직후처럼 past_track 누적이 적으면 estimate_heading_from_track()의
                # "짧은 창"(fast_disp_m) 경로가 양자화 잡음 섞인 값을 매 tick 다른 격자각도로
                # 내놓을 수 있음(같은 값에 고정되는 게 아니라 계속 "다른 값으로 틀림" —
                # gps_heading_value_frozen 체크로는 못 잡는 변종). gps_accumulated_path_m이
                # DEFAULT_HEADING_ANCHOR_PULL_MIN_ACCUM_PATH_M(긴 창 평균화가 실제로 적용되는
                # 기준, gps_heading_readiness()의 기본 min_disp_m과 동일) 이상 쌓이기 전에는
                # pull 자체를 보류 — 짧은 창 기반의 덜 평균화된 추정을 신뢰하지 않음.
                gps_track_ready = gps_accumulated_path_m >= DEFAULT_HEADING_ANCHOR_PULL_MIN_ACCUM_PATH_M
                if (self._heading_anchor_pull_ticks >= self.heading_anchor_pull_persist_ticks
                        and not self._gps_frozen_now
                        and not gps_heading_value_frozen
                        and gps_track_ready):
                    map_heading_rad = gps_heading_rad
                    source_label = "gps_track_pull"
            else:
                self._heading_anchor_pull_ticks = 0
                self._gps_heading_freeze_ticks = 0
                self._gps_heading_freeze_last_rad = None
            smoothed_deg = math.degrees(map_heading_rad)
            record["heading_diff_deg"] = None
            record["smoothed_heading_deg"] = smoothed_deg
            record["map_heading_source"] = source_label
            print(f"    [heading] IMU컴퍼스={imu_deg:+7.1f}°(미사용)  GPS궤적={record['gps_heading_deg']}  "
                  f"route_bearing_anchored(소스={source_label})={smoothed_deg:+7.1f}°  "
                  f"(pull_ticks={self._heading_anchor_pull_ticks})")
        elif self.heading_mode == "route_bearing" and route_bearing_rad_now is not None:
            # 2026-10-04 추가: --heading_mode route_bearing — gps_track/EMA를 아예 안 쓰고
            # 매 tick route_bearing_rad()를 그대로 heading으로 사용. route_bearing_rad_now를
            # 못 구하는 tick(route 아직 없음/현재 위치가 route 끝 근처)에는 이 조건 자체가
            # 거짓이 되어 아래 다른 분기(gps_track/route_aligned/imu_fallback)로 자연스럽게
            # 넘어가고, 다음 tick에 다시 쓸 수 있게 되면 자동으로 이 분기가 우선됨(매 tick
            # 새로 평가되는 if/elif라 별도 상태 추적 불필요).
            map_heading_rad = route_bearing_rad_now
            smoothed_deg = math.degrees(map_heading_rad)
            record["heading_diff_deg"] = None
            record["smoothed_heading_deg"] = smoothed_deg
            record["map_heading_source"] = "route_bearing"
            print(f"    [heading] IMU컴퍼스={imu_deg:+7.1f}°(미사용)  GPS궤적={record['gps_heading_deg']}"
                  f"(참고용, 미반영)  route_bearing 직접 사용(모드=route_bearing)={smoothed_deg:+7.1f}°")
        elif self.heading_mode == "route_aligned_fixed" and self._route_aligned_heading_rad is not None:
            # 2026-09-26 추가: --heading_mode route_aligned_fixed — 실외 테스트에서
            # 정지/저속 구간의 GPS 잡음이 gps_track/gps_ema_hold를 계속 흔드는 문제가
            # 실측됨(누적 이동거리/순변위가 어긋나는 지그재그 패턴, heading이 100°+
            # 튐). 이 모드에서는 "정렬 확인"으로 고정한 route_aligned 값을 실측 GPS
            # heading이 얼마나 들어오든 절대 넘겨주지 않고 런 내내 그대로 사용한다
            # (gps_track/gps_ema_hold 분기 자체를 건너뜀 — _heading_ema_vec도 아예
            # 안 건드려서 두 모드 상태가 서로 안 섞이게 함). gps_heading_deg는 위에서
            # 이미 진단용으로만 로그에 남음(실제 heading에는 영향 없음).
            map_heading_rad = self._route_aligned_heading_rad
            smoothed_deg = math.degrees(map_heading_rad)
            record["heading_diff_deg"] = None
            record["smoothed_heading_deg"] = smoothed_deg
            record["map_heading_source"] = "route_aligned"
            print(f"    [heading] IMU컴퍼스={imu_deg:+7.1f}°(미사용)  GPS궤적={record['gps_heading_deg']}"
                  f"(참고용, 미반영)  route-aligned 고정(모드=route_aligned_fixed)={smoothed_deg:+7.1f}°")
        elif gps_heading_rad is not None:
            is_first_real_acquisition = self._heading_ema_vec is None  # 2026-09-25: 전환 로그용
            # 2026-10-03 추가: rate limit — 첫 확보(시드)가 아니고 직전 tick 시각을 알 때만
            # 적용(첫 확보는 "스냅"이 맞는 동작이라 제외). 로봇이 dt_s 동안 물리적으로 돌 수
            # 있는 최대 각도(MAX_W*dt_s*마진)보다 raw 추정값이 더 크게 벌어져 있으면, 그
            # 한계까지만 이동한 지점을 입력으로 씀 — 한 번에 안 꺾이고 몇 틱에 걸쳐 수렴.
            if self._heading_ema_vec is not None and self._prev_step_ts is not None:
                dt_s = record["ts"] - self._prev_step_ts
                cur_angle = math.atan2(self._heading_ema_vec.imag, self._heading_ema_vec.real)
                raw_diff = (gps_heading_rad - cur_angle + math.pi) % (2 * math.pi) - math.pi
                max_dtheta = MAX_W * max(dt_s, 0.0) * HEADING_RATE_LIMIT_MARGIN
                if abs(raw_diff) > max_dtheta:
                    gps_heading_rad = cur_angle + math.copysign(max_dtheta, raw_diff)
            new_vec = complex(math.cos(gps_heading_rad), math.sin(gps_heading_rad))
            if self._heading_ema_vec is None:
                self._heading_ema_vec = new_vec  # 첫 확보 시엔 그대로 초기화(route_aligned로 절대 시드 안 함)
            else:
                self._heading_ema_vec = (1 - HEADING_EMA_ALPHA) * self._heading_ema_vec + HEADING_EMA_ALPHA * new_vec
            map_heading_rad = math.atan2(self._heading_ema_vec.imag, self._heading_ema_vec.real)
            smoothed_deg = math.degrees(map_heading_rad)
            diff = (smoothed_deg - imu_deg + 180) % 360 - 180
            record["heading_diff_deg"] = diff
            record["smoothed_heading_deg"] = smoothed_deg
            record["map_heading_source"] = "gps_track"
            print(f"    [heading] IMU컴퍼스={imu_deg:+7.1f}°  GPS궤적={record['gps_heading_deg']:+7.1f}°  "
                  f"EMA사용={smoothed_deg:+7.1f}°  차이(EMA-IMU)={diff:+7.1f}°")
            # 2026-09-25: route_aligned(사람이 정렬한 값) → gps_track(실측)으로
            # 처음 전환되는 그 tick만 명시적으로 로그 — 여기서 별도 스무딩은
            # 넣지 않음(사용자 요청대로), 대신 전환 자체가 보이게 함.
            if is_first_real_acquisition and self._route_aligned_heading_rad is not None:
                prev_deg = math.degrees(self._route_aligned_heading_rad)
                transition_diff = (smoothed_deg - prev_deg + 180) % 360 - 180
                self.state.log(f"heading source 전환: route_aligned({prev_deg:+.1f}°) → "
                                f"gps_track({smoothed_deg:+.1f}°), 차이={transition_diff:+.1f}°")
                self._log_jsonl({"type": "event", "ts": time.time(), "tick_id": tick_id,
                                  "event": "heading_source_transition",
                                  "from": "route_aligned", "to": "gps_track",
                                  "route_aligned_heading_deg": prev_deg,
                                  "first_gps_heading_deg": smoothed_deg,
                                  "diff_deg": transition_diff})
        elif self._heading_ema_vec is not None:
            # GPS 이동량이 잠깐 부족해도(제자리 회전/정지) 순간 IMU로 스냅하지 않고
            # 마지막으로 안정화된 EMA heading을 유지("관성") — 2026-09-18에 틱마다
            # 최대 190°까지 튀던 회전 불안정을 직접 렌더링 비교로 확인해서 도입.
            # USE_GYRO_FUSION=True면 그냥 고정하지 않고, 자이로 상대 회전량만큼
            # 계속 보정한다(estimate_yaw_delta_from_gyro 참고, 실험적).
            if USE_GYRO_FUSION and self._prev_step_ts is not None:
                dt_s = record["ts"] - self._prev_step_ts
                yaw_delta_rad = estimate_yaw_delta_from_gyro(raw_data, dt_s)
                cur_rad = math.atan2(self._heading_ema_vec.imag, self._heading_ema_vec.real)
                new_rad = cur_rad + yaw_delta_rad
                self._heading_ema_vec = complex(math.cos(new_rad), math.sin(new_rad))
                source_label = "gps_ema_gyro"
            else:
                source_label = "gps_ema_hold"
            map_heading_rad = math.atan2(self._heading_ema_vec.imag, self._heading_ema_vec.real)
            smoothed_deg = math.degrees(map_heading_rad)
            record["heading_diff_deg"] = None
            record["smoothed_heading_deg"] = smoothed_deg
            record["map_heading_source"] = source_label
            print(f"    [heading] IMU컴퍼스={imu_deg:+7.1f}°(무시)  GPS궤적=(이동량 부족)  "
                  f"EMA유지({source_label})={smoothed_deg:+7.1f}°")
        elif self._route_aligned_heading_rad is not None:
            # 2026-09-25 신규: 실측 GPS course heading이 아직 한 번도 확보되지
            # 않은 초반 구간에서, IMU 폴백 대신 사용자가 확정한 route-aligned
            # heading을 고정값으로 사용. _heading_ema_vec는 절대 건드리지 않음
            # (다음에 실측이 들어오면 위 gps_track 분기가 오염 없이 하드셋함).
            map_heading_rad = self._route_aligned_heading_rad
            smoothed_deg = math.degrees(map_heading_rad)
            record["heading_diff_deg"] = None
            record["smoothed_heading_deg"] = smoothed_deg
            record["map_heading_source"] = "route_aligned"
            print(f"    [heading] IMU컴퍼스={imu_deg:+7.1f}°(미사용)  "
                  f"route-aligned 고정={smoothed_deg:+7.1f}°  GPS궤적=(이동량 부족)")
        else:
            map_heading_rad = heading_rad  # 콜드스타트 폴백: route-aligned 확정도 안 됐고 GPS-heading도 없는 경우
            record["heading_diff_deg"] = None
            record["smoothed_heading_deg"] = None
            record["map_heading_source"] = "imu_fallback"
            print(f"    [heading] IMU컴퍼스={imu_deg:+7.1f}°(폴백 사용, EMA/route-align 없음)  "
                  f"GPS궤적=(이동량 부족, 추정불가)")

        # 2026-10-03 추가: route_bearing과 지속적으로 크게 어긋나면 강제 재동기화.
        # route_aligned_fixed/부트스트랩(아직 실측 전) 단계는 제외 — 그쪽은 의도적으로
        # 고정/보류된 값이라 건드리면 그 모드의 존재 이유가 없어짐(위 상수 설명 참고).
        # 단발 노이즈가 아니라 "계속" 어긋나는 경우만 잡으려고 PERSIST_TICKS 연속 요구.
        if record["map_heading_source"] in ("gps_track", "gps_ema_hold", "gps_ema_gyro") \
                and route_bearing_rad_now is not None:
            diff_deg = (math.degrees(map_heading_rad - route_bearing_rad_now) + 180) % 360 - 180
            if abs(diff_deg) >= HEADING_ROUTE_CORRECTION_THRESHOLD_DEG:
                self._heading_divergence_ticks += 1
            else:
                self._heading_divergence_ticks = 0
            if self._heading_divergence_ticks >= HEADING_ROUTE_CORRECTION_PERSIST_TICKS:
                prev_source = record["map_heading_source"]
                prev_deg = math.degrees(map_heading_rad)
                map_heading_rad = route_bearing_rad_now
                self._heading_ema_vec = complex(math.cos(map_heading_rad), math.sin(map_heading_rad))
                # 2026-10-04 추가: past_track도 같이 비워야 함 — 안 비우면
                # estimate_heading_from_track()이 다음 tick부터도 "보정 이전"(틀린
                # heading으로 주행하던 동안의) GPS 이력을 계속 거슬러 올라가 재계산하고,
                # 그 오염된 구간이 min_disp_m(1.5m)만큼 흘러나갈 때까지(약 15~16틱)
                # 또 똑같이 틀어진 값으로 수렴해버림 — 실측으로 확인됨
                # (deploy_20261004_103114.jsonl, route_corrected가 매번 15~16틱
                # 주기로 똑같은 -50°대 값에 반복 수렴). _confirm_route_alignment()가
                # 이미 하는 것과 동일한 조치.
                self.past_track.clear()
                smoothed_deg = math.degrees(map_heading_rad)
                record["smoothed_heading_deg"] = smoothed_deg
                record["map_heading_source"] = "route_corrected"
                self.state.log(f"⚠ heading이 route와 {diff_deg:+.1f}° 어긋난 채 "
                                f"{self._heading_divergence_ticks}틱 지속 — route_bearing"
                                f"({smoothed_deg:+.1f}°)으로 강제 재동기화 (이전: {prev_source} "
                                f"{prev_deg:+.1f}°)")
                self._log_jsonl({"type": "event", "ts": time.time(), "tick_id": tick_id,
                                  "event": "heading_route_correction",
                                  "from_source": prev_source, "from_heading_deg": prev_deg,
                                  "to_heading_deg": smoothed_deg, "diff_deg": diff_deg,
                                  "persist_ticks": self._heading_divergence_ticks})
                self._heading_divergence_ticks = 0
        else:
            self._heading_divergence_ticks = 0

        self._last_map_heading_source = record["map_heading_source"]
        self.state.update(map_heading_source=self._last_map_heading_source)
        self._prev_step_ts = record["ts"]  # USE_GYRO_FUSION dt 계산용

        # North-up 미리보기 갱신 (2026-09-25) — frame_buffer 준비와 무관하게 매 틱
        # 갱신해서, dry-run 초반부터 "route geometry 자체"를 heading-up 회전과
        # 분리해서 볼 수 있게 함.
        northup_img = self.map_builder.get_northup_preview_image(lat, lon, heading_rad=map_heading_rad)
        if northup_img is not None:
            self.state.update(map_img_northup=northup_img)

        if len(self.frame_buffer) < N_CTX + 1:
            self.state.log("context 채우는 중 ... 정지 유지")
            record.update(linear=0.0, angular=0.0, note="context_filling")
            self._log_jsonl(record)
            return 0.0, 0.0

        pred_xy_m = self.predict_waypoints(lat, lon, map_heading_rad, tick_id=tick_id)
        linear, angular = self.waypoint_to_control(pred_xy_m)
        linear, angular = clip_control(linear, angular)
        linear_before_override, angular_before_override = linear, angular

        # 2026-10-03/04 추가: 모델 예측을 안 믿고 route_bearing 기반으로 대체하는
        # 조건들 — 우선순위대로 하나만 적용(동시에 여러 개 해당해도 가장 위 것만).
        # 전부 override_reason 하나로 로그에 남겨서 어느 조건이 발동했는지 항상 알 수 있음.
        near_goal_override_applied = False
        straight_segment_override_applied = False
        off_route_safety_applied = False
        gps_freeze_safety_applied = False
        override_reason = None

        if self.enable_gps_freeze_guard and self._gps_frozen_now:
            # 2026-10-04 추가 (실측 사고, deploy_20261004_123749.jsonl): 위치 입력 자체가
            # 신뢰 불가능한 상태라 off_route_safety/near_goal/straight_segment 전부
            # route_bearing/거리 계산에 의존하므로 같이 못 믿음 — 가장 먼저, 무조건 정지.
            gps_freeze_safety_applied = True
            override_reason = "gps_freeze_safety"
            linear, angular = 0.0, 0.0
        elif self.enable_off_route_safety and self._is_off_route_now:
            # 2026-10-04 추가 (Harness 1 제안): is_off_route()는 allow_reroute와
            # 무관하게 항상 평가됨(_ensure_route_initialized() 참고) — 여기서는 그
            # 결과로 "가벼운" 조치만 취함(무거운 전체 재라우팅은 여전히 allow_reroute
            # 쪽에서만). deploy_20261004_111807.jsonl(실측 4.85m 이동·목표쪽 진행
            # 0.38m=약 85도 이탈)처럼 heading만으로는 못 잡는 "진짜 위치 이탈"의
            # 안전망 — heading_mode와 완전히 무관하게 작동.
            off_route_safety_applied = True
            override_reason = "off_route_safety"
            if self.off_route_safety_action == "stop":
                linear, angular = 0.0, 0.0
                self.state.log(f"⚠ 경로 이탈 감지(안전망, {self.off_route_safety_threshold_m:.1f}m 초과) — 정지")
            elif self._last_route_bearing_rad is not None:
                target_time_s = 3 * WAYPOINT_STRIDE_SEC
                # 2026-10-04 추가 (실측 사고 deploy_20261004_133317.jsonl): map_heading_rad는
                # route_bearing_anchored의 pull로 이 tick만 gps_track 노이즈(양자화 격자각도
                # 등)에 오염돼 있을 수 있음 — override는 "모델/노이즈 입력 무시하고 route
                # 지오메트리로만 조향"이 목적이라, 기준 heading도 이 tick에 새로 구한
                # route_bearing_rad_now(오염 없음)를 써야 일관됨. None이면(route 끝 근처 등)
                # map_heading_rad로 폴백.
                ref_heading_rad = route_bearing_rad_now if route_bearing_rad_now is not None else map_heading_rad
                linear, angular = route_bearing_to_control(
                    self._last_route_bearing_rad, ref_heading_rad, target_time_s)
                linear, angular = clip_control(linear, angular)
                self.state.log(f"⚠ 경로 이탈 감지(안전망, {self.off_route_safety_threshold_m:.1f}m 초과) — "
                                f"route_bearing 기반 조향으로 전환")
            else:
                linear, angular = 0.0, 0.0  # route_bearing도 없으면 안전하게 정지
        elif (self.near_goal_override_dist_m > 0 and dist_to_goal_m <= self.near_goal_override_dist_m
                and self._last_route_bearing_rad is not None):
            # 2026-10-03 추가 (Harness 1 §1-8/§3-1): future-route가 화면상 너무 짧아지는
            # 근거리에서는 모델 raw 출력이 실제 지도 내용과 무관하게 결정론적으로
            # 좌회전하는 게 실측됨 — DEFAULT_NEAR_GOAL_OVERRIDE_DIST_M 상수 설명 참고.
            target_time_s = 3 * WAYPOINT_STRIDE_SEC  # waypoint_to_control()의 target_step=2와 동일 시정수
            # 2026-10-04 추가: 아래 straight_segment와 동일 이유로 map_heading_rad(오염
            # 가능) 대신 route_bearing_rad_now를 기준으로 씀.
            ref_heading_rad = route_bearing_rad_now if route_bearing_rad_now is not None else map_heading_rad
            linear, angular = route_bearing_to_control(
                self._last_route_bearing_rad, ref_heading_rad, target_time_s)
            linear, angular = clip_control(linear, angular)
            near_goal_override_applied = True
            override_reason = "near_goal"
            self.state.log(f"near-goal override 발동(dist={dist_to_goal_m:.1f}m ≤ "
                            f"{self.near_goal_override_dist_m:.1f}m): 모델 예측 angular="
                            f"{angular_before_override:+.3f} 대신 route_bearing 기반 "
                            f"angular={angular:+.3f} 사용")
        elif self.enable_straight_segment_override and self._route_initialized:
            # 2026-10-04 추가 (사용자 요청): "OSM map이 직진 경로를 보여주면 로봇도
            # 반드시 직진하게" — 모델 자체의 미세 좌편향은 지도가 완벽해도 재현됨
            # (deploy_20261004_104301.jsonl tick 37, heading_route_diff_deg=0인데도
            # 좌회전 예측) — 지도를 아무리 정확하게 넣어도 안 없어지므로, 직진
            # 구간에서는 모델 예측 자체를 안 쓰는 것만이 구조적으로 보장되는 방법.
            # 짧은/긴 lookahead의 route_bearing 차이로 "당분간 직진"을 판정 — 턴이
            # 다가오면 두 값이 벌어져서 자동으로 모델 예측이 다시 쓰임(턴 구간은
            # 지금까지 보니 모델이 지도 내용을 비교적 잘 따라감, §1-2 20m+ 버킷 참고).
            near_b = self.map_builder.route_bearing_rad(lat, lon, lookahead_m=self.straight_near_lookahead_m)
            far_b = self.map_builder.route_bearing_rad(lat, lon, lookahead_m=self.straight_far_lookahead_m)
            if near_b is not None and far_b is not None:
                straight_diff_deg = abs((math.degrees(far_b - near_b) + 180) % 360 - 180)
                if straight_diff_deg <= self.straight_angle_threshold_deg:
                    target_time_s = 3 * WAYPOINT_STRIDE_SEC
                    # 2026-10-04 추가 (실측 사고 deploy_20261004_133317.jsonl, tick40-42):
                    # map_heading_rad가 route_bearing_anchored의 pull로 이 tick만
                    # gps_track 노이즈(양자화 격자각도)에 오염돼 있으면, 직진 구간인데도
                    # 그 noise를 "현재 방향"으로 오인해서 순간적으로 풀 강도(MAX_W) 조향을
                    # 냈던 버그 — near_b/far_b처럼 이 tick에 새로 구한 오염 없는
                    # route_bearing_rad_now를 기준으로 써서 노이즈에 면역되게 함.
                    ref_heading_rad = route_bearing_rad_now if route_bearing_rad_now is not None else map_heading_rad
                    linear, angular = route_bearing_to_control(near_b, ref_heading_rad, target_time_s)
                    linear, angular = clip_control(linear, angular)
                    straight_segment_override_applied = True
                    override_reason = "straight_segment"
                    self.state.log(f"직진 구간 override 발동(근/원거리 route_bearing 차이="
                                    f"{straight_diff_deg:.1f}° ≤ {self.straight_angle_threshold_deg:.0f}°): "
                                    f"모델 예측 angular={angular_before_override:+.3f} 대신 "
                                    f"route_bearing 기반 angular={angular:+.3f} 사용")

        # 2026-09-25 추가 디버그 필드: route bearing/heading 차이/지도 회전각/target
        # waypoint — 6번(시각화)에서 합의한 "GO 누르기 전에 확인할 수치들"
        map_rotation_deg = 90.0 - math.degrees(map_heading_rad)
        route_bearing_deg = (math.degrees(self._last_route_bearing_rad)
                              if self._last_route_bearing_rad is not None else None)
        heading_route_diff_deg = None
        if route_bearing_deg is not None:
            heading_route_diff_deg = (math.degrees(map_heading_rad) - route_bearing_deg + 180) % 360 - 180
        target_x, target_y = float(pred_xy_m[2][0]), float(pred_xy_m[2][1])

        record.update(linear=linear, angular=angular, pred_xy_m=pred_xy_m.tolist(),
                       map_rotation_deg=map_rotation_deg, route_bearing_deg=route_bearing_deg,
                       heading_route_diff_deg=heading_route_diff_deg,
                       target_waypoint_xy=[target_x, target_y],
                       map_replay_path=self._last_map_replay_path,
                       linear_before_override=linear_before_override,
                       angular_before_override=angular_before_override,
                       near_goal_override_applied=near_goal_override_applied,
                       straight_segment_override_applied=straight_segment_override_applied,
                       off_route_safety_applied=off_route_safety_applied,
                       gps_freeze_safety_applied=gps_freeze_safety_applied,
                       override_reason=override_reason,
                       is_off_route=self._is_off_route_now)
        self._log_jsonl(record)
        self.state.update(route_bearing_deg=route_bearing_deg,
                           heading_route_diff_deg=heading_route_diff_deg,
                           map_rotation_deg=map_rotation_deg,
                           target_waypoint_xy=(target_x, target_y),
                           near_goal_override_applied=near_goal_override_applied,
                           straight_segment_override_applied=straight_segment_override_applied,
                           off_route_safety_applied=off_route_safety_applied,
                           gps_freeze_safety_applied=gps_freeze_safety_applied,
                           override_reason=override_reason,
                           is_off_route=self._is_off_route_now,
                           gps_frozen=self._gps_frozen_now)
        return linear, angular

    def run(self):
        # SDK 서버의 헤드리스 브라우저(pyppeteer)는 첫 요청에서 Chrome 실행 + 페이지 접속 +
        # RTM join까지 해서 수 초~십수 초가 걸림. 실시간 루프(1s 타임아웃) 안에서 이 콜드스타트를
        # 맞으면 ReadTimeout으로 죽으므로, 루프 시작 전에 넉넉한 타임아웃으로 미리 깨워둔다.
        print("[deploy] SDK 서버 워밍업 중 (헤드리스 브라우저 초기화 대기)...")
        requests.get(f"{FRODOBOT_BASE}/data", timeout=30.0)
        # 2026-09-18: /data(GPS/텔레메트리)는 준비됐는데 카메라 RTM 채널은 아직 join 중이라
        # /v2/front가 404("Front frame not available")를 반환하는 경우가 실측됨 — poll_frodobot()이
        # 이걸 그대로 .json()해서 front_frame 없는 dict를 받아 KeyError로 죽었음. 카메라도 따로
        # 준비될 때까지 재시도.
        for attempt in range(30):
            r = requests.get(f"{FRODOBOT_BASE}/v2/front", timeout=5.0)
            if r.status_code == 200 and "front_frame" in r.json():
                break
            time.sleep(1.0)
        else:
            raise RuntimeError("카메라 스트림이 30초 안에 준비되지 않음 — SDK 서버/카메라 연결 확인 필요")
        print("[deploy] 워밍업 완료")

        print("[deploy] 시작 — Ctrl+C로 정지")
        try:
            while True:
                t0 = time.time()
                try:
                    linear, angular = self.step()
                except Exception as e:
                    # 추론/네트워크 등 어떤 에러든 로봇은 반드시 정지시키고 대시보드에 기록.
                    # 에러를 삼키고 계속 움직이는 건 안전상 절대 금지 — 여기서 멈추고 재발생시킴.
                    self.state.log_error(f"step() 실패: {e!r} — 정지 명령 전송 후 중단")
                    self._log_jsonl({"type": "event", "ts": time.time(),
                                      "event": "step_failed", "error": repr(e)})
                    self.send_control(0.0, 0.0)
                    raise
                # send_control() 호출 자체는 항상 그대로 유지(건너뛰지 않음) — SDK/로봇
                # 쪽에 "N초 안에 새 명령 없으면 자동 정지"하는 watchdog이 있다는 문서/코드를
                # earth-rovers-sdk 전체에서 찾아봤지만 없었다(확인 안 된 가정에 기대는 건
                # 위험 — 이전에 "GPS 없으면 안전할 것"이라는 확인 안 된 가정으로 실제
                # 로봇이 움직인 적이 있어서 같은 실수를 반복하지 않기 위함). "아무것도 안
                # 보낸다"보다 "명시적으로 0을 보낸다"가 더 안전한 선택.
                #
                # 2026-09-25: 두 조건을 모두 만족해야 non-zero 전송:
                #   1. control_stage == "LIVE" (대시보드에서 정렬확인→ARM→GO LIVE를
                #      명시적으로 거친 경우만 — 재시작 없이 같은 프로세스에서 전환됨)
                #   2. heading이 신뢰 가능(map_heading_source != "imu_fallback") — route
                #      정렬조차 안 된 채로 실수로 GO LIVE를 누르는 극단적 경우에 대한
                #      마지막 방어선. 이번 실험에서는 GO 전에 반드시 route-align을 먼저
                #      확정하므로 정상 흐름에서는 이 조건이 걸릴 일이 없어야 함 — 걸린다면
                #      그 자체가 "정렬을 안 하고 GO LIVE를 눌렀다"는 신호.
                #   3. (2026-09-26 추가) 아직 목표에 도착하지 않았어야 함(_goal_reached
                #      래치) — 이전엔 목표 근처에 도달해도 계속 명령이 나가던 실제 갭.
                heading_trustworthy = self._last_map_heading_source != "imu_fallback"
                is_live = self.control_stage == "LIVE"
                sent_linear, sent_angular = ((linear, angular)
                                              if (is_live and heading_trustworthy and not self._goal_reached)
                                              else (0.0, 0.0))
                t_ctrl = time.time()
                try:
                    control_status = self.send_control(sent_linear, sent_angular)
                except Exception as e:
                    # 2026-10-03 추가: send_control()의 주석은 "예외가 올라가면 정지가
                    # 강제된다"고 돼 있었지만 실제로는 그렇지 않았음 — 이 호출이
                    # try/except 밖에 있어서 예외가 while 루프 전체를 뚫고 올라가
                    # 아래 KeyboardInterrupt 핸들러에도 안 걸리고 정지 시도 없이
                    # 프로세스가 바로 죽었음. unity 쪽 sibling clone의 실제 필드
                    # 테스트(2026-09-26, 장시간 유지된 SDK 서버 세션 열화로 /control이
                    # ReadTimeout 반복)에서 이 구멍이 실측됨 — 여기서도 동일한 구조라
                    # 재현 가능한 위험이었음. step() 실패 때와 같은 원칙(에러를 삼키고
                    # 계속 움직이는 건 절대 금지, 멈추고 재발생)으로 정지를 최소 1회
                    # 재시도한 뒤 그대로 재발생시킴.
                    self.state.log_error(f"send_control() 실패: {e!r} — 정지 재시도 후 중단")
                    self._log_jsonl({"type": "event", "ts": time.time(),
                                      "event": "send_control_failed", "error": repr(e),
                                      "attempted_linear": sent_linear, "attempted_angular": sent_angular})
                    try:
                        self.send_control(0.0, 0.0)
                    except Exception as e2:
                        self.state.log_error(f"정지 재시도도 실패: {e2!r} — 그래도 프로세스 종료")
                    raise
                control_latency_ms = (time.time() - t_ctrl) * 1000.0
                self._log_jsonl({"type": "control_sent", "ts": time.time(),
                                  "linear": sent_linear, "angular": sent_angular,
                                  "http_status": control_status,
                                  "latency_ms": round(control_latency_ms, 1),
                                  "control_stage": self.control_stage,
                                  "heading_trustworthy": heading_trustworthy,
                                  "goal_reached": self._goal_reached,
                                  "map_heading_source": self._last_map_heading_source,
                                  "dry_run": not is_live,
                                  "computed_linear": linear, "computed_angular": angular})
                if is_live and not heading_trustworthy:
                    self.state.log_error("LIVE인데 heading_source=imu_fallback이라 (0,0) 강제 — "
                                          "route-align을 안 하고 GO LIVE를 눌렀을 가능성")
                elapsed = time.time() - t0
                loop_hz = round(1.0 / max(elapsed, 1e-6), 2)
                self.state.update(linear=sent_linear, angular=sent_angular, loop_hz=loop_hz,
                                   control_latency_ms=round(control_latency_ms, 1),
                                   computed_linear=linear, computed_angular=angular)
                prefix = "[GOAL REACHED] " if self._goal_reached else ("[LIVE] " if is_live else f"[{self.control_stage}] ")
                show_computed = not (is_live and not self._goal_reached)
                print(f"  {prefix}linear={sent_linear:+.3f} m/s  angular={sent_angular:+.3f} rad/s"
                      + (f"  (계산값: linear={linear:+.3f} angular={angular:+.3f})" if show_computed else "")
                      + f"  [/control {control_latency_ms:.0f}ms]")
                time.sleep(max(0.0, DT - elapsed))
                # 2026-10-03 추가: 이전엔 _goal_reached 래치가 명령만 (0,0)으로 묶어두고
                # 프로세스 자체는 안 끝나서 매번 직접 Ctrl+C로 꺼야 했음("주행 완료"라고
                # 부르기 애매한 상태) — 도착 즉시(정지 명령을 최소 한 번 보낸 이번 tick
                # 다음) 루프를 빠져나가 깔끔하게 종료.
                if self._goal_reached:
                    print("[deploy] 목표 도착 완료 — 자동 종료")
                    self._log_jsonl({"type": "event", "ts": time.time(), "event": "run_end",
                                      "reason": "goal_reached"})
                    break
        except KeyboardInterrupt:
            print("\n[deploy] 정지 요청됨 — 로봇 정지 명령 전송")
            self.send_control(0.0, 0.0)
            self._log_jsonl({"type": "event", "ts": time.time(), "event": "run_end",
                              "reason": "keyboard_interrupt"})
        finally:
            if self._dashboard_recorder is not None:
                print("[deploy] 대시보드 녹화 마무리 중 (최대 10초)...")
                self._dashboard_recorder.stop()
            # 2026-10-03 추가: logs/frames/<run_id>/의 ctx_*.jpg(카메라)/map_*.png(모델
            # 입력 지도) 시퀀스를 각각 mp4로 컴파일 — 개별 파일 수백 장 대신 영상으로
            # 훑어볼 수 있게. 제어 루프가 이미 끝난 뒤의 post-processing이라 안전하게
            # 동기 호출(최대 각 60초, ReplayLogger.compile_videos() 자체 타임아웃).
            print("[deploy] frames 영상 컴파일 중...")
            video_results = self.replay_logger.compile_videos()
            for label, (ok, info) in video_results.items():
                if ok:
                    print(f"[deploy]   {label}: {info}")
                else:
                    print(f"[deploy]   {label} 생략/실패: {info}")
            print(f"[deploy] 로그 저장 완료: {self.log_path}")
            self._log_fp.close()


if __name__ == "__main__":
    # 2026-10-04 추가: 실험 파라미터를 커맨드라인 플래그로만 하드코딩/타이핑하지 않고
    # config 파일(YAML)로 묶어서 관리할 수 있게 함 — 오늘 같은 날(heading_mode/
    # near_goal_override_dist_m/rate-limit 등 여러 실험 변형을 빠르게 반복) 매번 긴
    # 커맨드를 새로 조합하지 않아도 되게. --config로 먼저 로드한 값들을 기본값으로
    # 깔고, 그 뒤에 명시적으로 준 커맨드라인 플래그가 항상 우선함(set_defaults 뒤에
    # 최종 parse_args가 실행되는 순서 그대로 적용됨). ckpt/map_range/goal_lat/goal_lon은
    # "실수로 빠뜨리면 안 됨"이라 원래 required=True였는데, config로 줄 수도 있어야
    # 하니 required는 빼고 최종 parse 뒤에 수동으로 누락 여부를 검사함(안전장치
    # 자체는 그대로 유지, 어디서 왔는지만 config/CLI 둘 다 허용).
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=str, default=None,
                      help="실험 설정을 묶어둔 YAML 파일 경로 (deployment/configs/ 참고). "
                           "여기 적힌 값들이 기본값이 되고, 그 외에 명시적으로 준 "
                           "커맨드라인 플래그가 있으면 그게 항상 우선함.")
    pre_args, _ = pre.parse_known_args()

    p = argparse.ArgumentParser(parents=[pre])
    p.add_argument("--ckpt", type=str, default=None)
    p.add_argument("--map_range", type=float, default=None,
                   help="체크포인트를 학습시킨 맵 반경과 반드시 동일해야 함 "
                        "(baseline=25, 12m실험=12, 20m실험=20). 체크포인트마다 다르므로 기본값 없음 — 반드시 명시(CLI 또는 --config).")
    p.add_argument("--goal_lat", type=float, default=None)
    p.add_argument("--goal_lon", type=float, default=None)
    p.add_argument("--debug_port", type=int, default=8080,
                   help="모니터링 웹 대시보드 포트 (0이면 비활성화)")
    p.add_argument("--dry_run", action="store_true",
                   help="이 프로세스 전체에서 GO LIVE를 영구히 잠금(대시보드에서 눌러도 "
                        "거부됨) — 순수 관찰/검증 세션용. 플래그 없이 실행해도 항상 "
                        "DRY_RUN 상태로 시작하며, 실제 로봇에 명령이 나가려면 대시보드에서 "
                        "'정렬 확인' → 'ARM' → 'GO LIVE'를 명시적으로 눌러야 함 (재시작 불필요, "
                        "같은 프로세스 안에서 frame_buffer/GPS 궤적/heading 상태 그대로 유지).")
    p.add_argument("--heading_mode", type=str, default="auto",
                   choices=["auto", "route_aligned_fixed", "route_bearing", "route_bearing_anchored"],
                   help="auto(기본) = 정렬 확인은 실측 GPS heading이 잡히기 전까지의 "
                        "임시값일 뿐, 실측(gps_track)이 확보되면 자동으로 전환됨(기존 동작). "
                        "route_aligned_fixed = 정렬 확인으로 고정한 값을 런 내내 그대로 "
                        "사용, 이후 GPS 궤적이 얼마나 잡히든 절대 안 넘어감 — 턴이 없는 "
                        "직선 구간 전용(실제 턴은 전혀 못 따라감). "
                        "route_bearing = 매 tick route_bearing_rad()를 그대로 heading으로 "
                        "사용(gps_track/EMA 전혀 안 씀) — GPS 양자화 노이즈 영향 없이, 실제 "
                        "턴도 route 지오메트리를 통해 자연스럽게 따라감. 단, 로봇이 실제로 "
                        "경로를 벗어나도 전혀 못 알아챔 — 짧고 통제된 테스트 구간 전용 "
                        "(2026-10-04, docs/experiment_log.md §1-9/1-10 참고). "
                        "route_bearing_anchored = route_bearing을 기본값으로 쓰되 "
                        "gps_track이 --heading_anchor_pull_threshold_deg 이상 "
                        "--heading_anchor_pull_persist_ticks 연속으로 반박하면 그 tick만 "
                        "gps_track을 신뢰(Harness 1 제안, 2026-10-04) — route_bearing 단독보다 "
                        "드리프트에 조금 더 반응함.")
    p.add_argument("--goal_reach_threshold_m", type=float, default=DEFAULT_GOAL_REACH_THRESHOLD_M,
                   help=f"목표까지 이 거리(m) 이내로 들어오면 도착으로 간주하고 영구 정지 "
                        f"(기본 {DEFAULT_GOAL_REACH_THRESHOLD_M}m). 한 번 도착하면 다시 안 풀림.")
    p.add_argument("--near_goal_override_dist_m", type=float, default=DEFAULT_NEAR_GOAL_OVERRIDE_DIST_M,
                   help=f"목표까지 이 거리(m) 이내에서는 모델 raw 예측 대신 route_bearing 기반 "
                        f"조향으로 대체 (기본 {DEFAULT_NEAR_GOAL_OVERRIDE_DIST_M}m — 27m 체크포인트 "
                        f"기준 실측 보정값, 다른 체크포인트/map_range면 재측정 필요, "
                        f"docs/experiment_log.md §1-8 참고). 0 이하로 주면 비활성화(비교 테스트용).")
    p.add_argument("--heading_anchor_pull_threshold_deg", type=float,
                   default=DEFAULT_HEADING_ANCHOR_PULL_THRESHOLD_DEG,
                   help=f"heading_mode=route_bearing_anchored 전용 — gps_track이 route_bearing과 "
                        f"이 각도(°) 이상 어긋나야 '반박'으로 침 (기본 {DEFAULT_HEADING_ANCHOR_PULL_THRESHOLD_DEG}°).")
    p.add_argument("--heading_anchor_pull_persist_ticks", type=int,
                   default=DEFAULT_HEADING_ANCHOR_PULL_PERSIST_TICKS,
                   help=f"heading_mode=route_bearing_anchored 전용 — 반박이 몇 틱 연속돼야 그 tick에 "
                        f"gps_track을 신뢰할지 (기본 {DEFAULT_HEADING_ANCHOR_PULL_PERSIST_TICKS}틱).")
    p.add_argument("--enable_off_route_safety", action="store_true",
                   help="is_off_route() 기반 상시 이탈 감지 안전망 켬 (기본 꺼짐, 2026-10-04 Harness 1 "
                        "제안) — allow_reroute와 무관하게 항상 평가되고, 이탈 시 --off_route_safety_action "
                        "으로 지정한 가벼운 조치만 취함(무거운 전체 재라우팅은 여전히 --allow_reroute에서만).")
    p.add_argument("--off_route_safety_threshold_m", type=float,
                   default=DEFAULT_OFF_ROUTE_SAFETY_THRESHOLD_M,
                   help=f"--enable_off_route_safety의 이탈 판정 거리(m) (기본 {DEFAULT_OFF_ROUTE_SAFETY_THRESHOLD_M}m, "
                        f"is_off_route() 자체 기본값과 동일).")
    p.add_argument("--off_route_safety_action", type=str, default=DEFAULT_OFF_ROUTE_SAFETY_ACTION,
                   choices=["stop", "steer"],
                   help=f"--enable_off_route_safety 발동 시 조치 (기본 {DEFAULT_OFF_ROUTE_SAFETY_ACTION!r}) — "
                        f"'stop'=정지, 'steer'=route_bearing_to_control()로 직접 조향.")
    p.add_argument("--enable_straight_segment_override", action="store_true",
                   help="직진 구간에서 모델 예측 대신 route_bearing 기반 조향을 강제 (기본 꺼짐, "
                        "2026-10-04 사용자 요청 — '지도가 직진이면 로봇도 반드시 직진'을 구조적으로 "
                        "보장). 근/원거리 route_bearing 차이로 직진 여부 판정 — "
                        "--straight_near_lookahead_m/--straight_far_lookahead_m/"
                        "--straight_angle_threshold_deg로 조절.")
    p.add_argument("--straight_near_lookahead_m", type=float, default=DEFAULT_STRAIGHT_NEAR_LOOKAHEAD_M,
                   help=f"--enable_straight_segment_override 전용, 짧은 lookahead(m) (기본 "
                        f"{DEFAULT_STRAIGHT_NEAR_LOOKAHEAD_M}m).")
    p.add_argument("--straight_far_lookahead_m", type=float, default=DEFAULT_STRAIGHT_FAR_LOOKAHEAD_M,
                   help=f"--enable_straight_segment_override 전용, 긴 lookahead(m) (기본 "
                        f"{DEFAULT_STRAIGHT_FAR_LOOKAHEAD_M}m).")
    p.add_argument("--straight_angle_threshold_deg", type=float,
                   default=DEFAULT_STRAIGHT_ANGLE_THRESHOLD_DEG,
                   help=f"--enable_straight_segment_override 전용 — 근/원거리 route_bearing 차이가 "
                        f"이 각도(°) 이내면 '직진'으로 판정 (기본 {DEFAULT_STRAIGHT_ANGLE_THRESHOLD_DEG}°).")
    p.add_argument("--initial_heading_lookahead_m", type=float, default=INITIAL_HEADING_LOOKAHEAD_M,
                   help=f"'정렬 확인' 시 route tangent를 계산하는 lookahead 거리(m) "
                        f"(기본 {INITIAL_HEADING_LOOKAHEAD_M}m). 목표 근처처럼 경로가 국소적으로 "
                        f"꺾이는 구간에서 짧은 값이 실제 진행 방향과 크게 어긋난 heading을 "
                        f"고정시키는 사고가 실측됨(2026-09-26) — 그런 구간에서 정렬 확인이 "
                        f"필요하면 이 값을 5~10m로 늘려서 국소 꺾임에 덜 민감하게 만들 것.")
    p.add_argument("--disable_gps_freeze_guard", action="store_true",
                   help="GPS 위치 동결 안전장치를 끔 (기본 켜짐, 2026-10-04 실측 사고 "
                        "deploy_20261004_123749.jsonl 이후 추가 — speed>0인데 위치가 "
                        "--gps_freeze_persist_ticks 이상 --gps_freeze_min_disp_m도 못 움직이면 "
                        "정지 + heading_mode=route_bearing_anchored의 gps_track pull도 보류함. "
                        "다른 메커니즘과 달리 트레이드오프가 없는 순수 안전장치라 기본 켜짐 — "
                        "끄는 건 비교 테스트 등 특수한 경우만.")
    p.add_argument("--gps_freeze_min_disp_m", type=float, default=DEFAULT_GPS_FREEZE_MIN_DISP_M,
                   help=f"GPS 동결 판정 최소 이동거리(m) — 이보다 적게 움직이면 '안 움직인 것'으로 "
                        f"침 (기본 {DEFAULT_GPS_FREEZE_MIN_DISP_M}m).")
    p.add_argument("--gps_freeze_persist_ticks", type=int, default=DEFAULT_GPS_FREEZE_PERSIST_TICKS,
                   help=f"GPS 동결 판정에 필요한 연속 tick 수 (기본 {DEFAULT_GPS_FREEZE_PERSIST_TICKS}틱).")
    p.add_argument("--gps_freeze_speed_threshold_mps", type=float,
                   default=DEFAULT_GPS_FREEZE_SPEED_THRESHOLD_MPS,
                   help=f"이 속도(m/s) 이상을 로봇이 자기보고하는 동안에만 '동결'로 판정 — 로봇이 "
                        f"실제로 정지해 있어서 위치가 안 바뀌는 정상 상황은 제외 (기본 "
                        f"{DEFAULT_GPS_FREEZE_SPEED_THRESHOLD_MPS}m/s).")
    p.add_argument("--allow_reroute", action="store_true",
                   help="기본은 재라우팅 완전 비활성화(최초 route_init 경로만 런 내내 그대로 "
                        "사용). 목표에서 멀리 떨어진 거의 정지 상태에서도 is_off_route()가 "
                        "쿨다운마다 계속 True를 반환해 reroute가 반복 발생하고, 그때마다 OSRM이 "
                        "다른 경로를 반환해 계획 경로 방향이 요동치는 문제가 실측됨(2026-09-26, "
                        "2026-09-18에도 유사 현상 기록). 이 플래그를 주면 기존 동작(경로 이탈 "
                        "감지 시 재계산, REROUTE_DISABLE_NEAR_GOAL_M 보호 포함)으로 되돌림.")

    if pre_args.config:
        import yaml
        with open(pre_args.config) as f:
            cfg = yaml.safe_load(f) or {}
        unknown = set(cfg) - {a.dest for a in p._actions}
        if unknown:
            raise ValueError(f"--config {pre_args.config}에 모르는 키가 있음: {sorted(unknown)} "
                              f"(오타 확인 — 이 CLI가 아는 플래그 이름과 정확히 일치해야 함)")
        p.set_defaults(**cfg)
        print(f"[deploy] --config {pre_args.config} 로드됨: {cfg}")

    args = p.parse_args()

    missing = [name for name in ("ckpt", "map_range", "goal_lat", "goal_lon") if getattr(args, name) is None]
    if missing:
        raise SystemExit(f"다음 값이 CLI 플래그로도, --config로도 안 주어짐(실수 방지를 위해 "
                          f"필수): {missing} — 둘 중 하나로는 반드시 지정할 것.")

    deployer = OmniVLAEdgeDeployment(
        ckpt_path=args.ckpt, map_range_m=args.map_range,
        goal_lat=args.goal_lat, goal_lon=args.goal_lon,
        debug_port=args.debug_port, dry_run=args.dry_run,
        heading_mode=args.heading_mode,
        goal_reach_threshold_m=args.goal_reach_threshold_m,
        initial_heading_lookahead_m=args.initial_heading_lookahead_m,
        allow_reroute=args.allow_reroute,
        near_goal_override_dist_m=args.near_goal_override_dist_m,
        heading_anchor_pull_threshold_deg=args.heading_anchor_pull_threshold_deg,
        heading_anchor_pull_persist_ticks=args.heading_anchor_pull_persist_ticks,
        enable_off_route_safety=args.enable_off_route_safety,
        off_route_safety_threshold_m=args.off_route_safety_threshold_m,
        off_route_safety_action=args.off_route_safety_action,
        enable_straight_segment_override=args.enable_straight_segment_override,
        straight_near_lookahead_m=args.straight_near_lookahead_m,
        straight_far_lookahead_m=args.straight_far_lookahead_m,
        straight_angle_threshold_deg=args.straight_angle_threshold_deg,
        enable_gps_freeze_guard=not args.disable_gps_freeze_guard,
        gps_freeze_min_disp_m=args.gps_freeze_min_disp_m,
        gps_freeze_persist_ticks=args.gps_freeze_persist_ticks,
        gps_freeze_speed_threshold_mps=args.gps_freeze_speed_threshold_mps,
    )
    deployer.run()
