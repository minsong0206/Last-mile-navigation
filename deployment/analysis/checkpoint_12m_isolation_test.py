"""
좌편향 재조사 — 12m 체크포인트(구세대)로 같은 "map_range_m 분리" 분석 재현.

**중요 — 이 체크포인트는 §1-4(map_scale_isolation_test.py)의 27m 체크포인트와
세대가 다름**. HF에서 확인(`minsonganingee/omnivla-edge-rides11-odom-12m`, 생성일
2026-08-10)한 결과, 2026-09-05 geometry 변경 **이전**에 학습된 구세대 체크포인트임:

  | | 12m(구세대, 이 스크립트) | 20m_20260910(신세대, §1-4) |
  |---|---|---|
  | 맵 지오메트리 | 정중앙 대칭 crop(half-width) | 전방 reach + REAR_RATIO 앵커 |
  | map_range_m 의미 | half-width | 전방 reach |
  | 타일 소스/zoom | tile.openstreetmap.org, zoom18 | cartocdn voyager_nolabels, zoom19 |
  | WAYPOINT_STRIDE | 3 (≈2m horizon) | 7 (≈5m horizon) |
  | WAYPOINT_STRIDE_SEC(배포 상수) | 0.3 | 0.7 |

이 스크립트는 공간(지오메트리) 쪽은 `osm_pipeline/py/osm_map_generator_osrm_backup.py`
(구세대 render_frame — 정중앙 crop, REAR_RATIO 없음, 그대로 import해서 재사용)로
맞추고, 시간(waypoint 간격) 쪽은 `omnivla_edge_deploy.WAYPOINT_STRIDE_SEC`을 0.3으로
monkeypatch해서 맞춘다 (production 코드 주석에 이미 "구 체크포인트는 0.3으로 되돌려서
추론할 것"이라고 명시돼 있음 — deployment/omnivla_edge_deploy.py:113-117).

**남은 한계(의도적 타협, 결과 해석 시 반드시 감안)**: 구세대는 원래
tile.openstreetmap.org(zoom18) 타일로 학습됐으나, 그 서버는 2026-03 TOTP
스크래핑 방지로 이미 막혀있음(docs/0906.md). 이 스크립트는 **지오메트리(중심
배치/half-width 축척/회전)는 구세대 그대로**이지만 **타일 자체는 §1-4와 같은
cartocdn 캐시(zoom19)를 재사용**한다 — 지도 내용(도로/건물 모양)은 실제 서울
지역이라 동일하고, 색감/라벨 스타일만 다름. 축척·앵커·회전처럼 "모델이 어느
좌표를 어디서 보는지"를 결정하는 요소는 전부 올바르게 구세대로 맞췄으므로, 이
타일 소스 차이가 분석 결론(좌편향 유무)을 뒤집을 가능성은 낮다고 판단함 — 단,
완전히 동일한 학습 분포는 아님.

실행: python3 deployment/analysis/checkpoint_12m_isolation_test.py [run_id]
  (run_id 기본값: §1-4와 동일한 20260926_132829 — 27m 체크포인트가 기록한
   실제 GPS/heading/past_track/카메라 프레임을 그대로 재사용. 로봇 구동 없음.)
"""
import json
import math
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

REPO_ROOT = Path("/home/moai/Last-mile-navigation")
sys.path.insert(0, str(REPO_ROOT / "deployment"))
sys.path.insert(0, str(REPO_ROOT / "osm_pipeline" / "py"))
sys.path.insert(0, str(REPO_ROOT / "third_party" / "omnivla" / "inference"))
import os
os.environ.setdefault("CARTO_API_KEY", "cb1_2yj0_1_ffcc69174af013089c6c5da6")

import omnivla_edge_deploy as dep  # noqa: E402
from build_live_map import LiveMapBuilder, build_canvas  # noqa: E402  (canvas 재사용용, 신세대 zoom/tile)
import osm_map_generator_osrm_backup as oldgen  # noqa: E402  (구세대 render_frame만 가져다 씀, Arrow 미접근)

RUN_ID = sys.argv[1] if len(sys.argv) > 1 else "20260926_132829"
LOG_PATH = REPO_ROOT / "deployment" / "logs" / f"deploy_{RUN_ID}.jsonl"
FRAMES_DIR = REPO_ROOT / "deployment" / "logs"
CKPT_12M_PATH = REPO_ROOT / "checkpoints" / "omnivla_edge_rides11_odom_12m_hf_20260810" / "best.pth"

OLDGEN_WAYPOINT_STRIDE_SEC = 0.3
OLDGEN_MAP_RANGE_M = 12.0  # half-width

TARGET_BUCKETS = ["25-30m", "20-25m", "15-20m", "10-15m", "5-10m"]

# ── 1. 로그 파싱 (map_scale_isolation_test.py와 동일) ─────────────────────────────
run_start = None
route_init = None
steps = {}
with open(LOG_PATH) as f:
    for line in f:
        r = json.loads(line)
        t = r.get("type")
        if t == "run_start":
            run_start = r
        elif t == "event" and r.get("event") == "route_init":
            route_init = r
        elif t == "step":
            steps[r["tick_id"]] = r

assert run_start is not None and route_init is not None
print(f"[info] 원본 run: map_range_m={run_start['map_range_m']}  ckpt={run_start['ckpt_path']}  "
      f"heading_mode={run_start['heading_mode']}  (이 run의 실제 GPS/카메라만 재사용, "
      f"모델/지도는 12m 구세대로 전부 교체)")

route_latlon = np.array(route_init["route_latlon"])

aligned_ticks = sorted(
    tid for tid, s in steps.items()
    if s.get("map_heading_source") == "route_aligned"
    and s.get("heading_route_diff_deg") is not None
    and abs(s["heading_route_diff_deg"]) < 5.0
)

confirm_events = []
with open(LOG_PATH) as f:
    for line in f:
        r = json.loads(line)
        if r.get("type") == "event" and r.get("event") == "route_alignment_confirmed":
            confirm_events.append(r["ts"])
confirm_events.sort()

past_track_at = {}
history = []
last_confirm_idx = 0
for tid in sorted(steps.keys()):
    s = steps[tid]
    if not s.get("gps_fix_ok", True):
        past_track_at[tid] = list(history)
        continue
    while last_confirm_idx < len(confirm_events) and confirm_events[last_confirm_idx] <= s["ts"]:
        history = []
        last_confirm_idx += 1
    history.append((s["lat"], s["lon"]))
    history = history[-200:]
    past_track_at[tid] = list(history)

# ── 2. canvas + route — 신세대 LiveMapBuilder로 캐싱된 cartocdn 타일 재사용 ─────────
# (zoom/캔버스 생성 자체는 generation-agnostic 순수 투영 수학 — §1-4와 동일 캔버스 공유)
ng_builder = LiveMapBuilder()  # 기본 zoom(=19)만 쓰려는 용도, map_range_m은 안 씀
ng_builder._route_latlon = route_latlon
ng_builder._canvas = build_canvas(route_latlon[:, 0], route_latlon[:, 1],
                                   ng_builder.zoom, ng_builder.session)
canvas_bgr, gx0, gy0 = ng_builder._canvas
ZOOM = ng_builder.zoom
print(f"[info] canvas 재사용 완료 (zoom={ZOOM}, cartocdn 캐시 — 구세대 OSM-standard/zoom18 아님, 위 docstring 참고)")

mt_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(dep.IMG_MEAN, dep.IMG_STD),
])


def get_map_image_oldgen(lat, lon, heading_rad, past_track, out_size=96):
    """build_live_map.LiveMapBuilder.get_map_image()와 같은 전처리(과거 궤적 route-snap,
    ego를 가장 가까운 route 점으로 스냅)를 하되, 렌더링은 구세대 oldgen.render_frame()
    (정중앙 대칭 crop, REAR_RATIO 없음, map_range_m=half-width)으로 수행."""
    idx, _ = ng_builder._closest_route_idx(lat, lon)
    future_route = route_latlon[idx:]

    past_lats = past_lons = None
    if past_track:
        pt = np.array(past_track)
        d = np.hypot(pt[:, [0]] - route_latlon[:, 0][None, :],
                     pt[:, [1]] - route_latlon[:, 1][None, :])
        snapped_past = route_latlon[np.argmin(d, axis=1)]
        past_lats, past_lons = snapped_past[:, 0], snapped_past[:, 1]

    render_lat, render_lon = future_route[0]
    img = oldgen.render_frame(
        canvas_bgr, gx0, gy0, ZOOM,
        render_lat, render_lon, heading_rad,
        future_route[:, 0], future_route[:, 1],
        past_lats, past_lons,
        out_size=out_size, map_range_m=OLDGEN_MAP_RANGE_M,
    )
    return img


# ── 3. 12m 체크포인트 로드 ──────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = dep.OmniVLA_edge_odom(**dep.MODEL_PARAMS)
ckpt = torch.load(CKPT_12M_PATH, map_location="cpu")
missing_unexpected = model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=True)
model.to(device).eval()
print(f"[info] 12m 체크포인트 로드 OK (strict=True 통과) on {device}")

# WAYPOINT_STRIDE_SEC을 구세대 값으로 monkeypatch (dep.waypoint_to_control이 모듈
# 전역을 호출 시점에 참조하므로 이렇게 바꿔두면 바로 반영됨 — omnivla_edge_deploy.py는
# 수정하지 않음, 이 분석 스크립트의 프로세스 내에서만 유효)
dep.WAYPOINT_STRIDE_SEC = OLDGEN_WAYPOINT_STRIDE_SEC
print(f"[info] dep.WAYPOINT_STRIDE_SEC → {dep.WAYPOINT_STRIDE_SEC} (구세대 WAYPOINT_STRIDE=3 대응)")

obs_transform = transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.ToTensor(),
    transforms.Normalize(dep.IMG_MEAN, dep.IMG_STD),
])


def load_ctx_tensor(paths):
    frames = [obs_transform(Image.open(FRAMES_DIR / p).convert("RGB")) for p in paths]
    return torch.cat(frames, dim=0).unsqueeze(0).to(device)


def run_model(obs_stack, map_np):
    map_tensor = mt_transform(Image.fromarray(map_np)).unsqueeze(0).to(device)
    obs_cur = obs_stack[:, -3:]
    goal_pose = torch.zeros(1, 4, device=device)
    goal_mask = torch.zeros(1, dtype=torch.long, device=device)
    feat_text = torch.zeros(1, 512, device=device)
    cur_img = F.interpolate(obs_cur, (224, 224), mode="bilinear", align_corners=False)
    with torch.no_grad():
        pred, _, _ = model(obs_stack, goal_pose, map_tensor, obs_cur, goal_mask, feat_text, cur_img)
    pred_xy_m = pred[0, :, :2].detach().cpu().numpy() * dep.METRIC_WAYPOINT_SPACING
    linear, angular = dep.OmniVLAEdgeDeployment.waypoint_to_control(None, pred_xy_m)
    linear, angular = dep.clip_control(linear, angular)
    return angular


def bucket(dist):
    if dist >= 25:
        return "25-30m"
    if dist >= 20:
        return "20-25m"
    if dist >= 15:
        return "15-20m"
    if dist >= 10:
        return "10-15m"
    return "5-10m"


# ── 4. 실행 ───────────────────────────────────────────────────────────────────
results = defaultdict(list)
n_tested = 0
for tid in aligned_ticks:
    s = steps[tid]
    dist = s["dist_to_goal_m"]
    b = bucket(dist)
    if b not in TARGET_BUCKETS:
        continue

    lat, lon = s["lat"], s["lon"]
    heading_rad = math.radians(s["smoothed_heading_deg"])
    pt_real = past_track_at[tid] or None
    obs_stack = load_ctx_tensor(s["context_frame_paths"])

    map_img = get_map_image_oldgen(lat, lon, heading_rad, pt_real)
    ang = run_model(obs_stack, map_img)
    results[b].append((ang, s["angular"]))

    n_tested += 1
    if n_tested % 20 == 0:
        print(f"  ...{n_tested} ticks (tid={tid}, dist={dist:.1f}m, bucket={b}, "
              f"12m_pred={ang:+.4f}, orig_27m_run_logged={s['angular']:+.4f})")

print(f"\n[info] 총 {n_tested}개 tick 테스트 (버킷: {TARGET_BUCKETS})")

print("\n=== 12m 구세대 체크포인트 — 거리 버킷별 결과 (양수=좌, 음수=우) ===")
print(f"{'bucket':8s} {'n':>4s}  {'mean':>9s}  {'std':>8s}  {'n_left':>7s}  {'n_right':>8s}")
for b in TARGET_BUCKETS:
    vals = np.array([a for a, _ in results[b]])
    if len(vals) == 0:
        continue
    print(f"{b:8s} {len(vals):4d}  {vals.mean():+9.4f}  {vals.std():8.4f}  "
          f"{(vals > 0).sum():7d}  {(vals < 0).sum():8d}")

out_path = REPO_ROOT / "deployment" / "analysis" / f"checkpoint_12m_isolation_results_{RUN_ID}.json"
with open(out_path, "w") as f:
    json.dump({b: v for b, v in results.items()}, f, indent=2)
print(f"\n[info] raw results saved to {out_path}")
