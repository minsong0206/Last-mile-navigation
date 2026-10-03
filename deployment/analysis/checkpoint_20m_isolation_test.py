"""
좌편향 재조사 — 20m 체크포인트(구세대)로 §1-4/§1-6과 같은 "거리-버킷 분석" 재현.

checkpoint_12m_isolation_test.py(§1-6)와 완전히 동일한 방법론 — 차이는 체크포인트와
`map_range_m`뿐. HF에서 확인한 결과(`minsonganingee/omnivla-edge-rides11-odom-20m`,
생성일 2026-08-08) 이 체크포인트도 **2026-09-05 geometry 변경 이전 구세대**:

  | | 20m(구세대, 이 스크립트) | 27m(신세대, §1-2/1-4/1-5) |
  |---|---|---|
  | 맵 지오메트리 | 정중앙 대칭 crop(half-width) | 전방 reach + REAR_RATIO 앵커 |
  | map_range_m 의미 | half-width(=20.0) | 전방 reach |
  | 타일 소스/zoom | tile.openstreetmap.org, zoom18 | cartocdn voyager_nolabels, zoom19 |
  | WAYPOINT_STRIDE(학습) | 3 (≈2m horizon) | 7 (≈5m horizon) |
  | WAYPOINT_STRIDE_SEC(배포 상수) | 0.3 | 0.7 |

공간 쪽은 구세대 `osm_map_generator_osrm_backup.render_frame()`(import만, 수정
없음)으로, 시간 쪽은 `omnivla_edge_deploy.WAYPOINT_STRIDE_SEC`을 0.3으로
monkeypatch해서 맞춘다 (production 코드 주석의 지시 그대로 — 파일 수정 없음).

**한계(§1-6과 동일)**: 구세대 원 타일 소스(tile.openstreetmap.org, zoom18)가
TOTP 정책으로 막혀있어, 지오메트리는 구세대로 정확히 맞췄지만 타일 자체는
cartocdn 캐시(zoom19)를 재사용함 — 색감/라벨 스타일만 학습 분포와 다름.

체크포인트는 리포 안 `checkpoints/omnivla_edge_rides11_odom_20m_hf_20260808/`에
저장됨(HF에서 다운로드, 사용자 요청).

실행: python3 deployment/analysis/checkpoint_20m_isolation_test.py [run_id]
  (run_id 기본값: 20260926_132829 — 27m 체크포인트가 기록한 실제 GPS/heading/
   past_track/카메라 프레임을 그대로 재사용. 로봇 구동 없음.)
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
from build_live_map import LiveMapBuilder, build_canvas  # noqa: E402
import osm_map_generator_osrm_backup as oldgen  # noqa: E402

RUN_ID = sys.argv[1] if len(sys.argv) > 1 else "20260926_132829"
LOG_PATH = REPO_ROOT / "deployment" / "logs" / f"deploy_{RUN_ID}.jsonl"
FRAMES_DIR = REPO_ROOT / "deployment" / "logs"
CKPT_20M_PATH = REPO_ROOT / "checkpoints" / "omnivla_edge_rides11_odom_20m_hf_20260808" / "best.pth"

OLDGEN_WAYPOINT_STRIDE_SEC = 0.3
OLDGEN_MAP_RANGE_M = 20.0  # half-width

TARGET_BUCKETS = ["25-30m", "20-25m", "15-20m", "10-15m", "5-10m"]

# ── 1. 로그 파싱 ──────────────────────────────────────────────────────────────
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
      f"모델/지도는 20m 구세대로 전부 교체)")

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
ng_builder = LiveMapBuilder()
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


# ── 3. 20m 체크포인트 로드 ──────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = dep.OmniVLA_edge_odom(**dep.MODEL_PARAMS)
ckpt = torch.load(CKPT_20M_PATH, map_location="cpu")
model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=True)
model.to(device).eval()
print(f"[info] 20m 체크포인트 로드 OK (strict=True 통과) on {device}  path={CKPT_20M_PATH}")

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
              f"20m_pred={ang:+.4f}, orig_27m_run_logged={s['angular']:+.4f})")

print(f"\n[info] 총 {n_tested}개 tick 테스트 (버킷: {TARGET_BUCKETS})")

print("\n=== 20m 구세대 체크포인트 — 거리 버킷별 결과 (양수=좌, 음수=우) ===")
print(f"{'bucket':8s} {'n':>4s}  {'mean':>9s}  {'std':>8s}  {'n_left':>7s}  {'n_right':>8s}")
for b in TARGET_BUCKETS:
    vals = np.array([a for a, _ in results[b]])
    if len(vals) == 0:
        continue
    print(f"{b:8s} {len(vals):4d}  {vals.mean():+9.4f}  {vals.std():8.4f}  "
          f"{(vals > 0).sum():7d}  {(vals < 0).sum():8d}")

out_path = REPO_ROOT / "deployment" / "analysis" / f"checkpoint_20m_isolation_results_{RUN_ID}.json"
with open(out_path, "w") as f:
    json.dump({b: v for b, v in results.items()}, f, indent=2)
print(f"\n[info] raw results saved to {out_path}")
