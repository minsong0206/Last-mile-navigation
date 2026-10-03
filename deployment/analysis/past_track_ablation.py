"""
좌편향 재조사 — past_track(회색 지나온 경로 trail) ablation 실험.

deploy_20260926_132829.jsonl (heading_mode=route_aligned_fixed, 27m 체크포인트)에서
heading_route_diff_deg<5°로 필터링한 176개 tick(목표 25-30m -> 5-10m, 우->좌로
역전되는 패턴이 관찰된 구간)에 대해, 실제 production 코드(LiveMapBuilder.get_map_image,
OmniVLA_edge_odom 모델)를 그대로 재사용해서 두 가지 버전으로 다시 추론한다:

  A) 실제 past_track 재구성(그 시점까지 지나온 GPS 궤적 그대로) — 원본 재현 sanity check
  B) past_track=None (회색 trail 전혀 없음) — ablation

A의 angular 부호가 로그에 남은 원본 angular와 거리 버킷별로 얼추 맞으면 재현 파이프라인이
올바르다는 뜻. B에서도 같은 "멀면 우, 가까우면 좌" 패턴이 그대로 나오면 past_track은
원인이 아니라는 뜻이고, B에서 패턴이 사라지거나 뒤집히면 past_track 렌더링이 적어도
부분적인 원인이라는 뜻.

로봇 구동 전혀 없음 — 순수 오프라인 재추론.

실행: python3 deployment/analysis/past_track_ablation.py [run_id]
  run_id 생략 시 아래 기본값(2026-09-26 첫 발견 run) 사용. 다른 run을 재분석하려면
  `deploy_<run_id>.jsonl`이 heading_mode=route_aligned_fixed로 기록된 run이어야
  거리 버킷 필터링(heading_route_diff_deg<5°)이 의미 있다 — auto 모드 run은
  이 필터 조건을 만족하는 tick이 거의 없을 수 있음.
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
sys.path.insert(0, str(REPO_ROOT / "third_party" / "omnivla" / "inference"))
import os
os.environ.setdefault("CARTO_API_KEY", "cb1_2yj0_1_ffcc69174af013089c6c5da6")

import omnivla_edge_deploy as dep  # noqa: E402

RUN_ID = sys.argv[1] if len(sys.argv) > 1 else "20260926_132829"
LOG_PATH = REPO_ROOT / "deployment" / "logs" / f"deploy_{RUN_ID}.jsonl"
FRAMES_DIR = REPO_ROOT / "deployment" / "logs"  # map_replay_path/context_frame_paths는 이 기준 상대경로

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
MAP_RANGE_M = run_start["map_range_m"]
print(f"[info] map_range_m={MAP_RANGE_M}  ckpt={run_start['ckpt_path']}  heading_mode={run_start['heading_mode']}")

route_latlon = np.array(route_init["route_latlon"])
print(f"[info] route points={len(route_latlon)}")

# 분석 대상: heading_route_diff_deg<5°, map_heading_source=='route_aligned'
aligned_ticks = sorted(
    tid for tid, s in steps.items()
    if s.get("map_heading_source") == "route_aligned"
    and s.get("heading_route_diff_deg") is not None
    and abs(s["heading_route_diff_deg"]) < 5.0
)
print(f"[info] aligned tick count={len(aligned_ticks)} range=({aligned_ticks[0]},{aligned_ticks[-1]})")

# ── 2. past_track 실제 재구성 (tick -> list[(lat,lon)]) ─────────────────────────
# 두 번째 '정렬확인'이 past_track을 비운 시점 = aligned_ticks[0] 근방. 안전하게
# 전체 tick 범위를 돌면서 route_alignment_confirmed 이벤트 타임스탬프를 기준으로 clear.
confirm_events = []
with open(LOG_PATH) as f:
    for line in f:
        r = json.loads(line)
        if r.get("type") == "event" and r.get("event") == "route_alignment_confirmed":
            confirm_events.append(r["ts"])
confirm_events.sort()
print(f"[info] confirm events at ts={confirm_events}")

past_track_at = {}
history = []
last_confirm_idx = 0
all_tick_ids = sorted(steps.keys())
for tid in all_tick_ids:
    s = steps[tid]
    if not s.get("gps_fix_ok", True):
        past_track_at[tid] = list(history)
        continue
    # 이 tick의 poll 시점 이전에 발생한 confirm이 있으면 clear (step() 내부 순서와 동일:
    # _apply_pending_commands()가 past_track.append()보다 먼저 실행됨)
    while last_confirm_idx < len(confirm_events) and confirm_events[last_confirm_idx] <= s["ts"]:
        history = []
        last_confirm_idx += 1
    history.append((s["lat"], s["lon"]))
    history = history[-200:]
    past_track_at[tid] = list(history)

print(f"[info] past_track length at tick44={len(past_track_at.get(44, []))}, "
      f"at tick219={len(past_track_at.get(219, []))}")

# ── 3. LiveMapBuilder — 캐싱된 route로 바로 세팅 (OSRM 재쿼리 불필요) ──────────────
from build_live_map import LiveMapBuilder, build_canvas  # noqa: E402

builder = dep.LiveMapBuilder(map_range_m=MAP_RANGE_M)
builder._route_latlon = route_latlon
builder._canvas = build_canvas(route_latlon[:, 0], route_latlon[:, 1], builder.zoom, builder.session)
print("[info] canvas built OK (cache hit 여부는 네트워크 호출 없었으면 성공)")

# ── 4. 모델 로드 ──────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = dep.OmniVLA_edge_odom(**dep.MODEL_PARAMS)
ckpt = torch.load(REPO_ROOT / run_start["ckpt_path"], map_location="cpu")
model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=True)
model.to(device).eval()
print(f"[info] model loaded on {device}")

obs_transform = transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.ToTensor(),
    transforms.Normalize(dep.IMG_MEAN, dep.IMG_STD),
])


def load_ctx_tensor(paths, mirror=False):
    frames = []
    for p in paths:
        img = Image.open(FRAMES_DIR / p).convert("RGB")
        if mirror:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        frames.append(obs_transform(img))
    obs_stack = torch.cat(frames, dim=0).unsqueeze(0).to(device)  # (1,18,96,96)
    return obs_stack


def run_model(obs_stack, map_np):
    map_tensor = builder.transform(Image.fromarray(map_np)).unsqueeze(0).to(device)
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
    return angular, linear


# ── 5. 버킷 정의 ──────────────────────────────────────────────────────────────
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


results = defaultdict(lambda: {"orig": [], "A_real": [], "B_none": [], "C_mirror_map": [], "D_mirror_cam": []})

for i, tid in enumerate(aligned_ticks):
    s = steps[tid]
    lat, lon = s["lat"], s["lon"]
    heading_rad = math.radians(s["smoothed_heading_deg"])
    dist = s["dist_to_goal_m"]
    b = bucket(dist)

    obs_stack = load_ctx_tensor(s["context_frame_paths"])
    obs_stack_mirror = load_ctx_tensor(s["context_frame_paths"], mirror=True)

    # A) 실제 past_track 재구성
    pt_real = past_track_at[tid]
    map_A = builder.get_map_image(lat, lon, heading_rad, past_track=pt_real if pt_real else None)
    ang_A, _ = run_model(obs_stack, map_A)

    # B) past_track 없음 (ablation)
    map_B = builder.get_map_image(lat, lon, heading_rad, past_track=None)
    ang_B, _ = run_model(obs_stack, map_B)

    # C) B를 좌우 반전 — 카메라는 그대로, 지도(trail 없는 버전)만 mirror.
    map_C = map_B[:, ::-1, :].copy()
    ang_C, _ = run_model(obs_stack, map_C)

    # D) 지도는 B(원래), 카메라만 좌우 반전 — map의 영향과 분리해서 카메라 쪽 영향만 확인.
    ang_D, _ = run_model(obs_stack_mirror, map_B)

    results[b]["orig"].append(s["angular"])
    results[b]["A_real"].append(ang_A)
    results[b]["B_none"].append(ang_B)
    results[b]["C_mirror_map"].append(ang_C)
    results[b]["D_mirror_cam"].append(ang_D)

    if i % 30 == 0:
        print(f"  tick={tid} dist={dist:.1f}m bucket={b} orig={s['angular']:+.4f} "
              f"A_real={ang_A:+.4f} B_none={ang_B:+.4f} C_mirror_map={ang_C:+.4f} D_mirror_cam={ang_D:+.4f}")

print("\n=== 거리 버킷별 결과 (좌=양수, 우=음수) ===")
print(f"{'bucket':8s} {'n':>4s}  {'orig':>9s}  {'A_real':>9s}  "
      f"{'B_none':>9s}  {'C_mirMap':>9s}  {'D_mirCam':>9s}")
for b in ["25-30m", "20-25m", "15-20m", "10-15m", "5-10m"]:
    r = results[b]
    n = len(r["orig"])
    if n == 0:
        continue
    def mean(arr):
        return sum(arr) / len(arr)
    print(f"{b:8s} {n:4d}  {mean(r['orig']):+9.4f}  {mean(r['A_real']):+9.4f}  "
          f"{mean(r['B_none']):+9.4f}  {mean(r['C_mirror_map']):+9.4f}  {mean(r['D_mirror_cam']):+9.4f}")

out_path = REPO_ROOT / "deployment" / "analysis" / f"past_track_ablation_results_{RUN_ID}.json"
with open(out_path, "w") as f:
    json.dump({b: v for b, v in results.items()}, f, indent=2)
print(f"\n[info] raw results saved to {out_path}")
