"""
좌편향 재조사 — "맵 스케일(map_range_m)이 실제 goal 근접도와 무관하게 근거리 좌편향을
유발하는가" 분리 실험 (§1-2 past_track_ablation.py의 후속).

배경 (docs/experiment_log.md §1-2): 같은 run(deploy_20260926_132829.jsonl,
heading_mode=route_aligned_fixed, 27m 체크포인트)에서 past_track/지도/카메라를
전부 ablation/mirror 해봐도, goal까지 20m 이내에서는 "뭘 넣어도 좌회전"이라는
입력-무관 학습된 prior가 관찰됨. CLAUDE.md의 유력 가설 중 하나는 "맵 스케일(25-27m)이
실제 예측 horizon(~2m)에 비해 과하게 넓어서, goal에 가까워질수록 남은 실제 경로가
지도 위 몇 픽셀로 뭉개져 곡률 신호가 소실된다"는 것.

이 스크립트는 그 가설만 분리해서 테스트한다: 5-10m 버킷(좌편향이 가장 강하게
관찰된 real tick들)의 실제 GPS 위치/heading/past_track/카메라 프레임은 전부
그대로 두고, **지도를 렌더링하는 map_range_m만** 원래 값(27m)보다 작게(15/10/5m)
바꿔서 같은 route/canvas에서 다시 렌더링한다 (LiveMapBuilder는 canvas를
map_range_m과 독립적으로 한 번만 만들고, get_map_image()가 호출 시점의
self.map_range_m으로 crop/rescale하므로 canvas/route를 공유한 채로 바로 가능).

읽는 법:
  - 좌편향이 "실제로 goal에 가깝다"는 사실 자체(데이터 불균형 prior) 때문이라면,
    map_range_m을 줄여도(=남은 경로 곡률이 더 많은 픽셀로 또렷하게 보여도) 좌편향이
    그대로 남아야 한다.
  - 좌편향이 "맵이 너무 넓어서 근거리 곡률이 뭉개지는 것" 때문이라면, map_range_m을
    줄일수록(실제 축척이 실제 예측 horizon에 가까워질수록) 좌편향이 약해지거나
    사라지거나 원래 경로 방향(우회전 많음 — docs 참고)에 맞게 바뀌어야 한다.

주의: 모델은 27m 스케일로 학습됐으므로, 다른 map_range_m으로 렌더링한 입력은 모델
입장에서 분포 밖(OOD)이다. 그 자체가 "재학습 없이 지금 당장 쓸 해법"이라는 뜻은
아니고, 순수 진단용 — 어느 가설이 맞는지 가르는 게 목적.

로봇 구동 없음 — 순수 오프라인 재추론 (deployment/*.py는 읽기만 함).

실행: python3 deployment/analysis/map_scale_isolation_test.py [run_id]
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
from build_live_map import LiveMapBuilder, build_canvas  # noqa: E402

RUN_ID = sys.argv[1] if len(sys.argv) > 1 else "20260926_132829"
LOG_PATH = REPO_ROOT / "deployment" / "logs" / f"deploy_{RUN_ID}.jsonl"
FRAMES_DIR = REPO_ROOT / "deployment" / "logs"

TEST_MAP_RANGES = [27.0, 15.0, 10.0, 5.0]  # 27.0 = 원본(학습 때 스케일) 대조군
TARGET_BUCKETS = ["5-10m", "10-15m"]  # 좌편향이 가장 강했던 구간 (§1-2 표 참고)

# ── 1. 로그 파싱 (past_track_ablation.py와 동일 로직) ─────────────────────────────
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
ORIG_MAP_RANGE_M = run_start["map_range_m"]
print(f"[info] orig map_range_m={ORIG_MAP_RANGE_M}  ckpt={run_start['ckpt_path']}  "
      f"heading_mode={run_start['heading_mode']}")

route_latlon = np.array(route_init["route_latlon"])

aligned_ticks = sorted(
    tid for tid, s in steps.items()
    if s.get("map_heading_source") == "route_aligned"
    and s.get("heading_route_diff_deg") is not None
    and abs(s["heading_route_diff_deg"]) < 5.0
)
print(f"[info] aligned tick count={len(aligned_ticks)}")

# past_track 재구성 (past_track_ablation.py와 동일)
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
all_tick_ids = sorted(steps.keys())
for tid in all_tick_ids:
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

# ── 2. 공유 canvas 1번만 빌드, map_range_m별로 LiveMapBuilder 인스턴스만 다르게 ──────
_shared_session_builder = LiveMapBuilder(map_range_m=ORIG_MAP_RANGE_M)
shared_canvas = build_canvas(route_latlon[:, 0], route_latlon[:, 1],
                              _shared_session_builder.zoom, _shared_session_builder.session)
print("[info] shared canvas built (route 전체 1회만 — map_range_m과 무관, 공유)")

builders = {}
for mr in TEST_MAP_RANGES:
    b = LiveMapBuilder(map_range_m=mr)
    b._route_latlon = route_latlon
    b._canvas = shared_canvas  # canvas는 zoom에만 의존, map_range_m과 무관 → 공유 가능
    builders[mr] = b

# ── 3. 모델 로드 ──────────────────────────────────────────────────────────────
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


def load_ctx_tensor(paths):
    frames = [obs_transform(Image.open(FRAMES_DIR / p).convert("RGB")) for p in paths]
    return torch.cat(frames, dim=0).unsqueeze(0).to(device)


def run_model(obs_stack, map_np):
    builder_for_tf = builders[TEST_MAP_RANGES[0]]  # transform(Resize+Normalize)은 map_range_m과 무관, 재사용
    map_tensor = builder_for_tf.transform(Image.fromarray(map_np)).unsqueeze(0).to(device)
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
results = defaultdict(lambda: {mr: [] for mr in TEST_MAP_RANGES})
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

    for mr in TEST_MAP_RANGES:
        builder = builders[mr]
        map_img = builder.get_map_image(lat, lon, heading_rad, past_track=pt_real)
        ang = run_model(obs_stack, map_img)
        results[b][mr].append(ang)

    n_tested += 1
    if n_tested % 20 == 0:
        print(f"  ...{n_tested} ticks tested (tid={tid}, dist={dist:.1f}m, bucket={b})")

print(f"\n[info] 총 {n_tested}개 tick 테스트 (버킷: {TARGET_BUCKETS})")

# ── 5. 결과 ───────────────────────────────────────────────────────────────────
print("\n=== map_range_m별 평균 angular (양수=좌, 음수=우) ===")
print(f"{'bucket':8s} {'n':>4s}  " + "  ".join(f"{mr:>9.0f}m" for mr in TEST_MAP_RANGES))
for b in TARGET_BUCKETS:
    r = results[b]
    n = len(r[TEST_MAP_RANGES[0]])
    if n == 0:
        continue
    means = [sum(r[mr]) / len(r[mr]) for mr in TEST_MAP_RANGES]
    print(f"{b:8s} {n:4d}  " + "  ".join(f"{m:+9.4f}" for m in means))

out_path = REPO_ROOT / "deployment" / "analysis" / f"map_scale_isolation_results_{RUN_ID}.json"
with open(out_path, "w") as f:
    json.dump({b: {str(mr): v for mr, v in d.items()} for b, d in results.items()}, f, indent=2)
print(f"\n[info] raw results saved to {out_path}")
