"""
05_model_input_output visualization builder.

Reuses REAL production code paths:
  - osm_pipeline/py/osm_map_generator.py: build_canvas, render_frame, _densify_route,
    latlon_to_pixel_global, global_to_canvas, MAP_SIZE_PX, ZOOM, REAR_RATIO
  - deployment/omnivla_edge_deploy.py: estimate_heading_from_track, MODEL_PARAMS,
    IMG_MEAN/STD, METRIC_WAYPOINT_SPACING, WAYPOINT_STRIDE_SEC, N_CTX, CTX_STRIDE_SEC
  - third_party/omnivla/inference/model_omnivla_edge_odom.py: OmniVLA_edge_odom (actual checkpoint)

Only the LIVE SDK polling loop (poll_frodobot/run()) is not used, since there's no
connected robot -- real recorded sensor data from the HF dataset replaces it.
"""
import os, sys, json, math
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from PIL import Image
from huggingface_hub import hf_hub_download

from pathlib import Path as _Path
REPO_ROOT = str(_Path(__file__).resolve().parents[4])
sys.path.insert(0, os.path.join(REPO_ROOT, "osm_pipeline", "py"))
sys.path.insert(0, os.path.join(REPO_ROOT, "deployment"))
sys.path.insert(0, os.path.join(REPO_ROOT, "third_party", "omnivla", "inference"))

from osm_map_generator import (
    build_canvas, render_frame, _densify_route,
    latlon_to_pixel_global, global_to_canvas, meters_per_pixel,
    MAP_SIZE_PX, ZOOM, REAR_RATIO,
)
from omnivla_edge_deploy import (
    estimate_heading_from_track, MODEL_PARAMS, IMG_MEAN, IMG_STD,
    METRIC_WAYPOINT_SPACING, WAYPOINT_STRIDE_SEC, N_CTX, CTX_STRIDE_SEC,
)
from model_omnivla_edge_odom import OmniVLA_edge_odom
from build_live_map import LiveMapBuilder

MAP_RANGE_M = 20.0  # this checkpoint: omnivla_edge_rides11_odom_20m_20260910
CKPT_PATH = os.path.join(REPO_ROOT, "checkpoints", "omnivla_edge_rides11_odom_20m_20260910", "best.pth")
N_WAYPOINTS = 8
LAT_M = 111320.0

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "_cache")
DATA_DIR = os.path.join(CACHE_DIR, "hf_dataset")
IMG_DIR = os.path.join(DATA_DIR, "images")
os.makedirs(IMG_DIR, exist_ok=True)
HF_DATASET = "minsonganingee/frodobot-drive-2026-08-09_07-14-06"

OUT_DIR = os.path.join(CACHE_DIR, "viz_out")
os.makedirs(OUT_DIR, exist_ok=True)

# ── load raw dataset ──────────────────────────────────────────────────────
gps_rows = [json.loads(l) for l in open(f"{DATA_DIR}/gps_imu.jsonl")]
gps_t = [r["local_timestamp"] for r in gps_rows]
gps_ll = [(r["latitude"], r["longitude"]) for r in gps_rows]

ctrl_rows = [json.loads(l) for l in open(f"{DATA_DIR}/control.jsonl")]
ctrl_t = [r["local_timestamp"] for r in ctrl_rows]
ctrl_img = [r["image"] for r in ctrl_rows]


def past_track_upto(t):
    return [ll for tt, ll in zip(gps_t, gps_ll) if tt <= t]


def interp_gps(t):
    if t <= gps_t[0]:
        return gps_ll[0]
    if t >= gps_t[-1]:
        return gps_ll[-1]
    for i in range(1, len(gps_t)):
        if gps_t[i] >= t:
            t0_, t1_ = gps_t[i - 1], gps_t[i]
            la0, lo0 = gps_ll[i - 1]
            la1, lo1 = gps_ll[i]
            f = (t - t0_) / (t1_ - t0_) if t1_ > t0_ else 0.0
            return (la0 + f * (la1 - la0), lo0 + f * (lo1 - lo0))
    return gps_ll[-1]


def download_image(rel_path):
    local = os.path.join(DATA_DIR, rel_path)
    if os.path.exists(local):
        return local
    return hf_hub_download(HF_DATASET, rel_path, repo_type="dataset", local_dir=DATA_DIR)


def get_context_frames(t):
    """Replicate maybe_update_frame_buffer()'s gating: walk causally through the
    real recorded control.jsonl frames, keep a frame only if >=CTX_STRIDE_SEC has
    passed since the last kept frame. Returns the last (N_CTX+1) kept frames up to t,
    oldest first, as (timestamp, local_image_path)."""
    kept = []
    last_kept_t = None
    for tt, rel in zip(ctrl_t, ctrl_img):
        if tt > t:
            break
        if last_kept_t is None or (tt - last_kept_t) >= CTX_STRIDE_SEC:
            kept.append((tt, rel))
            last_kept_t = tt
    return kept[-(N_CTX + 1):]


def build_route_latlon():
    """Real recorded GPS trajectory as the 'future route', same as
    osm_map_generator.py::process_episode() (GT future GPS reused as fixed route --
    this is exactly how training builds its route, unlike deployment which uses live
    OSRM; here we replicate training's approach since we have the real ride recorded)."""
    lats = np.array([p[0] for p in gps_ll])
    lons = np.array([p[1] for p in gps_ll])
    gps_raw = np.column_stack([lats, lons])
    mask_unique = np.concatenate([[True], np.any(np.diff(gps_raw, axis=0) != 0, axis=1)])
    gps_unique = gps_raw[mask_unique]
    return _densify_route(gps_unique, step_m=1.0)


def make_live_map_builder(route_latlon, map_range_m=MAP_RANGE_M):
    """Real LiveMapBuilder instance, with the OSRM-dependent set_goal() bypassed --
    _route_latlon/_canvas are populated directly (route_latlon = real recorded GPS
    path; get_map_image()/route_bearing_rad()/draw_predicted_trajectory()/
    _closest_route_idx() below are then the UNMODIFIED production methods)."""
    import requests
    lmb = LiveMapBuilder(map_range_m=map_range_m)
    canvas_bgr, gx0, gy0 = build_canvas(route_latlon[:, 0], route_latlon[:, 1], ZOOM,
                                         requests.Session())
    lmb._route_latlon = route_latlon
    lmb._canvas = (canvas_bgr, gx0, gy0)
    return lmb


def compute_gt_waypoints(t, heading_rad):
    """GT future waypoints in ego-frame, same rotation formula as
    rides11_dataset.py::_get_waypoints(): x=dx_E*cos+dy_N*sin, y=-dx_E*sin+dy_N*cos.
    Derived from GT future GPS at this checkpoint's WAYPOINT_STRIDE_SEC=0.7 spacing,
    linearly interpolated from the raw ~1Hz gps_imu.jsonl (NOT the EKF-filtered
    high-rate ground truth used at training time -- see README caveat)."""
    lat0, lon0 = interp_gps(t)
    cos_h, sin_h = math.cos(heading_rad), math.sin(heading_rad)
    wps = []
    for k in range(1, N_WAYPOINTS + 1):
        t_fut = t + k * WAYPOINT_STRIDE_SEC
        lat1, lon1 = interp_gps(t_fut)
        dE = (lon1 - lon0) * LAT_M * math.cos(math.radians(lat0))
        dN = (lat1 - lat0) * LAT_M
        x_ego = dE * cos_h + dN * sin_h
        y_ego = -dE * sin_h + dN * cos_h
        wps.append([x_ego, y_ego])
    return np.array(wps, dtype=np.float32)


def prewarp_northup_crop(lmb, lat, lon, heading_rad, map_range_m=MAP_RANGE_M,
                          draw_ego=False):
    """North-up crop of the SAME tile canvas used by the real render_frame() call,
    before the heading-up warpAffine. Re-draws only the past(gray)/future(red)/goal
    route lines -- literally the same ~15 lines render_frame() itself draws pre-warp
    (see osm_map_generator.py::render_frame) -- since there is no production function
    that returns this intermediate frame (render_frame always returns the final
    warped result). The rotation/rescale/anchor math itself is NOT reimplemented here;
    that always comes from calling render_frame()/LiveMapBuilder.get_map_image()
    directly elsewhere in this script."""
    canvas_bgr, gx0, gy0 = lmb._canvas
    img = canvas_bgr.copy()
    idx, _ = lmb._closest_route_idx(lat, lon)
    route = lmb._route_latlon
    past_route, future_route = route[:idx + 1], route[idx:]

    def to_canvas_pts(seg):
        pts = []
        for la, lo in seg:
            gx, gy = latlon_to_pixel_global(la, lo, ZOOM)
            pts.append(global_to_canvas(gx, gy, gx0, gy0))
        return pts

    if len(past_route) >= 2:
        pts = to_canvas_pts(past_route)
        for k in range(1, len(pts)):
            cv2.line(img, pts[k - 1], pts[k], (160, 160, 160), 2, cv2.LINE_AA)
    ego_gx, ego_gy = latlon_to_pixel_global(future_route[0][0], future_route[0][1], ZOOM)
    ego_cx, ego_cy = global_to_canvas(ego_gx, ego_gy, gx0, gy0)
    if len(future_route) >= 1:
        pts = [(ego_cx, ego_cy)] + to_canvas_pts(future_route)
        for k in range(1, len(pts)):
            cv2.line(img, pts[k - 1], pts[k], (0, 0, 255), 2, cv2.LINE_AA)
        cv2.circle(img, pts[-1], 5, (0, 165, 255), -1, cv2.LINE_AA)

    if draw_ego:
        cv2.circle(img, (ego_cx, ego_cy), 7, (0, 200, 0), -1)
        cv2.circle(img, (ego_cx, ego_cy), 7, (255, 255, 255), 2)
        # heading arrow: East=0,North=+90 CCW math convention -> screen dx=cos(h), dy=-sin(h)
        arrow_len_px = 45
        dx = math.cos(heading_rad) * arrow_len_px
        dy = -math.sin(heading_rad) * arrow_len_px
        tip = (int(round(ego_cx + dx)), int(round(ego_cy + dy)))
        cv2.arrowedLine(img, (ego_cx, ego_cy), tip, (255, 0, 255), 3, cv2.LINE_AA, tipLength=0.35)

    rear_m = map_range_m * REAR_RATIO
    total_span_m = map_range_m + rear_m
    mpp = meters_per_pixel(lat, ZOOM)
    half_px = int((total_span_m / mpp) * 0.75)  # a bit wider than the final view for context
    h, w = img.shape[:2]
    x1, x2 = max(0, ego_cx - half_px), min(w, ego_cx + half_px)
    y1, y2 = max(0, ego_cy - half_px), min(h, ego_cy + half_px)
    crop = img[y1:y2, x1:x2]
    return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)


# ── model ──────────────────────────────────────────────────────────────────
_model = None


def get_model(device="cpu"):
    global _model
    if _model is None:
        m = OmniVLA_edge_odom(**MODEL_PARAMS)
        ckpt = torch.load(CKPT_PATH, map_location="cpu")
        m.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=True)
        m.to(device).eval()
        _model = m
    return _model


obs_transform = None


def get_obs_transform():
    global obs_transform
    if obs_transform is None:
        from torchvision import transforms
        obs_transform = transforms.Compose([
            transforms.Resize((96, 96)), transforms.ToTensor(),
            transforms.Normalize(IMG_MEAN, IMG_STD),
        ])
    return obs_transform


def run_inference(ctx_frames_pil, map_np, device="cpu"):
    """ctx_frames_pil: list of 6 PIL images oldest->newest. map_np: (224,224,3) uint8 RGB.
    Mirrors OmniVLAEdgeDeployment.build_inputs()/predict_waypoints() exactly."""
    tfm = get_obs_transform()
    frames = [tfm(im) for im in ctx_frames_pil]
    obs_stack = torch.cat(frames, dim=0).unsqueeze(0).to(device)  # (1,18,96,96)
    obs_cur = obs_stack[:, -3:]

    from torchvision import transforms
    map_tfm = transforms.Compose([
        transforms.Resize((96, 96)), transforms.ToTensor(),
        transforms.Normalize(IMG_MEAN, IMG_STD),
    ])
    map_tensor = map_tfm(Image.fromarray(map_np)).unsqueeze(0).to(device)

    goal_pose = torch.zeros(1, 4, device=device)
    goal_mask = torch.zeros(1, dtype=torch.long, device=device)
    feat_text = torch.zeros(1, 512, device=device)
    cur_img = F.interpolate(obs_cur, (224, 224), mode="bilinear", align_corners=False)

    model = get_model(device)
    with torch.no_grad():
        pred, _, _ = model(obs_stack, goal_pose, map_tensor, obs_cur, goal_mask, feat_text, cur_img)
    pred_xy_m = pred[0, :, :2].detach().cpu().numpy() * METRIC_WAYPOINT_SPACING
    return pred_xy_m


if __name__ == "__main__":
    print("module loaded OK")
    print("CTX_STRIDE_SEC", CTX_STRIDE_SEC, "WAYPOINT_STRIDE_SEC", WAYPOINT_STRIDE_SEC, "N_CTX", N_CTX)
