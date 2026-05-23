"""
Step 2: Ego-centric OSM Local Path Map Generation
  - Route polyline: local OSRM Docker (foot profile)
  - Base map:       OSM tile server (tile.openstreetmap.org)
  - Output:         224x224 ego-centric, heading-up PNG per frame

Supports segment-based episode_scores.json (v2):
  Each selected entry has 'episode' and 'segment' fields.
  The segment splitting logic is re-applied to find the correct frame indices.
  Output directory: osm_maps/episode_{ep:04d}_seg{seg:02d}/

OSRM servers (must be running):
  Perth  → localhost:5001
  Taipei → localhost:5002
  Tokyo  → localhost:5003

Usage:
  python osm_map_generator.py               # selected segments only
  python osm_map_generator.py --all_episodes
  python osm_map_generator.py --ep 9        # all segments of episode 9
  python osm_map_generator.py --ep 9 --seg 1  # specific segment
"""

import os, sys, json, math, time, argparse
import numpy as np
import pyarrow as pa
import cv2
import requests
from PIL import Image
from io import BytesIO
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))
from episode_selector import split_into_segments

# ── Paths ────────────────────────────────────────────────────────────────────
ARROW_PATH  = "/media/ms/WD_BLACK_4TB/Learning-to-Drive-Anywhere-with-MBRA/FrodoBots-2K/processed/frodobots_dataset/train/data-00000-of-00001.arrow"
SCORES_PATH = "/media/ms/WD_BLACK_4TB/Learning-to-Drive-Anywhere-with-MBRA/osm_pipeline/episode_scores.json"
OUT_ROOT    = "/media/ms/WD_BLACK_4TB/Learning-to-Drive-Anywhere-with-MBRA/osm_pipeline/osm_maps"
TILE_CACHE  = "/media/ms/WD_BLACK_4TB/Learning-to-Drive-Anywhere-with-MBRA/osm_pipeline/tile_cache"

# ── OSRM server mapping (region → port) ──────────────────────────────────────
def osrm_port(lat, lon):
    if lat < -30:                          return 5001  # Perth
    if 25 < lat < 26 and 121 < lon < 122: return 5002  # Taipei
    if 35 < lat < 36 and 139 < lon < 140: return 5003  # Tokyo
    raise ValueError(f"No OSRM server for lat={lat}, lon={lon}")

# ── Config ───────────────────────────────────────────────────────────────────
MAP_SIZE_PX  = 224
MAP_RANGE_M  = 25.0    # half-width of ego view in meters
GOAL_DIST_M  = 20.0    # future horizon distance
ZOOM         = 18      # OSM tile zoom
TILE_PX      = 256
ROUTE_COLOR  = (0, 0, 255)    # BGR red
PAST_COLOR   = (160, 160, 160)
ROUTE_WIDTH  = 4
EGO_COLOR    = (0, 200, 0)
USER_AGENT   = "MBRA-Research/1.0 (minmum0206@gmail.com)"


# ── OSM tile math ─────────────────────────────────────────────────────────────

def latlon_to_tile(lat, lon, zoom):
    n = 2 ** zoom
    tx = int((lon + 180.0) / 360.0 * n)
    lat_r = math.radians(lat)
    ty = int((1.0 - math.log(math.tan(lat_r) + 1.0 / math.cos(lat_r)) / math.pi) / 2.0 * n)
    return tx, ty

def latlon_to_pixel_global(lat, lon, zoom):
    """Pixel position in the global tile grid (not relative to any tile origin)."""
    n = 2 ** zoom
    x_frac = (lon + 180.0) / 360.0 * n
    lat_r = math.radians(lat)
    y_frac = (1.0 - math.log(math.tan(lat_r) + 1.0 / math.cos(lat_r)) / math.pi) / 2.0 * n
    return x_frac * TILE_PX, y_frac * TILE_PX   # global pixel (float)


# ── Tile fetching & stitching ─────────────────────────────────────────────────

def fetch_tile(tx, ty, zoom, session):
    os.makedirs(TILE_CACHE, exist_ok=True)
    cache = os.path.join(TILE_CACHE, f"{zoom}_{tx}_{ty}.png")
    if os.path.exists(cache):
        return Image.open(cache).convert("RGB")
    url = f"https://tile.openstreetmap.org/{zoom}/{tx}/{ty}.png"
    for attempt in range(3):
        try:
            r = session.get(url, headers={"User-Agent": USER_AGENT}, timeout=10)
            r.raise_for_status()
            img = Image.open(BytesIO(r.content)).convert("RGB")
            img.save(cache)
            time.sleep(0.05)
            return img
        except Exception:
            time.sleep(1.0 * (attempt + 1))
    return None

def build_canvas(lats, lons, zoom, session):
    """
    Stitch all tiles needed to cover the episode bounding box.
    Returns (canvas_bgr, gx0, gy0):
      gx0, gy0 = global pixel offset of canvas top-left corner.
    """
    tx_vals = [latlon_to_tile(lat, lon, zoom)[0] for lat, lon in zip(lats, lons)]
    ty_vals = [latlon_to_tile(lat, lon, zoom)[1] for lat, lon in zip(lats, lons)]
    tx_min, tx_max = min(tx_vals) - 1, max(tx_vals) + 1
    ty_min, ty_max = min(ty_vals) - 1, max(ty_vals) + 1

    n_tx = tx_max - tx_min + 1
    n_ty = ty_max - ty_min + 1
    canvas = Image.new("RGB", (n_tx * TILE_PX, n_ty * TILE_PX), (255, 255, 255))

    for dy in range(n_ty):
        for dx in range(n_tx):
            tile = fetch_tile(tx_min + dx, ty_min + dy, zoom, session)
            if tile:
                canvas.paste(tile, (dx * TILE_PX, dy * TILE_PX))

    canvas_bgr = cv2.cvtColor(np.array(canvas), cv2.COLOR_RGB2BGR)
    gx0 = tx_min * TILE_PX   # global pixel x of canvas left edge
    gy0 = ty_min * TILE_PX   # global pixel y of canvas top edge
    return canvas_bgr, gx0, gy0

def global_to_canvas(gx, gy, gx0, gy0):
    return int(gx - gx0), int(gy - gy0)


# ── Ego-centric rendering ─────────────────────────────────────────────────────

def meters_per_pixel(lat, zoom):
    return 156543.03392 * math.cos(math.radians(lat)) / (2 ** zoom)

def render_frame(canvas_bgr, gx0, gy0, zoom,
                 lat_curr, lon_curr, heading_rad,
                 future_lats, future_lons,  # fixed route: points ahead of current position
                 past_lats, past_lons,      # fixed route: points behind current position
                 out_size=MAP_SIZE_PX, map_range_m=MAP_RANGE_M):
    """
    Draw past (gray) and future (red) portions of the fixed planned route,
    then crop a square around ego and rotate heading-up.
    Returns (out_size, out_size, 3) uint8 RGB.
    """
    img = canvas_bgr.copy()

    # ── Past trajectory (gray) ──
    if past_lats is not None and len(past_lats) >= 2:
        pts = []
        for lat, lon in zip(past_lats, past_lons):
            gx, gy = latlon_to_pixel_global(lat, lon, zoom)
            cx, cy = global_to_canvas(gx, gy, gx0, gy0)
            pts.append((cx, cy))
        for k in range(1, len(pts)):
            cv2.line(img, pts[k-1], pts[k], PAST_COLOR, ROUTE_WIDTH, cv2.LINE_AA)

    # ── Future route (red), starting from current ego position ──
    if future_lats is not None and len(future_lats) >= 1:
        ego_gx, ego_gy = latlon_to_pixel_global(lat_curr, lon_curr, zoom)
        ego_cx, ego_cy = global_to_canvas(ego_gx, ego_gy, gx0, gy0)
        pts = [(ego_cx, ego_cy)]
        for lat, lon in zip(future_lats, future_lons):
            gx, gy = latlon_to_pixel_global(lat, lon, zoom)
            cx, cy = global_to_canvas(gx, gy, gx0, gy0)
            pts.append((cx, cy))
        for k in range(1, len(pts)):
            cv2.line(img, pts[k-1], pts[k], ROUTE_COLOR, ROUTE_WIDTH, cv2.LINE_AA)

    # ── Ego marker ──
    ego_gx, ego_gy = latlon_to_pixel_global(lat_curr, lon_curr, zoom)
    ego_cx, ego_cy = global_to_canvas(ego_gx, ego_gy, gx0, gy0)
    cv2.circle(img, (ego_cx, ego_cy), 7, EGO_COLOR, -1)
    cv2.circle(img, (ego_cx, ego_cy), 7, (255, 255, 255), 2)

    # ── Crop & rotate ────────────────────────────────────────────────────────
    mpp = meters_per_pixel(lat_curr, zoom)
    crop_r = int(map_range_m / mpp)  # half-side in pixels at native resolution

    # Use sqrt(2)*crop_r for pre-rotation crop to avoid corner clipping
    big_r = int(crop_r * math.sqrt(2)) + 4
    pad = big_r + 10
    img_pad = cv2.copyMakeBorder(img, pad, pad, pad, pad,
                                  cv2.BORDER_CONSTANT, value=(255, 255, 255))
    cx_p = ego_cx + pad
    cy_p = ego_cy + pad

    x1 = max(0, cx_p - big_r); x2 = min(img_pad.shape[1], cx_p + big_r)
    y1 = max(0, cy_p - big_r); y2 = min(img_pad.shape[0], cy_p + big_r)
    crop = img_pad[y1:y2, x1:x2]
    if crop.size == 0:
        return np.ones((out_size, out_size, 3), dtype=np.uint8) * 255

    # Make square
    h, w = crop.shape[:2]
    sq = max(h, w)
    sq_img = cv2.copyMakeBorder(crop,
        (sq-h)//2, (sq-h+1)//2, (sq-w)//2, (sq-w+1)//2,
        cv2.BORDER_CONSTANT, value=(255, 255, 255))

    # Rotate: OSM tiles are North-up.
    # heading_rad: East=0, North=π/2 (standard math)
    # To put heading at top: rotate by (90° - heading_deg) CCW
    heading_deg = math.degrees(heading_rad)
    rot_deg = 90.0 - heading_deg
    M = cv2.getRotationMatrix2D((sq/2, sq/2), rot_deg, 1.0)
    rotated = cv2.warpAffine(sq_img, M, (sq, sq),
                              flags=cv2.INTER_LINEAR,
                              borderValue=(255, 255, 255))

    # Centre-crop to crop_r*2 then resize
    c = sq // 2
    r = min(crop_r, c)
    final = rotated[c-r:c+r, c-r:c+r]
    if final.size == 0:
        final = rotated
    final = cv2.resize(final, (out_size, out_size), interpolation=cv2.INTER_LINEAR)
    return cv2.cvtColor(final, cv2.COLOR_BGR2RGB)


# ── Per-episode processing ────────────────────────────────────────────────────

def save_gps_csv(out_dir, lats, lons):
    gps_path = os.path.join(out_dir, "gps.csv")
    with open(gps_path, 'w') as f:
        f.write("frame_index,latitude,longitude\n")
        for i, (lat, lon) in enumerate(zip(lats, lons)):
            f.write(f"{i},{lat:.8f},{lon:.8f}\n")


def save_bev(canvas_bgr, gx0, gy0, zoom, lats, lons, out_dir, pad_px=80):
    """
    OSM 타일 위에 실제 GPS 경로를 파란 선으로 표시하여 저장.
    OSM 길찾기 결과 화면처럼 지도 + 경로선만 표시 (마커·텍스트 없음).
    zoom-1 레벨로 더 넓은 맥락을 포함.
    """
    bev_zoom = max(zoom - 1, 15)
    session = requests.Session()
    bev_canvas, bev_gx0, bev_gy0 = build_canvas(lats, lons, bev_zoom, session)
    img = bev_canvas.copy()

    # 전체 GPS 경로를 파란 선으로
    pts = []
    for lat, lon in zip(lats, lons):
        gx, gy = latlon_to_pixel_global(lat, lon, bev_zoom)
        cx, cy = global_to_canvas(gx, gy, bev_gx0, bev_gy0)
        pts.append((cx, cy))

    for k in range(1, len(pts)):
        cv2.line(img, pts[k-1], pts[k], (204, 102, 0), 4, cv2.LINE_AA)  # OSM 스타일 파란색(BGR)

    # 시작점·끝점 원형 마커
    cv2.circle(img, pts[0],  8, (0, 200, 0),   -1)   # 초록
    cv2.circle(img, pts[0],  8, (255,255,255),   2)
    cv2.circle(img, pts[-1], 8, (0,  50, 230),  -1)   # 파란 끝점
    cv2.circle(img, pts[-1], 8, (255,255,255),   2)

    # 경로 bbox + 여백 crop
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    x1 = max(0, min(xs) - pad_px)
    y1 = max(0, min(ys) - pad_px)
    x2 = min(img.shape[1], max(xs) + pad_px)
    y2 = min(img.shape[0], max(ys) + pad_px)
    crop = img[y1:y2, x1:x2]

    out_path = os.path.join(out_dir, "bev_overview.png")
    Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)).save(out_path)
    return out_path


def osrm_route_full(lats, lons, fp, port,
                    waypoint_interval_m=10.0, max_waypoints=25,
                    interp_step_m=2.0):
    """
    세그먼트 전체 OSRM 경로를 1회 요청.

    Waypoint 선택 전략:
      - EKF fp 기준 누적 이동거리 매 waypoint_interval_m마다 1개 샘플링
      - 최대 max_waypoints개 (OSRM URL 길이 제한 대응)
      - 시작점·끝점 항상 포함

    반환: shape (M, 2), [[lat, lon], ...], 2m 간격으로 densify됨
    """
    # EKF 누적 거리 기반 waypoint 선택
    fp = np.array(fp)
    cum = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(fp, axis=0), axis=1))])
    total_m = cum[-1]

    n_wp = max(2, min(max_waypoints, int(total_m / waypoint_interval_m) + 1))
    target_dists = np.linspace(0, total_m, n_wp)
    wp_idxs = [int(np.searchsorted(cum, d, side='left')) for d in target_dists]
    wp_idxs = sorted(set(np.clip(wp_idxs, 0, len(lats) - 1).tolist()))

    coords_str = ";".join(f"{lons[i]},{lats[i]}" for i in wp_idxs)
    url = (f"http://localhost:{port}/route/v1/foot/{coords_str}"
           f"?overview=full&geometries=geojson")
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        d = r.json()
        raw = d["routes"][0]["geometry"]["coordinates"]  # [[lon, lat], ...]
        route = np.array([[pt[1], pt[0]] for pt in raw])  # → [[lat, lon], ...]
    except Exception:
        route = np.column_stack([lats, lons])  # fallback: 실제 GPS 사용

    return _densify_route(route, interp_step_m)


def _densify_route(route_latlon, step_m=2.0):
    """Linearly interpolate route so consecutive points are ~step_m apart."""
    LAT_M = 111320.0  # metres per degree latitude
    dense = [route_latlon[0]]
    for k in range(1, len(route_latlon)):
        p0, p1 = route_latlon[k-1], route_latlon[k]
        dlat = (p1[0] - p0[0]) * LAT_M
        dlon = (p1[1] - p0[1]) * LAT_M * math.cos(math.radians(p0[0]))
        seg_m = math.sqrt(dlat**2 + dlon**2)
        n_steps = max(1, int(seg_m / step_m))
        for s in range(1, n_steps + 1):
            t = s / n_steps
            dense.append(p0 + t * (p1 - p0))
    return np.array(dense)


def map_frames_to_route(fp_seg, route_latlon, lats_seg, lons_seg):
    """
    각 프레임을 route 상의 인덱스에 매핑.
    - EKF filtered_position (UTM XY)의 누적 거리로 프레임 진행률 계산
    - OSRM route의 누적 거리로 route 진행률 계산
    - 두 진행률을 매핑하여 각 프레임의 closest route index 반환
    반환: (N,) int array, 각 프레임의 route index (단조 증가 보장)
    """
    LAT_M = 111320.0

    # EKF 누적 거리 (프레임 진행률)
    fp = np.array(fp_seg)
    fp_dists = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(fp, axis=0), axis=1))])
    fp_total = fp_dists[-1] if fp_dists[-1] > 0 else 1.0
    fp_frac = fp_dists / fp_total  # 0~1

    # OSRM route 누적 거리 (route 진행률)
    route = np.array(route_latlon)
    route_segs = []
    for k in range(1, len(route)):
        dlat = (route[k, 0] - route[k-1, 0]) * LAT_M
        dlon = (route[k, 1] - route[k-1, 1]) * LAT_M * math.cos(math.radians(route[k-1, 0]))
        route_segs.append(math.sqrt(dlat**2 + dlon**2))
    route_cum = np.concatenate([[0], np.cumsum(route_segs)])
    route_total = route_cum[-1] if route_cum[-1] > 0 else 1.0
    route_frac = route_cum / route_total  # 0~1

    # 각 프레임의 진행률에 대응하는 route index (단조 증가)
    result = np.searchsorted(route_frac, fp_frac, side='left')
    result = np.clip(result, 0, len(route) - 1)
    # 단조 증가 보장
    for i in range(1, len(result)):
        if result[i] < result[i-1]:
            result[i] = result[i-1]
    return result


def process_episode(ep, ep_data, out_dir,
                    goal_dist_m=GOAL_DIST_M, zoom=ZOOM, out_size=MAP_SIZE_PX):
    os.makedirs(out_dir, exist_ok=True)

    lats = np.array(ep_data['lats'])
    lons = np.array(ep_data['lons'])
    fp   = np.array(ep_data['filtered_pos'])
    fh   = np.array(ep_data['filtered_heading'])

    save_gps_csv(out_dir, lats, lons)

    port = osrm_port(lats.mean(), lons.mean())
    session = requests.Session()

    # OSM 타일 캔버스 1회 빌드
    canvas_bgr, gx0, gy0 = build_canvas(lats, lons, zoom, session)

    # BEV 조감도 저장
    save_bev(canvas_bgr, gx0, gy0, zoom, lats, lons, out_dir)

    # ── OSRM 경로 1회 요청 (세그먼트 전체, EKF 거리 기반 waypoint 샘플링) ────
    route_latlon = osrm_route_full(lats, lons, fp, port)
    # route_latlon: shape (M, 2), [[lat, lon], ...], 2m 간격 densified

    # 각 프레임의 route index를 EKF 진행률 기반으로 미리 계산 (1회)
    frame_route_idx = map_frames_to_route(fp, route_latlon, lats, lons)

    saved = 0
    for i in tqdm(range(len(lats)), desc=f"ep{ep:03d}", leave=False):
        out_path = os.path.join(out_dir, f"osm_map_{i:06d}.png")
        if os.path.exists(out_path):
            saved += 1
            continue

        closest_idx = frame_route_idx[i]

        # 고정 경로를 과거/미래로 분할
        past_route   = route_latlon[:closest_idx + 1]
        future_route = route_latlon[closest_idx + 1:]

        past_lats_r   = past_route[:, 0]   if len(past_route)   >= 2 else np.array([lats[i]])
        past_lons_r   = past_route[:, 1]   if len(past_route)   >= 2 else np.array([lons[i]])
        future_lats_r = future_route[:, 0] if len(future_route) >= 1 else np.array([])
        future_lons_r = future_route[:, 1] if len(future_route) >= 1 else np.array([])

        img = render_frame(
            canvas_bgr, gx0, gy0, zoom,
            lats[i], lons[i], fh[i],
            future_lats_r, future_lons_r,
            past_lats_r,   past_lons_r,
            out_size=out_size,
        )

        Image.fromarray(img).save(out_path)
        saved += 1

    return saved


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    os.makedirs(OUT_ROOT, exist_ok=True)

    with open(SCORES_PATH) as f:
        scores = json.load(f)

    # Build list of (episode, segment) pairs to process
    if args.ep is not None and args.seg is not None:
        targets = [(args.ep, args.seg)]
    elif args.ep is not None:
        ep_segs = {(s['episode'], s.get('segment', 0))
                   for s in scores if s['episode'] == args.ep}
        targets = sorted(ep_segs)
    elif args.all_episodes:
        targets = sorted({(s['episode'], s.get('segment', 0)) for s in scores})
    else:
        targets = sorted({(s['episode'], s.get('segment', 0))
                          for s in scores if s['selected']})

    print(f"Generating OSM maps for {len(targets)} segments")
    print(f"  zoom={args.zoom}, range={args.map_range}m, goal={args.goal_dist}m")
    print(f"  OSRM: Perth:5001  Taipei:5002  Tokyo:5003")

    print("Loading dataset...")
    table = pa.ipc.open_stream(open(ARROW_PATH, 'rb')).read_all()
    ep_idx_arr = np.array(table['episode_index'].to_pylist())
    lats_arr   = np.array(table['observation.latitude'].to_pylist())
    lons_arr   = np.array(table['observation.longitude'].to_pylist())
    fp_arr     = np.array(table['observation.filtered_position'].to_pylist())
    fh_arr     = np.array(table['observation.filtered_heading'].to_pylist())

    # Group targets by episode to avoid re-splitting per segment
    from collections import defaultdict
    by_ep = defaultdict(list)
    for ep, seg in targets:
        by_ep[ep].append(seg)

    for ep in tqdm(sorted(by_ep.keys()), desc="Episodes"):
        mask   = ep_idx_arr == ep
        ep_fp  = fp_arr[mask]
        ep_lats = lats_arr[mask]
        ep_lons = lons_arr[mask]
        ep_fh  = fh_arr[mask]

        # Re-split into segments using the same logic as episode_selector
        segments = split_into_segments(ep_fp, ep_lats, ep_lons)

        for seg_idx in sorted(by_ep[ep]):
            if seg_idx >= len(segments):
                tqdm.write(f"  ep{ep:03d}[{seg_idx}]: segment not found, skipping")
                continue

            seg = segments[seg_idx]
            idxs = seg['frame_indices']  # indices into episode frame array

            seg_data = {
                'lats':             ep_lats[idxs].tolist(),
                'lons':             ep_lons[idxs].tolist(),
                'filtered_pos':     ep_fp[idxs].tolist(),
                'filtered_heading': ep_fh[idxs].tolist(),
            }

            out_dir = os.path.join(OUT_ROOT, f"episode_{ep:04d}_seg{seg_idx:02d}")
            n = process_episode(ep, seg_data, out_dir,
                                goal_dist_m=args.goal_dist,
                                zoom=args.zoom,
                                out_size=args.out_size)
            tqdm.write(f"  ep{ep:03d}[{seg_idx}]: {n} maps → {out_dir}")

    print("\nDone.  © OpenStreetMap contributors")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--goal_dist",    type=float, default=GOAL_DIST_M)
    parser.add_argument("--map_range",    type=float, default=MAP_RANGE_M)
    parser.add_argument("--zoom",         type=int,   default=ZOOM)
    parser.add_argument("--out_size",     type=int,   default=MAP_SIZE_PX)
    parser.add_argument("--ep",           type=int,   default=None)
    parser.add_argument("--seg",          type=int,   default=None,
                        help="Specific segment index (used with --ep)")
    parser.add_argument("--all_episodes", action="store_true")
    args = parser.parse_args()
    main(args)
