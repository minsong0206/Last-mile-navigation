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
  Perth      → localhost:5001
  Taipei     → localhost:5002
  Tokyo      → localhost:5003
  Wuhan      → localhost:5004
  Manila     → localhost:5005
  Rome       → localhost:5006
  Wellington → localhost:5007
  Florida    → localhost:5008
  Brighton   → localhost:5009
  Madrid     → localhost:5010

Usage:
  python osm_map_generator.py               # selected segments only
  python osm_map_generator.py --all_episodes
  python osm_map_generator.py --ep 9        # all segments of episode 9
  python osm_map_generator.py --ep 9 --seg 1  # specific segment
"""

import os, sys, json, math, time, argparse, hashlib
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
ARROW_PATH  = "/media/ms/WD_BLACK_4TB/Learning-to-Drive-Anywhere-with-MBRA/FrodoBots-2K/processed/output_rides_11/train/data-00000-of-00001.arrow"
SCORES_PATH = "/media/ms/WD_BLACK_4TB/Learning-to-Drive-Anywhere-with-MBRA/osm_pipeline/osm_data/output_rides_11/episode_scores.json"
OUT_ROOT    = "/media/ms/WD_BLACK_4TB/Learning-to-Drive-Anywhere-with-MBRA/osm_pipeline/osm_data/output_rides_11/osm_maps_arrow"
# 타일 제공자별로 캐시 디렉터리를 분리 (map_image_spec.md 5절: 다른 제공자 타일이
# 같은 캐시에 섞여서 실제로 겪었던 사고 사례 — z/x/y만으로는 출처가 구분 안 됨)
TILE_PROVIDER = "cartocdn_voyager_nolabels"
TILE_CACHE  = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tile_cache", TILE_PROVIDER)

# ── OSRM server mapping (region → port) ──────────────────────────────────────
def osrm_port(lat, lon):
    if lat < -30:                            return 5001  # Perth
    if 24 < lat < 26 and 121 < lon < 122:   return 5002  # Taipei
    if 35 < lat < 36 and 139 < lon < 140:   return 5003  # Tokyo
    if 30 < lat < 31 and 114 < lon < 115:   return 5004  # Wuhan
    if 14 < lat < 15 and 120 < lon < 122:   return 5005  # Manila
    if 41 < lat < 43 and 12 < lon < 13:     return 5006  # Rome
    if -42 < lat < -41 and 174 < lon < 175: return 5007  # Wellington
    if 27 < lat < 28 and -81 < lon < -80:   return 5008  # Florida-B
    if 50 < lat < 51 and -1 < lon < 0:      return 5009  # Brighton
    if 40 < lat < 41 and -4 < lon < -3:     return 5010  # Madrid
    if 37.4 < lat < 37.7 and 126.8 < lon < 127.2: return 5011  # Seoul (서울과학기술대 실배포 테스트)
    raise ValueError(f"No OSRM server for lat={lat}, lon={lon}")

# ── Config ───────────────────────────────────────────────────────────────────
# 2026-09 맵 스케일 확정 (배포 튜닝안 "B안"): 정중앙 대칭 크롭 대신, 로봇을
# 화면 아래쪽에 앵커시켜 후방보다 전방을 넓게 본다. MAP_RANGE_M은 이제
# "half-width"가 아니라 "전방 reach(m)"이고, 후방 reach = MAP_RANGE_M*REAR_RATIO로
# 고정 비율 파생된다 (기존 검증된 20m-forward 대칭크롭과 96px 모델입력 기준
# 유효 해상도를 비슷하게 유지하기 위해 7:20 비율로 확정 — 더 이상 실험적으로
# 흔들지 않음). 스케일(m/px)은 (전방+후방)/MAP_SIZE_PX로 유일하게 결정되고
# 좌우 폭도 여기서 자동으로 나온다 (out_size가 정사각형이므로 half-width는
# 항상 (전방+후방)/2와 같음) — 별도 파라미터 아님.
MAP_SIZE_PX  = 224
MAP_RANGE_M  = 20.0    # forward reach in meters (앵커 기준, half-width 아님)
REAR_RATIO   = 0.35    # rear_m = MAP_RANGE_M * REAR_RATIO (20m→7m, 확정값)
GOAL_DIST_M  = 20.0    # future horizon distance
ZOOM         = 19      # OSM tile zoom (map_image_spec.md 규격값)
TILE_PX      = 256
ROUTE_COLOR  = (0, 0, 255)    # BGR red
PAST_COLOR   = (160, 160, 160)
GOAL_COLOR   = (0, 165, 255)  # BGR orange — route 끝(진짜 goal) 표시, 이전엔 안 그려지던 버그
ROUTE_WIDTH  = 2  # 교수님 피드백: 경로선을 더 얇게 (기존 4 → 2)
EGO_COLOR    = (0, 200, 0)
FILL_COLOR   = (200, 200, 200)  # 지도 밖 채움색 (map_image_spec.md 규격값, 기존 흰색에서 변경)
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

# 2026-09-06: cartocdn returns a static "API KEY REQUIRED" placeholder tile (HTTP 200,
# not an error status) when the request is missing/over-quota on its API key — this silently
# looked like a successful fetch and got cached as if it were real map content. The
# placeholder is a fixed, coordinate-independent image, so its content hash is a reliable,
# zero-false-positive signature (unlike heuristics like "low color diversity", which also
# flags plenty of genuinely blank/rural real tiles — checked empirically against 127 cached
# tiles, ~16% false-positive rate, rejected for that reason).
_BAD_TILE_MD5 = {
    "6975bf716d5d075bdf895ed0c0be8e50",  # cartocdn rastertiles "API KEY REQUIRED" placeholder
}

def fetch_tile(tx, ty, zoom, session):
    os.makedirs(TILE_CACHE, exist_ok=True)
    cache = os.path.join(TILE_CACHE, f"{zoom}_{tx}_{ty}.png")
    if os.path.exists(cache):
        return Image.open(cache).convert("RGB")
    url = f"https://basemaps.cartocdn.com/rastertiles/voyager_nolabels/{zoom}/{tx}/{ty}.png"
    # cartocdn now requires a free API key (carto.com/basemaps/apikey) for anonymous requests.
    # Key read from env var, never hardcoded here, to avoid committing a secret.
    carto_key = os.environ.get("CARTO_API_KEY")
    if carto_key:
        url += f"?key={carto_key}"
    for attempt in range(3):
        try:
            r = session.get(url, headers={"User-Agent": USER_AGENT}, timeout=10)
            r.raise_for_status()
            if hashlib.md5(r.content).hexdigest() in _BAD_TILE_MD5:
                raise RuntimeError(
                    f"tile {zoom}/{tx}/{ty} returned known placeholder (API key missing/"
                    f"quota exceeded?) - not caching, will retry"
                )
            img = Image.open(BytesIO(r.content)).convert("RGB")
            img.save(cache)
            time.sleep(0.05)
            return img
        except Exception as e:
            tqdm.write(f"  [fetch_tile] {zoom}/{tx}/{ty} attempt {attempt+1}/3 failed: {e}")
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
    Draw past (gray)/future (red) route + goal marker on the tile canvas,
    then a SINGLE affine warp (rotate heading-up + rescale to a fixed m/px +
    place ego at its anchor pixel), then draw the ego marker post-warp so it
    stays a fixed pixel size regardless of latitude/scale.

    map_range_m = forward reach in meters (NOT half-width — see REAR_RATIO
    above). rear reach = map_range_m*REAR_RATIO, so ego sits below center;
    left-right half-width is whatever (forward+rear)/2 works out to, since a
    single isotropic scale applies to the square out_size frame.
    Returns (out_size, out_size, 3) uint8 RGB.
    """
    img = canvas_bgr.copy()

    rear_m = map_range_m * REAR_RATIO
    total_span_m = map_range_m + rear_m
    scale_m_per_px = total_span_m / out_size
    anchor_px_x = 0.5 * out_size
    anchor_px_y = (map_range_m / total_span_m) * out_size

    ego_gx, ego_gy = latlon_to_pixel_global(lat_curr, lon_curr, zoom)
    ego_cx, ego_cy = global_to_canvas(ego_gx, ego_gy, gx0, gy0)

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
        pts = [(ego_cx, ego_cy)]
        for lat, lon in zip(future_lats, future_lons):
            gx, gy = latlon_to_pixel_global(lat, lon, zoom)
            cx, cy = global_to_canvas(gx, gy, gx0, gy0)
            pts.append((cx, cy))
        for k in range(1, len(pts)):
            cv2.line(img, pts[k-1], pts[k], ROUTE_COLOR, ROUTE_WIDTH, cv2.LINE_AA)

        # ── Goal marker (route end) — drawn pre-warp so it rotates/scales with the map ──
        cv2.circle(img, pts[-1], 5, GOAL_COLOR, -1, cv2.LINE_AA)

    # ── Single affine warp: rotate heading-up + rescale + place ego at anchor ──
    # filtered_heading: East=0, North=+90°, South=-90° (standard math CCW)
    # OSM tiles: North-up (위=북쪽, 오른쪽=동쪽)
    # 로봇 진행 방향이 이미지 위를 향하려면: rot_deg = 90 - heading_deg (OpenCV CCW 기준)
    # 검증: heading=-90°(남쪽) → rot=180° CCW → 남쪽이 위
    mpp = meters_per_pixel(lat_curr, zoom)      # native tile resolution at this latitude
    k = mpp / scale_m_per_px                     # native-px → output-px multiplier
    heading_deg = math.degrees(heading_rad)
    rot_deg = 90.0 - heading_deg

    # getRotationMatrix2D(center=(0,0), ...) → pure scale+rotate about the origin,
    # so we build it about the ego point and translate the result onto the anchor.
    R = cv2.getRotationMatrix2D((0, 0), rot_deg, k)
    a, b = R[0, 0], R[0, 1]
    c, d = R[1, 0], R[1, 1]
    tx = anchor_px_x - (a * ego_cx + b * ego_cy)
    ty = anchor_px_y - (c * ego_cx + d * ego_cy)
    M = np.array([[a, b, tx], [c, d, ty]], dtype=np.float64)

    warped = cv2.warpAffine(img, M, (out_size, out_size),
                             flags=cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_CONSTANT,
                             borderValue=FILL_COLOR)

    # ── Ego marker, drawn AFTER warp (fixed pixel size, independent of scale) ──
    anchor_pt = (int(round(anchor_px_x)), int(round(anchor_px_y)))
    cv2.circle(warped, anchor_pt, 7, EGO_COLOR, -1)
    cv2.circle(warped, anchor_pt, 7, (255, 255, 255), 2)

    return cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)


# ── Per-episode processing ────────────────────────────────────────────────────

def save_gps_csv(out_dir, lats, lons):
    gps_path = os.path.join(out_dir, "gps.csv")
    with open(gps_path, 'w') as f:
        f.write("frame_index,latitude,longitude\n")
        for i, (lat, lon) in enumerate(zip(lats, lons)):
            f.write(f"{i},{lat:.8f},{lon:.8f}\n")


def _bev_zoom_for_distance(total_m, base_zoom):
    """
    EKF 총 이동거리 기반 BEV zoom 레벨 결정.
      < 30m  → base_zoom     (가장 상세)
      30~100m → base_zoom - 1
      > 100m  → base_zoom - 2
    최소 zoom=15 보장.
    """
    if total_m < 30:
        return base_zoom
    elif total_m < 100:
        return max(base_zoom - 1, 15)
    else:
        return max(base_zoom - 2, 15)


def save_bev(canvas_bgr, gx0, gy0, zoom, lats, lons, out_dir,
             fp=None, start_heading_rad=None, pad_px=80):
    """
    BEV 조감도 저장.
    - 시작점(GPS[0]) 중심으로 crop
    - EKF 총 이동거리 기반 zoom 자동 조정 (fp 제공 시)
    - start_heading_rad 제공 시 odometry 지도와 동일한 heading-up 회전 적용
      (rot_deg = 270 - heading_deg, 진행 방향이 위를 향함)
    - 파란 경로선, 초록 시작점, 빨간 끝점
    """
    # zoom 결정
    if fp is not None:
        fp_arr = np.array(fp)
        total_m = float(np.sum(np.linalg.norm(np.diff(fp_arr, axis=0), axis=1)))
        bev_zoom = _bev_zoom_for_distance(total_m, zoom)
    else:
        total_m = None
        bev_zoom = max(zoom - 1, 15)

    session = requests.Session()
    bev_canvas, bev_gx0, bev_gy0 = build_canvas(lats, lons, bev_zoom, session)
    img = bev_canvas.copy()

    # GPS 경로 픽셀 좌표 계산
    pts = []
    for lat, lon in zip(lats, lons):
        gx, gy = latlon_to_pixel_global(lat, lon, bev_zoom)
        cx, cy = global_to_canvas(gx, gy, bev_gx0, bev_gy0)
        pts.append((cx, cy))

    for k in range(1, len(pts)):
        cv2.line(img, pts[k-1], pts[k], (204, 102, 0), 4, cv2.LINE_AA)

    # 시작점(초록)·끝점(빨간) 마커
    cv2.circle(img, pts[0],  10, (0, 200, 0),  -1)
    cv2.circle(img, pts[0],  10, (255, 255, 255), 2)
    cv2.circle(img, pts[-1], 10, (0,  50, 230), -1)
    cv2.circle(img, pts[-1], 10, (255, 255, 255), 2)

    # 시작점 중심으로 crop (회전 전)
    mpp = meters_per_pixel(lats[0], bev_zoom)
    if total_m is not None:
        radius_px = max(int(total_m * 1.5 / mpp), pad_px * 3)
    else:
        radius_px = pad_px * 3

    cx0, cy0 = pts[0]
    h_img, w_img = img.shape[:2]
    x1 = max(0, cx0 - radius_px)
    y1 = max(0, cy0 - radius_px)
    x2 = min(w_img, cx0 + radius_px)
    y2 = min(h_img, cy0 + radius_px)
    crop = img[y1:y2, x1:x2]

    # odometry 지도와 동일한 heading-up 회전 적용
    if start_heading_rad is not None:
        heading_deg = math.degrees(start_heading_rad)
        rot_deg = 270.0 - heading_deg  # odometry 지도와 동일한 공식
        h, w = crop.shape[:2]
        # 시작점의 crop 내 상대 좌표
        rel_cx = min(cx0 - x1, w - 1)
        rel_cy = min(cy0 - y1, h - 1)
        # 회전 후 이미지 크기 계산 (코너 잘림 방지)
        angle_r = math.radians(rot_deg)
        cos_a = abs(math.cos(angle_r))
        sin_a = abs(math.sin(angle_r))
        new_w = int(h * sin_a + w * cos_a) + 1
        new_h = int(h * cos_a + w * sin_a) + 1
        M = cv2.getRotationMatrix2D((rel_cx, rel_cy), rot_deg, 1.0)
        M[0, 2] += new_w / 2 - rel_cx
        M[1, 2] += new_h / 2 - rel_cy
        crop = cv2.warpAffine(crop, M, (new_w, new_h),
                              flags=cv2.INTER_LINEAR,
                              borderValue=(255, 255, 255))

    out_path = os.path.join(out_dir, "bev_overview.png")
    Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)).save(out_path)
    return out_path, bev_zoom, total_m


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
                    goal_dist_m=GOAL_DIST_M, zoom=ZOOM, out_size=MAP_SIZE_PX,
                    map_range_m=MAP_RANGE_M):
    os.makedirs(out_dir, exist_ok=True)

    lats = np.array(ep_data['lats'])
    lons = np.array(ep_data['lons'])
    fp   = np.array(ep_data['filtered_pos'])
    fh   = np.array(ep_data['filtered_heading'])

    save_gps_csv(out_dir, lats, lons)

    session = requests.Session()

    # OSM 타일 캔버스 1회 빌드
    canvas_bgr, gx0, gy0 = build_canvas(lats, lons, zoom, session)

    # BEV 조감도 저장 (시작점 중심, 거리 기반 zoom, north-up 고정)
    save_bev(canvas_bgr, gx0, gy0, zoom, lats, lons, out_dir, fp=fp)

    # ── 고정 경로: 실제 GPS 궤적 사용 (OSRM 제거) ───────────────────────────
    # GPS 해상도가 낮아 중복 좌표 제거 후 densify
    gps_raw = np.column_stack([lats, lons])
    # 연속 중복 제거 (GPS 해상도 한계로 같은 좌표가 여러 프레임 연속 등장)
    mask_unique = np.concatenate([[True], np.any(np.diff(gps_raw, axis=0) != 0, axis=1)])
    gps_unique = gps_raw[mask_unique]
    route_latlon = _densify_route(gps_unique, step_m=1.0)
    # route_latlon: shape (M, 2), [[lat, lon], ...], 1m 간격 densified

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
            map_range_m=map_range_m,
        )

        Image.fromarray(img).save(out_path)
        saved += 1

    return saved


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    out_root = args.out_root or OUT_ROOT
    os.makedirs(out_root, exist_ok=True)

    with open(args.scores_path, encoding="utf-8") as f:
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
    table = pa.ipc.open_stream(open(args.arrow_path, 'rb')).read_all()
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

            out_dir = os.path.join(out_root, f"episode_{ep:04d}_seg{seg_idx:02d}")
            n = process_episode(ep, seg_data, out_dir,
                                goal_dist_m=args.goal_dist,
                                zoom=args.zoom,
                                out_size=args.out_size,
                                map_range_m=args.map_range)
            tqdm.write(f"  ep{ep:03d}[{seg_idx}]: {n} maps -> {out_dir}")

    print("\nDone. (c) OpenStreetMap contributors")


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
    parser.add_argument("--out_root",     type=str,   default=None,
                        help="Override output root dir (default: OUT_ROOT constant)")
    parser.add_argument("--arrow_path",   type=str,   default=ARROW_PATH,
                        help="Override .arrow dataset path (default: ARROW_PATH constant, "
                             "which is hardcoded to the original author's machine)")
    parser.add_argument("--scores_path",  type=str,   default=SCORES_PATH,
                        help="Override episode_scores.json path (default: SCORES_PATH constant)")
    args = parser.parse_args()
    main(args)
