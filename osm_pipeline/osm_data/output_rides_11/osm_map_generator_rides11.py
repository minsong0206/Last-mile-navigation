"""
output_rides_11 전용 OSM 맵 생성기

ride0(Arrow 파이프라인)과 동일한 224x224 ego-heading-up 이미지 생성.
GPS 1Hz 기준으로 포인트당 1장 생성.
헤딩: filtered_heading 없음 → GPS 궤적 기반 진행 방향 추정.

출력: osm_pipeline/output_rides_11/osm_maps/ride_XXXXX_*/
  osm_map_NNNNNN.png  (224x224, ego-heading-up)
  bev_overview.png
  gps.csv
"""

import os, sys, glob, math, time, argparse
import numpy as np
import cv2
import requests
from PIL import Image
from io import BytesIO
from tqdm import tqdm
from collections import defaultdict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'py')))
from osm_map_generator import (
    build_canvas, render_frame, save_bev,
    save_gps_csv, _densify_route,
    latlon_to_pixel_global, global_to_canvas,
    MAP_SIZE_PX, MAP_RANGE_M, ZOOM, TILE_CACHE,
)

RIDES_ROOT = "/media/ms/WD_BLACK_4TB/Learning-to-Drive-Anywhere-with-MBRA/FrodoBots-2K/extracted/output_rides_11"
OUT_ROOT   = "/media/ms/WD_BLACK_4TB/Learning-to-Drive-Anywhere-with-MBRA/osm_pipeline/output_rides_11/osm_maps"

# 선별된 지역 중심 좌표 (0.01도 격자)
KEEP_CELLS = {
    (30.48, 114.30),   # 우한, 중국
    (14.70, 120.99),   # 마닐라-A, 필리핀
    (14.35, 121.05),   # 마닐라-B, 필리핀
    (41.98,  12.52),   # 로마, 이탈리아
    (-41.29, 174.77),  # 웰링턴, 뉴질랜드
    (24.96, 121.31),   # 타이베이, 대만
    (27.45, -80.33),   # 플로리다-B, 미국
    (50.83,  -0.12),   # 브라이턴-A, 영국
    (40.39,  -3.75),   # 마드리드-B, 스페인
    (40.34,  -3.76),   # 마드리드-C, 스페인
}


def load_gps(gps_path):
    lats, lons, timestamps = [], [], []
    with open(gps_path) as f:
        next(f)
        for line in f:
            p = line.strip().split(',')
            if len(p) >= 3:
                try:
                    lats.append(float(p[0]))
                    lons.append(float(p[1]))
                    timestamps.append(int(p[2]))
                except ValueError:
                    continue
    return np.array(lats), np.array(lons), np.array(timestamps)


def estimate_headings(lats, lons):
    """GPS 궤적 기반 헤딩 추정 (East=0, North=+π/2, rad)."""
    headings = np.zeros(len(lats))
    LAT_M = 111320.0
    for i in range(len(lats) - 1):
        dlat = (lats[i+1] - lats[i]) * LAT_M
        dlon = (lons[i+1] - lons[i]) * LAT_M * math.cos(math.radians(lats[i]))
        if abs(dlat) > 1e-6 or abs(dlon) > 1e-6:
            headings[i] = math.atan2(dlat, dlon)
        else:
            headings[i] = headings[i-1] if i > 0 else 0.0
    headings[-1] = headings[-2] if len(headings) > 1 else 0.0
    return headings


def map_frames_to_route(lats, lons, route_latlon):
    """각 GPS 포인트를 route 상의 인덱스에 매핑 (누적 거리 기반, 단조 증가)."""
    LAT_M = 111320.0
    # GPS 누적 거리
    gps_dists = [0.0]
    for k in range(1, len(lats)):
        dlat = (lats[k] - lats[k-1]) * LAT_M
        dlon = (lons[k] - lons[k-1]) * LAT_M * math.cos(math.radians(lats[k-1]))
        gps_dists.append(gps_dists[-1] + math.sqrt(dlat**2 + dlon**2))
    gps_dists = np.array(gps_dists)
    gps_total = gps_dists[-1] if gps_dists[-1] > 0 else 1.0
    gps_frac = gps_dists / gps_total

    # route 누적 거리
    route = np.array(route_latlon)
    route_segs = []
    for k in range(1, len(route)):
        dlat = (route[k,0] - route[k-1,0]) * LAT_M
        dlon = (route[k,1] - route[k-1,1]) * LAT_M * math.cos(math.radians(route[k-1,0]))
        route_segs.append(math.sqrt(dlat**2 + dlon**2))
    route_cum = np.concatenate([[0], np.cumsum(route_segs)])
    route_total = route_cum[-1] if route_cum[-1] > 0 else 1.0
    route_frac = route_cum / route_total

    result = np.searchsorted(route_frac, gps_frac, side='left')
    result = np.clip(result, 0, len(route) - 1)
    for i in range(1, len(result)):
        if result[i] < result[i-1]:
            result[i] = result[i-1]
    return result


def process_ride(ride_dir, out_dir, zoom=ZOOM, out_size=MAP_SIZE_PX):
    os.makedirs(out_dir, exist_ok=True)

    # GPS 로드
    gps_files = glob.glob(os.path.join(ride_dir, 'gps_data_*.csv'))
    if not gps_files:
        return 0
    lats, lons, timestamps = load_gps(gps_files[0])
    if len(lats) < 3:
        return 0

    # GPS 유효성 검사
    if not (-90 <= lats.mean() <= 90 and -180 <= lons.mean() <= 180):
        return 0

    save_gps_csv(out_dir, lats, lons)

    session = requests.Session()

    # OSM 타일 캔버스 1회 빌드
    canvas_bgr, gx0, gy0 = build_canvas(lats, lons, zoom, session)

    # BEV 조감도 저장 (EKF 없으므로 fp=None → zoom-1 고정)
    # GPS 기반 총 이동거리 계산
    LAT_M = 111320.0
    total_m = 0.0
    for k in range(1, len(lats)):
        dlat = (lats[k] - lats[k-1]) * LAT_M
        dlon = (lons[k] - lons[k-1]) * LAT_M * math.cos(math.radians(lats[k-1]))
        total_m += math.sqrt(dlat**2 + dlon**2)

    # fp 대신 GPS 거리로 BEV zoom 결정하기 위해 dummy fp 배열 생성
    dummy_fp = np.array([[0.0, 0.0], [total_m, 0.0]])
    save_bev(canvas_bgr, gx0, gy0, zoom, lats, lons, out_dir, fp=dummy_fp)

    # 고정 경로: GPS 궤적 densify
    gps_raw = np.column_stack([lats, lons])
    mask_unique = np.concatenate([[True], np.any(np.diff(gps_raw, axis=0) != 0, axis=1)])
    gps_unique = gps_raw[mask_unique]
    route_latlon = _densify_route(gps_unique, step_m=1.0)

    # 각 GPS 포인트 → route index 매핑
    frame_route_idx = map_frames_to_route(lats, lons, route_latlon)

    # 헤딩 추정
    headings = estimate_headings(lats, lons)

    saved = 0
    for i in range(len(lats)):
        out_path = os.path.join(out_dir, f"osm_map_{i:06d}.png")
        if os.path.exists(out_path):
            saved += 1
            continue

        closest_idx = frame_route_idx[i]
        past_route   = route_latlon[:closest_idx + 1]
        future_route = route_latlon[closest_idx + 1:]

        past_lats_r   = past_route[:, 0]   if len(past_route)   >= 2 else np.array([lats[i]])
        past_lons_r   = past_route[:, 1]   if len(past_route)   >= 2 else np.array([lons[i]])
        future_lats_r = future_route[:, 0] if len(future_route) >= 1 else np.array([])
        future_lons_r = future_route[:, 1] if len(future_route) >= 1 else np.array([])

        img = render_frame(
            canvas_bgr, gx0, gy0, zoom,
            lats[i], lons[i], headings[i],
            future_lats_r, future_lons_r,
            past_lats_r,   past_lons_r,
            out_size=out_size,
        )

        Image.fromarray(img).save(out_path)
        saved += 1

    return saved


def get_cell(lats, lons):
    return (round(float(np.mean(lats)), 2), round(float(np.mean(lons)), 2))


def main(args):
    os.makedirs(OUT_ROOT, exist_ok=True)

    # 전체 라이드 목록 로드
    all_rides = sorted([
        r for r in os.listdir(RIDES_ROOT)
        if os.path.isdir(os.path.join(RIDES_ROOT, r)) and r.startswith('ride_')
    ])

    # 선별 지역에 속하는 라이드만 필터링
    selected_rides = []
    for ride in all_rides:
        gf = glob.glob(os.path.join(RIDES_ROOT, ride, 'gps_data_*.csv'))
        if not gf:
            continue
        try:
            ls, lo = [], []
            with open(gf[0]) as f:
                next(f)
                for line in f:
                    p = line.strip().split(',')
                    if len(p) >= 2:
                        ls.append(float(p[0])); lo.append(float(p[1]))
            if ls:
                cell = (round(float(np.mean(ls)), 2), round(float(np.mean(lo)), 2))
                if cell in KEEP_CELLS:
                    selected_rides.append(ride)
        except Exception:
            continue

    if args.ride:
        selected_rides = [r for r in selected_rides if args.ride in r]

    print(f"Processing {len(selected_rides)} rides → {OUT_ROOT}")

    total_maps = 0
    for ride in tqdm(selected_rides, desc="Rides"):
        ride_dir = os.path.join(RIDES_ROOT, ride)
        out_dir  = os.path.join(OUT_ROOT, ride)
        n = process_ride(ride_dir, out_dir, zoom=args.zoom, out_size=args.out_size)
        total_maps += n
        tqdm.write(f"  {ride}: {n} maps")

    print(f"\nDone. Total maps: {total_maps:,}  © OpenStreetMap contributors")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--zoom",     type=int, default=ZOOM)
    parser.add_argument("--out_size", type=int, default=MAP_SIZE_PX)
    parser.add_argument("--ride",     type=str, default=None, help="특정 ride 이름(부분 매칭)")
    args = parser.parse_args()
    main(args)
