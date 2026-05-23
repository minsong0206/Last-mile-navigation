"""
고정 경로 방식 검증: ep9 seg0의 처음 5프레임 샘플 이미지 생성
"""
import os, sys, json
import numpy as np
import pyarrow as pa
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from episode_selector import split_into_segments
from osm_map_generator import (
    build_canvas, save_gps_csv, osrm_route_full,
    map_frames_to_route, render_frame,
    latlon_to_pixel_global, global_to_canvas,
    osrm_port, ZOOM, MAP_SIZE_PX,
)
import requests, cv2

ARROW_PATH = "/media/ms/WD_BLACK_4TB/Learning-to-Drive-Anywhere-with-MBRA/FrodoBots-2K/processed/frodobots_dataset/train/data-00000-of-00001.arrow"
OUT_DIR    = "/media/ms/WD_BLACK_4TB/Learning-to-Drive-Anywhere-with-MBRA/osm_pipeline/test_fixed_route"
EP, SEG    = 9, 0
N_SAMPLE   = 5  # 처음 N프레임만

os.makedirs(OUT_DIR, exist_ok=True)

print("Arrow 로딩...")
table = pa.ipc.open_stream(open(ARROW_PATH, 'rb')).read_all()
ep_idx_arr = np.array(table['episode_index'].to_pylist())
lats_arr   = np.array(table['observation.latitude'].to_pylist())
lons_arr   = np.array(table['observation.longitude'].to_pylist())
fp_arr     = np.array(table['observation.filtered_position'].to_pylist())
fh_arr     = np.array(table['observation.filtered_heading'].to_pylist())

mask    = ep_idx_arr == EP
ep_fp   = fp_arr[mask]
ep_lats = lats_arr[mask]
ep_lons = lons_arr[mask]
ep_fh   = fh_arr[mask]

segments = split_into_segments(ep_fp, ep_lats, ep_lons)
seg  = segments[SEG]
idxs = seg['frame_indices']

lats = ep_lats[idxs]
lons = ep_lons[idxs]
fp   = ep_fp[idxs]
fh   = ep_fh[idxs]

print(f"ep{EP} seg{SEG}: {len(lats)} frames")

port = osrm_port(lats.mean(), lons.mean())
session = requests.Session()

print("OSM 타일 캔버스 빌드 중...")
canvas_bgr, gx0, gy0 = build_canvas(lats, lons, ZOOM, session)

print(f"OSRM 경로 1회 요청 (port {port})...")
route_latlon = osrm_route_full(lats, lons, port)
print(f"  → 경로 포인트 수: {len(route_latlon)}")

print("EKF 진행률 기반 frame→route 매핑 계산...")
frame_route_idx = map_frames_to_route(fp, route_latlon, lats, lons)
print(f"  frame_route_idx[0:10]: {frame_route_idx[:10]}")

print(f"샘플 {N_SAMPLE}장 생성 중...")
# 5프레임 간격으로 샘플링하여 변화 확인
sample_frames = list(range(0, min(N_SAMPLE * 50, len(lats)), 50))[:N_SAMPLE]
for i in sample_frames:
    closest_idx = frame_route_idx[i]

    past_route   = route_latlon[:closest_idx + 1]
    future_route = route_latlon[closest_idx + 1:]

    past_lats_r   = past_route[:, 0]   if len(past_route)   >= 2 else np.array([lats[i]])
    past_lons_r   = past_route[:, 1]   if len(past_route)   >= 2 else np.array([lons[i]])
    future_lats_r = future_route[:, 0] if len(future_route) >= 1 else np.array([])
    future_lons_r = future_route[:, 1] if len(future_route) >= 1 else np.array([])

    img = render_frame(
        canvas_bgr, gx0, gy0, ZOOM,
        lats[i], lons[i], fh[i],
        future_lats_r, future_lons_r,
        past_lats_r,   past_lons_r,
        out_size=MAP_SIZE_PX,
    )

    out_path = os.path.join(OUT_DIR, f"frame_{i:04d}.png")
    Image.fromarray(img).save(out_path)
    print(f"  [frame {i:4d}] closest_idx={closest_idx:3d}/{len(route_latlon)}, past={len(past_route):3d}, future={len(future_route):3d} → {out_path}")

print(f"\n완료. 결과: {OUT_DIR}")
