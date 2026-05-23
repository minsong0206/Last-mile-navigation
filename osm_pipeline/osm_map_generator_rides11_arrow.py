"""
osm_map_generator_rides11_arrow.py

Arrow(Zarr) 기준으로 rides_11 OSM 맵 재생성.

기존 osm_map_generator_rides11.py와의 차이:
  - 입력: 원본 GPS CSV → Arrow Zarr (카메라 10Hz 보간 완료)
  - 헤딩: estimate_headings() 추정 → observation.filtered_heading 직접 사용
  - 루프: ride 폴더 → episode
  - 파일명: osm_map_{원본GPS인덱스}.png → osm_map_{Arrow_frame_index:06d}.png
  - 출력폴더: ride_XXXXX/ → episode_{ep_idx:06d}/

출력 구조:
  <OUT_ROOT>/
    episode_000000/
      osm_map_000000.png   ← episode 0, frame_index 0
      osm_map_000001.png
      ...
      bev_overview.png
      gps.csv              ← 해당 episode의 lat/lon (Arrow 기준)
    episode_000001/
      ...

실행:
  python osm_map_generator_rides11_arrow.py \
      --zarr_path /path/to/dataset_cache.zarr \
      --out_root  /path/to/osm_maps_arrow \
      [--episode 42]        # 특정 episode만 처리 (디버그용)
      [--zoom 18]
      [--out_size 224]
      [--workers 4]         # 병렬 episode 처리 수
"""

import os, sys, math, argparse
import numpy as np
import zarr
import requests
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'py')))
from osm_map_generator import (
    build_canvas, render_frame, save_bev,
    save_gps_csv, _densify_route,
    MAP_SIZE_PX, ZOOM,
)

# ── 경로 설정 ──────────────────────────────────────────────────────
ZARR_PATH = "/media/ms/WD_BLACK_4TB/Learning-to-Drive-Anywhere-with-MBRA/FrodoBots-2K/processed/frodobots_dataset/dataset_cache.zarr"
OUT_ROOT  = "/media/ms/WD_BLACK_4TB/Learning-to-Drive-Anywhere-with-MBRA/osm_pipeline/output_rides_11/osm_maps_arrow"


# ── Episode 데이터 추출 ────────────────────────────────────────────
def load_episode(zarr_root, ep_idx: int) -> dict:
    """
    Zarr에서 특정 episode의 데이터를 추출.
    반환: {lats, lons, headings, frame_indices} numpy array
    """
    ep_mask = zarr_root["episode_index"][:] == ep_idx

    return {
        "lats":          zarr_root["observation.latitude"][ep_mask],
        "lons":          zarr_root["observation.longitude"][ep_mask],
        "headings":      zarr_root["observation.filtered_heading"][ep_mask],
        "frame_indices": zarr_root["frame_index"][ep_mask],
    }


def map_frames_to_route(lats, lons, route_latlon):
    """각 프레임을 densify된 route 상의 인덱스에 매핑 (단조 증가 보장)."""
    LAT_M = 111320.0

    gps_dists = [0.0]
    for k in range(1, len(lats)):
        dlat = (lats[k] - lats[k-1]) * LAT_M
        dlon = (lons[k] - lons[k-1]) * LAT_M * math.cos(math.radians(lats[k-1]))
        gps_dists.append(gps_dists[-1] + math.sqrt(dlat**2 + dlon**2))
    gps_dists  = np.array(gps_dists)
    gps_total  = gps_dists[-1] if gps_dists[-1] > 0 else 1.0
    gps_frac   = gps_dists / gps_total

    route      = np.array(route_latlon)
    route_segs = []
    for k in range(1, len(route)):
        dlat = (route[k,0] - route[k-1,0]) * LAT_M
        dlon = (route[k,1] - route[k-1,1]) * LAT_M * math.cos(math.radians(route[k-1,0]))
        route_segs.append(math.sqrt(dlat**2 + dlon**2))
    route_cum   = np.concatenate([[0], np.cumsum(route_segs)])
    route_total = route_cum[-1] if route_cum[-1] > 0 else 1.0
    route_frac  = route_cum / route_total

    result = np.searchsorted(route_frac, gps_frac, side='left')
    result = np.clip(result, 0, len(route) - 1)
    for i in range(1, len(result)):
        if result[i] < result[i-1]:
            result[i] = result[i-1]
    return result


# ── Episode 처리 ───────────────────────────────────────────────────
def process_episode(ep_idx: int, ep_data: dict, out_dir: str,
                    zoom: int = ZOOM, out_size: int = MAP_SIZE_PX) -> int:
    """
    Arrow 기준 episode 하나의 OSM 맵을 생성.
    파일명: osm_map_{frame_index:06d}.png  (Arrow frame_index 기준)
    """
    os.makedirs(out_dir, exist_ok=True)

    lats          = ep_data["lats"]
    lons          = ep_data["lons"]
    headings      = ep_data["headings"]   # EKF filtered, 추정 불필요
    frame_indices = ep_data["frame_indices"]

    if len(lats) < 3:
        return 0

    if not (-90 <= lats.mean() <= 90 and -180 <= lons.mean() <= 180):
        return 0

    # GPS CSV 저장 (Arrow 기준)
    save_gps_csv(out_dir, lats, lons)

    session = requests.Session()

    # OSM 타일 캔버스 빌드 (episode 전체 범위 커버)
    canvas_bgr, gx0, gy0 = build_canvas(lats, lons, zoom, session)

    # BEV 조감도 저장
    LAT_M = 111320.0
    total_m = sum(
        math.sqrt(
            ((lats[k]-lats[k-1]) * LAT_M)**2 +
            ((lons[k]-lons[k-1]) * LAT_M * math.cos(math.radians(lats[k-1])))**2
        )
        for k in range(1, len(lats))
    )
    dummy_fp = np.array([[0.0, 0.0], [total_m, 0.0]])
    save_bev(canvas_bgr, gx0, gy0, zoom, lats, lons, out_dir, fp=dummy_fp)

    # Densify route (과거/미래 경로 분리용)
    gps_raw    = np.column_stack([lats, lons])
    mask_uniq  = np.concatenate([[True], np.any(np.diff(gps_raw, axis=0) != 0, axis=1)])
    gps_unique = gps_raw[mask_uniq]
    route_latlon     = _densify_route(gps_unique, step_m=1.0)
    frame_route_idx  = map_frames_to_route(lats, lons, route_latlon)

    saved = 0
    for i, frame_idx in enumerate(frame_indices):
        # ★ 파일명 = Arrow frame_index (episode마다 0부터 시작)
        out_path = os.path.join(out_dir, f"osm_map_{frame_idx:06d}.png")
        if os.path.exists(out_path):
            saved += 1
            continue

        closest_idx  = frame_route_idx[i]
        past_route   = route_latlon[:closest_idx + 1]
        future_route = route_latlon[closest_idx + 1:]

        past_lats_r   = past_route[:, 0]   if len(past_route)   >= 2 else np.array([lats[i]])
        past_lons_r   = past_route[:, 1]   if len(past_route)   >= 2 else np.array([lons[i]])
        future_lats_r = future_route[:, 0] if len(future_route) >= 1 else np.array([])
        future_lons_r = future_route[:, 1] if len(future_route) >= 1 else np.array([])

        img = render_frame(
            canvas_bgr, gx0, gy0, zoom,
            lats[i], lons[i],
            headings[i],          # ★ EKF filtered heading 직접 사용
            future_lats_r, future_lons_r,
            past_lats_r,   past_lons_r,
            out_size=out_size,
        )

        Image.fromarray(img).save(out_path)
        saved += 1

    return saved


# ── Worker (멀티프로세싱용) ────────────────────────────────────────
def _worker(args):
    ep_idx, zarr_path, out_root, zoom, out_size = args
    try:
        zroot   = zarr.open(zarr_path, mode='r')
        ep_data = load_episode(zroot, ep_idx)
        out_dir = os.path.join(out_root, f"episode_{ep_idx:06d}")
        n       = process_episode(ep_idx, ep_data, out_dir, zoom, out_size)
        return ep_idx, n, None
    except Exception as e:
        import traceback
        return ep_idx, 0, traceback.format_exc()


# ── Main ───────────────────────────────────────────────────────────
def main(args):
    os.makedirs(args.out_root, exist_ok=True)

    print(f"Loading Zarr: {args.zarr_path}")
    zroot = zarr.open(args.zarr_path, mode='r')

    all_ep = np.unique(zroot["episode_index"][:])
    print(f"Total episodes: {len(all_ep)}")

    # 특정 episode만 처리 (디버그용)
    if args.episode is not None:
        all_ep = [args.episode]
        print(f"Processing single episode: {args.episode}")

    total_maps = 0

    if args.workers == 1:
        # 단일 프로세스
        for ep_idx in tqdm(all_ep, desc="Episodes"):
            ep_data = load_episode(zroot, ep_idx)
            out_dir = os.path.join(args.out_root, f"episode_{ep_idx:06d}")
            n = process_episode(ep_idx, ep_data, out_dir, args.zoom, args.out_size)
            total_maps += n
            tqdm.write(f"  ep{ep_idx:04d}: {n} maps → {out_dir}")
    else:
        # 멀티프로세스
        worker_args = [
            (int(ep_idx), args.zarr_path, args.out_root, args.zoom, args.out_size)
            for ep_idx in all_ep
        ]
        with ProcessPoolExecutor(max_workers=args.workers) as exe:
            futures = {exe.submit(_worker, a): a[0] for a in worker_args}
            with tqdm(total=len(futures), desc="Episodes") as pbar:
                for fut in as_completed(futures):
                    ep_idx, n, err = fut.result()
                    if err:
                        tqdm.write(f"  ep{ep_idx:04d} ERROR: {err[:200]}")
                    else:
                        tqdm.write(f"  ep{ep_idx:04d}: {n} maps")
                        total_maps += n
                    pbar.update(1)

    print(f"\nDone. Total maps: {total_maps:,}  © OpenStreetMap contributors")
    print(f"Output: {args.out_root}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--zarr_path", type=str, default=ZARR_PATH)
    parser.add_argument("--out_root",  type=str, default=OUT_ROOT)
    parser.add_argument("--episode",   type=int, default=None,
                        help="특정 episode만 처리 (디버그용)")
    parser.add_argument("--zoom",      type=int, default=ZOOM)
    parser.add_argument("--out_size",  type=int, default=MAP_SIZE_PX)
    parser.add_argument("--workers",   type=int, default=4,
                        help="병렬 처리 프로세스 수 (OSM 타일 요청 부하 고려, 4 권장)")
    args = parser.parse_args()
    main(args)
