"""
Step 3: Dataset Builder

선별된 세그먼트의 각 프레임을 학습 샘플로 패키징:
  current_image.jpg    — 현재 프레임 전방 카메라
  past_image_1.jpg     — 1 스텝 이전 전방
  past_image_2.jpg     — 2 스텝 이전 전방
  rear_image.jpg       — 현재 프레임 후방 카메라
  osm_map.png          — 해당 프레임의 ego-centric OSM 지도 (Step 2 결과)
  metadata.json        — GPS, timestamp, episode/segment/frame 정보

프레임 매핑:
  Arrow: observation.images.front → {path: "videos/...", timestamp: float}
  video_timestamp × 10 + 1 = frames 디렉토리의 1-based 파일 번호

사용법:
  python dataset_builder.py --ep 6              # ep6 전체 세그먼트
  python dataset_builder.py --ep 6 --seg 1      # ep6 seg1만
  python dataset_builder.py                     # 선별된 전체 세그먼트

출력:
  osm_pipeline/dataset/episode_{ep:04d}_seg{seg:02d}/sample_{frame:06d}/
"""

import os
import sys
import json
import shutil
import argparse
import numpy as np
import pyarrow as pa
import cv2
from tqdm import tqdm
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))
from episode_selector import split_into_segments

# ── 경로 설정 ─────────────────────────────────────────────────────────────────
ARROW_PATH    = "/media/ms/WD_BLACK_4TB/Learning-to-Drive-Anywhere-with-MBRA/FrodoBots-2K/processed/frodobots_dataset/train/data-00000-of-00001.arrow"
FRAMES_ROOT   = "/media/ms/WD_BLACK_4TB/Learning-to-Drive-Anywhere-with-MBRA/FrodoBots-2K/processed/frodobots_dataset/frames"
VIDEOS_ROOT   = "/media/ms/WD_BLACK_4TB/Learning-to-Drive-Anywhere-with-MBRA/FrodoBots-2K/processed/frodobots_dataset/videos"
OSM_MAPS_ROOT = "/media/ms/WD_BLACK_4TB/Learning-to-Drive-Anywhere-with-MBRA/osm_pipeline/osm_maps"
SCORES_PATH   = "/media/ms/WD_BLACK_4TB/Learning-to-Drive-Anywhere-with-MBRA/osm_pipeline/episode_scores.json"
OUT_ROOT      = "/media/ms/WD_BLACK_4TB/Learning-to-Drive-Anywhere-with-MBRA/osm_pipeline/dataset"

N_PAST = 2  # 과거 컨텍스트 이미지 수


# ── 헬퍼 함수 ─────────────────────────────────────────────────────────────────

def get_ride_name(video_path):
    """'videos/ride_XXXXX_YYYYYYY_front_camera.mp4' → 'ride_XXXXX_YYYYYYY'"""
    basename = os.path.basename(video_path)
    return basename.replace("_front_camera.mp4", "").replace("_rear_camera.mp4", "")


def front_frame_path(ride_name, video_ts):
    """전방: frames 디렉토리의 pre-extracted jpg (10fps, 1-based)"""
    frame_num = round(video_ts * 10) + 1
    return os.path.join(FRAMES_ROOT, ride_name, f"{frame_num:06d}.jpg")


def extract_rear_frame(cap, video_ts, dst_path):
    """
    후방: rear_camera.mp4에서 video_ts(초) 위치의 프레임을 추출하여 저장.
    cap은 호출자가 열고 닫는 cv2.VideoCapture 객체.
    """
    cap.set(cv2.CAP_PROP_POS_MSEC, video_ts * 1000)
    ret, frame = cap.read()
    if ret:
        cv2.imwrite(dst_path, frame)
        return True
    return False


# ── 후방 VideoCapture 캐시 (세그먼트 내 재사용) ───────────────────────────────

class RearCapCache:
    """ride별 VideoCapture를 캐싱하여 반복 open/close 방지."""
    def __init__(self):
        self._cache = {}

    def get(self, ride_name):
        if ride_name not in self._cache:
            mp4 = os.path.join(VIDEOS_ROOT, f"{ride_name}_rear_camera.mp4")
            if not os.path.exists(mp4):
                return None
            self._cache[ride_name] = cv2.VideoCapture(mp4)
        return self._cache[ride_name]

    def release_all(self):
        for cap in self._cache.values():
            cap.release()
        self._cache.clear()


# ── 세그먼트 데이터 빌드 ─────────────────────────────────────────────────────

def build_segment(ep, seg_idx, seg, ep_arrow_data, out_root):
    """
    하나의 세그먼트에 대해 학습 샘플 디렉토리를 생성.
    전방: frames/ pre-extracted jpg 복사
    후방: rear_camera.mp4에서 video_ts로 직접 추출
    """
    frame_indices = seg['frame_indices']
    lats  = seg['lats']
    lons  = seg['lons']

    img_col      = ep_arrow_data['img_col']
    img_rear_col = ep_arrow_data['img_rear_col']
    ts_col       = ep_arrow_data['ts_col']

    seg_dir = os.path.join(out_root, f"episode_{ep:04d}_seg{seg_idx:02d}")
    osm_dir = os.path.join(OSM_MAPS_ROOT, f"episode_{ep:04d}_seg{seg_idx:02d}")
    os.makedirs(seg_dir, exist_ok=True)

    rear_cache = RearCapCache()
    saved = 0
    skipped = 0

    for local_i, ep_frame_i in enumerate(tqdm(frame_indices,
                                               desc=f"ep{ep:03d}[{seg_idx}]",
                                               leave=False)):
        sample_dir = os.path.join(seg_dir, f"sample_{local_i:06d}")
        if os.path.exists(sample_dir) and len(os.listdir(sample_dir)) >= 5:
            saved += 1
            continue
        os.makedirs(sample_dir, exist_ok=True)

        # ── 전방 현재 프레임 ─────────────────────────────────────────────────
        img_meta  = img_col[ep_frame_i]
        ride_name = get_ride_name(img_meta['path'])
        video_ts  = img_meta['timestamp']
        curr_path = front_frame_path(ride_name, video_ts)

        if not os.path.exists(curr_path):
            skipped += 1
            continue
        shutil.copy2(curr_path, os.path.join(sample_dir, "current_image.jpg"))

        # ── 전방 과거 컨텍스트 ───────────────────────────────────────────────
        for p in range(1, N_PAST + 1):
            past_ep_i = frame_indices[max(0, local_i - p)]
            past_meta = img_col[past_ep_i]
            past_path = front_frame_path(get_ride_name(past_meta['path']),
                                         past_meta['timestamp'])
            dst = os.path.join(sample_dir, f"past_image_{p}.jpg")
            shutil.copy2(past_path if os.path.exists(past_path) else curr_path, dst)

        # ── 후방 현재 프레임 (MP4에서 직접 추출) ────────────────────────────
        rear_meta = img_rear_col[ep_frame_i]
        rear_name = get_ride_name(rear_meta['path'])
        rear_ts   = rear_meta['timestamp']
        rear_cap  = rear_cache.get(rear_name)
        if rear_cap is not None:
            extract_rear_frame(rear_cap, rear_ts,
                               os.path.join(sample_dir, "rear_image.jpg"))

        # ── OSM ego-centric 지도 ─────────────────────────────────────────────
        osm_src = os.path.join(osm_dir, f"osm_map_{local_i:06d}.png")
        if os.path.exists(osm_src):
            shutil.copy2(osm_src, os.path.join(sample_dir, "osm_map.png"))

        # ── metadata.json ────────────────────────────────────────────────────
        meta = {
            "episode":       ep,
            "segment":       seg_idx,
            "local_frame":   local_i,
            "episode_frame": int(ep_frame_i),
            "timestamp":     float(ts_col[ep_frame_i]),
            "front_video_ts": float(video_ts),
            "rear_video_ts":  float(rear_ts),
            "ride_name":     ride_name,
            "gps": {
                "latitude":  float(lats[local_i]),
                "longitude": float(lons[local_i]),
            },
        }
        with open(os.path.join(sample_dir, "metadata.json"), 'w') as f:
            json.dump(meta, f, indent=2)

        saved += 1

    rear_cache.release_all()
    return saved, skipped


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    os.makedirs(OUT_ROOT, exist_ok=True)

    with open(SCORES_PATH) as f:
        scores = json.load(f)

    # 처리 대상 (episode, segment) 목록
    if args.ep is not None and args.seg is not None:
        targets = [(args.ep, args.seg)]
    elif args.ep is not None:
        targets = sorted({(s['episode'], s.get('segment', 0))
                          for s in scores
                          if s['episode'] == args.ep and s['selected']})
    else:
        targets = sorted({(s['episode'], s.get('segment', 0))
                          for s in scores if s['selected']})

    print(f"Dataset Builder 시작: {len(targets)}개 세그먼트")
    print(f"출력 위치: {OUT_ROOT}")

    print("Arrow 데이터 로딩 중...")
    table   = pa.ipc.open_stream(open(ARROW_PATH, 'rb')).read_all()
    ep_idx  = np.array(table['episode_index'].to_pylist())
    ts_all  = np.array(table['timestamp'].to_pylist())
    img_all      = table['observation.images.front'].to_pylist()
    img_rear_all = table['observation.images.rear'].to_pylist()
    fp_all  = np.array(table['observation.filtered_position'].to_pylist())
    lats_all = np.array(table['observation.latitude'].to_pylist())
    lons_all = np.array(table['observation.longitude'].to_pylist())

    by_ep = defaultdict(list)
    for ep, seg in targets:
        by_ep[ep].append(seg)

    total_saved = 0
    total_skipped = 0

    for ep in tqdm(sorted(by_ep.keys()), desc="에피소드"):
        mask    = ep_idx == ep
        ep_rows = np.where(mask)[0]   # Arrow 전체 테이블에서 이 episode의 행 번호

        ep_fp   = fp_all[mask]
        ep_lats = lats_all[mask]
        ep_lons = lons_all[mask]

        # episode 내 인덱스 → Arrow 전체 행 번호 변환을 위한 매핑
        ep_img_col      = [img_all[r]      for r in ep_rows]
        ep_img_rear_col = [img_rear_all[r] for r in ep_rows]
        ep_ts_col       = ts_all[ep_rows]

        ep_arrow = {
            'img_col':      ep_img_col,
            'img_rear_col': ep_img_rear_col,
            'ts_col':       ep_ts_col,
        }

        segments = split_into_segments(ep_fp, ep_lats, ep_lons)

        for seg_idx in sorted(by_ep[ep]):
            if seg_idx >= len(segments):
                tqdm.write(f"  ep{ep:03d}[{seg_idx}]: 세그먼트 없음, 스킵")
                continue

            seg = segments[seg_idx]
            saved, skipped = build_segment(ep, seg_idx, seg, ep_arrow, OUT_ROOT)
            total_saved   += saved
            total_skipped += skipped
            tqdm.write(f"  ep{ep:03d}[{seg_idx}]: {saved}개 샘플 저장"
                       + (f", {skipped}개 이미지 없음" if skipped else ""))

    print(f"\n=== 완료 ===")
    print(f"총 샘플: {total_saved:,}개  (이미지 없음: {total_skipped}개)")
    print(f"출력: {OUT_ROOT}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ep",  type=int, default=None)
    parser.add_argument("--seg", type=int, default=None)
    args = parser.parse_args()
    main(args)
