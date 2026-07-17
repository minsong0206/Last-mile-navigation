"""
rides11_dataset.py

FrodoBots rides_11 Dataset for OmniVLA-Edge fine-tuning.

설계 기준 (rides11_datatype_design.md):
  - 1 sample = 1 valid frame index fi within a selected segment
  - obs_stack:  (18, 96, 96) — [ctx5..ctx1, obs] 6프레임 채널 concat
                frame_indices: fi-15, fi-12, fi-9, fi-6, fi-3, fi
  - map_img:    osm_maps/episode_{ep:04d}_seg{seg:02d}/osm_map_{fi:06d}.png
  - gt_waypoints: (8,2) ego-frame metres, normalized by METRIC_WAYPOINT_SPACING=0.125

Valid frame condition:
  fi >= seg_fi_start + 15  (ctx 5장 확보, PAST_MARGIN=15)
  fi <= seg_fi_end - 24    (future 8 waypoints × 3프레임 확보)

사용:
  ds = Rides11Dataset(arrow_path, scores_path, osm_root, video_root)
  sample = ds[0]
"""

import os
import math
import json
import numpy as np
import pyarrow as pa
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import av   # PyAV — VideoFrame 디코딩

# ── 상수 ──────────────────────────────────────────────────────────────────────
METRIC_WAYPOINT_SPACING = 0.125      # 0.25 * 0.5 m — OmniVLA-Edge 기준
N_WAYPOINTS             = 8          # gt_waypoints 개수
WAYPOINT_STRIDE         = 3          # 프레임 간격 (0.3초)
CTX_STRIDE              = 3          # context 프레임 간격
N_CTX                   = 5          # 과거 이미지 수 (context_size=5, pretrained weight 호환)
PAST_MARGIN             = CTX_STRIDE * N_CTX    # = 15
FUTURE_MARGIN           = WAYPOINT_STRIDE * N_WAYPOINTS  # = 24

IMG_SIZE_OBS  = (96, 96)    # obs/ctx/map 이미지 크기 (OmniVLA-Edge 기준)

IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD  = [0.229, 0.224, 0.225]

# ── 이미지 transform ──────────────────────────────────────────────────────────
def make_obs_transform():
    return transforms.Compose([
        transforms.Resize(IMG_SIZE_OBS),
        transforms.ToTensor(),
        transforms.Normalize(IMG_MEAN, IMG_STD),
    ])

def make_map_transform():
    return transforms.Compose([
        transforms.Resize(IMG_SIZE_OBS),
        transforms.ToTensor(),
        transforms.Normalize(IMG_MEAN, IMG_STD),
    ])


# ── VideoFrame 디코딩 ─────────────────────────────────────────────────────────
class VideoReader:
    """
    mp4 파일을 열어두고 특정 timestamp의 프레임을 RGB PIL Image로 반환.
    episode별로 캐시하여 반복 open을 방지.
    """
    def __init__(self):
        self._cache: Dict[str, av.container.InputContainer] = {}

    def _open(self, path: str) -> av.container.InputContainer:
        if path not in self._cache:
            self._cache[path] = av.open(path)
        return self._cache[path]

    def get_frame(self, video_path: str, timestamp_s: float) -> Image.Image:
        """
        timestamp_s 초에 해당하는 프레임을 RGB PIL Image로 반환.
        """
        container = self._open(video_path)
        stream = container.streams.video[0]
        # stream=stream을 넘기면 offset은 마이크로초가 아니라 그 stream의 time_base 단위로
        # 해석됨 (AV_TIME_BASE=1e6은 stream 인자 없이 컨테이너 전체를 seek할 때만 적용).
        # 이전 코드는 마이크로초 값을 그대로 넘겨서 항상 영상 끝 근처로 seek되는 버그가 있었음.
        pts_target = int(timestamp_s / stream.time_base)
        container.seek(pts_target, stream=stream, backward=True, any_frame=False)
        for frame in container.decode(stream):
            return frame.to_image().convert("RGB")
        # fallback: 첫 프레임
        container.seek(0, stream=stream)
        for frame in container.decode(stream):
            return frame.to_image().convert("RGB")

    def close_all(self):
        for c in self._cache.values():
            c.close()
        self._cache.clear()


# ── Arrow 로드 & lookup table 구성 ───────────────────────────────────────────
def build_lookup(
    arrow_path: str,
    scores_path: str,
) -> Tuple[List[Tuple[int,int,int]], Dict, np.ndarray, np.ndarray, np.ndarray]:
    """
    Arrow와 episode_scores.json을 읽어서:
      1. valid_samples: List of (ep, seg, fi)
      2. global_row_map: dict[(ep, fi)] → global Arrow row index
      3. filtered_position: (N, 2) array
      4. filtered_heading:  (N,) array
      5. video_path_arr: (N,) — 각 global row의 video 파일 경로 (str)

    반환값은 dataloader __init__에서 한 번만 호출.
    """
    print(f"[Dataset] Loading Arrow: {arrow_path}")
    table = pa.ipc.open_stream(open(arrow_path, "rb")).read_all()

    ep_idx_arr  = np.array(table["episode_index"].to_pylist(),  dtype=np.int64)
    fi_arr      = np.array(table["frame_index"].to_pylist(),    dtype=np.int64)
    fp_arr      = np.array(table["observation.filtered_position"].to_pylist(), dtype=np.float32)
    fh_arr      = np.array(table["observation.filtered_heading"].to_pylist(),  dtype=np.float32)

    # VideoFrame path & timestamp
    vf_list     = table["observation.images.front"].to_pylist()
    # vf_list[i] = {"path": "videos/ride_XXXX_front_camera.mp4", "timestamp": float}
    vf_paths    = np.array([v["path"] for v in vf_list])
    vf_ts       = np.array([v["timestamp"] for v in vf_list], dtype=np.float32)

    # global_row_map: (ep, fi) → global row index
    print("[Dataset] Building global_row_map ...")
    global_row_map = {}
    for row in range(len(ep_idx_arr)):
        key = (int(ep_idx_arr[row]), int(fi_arr[row]))
        global_row_map[key] = row

    # episode_scores.json 로드
    print(f"[Dataset] Loading scores: {scores_path}")
    with open(scores_path) as f:
        scores = json.load(f)
    selected = [s for s in scores if s["selected"]]
    print(f"[Dataset] Selected segments: {len(selected)}")

    # episode_selector의 split_into_segments 재실행해서 frame_indices 복원
    # (scores에 frame_indices가 없으므로 Arrow에서 다시 split)
    from episode_selector import split_into_segments

    # episode별 fp, fi 캐시
    # (lat/lon 전체 컬럼은 루프 밖에서 한 번만 변환 — 이전에는 episode마다 매번
    #  전체 테이블을 to_pylist()로 재변환해 O(episodes × N)로 느렸음)
    lat_arr = np.array(table["observation.latitude"].to_pylist(), dtype=np.float64)
    lon_arr = np.array(table["observation.longitude"].to_pylist(), dtype=np.float64)
    ep_data_cache = {}
    for ep in sorted(set(s["episode"] for s in selected)):
        mask = ep_idx_arr == ep
        ep_data_cache[ep] = {
            "fp":  fp_arr[mask],
            "fi":  fi_arr[mask],
            "lats": lat_arr[mask],
            "lons": lon_arr[mask],
        }

    # valid_samples 구성
    valid_samples = []
    for s in selected:
        ep  = s["episode"]
        seg = s["segment"]
        ep_d = ep_data_cache[ep]

        segments = split_into_segments(ep_d["fp"], ep_d["lats"], ep_d["lons"])
        if seg >= len(segments):
            continue

        seg_obj = segments[seg]
        frame_indices = seg_obj["frame_indices"]  # episode-local fi 배열
        fi_start = int(frame_indices[0])
        fi_end   = int(frame_indices[-1])

        for seg_local_idx, fi in enumerate(frame_indices):
            fi = int(fi)
            # Valid condition
            if fi < fi_start + PAST_MARGIN:
                continue
            if fi > fi_end - FUTURE_MARGIN:
                continue
            # seg_local_idx: osm_map_{seg_local_idx:06d}.png 파일명과 1:1 대응
            valid_samples.append((ep, seg, fi, seg_local_idx))

    print(f"[Dataset] Valid samples: {len(valid_samples):,}")
    # valid_samples: List of (ep, seg, fi, seg_local_idx)
    # seg_local_idx: 세그먼트 내 0-based 인덱스 → osm_map_{seg_local_idx:06d}.png
    return valid_samples, global_row_map, fp_arr, fh_arr, vf_paths, vf_ts


# ── Dataset ───────────────────────────────────────────────────────────────────
class Rides11Dataset(Dataset):
    """
    FrodoBots rides_11 Dataset.

    Args:
        arrow_path:  .arrow 파일 경로
        scores_path: episode_scores.json 경로
        osm_root:    osm_maps/ 루트 (episode_{ep:04d}_seg{seg:02d}/ 하위)
        video_root:  videos/ 루트 (mp4 파일들이 있는 폴더)
        normalize:   이미지 정규화 여부
    """

    def __init__(
        self,
        arrow_path:  str,
        scores_path: str,
        osm_root:    str,
        video_root:  str,
        normalize:   bool = True,
    ):
        self.osm_root   = Path(osm_root)
        # video_root: Arrow의 "videos/ride_XXXX.mp4" 기준 루트
        # 예: .../output_rides_11/  → video_root / "videos/ride_XXXX.mp4" 가 실제 경로
        self.video_root = Path(video_root)

        (
            self.valid_samples,
            self.global_row_map,
            self.fp_arr,
            self.fh_arr,
            self.vf_paths,
            self.vf_ts,
        ) = build_lookup(arrow_path, scores_path)

        self.obs_transform = make_obs_transform() if normalize else transforms.Compose([
            transforms.Resize(IMG_SIZE_OBS), transforms.ToTensor()])
        self.map_transform = make_map_transform() if normalize else transforms.Compose([
            transforms.Resize(IMG_SIZE_OBS), transforms.ToTensor()])

        # VideoReader는 worker별로 생성 (fork 문제 방지)
        self._video_reader: Optional[VideoReader] = None

    @property
    def video_reader(self) -> VideoReader:
        if self._video_reader is None:
            self._video_reader = VideoReader()
        return self._video_reader

    def __len__(self) -> int:
        return len(self.valid_samples)

    def _get_frame(self, ep: int, fi: int) -> Image.Image:
        """
        (ep, fi) → RGB PIL Image.

        frames/ 디렉토리에 추출된 JPEG가 있으면 직접 읽고,
        없으면 mp4에서 디코딩 (fallback).
        """
        # 추출된 JPEG 경로: frames/episode_{ep:04d}/{fi:06d}.jpg
        jpeg_path = self.video_root / "frames" / f"episode_{ep:04d}" / f"{fi:06d}.jpg"
        if jpeg_path.exists():
            return Image.open(str(jpeg_path)).convert("RGB")

        # fallback: mp4 디코딩 (frames 미추출 시)
        row = self.global_row_map[(ep, fi)]
        rel_path = self.vf_paths[row]
        ts       = float(self.vf_ts[row])
        abs_path = str(self.video_root / rel_path)
        return self.video_reader.get_frame(abs_path, ts)

    def _get_waypoints(self, ep: int, fi: int) -> torch.Tensor:
        """
        fi 기준 미래 8 waypoints를 ego-frame으로 변환 후 정규화.
        반환: (8, 2) float32 — (x_ego, y_ego) / METRIC_WAYPOINT_SPACING
        """
        row_curr = self.global_row_map[(ep, fi)]
        pos_curr = self.fp_arr[row_curr]             # (2,) — EKF position
        hdg_curr = float(self.fh_arr[row_curr])      # heading (rad)

        cos_h = math.cos(hdg_curr)
        sin_h = math.sin(hdg_curr)

        waypoints = []
        for k in range(1, N_WAYPOINTS + 1):
            fi_fut = fi + k * WAYPOINT_STRIDE
            row_fut = self.global_row_map.get((ep, fi_fut))
            if row_fut is None:
                # 경계 밖 — 마지막 값으로 pad
                waypoints.append(waypoints[-1] if waypoints else [0.0, 0.0])
                continue
            pos_fut = self.fp_arr[row_fut]
            dx = pos_fut[0] - pos_curr[0]
            dy = pos_fut[1] - pos_curr[1]
            # 전역 (E, N) → ego frame (forward, left)
            x_ego =  dx * cos_h + dy * sin_h
            y_ego = -dx * sin_h + dy * cos_h
            # 정규화
            waypoints.append([
                x_ego / METRIC_WAYPOINT_SPACING,
                y_ego / METRIC_WAYPOINT_SPACING,
            ])

        return torch.tensor(waypoints, dtype=torch.float32)  # (8, 2)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ep, seg, fi, seg_local_idx = self.valid_samples[idx]

        # ── obs_stack (과거 5장 + 현재 1장 = 6프레임, context_size=5) ───────────
        # 순서: [fi-15, fi-12, fi-9, fi-6, fi-3, fi] → cat → (18, 96, 96)
        ctx_imgs = [
            self.obs_transform(self._get_frame(ep, fi - CTX_STRIDE * (N_CTX - k)))
            for k in range(N_CTX)
        ]  # ctx_imgs[0]=fi-15, ..., ctx_imgs[4]=fi-3, 각각 (3,96,96)
        obs_pil = self._get_frame(ep, fi)
        obs_img = self.obs_transform(obs_pil)            # (3, 96, 96)
        obs_stack = torch.cat([*ctx_imgs, obs_img], dim=0)  # (18, 96, 96)

        # ── map_images (3ch) ─────────────────────────────────────────────────
        # OmniVLA-edge-odom: goal_encoder expects (3, 96, 96) — OSM 맵 1장만 사용
        # osm_map_generator.py는 세그먼트 내 0-based 인덱스로 파일명 생성
        # → seg_local_idx 사용 (fi가 아님)
        map_path = (
            self.osm_root
            / f"episode_{ep:04d}_seg{seg:02d}"
            / f"osm_map_{seg_local_idx:06d}.png"
        )
        map_pil  = Image.open(map_path).convert("RGB")
        map_images = self.map_transform(map_pil)         # (3, 96, 96)

        # ── gt_waypoints ──────────────────────────────────────────────────────
        gt_waypoints = self._get_waypoints(ep, fi)       # (8, 2)

        return {
            "obs_stack":    obs_stack,      # (18, 96, 96) — 6프레임 스택
            "map_images":   map_images,     # (3, 96, 96)  — OSM 맵 1장 (odom3ch model)
            "gt_waypoints": gt_waypoints,   # (8, 2)
            # metadata
            "episode_index":   torch.tensor(ep,            dtype=torch.long),
            "frame_index":     torch.tensor(fi,            dtype=torch.long),
            "segment":         torch.tensor(seg,           dtype=torch.long),
            "seg_local_idx":   torch.tensor(seg_local_idx, dtype=torch.long),
        }

    def __del__(self):
        if self._video_reader is not None:
            self._video_reader.close_all()


# ── 단위 테스트 ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    from torch.utils.data import DataLoader

    parser = argparse.ArgumentParser()
    parser.add_argument("--arrow",   required=True)
    parser.add_argument("--scores",  required=True)
    parser.add_argument("--osm",     required=True)
    parser.add_argument("--videos",  required=True,
                        help="Arrow rel_path 기준 루트 (예: .../output_rides_11/)")
    parser.add_argument("--n",       type=int, default=3, help="테스트 샘플 수")
    args = parser.parse_args()

    ds = Rides11Dataset(
        arrow_path  = args.arrow,
        scores_path = args.scores,
        osm_root    = args.osm,
        video_root  = args.videos,
    )
    print(f"\nDataset size: {len(ds):,}")

    # 샘플 shape 확인
    for i in range(min(args.n, len(ds))):
        s = ds[i]
        ep  = s["episode_index"].item()
        fi  = s["frame_index"].item()
        seg = s["segment"].item()
        print(f"\n[Sample {i}] ep={ep}, seg={seg}, fi={fi}")
        for k, v in s.items():
            if isinstance(v, torch.Tensor):
                print(f"  {k}: shape={tuple(v.shape)}, "
                      f"min={v.float().min():.3f}, max={v.float().max():.3f}")

    # DataLoader 테스트 (num_workers=0)
    print("\n[DataLoader test] batch_size=4, num_workers=0")
    loader = DataLoader(ds, batch_size=4, shuffle=True, num_workers=0)
    batch = next(iter(loader))
    print("  obs_stack:    ", tuple(batch["obs_stack"].shape))
    print("  map_images:   ", tuple(batch["map_images"].shape))
    print("  gt_waypoints: ", tuple(batch["gt_waypoints"].shape))
    print("\nDone.")
