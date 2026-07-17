"""
extract_frames.py

Arrow의 timestamp 기준으로 mp4에서 프레임을 추출해 JPEG로 저장.

출력 구조:
  frames/
    episode_0000/
      000000.jpg   ← frame_index=0
      000001.jpg   ← frame_index=1
      ...
    episode_0001/
      ...

Arrow의 (episode_index, frame_index, timestamp, video_path)를 읽어,
각 mp4에서 해당 timestamp의 프레임을 PyAV seek으로 추출.

실행:
  conda run -n mbra python scripts/data/extract_frames.py
  conda run -n mbra python scripts/data/extract_frames.py --workers 8
  conda run -n mbra python scripts/data/extract_frames.py --resume
  conda run -n mbra python scripts/data/extract_frames.py --ep 0 5
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import pyarrow as pa
import av
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]

ARROW_PATH  = str(REPO_ROOT / "FrodoBots-2K/processed/output_rides_11/train/data-00000-of-00001.arrow")
VIDEO_ROOT  = REPO_ROOT / "FrodoBots-2K/processed/output_rides_11"
OUT_ROOT    = REPO_ROOT / "FrodoBots-2K/processed/output_rides_11/frames"

JPEG_QUALITY = 90   # JPEG 품질 (1~95, 높을수록 고화질/용량 큼)
IMG_SIZE     = None  # None = 원본 크기 유지. (96,96) 등으로 설정하면 리사이즈


def extract_episode(args):
    """
    하나의 episode에 대해 모든 프레임을 추출.

    Arrow timestamp = mp4 내부 PTS (초 단위).
    seek이 이 mp4 포맷에서 신뢰할 수 없으므로 (항상 마지막 프레임으로 jump),
    순차 디코딩으로 전체 프레임을 읽으면서 Arrow timestamp에 nearest-neighbor 매핑.

    매핑 조건: decoded_pts >= target_ts - tolerance (0.03s)
    """
    ep, rows, resume = args
    out_dir = OUT_ROOT / f"episode_{ep:04d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # video별로 그룹화
    video_groups = {}
    for fi, ts, vpath in rows:
        if vpath not in video_groups:
            video_groups[vpath] = []
        video_groups[vpath].append((fi, ts))

    saved = 0
    errors = 0
    TOLERANCE = 0.03  # 30ms — 20Hz 프레임 간격(50ms)의 절반

    for vpath, frames in video_groups.items():
        abs_path = str(VIDEO_ROOT / vpath)
        if not os.path.exists(abs_path):
            errors += len(frames)
            continue

        try:
            # timestamp 오름차순 정렬 (순차 디코딩과 일치)
            frames_sorted = sorted(frames, key=lambda x: x[1])

            # resume: 이미 모두 추출됐으면 건너뜀
            if resume:
                all_exist = all(
                    (out_dir / f"{fi:06d}.jpg").exists()
                    for fi, _ in frames_sorted
                )
                if all_exist:
                    saved += len(frames_sorted)
                    continue

            container = av.open(abs_path)
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"

            target_idx = 0
            n_targets = len(frames_sorted)

            for frame in container.decode(stream):
                if target_idx >= n_targets:
                    break

                pts = float(frame.pts * stream.time_base) if frame.pts is not None else None
                if pts is None:
                    continue

                # 현재 decoded PTS가 target timestamp에 도달했는지 확인
                fi_target, ts_target = frames_sorted[target_idx]
                if pts >= ts_target - TOLERANCE:
                    out_path = out_dir / f"{fi_target:06d}.jpg"

                    if not (resume and out_path.exists()):
                        frame_img = frame.to_image().convert("RGB")
                        if IMG_SIZE is not None:
                            frame_img = frame_img.resize(IMG_SIZE, Image.BILINEAR)
                        frame_img.save(str(out_path), "JPEG", quality=JPEG_QUALITY)

                    saved += 1
                    target_idx += 1

                    # 같은 프레임이 여러 target에 매핑될 수 있음 (연속 확인)
                    while target_idx < n_targets:
                        fi_next, ts_next = frames_sorted[target_idx]
                        if pts >= ts_next - TOLERANCE:
                            out_path_next = out_dir / f"{fi_next:06d}.jpg"
                            if not (resume and out_path_next.exists()):
                                frame_img = frame.to_image().convert("RGB")
                                if IMG_SIZE is not None:
                                    frame_img = frame_img.resize(IMG_SIZE, Image.BILINEAR)
                                frame_img.save(str(out_path_next), "JPEG", quality=JPEG_QUALITY)
                            saved += 1
                            target_idx += 1
                        else:
                            break

            container.close()

            # 끝까지 디코딩했는데 남은 target이 있으면 에러
            errors += (n_targets - target_idx)

        except Exception:
            errors += len(frames)

    return ep, saved, errors


def main(args):
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    # Arrow 로드
    print("Loading Arrow...")
    table = pa.ipc.open_stream(open(ARROW_PATH, "rb")).read_all()
    ep_arr    = np.array(table["episode_index"].to_pylist(), dtype=np.int64)
    fi_arr    = np.array(table["frame_index"].to_pylist(),   dtype=np.int64)
    vf_list   = table["observation.images.front"].to_pylist()
    ts_arr    = np.array([v["timestamp"] for v in vf_list],  dtype=np.float64)
    path_arr  = np.array([v["path"]      for v in vf_list])

    total_eps = int(ep_arr.max()) + 1
    print(f"Total episodes: {total_eps}, frames: {len(ep_arr):,}, videos: {len(np.unique(path_arr))}")

    # episode 범위 필터
    ep_start = args.ep[0] if args.ep else 0
    ep_end   = args.ep[1] if args.ep else total_eps - 1

    # episode별 row 묶기
    ep_jobs = []
    for ep in range(ep_start, ep_end + 1):
        mask = ep_arr == ep
        rows = list(zip(
            fi_arr[mask].tolist(),
            ts_arr[mask].tolist(),
            path_arr[mask].tolist(),
        ))
        if rows:
            ep_jobs.append((ep, rows, args.resume))

    print(f"Extracting episodes {ep_start}~{ep_end} ({len(ep_jobs)} episodes)")
    print(f"Output: {OUT_ROOT}")
    print(f"Workers: {args.workers}, JPEG quality: {JPEG_QUALITY}")

    total_saved  = 0
    total_errors = 0

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(extract_episode, job): job[0] for job in ep_jobs}
        with tqdm(total=len(ep_jobs), desc="Episodes") as pbar:
            for future in as_completed(futures):
                ep, saved, errors = future.result()
                total_saved  += saved
                total_errors += errors
                pbar.update(1)
                if errors > 0:
                    pbar.write(f"  ep{ep:04d}: {saved} saved, {errors} errors")

    print(f"\nDone. saved={total_saved:,}  errors={total_errors:,}")
    print(f"Output: {OUT_ROOT}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4,
                        help="병렬 worker 수 (기본 4)")
    parser.add_argument("--resume",  action="store_true",
                        help="이미 추출된 파일 건너뜀")
    parser.add_argument("--ep",      type=int, nargs=2, default=None,
                        metavar=("START", "END"),
                        help="추출할 episode 범위 (예: --ep 0 10)")
    args = parser.parse_args()
    main(args)
