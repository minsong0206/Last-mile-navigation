"""
osm_map 파일명을 세그먼트 내 0-based 인덱스 → Arrow frame_index 기준으로 rename.

기존: osm_map_000000.png ~ osm_map_000718.png  (세그먼트 내 순서)
변경: osm_map_000511.png ~ osm_map_001229.png  (Arrow episode 내 frame_index)

--dry_run: 실제 rename 없이 검증만 수행 (기본값)
--execute: 실제 rename 수행
"""

import os, sys, json, argparse
import numpy as np
import pyarrow as pa

sys.path.insert(0, "/media/ms/WD_BLACK_4TB/Learning-to-Drive-Anywhere-with-MBRA/osm_pipeline/py")
from episode_selector import split_into_segments

ARROW    = "/media/ms/WD_BLACK_4TB/Learning-to-Drive-Anywhere-with-MBRA/FrodoBots-2K/processed/output_rides_11/train/data-00000-of-00001.arrow"
OSM_MAPS = "/media/ms/WD_BLACK_4TB/Learning-to-Drive-Anywhere-with-MBRA/osm_pipeline/osm_data/output_rides_11/osm_maps"
SCORES   = "/media/ms/WD_BLACK_4TB/Learning-to-Drive-Anywhere-with-MBRA/osm_pipeline/osm_data/output_rides_11/episode_scores.json"


def main(args):
    print("Loading Arrow dataset...")
    table = pa.ipc.open_stream(open(ARROW, 'rb')).read_all()
    ei   = np.array(table['episode_index'])
    fp   = np.array(table['observation.filtered_position'].to_pylist())
    lats = np.array(table['observation.latitude'])
    lons = np.array(table['observation.longitude'])

    with open(SCORES) as f:
        scores = json.load(f)
    selected = [s for s in scores if s['selected']]
    print(f"Selected segments: {len(selected)}")

    total_renamed = 0
    total_errors  = 0

    for s in selected:
        ep, seg = s['episode'], s['segment']
        folder = os.path.join(OSM_MAPS, f"episode_{ep:04d}_seg{seg:02d}")

        if not os.path.isdir(folder):
            print(f"  [SKIP] {folder} not found")
            continue

        # Arrow에서 이 세그먼트의 frame_indices 계산
        mask    = ei == ep
        segs    = split_into_segments(fp[mask], lats[mask], lons[mask])
        if seg >= len(segs):
            print(f"  [ERROR] ep={ep} seg={seg} out of range (Arrow has {len(segs)} segs)")
            total_errors += 1
            continue

        frame_indices = segs[seg]['frame_indices']  # 에피소드 내 절대 frame_index 배열

        # 현재 파일 목록 (0-based 순서)
        files = sorted(
            [f for f in os.listdir(folder) if f.startswith('osm_map_') and f.endswith('.png')],
            key=lambda f: int(f[8:14])
        )

        if len(files) != len(frame_indices):
            print(f"  [ERROR] {folder}: files={len(files)} != arrow={len(frame_indices)}")
            total_errors += 1
            continue

        # 이미 Arrow 인덱스로 rename된 경우 skip
        first_file_idx = int(files[0][8:14])
        if first_file_idx == frame_indices[0]:
            total_renamed += len(files)
            continue

        if args.dry_run:
            # 첫 3개만 출력
            for i in range(min(3, len(files))):
                print(f"  [DRY] {folder}/{files[i]}  →  osm_map_{frame_indices[i]:06d}.png")
            if len(files) > 3:
                print(f"         ... ({len(files)}개 총)")
            total_renamed += len(files)
            continue

        # 충돌 방지: 임시 이름으로 먼저 rename 후 최종 이름으로 rename
        tmp_names = []
        for i, fname in enumerate(files):
            src = os.path.join(folder, fname)
            tmp = os.path.join(folder, f"__tmp_{i:06d}.png")
            os.rename(src, tmp)
            tmp_names.append(tmp)

        for i, tmp in enumerate(tmp_names):
            dst = os.path.join(folder, f"osm_map_{frame_indices[i]:06d}.png")
            os.rename(tmp, dst)

        total_renamed += len(files)

    print()
    if args.dry_run:
        print(f"[DRY RUN] rename 예정 파일: {total_renamed:,}개  오류: {total_errors}개")
        print("실제 실행하려면 --execute 옵션을 사용하세요.")
    else:
        print(f"[DONE] renamed: {total_renamed:,}개  오류: {total_errors}개")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--dry_run", action="store_true", default=True,
                       help="검증만 수행 (기본값)")
    group.add_argument("--execute", dest="dry_run", action="store_false",
                       help="실제 rename 수행")
    args = parser.parse_args()
    main(args)
