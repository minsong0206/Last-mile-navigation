"""
traj_data.pkl (GPS Kalman) 기반으로 odom_{i}.jpg 생성
- MBRA inference 없음
- gnm_format/ride_xxx/traj_data.pkl → processed_gnm/ride_xxx/odom_{i}.jpg
"""

import os
import shutil
import pickle
import argparse
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm

GNM_ROOT = "/media/ms/WD_BLACK_4TB/Learning-to-Drive-Anywhere-with-MBRA/FrodoBots-2K/gnm_format"
OUT_ROOT = "/media/ms/WD_BLACK_4TB/Learning-to-Drive-Anywhere-with-MBRA/FrodoBots-2K/processed_gnm"

MAP_PX     = 256
MAP_SIZE_M = 20.0


def make_odom_map(positions, yaws, curr_idx):
    """
    GPS 기반 heading-up BEV odom map
    positions: (N,2) — pos[i] - pos[0] origin 정렬
    yaws: (N,)
    curr_idx: 현재 프레임
    """
    img = np.ones((MAP_PX, MAP_PX, 3), dtype=np.uint8) * 255
    curr_pos = positions[curr_idx]
    curr_yaw = yaws[curr_idx]
    scale = MAP_PX / MAP_SIZE_M

    def to_px(p):
        rel = p - curr_pos
        c, s = np.cos(-curr_yaw), np.sin(-curr_yaw)
        rx =  c * rel[0] - s * rel[1]
        ry =  s * rel[0] + c * rel[1]
        px_x = int(MAP_PX / 2 - ry * scale)
        px_y = int(MAP_PX / 2 - rx * scale)
        return (px_x, px_y)

    # 지나온 경로 (회색, 두께 6)
    for i in range(1, curr_idx + 1):
        cv2.line(img, to_px(positions[i-1]), to_px(positions[i]), (180, 180, 180), 6)

    # 앞으로 갈 경로 (파란, 두께 8)
    for i in range(curr_idx, len(positions) - 1):
        cv2.line(img, to_px(positions[i]), to_px(positions[i+1]), (200, 80, 0), 8)

    # 현재 위치 (원, 항상 중앙)
    cp = (MAP_PX // 2, MAP_PX // 2)
    cv2.circle(img, cp, 10, (200, 80, 0), -1)
    cv2.circle(img, cp, 10, (255, 255, 255), 2)

    return img


def process_ride(ride_name, gnm_root, out_root):
    in_dir  = os.path.join(gnm_root, ride_name)
    out_dir = os.path.join(out_root, ride_name)
    os.makedirs(out_dir, exist_ok=True)

    pkl_path = os.path.join(in_dir, "traj_data.pkl")
    if not os.path.exists(pkl_path):
        print(f"  skip {ride_name}: no traj_data.pkl")
        return False

    with open(pkl_path, "rb") as f:
        d = pickle.load(f)

    positions = np.array(d["pos"]) - d["pos"][0]  # origin 정렬
    yaws      = np.array(d["yaw"])
    N = len(positions)

    # jpg 복사
    jpgs = sorted([f for f in os.listdir(in_dir) if f.endswith(".jpg")],
                  key=lambda x: int(x.replace(".jpg", "")))
    for jpg in jpgs:
        shutil.copy(os.path.join(in_dir, jpg), os.path.join(out_dir, jpg))

    # odom map 생성
    for i in range(N):
        omap = make_odom_map(positions, yaws, i)
        omap_rgb = cv2.cvtColor(omap, cv2.COLOR_BGR2RGB)
        Image.fromarray(omap_rgb).save(os.path.join(out_dir, f"odom_{i}.jpg"))

    # traj_data.pkl 복사 (원본 그대로)
    shutil.copy(pkl_path, os.path.join(out_dir, "traj_data.pkl"))

    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gnm_root", default=GNM_ROOT)
    parser.add_argument("--out_root", default=OUT_ROOT)
    parser.add_argument("--rides",    nargs="*", default=None)
    args = parser.parse_args()

    os.makedirs(args.out_root, exist_ok=True)
    rides = args.rides or sorted(os.listdir(args.gnm_root))
    rides = [r for r in rides if os.path.isdir(os.path.join(args.gnm_root, r))]

    print(f"총 {len(rides)}개 ride 처리")
    success = 0
    for ride in tqdm(rides):
        ok = process_ride(ride, args.gnm_root, args.out_root)
        if ok:
            success += 1

    print(f"완료: {success}/{len(rides)} → {args.out_root}")


if __name__ == "__main__":
    main()
