import sys, os, json, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from PIL import Image
import build_viz as bv

SCENARIOS = {
    "straight": 1786194922.35,
    "curve":    1786194897.35,
    "turn":     1786194956.35,
}

route_latlon = bv.build_route_latlon()
lmb = bv.make_live_map_builder(route_latlon)

results = {}
for label, t in SCENARIOS.items():
    print(f"=== {label} (t={t}) ===")
    lat, lon = bv.interp_gps(t)
    heading_rad = bv.estimate_heading_from_track(bv.past_track_upto(t))
    heading_deg = math.degrees(heading_rad)
    route_bearing_rad = lmb.route_bearing_rad(lat, lon)
    route_bearing_deg = math.degrees(route_bearing_rad) if route_bearing_rad is not None else None
    map_rotation_deg = 90.0 - heading_deg
    heading_route_diff_deg = None
    if route_bearing_deg is not None:
        heading_route_diff_deg = (heading_deg - route_bearing_deg + 180) % 360 - 180

    # context frames: download + load
    kept = bv.get_context_frames(t)
    assert len(kept) == bv.N_CTX + 1, f"only {len(kept)} context frames available for {label}"
    ctx_paths = [bv.download_image(rel) for _, rel in kept]
    ctx_pil = [Image.open(p).convert("RGB") for p in ctx_paths]

    # map render (real production path)
    map_np = lmb.get_map_image(lat, lon, heading_rad, past_track=bv.past_track_upto(t))

    # inference (real checkpoint)
    pred_xy_m = bv.run_inference(ctx_pil, map_np)
    gt_xy_m = bv.compute_gt_waypoints(t, heading_rad)

    ade = float(np.mean(np.linalg.norm(pred_xy_m - gt_xy_m, axis=1)))
    fde = float(np.linalg.norm(pred_xy_m[-1] - gt_xy_m[-1]))

    overlay_pred = lmb.draw_predicted_trajectory(map_np, pred_xy_m, color=(0, 255, 255))

    # pre-warp north-up panels (only really need these for the detailed sample, but cheap to build for all)
    northup_raw = bv.prewarp_northup_crop(lmb, lat, lon, heading_rad, draw_ego=False)
    northup_ego = bv.prewarp_northup_crop(lmb, lat, lon, heading_rad, draw_ego=True)

    out = os.path.join(bv.OUT_DIR, label)
    os.makedirs(out, exist_ok=True)
    for i, p in enumerate(ctx_pil):
        p.save(os.path.join(out, f"ctx_{i}.jpg"))
    Image.fromarray(map_np).save(os.path.join(out, "map_headingup.png"))
    Image.fromarray(map_np).resize((96, 96), Image.BILINEAR).save(os.path.join(out, "map_input96.png"))
    Image.fromarray(overlay_pred).save(os.path.join(out, "map_overlay_pred.png"))
    Image.fromarray(northup_raw).save(os.path.join(out, "northup_raw.png"))
    Image.fromarray(northup_ego).save(os.path.join(out, "northup_ego.png"))

    results[label] = dict(
        t=t, lat=lat, lon=lon, heading_deg=heading_deg,
        route_bearing_deg=route_bearing_deg, map_rotation_deg=map_rotation_deg,
        heading_route_diff_deg=heading_route_diff_deg,
        ctx_rel_paths=[rel for _, rel in kept], ctx_timestamps=[tt for tt, _ in kept],
        pred_xy_m=pred_xy_m.tolist(), gt_xy_m=gt_xy_m.tolist(),
        ade_m=ade, fde_m=fde,
        out_dir=out,
    )
    print(json.dumps(results[label], indent=2)[:800])

with open(os.path.join(bv.OUT_DIR, "results.json"), "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved results.json")
