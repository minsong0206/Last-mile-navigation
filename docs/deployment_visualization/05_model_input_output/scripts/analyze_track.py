"""Step 1: analyze the raw HF ride to find valid ticks + classify straight/curve/turn.
Uses the REAL deployment heading estimator (imported, not reimplemented)."""
import sys, os, json, math
from pathlib import Path
SCRIPT_DIR = Path(__file__).resolve().parent            # .../05_model_input_output/scripts
VIZ_DIR = SCRIPT_DIR.parent                              # .../05_model_input_output
REPO_ROOT = VIZ_DIR.parents[2]                            # .../Last-mile-navigation
sys.path.insert(0, str(REPO_ROOT / "deployment"))
from omnivla_edge_deploy import estimate_heading_from_track, N_CTX  # real deploy code

DATA_DIR = str(VIZ_DIR / "_cache" / "hf_dataset")
CTX_STRIDE_SEC = 0.3
WAYPOINT_STRIDE_SEC = 0.7
N_WAYPOINTS = 8

gps_rows = [json.loads(l) for l in open(f"{DATA_DIR}/gps_imu.jsonl")]
gps_t = [r["local_timestamp"] for r in gps_rows]
gps_ll = [(r["latitude"], r["longitude"]) for r in gps_rows]

ctrl_rows = [json.loads(l) for l in open(f"{DATA_DIR}/control.jsonl")]
ctrl_t = [r["local_timestamp"] for r in ctrl_rows]

t0, t_end = ctrl_t[0], ctrl_t[-1]
gps_t0, gps_t_end = gps_t[0], gps_t[-1]

def past_track_upto(t):
    return [ll for tt, ll in zip(gps_t, gps_ll) if tt <= t]

def interp_gps(t):
    if t <= gps_t[0]:
        return gps_ll[0]
    if t >= gps_t[-1]:
        return gps_ll[-1]
    for i in range(1, len(gps_t)):
        if gps_t[i] >= t:
            t0_, t1_ = gps_t[i-1], gps_t[i]
            la0, lo0 = gps_ll[i-1]
            la1, lo1 = gps_ll[i]
            f = (t - t0_) / (t1_ - t0_) if t1_ > t0_ else 0.0
            return (la0 + f*(la1-la0), lo0 + f*(lo1-lo0))
    return gps_ll[-1]

LAT_M = 111320.0
def heading_at(t):
    return estimate_heading_from_track(past_track_upto(t))

# valid tick window: need >=6 camera frames spanning >= N_CTX*CTX_STRIDE_SEC before t,
# and >= N_WAYPOINTS*WAYPOINT_STRIDE_SEC of future time within the ride for GT.
min_t = t0 + N_CTX * CTX_STRIDE_SEC
max_t = t_end - N_WAYPOINTS * WAYPOINT_STRIDE_SEC
min_t = max(min_t, gps_t0 + 1.5)   # heading estimator needs >=1.5m accumulated path typically
max_t = min(max_t, gps_t_end)
print(f"valid tick window: [{min_t:.2f}, {max_t:.2f}]  (ride t0={t0:.2f} t_end={t_end:.2f})")

# sample candidate ticks every 1.0s across valid window, compute heading now and heading ~3s ahead
samples = []
t = min_t
while t <= max_t:
    h_now = heading_at(t)
    t_fwd = min(t + 3.0, max_t)
    h_fwd = heading_at(t_fwd)
    if h_now is not None and h_fwd is not None:
        dh = math.degrees(h_fwd - h_now)
        dh = (dh + 180) % 360 - 180
        lat, lon = interp_gps(t)
        samples.append((t, math.degrees(h_now), dh, lat, lon))
    t += 1.0

print(f"\n{'t':>10} {'heading_deg':>12} {'dheading_3s':>12}  classification")
for t, h, dh, lat, lon in samples:
    if abs(dh) < 10:
        cls = "STRAIGHT"
    elif abs(dh) < 35:
        cls = "CURVE"
    else:
        cls = "TURN"
    print(f"{t:10.2f} {h:12.1f} {dh:12.1f}  {cls}")
