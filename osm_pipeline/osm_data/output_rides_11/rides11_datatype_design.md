# rides_11 Dataset — Data Type Design

## 1. Sample Unit

**1 sample = 1 valid frame index `i`** within a selected segment.

```
Arrow row: (episode_index=ep, frame_index=i)
```

---

## 2. Valid Frame Condition

Frame `i` is valid when **all three conditions** hold:

```python
# Condition 1: enough past frames for ctx (within same episode)
fi >= seg_frame_indices[0] + 6

# Condition 2: enough future frames for gt_waypoints (within same episode)
fi <= seg_frame_indices[-1] - 24

# Condition 3: frame belongs to a selected segment
(ep, fi) in selected_segment_frame_indices
```

### Statistics (rides_11)
| | Count |
|---|---|
| Selected segments | 538 |
| Raw frames in selected segments | 457,676 |
| **Valid samples** | **441,536** |
| Valid ratio | 96.5% |

### Why episode boundary matters
Arrow global rows are **not** episode-continuous.
`ep=1, fi=6` is at global row 1409, but `global_row - 6 = 1403` belongs to ep=0.
→ **Always use `frame_index` (episode-local) for offset arithmetic, never global row offsets.**

---

## 3. Input Components

### 3-1. obs_img
```
Arrow: observation.images.front  (VideoFrame)
  path:      videos/ride_{id}_front_camera.mp4
  timestamp: fi × 0.1 s  (10 Hz, verified: Δt=0.1s per frame)

Access via LeRobot VideoFrame API — no jpg extraction needed.
```

### 3-2. ctx_img_t1  (~0.3s ago)
```
Same episode, frame_index = i - 3
timestamp = ts[i] - 0.3s
```

### 3-3. ctx_img_t2  (~0.6s ago)
```
Same episode, frame_index = i - 6
timestamp = ts[i] - 0.6s
```

> **Note:** ctx frames must be fetched by `(episode_index=ep, frame_index=i-3/i-6)`,
> not by `global_row - 3/6`.

### 3-4. map_img
```
Root: osm_pipeline/osm_data/output_rides_11/osm_maps/
File: episode_{ep:04d}_seg{seg:02d}/osm_map_{i:06d}.png
Size: 224×224 px, RGB, ego-heading-up
```

- `ep` = Arrow `episode_index` (0-based)
- `seg` = segment index within episode (from episode_scores.json)
- `i`  = Arrow `frame_index` (1:1 correspondence, renamed 2026-05-23)
- osm_maps/ has 539 folders; 1 extra (`ride_28156_20240406074249`) is a legacy
  non-selected folder — ignore it in the dataloader.

---

## 4. Ground Truth Waypoints

### Formula
```python
METRIC_WAYPOINT_SPACING = 0.25 * 0.5  # = 0.125 m  (matches train_utils.py L1844)

heading_i = filtered_heading[i]   # EKF heading (rad), East=0 CCW
future_idx = [i+3, i+6, i+9, i+12, i+15, i+18, i+21, i+24]  # 8 waypoints

waypoints_m = []
for j in future_idx:
    dx = filtered_position[j] - filtered_position[i]   # global displacement (m)
    x_ego =  dx[0]*cos(heading_i) + dx[1]*sin(heading_i)   # forward  (+X)
    y_ego = -dx[0]*sin(heading_i) + dx[1]*cos(heading_i)   # left     (+Y)
    waypoints_m.append((x_ego, y_ego))

# Normalize to match model action space (same as train_utils.py)
gt_waypoints = waypoints_m / METRIC_WAYPOINT_SPACING   # unit: normalised steps
# Output: (8, 2) float32
```

### Normalization rationale
`METRIC_WAYPOINT_SPACING = 0.125 m` is the canonical value used in:
- `train_utils.py:1844` — `metric_waypoint_spacing = 0.25*0.5`
- `build_odom_dataset.py` — `TRAIN_METRIC_SPACING = 0.25 * 0.5`
- `trajectory_map_generator.py` — same constant

Dataloader outputs **normalised** waypoints (divide by 0.125).
To recover metres: `waypoints_m = gt_waypoints * 0.125`.

---

## 5. Selected Segment Index (lookup table)

At dataloader init, build a lookup from Arrow + episode_scores.json:

```python
# Pseudocode — build once at init
import json, numpy as np, pyarrow as pa
from episode_selector import split_into_segments

valid_samples = []   # list of (ep, seg, fi)
global_row_map = {}  # (ep, fi) → global Arrow row index

table = pa.ipc.open_stream(open(ARROW_PATH, 'rb')).read_all()
ei_arr = np.array(table['episode_index'])
fi_arr = np.array(table['frame_index'])
fp_arr = np.array(table['observation.filtered_position'].to_pylist())
lats   = np.array(table['observation.latitude'])
lons   = np.array(table['observation.longitude'])

# Build (ep, fi) → global row lookup
for row, (ep, fi) in enumerate(zip(ei_arr, fi_arr)):
    global_row_map[(int(ep), int(fi))] = row

with open(SCORES_PATH) as f:
    scores = json.load(f)

for s in scores:
    if not s['selected']:
        continue
    ep, seg = s['episode'], s['segment']
    mask = ei_arr == ep
    segs = split_into_segments(fp_arr[mask], lats[mask], lons[mask])
    if seg >= len(segs):
        continue
    frame_indices = segs[seg]['frame_indices']   # episode-local fi array
    fi_start = frame_indices[0]
    fi_end   = frame_indices[-1]
    for fi in frame_indices:
        if fi >= fi_start + 6 and fi <= fi_end - 24:
            valid_samples.append((ep, seg, int(fi)))

# Total: 441,536 valid samples
```

Sample lookup at index `k`:
```python
ep, seg, fi = valid_samples[k]
row         = global_row_map[(ep, fi)]     # global Arrow row
row_minus3  = global_row_map[(ep, fi-3)]   # ctx_img_t1
row_minus6  = global_row_map[(ep, fi-6)]   # ctx_img_t2
```

---

## 6. Full Sample Schema

```python
{
  # Inputs
  'obs_img':      Tensor[3, H, W],       # current frame (VideoFrame at fi)
  'ctx_img_t1':   Tensor[3, H, W],       # fi-3  (~0.3s ago)
  'ctx_img_t2':   Tensor[3, H, W],       # fi-6  (~0.6s ago)
  'map_img':      Tensor[3, 224, 224],   # osm ego-map

  # Labels (normalised by METRIC_WAYPOINT_SPACING = 0.125 m)
  'gt_waypoints': Tensor[8, 2],          # (x_ego, y_ego) / 0.125

  # Metadata (for debug / loss masking)
  'episode_index': int,
  'frame_index':   int,
  'segment':       int,
}
```

---

## 7. Data Source Files

| Component | Path |
|---|---|
| obs_img / ctx_img | `FrodoBots-2K/processed/output_rides_11/train/data-00000-of-00001.arrow` |
| map_img | `osm_pipeline/osm_data/output_rides_11/osm_maps/episode_{ep:04d}_seg{seg:02d}/osm_map_{fi:06d}.png` |
| filtered_position | Arrow column `observation.filtered_position` (float32, 2) |
| filtered_heading | Arrow column `observation.filtered_heading` (float32) |
| segment lookup | `osm_pipeline/osm_data/output_rides_11/episode_scores.json` |
| split logic | `osm_pipeline/py/episode_selector.py::split_into_segments()` |
