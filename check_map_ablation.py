"""
map=zeros / map=noise / map=shuffled 일 때 ADE/FDE 변화로
OSM map이 실제로 쓰이는지 확인.
"""
import sys, math
import numpy as np
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F

sys.path.insert(0, "/media/ms/WD_BLACK_4TB/OmniVLA/OmniVLA/inference")
from model_omnivla_edge_odom import OmniVLA_edge_odom

sys.path.insert(0, "/media/ms/WD_BLACK_4TB/Learning-to-Drive-Anywhere-with-MBRA/osm_pipeline/py")
from rides11_dataset import Rides11Dataset, METRIC_WAYPOINT_SPACING

BASE = Path("/media/ms/WD_BLACK_4TB/Learning-to-Drive-Anywhere-with-MBRA")

CKPT    = BASE / "checkpoints/omnivla_edge_rides11_odom/best.pth"
ARROW   = BASE / "FrodoBots-2K/processed/output_rides_11/train/data-00000-of-00001.arrow"
SCORES  = BASE / "osm_pipeline/osm_data/output_rides_11/episode_scores.json"
OSM_ROOT= BASE / "osm_pipeline/osm_data/output_rides_11/osm_maps_arrow"
VID_ROOT= BASE / "FrodoBots-2K/processed/output_rides_11"
OUT_DIR = BASE / "attention_analysis"
OUT_DIR.mkdir(exist_ok=True)

DEVICE = torch.device("cpu")
BS = 16
N_BATCHES = 10   # 10×16 = 160 samples

# ── load model ────────────────────────────────────────────────────────────────
model = OmniVLA_edge_odom(
    context_size=5, len_traj_pred=8, learn_angle=True,
    obs_encoder="efficientnet-b0", obs_encoding_size=1024,
    late_fusion=False, mha_num_attention_heads=4,
    mha_num_attention_layers=4, mha_ff_dim_factor=4,
)
ckpt = torch.load(CKPT, map_location="cpu")
model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=True)
model.eval()
print("Model loaded.")

# ── load dataset ──────────────────────────────────────────────────────────────
ds = Rides11Dataset(str(ARROW), str(SCORES), str(OSM_ROOT), str(VID_ROOT))
indices = np.random.choice(len(ds), BS * N_BATCHES, replace=False)
print(f"Using {len(indices)} samples ({N_BATCHES} batches of {BS})")

# ── helpers ───────────────────────────────────────────────────────────────────
def make_inputs(batch):
    obs_stack  = torch.stack([b["obs_stack"]    for b in batch])   # (B,18,96,96)
    map_images = torch.stack([b["map_images"]   for b in batch])   # (B,3,96,96)
    gt_wp      = torch.stack([b["gt_waypoints"] for b in batch])   # (B,8,2)
    B = obs_stack.shape[0]
    obs_cur   = obs_stack[:, -3:]
    goal_pose = torch.zeros(B, 4)
    goal_img  = obs_cur
    goal_mask = torch.zeros(B, dtype=torch.long)
    feat_text = torch.zeros(B, 512)
    cur_img   = F.interpolate(obs_cur, (224, 224), mode='bilinear', align_corners=False)
    return obs_stack, goal_pose, map_images, goal_img, goal_mask, feat_text, cur_img, gt_wp

def ade_fde(pred, gt):
    p = pred.detach().numpy() * METRIC_WAYPOINT_SPACING
    g = gt.detach().numpy()   * METRIC_WAYPOINT_SPACING
    d = np.sqrt(((p - g)**2).sum(-1))   # (B,8)
    return d.mean(-1).mean(), d[:,-1].mean()

# ── run ───────────────────────────────────────────────────────────────────────
results = {k: {"ade": [], "fde": []} for k in ["normal","zeros","noise","shuffled"]}

for i in range(N_BATCHES):
    batch = [ds[int(j)] for j in indices[i*BS:(i+1)*BS]]
    obs, gp, map_img, gi, gm, ft, ci, gt = make_inputs(batch)

    with torch.no_grad():
        # normal
        pred, _, _ = model(obs, gp, map_img, gi, gm, ft, ci)
        a, f = ade_fde(pred[:,:,:2], gt); results["normal"]["ade"].append(a); results["normal"]["fde"].append(f)
        # zeros
        pred, _, _ = model(obs, gp, torch.zeros_like(map_img), gi, gm, ft, ci)
        a, f = ade_fde(pred[:,:,:2], gt); results["zeros"]["ade"].append(a); results["zeros"]["fde"].append(f)
        # noise
        pred, _, _ = model(obs, gp, torch.randn_like(map_img), gi, gm, ft, ci)
        a, f = ade_fde(pred[:,:,:2], gt); results["noise"]["ade"].append(a); results["noise"]["fde"].append(f)
        # shuffled (배치 내 다른 샘플 맵)
        pred, _, _ = model(obs, gp, map_img[torch.randperm(BS)], gi, gm, ft, ci)
        a, f = ade_fde(pred[:,:,:2], gt); results["shuffled"]["ade"].append(a); results["shuffled"]["fde"].append(f)

    if (i+1) % 5 == 0:
        print(f"  {i+1}/{N_BATCHES} batches done")

# ── print results ─────────────────────────────────────────────────────────────
print("\n" + "─"*55)
print(f"{'Condition':<12}  {'ADE(m)':>8}  {'FDE(m)':>8}  {'ΔADE':>10}  {'ΔADE%':>7}")
print("─"*55)
base_ade = np.mean(results["normal"]["ade"])
base_fde = np.mean(results["normal"]["fde"])
print(f"{'normal':<12}  {base_ade:8.4f}  {base_fde:8.4f}  {'(baseline)':>10}")
for cond in ["zeros","noise","shuffled"]:
    a = np.mean(results[cond]["ade"])
    f = np.mean(results[cond]["fde"])
    da = a - base_ade
    pct = da / base_ade * 100
    print(f"{'map='+cond:<12}  {a:8.4f}  {f:8.4f}  {da:+10.4f}  {pct:+6.1f}%")
print("─"*55)

delta = np.mean(results["zeros"]["ade"]) - base_ade
if delta > 0.005:
    print(f"\n✓ MAP이 사용되고 있음: zeros 교체 시 ADE +{delta:.4f}m 악화")
else:
    print(f"\n⚠ MAP 영향 미미: ADE 변화 {delta:+.4f}m — 학습이 더 필요하거나 map 활용 안 됨")

# ── bar chart ─────────────────────────────────────────────────────────────────
conds  = ["normal","zeros","noise","shuffled"]
ades   = [np.mean(results[c]["ade"]) for c in conds]
colors = ["#4CAF50","#9E9E9E","#F44336","#FF9800"]

fig, ax = plt.subplots(figsize=(6,4))
bars = ax.bar(conds, ades, color=colors, edgecolor='white', width=0.5)
ax.axhline(ades[0], color='green', ls='--', lw=1, alpha=0.6, label=f"normal={ades[0]:.4f}")
ax.set_ylabel("ADE (m)")
ax.set_title("Map Ablation Test\n(map을 교체했을 때 ADE 변화)")
for bar, v in zip(bars, ades):
    ax.text(bar.get_x()+bar.get_width()/2, v+0.001, f"{v:.4f}",
            ha='center', va='bottom', fontsize=9)
ax.legend(fontsize=8)
fig.tight_layout()
out = OUT_DIR / "ablation_result.png"
fig.savefig(out, dpi=150)
plt.close()
print(f"\nSaved: {out}")
