"""
check_map_attention.py

OSM map이 모델 학습에 실제로 반영되는지 3가지 방법으로 확인.

방법 1: Ablation — map을 zeros로 교체했을 때 ADE/FDE 변화
방법 2: GradCAM  — OSM map의 어느 영역이 action에 기여하는지 히트맵
방법 3: Attention weights — transformer 각 레이어에서 map token attention 추출

실행 (host mbra 환경):
  cd /media/ms/WD_BLACK_4TB/Learning-to-Drive-Anywhere-with-MBRA
  conda run -n mbra python scripts/analysis/check_map_attention.py \
      --ckpt checkpoints/omnivla_edge_rides11_odom/best.pth \
      --method all          # ablation | gradcam | attention | all
"""

import os, sys, argparse, math
import numpy as np
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "third_party" / "omnivla" / "inference"))

from model_omnivla_edge_odom import OmniVLA_edge_odom

sys.path.insert(0, str(REPO_ROOT / "osm_pipeline" / "py"))
from rides11_dataset import Rides11Dataset, METRIC_WAYPOINT_SPACING

BASE     = Path("/media/ms/WD_BLACK_4TB")
ARROW    = BASE / "Learning-to-Drive-Anywhere-with-MBRA/FrodoBots-2K/processed/output_rides_11/train/data-00000-of-00001.arrow"
SCORES   = BASE / "Learning-to-Drive-Anywhere-with-MBRA/osm_pipeline/osm_data/output_rides_11/episode_scores.json"
OSM_ROOT = BASE / "Learning-to-Drive-Anywhere-with-MBRA/osm_pipeline/osm_data/output_rides_11/osm_maps_arrow"
VIDEO_ROOT = BASE / "Learning-to-Drive-Anywhere-with-MBRA/FrodoBots-2K/processed/output_rides_11"
LABELS   = BASE / "Learning-to-Drive-Anywhere-with-MBRA/dataset_analysis/dataset_labels.json"
OUT_DIR  = BASE / "Learning-to-Drive-Anywhere-with-MBRA/attention_analysis"
OUT_DIR.mkdir(exist_ok=True)

MODEL_PARAMS = dict(
    model_type="omnivla-edge", context_size=5, len_traj_pred=8,
    learn_angle=True, obs_encoder="efficientnet-b0", obs_encoding_size=1024,
    late_fusion=False, mha_num_attention_heads=4, mha_num_attention_layers=4,
    mha_ff_dim_factor=4,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")


# ── Model utils ───────────────────────────────────────────────────────────────

def load_model(ckpt_path):
    model = OmniVLA_edge_odom(
        context_size=MODEL_PARAMS["context_size"],
        len_traj_pred=MODEL_PARAMS["len_traj_pred"],
        learn_angle=MODEL_PARAMS["learn_angle"],
        obs_encoder=MODEL_PARAMS["obs_encoder"],
        obs_encoding_size=MODEL_PARAMS["obs_encoding_size"],
        late_fusion=MODEL_PARAMS["late_fusion"],
        mha_num_attention_heads=MODEL_PARAMS["mha_num_attention_heads"],
        mha_num_attention_layers=MODEL_PARAMS["mha_num_attention_layers"],
        mha_ff_dim_factor=MODEL_PARAMS["mha_ff_dim_factor"],
    )
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=True)
    model.to(DEVICE).eval()
    print(f"Loaded checkpoint: {ckpt_path}")
    return model


def load_dataset(n_samples=200):
    ds = Rides11Dataset(
        arrow_path=str(ARROW),
        scores_path=str(SCORES),
        osm_root=str(OSM_ROOT),
        video_root=str(VIDEO_ROOT),
    )
    # Subsample for speed
    indices = np.random.choice(len(ds), min(n_samples, len(ds)), replace=False).tolist()
    samples = []
    for i in indices:
        try:
            samples.append(ds[i])
        except Exception:
            pass
    print(f"Loaded {len(samples)} samples from dataset")
    return samples


def batch_from_samples(samples, device):
    """
    dataset 반환값(obs_stack, map_images, gt_waypoints)으로
    모델 forward에 필요한 모든 입력 구성.
    finetune_omnivla_edge.py의 prepare_model_inputs()와 동일 로직.
    """
    obs_stack  = torch.stack([s["obs_stack"]  for s in samples]).to(device)  # (B, 18, 96, 96)
    map_images = torch.stack([s["map_images"] for s in samples]).to(device)  # (B, 3, 96, 96)
    gt_wp      = torch.stack([s["gt_waypoints"] for s in samples]).to(device)

    B = obs_stack.shape[0]
    obs_cur    = obs_stack[:, -3:, :, :]                                     # (B, 3, 96, 96)
    goal_pose  = torch.zeros(B, 4, device=device)
    goal_img   = obs_cur
    goal_mask  = torch.zeros(B, dtype=torch.long, device=device)
    feat_text  = torch.zeros(B, 512, device=device)
    cur_img    = F.interpolate(obs_cur, size=(224, 224), mode='bilinear', align_corners=False)

    return obs_stack, goal_pose, map_images, goal_img, goal_mask, feat_text, cur_img, gt_wp


def compute_ade_fde(pred, gt):
    """pred, gt: (B, 8, 2) in normalized units → convert to meters."""
    pred_m = pred.detach().cpu().numpy() * METRIC_WAYPOINT_SPACING
    gt_m   = gt.detach().cpu().numpy() * METRIC_WAYPOINT_SPACING
    diff   = pred_m - gt_m
    dist   = np.sqrt((diff**2).sum(axis=-1))  # (B, 8)
    ade    = dist.mean(axis=-1).mean()         # scalar
    fde    = dist[:, -1].mean()
    return float(ade), float(fde)


# ─────────────────────────────────────────────────────────────────────────────
# 방법 1: Ablation Test
# ─────────────────────────────────────────────────────────────────────────────

def run_ablation(model, samples, n_batch=200):
    """
    Normal map vs. corrupted map → ADE/FDE 비교.
    3가지 corruption: zeros, gaussian noise, shuffled (다른 샘플의 맵)
    """
    print("\n" + "="*60)
    print("방법 1: ABLATION TEST")
    print("="*60)

    import random
    # Build batches of 16
    BS = 16
    results = {"normal": [], "zeros": [], "noise": [], "shuffled": []}

    all_samples = samples[:]
    for start in range(0, len(all_samples) - BS, BS):
        batch = all_samples[start:start+BS]
        obs, goal_pose, map_img, goal_img, goal_mask, feat_text, cur_img, gt_wp = \
            batch_from_samples(batch, DEVICE)

        with torch.no_grad():
            # Normal
            pred, _, _ = model(obs, goal_pose, map_img, goal_img, goal_mask, feat_text, cur_img)
            ade, fde = compute_ade_fde(pred[:, :, :2], gt_wp)
            results["normal"].append((ade, fde))

            # Zeros
            pred_z, _, _ = model(obs, goal_pose, torch.zeros_like(map_img), goal_img, goal_mask, feat_text, cur_img)
            ade, fde = compute_ade_fde(pred_z[:, :, :2], gt_wp)
            results["zeros"].append((ade, fde))

            # Gaussian noise
            noisy = torch.randn_like(map_img)
            pred_n, _, _ = model(obs, goal_pose, noisy, goal_img, goal_mask, feat_text, cur_img)
            ade, fde = compute_ade_fde(pred_n[:, :, :2], gt_wp)
            results["noise"].append((ade, fde))

            # Shuffled map (다른 batch의 맵 사용 — 동일 배치 내 shift)
            shuffled_map = map_img[torch.randperm(BS)]
            pred_s, _, _ = model(obs, goal_pose, shuffled_map, goal_img, goal_mask, feat_text, cur_img)
            ade, fde = compute_ade_fde(pred_s[:, :, :2], gt_wp)
            results["shuffled"].append((ade, fde))

    print(f"\n{'Condition':<12}  {'ADE (m)':>10}  {'FDE (m)':>10}  {'ΔADE vs normal':>14}")
    ade_normal = np.mean([x[0] for x in results["normal"]])
    fde_normal = np.mean([x[1] for x in results["normal"]])
    print(f"{'normal':<12}  {ade_normal:10.4f}  {fde_normal:10.4f}  {'baseline':>14}")
    for cond in ["zeros", "noise", "shuffled"]:
        ade_c = np.mean([x[0] for x in results[cond]])
        fde_c = np.mean([x[1] for x in results[cond]])
        delta = ade_c - ade_normal
        pct   = delta / ade_normal * 100
        print(f"{'map='+cond:<12}  {ade_c:10.4f}  {fde_c:10.4f}  {delta:>+10.4f}m ({pct:+.1f}%)")

    # Bar chart
    conds = ["normal", "zeros", "noise", "shuffled"]
    ades  = [np.mean([x[0] for x in results[c]]) for c in conds]
    colors = ["#4CAF50", "#9E9E9E", "#F44336", "#FF9800"]
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(conds, ades, color=colors, edgecolor='white')
    ax.axhline(ades[0], color='green', ls='--', lw=1, alpha=0.5)
    ax.set_ylabel("ADE (m)")
    ax.set_title("Map Ablation: ADE with different map inputs\n(higher = model was using the map)")
    for bar, val in zip(bars, ades):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f"{val:.4f}", ha='center', va='bottom', fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "ablation_ade.png", dpi=150)
    plt.close(fig)
    print(f"\n  → Saved: {OUT_DIR}/ablation_ade.png")

    delta_zeros = np.mean([x[0] for x in results["zeros"]]) - ade_normal
    if delta_zeros > 0.01:
        print(f"\n  ✓ MAP IS BEING USED: map=zeros가 ADE를 {delta_zeros:+.4f}m 악화시킴")
    else:
        print(f"\n  ⚠ MAP IMPACT WEAK: zeros map과 normal map의 ADE 차이 {delta_zeros:+.4f}m (미미함)")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# 방법 2: GradCAM on goal_encoder (OSM map encoder)
# ─────────────────────────────────────────────────────────────────────────────

class GradCAM:
    """EfficientNet 마지막 conv block에 GradCAM 적용."""
    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.activations = None
        self._hook_handles = []

    def _register_hooks(self):
        target_layer = self.model.goal_encoder._blocks[-1]

        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self._hook_handles.append(target_layer.register_forward_hook(forward_hook))
        self._hook_handles.append(target_layer.register_full_backward_hook(backward_hook))

    def _remove_hooks(self):
        for h in self._hook_handles:
            h.remove()
        self._hook_handles.clear()

    def compute(self, obs, goal_pose, map_img, goal_img, goal_mask, feat_text, cur_img):
        """
        map_img 1장에 대한 GradCAM 반환: (H, W) numpy heatmap (0~1).
        Loss signal: action_pred의 L2 norm (가장 큰 변화를 유발하는 방향).
        """
        self._register_hooks()
        map_img = map_img.clone().requires_grad_(True)

        pred, _, _ = self.model(obs, goal_pose, map_img, goal_img, goal_mask, feat_text, cur_img)
        # action magnitude를 loss로 사용 → 가장 큰 prediction 변화를 유발하는 map 영역 탐색
        loss = pred[:, :, :2].norm()
        self.model.zero_grad()
        loss.backward()

        self._remove_hooks()

        if self.gradients is None or self.activations is None:
            return None

        # GradCAM: channel-wise weighted sum of activations
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (B, C, 1, 1)
        cam = (weights * self.activations).sum(dim=1)             # (B, H, W)
        cam = F.relu(cam)
        cam = cam[0].cpu().numpy()

        if cam.max() > 0:
            cam = cam / cam.max()
        return cam


def run_gradcam(model, samples, n_viz=12):
    """
    Straight / sharp_turn 세그먼트에서 GradCAM 시각화 비교.
    """
    import json
    print("\n" + "="*60)
    print("방법 2: GRADCAM on OSM map")
    print("="*60)

    gradcam = GradCAM(model)

    # scenario별로 샘플 선택
    try:
        labels = {l["seg_key"]: l for l in json.load(open(LABELS))}
    except Exception:
        labels = {}
        print("  (dataset_labels.json 없음 — 전체 샘플에서 랜덤 선택)")

    selected = samples[:n_viz]

    fig, axes = plt.subplots(3, n_viz, figsize=(n_viz * 2.5, 8))
    if n_viz == 1:
        axes = axes[:, np.newaxis]

    for col, sample in enumerate(selected):
        obs_stack, goal_pose, map_img, goal_img, goal_mask, feat_text, cur_img, _ = \
            batch_from_samples([sample], DEVICE)
        obs = obs_stack

        # Row 0: OSM map (원본)
        osm_np = sample["map_images"].permute(1, 2, 0).cpu().numpy()
        # denormalize
        mean = np.array([0.485, 0.456, 0.406])
        std  = np.array([0.229, 0.224, 0.225])
        osm_np = np.clip(osm_np * std + mean, 0, 1)
        axes[0, col].imshow(osm_np)
        axes[0, col].set_xticks([]); axes[0, col].set_yticks([])
        if col == 0: axes[0, col].set_ylabel("OSM map", fontsize=8)

        # Row 1: current obs image
        ctx_np = sample["obs_stack"][-3:].permute(1, 2, 0).cpu().numpy()
        ctx_np = np.clip(ctx_np * std + mean, 0, 1)
        axes[1, col].imshow(ctx_np)
        axes[1, col].set_xticks([]); axes[1, col].set_yticks([])
        if col == 0: axes[1, col].set_ylabel("Current obs", fontsize=8)

        # Row 2: GradCAM on OSM map
        cam = gradcam.compute(obs, goal_pose, map_img, goal_img, goal_mask, feat_text, cur_img)
        if cam is not None:
            cam_resized = np.array(
                __import__('PIL').Image.fromarray((cam * 255).astype(np.uint8)).resize((96, 96), __import__('PIL').Image.BILINEAR)
            ) / 255.0
            axes[2, col].imshow(osm_np)
            axes[2, col].imshow(cam_resized, alpha=0.6, cmap='jet')
        axes[2, col].set_xticks([]); axes[2, col].set_yticks([])
        if col == 0: axes[2, col].set_ylabel("GradCAM", fontsize=8)

    fig.suptitle("GradCAM on OSM map — goal_encoder 마지막 block\n(빨강=high gradient, 파랑=low)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT_DIR / "gradcam_osm.png", dpi=150)
    plt.close(fig)
    print(f"  → Saved: {OUT_DIR}/gradcam_osm.png")


# ─────────────────────────────────────────────────────────────────────────────
# 방법 3: Transformer Attention Weights
# ─────────────────────────────────────────────────────────────────────────────

class AttentionExtractor:
    """
    MultiLayerDecoder_mask3 내부 TransformerEncoder의 각 레이어
    attention weight를 hook으로 추출.

    PyTorch TransformerEncoderLayer는 기본적으로 attn weight를 반환하지 않으므로
    MultiheadAttention의 forward를 monkey-patch하여 weight 저장.
    """
    def __init__(self, model):
        self.model = model
        self.attn_weights = []  # list of (num_layers,) -> (B, n_heads, seq, seq)
        self._handles = []

    def _patch_layer(self, layer, layer_idx):
        orig_attn_forward = layer.self_attn.forward

        extractor = self  # closure reference

        def patched_forward(query, key, value, **kwargs):
            kwargs["need_weights"] = True
            kwargs["average_attn_weights"] = False  # keep per-head
            out, weights = orig_attn_forward(query, key, value, **kwargs)
            if len(extractor.attn_weights) <= layer_idx:
                extractor.attn_weights.extend([None] * (layer_idx + 1 - len(extractor.attn_weights)))
            extractor.attn_weights[layer_idx] = weights.detach().cpu()
            return out, weights

        layer.self_attn.forward = patched_forward

    def register(self):
        decoder = self.model.decoder.sa_decoder
        for i, layer in enumerate(decoder.layers):
            self._patch_layer(layer, i)

    def reset(self):
        self.attn_weights = []


def run_attention(model, samples, n_viz=16):
    """
    각 샘플에서 map token (index 7)에 대한 attention weight 추출.
    시나리오별 평균 map-attention 비교.
    """
    import json
    print("\n" + "="*60)
    print("방법 3: TRANSFORMER ATTENTION WEIGHTS")
    print("="*60)

    extractor = AttentionExtractor(model)
    extractor.register()

    try:
        labels = {l["seg_key"]: l for l in json.load(open(LABELS))}
    except Exception:
        labels = {}

    MAP_TOKEN_IDX = 7  # tokens 순서: 0-5=obs, 6=goal_pose, 7=map, 8=goal_img, 9=lan

    # Active tokens after masking (goal_mask=0): 0-5 obs + 7 map
    # After masking 6,8,9 → effectively positions 0-5 (obs) and 6 (map, originally 7)
    # But mask is applied in attention not token removal, so position 7 is still index 7

    all_map_attn = []  # per-sample: mean attention to map token across all layers and heads
    scenario_attn = {}

    for sample in samples[:200]:
        extractor.reset()
        obs       = sample["obs_img"].unsqueeze(0).to(DEVICE)
        goal_pose = sample["goal_pose"].unsqueeze(0).to(DEVICE)
        map_img   = sample["map_img"].unsqueeze(0).to(DEVICE)
        goal_img  = sample["goal_img"].unsqueeze(0).to(DEVICE)
        goal_mask = sample["goal_mask"].unsqueeze(0).to(DEVICE)
        feat_text = sample["feat_text"].unsqueeze(0).to(DEVICE)
        cur_img   = sample["current_img"].unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            model(obs, goal_pose, map_img, goal_img, goal_mask, feat_text, cur_img)

        if not extractor.attn_weights:
            print("  ⚠ attention weight 추출 실패 — PyTorch 버전 호환 문제일 수 있음")
            break

        # attn_weights: list of (1, n_heads, seq_len, seq_len) per layer
        # 각 obs 토큰이 map 토큰(index 7)에 얼마나 attend하는지 추출
        layer_map_attns = []
        for layer_w in extractor.attn_weights:
            # layer_w: (1, n_heads, seq, seq) — [query, key]
            # obs tokens (0-5) attending to map token (7)
            obs_to_map = layer_w[0, :, :6, MAP_TOKEN_IDX]  # (n_heads, 6)
            layer_map_attns.append(obs_to_map.mean().item())

        mean_map_attn = np.mean(layer_map_attns)
        all_map_attn.append(mean_map_attn)

        # scenario 기반 집계
        seg_key = sample.get("seg_key", "")
        scenario = labels.get(seg_key, {}).get("scenario", "unknown")
        if scenario not in scenario_attn:
            scenario_attn[scenario] = []
        scenario_attn[scenario].append(mean_map_attn)

    if not all_map_attn:
        return

    print(f"\n  전체 평균 map token attention: {np.mean(all_map_attn):.4f}")
    print(f"  (랜덤 uniform 기댓값: 1/{MAP_TOKEN_IDX+1} ≈ {1/10:.4f})")
    if np.mean(all_map_attn) > 1/10 * 1.5:
        print("  ✓ map token이 uniform보다 높은 attention을 받고 있음")
    else:
        print("  ⚠ map token attention이 weak — 학습 초기이거나 map을 잘 활용 못할 수 있음")

    print("\n  시나리오별 map token attention:")
    for sc, vals in sorted(scenario_attn.items(), key=lambda x: -np.mean(x[1])):
        print(f"    {sc:20s}  mean={np.mean(vals):.4f}  n={len(vals)}")

    # Visualization 1: histogram of map attention
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(all_map_attn, bins=30, color="#2196F3", edgecolor='white', alpha=0.8)
    axes[0].axvline(1/10, color='red', ls='--', lw=1.5, label=f"Random baseline (1/10={1/10:.3f})")
    axes[0].axvline(np.mean(all_map_attn), color='green', ls='-', lw=1.5,
                    label=f"Mean={np.mean(all_map_attn):.4f}")
    axes[0].set_xlabel("Avg attention weight → map token")
    axes[0].set_ylabel("# samples")
    axes[0].set_title("Map Token Attention Distribution")
    axes[0].legend(fontsize=8)

    # Visualization 2: scenario comparison
    scenarios = [sc for sc, vals in scenario_attn.items() if len(vals) >= 3]
    means     = [np.mean(scenario_attn[sc]) for sc in scenarios]
    stds      = [np.std(scenario_attn[sc]) for sc in scenarios]
    y_pos     = np.arange(len(scenarios))
    axes[1].barh(y_pos, means, xerr=stds, align='center', color="#FF9800", alpha=0.8,
                 error_kw=dict(ecolor='gray', capsize=3))
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels(scenarios, fontsize=8)
    axes[1].axvline(1/10, color='red', ls='--', lw=1, label="Random")
    axes[1].set_xlabel("Mean attention to map token")
    axes[1].set_title("Map Attention by Scenario\n(교차로/코너에서 높아야 정상)")
    axes[1].legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "transformer_map_attention.png", dpi=150)
    plt.close(fig)
    print(f"\n  → Saved: {OUT_DIR}/transformer_map_attention.png")

    # Visualization 3: per-layer attention to map token
    if extractor.attn_weights:
        extractor.reset()
        obs       = samples[0]["obs_img"].unsqueeze(0).to(DEVICE)
        goal_pose = samples[0]["goal_pose"].unsqueeze(0).to(DEVICE)
        map_img   = samples[0]["map_img"].unsqueeze(0).to(DEVICE)
        goal_img  = samples[0]["goal_img"].unsqueeze(0).to(DEVICE)
        goal_mask = samples[0]["goal_mask"].unsqueeze(0).to(DEVICE)
        feat_text = samples[0]["feat_text"].unsqueeze(0).to(DEVICE)
        cur_img   = samples[0]["current_img"].unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            model(obs, goal_pose, map_img, goal_img, goal_mask, feat_text, cur_img)

        n_layers = len(extractor.attn_weights)
        n_heads  = extractor.attn_weights[0].shape[1]
        seq_len  = extractor.attn_weights[0].shape[2]

        fig, axes = plt.subplots(1, n_layers, figsize=(n_layers * 3, 3.5))
        if n_layers == 1:
            axes = [axes]
        token_labels = [f"obs{i}" for i in range(6)] + ["goal_pos", "MAP", "goal_img", "lan"][:seq_len-6]

        for i, (ax, layer_w) in enumerate(zip(axes, extractor.attn_weights)):
            # Average over heads: (seq, seq)
            avg = layer_w[0].mean(dim=0).numpy()  # (seq, seq)
            im = ax.imshow(avg, cmap='Blues', aspect='auto', vmin=0)
            ax.set_title(f"Layer {i+1}", fontsize=9)
            ax.set_xticks(range(len(token_labels)))
            ax.set_yticks(range(len(token_labels)))
            ax.set_xticklabels(token_labels, rotation=45, ha='right', fontsize=6)
            ax.set_yticklabels(token_labels, fontsize=6)
            # Highlight MAP column
            ax.axvline(MAP_TOKEN_IDX - 0.5, color='red', lw=1.5, alpha=0.7)
            ax.axvline(MAP_TOKEN_IDX + 0.5, color='red', lw=1.5, alpha=0.7)
            plt.colorbar(im, ax=ax, fraction=0.046)

        fig.suptitle("Transformer Self-Attention per layer (head-averaged)\nRed lines = MAP token column",
                     fontsize=10)
        fig.tight_layout()
        fig.savefig(OUT_DIR / "attention_matrix_per_layer.png", dpi=150)
        plt.close(fig)
        print(f"  → Saved: {OUT_DIR}/attention_matrix_per_layer.png")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",   type=str,
                        default="checkpoints/omnivla_edge_rides11_odom/best.pth")
    parser.add_argument("--method", type=str, default="all",
                        choices=["ablation", "gradcam", "attention", "all"])
    parser.add_argument("--n_samples", type=int, default=200)
    args = parser.parse_args()

    model   = load_model(args.ckpt)
    samples = load_dataset(args.n_samples)

    if not samples:
        print("ERROR: dataset 로드 실패")
        return

    if args.method in ("ablation", "all"):
        run_ablation(model, samples)

    if args.method in ("gradcam", "all"):
        run_gradcam(model, samples)

    if args.method in ("attention", "all"):
        run_attention(model, samples)

    print(f"\n모든 결과 저장 위치: {OUT_DIR}")


if __name__ == "__main__":
    main()
