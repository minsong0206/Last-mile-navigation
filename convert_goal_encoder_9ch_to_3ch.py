"""
convert_goal_encoder_9ch_to_3ch.py

OmniVLA-Edge 체크포인트의 goal_encoder (in_channels=9) →
odom map 전용 goal_encoder (in_channels=3) 로 변환.

변환 전략:
  _conv_stem.weight: (32, 9, 3, 3) → (32, 3, 3, 3)
  9ch를 3ch 그룹 3개로 나누어 평균 (채널 의미: OSM(0-2)+OSM(3-5)+obs(6-8))
  나머지 모든 weight: 1:1 복사 (shape 동일)

실행:
  conda run -n mbra python convert_goal_encoder_9ch_to_3ch.py \
    --input  /media/ms/WD_BLACK_4TB/OmniVLA/omnivla-edge/omnivla-edge.pth \
    --output /media/ms/WD_BLACK_4TB/OmniVLA/omnivla-edge/omnivla-edge-odom3ch.pth
"""

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "third_party" / "omnivla" / "inference"))


def convert(input_path: str, output_path: str, verify: bool = True) -> None:
    device = torch.device("cpu")

    print(f"[load] {input_path}")
    state_dict = torch.load(input_path, map_location=device)

    # ── 기존 weight 확인 ──────────────────────────────────────────────────────
    key = "goal_encoder._conv_stem.weight"
    w9 = state_dict[key]
    assert w9.shape == (32, 9, 3, 3), f"예상 shape (32,9,3,3) ≠ {w9.shape}"
    print(f"[before] {key}: {tuple(w9.shape)}")

    # ── 핵심 변환: 3ch 그룹 평균 ─────────────────────────────────────────────
    # 채널 구성: osm_cur(0-2) | osm_cur(3-5) | obs_cur(6-8)
    # 3채널로 압축 → 3개 그룹의 평균
    w3 = (w9[:, 0:3, :, :] + w9[:, 3:6, :, :] + w9[:, 6:9, :, :]) / 3.0
    assert w3.shape == (32, 3, 3, 3)
    print(f"[after]  {key}: {tuple(w3.shape)}")

    # ── 새 state_dict 구성 ───────────────────────────────────────────────────
    new_state_dict = {}
    n_copied = 0
    for k, v in state_dict.items():
        if k == key:
            new_state_dict[k] = w3
        else:
            new_state_dict[k] = v
            n_copied += 1

    print(f"[info] {n_copied} keys copied unchanged, 1 key converted")

    # ── 저장 ─────────────────────────────────────────────────────────────────
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(new_state_dict, output_path)
    print(f"[save] {output_path}")

    # ── 검증: 3ch 모델에 실제 로드 ───────────────────────────────────────────
    if verify:
        from model_omnivla_edge import OmniVLA_edge
        from efficientnet_pytorch import EfficientNet

        # goal_encoder를 3ch로 패치한 모델 인스턴스 생성
        model = OmniVLA_edge(
            context_size=5,
            len_traj_pred=8,
            learn_angle=True,
            obs_encoder="efficientnet-b0",
            obs_encoding_size=1024,
            late_fusion=False,
            mha_num_attention_heads=4,
            mha_num_attention_layers=4,
            mha_ff_dim_factor=4,
        )
        # goal_encoder를 3ch로 교체 (로드 전에 shape 맞춤)
        model.goal_encoder = EfficientNet.from_name("efficientnet-b0", in_channels=3)

        missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
        if missing:
            print(f"[warn] missing keys ({len(missing)}): {missing[:5]}")
        if unexpected:
            print(f"[warn] unexpected keys ({len(unexpected)}): {unexpected[:5]}")

        # goal_encoder._conv_stem.weight 값 직접 비교
        loaded_w = dict(model.named_parameters())["goal_encoder._conv_stem.weight"]
        assert torch.allclose(loaded_w, w3), "로드된 weight가 변환값과 다름"
        print("[verify] weight load OK ✓")

        # 더미 forward 테스트 — 모델 forward 내부가 get_device() 사용 (CPU=-1 버그)
        # GPU 있을 때만 실행
        if torch.cuda.is_available():
            dev = torch.device("cuda:0")
            model = model.to(dev).eval()
            with torch.no_grad():
                B = 2
                dummy_obs      = torch.zeros(B, 18, 96, 96,   device=dev)
                dummy_goal     = torch.zeros(B, 4,             device=dev)
                dummy_map      = torch.zeros(B, 3, 96, 96,    device=dev)  # ← 3ch
                dummy_goal_img = torch.zeros(B, 3, 96, 96,    device=dev)
                dummy_mask     = torch.zeros(B, dtype=torch.long, device=dev)
                dummy_text     = torch.zeros(B, 512,           device=dev)
                dummy_cur      = torch.zeros(B, 3, 224, 224,  device=dev)

                out, _, _ = model(
                    dummy_obs, dummy_goal, dummy_map,
                    dummy_goal_img, dummy_mask, dummy_text, dummy_cur,
                )
            assert out.shape == (B, 8, 4), f"출력 shape 오류: {out.shape}"
            print(f"[verify] forward OK — action_pred shape: {tuple(out.shape)}")
        else:
            print("[verify] CUDA 없음 — forward 테스트 스킵 (GPU에서 실행 시 확인)")

        print("[verify] 변환 성공 ✓")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", "-i",
        default="/media/ms/WD_BLACK_4TB/OmniVLA/omnivla-edge/omnivla-edge.pth",
        help="원본 체크포인트 (9ch goal_encoder)",
    )
    parser.add_argument(
        "--output", "-o",
        default="/media/ms/WD_BLACK_4TB/OmniVLA/omnivla-edge/omnivla-edge-odom3ch.pth",
        help="출력 체크포인트 (3ch goal_encoder)",
    )
    parser.add_argument(
        "--no_verify", action="store_true",
        help="forward 검증 스킵 (모델 import 없이 변환만)",
    )
    args = parser.parse_args()

    convert(args.input, args.output, verify=not args.no_verify)


if __name__ == "__main__":
    main()
