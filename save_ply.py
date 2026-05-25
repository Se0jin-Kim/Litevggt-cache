"""
PLY 포인트 클라우드 저장 스크립트

사용법:
    TORCH_COMPILE_DISABLE=1 python save_ply.py \
        --scene_dir /workspace/litevggt/Litevggt-cache/litevggt_dataset/scannet/scene0025_01 \
        --ckpt_path /workspace/litevggt/Litevggt-cache/te_dict.pt \
        --output_dir ./ply_results \
        --max_frames 96 \
        --conf_thresh 3.0 \
        --mode both
"""

import os, sys, argparse
import numpy as np
import torch
from pathlib import Path

os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
sys.path.insert(0, "/workspace/litevggt/Litevggt-cache")

import transformer_engine.pytorch as te
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images


def save_ply(path, points, colors=None):
    """numpy array → ply 파일 저장"""
    N = points.shape[0]
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {N}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        if colors is not None:
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for i in range(N):
            line = f"{points[i,0]:.6f} {points[i,1]:.6f} {points[i,2]:.6f}"
            if colors is not None:
                r, g, b = int(colors[i,0]*255), int(colors[i,1]*255), int(colors[i,2]*255)
                line += f" {r} {g} {b}"
            f.write(line + "\n")
    print(f"  저장 완료: {path}  ({N:,}점)")


def run(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 모델 로드
    print("모델 로드 중...")
    model = VGGT().to(device)
    ckpt = torch.load(args.ckpt_path, map_location=device)
    state_dict = ckpt.get("state_dict", ckpt.get("model", ckpt))
    model.load_state_dict(state_dict, strict=False)
    model = model.to(torch.bfloat16).eval()
    print("완료")

    # 이미지 로드
    scene_dir = Path(args.scene_dir)
    color_dir = scene_dir / "color_96"
    if not color_dir.exists():
        color_dir = scene_dir / "color"
    image_paths = sorted(list(color_dir.glob("*.jpg")) + list(color_dir.glob("*.png")))
    image_paths = image_paths[:args.max_frames]
    print(f"{len(image_paths)}장 로드 중...")

    images = load_and_preprocess_images([str(p) for p in image_paths]).to(device)
    N = images.shape[1] if images.ndim == 5 else images.shape[0]
    pw = images.shape[-1] // 14
    ph = images.shape[-2] // 14
    model.update_patch_dimensions(pw, ph)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scene_name = scene_dir.name

    def infer(use_adaptive_cache, label):
        import time
        # aggregator.forward에 직접 플래그 전달
        orig_fwd = model.aggregator.forward
        _flag = use_adaptive_cache  # 클로저 캡처용

        def patched_fwd(img, **kw):
            # 재귀 방지: 원본 함수 직접 호출
            return orig_fwd.__func__(model.aggregator, img,
                                     use_adaptive_cache=_flag)                    if hasattr(orig_fwd, '__func__')                    else orig_fwd(img, use_adaptive_cache=_flag)

        model.aggregator.forward = patched_fwd
        torch.cuda.synchronize()
        t0 = time.time()
        try:
            with torch.no_grad():
                with te.fp8_autocast(enabled=False):
                    pred = model(images)
        finally:
            model.aggregator.forward = orig_fwd
        torch.cuda.synchronize()
        elapsed = time.time() - t0
        print(f"  [{label}] 추론 완료: {elapsed:.2f}s")
        return pred, elapsed

    modes = []
    if args.mode in ("original", "both"):
        modes.append((False, "original"))
    if args.mode in ("adaptive", "both"):
        modes.append((True, "adaptive"))

    for flag, label in modes:
        print(f"\n[{label}] 포인트 클라우드 생성 중...")
        pred, elapsed = infer(flag, label)

        # world_points: [1, N, H, W, 3]
        # world_points_conf: [1, N, H, W]
        wpts  = pred["world_points"][0].float().cpu().numpy()    # [N, H, W, 3]
        wconf = pred["world_points_conf"][0].float().cpu().numpy() # [N, H, W]
        imgs  = pred["images"][0].float().cpu().numpy()           # [N, 3, H, W]

        all_points = []
        all_colors = []

        for i in range(wpts.shape[0]):
            pts  = wpts[i].reshape(-1, 3)    # [H*W, 3]
            conf = wconf[i].reshape(-1)       # [H*W]
            # 색상: [3, H, W] → [H*W, 3]
            rgb = imgs[i].transpose(1, 2, 0).reshape(-1, 3)

            # confidence 필터링
            mask = conf > args.conf_thresh
            pts  = pts[mask]
            rgb  = rgb[mask]

            all_points.append(pts)
            all_colors.append(rgb)

        all_points = np.concatenate(all_points, axis=0)
        all_colors = np.concatenate(all_colors, axis=0)
        all_colors = np.clip(all_colors, 0, 1)

        ply_path = out_dir / f"{scene_name}_{label}.ply"
        save_ply(str(ply_path), all_points, all_colors)

    print(f"\n결과 저장 위치: {out_dir}")
    print("PLY 파일은 MeshLab, CloudCompare, Open3D 등으로 열 수 있습니다.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_dir", type=str,
        default="/workspace/litevggt/Litevggt-cache/litevggt_dataset/scannet/scene0025_01")
    parser.add_argument("--ckpt_path", type=str,
        default="/workspace/litevggt/Litevggt-cache/te_dict.pt")
    parser.add_argument("--output_dir", type=str, default="./ply_results")
    parser.add_argument("--max_frames", type=int, default=96)
    parser.add_argument("--conf_thresh", type=float, default=3.0,
        help="world_points_conf 임계값 (높을수록 노이즈 적음)")
    parser.add_argument("--mode", choices=["original", "adaptive", "both"],
        default="both", help="어떤 모드로 생성할지")
    args = parser.parse_args()
    run(args)
