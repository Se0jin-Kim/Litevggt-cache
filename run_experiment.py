"""
Adaptive Cache Scheduling 실험 스크립트
"""

import os
import sys
import time
import json
import argparse
import numpy as np
import torch
import warnings
import logging
from pathlib import Path

os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ""))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

logging.getLogger("dinov2").setLevel(logging.WARNING)
warnings.filterwarnings("ignore")

import transformer_engine.pytorch as te
from transformer_engine.common.recipe import Format, DelayedScaling
from vggt.models.vggt import VGGT
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import closed_form_inverse_se3
from vggt.utils.rotation import mat_to_quat

torch.set_float32_matmul_precision("highest")
torch.backends.cudnn.allow_tf32 = False


def load_model(ckpt_path, device):
    print(f"모델 로드: {ckpt_path}")
    model = VGGT().to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt.get("state_dict", ckpt.get("model", ckpt))
    model.load_state_dict(state_dict, strict=False)
    model = model.to(torch.bfloat16).eval()
    print("모델 로드 완료")
    return model


def rotation_angle(rot_gt, rot_pred, eps=1e-15):
    q_pred = mat_to_quat(rot_pred)
    q_gt   = mat_to_quat(rot_gt)
    loss_q = (1 - (q_pred * q_gt).sum(dim=1) ** 2).clamp(min=eps)
    return torch.arccos(1 - 2 * loss_q) * 180 / np.pi


def translation_angle(tvec_gt, tvec_pred, eps=1e-15):
    def norm(v): return v / (torch.norm(v, dim=1, keepdim=True) + eps)
    loss_t = torch.clamp_min(1.0 - (norm(tvec_pred) * norm(tvec_gt)).sum(dim=1) ** 2, eps)
    deg = torch.acos(torch.sqrt(1 - loss_t)) * 180 / np.pi
    return torch.min(deg, (180 - deg).abs())


def build_pair_index(N):
    return torch.combinations(torch.arange(N), 2).unbind(-1)


def compute_pose_errors(pred_se3, gt_se3, N):
    i1, i2 = build_pair_index(N)
    rel_gt   = gt_se3[i1].bmm(closed_form_inverse_se3(gt_se3[i2]))
    rel_pred = pred_se3[i1].bmm(closed_form_inverse_se3(pred_se3[i2]))
    r_err = rotation_angle(rel_gt[:, :3, :3], rel_pred[:, :3, :3])
    t_err = translation_angle(rel_gt[:, :3, 3], rel_pred[:, :3, 3])
    return r_err.cpu().numpy(), t_err.cpu().numpy()


def calculate_auc(r_err, t_err, thresholds=(5, 15, 30)):
    max_err = np.max(np.stack([r_err, t_err], axis=1), axis=1)
    results = {}
    for thr in thresholds:
        hist, _ = np.histogram(max_err, bins=np.arange(thr + 1))
        results[f"AUC@{thr}"] = float(np.mean(np.cumsum(hist / len(max_err))))
    return results


def run_inference(model, images, use_adaptive_cache):
    fp8_recipe = DelayedScaling(
        fp8_format=Format.E4M3,
        amax_history_len=80,
        amax_compute_algo="max",
    )
    # use_adaptive_cache 플래그 주입
    original_forward = model.aggregator.forward
    def patched_forward(img, **kw):
        return original_forward(img, use_adaptive_cache=use_adaptive_cache)
    model.aggregator.forward = patched_forward

    torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        with te.fp8_autocast(enabled=False):
            predictions = model(images)
    torch.cuda.synchronize()
    elapsed = time.time() - t0

    model.aggregator.forward = original_forward
    return predictions, elapsed


def evaluate_scene_dtu(model, sample, device):
    scene   = sample["scene"]
    images  = sample["imgs"].to(device)[:48]
    gt_poses = sample["poses"][:48]
    N = images.shape[0]

    pw = images.shape[-1] // 14
    ph = images.shape[-2] // 14
    model.update_patch_dimensions(pw, ph)

    scene_results = {}

    for mode, flag in [("original", False), ("adaptive", True)]:
        recompute_log = []

        if flag:
            from vggt.merging.adaptive_cache import AdaptiveCacheScheduler
            orig_fn = AdaptiveCacheScheduler.should_recompute
            def patched(self, layer_idx, x, b_idx):
                result, reason = orig_fn(self, layer_idx, x, b_idx)
                drift = None
                if self.cached_dst_feat is not None and b_idx is not None:
                    import torch.nn.functional as F
                    dst_now = x[:, b_idx[0, :, 0], :]
                    cos = F.cosine_similarity(dst_now, self.cached_dst_feat, dim=-1).mean().item()
                    drift = round(1.0 - cos, 4)
                recompute_log.append((layer_idx, reason, drift))
                return result, reason
            AdaptiveCacheScheduler.should_recompute = patched

        try:
            predictions, elapsed = run_inference(model, images, flag)
        finally:
            if flag:
                AdaptiveCacheScheduler.should_recompute = orig_fn

        with torch.amp.autocast("cuda", dtype=torch.float64):
            extrinsic, _ = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
            pred_se3_raw = extrinsic[0]
            add_row = torch.tensor([0, 0, 0, 1], device=device).expand(N, 1, 4)
            pred_se3 = torch.cat((pred_se3_raw, add_row), dim=1)
            gt_se3_np = gt_poses.numpy() if isinstance(gt_poses, torch.Tensor) else gt_poses
            gt_se3 = torch.from_numpy(gt_se3_np).to(device)
            if gt_se3.shape[1] == 3:
                gt_se3 = torch.cat((gt_se3, add_row), dim=1)

        r_err, t_err = compute_pose_errors(pred_se3.double(), gt_se3.double(), N)
        auc = calculate_auc(r_err, t_err)
        auc["inference_time_s"] = elapsed

        recompute_count = sum(1 for _, r, _ in recompute_log if r != "cache_hit") if recompute_log else 0
        drift_vals = [d for _, _, d in recompute_log if d is not None] if recompute_log else []

        print(f"  [{mode:8s}] AUC@30={auc['AUC@30']:.4f}  AUC@15={auc['AUC@15']:.4f}  "
              f"시간={elapsed:.2f}s", end="")
        if flag and recompute_log:
            print(f"  재계산={recompute_count}/24  "
                  f"drift_avg={np.mean(drift_vals):.4f}" if drift_vals else "", end="")
        print()

        scene_results[mode] = auc

    return scene, scene_results


def run_experiment(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = load_model(args.ckpt_path, device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = {"original": {}, "adaptive": {}}

    if args.dataset == "dtu":
        print("\n" + "="*60)
        print("DTU 데이터셋 실험")
        print("="*60)
        sys.path.insert(0, str(Path(__file__).parent / "eval"))
        from data import DTUDataset
        dataset = DTUDataset(root_dir=str(args.data_dir))

        for sample in dataset:
            scene = sample["scene"]
            print(f"\n[{scene}]  {min(sample['imgs'].shape[0], 48)}장")
            try:
                scene_name, scene_res = evaluate_scene_dtu(model, sample, device)
                for mode in ("original", "adaptive"):
                    all_results[mode][scene_name] = scene_res[mode]
            except Exception as e:
                print(f"  오류: {e}")
                import traceback; traceback.print_exc()

    # ── ScanNet ──
    elif args.dataset == "scannet":
        print("\n" + "="*60)
        print("ScanNet 데이터셋 실험")
        print("="*60)

        from vggt.utils.load_fn import load_and_preprocess_images

        data_dir = Path(args.data_dir)
        scenes = []
        for scene_dir in sorted(data_dir.iterdir()):
            if not scene_dir.is_dir():
                continue
            color_dir = scene_dir / "color_96"
            if not color_dir.exists():
                color_dir = scene_dir / "color"
            if not color_dir.exists():
                continue
            image_paths = sorted(list(color_dir.glob("*.jpg")) + list(color_dir.glob("*.png")))
            if len(image_paths) < 10:
                continue
            image_paths = image_paths[:args.max_frames]
            scenes.append({"name": scene_dir.name, "image_paths": image_paths,
                           "pose_dir": scene_dir / "pose"})

        print(f"{len(scenes)}개 씬 발견")

        for scene_info in scenes:
            scene = scene_info["name"]
            image_paths = [str(p) for p in scene_info["image_paths"]]
            pose_dir = scene_info["pose_dir"]
            print(f"\n[{scene}]  {len(image_paths)}장")

            try:
                images = load_and_preprocess_images(image_paths).to(device)
            except Exception as e:
                print(f"  이미지 로드 실패: {e}")
                continue

            N = images.shape[0]
            model.update_patch_dimensions(images.shape[-1]//14, images.shape[-2]//14)

            # GT 포즈 로드
            gt_poses = []
            valid = True
            for img_path in image_paths:
                stem = Path(img_path).stem
                pose_file = pose_dir / f"{stem}.txt"
                if not pose_file.exists():
                    valid = False
                    break
                pose = np.loadtxt(str(pose_file)).astype(np.float32)
                if pose.shape != (4, 4) or np.any(np.isnan(pose)) or np.any(np.isinf(pose)):
                    valid = False
                    break
                gt_poses.append(pose)

            if not valid or len(gt_poses) != N:
                print(f"  포즈 로드 실패 — 속도만 측정")
                gt_poses = None

            for mode, flag in [("original", False), ("adaptive", True)]:
                recompute_log = []
                if flag:
                    from vggt.merging.adaptive_cache import AdaptiveCacheScheduler
                    orig_fn = AdaptiveCacheScheduler.should_recompute
                    def patched(self, layer_idx, x, b_idx):
                        result, reason = orig_fn(self, layer_idx, x, b_idx)
                        recompute_log.append((layer_idx, reason))
                        return result, reason
                    AdaptiveCacheScheduler.should_recompute = patched

                orig_fwd = model.aggregator.forward
                def patched_fwd(img, **kw):
                    return orig_fwd(img, use_adaptive_cache=flag)
                model.aggregator.forward = patched_fwd

                import transformer_engine.pytorch as te
                torch.cuda.synchronize()
                t0 = time.time()
                with torch.no_grad():
                    with te.fp8_autocast(enabled=False):
                        predictions = model(images)
                torch.cuda.synchronize()
                elapsed = time.time() - t0

                model.aggregator.forward = orig_fwd
                if flag:
                    AdaptiveCacheScheduler.should_recompute = orig_fn

                metrics = {"inference_time_s": elapsed}

                if gt_poses is not None:
                    try:
                        with torch.amp.autocast("cuda", dtype=torch.float64):
                            from vggt.utils.pose_enc import pose_encoding_to_extri_intri
                            extrinsic, _ = pose_encoding_to_extri_intri(
                                predictions["pose_enc"], images.shape[-2:])
                            pred_se3_raw = extrinsic[0]
                            add_row = torch.tensor([0,0,0,1], device=device).expand(N,1,4)
                            pred_se3 = torch.cat((pred_se3_raw, add_row), dim=1)
                            gt_np = np.stack(gt_poses, axis=0)
                            gt_se3 = torch.from_numpy(gt_np).to(device)
                            if gt_se3.shape[1] == 3:
                                gt_se3 = torch.cat((gt_se3, add_row), dim=1)
                        r_err, t_err = compute_pose_errors(pred_se3.double(), gt_se3.double(), N)
                        auc = calculate_auc(r_err, t_err)
                        metrics.update(auc)
                    except Exception as e:
                        print(f"  포즈 오차 계산 실패: {e}")

                all_results[mode][scene] = metrics

                recompute_count = sum(1 for _, r in recompute_log if r != "cache_hit") if recompute_log else 0
                auc30_str = f"  AUC@30={metrics['AUC@30']:.4f}" if "AUC@30" in metrics else ""
                print(f"  [{mode:8s}] 시간={elapsed:.2f}s{auc30_str}", end="")
                if flag and recompute_log:
                    print(f"  재계산={recompute_count}/24", end="")
                print()

    # ── 요약 ──────────────────────────────────────────────────────────────────
    common = set(all_results["original"]) & set(all_results["adaptive"])
    print("\n" + "="*60)
    print(f"결과 요약  ({len(common)}개 씬)")
    print("="*60)

    if common:
        def mean_metric(mode, key):
            return np.mean([all_results[mode][s][key] for s in common])

        for key in ("AUC@30", "AUC@15", "AUC@5"):
            o = mean_metric("original", key)
            a = mean_metric("adaptive", key)
            print(f"  {key}  기존={o:.4f}  적응형={a:.4f}  차이={a-o:+.4f}")

        ot = mean_metric("original", "inference_time_s")
        at = mean_metric("adaptive", "inference_time_s")
        print(f"  시간    기존={ot:.2f}s  적응형={at:.2f}s  속도향상={ot/at:.2f}x")

        all_results["summary"] = {
            "num_scenes": len(common),
            "speedup": ot / at,
            **{f"orig_{k}":  mean_metric("original", k) for k in ("AUC@30","AUC@15","AUC@5")},
            **{f"adap_{k}":  mean_metric("adaptive", k) for k in ("AUC@30","AUC@15","AUC@5")},
            **{f"delta_{k}": mean_metric("adaptive", k) - mean_metric("original", k)
               for k in ("AUC@30","AUC@15","AUC@5")},
        }

    out_file = out_dir / f"results_{args.dataset}.json"
    with open(out_file, "w") as f:
        def cvt(o):
            if isinstance(o, (np.floating, np.integer)): return float(o)
            if isinstance(o, np.ndarray): return o.tolist()
            raise TypeError
        json.dump(all_results, f, indent=2, default=cvt)
    print(f"\n결과 저장: {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["dtu", "scannet"], required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--ckpt_path", type=str,
                        default="/workspace/litevggt/Litevggt-cache/te_dict.pt")
    parser.add_argument("--output_dir", type=str, default="./experiment_results")
    parser.add_argument("--max_frames", type=int, default=96)
    args = parser.parse_args()
    run_experiment(args)
