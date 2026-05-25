"""
run_chunked.py
--------------
총 N장의 이미지를 chunk_size장씩 균등 분할하여 순차 추론 후
모든 청크의 PLY 포인트 클라우드를 하나로 합치는 스크립트.

사용 예시:
    python run_chunked.py \
        --ckpt_path /workspace/litevggt/Litevggt-cache/LiteVGGT-repo/te_dict.pt \
        --img_dir   /workspace/litevggt/Litevggt-cache/LiteVGGT-repo/litevggt_dataset/scannet1/color \
        --output_dir ./recon_result_scannet1_600 \
        --total_frames 600 \
        --chunk_size 192
"""

import os
import time
import json
import argparse
import subprocess
import numpy as np


# ── PLY I/O helpers ──────────────────────────────────────────────────────────

def read_ply(path):
    """ASCII PLY를 읽어 (N,3) points, (N,3) colors 반환."""
    with open(path, "r") as f:
        lines = f.readlines()

    # 헤더 끝 위치 파악
    header_end = 0
    n_vertex = 0
    for i, line in enumerate(lines):
        if line.startswith("element vertex"):
            n_vertex = int(line.strip().split()[-1])
        if line.strip() == "end_header":
            header_end = i + 1
            break

    points = []
    colors = []
    for line in lines[header_end: header_end + n_vertex]:
        vals = line.strip().split()
        points.append([float(vals[0]), float(vals[1]), float(vals[2])])
        colors.append([int(vals[3]),   int(vals[4]),   int(vals[5])])

    return np.array(points, dtype=np.float32), np.array(colors, dtype=np.uint8)


def write_ply(path, points, colors):
    """(N,3) points + (N,3) uint8 colors를 ASCII PLY로 저장."""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for p, c in zip(points, colors):
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {c[0]} {c[1]} {c[2]}\n")
    print(f"✅ Merged PLY saved: {path} ({len(points)} points)")


# ── 이미지 목록 수집 ─────────────────────────────────────────────────────────

def collect_images(img_dir):
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".gif"}
    files = sorted([
        f for f in os.listdir(img_dir)
        if os.path.splitext(f)[1].lower() in exts
    ])
    return [os.path.join(img_dir, f) for f in files]


# ── 임시 img_dir 생성 (symlink) ──────────────────────────────────────────────

def make_chunk_dir(all_images, indices, tmp_base, chunk_idx):
    """청크에 해당하는 이미지만 담은 임시 디렉토리를 심볼릭 링크로 구성."""
    chunk_dir = os.path.join(tmp_base, f"chunk_{chunk_idx:03d}")
    os.makedirs(chunk_dir, exist_ok=True)
    for rank, idx in enumerate(indices):
        src = all_images[idx]
        ext = os.path.splitext(src)[1]
        dst = os.path.join(chunk_dir, f"{rank:06d}{ext}")
        if not os.path.exists(dst):
            os.symlink(src, dst)
    return chunk_dir


# ── 메인 ─────────────────────────────────────────────────────────────────────

def get_args():
    parser = argparse.ArgumentParser("run_chunked — LiteVGGT chunked inference")
    parser.add_argument("--ckpt_path",    type=str, required=True)
    parser.add_argument("--img_dir",      type=str, required=True)
    parser.add_argument("--output_dir",   type=str, required=True)
    parser.add_argument("--total_frames", type=int, default=300,
                        help="전체에서 균등 샘플링할 프레임 수 (300/600/900/1200)")
    parser.add_argument("--chunk_size",   type=int, default=192,
                        help="한 번에 GPU에 올릴 최대 프레임 수 (기본 192)")
    parser.add_argument("--keep_ratio",   type=float, default=0.42)
    parser.add_argument("--device",       type=str,   default="cuda:0")
    parser.add_argument("--run_demo",     type=str,
                        default="/workspace/litevggt/Litevggt-cache/LiteVGGT-repo/run_demo.py",
                        help="run_demo.py 경로")
    return parser.parse_args()


def main():
    args = get_args()

    # 1) 전체 이미지 목록
    all_images = collect_images(args.img_dir)
    total_available = len(all_images)
    print(f"📂 총 이미지 수: {total_available}")

    # 2) total_frames 만큼 균등 샘플링
    n = min(args.total_frames, total_available)
    sampled_indices = np.linspace(0, total_available - 1, n, dtype=int).tolist()
    print(f"🎯 균등 샘플링: {n}장 (chunk_size={args.chunk_size})")

    # 3) chunk_size 단위로 분할 (8의 배수 맞춤)
    effective_chunk = (args.chunk_size // 8) * 8
    chunks = []
    for start in range(0, n, effective_chunk):
        chunk = sampled_indices[start: start + effective_chunk]
        if len(chunk) >= 8:          # 8장 미만 청크는 버림
            chunks.append(chunk)
        else:
            print(f"⚠️  마지막 청크 {len(chunk)}장 → 너무 작아서 건너뜀")
    print(f"📦 청크 수: {len(chunks)} (각 최대 {effective_chunk}장)")

    # 4) 임시 디렉토리
    tmp_base = os.path.join(args.output_dir, "_tmp_chunks")
    os.makedirs(tmp_base, exist_ok=True)

    # 5) 청크별 추론
    chunk_ply_paths = []
    chunk_times = []
    total_start = time.time()
    for ci, chunk_indices in enumerate(chunks):
        print(f"\n{'='*50}")
        print(f"🚀 청크 {ci+1}/{len(chunks)}  ({len(chunk_indices)}장)")

        chunk_img_dir  = make_chunk_dir(all_images, chunk_indices, tmp_base, ci)
        chunk_out_dir  = os.path.join(args.output_dir, f"chunk_{ci:03d}")
        chunk_ply_path = os.path.join(chunk_out_dir, "recon.ply")

        # 이미 완료된 청크는 스킵
        if os.path.exists(chunk_ply_path):
            print(f"⏭️  이미 존재 → 스킵: {chunk_ply_path}")
            chunk_ply_paths.append(chunk_ply_path)
            continue

        cmd = [
            "python", args.run_demo,
            "--ckpt_path",  args.ckpt_path,
            "--img_dir",    chunk_img_dir,
            "--output_dir", chunk_out_dir,
            "--keep_ratio", str(args.keep_ratio),
            "--device",     args.device,
        ]
        env = os.environ.copy()
        env["TORCH_COMPILE_DISABLE"]  = "1"
        env["TORCHINDUCTOR_DISABLE"]  = "1"
        env["PYTHONPATH"] = "/workspace/litevggt/Litevggt-cache/LiteVGGT-repo"

        chunk_start_time = time.time()
        result = subprocess.run(cmd, env=env)
        chunk_elapsed = time.time() - chunk_start_time
        chunk_times.append(chunk_elapsed)
        print(f"  ⏱️  청크 {ci+1} 시간: {chunk_elapsed:.2f}s")
        if result.returncode != 0:
            print(f"❌ 청크 {ci+1} 실패! (returncode={result.returncode})")
            continue

        if os.path.exists(chunk_ply_path):
            chunk_ply_paths.append(chunk_ply_path)
            print(f"✅ 청크 {ci+1} 완료: {chunk_ply_path}")
        else:
            print(f"⚠️  청크 {ci+1}: PLY 파일이 생성되지 않았습니다.")

    # 6) PLY 병합
    print(f"\n{'='*50}")
    print(f"🔗 PLY 병합 시작 ({len(chunk_ply_paths)}개 청크)")

    all_points = []
    all_colors = []
    for ply_path in chunk_ply_paths:
        pts, cols = read_ply(ply_path)
        all_points.append(pts)
        all_colors.append(cols)
        print(f"   읽음: {ply_path} ({len(pts)} points)")

    if not all_points:
        print("❌ 병합할 PLY가 없습니다.")
        return

    merged_points = np.concatenate(all_points, axis=0)
    merged_colors = np.concatenate(all_colors, axis=0)
    print(f"📊 병합 전 총 포인트: {len(merged_points)}")

    merged_ply_path = os.path.join(args.output_dir, "recon_merged.ply")
    write_ply(merged_ply_path, merged_points, merged_colors)
    print(f"\n🎉 완료! → {merged_ply_path}")

    # 7) 시간 저장
    total_elapsed = time.time() - total_start
    timing = {
        "total_frames": args.total_frames,
        "chunk_size": args.chunk_size,
        "num_chunks": len(chunk_times),
        "chunk_times_s": chunk_times,
        "total_inference_time_s": sum(chunk_times),
        "total_elapsed_s": total_elapsed,
        "output_dir": args.output_dir,
    }
    timing_path = os.path.join(args.output_dir, "timing.json")
    with open(timing_path, "w") as f:
        json.dump(timing, f, indent=2)
    print(f"⏱️  총 추론 시간: {sum(chunk_times):.2f}s / 전체 소요: {total_elapsed:.2f}s")
    print(f"📄 타이밍 저장: {timing_path}")


if __name__ == "__main__":
    main()