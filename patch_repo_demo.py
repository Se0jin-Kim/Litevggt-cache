path = "/workspace/litevggt/Litevggt-cache/LiteVGGT-repo/run_demo.py"

with open(path, "r") as f:
    lines = f.readlines()

out = []
for line in lines:
    # 1) FP8 비활성화
    line = line.replace(
        "with te.fp8_autocast(enabled=True",
        "with te.fp8_autocast(enabled=False"
    )
    # 2) return parser 바로 앞에 max_frames 인자 삽입 (1회만)
    if line.strip() == "return parser" and "max_frames" not in "".join(out):
        indent = line[: len(line) - len(line.lstrip())]
        out.append(indent + 'parser.add_argument("--max_frames", type=int, default=None, help="max number of frames to use (uniform sampling)")\n')
    # 3) N = len(images) 바로 뒤에 샘플링 로직 삽입 (1회만)
    out.append(line)
    if line.strip() == "N = len(images)" and "max_frames" not in "".join(out[:-1][-20:]):
        out.append("    if args.max_frames is not None and N > args.max_frames:\n")
        out.append("        import numpy as _np\n")
        out.append("        indices = _np.linspace(0, N-1, args.max_frames, dtype=int).tolist()\n")
        out.append("        images = [images[i] for i in indices]\n")
        out.append("        N = len(images)\n")
        out.append('        print(f"Sampled {N} frames from original {N} frames")\n')

with open(path, "w") as f:
    f.writelines(out)

print("Done!")