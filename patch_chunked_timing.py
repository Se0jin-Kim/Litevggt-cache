"""
patch_chunked_timing.py
-----------------------
run_chunked.py에 시간 측정 및 JSON 저장 기능을 추가하는 패치.
output_dir/timing.json 에 저장됨.
"""

path = "/workspace/litevggt/Litevggt-cache/run_chunked.py"

with open(path, "r") as f:
    code = f.read()

# 1) import time, json 추가
if "import time" not in code:
    code = code.replace("import os\n", "import os\nimport time\nimport json\n")

# 2) 청크 루프 시작 전에 타이머 시작
old = "    # 5) 청크별 추론\n    chunk_ply_paths = []"
new = """    # 5) 청크별 추론
    chunk_ply_paths = []
    chunk_times = []
    total_start = time.time()"""
code = code.replace(old, new)

# 3) subprocess.run 앞뒤로 시간 측정
old = "        result = subprocess.run(cmd, env=env)"
new = """        chunk_start_time = time.time()
        result = subprocess.run(cmd, env=env)
        chunk_elapsed = time.time() - chunk_start_time
        chunk_times.append(chunk_elapsed)
        print(f"  ⏱️  청크 {ci+1} 시간: {chunk_elapsed:.2f}s")"""
code = code.replace(old, new)

# 4) 완료 메시지 뒤에 JSON 저장
old = '    print(f"\\n🎉 완료! → {merged_ply_path}")'
new = '''    print(f"\\n🎉 완료! → {merged_ply_path}")

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
    print(f"📄 타이밍 저장: {timing_path}")'''
code = code.replace(old, new)

with open(path, "w") as f:
    f.write(code)

print("✅ 패치 완료!")
print("결과: output_dir/timing.json 에 시간 저장됨")