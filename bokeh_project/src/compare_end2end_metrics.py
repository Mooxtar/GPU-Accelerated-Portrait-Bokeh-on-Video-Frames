import os, csv, argparse, math

def read_rows(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, "r", newline="") as f:
        r = csv.DictReader(f)
        rows = list(r)
    if not rows:
        raise RuntimeError(f"No rows in {path}")
    return rows

def pick_row(rows, prefer_mode=None, match=None):
    cand = rows
    if prefer_mode is not None:
        cand2 = [x for x in cand if x.get("mode","").strip().lower() == prefer_mode.lower()]
        if cand2:
            cand = cand2
    if match:
        def ok(x):
            for k, v in match.items():
                if v is None:
                    continue
                if str(x.get(k, "")) != str(v):
                    return False
            return True
        cand2 = [x for x in cand if ok(x)]
        if cand2:
            cand = cand2
    return cand[-1]

def ffloat(x, default=float("nan")):
    try:
        return float(x)
    except Exception:
        return default

def fint(x, default=None):
    try:
        return int(float(x))
    except Exception:
        return default

def valid_pos(x):
    return (x is not None) and (not math.isnan(x)) and (x > 0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpu_csv", required=True)
    ap.add_argument("--gpu_csv", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--match_settings", action="store_true")
    args = ap.parse_args()

    cpu_rows = read_rows(args.cpu_csv)
    gpu_rows = read_rows(args.gpu_csv)

    match = None
    if args.match_settings:
        cpu_last = pick_row(cpu_rows, prefer_mode="cpu")
        match = {
            "clips": cpu_last.get("clips"),
            "fps": cpu_last.get("fps"),
            "blur_radius": cpu_last.get("blur_radius"),
            "feather": cpu_last.get("feather"),
            "crf": cpu_last.get("crf"),
        }

    cpu = pick_row(cpu_rows, prefer_mode="cpu", match=match)
    gpu = pick_row(gpu_rows, prefer_mode="gpu", match=match)

    cpu_frames = fint(cpu.get("frames"))
    gpu_frames = fint(gpu.get("frames"))
    if cpu_frames is not None and gpu_frames is not None:
        frames = min(cpu_frames, gpu_frames)
    else:
        frames = cpu_frames if cpu_frames is not None else gpu_frames

    cpu_total_s = ffloat(cpu.get("total_s"))
    gpu_total_s = ffloat(gpu.get("total_s"))

    cpu_compute_s = ffloat(cpu.get("compute_s"))
    gpu_compute_s = ffloat(gpu.get("gpu_compute_total_s", gpu.get("compute_s")))

    speedup_total = (cpu_total_s / gpu_total_s) if (valid_pos(gpu_total_s) and not math.isnan(cpu_total_s)) else float("nan")
    speedup_compute = (cpu_compute_s / gpu_compute_s) if (valid_pos(gpu_compute_s) and not math.isnan(cpu_compute_s)) else float("nan")

    cpu_ms_pf = ffloat(cpu.get("ms_per_frame_total"))
    gpu_ms_pf = ffloat(gpu.get("ms_per_frame_total"))

    out_dir = os.path.dirname(args.out_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    header = [
        "frames",
        "cpu_total_s","gpu_total_s","speedup_total",
        "cpu_compute_s","gpu_compute_s","speedup_compute",
        "cpu_ms_pf","gpu_ms_pf",
        "cpu_out_video","gpu_out_video",
        "cpu_csv","gpu_csv"
    ]
    row = [
        frames,
        f"{cpu_total_s:.6f}", f"{gpu_total_s:.6f}", f"{speedup_total:.6f}",
        f"{cpu_compute_s:.6f}", f"{gpu_compute_s:.6f}", f"{speedup_compute:.6f}",
        f"{cpu_ms_pf:.6f}", f"{gpu_ms_pf:.6f}",
        cpu.get("out_video",""),
        gpu.get("out_video",""),
        args.cpu_csv,
        args.gpu_csv
    ]

    write_header = not os.path.exists(args.out_csv)
    with open(args.out_csv, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(header)
        w.writerow(row)

    print("Wrote comparison:", args.out_csv)
    print("Frames:", frames)
    print("Total speedup (CPU/GPU):", f"{speedup_total:.3f}x")
    print("Compute speedup (CPU/GPU):", f"{speedup_compute:.3f}x")
    print("CPU video:", cpu.get("out_video",""))
    print("GPU video:", gpu.get("out_video",""))

if __name__ == "__main__":
    main()
