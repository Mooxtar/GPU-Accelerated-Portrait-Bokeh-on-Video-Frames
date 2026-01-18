import os, glob, time, csv, math, argparse, subprocess
import numpy as np
import cv2
from tqdm import tqdm
from numba import cuda

@cuda.jit
def box_blur_h_u8(inp, out, radius):
    y, x = cuda.grid(2)
    H, W, C = inp.shape
    if y >= H or x >= W:
        return
    r = radius
    x0 = x - r
    if x0 < 0:
        x0 = 0
    x1 = x + r
    if x1 >= W:
        x1 = W - 1
    cnt = x1 - x0 + 1
    for ch in range(3):
        s = 0
        for xx in range(x0, x1 + 1):
            s += int(inp[y, xx, ch])
        out[y, x, ch] = s // cnt

@cuda.jit
def box_blur_v_u8(inp, out, radius):
    y, x = cuda.grid(2)
    H, W, C = inp.shape
    if y >= H or x >= W:
        return
    r = radius
    y0 = y - r
    if y0 < 0:
        y0 = 0
    y1 = y + r
    if y1 >= H:
        y1 = H - 1
    cnt = y1 - y0 + 1
    for ch in range(3):
        s = 0
        for yy in range(y0, y1 + 1):
            s += int(inp[yy, x, ch])
        out[y, x, ch] = s // cnt

@cuda.jit
def blend_u8(sharp, blur, mask_u8, out):
    y, x = cuda.grid(2)
    H, W, C = sharp.shape
    if y >= H or x >= W:
        return
    a = float(mask_u8[y, x]) / 255.0
    ia = 1.0 - a
    for ch in range(3):
        out[y, x, ch] = np.uint8(a * float(sharp[y, x, ch]) + ia * float(blur[y, x, ch]))

def make_mask(depth_u8, feather=11, keep_topk=2, min_area_frac=0.01):
    d = depth_u8
    _, m1 = cv2.threshold(d, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, m2 = cv2.threshold(255 - d, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    def score(mask):
        area = mask.mean() / 255.0
        if area < 0.02 or area > 0.80:
            return -1e9
        return -abs(area - 0.30)
    mask = m1 if score(m1) >= score(m2) else m2
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    h, w = mask.shape
    min_area = int(min_area_frac * h * w)
    num, lab, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)
    if num > 1:
        comps = []
        for cid in range(1, num):
            area = int(stats[cid, cv2.CC_STAT_AREA])
            if area >= min_area:
                comps.append((area, cid))
        comps.sort(reverse=True)
        keep = comps[:max(1, int(keep_topk))]
        out = np.zeros((h, w), dtype=np.uint8)
        for _, cid in keep:
            out[lab == cid] = 255
        mask = out
    kd = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.dilate(mask, kd, iterations=1)
    if feather > 0:
        mask = cv2.GaussianBlur(mask, (feather | 1, feather | 1), 0)
    return mask.astype(np.uint8)

class GPUBokeh:
    def __init__(self, H, W, blur_radius, threads=(16,16)):
        self.H, self.W = H, W
        self.r = int(blur_radius)
        self.threads = threads
        self.blocks = (math.ceil(H / threads[0]), math.ceil(W / threads[1]))
        self.d_rgb = cuda.device_array((H, W, 3), np.uint8)
        self.d_tmp = cuda.device_array((H, W, 3), np.uint8)
        self.d_blur = cuda.device_array((H, W, 3), np.uint8)
        self.d_out = cuda.device_array((H, W, 3), np.uint8)
        self.d_mask = cuda.device_array((H, W), np.uint8)
        self.stream = cuda.stream()
        self.e0 = cuda.event()
        self.e1 = cuda.event()
        self.e2 = cuda.event()
        self.e3 = cuda.event()

    def run(self, rgb, mask):
        self.e0.record(self.stream)
        self.d_rgb.copy_to_device(rgb, stream=self.stream)
        self.d_mask.copy_to_device(mask, stream=self.stream)
        self.e1.record(self.stream)
        self.e1.synchronize()
        h2d_ms = cuda.event_elapsed_time(self.e0, self.e1)

        self.e1.record(self.stream)
        box_blur_h_u8[self.blocks, self.threads, self.stream](self.d_rgb, self.d_tmp, self.r)
        box_blur_v_u8[self.blocks, self.threads, self.stream](self.d_tmp, self.d_blur, self.r)
        blend_u8[self.blocks, self.threads, self.stream](self.d_rgb, self.d_blur, self.d_mask, self.d_out)
        self.e2.record(self.stream)
        self.e2.synchronize()
        kernels_ms = cuda.event_elapsed_time(self.e1, self.e2)

        self.e2.record(self.stream)
        out = self.d_out.copy_to_host(stream=self.stream)
        self.e3.record(self.stream)
        self.e3.synchronize()
        d2h_ms = cuda.event_elapsed_time(self.e2, self.e3)

        return out, h2d_ms, kernels_ms, d2h_ms

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--packed_dir", required=True)
    ap.add_argument("--topk", type=int, default=80)
    ap.add_argument("--out_video", required=True)
    ap.add_argument("--metrics_csv", required=True)
    ap.add_argument("--blur_radius", type=int, default=18)
    ap.add_argument("--feather", type=int, default=11)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--limit_total_frames", type=int, default=0)
    ap.add_argument("--crf", type=int, default=23)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.packed_dir, "*.npz")))[:args.topk]
    if not files:
        raise RuntimeError(f"No npz files in {args.packed_dir}")

    os.makedirs(os.path.dirname(args.metrics_csv), exist_ok=True)
    os.makedirs(os.path.dirname(args.out_video), exist_ok=True)

    d0 = np.load(files[0], allow_pickle=True)
    rgb0 = d0["rgb"]
    H, W, _ = rgb0[0].shape

    gpu = GPUBokeh(H, W, args.blur_radius)
    _ = gpu.run(np.zeros((H, W, 3), np.uint8), np.zeros((H, W), np.uint8))

    ffmpeg_cmd = [
        "ffmpeg","-y","-hide_banner","-loglevel","error",
        "-f","rawvideo","-pix_fmt","bgr24","-s",f"{W}x{H}","-r",str(args.fps),
        "-i","-",
        "-c:v","libx264","-pix_fmt","yuv420p","-crf",str(args.crf),
        args.out_video
    ]
    proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

    t_load = t_mask = t_pipe = 0.0
    h2d_ms_sum = kernels_ms_sum = d2h_ms_sum = 0.0
    frames = 0
    t_all0 = time.perf_counter()

    for fpath in tqdm(files, desc=f"GPU stream {len(files)} clips"):
        if args.limit_total_frames and frames >= args.limit_total_frames:
            break
        tl0 = time.perf_counter()
        d = np.load(fpath, allow_pickle=True)
        rgb = d["rgb"]; depth = d["depth"]
        tl1 = time.perf_counter()
        t_load += (tl1 - tl0)
        T = rgb.shape[0]
        for i in range(T):
            if args.limit_total_frames and frames >= args.limit_total_frames:
                break
            tm0 = time.perf_counter()
            mask = make_mask(depth[i], feather=args.feather)
            tm1 = time.perf_counter()
            t_mask += (tm1 - tm0)

            out, h2d_ms, kernels_ms, d2h_ms = gpu.run(rgb[i], mask)
            h2d_ms_sum += h2d_ms
            kernels_ms_sum += kernels_ms
            d2h_ms_sum += d2h_ms

            tp0 = time.perf_counter()
            bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
            proc.stdin.write(bgr.tobytes())
            tp1 = time.perf_counter()
            t_pipe += (tp1 - tp0)

            frames += 1

    proc.stdin.close()
    proc.wait()
    t_all1 = time.perf_counter()
    total = (t_all1 - t_all0)
    ms_pf = 1000.0 * total / frames if frames else float("nan")

    h2d_s = h2d_ms_sum / 1000.0
    kernels_s = kernels_ms_sum / 1000.0
    d2h_s = d2h_ms_sum / 1000.0
    gpu_compute_total_s = h2d_s + kernels_s + d2h_s

    header = ["mode","clips","frames","fps","blur_radius","feather","crf",
              "load_s","mask_s","h2d_s","kernels_s","d2h_s","gpu_compute_total_s",
              "pipewrite_s","total_s","ms_per_frame_total","out_video"]
    row = ["gpu",len(files),frames,args.fps,args.blur_radius,args.feather,args.crf,
           f"{t_load:.3f}",f"{t_mask:.3f}",f"{h2d_s:.3f}",f"{kernels_s:.3f}",f"{d2h_s:.3f}",f"{gpu_compute_total_s:.3f}",
           f"{t_pipe:.3f}",f"{total:.3f}",f"{ms_pf:.3f}",args.out_video]

    write_header = not os.path.exists(args.metrics_csv)
    with open(args.metrics_csv, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(header)
        w.writerow(row)

    print("Wrote:", args.out_video)
    print("Appended metrics:", args.metrics_csv)

if __name__ == "__main__":
    main()
