import os, glob, time, csv, argparse, subprocess
import numpy as np
import cv2
from tqdm import tqdm

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

def bokeh_cpu(rgb, mask, blur_radius):
    k = 2*blur_radius + 1
    blur = cv2.blur(rgb, (k, k))  
    a = (mask.astype(np.float32)/255.0)[...,None]
    out = (a*rgb.astype(np.float32) + (1-a)*blur.astype(np.float32)).astype(np.uint8)
    return out

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
    H,W,_ = rgb0[0].shape

    ffmpeg_cmd = [
        "ffmpeg","-y","-hide_banner","-loglevel","error",
        "-f","rawvideo","-pix_fmt","bgr24","-s",f"{W}x{H}","-r",str(args.fps),
        "-i","-",
        "-c:v","libx264","-pix_fmt","yuv420p","-crf",str(args.crf),
        args.out_video
    ]
    proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

    t_load=t_mask=t_cpu=t_pipe=0.0
    frames=0
    t_all0 = time.perf_counter()

    for fpath in tqdm(files, desc=f"CPU stream {len(files)} clips"):
        if args.limit_total_frames and frames >= args.limit_total_frames:
            break

        tl0=time.perf_counter()
        d=np.load(fpath, allow_pickle=True)
        rgb=d["rgb"]; depth=d["depth"]
        tl1=time.perf_counter()
        t_load += (tl1-tl0)

        T=rgb.shape[0]
        for i in range(T):
            if args.limit_total_frames and frames >= args.limit_total_frames:
                break

            tm0=time.perf_counter()
            mask=make_mask(depth[i], feather=args.feather)
            tm1=time.perf_counter()
            t_mask += (tm1-tm0)

            tc0=time.perf_counter()
            out=bokeh_cpu(rgb[i], mask, args.blur_radius)   
            tc1=time.perf_counter()
            t_cpu += (tc1-tc0)

            tp0=time.perf_counter()
            bgr=cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
            proc.stdin.write(bgr.tobytes())
            tp1=time.perf_counter()
            t_pipe += (tp1-tp0)

            frames += 1

    proc.stdin.close()
    proc.wait()
    t_all1 = time.perf_counter()
    total = (t_all1 - t_all0)
    ms_pf = 1000.0*total/frames if frames else float("nan")

    header=["mode","clips","frames","fps","blur_radius","feather","crf",
            "load_s","mask_s","compute_s","pipewrite_s","total_s","ms_per_frame_total","out_video"]
    row=["cpu",len(files),frames,args.fps,args.blur_radius,args.feather,args.crf,
         f"{t_load:.3f}",f"{t_mask:.3f}",f"{t_cpu:.3f}",f"{t_pipe:.3f}",
         f"{total:.3f}",f"{ms_pf:.3f}",args.out_video]

    write_header = not os.path.exists(args.metrics_csv)
    with open(args.metrics_csv,"a",newline="") as f:
        w=csv.writer(f)
        if write_header: w.writerow(header)
        w.writerow(row)

    print("Wrote:", args.out_video)
    print("Appended metrics:", args.metrics_csv)

if __name__=="__main__":
    main()
