import os, glob, argparse
import numpy as np
import cv2
from tqdm import tqdm

def idx(p): return int(os.path.basename(p).split("_")[1].split(".")[0])

def read_rgb(path):
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None: raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.uint8)

def read_depth(path):
    d = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if d is None: raise FileNotFoundError(path)
    return d.astype(np.uint8)

def sorted_pairs(seq_dir):
    rgb = sorted(glob.glob(os.path.join(seq_dir, "rgb_*.png")))
    dep = sorted(glob.glob(os.path.join(seq_dir, "depth_*.png")))
    dep_map = {idx(p): p for p in dep}
    pairs = [(idx(r), r, dep_map[idx(r)]) for r in rgb if idx(r) in dep_map]
    pairs.sort(key=lambda t: t[0])
    return pairs

def collect_top_clips(root, topk):
    clips = []
    for dirpath, _, filenames in os.walk(root):
        if any(f.startswith("rgb_") and f.endswith(".png") for f in filenames):
            n = len(glob.glob(os.path.join(dirpath, "rgb_*.png")))
            if n > 0:
                rel = os.path.relpath(dirpath, root)
                clips.append((n, rel))
    clips.sort(reverse=True)
    return clips[:topk]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--topk", type=int, default=150)
    ap.add_argument("--limit_frames_per_clip", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    clips = collect_top_clips(args.root, args.topk)
    print("Packing clips:", len(clips))

    for n, rel in tqdm(clips, desc="Pack to npz"):
        seq_dir = os.path.join(args.root, rel)
        pairs = sorted_pairs(seq_dir)
        if args.limit_frames_per_clip and args.limit_frames_per_clip > 0:
            pairs = pairs[:args.limit_frames_per_clip]
        if not pairs:
            continue

        rgbs = []
        deps = []
        for _, rp, dp in pairs:
            rgbs.append(read_rgb(rp))
            deps.append(read_depth(dp))

        rgb_arr = np.stack(rgbs, axis=0)  
        dep_arr = np.stack(deps, axis=0)   

        out_path = os.path.join(args.out_dir, rel.replace("/", "__") + ".npz")
        np.savez_compressed(out_path, rgb=rgb_arr, depth=dep_arr, rel=rel, frames=len(pairs))
    print("Done. Out:", args.out_dir)

if __name__ == "__main__":
    main()
