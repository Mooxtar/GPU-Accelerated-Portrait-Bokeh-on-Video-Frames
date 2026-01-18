# GPU-Accelerated Portrait Bokeh on Video Frames (RGB + Depth)

This project implements an end-to-end pipeline that creates a portrait-style bokeh effect (sharp foreground, blurred background) using synchronized RGB and depth frames from a Kinect dataset. We provide a CPU baseline and a GPU-accelerated version (CUDA kernels via Numba), then compare their performance using detailed timing metrics while producing final MP4 outputs via FFmpeg streaming.

---

## Project idea

We use the depth frame to build a soft foreground mask `a = mask / 255` and blend sharp and blurred images:

out = a * sharp + (1 - a) * blurred

The CPU and GPU versions use the same formula. The GPU version accelerates the expensive blur and blend step.

---

## Dataset

We use the Two Person Interaction Kinect Dataset (Kaggle):
https://www.kaggle.com/datasets/dasmehdixtr/two-person-interaction-kinect-dataset/code/data

Each clip contains synchronized pairs:

- rgb_XXXXXX.png (640×480 RGB, 8-bit per channel)
- depth_XXXXXX.png (640×480 depth, stored as 8-bit grayscale in our export)

Depth values are not only 0 or 255. They can be any value in 0–255. In many Kinect-derived exports, 0 often appears as invalid or missing depth, while other values represent near-to-far in a quantized scale.

---

## Repository structure

GPU-Accelerated-Portrait-Bokeh-on-Video-Frames/
  bokeh_project/
    src/
      pack_clips_to_npz.py
      render_topk_stream_to_video_cpu.py
      render_topk_stream_to_video_gpu.py
      compare_end2end_metrics.py
    scripts/
      end2end_cpu_top80.sbatch
      end2end_gpu_top80.sbatch
    results/
      top80_onevideo_cpu_metrics.csv
      top80_onevideo_gpu_metrics.csv
      top80_onevideo_compare.csv
    logs/
      *.out, *.err
  bokeh_run/
    data/
      .gitkeep
    packed_npz/
      .gitkeep
    final_end2end/videos/
      .gitkeep

---

## Why bokeh_run is empty in the repo

bokeh_run is a workspace folder. It is intentionally empty in Git (only .gitkeep) because it is meant to store large generated artifacts that should not be committed:

- raw dataset frames: bokeh_run/data/archive/...
- packed npz clips: bokeh_run/packed_npz/...
- output MP4 videos: bokeh_run/final_end2end/videos/...

This keeps the repository lightweight and avoids uploading gigabytes of data.

---

## How to run (end-to-end)

### 1) Clone the repo

If SSH cloning fails, use HTTPS:

git clone https://github.com/Mooxtar/GPU-Accelerated-Portrait-Bokeh-on-Video-Frames.git
cd GPU-Accelerated-Portrait-Bokeh-on-Video-Frames

---

### 2) Download and place the dataset

Download the dataset ZIP from Kaggle:
https://www.kaggle.com/datasets/dasmehdixtr/two-person-interaction-kinect-dataset/code/data

Extract it and place the dataset archive folder into:

bokeh_run/data/archive/

After this you should have clip folders like:

bokeh_run/data/archive/s01s02/06/001/
bokeh_run/data/archive/s02s07/...

Each clip folder should contain rgb_*.png and depth_*.png.

---

### 3) Pack clips into NPZ (reduces I/O overhead)

The raw dataset contains many small PNG files. On HPC systems this creates a strong I/O bottleneck. To reduce overhead, we pack each clip into a single compressed NPZ file containing:

- rgb: (T, H, W, 3) uint8
- depth: (T, H, W) uint8
- rel: relative path metadata
- frames: frame count

Example (pack top 80 longest clips):

python3 bokeh_project/src/pack_clips_to_npz.py \
  --root bokeh_run/data/archive \
  --out_dir bokeh_run/packed_npz/top80 \
  --topk 80

Output:
- bokeh_run/packed_npz/top80/*.npz

If you hit disk quota, pack fewer clips or limit frames per clip:

python3 bokeh_project/src/pack_clips_to_npz.py \
  --root bokeh_run/data/archive \
  --out_dir bokeh_run/packed_npz/top80 \
  --topk 80 \
  --limit_frames_per_clip 40

---

### 4) Run CPU end-to-end pipeline (creates MP4 + metrics)

This reads packed NPZ clips, computes masks on CPU, applies CPU blur + blend, and streams frames directly into FFmpeg to produce one merged MP4.

python3 bokeh_project/src/render_topk_stream_to_video_cpu.py \
  --packed_dir bokeh_run/packed_npz/top80 \
  --topk 80 \
  --out_video bokeh_run/final_end2end/videos/top80_cpu.mp4 \
  --metrics_csv bokeh_project/results/top80_onevideo_cpu_metrics.csv \
  --blur_radius 18 \
  --feather 11 \
  --fps 30 \
  --crf 23

Output:
- bokeh_run/final_end2end/videos/top80_cpu.mp4
- bokeh_project/results/top80_onevideo_cpu_metrics.csv

---

### 5) Run GPU end-to-end pipeline (creates MP4 + metrics)

This uses the same mask generation on CPU, but accelerates blur + blend on GPU using CUDA kernels. It also reports pure GPU timing using CUDA events:

- h2d_s: host-to-device copies
- kernels_s: blur + blend kernels only
- d2h_s: device-to-host copies
- gpu_compute_total_s = h2d_s + kernels_s + d2h_s

python3 bokeh_project/src/render_topk_stream_to_video_gpu.py \
  --packed_dir bokeh_run/packed_npz/top80 \
  --topk 80 \
  --out_video bokeh_run/final_end2end/videos/top80_gpu.mp4 \
  --metrics_csv bokeh_project/results/top80_onevideo_gpu_metrics.csv \
  --blur_radius 18 \
  --feather 11 \
  --fps 30 \
  --crf 23

Output:
- bokeh_run/final_end2end/videos/top80_gpu.mp4
- bokeh_project/results/top80_onevideo_gpu_metrics.csv

---

### 6) Compare CPU vs GPU metrics

This reads the CPU and GPU metrics CSV files and writes a single comparison CSV with speedups:

python3 bokeh_project/src/compare_end2end_metrics.py \
  --cpu_csv bokeh_project/results/top80_onevideo_cpu_metrics.csv \
  --gpu_csv bokeh_project/results/top80_onevideo_gpu_metrics.csv \
  --out_csv bokeh_project/results/top80_onevideo_compare.csv \
  --match_settings

Output:
- bokeh_project/results/top80_onevideo_compare.csv

---

## Running on Mahti (Slurm)

We provide Slurm scripts:

- bokeh_project/scripts/end2end_cpu_top80.sbatch
- bokeh_project/scripts/end2end_gpu_top80.sbatch

Typical usage:

sbatch bokeh_project/scripts/end2end_cpu_top80.sbatch
sbatch bokeh_project/scripts/end2end_gpu_top80.sbatch

Logs are written into:
- bokeh_project/logs/*.out
- bokeh_project/logs/*.err

Note: the CPU job should use a CPU partition on your cluster (on Mahti you should use a valid CPU partition configured for your course/project). The GPU job uses a GPU partition and requests an A100.

---

## What metrics mean

Both end-to-end scripts output timing breakdowns.

CPU CSV fields:
- load_s: reading NPZ clips from disk
- mask_s: depth-based foreground mask generation (OpenCV)
- compute_s: CPU blur + blend time
- pipewrite_s: time to push raw frames to FFmpeg stdin
- total_s: full end-to-end runtime
- ms_per_frame_total: total_s / frames

GPU CSV fields:
- load_s: reading NPZ clips from disk
- mask_s: mask generation on CPU
- h2d_s: RGB + mask transfers to GPU
- kernels_s: pure GPU kernel time (blur + blend)
- d2h_s: result transfers back to CPU
- gpu_compute_total_s: h2d_s + kernels_s + d2h_s
- pipewrite_s: time to stream frames to FFmpeg
- total_s: full end-to-end runtime
- ms_per_frame_total: total_s / frames

Comparison CSV fields:
- speedup_total = cpu_total_s / gpu_total_s
- speedup_compute = cpu_compute_s / gpu_compute_s
- cpu_ms_pf and gpu_ms_pf compare total time per frame

---

## Results (measured on top80 merged run)

Example comparison row:

frames = 2608  
cpu_total_s = 54.118  
gpu_total_s = 38.643  
speedup_total = 1.400x  

cpu_compute_s = 14.541  
gpu_compute_s = 2.940  
speedup_compute = 4.946x  

cpu_ms_pf = 20.751 ms/frame  
gpu_ms_pf = 14.817 ms/frame  

This shows that GPU kernels accelerate the compute part strongly, but the full end-to-end speedup is smaller because the total time also includes loading, mask creation on CPU, and video encoding.

---

## Notes and common issues

1) Disk quota / libpng write errors  
If you write PNG frames to disk, you can easily hit quota. This project streams frames directly to FFmpeg to avoid writing thousands of PNGs.

2) FFmpeg requirement  
You need ffmpeg installed and available in PATH. The scripts call ffmpeg to create MP4.

3) CUDA requirement for GPU run  
GPU run requires a CUDA-capable GPU and a working Numba CUDA setup.

---

## Summary

This repository provides a full reproducible workflow:

1. Download dataset from Kaggle and place into bokeh_run/data/archive
2. Pack top-K clips into NPZ for faster I/O
3. Run CPU end-to-end pipeline to produce MP4 and metrics
4. Run GPU end-to-end pipeline to produce MP4 and metrics (with pure kernel timing)
5. Compare results using a dedicated comparison script

If you want to change the experiment, the main knobs are:
- --topk
- --blur_radius
- --feather
- --fps
- --crf
- --limit_total_frames (optional)
