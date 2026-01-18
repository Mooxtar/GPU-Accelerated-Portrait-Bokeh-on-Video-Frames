#!/bin/bash
#SBATCH --account=project_2016196
#SBATCH --partition=gputest
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --output=/users/mrabayev/gpu_course/GP25_Labs/bokeh_project/logs/end2end_gpu_top80_%j.out

module --force purge
module load python-data cuda

cd /users/mrabayev/gpu_course/GP25_Labs
source /users/mrabayev/gpu_course/GP25_Labs/bokeh_run/venv/bokeh/bin/activate

PACKED=/users/mrabayev/gpu_course/GP25_Labs/bokeh_run/packed_npz/top80
OUT_VIDEO=/users/mrabayev/gpu_course/GP25_Labs/bokeh_run/final_end2end/videos/top80_gpu.mp4
CSV=/users/mrabayev/gpu_course/GP25_Labs/bokeh_project/results/top80_onevideo_gpu_metrics.csv

mkdir -p /users/mrabayev/gpu_course/GP25_Labs/bokeh_run/final_end2end/videos
mkdir -p /users/mrabayev/gpu_course/GP25_Labs/bokeh_project/results

python3 /users/mrabayev/gpu_course/GP25_Labs/bokeh_project/src/render_topk_stream_to_video_gpu.py \
  --packed_dir "$PACKED" \
  --topk 80 \
  --out_video "$OUT_VIDEO" \
  --metrics_csv "$CSV" \
  --blur_radius 18 \
  --feather 11 \
  --fps 30 \
  --crf 23
