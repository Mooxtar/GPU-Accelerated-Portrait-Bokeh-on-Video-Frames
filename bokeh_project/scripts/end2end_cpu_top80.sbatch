#!/bin/bash
#SBATCH --account=project_2016196
#SBATCH --partition=test
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --output=/users/mrabayev/gpu_course/GP25_Labs/bokeh_project/logs/end2end_cpu_top80_%j.out

module --force purge
module load python-data

cd /users/mrabayev/gpu_course/GP25_Labs
source bokeh_run/venv/bokeh/bin/activate

PACKED=/users/mrabayev/gpu_course/GP25_Labs/bokeh_run/packed_npz/top80
OUT_VIDEO=/users/mrabayev/gpu_course/GP25_Labs/bokeh_run/final_end2end/videos/top80_cpu.mp4
CSV=/users/mrabayev/gpu_course/GP25_Labs/bokeh_project/results/top80_onevideo_cpu_metrics.csv

mkdir -p /users/mrabayev/gpu_course/GP25_Labs/bokeh_run/final_end2end/videos
mkdir -p /users/mrabayev/gpu_course/GP25_Labs/bokeh_project/results

python3 bokeh_project/src/render_topk_stream_to_video_cpu.py \
  --packed_dir "$PACKED" \
  --topk 80 \
  --out_video "$OUT_VIDEO" \
  --metrics_csv "$CSV" \
  --blur_radius 18 \
  --feather 11 \
  --fps 30 \
  --crf 23
