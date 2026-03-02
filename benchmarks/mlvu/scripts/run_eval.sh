MODEL=lmms-lab/llava-onevision-qwen2-7b-ov

#################################
FPS=0.5
CACHE_SIZE=12000
CLIP_FRAME_NUM=32
RECENT_FRAMES=2
SCATTER_RATIO=0.5
RETR_RATIO=0.4
#################################

EXP_TAG=cache_${CACHE_SIZE}_clip_${CLIP_FRAME_NUM}_recent_${RECENT_FRAMES}_scatter_${SCATTER_RATIO}_retr_${RETR_RATIO}
output_dir="benchmarks/mlvu/outputs/${FPS}fps/${EXP_TAG}"

export HF_ENDPOINT="https://hf-mirror.com"
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64
################################# Run ##################################

gpu_list="${CUDA_VISIBLE_DEVICES:-0,1}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

for IDX in $(seq 0 $((CHUNKS-1))); do
  CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python3 benchmarks/mlvu/run_streaming_qa.py \
    --model_path $MODEL \
    --gt_file benchmarks/mlvu/dev_debug_mc.json \
    --video_path /home/zyning/_datasets/MLVU_dev/MLVU_dev/MLVU/video \
    --output_dir ${output_dir} \
    --output_name ${CHUNKS}_${IDX} \
    --num_chunks ${CHUNKS} \
    --chunk_idx ${IDX} \
    --test_size ${test_size} \
    --max_tokens 8 \
    --clip_frame_num $CLIP_FRAME_NUM \
    --fps $FPS \
    --cache_size ${CACHE_SIZE} \
    --recent_frames ${RECENT_FRAMES} \
    --scatter_ratio ${SCATTER_RATIO} \
    --retr_ratio ${RETR_RATIO} &
done

wait

output_file=${output_dir}/merge.jsonl

# Clear out the output file if it exists.
> "${output_file}"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ${output_dir}/${CHUNKS}_${IDX}.json >> "${output_file}"
done

################################# Eval ##################################

python3 benchmarks/mlvu/eval.py \
    --pred_path ${output_file}
