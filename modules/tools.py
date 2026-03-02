import argparse
import numpy as np
import math
from decord import VideoReader, cpu
from decord._ffi.base import DECORDError

import torch
from modules.cache_utils import StreamCache
from models.llava_next.mm_utils import tokenizer_image_token
from models.llava_next.constants import IMAGE_TOKEN_INDEX
from modules.model_builder import format_mcqa_prompt


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]

def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


# offline
def load_video(video_path, num_frames):
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frame_num = len(vr)
    num_frames = min(num_frames, total_frame_num)

    uniform_sampled_frames = np.linspace(0, total_frame_num - 1, num_frames, dtype=int)
    frame_idx = uniform_sampled_frames.tolist()
    for i in range(5):
        try:
            spare_frames = vr.get_batch(frame_idx).asnumpy()
            return spare_frames
        except DECORDError as e:
            continue

    return spare_frames

# online
def load_video_fps(video_path, fps, start=0, end=-1):
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frame_num = len(vr)
    video_fps = round(vr.get_avg_fps())

    # For videos longer than 30 minutes, reduce the FPS to 0.2
    if 30*60*video_fps < total_frame_num:
        fps = 0.2

    stride = int(video_fps / fps)

    start = start * video_fps
    if end == -1:
        end = total_frame_num
    else:
        end = end * video_fps
    
    end = min(end, total_frame_num)
    start = max(0, start)
    if end <= start:
        end = start + 1

    frame_idx = [i for i in range(start, end, stride)]
    for i in range(5):
        try:
            spare_frames = vr.get_batch(frame_idx).asnumpy()
            return spare_frames
        except DECORDError as e:
            continue

    return spare_frames


def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser()

    # Define the command-line arguments
    parser.add_argument("--video_path", default="/data/ning/.dataset/MovieChat-1K-test/videos")
    parser.add_argument("--gt_file", default="/data/ning/.dataset/MovieChat-1K-test/gt/gt")
    parser.add_argument("--output_dir", default="outputs/MovieChat/tmp")
    parser.add_argument("--output_name", default="pred")
    parser.add_argument("--model_path", type=str, default="lmms-lab/LLaVA-Video-7B-Qwen2")
    parser.add_argument("--chunk_idx", type=int, default=0)
    parser.add_argument("--num_chunks", type=int, default=1)

    parser.add_argument("--fps", type=float, default=1)
    parser.add_argument("--max_tokens", type=int, default=128)
    parser.add_argument("--test_size", type=int, default=500)
    parser.add_argument("--target_duration", type=str, default="all")
    parser.add_argument("--model_arch", type=str, default="llava_next")

    parser.add_argument("--clip_frame_num", type=int, default=32)
    parser.add_argument("--cache_size", type=int, default=12000)
    parser.add_argument("--recent_frames", type=int, default=2)
    parser.add_argument("--scatter_ratio", type=float, default=0.5)
    parser.add_argument("--chunk_size", type=int, default=16)
    parser.add_argument("--retr_ratio", type=float, default=0.4)


    return parser.parse_args()


init_prefix = '<|im_start|>system \nYou are a helpful assistant.<|im_end|>\n<|im_start|>user \n'
suffix = "<|im_end|>\n<|im_start|>assistant\nBest option: ("


def frame_encoding(
    videos,
    model,
    tokenizer,
    past_key_values,
    args,
):
    qs = '<image>'
    
    if past_key_values is None:
        past_key_values = StreamCache.from_legacy_cache(past_key_values)
        past_key_values.initialize(args.cache_size, args.recent_frames, args.scatter_ratio, args.chunk_size)
        prompt = init_prefix + qs
    else:
        prompt = qs

    prefix_len = tokenizer_image_token(init_prefix, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").shape[0]
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()    

    with torch.inference_mode():
        output_ids = model.generate(
            inputs=input_ids, 
            images=videos, 
            modalities=["video"], 
            past_key_values=past_key_values,
            do_sample=False, 
            temperature=0.0, 
            max_new_tokens=1, 
            use_cache=True, 
            prefix_len = prefix_len,
            sparse="no-no-no",
            )
             
    past_key_values.flush_cache()

    return past_key_values

def response(
    videos,
    ques,
    question,
    model,
    tokenizer,
    past_key_values,
    max_tokens,
    args,
):
    annotation = ""
    qs = annotation + '\n' + ques + suffix if videos is None else '<image>' + annotation + '\n' + ques + suffix
    
    if past_key_values is None:
        past_key_values = StreamCache.from_legacy_cache(past_key_values)
        past_key_values.initialize(args.cache_size, args.recent_frames, args.scatter_ratio, args.chunk_size)
        prompt = init_prefix + qs
    else:
        prompt = qs
    
    prefix_len = tokenizer_image_token(init_prefix, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").shape[0]
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()    
    question = f"Question: {question}"
    observation_len = tokenizer_image_token(question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").shape[0]
    past_key_values.observation_len = observation_len

    sparse = f"no-{args.scatter_ratio}-{args.retr_ratio}"
    past_key_values.backup_cache()

    with torch.inference_mode():
        output_ids = model.generate(
            inputs=input_ids, 
            images=videos, 
            modalities=["video"], 
            past_key_values=past_key_values,
            do_sample=False, 
            temperature=0.0, 
            max_new_tokens=max_tokens, 
            use_cache=True, 
            prefix_len=prefix_len,
            sparse=sparse,
            )
   
    past_key_values.reset_cache()
    output = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

    return output
