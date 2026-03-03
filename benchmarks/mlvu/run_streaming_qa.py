import argparse
import torch
import gc

import json
import os
import math
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.append(project_root)
from modules.model_builder import load_pretrained_model_arch, format_mcqa_prompt
from modules.tools import *

from tqdm import tqdm

import pandas as pd

import numpy as np
import warnings
warnings.filterwarnings("ignore")


def run_inference(args):
 
    tokenizer, model, image_processor = load_pretrained_model_arch(args.model_path, args.model_arch)

    test_size = args.test_size
    json_files = json.load(open(args.gt_file, "r"))
   
    dataset_size = len(json_files)
    if dataset_size < test_size:
        test_size = dataset_size
    
    json_files = json_files[:test_size]
    json_files = get_chunk(json_files, args.num_chunks, args.chunk_idx)

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    ans_file = open(os.path.join(output_dir, f"{args.output_name}.json"), "a")
    num_lines = 0
    with open(os.path.join(output_dir, f"{args.output_name}.json"), "r") as test_file:
        for _ in test_file:
            num_lines += 1
    already_len = num_lines
  

    clip_frame_num = args.clip_frame_num

    past_key_values = None
    video_time_pos = 0
    count = 0

    for sample in tqdm(json_files):
            
        video_id = sample["video_id"]
        video_path = sample["video_path"]
        parts = video_path.split('/')
        suffix = '/'.join(parts[-2:])
        load_video_path = os.path.join(args.video_path, suffix)

        all_video = None
        video_time_pos = 0
        del past_key_values
        past_key_values = None

        for conversation in sample['conversations']:
            count += 1
            if count<=already_len:
                continue

            question = conversation["question"]
            choices = conversation["choices"]
            answer = conversation["answer"]

            letters = ['A', 'B', 'C', 'D']
            letter = ''
            idx = 0
            for choice in choices:
                if choice == answer:
                    letter = letters[idx]
                    break
                idx += 1
            answer = letter

            question_type = conversation["question_type"]

            if all_video is None:
                all_video = load_video_fps(load_video_path, args.fps)
                question_time = len(all_video)

            # Frame encoding
            while video_time_pos < question_time:
                next_time_pos = min(video_time_pos + clip_frame_num, question_time)
                video = all_video[video_time_pos: next_time_pos]
                video = image_processor.preprocess(video, return_tensors="pt")["pixel_values"].half().cuda()
                video = [video]
                video_time_pos= next_time_pos
                past_key_values = frame_encoding(
                    video,
                    model,
                    tokenizer,
                    past_key_values,
                    args,
                )
            
            video = None
            ques = format_mcqa_prompt(question, choices)
            output = response(
                video,
                ques,
                question,
                model,
                tokenizer,
                past_key_values,
                args.max_tokens,
                args,
            )
            gc.collect()
            torch.cuda.empty_cache()
        
            results = {}
            results["video_id"] = video_id
            results["video_path"] = video_path
            results["task_type"] = question_type
            results['question'] = question
            results['answer'] = answer
            
            results['pred'] = output
            ans_file.write(json.dumps(results) + "\n")
            ans_file.flush()
   
    ans_file.close()



if __name__ == "__main__":
    
    args = parse_args()
    run_inference(args)