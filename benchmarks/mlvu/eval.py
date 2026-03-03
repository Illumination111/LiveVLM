import argparse
from tqdm import tqdm
import numpy as np
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_path", default=r"", help="The path to file containing prediction.")
    args = parser.parse_args()
    return args


def map_prediction_to_option(pred):
    pred_option = "none"
    if isinstance(pred, str):
        prediction_letter = pred[0]
        if prediction_letter in "abcdABCD":
            pred_option = prediction_letter.lower()
        if "answer is " in pred:
            pred = pred[pred.index("answer is"):]
        if "A:" in pred or "A)" in pred:
            pred_option = "a"
        elif "B:" in pred or "B)" in pred:
            pred_option = "b"
        elif "C:" in pred or "C)" in pred:
            pred_option = "c"
        elif "D:" in pred or "D)" in pred:
            pred_option = "d"
        # elif "E:" in pred or "E)" in pred:
        #     pred_option = "e"
    return pred_option


def check_ans(pred, gt):
    flag = False

    pred_option = map_prediction_to_option(pred)

    if pred_option not in "abcd":
        print(f"Model does not follow the instruction: {pred}")
    elif pred_option == gt.lower():
        flag = True

    return flag


def main():
    # Parse arguments.
    args = parse_args()

    file = open(args.pred_path)
    new_pred_contents = [eval(i.strip()) for i in file.readlines()]

    task_accuracy = {}
    for new_pred_content in tqdm(new_pred_contents):
        task_type = new_pred_content["task_type"]
        if task_type not in task_accuracy:
            task_accuracy[task_type] = {
                "yes_count": 0,
                "no_count": 0,
            }

        gt = new_pred_content["answer"].lower()
        if check_ans(
            pred=new_pred_content["pred"],
            gt=gt,
        ):
            task_accuracy[task_type]["yes_count"] += 1
        else:
            task_accuracy[task_type]["no_count"] += 1

    report_dir = os.path.dirname(args.pred_path)+"/report.txt"
    f = open(report_dir, "w")

    for task_name in task_accuracy:
        yes_count = task_accuracy[task_name]["yes_count"]
        no_count = task_accuracy[task_name]["no_count"]
        accuracy = yes_count / (yes_count + no_count)
        print(task_name)
        print("\tYes count:", yes_count)
        print("\tAll count:", yes_count + no_count)
        print("\tAccuracy:", accuracy)

        print(task_name, file=f, flush=True)
        print("\tYes count:", yes_count, file=f, flush=True)
        print("\tAll count:", yes_count + no_count, file=f, flush=True)
        print("\tAccuracy:", accuracy, file=f, flush=True)
    
    yes_count = 0
    no_count = 0
    for task_name in task_accuracy:
        yes_count += task_accuracy[task_name]["yes_count"]
        no_count += task_accuracy[task_name]["no_count"]
    accuracy = yes_count / (yes_count + no_count)
    print("All")
    print("\tYes count:", yes_count)
    print("\tAll count:", yes_count + no_count)
    print("\tAccuracy:", accuracy)

    print("All", file=f, flush=True)
    print("\tYes count:", yes_count, file=f, flush=True)
    print("\tAll count:", yes_count + no_count, file=f, flush=True)
    print("\tAccuracy:", accuracy, file=f, flush=True)
   
    f.close()


if __name__ == "__main__":
    main()