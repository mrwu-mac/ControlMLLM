"""
Usage:
- Eval Prediction:
python task/RTC/eval.py --pred_file=[your generated result by running task/RTC/llava_rtc.sh]

"""
import argparse
import json
import os
import re
import random
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import textwrap
from tqdm import tqdm
import pdb
import numpy as np

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ann_file', type=str, default='data/RTC/question_rtc.json')
    parser.add_argument('--pred_file', type=str, default='outputs/llava_7b_rtc_box.json')
    return parser.parse_args()

def remove_not_phrases_v2(text):
    # Pattern covers the start of a phrase up to and including 'not' and any following characters until a comma or period
    pattern = r"\s+not[^,.]*[,.]"
    text = re.sub(pattern, "", text)
    pattern = r"\s+no[^,.]*[,.]"
    text = re.sub(pattern, "", text)
    return text 

if __name__ == "__main__":
    args = get_args()
    # Fix the random seed
    random.seed(42)
    if os.path.isfile(args.pred_file):
        predictions = [json.loads(line) for line in open(args.pred_file)]
    elif os.path.isdir(args.pred_file):
        predictions = [json.loads(line) for pred_file in os.listdir(args.pred_file) for line in open(os.path.join(args.pred_file, pred_file))]
    else:
        raise NotImplementedError('Not supported file format.')
    

    annotations = [json.loads(line) for line in open(args.ann_file)]
    assert len(predictions) == len(annotations)

    for t in range(2):
        our_rel_max = 0
        our_rel_sum = 0

        total_correct = 0
        num = 0
        for idx, (i, ann) in enumerate(tqdm(zip(predictions, annotations))):
            num += 1

            # Process name and synonyms
            ann['name'] = ann['label'].replace('_', ' ').strip()
            ann['synonyms'] = []

            # Match Result
            processed_text = remove_not_phrases_v2(i['answers'][t])
            if ann['name'] in processed_text or any(syn_i in processed_text for syn_i in ann['synonyms']):
                total_correct += 1
            else:
                pass
        
        acc = total_correct / num
        our_rel_max /= num
        our_rel_sum /= num
        
        name = 'Baseline' if t==0 else 'Ours'
        print(f'{name} ---> Acc:{acc*100:.3f}%')
   
