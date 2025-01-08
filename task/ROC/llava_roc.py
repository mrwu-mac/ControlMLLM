
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '6,7'
from visualizer import get_local
get_local.activate()
import math
import argparse
from PIL import Image, ImageDraw
import torch
import requests
from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig
from utils import compute_ca_loss, show_image_relevance
from torchviz import make_dot
import clip
import torchvision.transforms as transforms
import numpy as np
import math
import json
import cv2
from tqdm import tqdm

_, preprocess = clip.load("ViT-B/32", device='cpu', jit=False)


def parse_args():
    parser = argparse.ArgumentParser(description="Hyperparameters for the model")

    # Visual prompt options
    parser.add_argument('--visual_prompt', type=str, choices=['Box', 'Mask', 'Scribble', 'Point'], default='Box', help='Visual prompt method')

    # Model paths
    parser.add_argument('--model_path', type=str, default="pretrained_models/llava-1.5-7b-hf", help='Path to the pretrained model')
    parser.add_argument('--data_path', type=str, default="data/ROC/LVIS", help='Path to the dataset')
    parser.add_argument('--question_file', type=str, default='data/ROC/question_roc.json', help='Path to the question file')
    parser.add_argument('--answers_file', type=str, default='outputs/llava_roc.json', help='Path to the answers file')

    # Attention dimensions and input size of image
    parser.add_argument('--H', type=int, default=24, help='Height of the attention map')
    parser.add_argument('--W', type=int, default=24, help='Width of the attention map')
    parser.add_argument('--n_px', type=int, default=224, help='Input size of image')

    # Optimization parameters
    parser.add_argument('--alpha', type=float, default=400, help='Alpha parameter')
    parser.add_argument('--T', type=int, default=5, help='T parameter')

    # Flags
    parser.add_argument('--show_att', action='store_true', help='Flag to show attention maps')
    parser.add_argument('--early_stop', action='store_true', help='Enable early stopping')
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')

    # Learning rate and other optimization hyperparameters
    parser.add_argument('--lr', type=float, default=1, help='Learning rate')
    parser.add_argument('--beta', type=float, default=0.5, help='Beta parameter')

    # Threshold for loss change percentage
    parser.add_argument('--loss_change_percent_threshold', type=float, default=25, help='Loss change percentage threshold')

    args = parser.parse_args()
    return args



def main():
    args = parse_args()

    # Define transforms
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(args.n_px, interpolation=transforms.InterpolationMode.NEAREST),
        transforms.CenterCrop(args.n_px),
        transforms.Resize(args.H, interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor(),
    ])

    # Load model and processor
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    model = LlavaForConditionalGeneration.from_pretrained(
        args.model_path,
        quantization_config=quantization_config,
        device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(args.model_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for p in model.parameters():
        p.requires_grad = False

    questions = [json.loads(q) for q in open(args.question_file, "r")]

    answers_file = os.path.expanduser(args.answers_file)

    if os.path.exists(answers_file):
        answers = [json.loads(q) for q in open(answers_file, "r")]
        answer_ids = [q['question_id'] for q in answers]

    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    if args.resume:
        ans_file = open(answers_file, "a")
    else:
        ans_file = open(answers_file, "w")

    for q in tqdm(questions, desc="Processing Questions"):
        qid = q['id']
        if args.resume and answer_ids and qid in answer_ids:
            continue

        image_path = os.path.join(args.data_path, 'image', q['image_path'].split('/')[-2], q['image_path'].split('/')[-1])
        # print(image_path)
        question = q['text'].replace('<location> ', '')
        # print(question)
        label = q['name']
        # print(label)
        bbox = q['bbox']
        mask_path = os.path.join(args.data_path, 'mask', q['seg_mask'])
        point = q['center_point']
        scribble = q['scribble']

        prompt = "USER: <image>\n{} ASSISTANT:".format(question)
        image = Image.open(image_path)
        iw, ih = image.size

        inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)
        img_token_idx = int(torch.where(inputs['input_ids'] == 32000)[1])

        mask = np.zeros((ih, iw))
        
        if args.visual_prompt == 'Box':
            x_min, y_min, x_max, y_max = int(bbox[0]), int(bbox[1]), int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])
            mask[y_min: y_max, x_min: x_max] = 1
            mask = transform(mask.numpy())[0]

        elif args.visual_prompt == 'Mask':
            mask = Image.open(mask_path)
            mask = transform(np.array(mask))[0]

        elif args.visual_prompt == 'Scribble':
            for scri in scribble:
                mask[int(scri[1]), int(scri[0])] = 1
            mask = ((1-mask) * 255).astype(np.uint8)
            distance_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
            distance_transform_normalized = cv2.normalize(distance_transform, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            mask = distance_transform_normalized
            mask = transform(np.array(mask))[0]
    
        elif args.visual_prompt == 'Point':
            mask[int(point[1]), int(point[0])] = 1
            mask = ((1-mask) * 255).astype(np.uint8)
            distance_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
            distance_transform_normalized = cv2.normalize(distance_transform, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            mask = distance_transform_normalized
            mask = transform(np.array(mask))[0]
        

        mask = mask.cuda() if torch.cuda.is_available() else mask

        with torch.no_grad():
            inputs_embeds = model.get_input_embeddings()(inputs.input_ids)
            if inputs.pixel_values is not None and inputs.input_ids.shape[1] != 1:
                image_outputs = model.vision_tower(inputs.pixel_values, output_hidden_states=True)
                # this is not memory efficient at all (output_hidden_states=True) will save all the hidden stated.
                selected_image_feature = image_outputs.hidden_states[model.config.vision_feature_layer]
                
                vision_feature_select_strategy = model.config.vision_feature_select_strategy
                if vision_feature_select_strategy == "default":
                    selected_image_feature = selected_image_feature[:, 1:]
                elif vision_feature_select_strategy == "full":
                    selected_image_feature = selected_image_feature
                else:
                    raise ValueError(
                        f"Unexpected select feature strategy: {model.config.vision_feature_select_strategy}"
                    )

                image_features = model.multi_modal_projector(selected_image_feature)
                inputs_embeds, attention_mask, labels, position_ids = model._merge_input_ids_with_image_features(
                    image_features, inputs_embeds, inputs.input_ids, inputs.attention_mask, labels=None
                )
                if labels is None:
                    labels = torch.full_like(attention_mask, model.config.ignore_index).to(torch.long)

        visual_prompt = torch.nn.Parameter(torch.zeros_like(inputs_embeds[:, img_token_idx:img_token_idx + args.H * args.W, :]))

        loss_history = []
        vprompt_history = visual_prompt

        output_T = []
        rel_T = []

        for _ in range(args.T):
            new_inputs_embeds = inputs_embeds
            vprompt_cur = args.beta * visual_prompt + (1 - args.beta) * vprompt_history  # EMA
            new_inputs_embeds[:, img_token_idx:img_token_idx + args.H * args.W, :] += vprompt_cur
            outputs = model.generate(inputs_embeds=new_inputs_embeds, attention_mask=attention_mask, max_new_tokens=30, return_dict_in_generate=True, output_scores=True)

            generate_ids = outputs.sequences

            output = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            # print(output)
            if _ == 0:
                output_T.append(output)

            ori_attention_maps = get_local.cache['LlamaSdpaAttention.forward']
            attention_maps = [att for i, att in enumerate(ori_attention_maps) if att.shape[-2] > 1]

            rel_T.append([0, 0])

            mean_att = torch.cat([att.to(device) for att in attention_maps], 0).mean(0)

            if args.show_att:
                fig = show_image_relevance(
                    mean_att[:, img_token_idx + args.H * args.W:, img_token_idx:img_token_idx + args.H * args.W].mean(axis=0).mean(axis=0),
                    image,
                    orig_image=image,
                    mask=mask,
                    preprocess=None,
                    only_map=True,
                    show_mask=True
                )
                fig.savefig('vis/img_tmp_{}.png'.format(_), dpi=300, bbox_inches='tight')

            target2img_rel = mean_att[:, img_token_idx + args.H * args.W:, img_token_idx:img_token_idx + args.H * args.W].mean(axis=0).mean(axis=0).unsqueeze(0)
            loss = args.alpha * compute_ca_loss(target2img_rel.to(mask.device), masks=[mask], choice=args.visual_prompt, object_positions=None)

            loss_history.append(loss.item())
            if args.early_stop:
                if len(loss_history) >= 2:
                    loss_change_percent = abs((loss_history[-1] - loss_history[0]) / loss_history[0]) * 100
                    if loss_change_percent > args.loss_change_percent_threshold:
                        # print("Loss change percentage exceeds threshold. Stop.")
                        break

            vprompt_history = vprompt_cur

            grad_cond = torch.autograd.grad(loss.requires_grad_(True), [visual_prompt], retain_graph=True)[0]
            visual_prompt = visual_prompt - args.lr * grad_cond

            get_local.clear()
            torch.cuda.empty_cache()

        with torch.no_grad():
            get_local.clear()
            torch.cuda.empty_cache()
            new_inputs_embeds = inputs_embeds
            vprompt_history = args.beta * visual_prompt + (1 - args.beta) * vprompt_history  # EMA
            new_inputs_embeds[:, img_token_idx:img_token_idx + args.H * args.W, :] += vprompt_history
            outputs = model.generate(inputs_embeds=new_inputs_embeds, attention_mask=attention_mask, max_new_tokens=30, return_dict_in_generate=True, output_scores=True)

            generate_ids = outputs.sequences
            output = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            # print(output)
            output_T.append(output)

            rel_T.append([0, 0])
            get_local.clear()
            torch.cuda.empty_cache()

        ans_file.write(json.dumps({
            "question_id": qid,
            "answers": output_T,
            "relevancy": rel_T,
            "label": label
        }) + '\n')

    ans_file.close()

if __name__ == "__main__":
    main()