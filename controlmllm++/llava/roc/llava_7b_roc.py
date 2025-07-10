import os

# To get attention weights
from visualizer import get_local

get_local.activate()

import argparse
from PIL import Image, ImageDraw
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig
from utils import compute_ca_loss, show_image_relevance
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import numpy as np
import json
import clip
import cv2
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"
_, preprocess = clip.load("ViT-B/32", device='cpu', jit=False)


def parse_args():
    parser = argparse.ArgumentParser(description="Hyperparameters for the model")

    # Visual prompt options
    parser.add_argument('--visual_prompt', type=str, choices=['Box', 'Mask', 'Scribble', 'Point'], default='Box',
                        help='Visual prompt method')

    # Model paths
    parser.add_argument('--model_path', type=str, default="pretrained_models/llava-1.5-7b-hf",
                        help='Path to the pretrained model')
    parser.add_argument('--data_path', type=str, default="dataset/LVIS", help='Path to the dataset')
    parser.add_argument('--question_file', type=str, default='dataset/LVIS/question_roc.json', help='Path to the question file')
    parser.add_argument('--set', type=str, default='test', choices=['test', 'val'], help="Use test or validation split")
    parser.add_argument('--answers_file', type=str, default='outputs/llava_7b_roc.json', help='Path to the answers file')

    # Generation parameters
    parser.add_argument('--max_new_tokens', type=int, default=30, help='Maximum number of new tokens to generate')

    # Attention dimensions and input size of image
    parser.add_argument('--H', type=int, default=24, help='Height of the attention map')
    parser.add_argument('--W', type=int, default=24, help='Width of the attention map')
    parser.add_argument('--n_px', type=int, default=224, help='Input size of image')

    # Optimization hyperparameters
    parser.add_argument('--lr', type=float, default=0.03, help='Learning rate for ADAM')
    # parser.add_argument('--beta', type=float, default=0.5, help='Beta parameter')
    parser.add_argument('--alpha', type=float, default=400, help='Alpha parameter')
    parser.add_argument('--T', type=int, default=4, help='T parameter')
    parser.add_argument('--mu', type=float, default=0.4)
    # parser.add_argument('--early_stop', action='store_true', help='Enable early stopping')
    # parser.add_argument('--loss_change_percent_threshold', type=float, default=25, help='Loss change percentage threshold')

    # ControlMLLM++ parameters
    parser.add_argument('--use_cd', action='store_true', help='Use Comparative Decoding')
    parser.add_argument('--cd_alpha', type=float, default=0.7, help='Comparative Decoding alpha parameter')
    parser.add_argument('--cd_beta', type=float, default=0.1, help='Comparative Decoding beta parameter')

    parser.add_argument('--start_layer', type=int, default=14, help='Start layer for attention')
    parser.add_argument('--end_layer', type=int, default=26, help='End layer for attention')

    # Flags for visualization
    parser.add_argument('--show_att', action='store_true', help='Flag to show attention maps')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    ATT_LAYER_START = args.start_layer
    ATT_LAYER_END = args.end_layer

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

    for p in model.parameters():
        p.requires_grad = False

    # Load questions from question file
    questions = [json.loads(q) for q in open(args.question_file, "r")]
    if args.set == 'test':
        questions = questions[:1548]
    else:
        questions = questions[1548:]
    # Open answers file
    answers_file = os.path.expanduser(args.answers_file)
    ans_file = open(answers_file, "w")

    for q in tqdm(questions, desc="Processing Questions"):
        qid = q['id']
        label = q['name']  # ground truth
        question = q['text'].replace('<location> ', '')
        prompt = "USER: <image>\n{} ASSISTANT:".format(question)

        image_path = os.path.join(args.data_path, 'image', q['image_path'].split('/')[-2],
                                  q['image_path'].split('/')[-1])
        image = Image.open(image_path)
        iw, ih = image.size

        inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)
        img_token_idx = int(torch.where(inputs['input_ids'] == 32000)[1])

        # =========== for visual prompt ==============
        mask = np.zeros((ih, iw))
        # Box
        bbox = q['bbox']
        # Mask
        mask_path = os.path.join(args.data_path, 'mask', q['seg_mask'])
        # Point
        point = q['center_point']
        # Scribble
        scribble = q['scribble']
        if args.visual_prompt == 'Box':
            x_min, y_min, x_max, y_max = int(bbox[0]), int(bbox[1]), int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])
            mask[y_min: y_max, x_min: x_max] = 1
            mask = transform(mask)[0]

        elif args.visual_prompt == 'Mask':
            mask = Image.open(mask_path)
            mask = transform(np.array(mask))[0]

        elif args.visual_prompt == 'Scribble':
            for scri in scribble:
                mask[int(scri[1]), int(scri[0])] = 1
            mask = ((1 - mask) * 255).astype(np.uint8)
            distance_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
            distance_transform_normalized = cv2.normalize(distance_transform, None, 0, 255, cv2.NORM_MINMAX,
                                                          dtype=cv2.CV_8U)
            mask = distance_transform_normalized
            mask = transform(np.array(mask))[0]

        elif args.visual_prompt == 'Point':
            mask[int(point[1]), int(point[0])] = 1
            mask = ((1 - mask) * 255).astype(np.uint8)
            distance_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
            distance_transform_normalized = cv2.normalize(distance_transform, None, 0, 255, cv2.NORM_MINMAX,
                                                          dtype=cv2.CV_8U)
            mask = distance_transform_normalized
            mask = transform(np.array(mask))[0]
        # =========== for visual prompt ==============

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

        visual_prompt = torch.nn.Parameter(torch.zeros_like(
            inputs_embeds[:, img_token_idx:img_token_idx + args.H * args.W, :])
        )

        hyperparams = {'lr': args.lr, 't': 1}
        beta1 = 0.9
        beta2 = 0.999
        epsilon = 1e-3
        # Initialize ADAM parameters
        m = torch.zeros_like(visual_prompt)
        s = torch.zeros_like(visual_prompt)
        state = {'m': m, 's': s}
        output_T = []

        for _ in range(args.T):
            if _ == args.T - 1:
                tmp_use_cd = args.use_cd
                tmp_max_new_tokens = args.max_new_tokens
                use_grad = False
            else:
                tmp_use_cd = False
                tmp_max_new_tokens = 1
                use_grad = True

            new_inputs_embeds = inputs_embeds.clone()
            new_inputs_embeds[:, img_token_idx:img_token_idx + args.H * args.W, :] += visual_prompt

            # ControlMLLM temp output
            with torch.set_grad_enabled(use_grad):
                outputs = model.generate(
                    inputs_embeds=new_inputs_embeds,
                    attention_mask=attention_mask,
                    max_new_tokens=tmp_max_new_tokens,
                    return_dict_in_generate=True,
                    output_scores=True,
                    use_cd=tmp_use_cd,
                    uncond_inputs_embeds=inputs_embeds,
                    cd_alpha=args.cd_alpha,
                    cd_beta=args.cd_beta
                )

            if _ == args.T - 1:
                generate_ids = outputs.sequences
                output = processor.batch_decode(
                    generate_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False
                )[0]
                output_T.append(output)
                get_local.clear()
                torch.cuda.empty_cache()
                break

            # len(ori_attention_maps) = num_tokens * num_tokens
            ori_attention_maps = get_local.cache['LlamaSdpaAttention.forward']

            attn_layers = 32  # attention layers
            if args.use_cd:
                ori_attention_maps = [
                    attn for i in range(0, len(ori_attention_maps) // attn_layers, 2)
                    for attn in ori_attention_maps[i * attn_layers: (i + 1) * attn_layers]
                ]

            # Filter the first token's attention map, which lacks KV cache
            attention_maps = [
                # len(attention_maps) = 32
                att for i, att in enumerate(ori_attention_maps) if att.shape[-2] > 1
            ]

            # mean_att.shape = torch.Size([32, 603, 603]) (example)
            mean_att = torch.cat([att.to(device) for att in attention_maps[ATT_LAYER_START:ATT_LAYER_END]], 0).mean(0)

            if args.show_att:
                fig = show_image_relevance(
                    mean_att[:,
                    -1,
                    img_token_idx:img_token_idx + args.H * args.W
                    ].mean(axis=0),
                    image,
                    orig_image=image,
                    mask=mask,
                    preprocess=preprocess,
                    only_map=True,
                    show_mask=False
                )
                # You can save the figure if you want

            # Extract target-to-image relevance
            # target2img_rel.shape = torch.Size([1, 576])
            target2img_rel = mean_att[:,
                             -1,
                             img_token_idx:img_token_idx + args.H * args.W
                             ].mean(axis=0).unsqueeze(0)

            loss = args.alpha * compute_ca_loss(
                target2img_rel.to(mask.device),
                masks=[mask],
                choice=args.visual_prompt,
                mu=args.mu,
                object_positions=None
            )
            # ADAM for visual prompt
            grad_cond = torch.autograd.grad(
                loss.requires_grad_(True),
                [visual_prompt],
                retain_graph=True
            )[0]
            state['m'] = beta1 * state['m'] + (1 - beta1) * grad_cond
            state['s'] = beta2 * state['s'] + (1 - beta2) * grad_cond.pow(2)
            m_hat = state['m'] / (1 - beta1 ** hyperparams['t'])
            s_hat = state['s'] / (1 - beta2 ** hyperparams['t'])
            visual_prompt = visual_prompt - hyperparams['lr'] * m_hat / (torch.sqrt(s_hat) + epsilon)
            hyperparams['t'] += 1
            get_local.clear()
            torch.cuda.empty_cache()

        ans_file.write(json.dumps({
            "question_id": qid,
            "answers": output_T,
            "label": label
        }) + '\n')
        ans_file.flush()

    ans_file.close()


if __name__ == "__main__":
    main()