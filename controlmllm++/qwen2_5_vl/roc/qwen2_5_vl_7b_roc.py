import os, json, argparse, torch, numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import (
    AutoProcessor, Qwen2_5_VLForConditionalGeneration
)
from qwen_vl_utils import process_vision_info
from qwen_utils import (
    get_grid_shape, build_mask_from_bbox, compute_activation_loss_qwen
)

device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------- CLI ----------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model_path', default="pretrained_models/Qwen2.5-VL-7B-Instruct")
    p.add_argument('--data_path', default="dataset/LVIS")
    p.add_argument('--question_file', default='dataset/LVIS/question_roc.json')
    p.add_argument('--set', type=str, default='test', choices=['test', 'val'], help="Use test or validation split")
    p.add_argument('--answers_file', default='outputs/qwen2_5_vl_7b_roc.json')

    p.add_argument('--visual_prompt', choices=['Box'], default='Box')
    p.add_argument('--prompt_type', type=str, default='box', choices=['box', 'color', 'none'])
    p.add_argument('--max_new_tokens', type=int, default=30)
    p.add_argument('--lr', type=float, default=0.02)
    p.add_argument('--alpha', type=float, default=400)
    p.add_argument('--T', type=int, default=2)
    p.add_argument('--mu', type=float, default=0.4)

    p.add_argument('--use_cd', action='store_true')
    p.add_argument('--cd_alpha', type=float, default=0.01)
    p.add_argument('--cd_beta', type=float, default=0.1)

    p.add_argument('--start_layer', type=int, default=12, help='Start layer for attention')
    p.add_argument('--end_layer', type=int, default=28, help='End layer for attention')

    # p.add_argument('--show_att', action='store_true')
    return p.parse_args()


def main():
    args = parse_args()
    ATT_LAYER_START = args.start_layer
    ATT_LAYER_END = args.end_layer

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        attn_implementation="eager",
        device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(
        args.model_path, trust_remote_code=True,
        padding_side='left', use_fast=True
    )
    for p in model.parameters():
        p.requires_grad = False

    questions = [json.loads(l) for l in open(args.question_file)]
    if args.set == 'test':
        questions = questions[:1548]
    else:
        questions = questions[1548:]
    ans_file = open(os.path.expanduser(args.answers_file), "w")

    beta1, beta2, eps = 0.9, 0.999, 1e-3

    for q in tqdm(questions, desc="Qwen‑ROC"):
        img_path = os.path.join(args.data_path, 'image',
                                *q['image_path'].split('/')[-2:])
        image = Image.open(img_path).convert("RGB")
        bbox = q['bbox']
        x_min, y_min, x_max, y_max = int(bbox[0]), int(bbox[1]), int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])
        box = [x_min, y_min, x_max, y_max]
        if args.prompt_type == 'box':
            location_text = f"box @ [{box[0]}, {box[1]}, {box[2]}, {box[3]}]"
        elif args.prompt_type == 'color':
            location_text = "in red bounding box "
            draw = ImageDraw.Draw(image)
            draw.rectangle(box, outline='red', width=2)
        else:
            location_text = ""
        qid, label = q['id'], q['name']
        question = q['text'].replace('<location> ', f'{location_text} ')
        question = question + " Answer the question using a single word or phrase."

        msgs = [{
            "role": "user",
            "content": [
                {"type": "image", "image": img_path, "max_pixels": 512 * 28 * 28},
                {"type": "text", "text": question}
            ]
        }]
        image_inputs, _ = process_vision_info(msgs)
        grid_shape = get_grid_shape(processor, image_inputs)

        text_prompt = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text_prompt],
                           images=image_inputs,
                           padding=True,
                           return_tensors="pt").to(device)

        vision_start_token_id = processor.tokenizer.convert_tokens_to_ids('<|vision_start|>')
        vision_end_token_id = processor.tokenizer.convert_tokens_to_ids('<|vision_end|>')
        pos = inputs['input_ids'].tolist()[0].index(vision_start_token_id) + 1
        pos_end = inputs['input_ids'].tolist()[0].index(vision_end_token_id)

        mask = build_mask_from_bbox(box, image_size=image.size,
                                    grid_shape=grid_shape, device=device)

        model.visual_prompt = torch.nn.Parameter(torch.zeros(
            (grid_shape[0] * grid_shape[1], model.config.hidden_size), dtype=model.dtype)).to(device)
        hyperparams = {'lr': args.lr, 't': 1}
        m = torch.zeros_like(model.visual_prompt)
        s = torch.zeros_like(model.visual_prompt)
        state = {'m': m, 's': s}

        output_T = []
        for tt in range(args.T):
            is_last = tt == args.T - 1
            with torch.set_grad_enabled(not is_last):
                if is_last:
                    generated_ids = model.generate(**inputs,
                                                   max_new_tokens=args.max_new_tokens)
                    generated_ids_trimmed = [
                        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                    ]
                    output_text = processor.batch_decode(
                        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                    )
                    output_T.append(output_text)
                    break
                else:
                    out = model(**inputs, output_attentions=True)
                mean_att = torch.cat([att.to(device) for att in out.attentions[ATT_LAYER_START:ATT_LAYER_END]], 0).mean(
                    0)
                tgt2img = mean_att[:, -1, pos:pos_end].mean(0, keepdim=True)  # [1, H*W]
                loss = args.alpha * compute_activation_loss_qwen(tgt2img, [mask])

            grad_cond = torch.autograd.grad(loss, model.visual_prompt, retain_graph=False)[0]
            state['m'] = beta1 * state['m'] + (1 - beta1) * grad_cond
            state['s'] = beta2 * state['s'] + (1 - beta2) * grad_cond.pow(2)
            m_hat = state['m'] / (1 - beta1 ** hyperparams['t'])
            s_hat = state['s'] / (1 - beta2 ** hyperparams['t'])
            model.visual_prompt = model.visual_prompt - hyperparams['lr'] * m_hat / (torch.sqrt(s_hat) + eps)
            hyperparams['t'] += 1
            torch.cuda.empty_cache()

        ans_file.write(json.dumps({"question_id": qid, "answers": output_T, "label": label}) + '\n')
        ans_file.flush()

    ans_file.close()


if __name__ == "__main__":
    main()
