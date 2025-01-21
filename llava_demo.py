'''demo'''
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '4,5'
from visualizer import get_local
get_local.activate()
import math
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
import cv2

import gradio as gr
from PIL import Image,ImageDraw


_, preprocess = clip.load("ViT-B/32", device='cpu', jit=False)
H, W = 24, 24
n_px = 224
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(n_px, interpolation=transforms.InterpolationMode.NEAREST),
    transforms.CenterCrop(n_px),
    transforms.Resize(H, interpolation=transforms.InterpolationMode.NEAREST),
    transforms.ToTensor(),
])


image_path = "assets/GettyImages-1215928137.jpg"

image = Image.open(image_path)
iw, ih = image.size

img = image

lr = 1
show_att = False
early_stop = True

loss_change_percent_threshold = 25  


quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

model = LlavaForConditionalGeneration.from_pretrained("pretrained_models/llava-1.5-7b-hf", quantization_config=quantization_config, device_map="auto")
processor = AutoProcessor.from_pretrained("pretrained_models/llava-1.5-7b-hf")
device = "cuda" if torch.cuda.is_available() else "cpu"


def method(mask_input, prompt_input, ori_img, choice, T, alpha, beta, max_new_token):
    prompt = "USER: <image>\n{}ASSISTANT:".format(prompt_input)
    image = ori_img
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)

    img_token_idx = int(torch.where(inputs['input_ids'] == 32000)[1])
    
    if choice in ["Scribble", "Point"]:
        mask_input = ((1-mask_input) * 255).astype(np.uint8)
        distance_transform = cv2.distanceTransform(mask_input, cv2.DIST_L2, 5)
        distance_transform_normalized = cv2.normalize(distance_transform, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        mask_input = distance_transform_normalized
        
    mask = transform(mask_input)[0]
    mask = mask.cuda() if torch.cuda.is_available() else mask
 
    # Generate input embeds
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
    

    # original output
    with torch.no_grad():
        outputs = model.generate(inputs_embeds=inputs_embeds, attention_mask=attention_mask, max_new_tokens=max_new_token, return_dict_in_generate=True, output_scores=True)
        cache = get_local.cache

        generate_ids = outputs.sequences
        logits = outputs.scores
    
        result_ids = generate_ids

        output = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        print(output)
        get_local.clear()
        torch.cuda.empty_cache()


    # init learnable Latent Variable
    visual_prompt = torch.nn.Parameter(torch.zeros_like(inputs_embeds[:,img_token_idx:img_token_idx+H*W,:]))
    loss_history = []
    vprompt_history = visual_prompt

    # start optimization x T
    loss_history = []
    for _ in range(T):
        new_inputs_embeds = inputs_embeds
        vprompt_cur = beta * visual_prompt + (1-beta) * vprompt_history  # EMA
        new_inputs_embeds[:,img_token_idx:img_token_idx+H*W,:] += vprompt_cur
        outputs = model.generate(inputs_embeds=new_inputs_embeds, attention_mask=attention_mask, max_new_tokens=1, return_dict_in_generate=True, output_scores=True)
        cache = get_local.cache

        generate_ids = outputs.sequences
        logits = outputs.scores
    
        result_ids = generate_ids

        output = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        # print(output)

        ori_attention_maps = cache['LlamaSdpaAttention.forward']
        attention_maps = [att.to(device) for i,att in enumerate(ori_attention_maps) if att.shape[-2] > 1]

        mean_att = torch.cat(attention_maps, 0).mean(0)

        fig = None
        if show_att:
            fig = show_image_relevance( mean_att[:, img_token_idx+H*W:,img_token_idx:img_token_idx+H*W].mean(axis=0).mean(axis=0), image, orig_image=image, mask=mask, preprocess=preprocess, only_map=True, show_mask=True)

            fig.savefig('vis/img_tmp_{}.png'.format(_),dpi=300,bbox_inches='tight')

    
        target2img_rel = mean_att[:, img_token_idx+H*W:,img_token_idx:img_token_idx+H*W].mean(axis=0).mean(axis=0).unsqueeze(0)
       
        loss = alpha * compute_ca_loss(target2img_rel.to(mask.device), masks=[mask], choice=choice, object_positions=None)
        # print(loss)
        loss_history.append(loss.item())

        if early_stop:
            if len(loss_history) > 2:
                loss_change_percent = np.abs((loss_history[-1] - loss_history[0]) / loss_history[0]) * 100
                
                if loss_change_percent > loss_change_percent_threshold:
                    print("Loss change percentage exceeds threshold. Stop.")

                    break
            
            if len(loss_history) > 1:
                if loss_history[-1] > loss_history[-2]:
                    break

        vprompt_history = vprompt_cur
        grad_cond = torch.autograd.grad(loss.requires_grad_(True), [visual_prompt], retain_graph=True)[0]
        
        visual_prompt = visual_prompt - lr * grad_cond
    
    
        get_local.clear()
        torch.cuda.empty_cache()

    # final output
    with torch.no_grad():
        get_local.clear()
        torch.cuda.empty_cache()
        new_inputs_embeds = inputs_embeds
        vprompt_cur = beta * visual_prompt + (1-beta) * vprompt_history  # EMA
        new_inputs_embeds[:,img_token_idx:img_token_idx+H*W,:] += vprompt_cur
        outputs = model.generate(inputs_embeds=new_inputs_embeds, attention_mask=attention_mask, max_new_tokens=max_new_token, return_dict_in_generate=True, output_scores=True)
        
        generate_ids = outputs.sequences
        logits = outputs.scores
    
        result_ids = generate_ids

        output = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        print(output)
        get_local.clear()
        torch.cuda.empty_cache()
    return output, fig


def process_image(image_1, choice, prompt, T, alpha, beta, max_new_token):
    if choice == "Boxes":
        mask_np = np.zeros_like(image_1["layers"][0])

        for image in image_1["layers"]:
            image = np.array(image)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                mask_np[y:y + h, x:x + w] = [1, 1, 1]

    else:
        mask_np = np.zeros_like(image_1["layers"][0])
        for image in image_1["layers"]:
            image = np.array(image)
            non_black_mask = (image[:, :, 0] != 0) | (image[:, :, 1] != 0) | (image[:, :, 2] != 0)
            non_zero_indices = np.where(non_black_mask)
            mask_np[non_zero_indices] = [1, 1, 1]  
        
    ori_img = image_1["background"]
    mean_values = np.mean(mask_np, axis=2)
    mask_input = mean_values
    output, att = method(mask_input, prompt, ori_img, choice, T, alpha, beta, max_new_token)
    _demo_output_text = output  

    return image_1["layers"][0], _demo_output_text
    

demo = gr.Interface(
    fn=process_image,
    inputs=[
        gr.ImageEditor(
            brush=gr.Brush(
                colors=[
                    "rgb(255, 0, 0)",  # red
                    "rgb(0, 255, 0)",  # green
                    "rgb(0, 0, 255)",  # blue
                    "rgb(255, 255, 0)",  # yellow
                    "rgb(0, 0, 0)"  # black
                ],
                default_size=7 
            ),
            type="pil",
            label="Color Sketch Pad",
            image_mode='RGB',
            value=img
        ),
        gr.Radio(
            choices=["Scribble", "Boxes", "Mask", "Point"],
            label="Select Tools"
        ),
        gr.Textbox(label='prompt'),
        gr.Slider(minimum=0, maximum=10, step=1, value=5, label="T"),
        gr.Slider(minimum=100, maximum=1000, step=100, value=400, label="alpha"),
        gr.Slider(minimum=0, maximum=1, step=0.05, value=0.5, label="beta"),
        gr.Slider(minimum=1, maximum=2048, step=2, value=512, label="max_new_token")
    ],
    outputs=["image", "text"],
    title="demo"
)
demo.launch(server_port=8008)





