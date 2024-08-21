'''demo'''
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '4'
from visualizer import get_local
get_local.activate()
import math
from PIL import Image, ImageDraw
import torch
import requests
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration, BitsAndBytesConfig
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
H, W = 16, 16
n_px = 224
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((n_px,n_px), interpolation=transforms.InterpolationMode.NEAREST),
    # transforms.CenterCrop(n_px),
    transforms.Resize((H,W), interpolation=transforms.InterpolationMode.NEAREST),
    transforms.ToTensor(),
])


image_path = "assets/GettyImages-1215928137.jpg"

image = Image.open(image_path)
iw, ih = image.size

img = image

lr = 1
show_att = True
early_stop = True

loss_change_percent_threshold = 25  


quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

model = InstructBlipForConditionalGeneration.from_pretrained("pretrained_models/instructblip-vicuna-7b-hf", quantization_config=quantization_config, device_map="auto")
processor = InstructBlipProcessor.from_pretrained("pretrained_models/instructblip-vicuna-7b-hf")
for p in model.language_model.parameters():
    p.requires_grad = False
device = "cuda" if torch.cuda.is_available() else "cpu"


def method(mask_input, prompt_input, ori_img, choice, T, alpha, beta, max_new_token):
    prompt = prompt_input
    image = ori_img
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)
    
    if choice in ["Scribble", "Point"]:
        mask_input = ((1-mask_input) * 255).astype(np.uint8)
        distance_transform = cv2.distanceTransform(mask_input, cv2.DIST_L2, 5)
        distance_transform_normalized = cv2.normalize(distance_transform, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        mask_input = distance_transform_normalized
        
    mask = transform(mask_input)[0]
    mask = mask.cuda() if torch.cuda.is_available() else mask
 
    # original output
    with torch.no_grad():
        outputs = model.generate(
                **inputs,
                do_sample=False,
                num_beams=5,
                max_length=50,
                min_length=1,
                top_p=0.9,
                repetition_penalty=1.5,
                length_penalty=1.0,
                temperature=1,
        )
        cache = get_local.cache

        generated_text = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
        print(generated_text)
        get_local.clear()
        torch.cuda.empty_cache()


    # init learnable Latent Variable
    visual_prompt = torch.nn.Parameter(torch.zeros((1, H*W+1, model.qformer.config.encoder_hidden_size))).cuda()
    vprompt_history = visual_prompt
    model.visual_prompt = visual_prompt

    # start optimization x T
    loss_history = []
    for _ in range(T):
        model.vprompt_cur = beta * model.visual_prompt + (1-beta) * vprompt_history  # EMA
        outputs = model.generate(
                **inputs,
                do_sample=False,
                num_beams=5,
                max_length=10,
                min_length=1,
                top_p=0.9,
                repetition_penalty=1.5,
                length_penalty=1.0,
                temperature=1,
        )

        generated_text = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()

        cache = get_local.cache
        ori_attention_maps = cache['InstructBlipQFormerMultiHeadAttention.forward']

        attention_maps = [att.to(device) for i,att in enumerate(ori_attention_maps) if att.shape[-2] != att.shape[-1]]

        mean_att = torch.cat(attention_maps, 0).mean(0)

        fig = None
        if show_att:
            fig = show_image_relevance( mean_att[:, :,1:].mean(axis=0).mean(axis=0), image, orig_image=image, mask=mask, preprocess=transforms.Compose([transforms.Resize((n_px,n_px), interpolation=transforms.InterpolationMode.BICUBIC),transforms.ToTensor()]), only_map=True, show_mask=True,att_hw=(H,W))

            fig.savefig('vis/img_tmp_{}.png'.format(_),dpi=300,bbox_inches='tight')

        target2img_rel = mean_att[:, :,1:].mean(axis=0).mean(axis=0).unsqueeze(0)
        # print(target2img_rel.shape)
        loss = alpha * compute_ca_loss(target2img_rel.to(mask.device), masks=[mask], object_positions=None)
        print(loss)
        loss_history.append(loss.item())

        if early_stop:
        ### early_stop
            if len(loss_history) >= 2:
                loss_change_percent = np.abs((loss_history[-1] - loss_history[0]) / loss_history[0]) * 100
                
                if loss_change_percent > loss_change_percent_threshold:
                    print("Loss change percentage exceeds threshold. Stop.")
                    break
        
        vprompt_history = model.vprompt_cur
        grad_cond = torch.autograd.grad(loss.requires_grad_(True), [model.visual_prompt], retain_graph=True)[0]
        
        model.visual_prompt = model.visual_prompt - lr * grad_cond
        
        get_local.clear()
        torch.cuda.empty_cache()

    # final output
    with torch.no_grad():
        get_local.clear()
        torch.cuda.empty_cache()
        model.vprompt_cur = beta * model.visual_prompt + (1-beta) * vprompt_history  # EMA
        outputs = model.generate(
                **inputs,
                do_sample=False,
                num_beams=5,
                max_length=max_new_token,
                min_length=1,
                top_p=0.9,
                repetition_penalty=1.5,
                length_penalty=1.0,
                temperature=1,
        )
        cache = get_local.cache
        ori_attention_maps = cache['InstructBlipQFormerMultiHeadAttention.forward']

        generated_text = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
        print(generated_text)
        get_local.clear()
        torch.cuda.empty_cache()
    return generated_text, fig


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





