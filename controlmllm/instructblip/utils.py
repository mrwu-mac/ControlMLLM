import torch
import math
from PIL import Image, ImageDraw, ImageFont
import logging
import os
import torchvision.transforms as T
import json
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
import cv2
from matplotlib.patches import Rectangle



def gaussian(x, mu, sigma):
    return torch.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) / (torch.sqrt(2 * torch.tensor(3.14159265358979323846)) * sigma)

def compute_ca_loss(rel_map, masks, choice=None, object_positions=None):
    loss = 0
    object_number = len(masks)
    if object_number == 0:
        return torch.tensor(0).float().cuda() if torch.cuda.is_available() else torch.tensor(0).float()


    attn_map = rel_map 

    b = attn_map.shape[0]
    # H, W = 24, 24
    H, W = masks[0].shape
    for obj_idx in range(object_number):
        obj_loss = 0
        mask = masks[obj_idx]
        
           
        ca_map_obj = attn_map.reshape(b, H, W)

        if choice and choice in ["Scribble", "Point"]:
            
            activation_value = (ca_map_obj * gaussian(mask,0,0.1)).reshape(b, -1).sum(dim=-1)/ca_map_obj.reshape(b, -1).sum(dim=-1)
        else:
            activation_value = (ca_map_obj * mask).reshape(b, -1).sum(dim=-1)/ca_map_obj.reshape(b, -1).sum(dim=-1)

        obj_loss += torch.mean((1 - activation_value) ** 2)
        
        loss += obj_loss

    return loss



def show_image_relevance(image_relevance, image, orig_image, preprocess, mask=None, only_map=False, show_mask=False, att_hw=(24,24)):
    # create heatmap from mask on image
    def show_cam_on_image(img, mask):
        heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        cam = heatmap + np.float32(img)
        cam = cam / np.max(cam)
        return cam

    if only_map:
        plt.plot()
        fig = plt.gcf()
        # fig, axs = plt.subplots(1, 1)
        dim = int(image_relevance.numel() ** 0.5)
        image_relevance = image_relevance.reshape(1, 1, dim, dim)
        image_relevance = torch.nn.functional.interpolate(image_relevance, size=224, mode='bilinear')
        image_relevance = image_relevance.reshape(224, 224).cuda().data.cpu().numpy()
        image_relevance = (image_relevance - image_relevance.min()) / (image_relevance.max() - image_relevance.min())
        
        image = preprocess(image)
        image = image.permute(1, 2, 0).data.cpu().numpy()
        image = (image - image.min()) / (image.max() - image.min())
        vis = show_cam_on_image(image, image_relevance)
        vis = np.uint8(255 * vis)
        vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)

        
        plt.imshow(vis)
        # plt.imshow(image)
        plt.axis('off')
        if show_mask:
            # draw = ImageDraw.Draw(fig)
            mask = mask.reshape(1,1,att_hw[0],att_hw[1])
            # mask = mask.reshape(1,1,16,16)
            mask = torch.nn.functional.interpolate(mask, size=224, mode='nearest')
            mask = mask.reshape(224, 224).cuda().data.cpu().numpy()
            mask_image = (mask * 255).astype(np.uint8)
            cv2.imwrite('vis/mask.png',mask_image)
            
    else:
        fig, axs = plt.subplots(1, 2)
        axs[0].imshow(orig_image)
        axs[0].axis('off')

        dim = int(image_relevance.numel() ** 0.5)
        image_relevance = image_relevance.reshape(1, 1, dim, dim)
        image_relevance = torch.nn.functional.interpolate(image_relevance, size=224, mode='BICUBIC')
        image_relevance = image_relevance.reshape(224, 224).cuda().data.cpu().numpy()
        image_relevance = (image_relevance - image_relevance.min()) / (image_relevance.max() - image_relevance.min())
        # _, preprocess = clip.load("ViT-B/32", device='cpu', jit=False)
        image = preprocess(image)
        image = image.permute(1, 2, 0).data.cpu().numpy()
        image = (image - image.min()) / (image.max() - image.min())
        vis = show_cam_on_image(image, image_relevance)
        vis = np.uint8(255 * vis)
        vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)
        axs[1].imshow(vis)
        axs[1].axis('off')

    return fig