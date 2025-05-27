import numpy as np
from torchvision import transforms
import torch

def att_window_from_bbox(bbox, image_size, att_shape, clip=True):
    x1, y1, x2, y2 = bbox
    img_w, img_h    = image_size
    att_h, att_w    = att_shape

    block_w = img_w / att_w
    block_h = img_h / att_h

    x_start = int(np.floor(x1 / block_w))
    y_start = int(np.floor(y1 / block_h))
    x_end   = int(np.ceil (x2 / block_w))
    y_end   = int(np.ceil (y2 / block_h))

    if clip:
        x_start = max(0, min(x_start, att_w))
        y_start = max(0, min(y_start, att_h))
        x_end   = max(0, min(x_end,   att_w))
        y_end   = max(0, min(y_end,   att_h))

    return x_start, y_start, x_end, y_end


def get_grid_shape(processor, image_inputs):
    with torch.no_grad():
        aux = processor.image_processor(images=image_inputs)
    h, w = int(aux["image_grid_thw"][0, 1]/2), int(aux["image_grid_thw"][0, 2]/2)
    # print(f"get_grid_shape: {h}, {w}")
    return h, w

def build_mask_from_bbox(bbox, image_size, grid_shape, device="cpu"):
    h, w = grid_shape
    x0, y0, x1, y1 = att_window_from_bbox(bbox, image_size, (h, w))
    # print(f"att_window_from_bbox: {x0}, {y0}, {x1}, {y1}")
    mask = torch.zeros((h, w), dtype=torch.float32, device=device)
    mask[y0:y1, x0:x1] = 1.0
    return mask

def compute_activation_loss_qwen(rel_map, masks, eps=1e-6):
    if len(masks) == 0:
        return torch.tensor(0., device=rel_map.device)

    B, HW = rel_map.shape
    H, W  = masks[0].shape
    rel_map = rel_map.reshape(B, H, W)

    total_att = rel_map.reshape(B, -1).sum(-1, keepdim=True) + eps

    loss = 0.
    for m in masks:
        act = (rel_map * m).reshape(B, -1).sum(-1) / total_att.squeeze(-1)
        loss += torch.mean((1.0 - act)**2)
    return loss / len(masks)