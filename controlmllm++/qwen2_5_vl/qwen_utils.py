import numpy as np
from torchvision import transforms
import torch

def att_window_from_bbox(bbox, image_size, att_shape, clip=True):
    """
    将原图上的边界框 (bbox) 映射到注意力图上的窗口坐标。

    Args:
        bbox (tuple): (x1, y1, x2, y2) —— 原图像素坐标，左上角 & 右下角。
        image_size (tuple): (width, height) —— 原图尺寸（像素）。
        att_shape (tuple): (h, w) —— 注意力图尺寸；如 (24, 24)、(16, 16)。
        clip (bool): 是否将结果裁剪到 [0, h-1] / [0, w-1] 区间内，默认 True。

    Returns:
        tuple: (x_start, y_start, x_end, y_end) —— 注意力图的整数网格坐标
               （含首、不含尾，可直接用于切片：att[y_start:y_end, x_start:x_end]）。
    """
    x1, y1, x2, y2 = bbox
    img_w, img_h    = image_size
    att_h, att_w    = att_shape            # 高在前、宽在后（与 numpy 下标一致）

    # --- 1. 计算一个 attention block 在原图中覆盖的像素大小 ---
    block_w = img_w / att_w               # 每列 block 覆盖的宽度
    block_h = img_h / att_h               # 每行 block 覆盖的高度

    # --- 2. bbox → 注意力网格下标 ---
    #   • 起点用 floor：只要 bbox 的像素落在这个 block 内，就算作包含
    #   • 终点用 ceil 取后减 1：确保完全覆盖 bbox
    x_start = int(np.floor(x1 / block_w))
    y_start = int(np.floor(y1 / block_h))
    x_end   = int(np.ceil (x2 / block_w))   # 不含尾：切片时直接用
    y_end   = int(np.ceil (y2 / block_h))

    # --- 3. 可选裁剪，防止越界 ---
    if clip:
        x_start = max(0, min(x_start, att_w))   # 注意尾部可等于 att_w
        y_start = max(0, min(y_start, att_h))
        x_end   = max(0, min(x_end,   att_w))
        y_end   = max(0, min(y_end,   att_h))

    return x_start, y_start, x_end, y_end


# ------------------------------------------------------------
# 获取 (H, W) 以及一次性生成 attention_mask → loss
# ------------------------------------------------------------
def get_grid_shape(processor, image_inputs):
    """
    Qwen2.5 的 image_processor 会返回字典:
    'image_grid_thw': [B, 3]  # (T, H, W)
    其中 H, W 即 patch 网格尺寸
    """
    with torch.no_grad():
        aux = processor.image_processor(images=image_inputs)
    # 因为模型中视觉 token 的 resolution 经过 patch embedding 或者 pooling 后被降采样了（比如每 patch 对应 2x2 或更大区域）。
    h, w = int(aux["image_grid_thw"][0, 1]/2), int(aux["image_grid_thw"][0, 2]/2)
    # print(f"get_grid_shape: {h}, {w}")
    return h, w

def build_mask_from_bbox(bbox, image_size, grid_shape, device="cpu"):
    """利用写好的 att_window_from_bbox() 得到网格坐标，再转成 H×W 的 0/1 mask"""
    h, w = grid_shape
    x0, y0, x1, y1 = att_window_from_bbox(bbox, image_size, (h, w))
    # print(f"att_window_from_bbox: {x0}, {y0}, {x1}, {y1}")
    mask = torch.zeros((h, w), dtype=torch.float32, device=device)
    mask[y0:y1, x0:x1] = 1.0
    return mask

def compute_activation_loss_qwen(rel_map, masks, eps=1e-6):
    """
    rel_map:   [B, H*W]  —— answer-start → image‑token 的注意力
    masks:     list[ torch.FloatTensor(H, W) ]
    return:    单个标注区域的平均 (1 - activation)^2
    """
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