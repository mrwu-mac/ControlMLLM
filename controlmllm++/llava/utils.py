import torch
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as T
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
import cv2
from matplotlib.patches import Rectangle


def gaussian(x, mu, sigma):
    return torch.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) / (
                torch.sqrt(2 * torch.tensor(3.14159265358979323846)) * sigma)


def compute_ca_loss(rel_map, masks, choice=None, mu=0.3, object_positions=None):
    loss = 0
    object_number = len(masks)  # Single Region = 1
    if object_number == 0:
        return torch.tensor(0).float().cuda() if torch.cuda.is_available() else torch.tensor(0).float()

    attn_map = rel_map  # [1, 576]
    b = attn_map.shape[0]
    H, W = masks[0].shape  # H, W = 24, 24

    attention_map_reshaped = attn_map.reshape(b, H, W)  # [1, 24, 24]
    attention_sum = attention_map_reshaped.reshape(b, -1).sum(dim=-1, keepdim=True)

    for obj_idx in range(object_number):
        obj_loss = 0
        mask = masks[obj_idx]

        if choice and choice in ["Scribble", "Point"]:
            activation_value = (attention_map_reshaped * gaussian(mask, 0, mu)).reshape(b, -1).sum(
                dim=-1) / attention_sum
        else:
            activation_value = (attention_map_reshaped * mask).reshape(b, -1).sum(dim=-1) / attention_sum
        obj_loss += torch.mean((1 - activation_value) ** 2)
        loss += obj_loss

    return loss

def show_image_relevance(image_relevance, image, orig_image, preprocess, mask=None, only_map=False, show_mask=False,
                         att_hw=(24, 24)):
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
            mask = mask.reshape(1, 1, att_hw[0], att_hw[1])
            # mask = mask.reshape(1,1,16,16)
            mask = torch.nn.functional.interpolate(mask, size=224, mode='nearest')
            mask = mask.reshape(224, 224).cuda().data.cpu().numpy()
            mask_image = (mask * 255).astype(np.uint8)
            cv2.imwrite('vis/mask.png', mask_image)

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


def show_text_relevance(text_relevance, text_tokens, mask=None, only_map=False):
    # Create horizontal bar plot from text relevance
    def show_cam_on_text(tokens, relevance):
        # For simplicity, we'll just visualize the attention distribution across tokens
        fig, ax = plt.subplots(figsize=(10, len(tokens) * 0.5))  # Adjust the height based on the number of tokens

        # Plot the relevance as a horizontal bar chart
        ax.barh(range(len(tokens)), relevance, color='blue', height=0.7)  # height adjusts the thickness of bars
        ax.set_yticks(range(len(tokens)))
        ax.set_yticklabels(tokens)  # Display the tokens on the y-axis
        ax.set_xlabel("Attention")
        ax.set_ylabel("Tokens")

        # Invert y-axis to make the first token at the top
        ax.invert_yaxis()

        plt.tight_layout()
        return fig

    if only_map:
        # Normalize the relevance map
        text_relevance = (text_relevance - text_relevance.min()) / (text_relevance.max() - text_relevance.min())

        fig = show_cam_on_text(text_tokens, text_relevance.cuda().data.cpu().numpy())
        plt.show()

    return fig


def show_text_relevance_2d(text_relevance, text_tokens, mask=None, only_map=False):
    # print(text_relevance.shape)
    # Create heatmap from text relevance
    def show_cam_on_text(tokens, relevance):
        # For simplicity, we'll just visualize the attention distribution across tokens
        fig, ax = plt.subplots(figsize=(12, 12))  # Adjust the size for better visualization

        # Plot the relevance as a heatmap
        cax = ax.imshow(relevance, cmap="hot", interpolation="nearest")  # Use a heatmap with "hot" colormap
        fig.colorbar(cax)  # Add colorbar to represent attention scale

        ax.set_xticks(range(len(tokens)))
        ax.set_yticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=90)
        ax.set_yticklabels(tokens)

        ax.set_xlabel("Tokens")
        ax.set_ylabel("Tokens")

        # ax.invert_yaxis()

        plt.tight_layout()
        return fig

    if only_map:
        # Normalize the relevance map
        # text_relevance = (text_relevance - text_relevance.min()) / (text_relevance.max() - text_relevance.min())

        fig = show_cam_on_text(text_tokens, text_relevance.cuda().data.cpu().numpy())
        plt.show()

    return fig

