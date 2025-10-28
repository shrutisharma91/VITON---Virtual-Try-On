import os
import io
import cv2
import numpy as np
from PIL import Image
from rembg import remove
import torch
import torch.nn.functional as F

def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def load_image_as_numpy(path, size=(256, 256)):
    """Load image and return [H,W,3] uint8 RGB numpy array resized to size."""
    img = Image.open(path).convert('RGB')
    img = img.resize(size)
    arr = np.array(img)
    return arr

def load_image_as_tensor(path, size=(256, 256), device='cpu'):
    """Load image and return normalized Tensor CxHxW float32 in [0,1]."""
    arr = load_image_as_numpy(path, size)
    t = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
    return t.to(device)

def save_tensor_as_image(tensor, path):
    """Tensor CxHxW or 1xCxHxW -> save to disk as PNG/JPG."""
    if tensor.dim() == 4:
        tensor = tensor[0]
    arr = (tensor.detach().cpu().clamp(0, 1).numpy() * 255).astype(np.uint8)
    arr = np.transpose(arr, (1, 2, 0))
    Image.fromarray(arr).save(path)

def remove_background_pil(image_path):
    """Remove background using rembg; returns PIL.Image (RGB)."""
    with open(image_path, 'rb') as f:
        input_bytes = f.read()
    result = remove(input_bytes)
    img = Image.open(io.BytesIO(result)).convert('RGBA')  # rembg returns with alpha
    # convert to RGB with white background
    bg = Image.new("RGB", img.size, (255, 255, 255))
    bg.paste(img, mask=img.split()[3])  # paste using alpha channel
    return bg

def generate_cloth_mask_from_rembg(cloth_path, out_mask_path, size=(256,256)):
    """Create binary cloth mask (0/255) using rembg alpha channel and save it."""
    with open(cloth_path, 'rb') as f:
        result = remove(f.read())
    img = Image.open(io.BytesIO(result)).convert('RGBA')
    alpha = img.split()[3].resize(size)
    alpha_arr = np.array(alpha)
    # binarize
    mask = (alpha_arr > 10).astype(np.uint8) * 255
    Image.fromarray(mask).save(out_mask_path)

def compute_affine_from_keypoints(kps, img_size=(256,256)):
    """
    Estimate a simple 2x3 affine transform for garment placement using keypoints.
    kps: list of (x, y) normalized coordinates from Mediapipe (0..1) or None.
    The function uses shoulders and hips to get center, scale and rotation.
    Returns 2x3 float32 numpy matrix suitable for cv2.warpAffine.
    """
    H, W = img_size
    # default identity
    M = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)

    if kps is None or len(kps) < 1:
        return M

    # Mediapipe landmark indices: use left_shoulder=11, right_shoulder=12, left_hip=23, right_hip=24
    # but kps here expected as list of (x,y) with same ordering; caller should pass those indices.
    try:
        left_sh = kps[11]
        right_sh = kps[12]
        left_hip = kps[23]
        right_hip = kps[24]
    except Exception:
        # fallback to first two points
        if len(kps) >= 4:
            left_sh, right_sh, left_hip, right_hip = kps[0], kps[1], kps[2], kps[3]
        else:
            return M

    # convert normalized coords to pixels
    def to_px(pt):
        return np.array([pt[0] * W, pt[1] * H], dtype=np.float32)

    sL = to_px(left_sh)
    sR = to_px(right_sh)
    hL = to_px(left_hip)
    hR = to_px(right_hip)

    # torso center
    top_center = (sL + sR) / 2.0
    bottom_center = (hL + hR) / 2.0
    center = (top_center + bottom_center) / 2.0

    # torso width & height
    width = np.linalg.norm(sR - sL)
    height = np.linalg.norm(bottom_center - top_center)

    # desired garment reference rectangle (target): we map cloth center to torso center.
    # compute angle (radians) from shoulders vector
    vec = sR - sL
    angle = np.arctan2(vec[1], vec[0])  # rotation of shoulders
    cos = np.cos(angle)
    sin = np.sin(angle)

    # scale factor (assume cloth image width ~ W). Use width/cloth_w_scale later at warp time.
    scale = max(width / (W * 0.6), 0.5)  # heuristic so garment roughly fits torso

    # build affine matrix: rotation * scale + translation to center
    M = np.array([
        [scale * cos, -scale * sin, center[0] - W / 2.0 * (scale * cos - 1)],
        [scale * sin,  scale * cos, center[1] - H / 2.0 * (scale * cos - 1)]
    ], dtype=np.float32)

    return M

def apply_affine_cv2(image_np, M, dsize=(256,256)):
    """Apply 2x3 affine matrix M (numpy) to image (H,W,3) using cv2.warpAffine."""
    warped = cv2.warpAffine(image_np, M, dsize, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return warped

def torch_affine_warp(tensor, theta, size):
    """
    Batched warp using PyTorch grid_sample.
    tensor: BxCxHxW
    theta: Bx2x3 (affine matrices)
    size: (H,W)
    """
    B, C, H, W = tensor.shape
    grid = F.affine_grid(theta, size=(B, C, size[0], size[1]), align_corners=False)
    return F.grid_sample(tensor, grid, padding_mode='border', align_corners=False)
