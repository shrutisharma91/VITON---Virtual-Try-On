import os
import cv2
import torch
import numpy as np
import mediapipe as mp

from datasets import TryOnDataset
from networks import FuseNet
from utils import (
    ensure_dir, load_image_as_tensor, save_tensor_as_image,
    remove_background_pil, generate_cloth_mask_from_rembg,
    compute_affine_from_keypoints, apply_affine_cv2, torch_affine_warp
)

# Config
IMG_SIZE = (256, 256)  # H, W
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PERSON_DIR = 'datasets/person'
CLOTH_DIR = 'datasets/clothes'
OUT_DIR = 'datasets/output'
USE_NETWORK_FUSION = True  # if False will use simple blending

ensure_dir(OUT_DIR)

# Mediapipe pose setup
mp_pose = mp.solutions.pose
pose_detector = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

def mediapipe_keypoints(np_img):
    """Return list of mediapipe normalized keypoints (x,y) or None."""
    # Mediapipe expects RGB
    rgb = cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB)
    results = pose_detector.process(rgb)
    if not results.pose_landmarks:
        return None
    kps = []
    for lm in results.pose_landmarks.landmark:
        kps.append((lm.x, lm.y))
    return kps

def simple_alpha_blend(person_np, warped_np, cloth_mask=None, alpha=0.7):
    """Blend warped cloth onto person. cloth_mask optional (H,W) uint8 0/255."""
    person = person_np.astype(np.float32) / 255.0
    warped = warped_np.astype(np.float32) / 255.0
    if cloth_mask is None:
        blended = person * (1 - alpha) + warped * alpha
    else:
        mask = (cloth_mask.astype(np.float32) / 255.0)[:, :, None]
        blended = person * (1 - mask) + warped * mask
    return (blended * 255).astype(np.uint8)

def main():
    dataset = TryOnDataset(PERSON_DIR, CLOTH_DIR, size=IMG_SIZE)
    fuse_net = FuseNet().to(DEVICE)
    fuse_net.eval()

    for i in range(len(dataset)):
        sample = dataset[i]
        person_np = sample['person_np'][:, :, ::-1]  # PIL->RGB->numpy gives RGB; keep as RGB for mediapipe needs
        cloth_np = sample['cloth_np'][:, :, ::-1]

        # 1) compute pose keypoints
        # Mediapipe expects BGR input, we convert:
        person_bgr = cv2.cvtColor(person_np, cv2.COLOR_RGB2BGR)
        kps = mediapipe_keypoints(person_bgr)  # normalized coordinates

        # 2) compute approximate affine matrix from keypoints
        M = compute_affine_from_keypoints(kps, img_size=IMG_SIZE)

        # 3) warp cloth using cv2 (fast and simple)
        warped_np = apply_affine_cv2(cloth_np, M, dsize=(IMG_SIZE[1], IMG_SIZE[0]))

        # Optional: create cloth mask from rembg if masks not available (slow)
        mask_path = None  # you can set path if you already generated masks
        cloth_mask = None

        # 4) fusion: either a small network (FuseNet) or simple alpha blend
        if USE_NETWORK_FUSION:
            # convert to tensors (B x C x H x W) normalized
            person_t = torch.from_numpy(person_np).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE) / 255.0
            warped_t = torch.from_numpy(warped_np).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE) / 255.0
            with torch.no_grad():
                out_t = fuse_net(person_t, warped_t)[0]
            out_path = os.path.join(OUT_DIR, f"tryon_{i}_{sample['person_name']}_{sample['cloth_name']}.jpg")
            save_tensor_as_image(out_t, out_path)
        else:
            # simple blending
            blended = simple_alpha_blend(person_np, warped_np, cloth_mask=cloth_mask, alpha=0.7)
            out_path = os.path.join(OUT_DIR, f"tryon_{i}_{sample['person_name']}_{sample['cloth_name']}.jpg")
            cv2.imwrite(out_path, cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))

        print(f"[{i+1}/{len(dataset)}] saved:", out_path)

    print("Done. Results in", OUT_DIR)

if __name__ == "__main__":
    main()
