import argparse
import os
import torch
from torch import nn
from torch.nn import functional as F
import torchgeometry as tgm
from utils import gen_noise, load_checkpoint, save_images

# Dummy models for prototype (replace with real networks later)
class DummySegGenerator(nn.Module):
    def forward(self, x):
        return torch.rand(x.size(0), 13, 256, 192, device=x.device)

class DummyGMM(nn.Module):
    def forward(self, x, y):
        grid = torch.rand(x.size(0), 256, 192, 2, device=x.device) * 2 - 1
        return None, grid

class DummyALIASGenerator(nn.Module):
    def forward(self, x1, x2, x3, x4):
        B, C, H, W = x1.size()
        return torch.rand(B, 3, H, W, device=x1.device)

def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='my_small_demo')
    parser.add_argument('--dataset_dir', type=str, default='./datasets/mydata/')
    parser.add_argument('--save_dir', type=str, default='./results/')
    parser.add_argument('--seg_checkpoint', type=str, default='./checkpoints/seg_final.pth')
    parser.add_argument('--gmm_checkpoint', type=str, default='./checkpoints/gmm_final.pth')
    parser.add_argument('--alias_checkpoint', type=str, default='./checkpoints/alias_final.pth')
    return parser.parse_args()

def test(opt, seg, gmm, alias):
    os.makedirs(os.path.join(opt.save_dir, opt.name), exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load images (for now just use random tensors for demo)
    img_agnostic = torch.rand(1, 3, 256, 192, device=device)
    parse_agnostic = torch.rand(1, 13, 256, 192, device=device)
    pose = torch.rand(1, 18, 256, 192, device=device)
    c = torch.rand(1, 3, 256, 192, device=device)
    cm = torch.rand(1, 1, 256, 192, device=device)

    seg.eval(); gmm.eval(); alias.eval()

    with torch.no_grad():
        seg_input = torch.cat((cm, c * cm, parse_agnostic, pose, gen_noise(cm.size()).to(device)), dim=1)
        parse_pred_down = seg(seg_input)
        parse_pred = F.interpolate(parse_pred_down, size=(256, 192), mode='bilinear')
        parse_pred = parse_pred.argmax(dim=1)[:, None]
        parse_old = torch.zeros(parse_pred.size(0), 13, 256, 192, dtype=torch.float, device=device)
        parse_old.scatter_(1, parse_pred, 1.0)
        parse = parse_old

        _, warped_grid = gmm(img_agnostic, c)
        warped_c = F.grid_sample(c, warped_grid, padding_mode='border')
        warped_cm = F.grid_sample(cm, warped_grid, padding_mode='border')

        misalign_mask = parse[:, 2:3] - warped_cm
        misalign_mask[misalign_mask < 0.0] = 0.0
        parse_div = torch.cat((parse, misalign_mask), dim=1)
        parse_div[:, 2:3] -= misalign_mask

        output = alias(torch.cat((img_agnostic, pose, warped_c), dim=1), parse, parse_div, misalign_mask)
        save_images(output, ['output_demo.jpg'], os.path.join(opt.save_dir, opt.name))
        print(f"✅ Output saved to: {os.path.join(opt.save_dir, opt.name)}")

def main():
    opt = get_opt()
    seg = DummySegGenerator()
    gmm = DummyGMM()
    alias = DummyALIASGenerator()
    test(opt, seg, gmm, alias)

if __name__ == '__main__':
    main()
