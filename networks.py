import torch
import torch.nn as nn

class FuseNet(nn.Module):
    """
    Small fusion network that takes concatenated person+warped-cloth (6 channels)
    and outputs a blended image (3 channels).
    This is intentionally tiny for student demo.
    """
    def __init__(self):
        super(FuseNet, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(6, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 3, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, person_tensor, warped_cloth_tensor):
        x = torch.cat([person_tensor, warped_cloth_tensor], dim=1)  # B x 6 x H x W
        return self.net(x)
