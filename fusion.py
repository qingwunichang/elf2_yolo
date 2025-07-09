import torch
import torch.nn as nn
import torch.nn.functional as F

class AlphaPredictor(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),  # 全局特征
            nn.Conv2d(16, 1, kernel_size=1),  # 输出通道为1
            nn.Sigmoid()  # 保证alpha在0~1之间
        )

    def forward(self, x):
        return self.conv(x)  # 输出形状为 [B, 1, 1, 1]

class AdaptiveImageFusionModel(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.alpha_predictor = AlphaPredictor(in_channels)

    def forward(self, original, highlight):
        alpha = self.alpha_predictor(original)  # [B,1,1,1]
        return alpha * original + (1 - alpha) * highlight
