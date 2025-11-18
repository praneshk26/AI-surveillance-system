import torch
import torch.nn as nn
import cv2

class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.ReLU(),

            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),

            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),

            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU()
        )
        
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, img_tensor):
        features = self.encoder(img_tensor)
        pooled = self.global_pool(features)
        flattened = pooled.view(pooled.size(0), -1)
        return flattened  # output dim = 128


def preprocess_image(crop):
    """
    Converts a cropped image to a PyTorch tensor (3x128x128)
    """
    if crop is None or crop.size == 0:
        return None

    img = cv2.resize(crop, (128, 128))
    img = img / 255.0
    img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)
    img = img.unsqueeze(0)  # add batch dim
    return img


# Quick test
if __name__ == "__main__":
    model = FeatureExtractor()
    dummy = torch.randn(1, 3, 128, 128)
    out = model(dummy)
    print("Feature size:", out.shape)
