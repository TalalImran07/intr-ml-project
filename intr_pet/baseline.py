"""R3 baseline: ImageNet ResNet-50 fine-tuned on Oxford-IIIT Pet.  Written by our team.

The paper's baseline (Table 2) is an ImageNet-1K ResNet-50 fine-tuned on each dataset
(Pet: 89.5%). We retrain it with exactly the same images, augmentation, image sizes and
padded batches as our INTR runs, so the INTR-vs-ResNet comparison is like-for-like.

Because batches are zero-padded to the largest image (as INTR/DETR need), plain global
average pooling would average padding too. We therefore pool only over real pixels using
the same padding mask INTR uses ("masked average pooling"). With one unpadded image this
is identical to the standard ResNet-50.
"""
import torch
import torch.nn.functional as F
import torchvision
from torch import nn


class MaskedResNet50(nn.Module):
    def __init__(self, num_classes=37, imagenet=True):
        super().__init__()
        weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V1 if imagenet else None
        net = torchvision.models.resnet50(weights=weights)
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = net.layer1, net.layer2, net.layer3, net.layer4
        self.fc = nn.Linear(2048, num_classes)

    def features(self, x):
        """Last convolutional feature map [B, 2048, h, w] (used by Grad-CAM)."""
        return self.layer4(self.layer3(self.layer2(self.layer1(self.stem(x)))))

    def pool_classify(self, feat, mask):
        valid = (~F.interpolate(mask[:, None].float(), size=feat.shape[-2:]).bool()).float()
        pooled = (feat * valid).sum((2, 3)) / valid.sum((2, 3)).clamp(min=1.0)
        return self.fc(pooled)

    def forward(self, samples):
        """samples: INTR NestedTensor (tensors [B,3,H,W], mask [B,H,W], True = padding)."""
        x, mask = samples.decompose()
        return self.pool_classify(self.features(x), mask)
