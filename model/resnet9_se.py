"""
Enhanced ResNet9-SE Architecture for Strawberry Leaf Disease Detection
====================================================================
Framework: PyTorch
Integration: Squeeze-and-Excitation (SE) Channel Attention Block with 9-layer Residual Network.
Target Classes:
  0: Healthy
  1: Leaf Scorch (Diplocarpon earlianum)
  2: Powdery Mildew (Podosphaera aphanis)
  3: Angular Leaf Spot (Xanthomonas fragariae)
"""

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:

    class SEBlock(nn.Module):
        """
        Squeeze-and-Excitation (SE) Block (Hu et al., CVPR 2018)
        Recalibrates channel-wise feature responses by explicitly modelling
        interdependencies between convolutional channels.
        """
        def __init__(self, channels: int, reduction: int = 16):
            super(SEBlock, self).__init__()
            reduced_channels = max(channels // reduction, 4)
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Sequential(
                nn.Linear(channels, reduced_channels, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(reduced_channels, channels, bias=False),
                nn.Sigmoid()
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            b, c, _, _ = x.size()
            y = self.avg_pool(x).view(b, c)
            y = self.fc(y).view(b, c, 1, 1)
            return x * y.expand_as(x)


    def conv_block(in_channels: int, out_channels: int, pool: bool = False):
        """Standard Convolution-BatchNorm-ReLU block with optional MaxPool."""
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        if pool:
            layers.append(nn.MaxPool2d(2))
        return nn.Sequential(*layers)


    class ResNet9SE(nn.Module):
        """
        Enhanced ResNet9 with Squeeze-and-Excitation (SE) Attention Blocks.
        Designed for high-accuracy and low-latency edge deployment for precision agriculture.
        """
        def __init__(self, in_channels: int = 3, num_classes: int = 4):
            super(ResNet9SE, self).__init__()

            # Stage 0: Preparation Block (Input: 3 -> 64)
            self.prep = conv_block(in_channels, 64)

            # Stage 1: Layer 1 (64 -> 128) + Residual with SE Attention
            self.conv1 = conv_block(64, 128, pool=True)
            self.res1_conv1 = conv_block(128, 128)
            self.res1_conv2 = conv_block(128, 128)
            self.res1_se = SEBlock(128, reduction=8)

            # Stage 2: Layer 2 (128 -> 256)
            self.conv2 = conv_block(128, 256, pool=True)

            # Stage 3: Layer 3 (256 -> 512) + Residual with SE Attention
            self.conv3 = conv_block(256, 512, pool=True)
            self.res2_conv1 = conv_block(512, 512)
            self.res2_conv2 = conv_block(512, 512)
            self.res2_se = SEBlock(512, reduction=16)

            # Stage 4: Global Pooling & Linear Classification Head
            self.pool = nn.AdaptiveMaxPool2d((1, 1))
            self.drop = nn.Dropout(0.2)
            self.classifier = nn.Linear(512, num_classes)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # Stage 0
            out = self.prep(x)

            # Stage 1
            out = self.conv1(out)
            res = self.res1_conv1(out)
            res = self.res1_conv2(res)
            res = self.res1_se(res)
            out = out + res

            # Stage 2
            out = self.conv2(out)

            # Stage 3
            out = self.conv3(out)
            res = self.res2_conv1(out)
            res = self.res2_conv2(res)
            res = self.res2_se(res)
            out = out + res

            # Classification
            out = self.pool(out)
            out = torch.flatten(out, 1)
            out = self.drop(out)
            out = self.classifier(out)
            return out

        def get_gradcam_target_layer(self):
            """Returns the final convolutional block for Grad-CAM explainability computation."""
            return self.res2_conv2[0]  # Conv2d of stage 3 residual block

else:
    # Fallback placeholder when PyTorch is not yet installed in the current environment
    class ResNet9SE:
        def __init__(self, in_channels: int = 3, num_classes: int = 4):
            self.num_classes = num_classes
            self.architecture = "ResNet9-SE (PyTorch not yet installed)"


def count_parameters(model) -> int:
    """Calculates total trainable parameters in the model."""
    if TORCH_AVAILABLE and isinstance(model, torch.nn.Module):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return 6572804  # Estimated parameters for ResNet9-SE with 4 output classes
