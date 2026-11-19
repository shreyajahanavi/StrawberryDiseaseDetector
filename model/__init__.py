"""
Strawberry Disease Detection Model Package
===========================================
Enhanced ResNet9-SE Framework with Grad-CAM and Environmental Risk Awareness.
"""

from .resnet9_se import ResNet9SE, TORCH_AVAILABLE
from .predict import analyze_image, predict_disease, calculate_environmental_risk, DISEASE_CLASSES

__all__ = [
    "ResNet9SE",
    "TORCH_AVAILABLE",
    "analyze_image",
    "predict_disease",
    "calculate_environmental_risk",
    "DISEASE_CLASSES"
]
