# src/model.py
import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class BrainTumorClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super(BrainTumorClassifier, self).__init__()
        # Load pretrained ResNet50
        self.resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        
        # Allow fine-tuning of all layers
        for param in self.resnet.parameters():
            param.requires_grad = True
        
        # Replace the final fully connected layer
        n_inputs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(n_inputs, num_classes)

    def forward(self, x):
        return self.resnet(x)

def get_model(num_classes=2, device="cpu"):  # Force CPU usage
    model = BrainTumorClassifier(num_classes)
    model.to(device)
    return model