import torch
import torch.nn as nn
import torch.nn.functional as F

class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        
        # Convolutional Feature Extractor
        self.features = nn.Sequential(
            # Layer 1
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            # Layer 2
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            # Layer 3
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            
            # Layer 4
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        
        # Fully Connected Classifier
        self.classifier = nn.Sequential(
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        # x input size: (batch_size, 3, 227, 227)
        x = self.features(x)
        
        # Flatten the output of convolutions for the dense layers
        x = x.view(x.size(0), -1) # size: (batch_size, 9216)
        
        x = self.classifier(x)
        
        # Use softmax for multi-class probability (dim=1 is the class dimension)
        return F.softmax(x, dim=1)

if __name__ == "__main__":
    # Create model instance
    model = AlexNet(num_classes=1000)
    
    # Generate a dummy batch of 1 image: (Batch, Channels, Height, Width)
    dummy_input = torch.randn(1, 3, 227, 227)
    
    # Forward pass
    output = model(dummy_input)
    print(f"Output shape: {output.shape}") # Should be (1, 1000)
