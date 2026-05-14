import torch
import torch.nn as nn


class CNN2DStaticModel(nn.Module):
    """
    2D CNN for new ultrasound format.
    Expected input shape: (B, 1, 4, depth)
    """

    def __init__(self, num_classes: int = 20):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(1, 51), stride=(1, 1), padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 5), stride=(1, 5)),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(1, 23), stride=(1, 1), padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 5), stride=(1, 5)),
        )

        # Keep classifier stable for variable input depth.
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 32))

        self.features = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(32 * 32, 48),
            nn.Dropout(p=0.5),
            nn.ReLU(inplace=True),
            nn.Linear(48, 20),
            nn.ReLU(inplace=True),
        )

        self.classifier = nn.Linear(20, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.adaptive_pool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.features(x)
        x = self.classifier(x)
        return x
