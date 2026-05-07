# likely need to random-crop/use a small obj detector for 128x128 image
import torch
import torch.nn as nn

class DepthwiseSearchableBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size=3, stride=1, padding=1):
        super(DepthwiseSearchableBlock, self).__init__()
        self.depthwisearchable = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=kernel_size,
                padding=padding,
                stride=stride,
                groups=in_channels,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                bias=False
            ),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.depthwisearchable(x)

# 3 depthwise, global avg pooling, final linear layer
class PoseRegressorModel(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_features:int = 5):
        super(PoseRegressorModel, self).__init__()

        self.features = nn.Sequential(
            DepthwiseSearchableBlock(in_channels, hidden_channels, 3, 2, 1),
            DepthwiseSearchableBlock(hidden_channels, hidden_channels*2, 3, 2, 1),
            DepthwiseSearchableBlock(hidden_channels*2, hidden_channels*4, 3, 2, 1)
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.linear = nn.Linear(
            in_features=hidden_channels*4,
            out_features=out_features,
            bias=True
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)

        # 4D => 2D flatten
        x = torch.flatten(x, 1)
        
        return self.linear(x)