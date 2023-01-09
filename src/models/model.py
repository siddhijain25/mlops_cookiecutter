from torch import Tensor, nn


class MyAwesomeModel(nn.Module):
    """
    Basic neural network
    """

    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 64, 3),  # [N, 64, 26]
            nn.LeakyReLU(),
            nn.Conv2d(64, 32, 3),  # [N, 32, 24]
            nn.LeakyReLU(),
            nn.Conv2d(32, 16, 3),  # [N, 16, 22]
            nn.LeakyReLU(),
            nn.Conv2d(16, 8, 3),  # [N, 8, 20]
            nn.LeakyReLU(),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(8 * 20 * 20, 128), nn.Dropout(), nn.Linear(128, 10)
        )

    def forward(self, x: Tensor) -> Tensor:
        """Runs inference on input x
        Args:
            x: tensor with shape [N, 1, 28, 28]

        Returns:
            log_probs: tensor with log probabilities with shape [N, 10]

        """
        if x.ndim != 4:
            raise ValueError('Expected input to a 4D tensor')
        if x.shape[1] != 1 or x.shape[2] != 28 or x.shape[3] != 28:
            raise ValueError('Expected each sample to have shape [1, 28, 28]')
        return self.classifier(self.backbone(x))


# from torch import nn, unsqueeze
# from torch.nn.modules.conv import Conv2d


# class MyAwesomeModel(nn.Module):
#     def __init__(self):
#         super().__init__()
#         # build convolutional layers
#         self.conv = nn.Sequential(
#             nn.Conv2d(1, 4, kernel_size=3, padding=1, stride=1),
#             nn.Dropout(p=0.2),
#             nn.ReLU(),
#             nn.MaxPool2d(2, 2),
#             nn.Conv2d(4, 8, 3, 1, 1),
#             nn.Dropout(p=0.2),
#             nn.ReLU(),
#             nn.Conv2d(8, 8, 3, 1, 1),
#             nn.ReLU(),
#             nn.Dropout(p=0.2),
#             nn.ReLU(),
#             nn.MaxPool2d(2, 2),
#             nn.Conv2d(8, 16, 3, 1, 1),
#             nn.Dropout(p=0.2),
#             nn.ReLU(),
#         )

#         # build fully connected layers
#         self.fc = nn.Sequential(
#             nn.Linear(7 * 7 * 16, 256),
#             nn.Dropout(p=0.2),
#             nn.ReLU(),
#             nn.Linear(256, 64),
#             nn.Dropout(p=0.2),
#             nn.ReLU(),
#             nn.Linear(64, 10),
#             nn.LogSoftmax(dim=1),
#         )

#     # forward pass
#     def forward(self, x):
#         """
#         Performs a forward pass of a neural network, given a network class.
#         """
#         x = unsqueeze(x, dim=1)
#         x = self.conv(x)
#         x = x.view(x.size(0), -1)
#         out = self.fc(x)
#         return out
