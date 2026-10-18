from torch import nn


def create_model(number_of_classes: int) -> nn.Module:
    """Create a CNN whose final output matches the number of fruit classes."""
    return nn.Sequential(
        nn.Conv2d(
            in_channels=3,
            out_channels=8,
            kernel_size=3,
            padding=1,
        ),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),

        nn.Conv2d(
            in_channels=8,
            out_channels=16,
            kernel_size=3,
            padding=1,
        ),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),

        nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            padding=1,
        ),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),

        nn.Flatten(),

        nn.Linear(
            in_features=32 * 16 * 16,
            out_features=64,
        ),
        nn.ReLU(),

        nn.Linear(
            in_features=64,
            out_features=number_of_classes,
        ),
    )
