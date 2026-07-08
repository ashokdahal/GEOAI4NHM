"""The U-Net architecture used for burn scar mapping, together with a small helper to load it.

The architecture defined here is identical to the one built and trained in Exercise 2, so
that weights saved from that exercise can be loaded directly by this pipeline.
"""

import torch
import torch.nn as nn


def conv_block(in_channels, out_channels):
    """Two 3 by 3 convolutions, each followed by a ReLU activation."""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    def __init__(self, in_channels=6, num_classes=2):
        super().__init__()

        self.enc1 = conv_block(in_channels, 32)
        self.enc2 = conv_block(32, 64)
        self.enc3 = conv_block(64, 128)
        self.enc4 = conv_block(128, 256)
        self.pool = nn.MaxPool2d(kernel_size=2)

        self.bottleneck = conv_block(256, 512)

        self.up4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec4 = conv_block(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = conv_block(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = conv_block(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = conv_block(64, 32)

        self.classifier = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.classifier(d1)


def load_model(checkpoint_path=None, device="cpu", in_channels=6, num_classes=2):
    """Build the U-Net and optionally load trained weights from a checkpoint file.

    If checkpoint_path is not given, the network keeps its random initial weights. This is
    enough to exercise every step of the pipeline mechanically, but predictions will only be
    meaningful once weights trained as in Exercise 2 are supplied here.
    """
    model = UNet(in_channels=in_channels, num_classes=num_classes).to(device)

    if checkpoint_path is not None:
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
    else:
        print("No checkpoint provided, running with randomly initialised weights. "
              "Pass checkpoint_path once you have trained weights from Exercise 2.")

    model.eval()
    return model
