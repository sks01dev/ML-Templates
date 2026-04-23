import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    """(Convolution => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_classes=2):
        super(UNet, self).__init__()

        # Encoder (Downsampling)
        self.down1 = DoubleConv(in_channels, 64)
        self.down2 = DoubleConv(64, 128)
        self.down3 = DoubleConv(128, 256)
        self.down4 = DoubleConv(256, 512)
        self.bottleneck = DoubleConv(512, 1024)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Decoder (Upsampling)
        self.up_trans1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.up_conv1 = DoubleConv(1024, 512) # 512 (up) + 512 (skip connection)

        self.up_trans2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up_conv2 = DoubleConv(512, 256)

        self.up_trans3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up_conv3 = DoubleConv(256, 128)

        self.up_trans4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up_conv4 = DoubleConv(128, 64)

        # Final Output Layer
        self.final_conv = nn.Conv2d(64, out_classes, kernel_size=1)

    def forward(self, x):
        # Encoder path
        c1 = self.down1(x)
        p1 = self.pool(c1)

        c2 = self.down2(p1)
        p2 = self.pool(c2)

        c3 = self.down3(p2)
        p3 = self.pool(c3)

        c4 = self.down4(p3)
        p4 = self.pool(c4)

        # Bottleneck
        b = self.bottleneck(p4)

        # Decoder path with Skip Connections
        d1 = self.up_trans1(b)
        # torch.cat combines the upsampled features with encoder features
        u1 = self.up_conv1(torch.cat([d1, c4], dim=1))

        d2 = self.up_trans2(u1)
        u2 = self.up_conv2(torch.cat([d2, c3], dim=1))

        d3 = self.up_trans3(u2)
        u3 = self.up_conv3(torch.cat([d3, c2], dim=1))

        d4 = self.up_trans4(u3)
        u4 = self.up_conv4(torch.cat([d4, c1], dim=1))

        return self.final_conv(u4)

if __name__ == "__main__":
    # Test with standard input (Batch, Channels, H, W)
    # Using 512x512 instead of 572x572 for simplicity
    image = torch.randn((1, 1, 512, 512))
    model = UNet(in_channels=1, out_classes=2)
    output = model(image)
    print(f"Input Shape: {image.shape}")
    print(f"Output Shape: {output.shape}")
