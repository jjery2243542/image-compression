import torch 
import torch.nn as nn
import torch.nn.functional as F
from utils import cc

class Encoder(nn.Module):
    def __init__(self, w=128, h=128, in_channel=3, out_channel=64):
        super(Encoder, self).__init__()
        self.cnn_layer1 = nn.Sequential(
                nn.ReflectionPad2d(2),
                nn.Conv2d(in_channel, 64, kernel_size=5, stride=2),
                nn.ReLU(),
                nn.ReflectionPad2d(2),
                nn.Conv2d(64, 128, kernel_size=5, stride=2),
                nn.ReLU()
            )

        self.cnn_layer2s = nn.ModuleList([nn.Conv2d(128, 128, kernel_size=3) for _ in range(3)])

        self.cnn_layer3 = nn.Sequential(
                nn.Conv2d(128, out_channel, kernel_size=3, stride=2),
                nn.ReLU()
            )

    def forward(self, x):
        out = self.cnn_layer1(x)
        for layer in self.cnn_layer2s:
            pad_out = F.pad(out, pad=(1,1,1,1), mode='reflect')
            res = F.relu(layer(pad_out))
            out = res + out
        out = F.pad(out, pad=(1,1,1,1), mode='reflect')
        out = self.cnn_layer3(out)
        return out

class Decoder(nn.Module):
    def __init__(self, w=128, h=128, in_channel=64, out_channel=3):
        super(Decoder, self).__init__()
        self.conv_layer1 = nn.Sequential(
                nn.ConvTranspose2d(64, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.ReLU()
            )
        self.conv_layer2s = nn.ModuleList(
                [nn.ConvTranspose2d(128, 128, kernel_size=3, padding=1) for _ in range(3)])
        self.conv_layer3 = nn.Sequential(
                nn.ConvTranspose2d(128, 64, stride=2, kernel_size=5, padding=2, output_padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(64, out_channel, stride=2, kernel_size=5, padding=2, output_padding=1)
            )

    def forward(self, x):
        out = self.conv_layer1(x)
        for layer in self.conv_layer2s:
            res = F.relu(layer(out))
            out = out + res
        out = self.conv_layer3(out)
        return out

if __name__ == '__main__':
    enc = cc(Encoder())
    dec = cc(Decoder())
    data = cc(torch.randn(16, 3, 128, 128))
    e = enc(data)
    d = dec(e)
    print(d.size())
