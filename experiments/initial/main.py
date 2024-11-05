from fractalnet.layers.github.fractal_net import FractalNet
from torchinfo import summary
from torchvision.models import resnet18
from torch import nn
import torch
from experiments.ssd.ssd import build_ssd_fractal, build_ssd
from experiments.ssd.data import dfg

model = build_ssd('train', dfg, 300, 201)
# model = build_ssd_fractal('train', dfg, 300, 201)

img = torch.randn((2, 3, 300, 300))
out = model(img)

# summary(model, (2, 3, 300, 300))
exit()

fractalnet = FractalNet((3, 300, 300, 201), 4, 64, 0.15, [0.1, 0.2, 0.3, 0.4], 0.5)
fractalnet.layers[-3].padding = 1

print("FRACTALNET:")
print(fractalnet)
summary(fractalnet, (1, 3, 300, 300), features_only=True)

resnet = resnet18()
resnet = nn.Sequential(*list(resnet.children())[:7])
conv4_block1 = resnet[-1][0]
conv4_block1.conv1.stride = (1, 1)
conv4_block1.conv2.stride = (1, 1)
conv4_block1.downsample[0].stride = (1, 1)

print("RESNET:")
print(resnet)
summary(resnet, (1, 3, 300, 300))
