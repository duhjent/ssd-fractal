from experiments.ssd.ssd import build_ssd_fractal
from experiments.ssd.data import dfg as cfg
import torch

ssd_net = build_ssd_fractal("train", cfg, cfg["min_dim"], cfg["num_classes"])
data = torch.randn((2, 3, 300, 300))

ssd_net(data)
