import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import v2 as transforms

from experiments.classification.data import DFGClassification
from models.fractalnet.fractal_net import FractalNet


def main():
    model = FractalNet(
        data_shape=(3, 64, 64, 200),
        n_columns=4,
        init_channels=128,
        p_ldrop=0.15,
        dropout_probs=[0, 0.1, 0.2, 0.3, 0.4],
        gdrop_ratio=0.5,
    )

    dataset = DFGClassification(
        "./data/dfg/JPEGImages",
        "./data/dfg/train.bak.json",
        transform=transforms.Compose(
            [
                transforms.ToImage(),
                transforms.ToDtype(torch.float32, scale=True),
                transforms.Normalize(
                    [0.4397, 0.4331, 0.4526], [0.2677, 0.2707, 0.2906]
                ),
                transforms.Resize((64, 64)),
            ]
        ),
    )

    loader = DataLoader(dataset, batch_size=64)

    criterion = nn.CrossEntropyLoss()

    img, tgt = next(iter(loader))
    tgt = F.one_hot(tgt, 200).to(torch.float32)

    out = model(img)

    loss = criterion(out, tgt)
    loss.backward()

    grads = {
        name: param.grad.data.view(-1).cpu().clone().numpy()
        for name, param in model.named_parameters()
        if "weight" in name and "conv" in name and param.grad is not None
    }

    fig, axs = plt.subplots(1, 5, figsize=(20, 5))
    fig.tight_layout()

    for i, ax in enumerate(axs):
        layer_key = f"layers.{i * 2}.columns.3.7.conv.weight"
        sns.histplot(data=grads[layer_key], bins=30, kde=True, ax=ax)

    plt.show()


if __name__ == "__main__":
    main()
