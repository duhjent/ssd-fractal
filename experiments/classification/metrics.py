import argparse
import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F
import numpy as np
from torchvision.models import vgg16

from torchvision.transforms import v2 as transforms

from experiments.classification.data import DFGClassification
from experiments.classification.model import create_fractalnet, create_vgg16
from models.fractalnet.fractal_net import FractalNet

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
)
import os
from os import path
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="fractal", choices=["fractal", "vgg"])
parser.add_argument("--annot_path", type=str, required=True)
parser.add_argument("--img_path", type=str, required=True)
parser.add_argument("--weight_path", type=str, required=True)
parser.add_argument("--cuda", action="store_true")
parser.add_argument("--batch_size", type=int, required=True)
parser.add_argument("--num_workers", type=int, default=0)
parser.add_argument("--out_path", type=str, required=True)
parser.add_argument("--comment", type=str, required=True)

args = parser.parse_args()


def main():
    if not path.exists(args.out_path):
        os.mkdir(args.out_path)

    device = "cuda" if args.cuda else "cpu"

    if args.model == "fractal":
        model = create_fractalnet().to(device)
    elif args.model == "vgg":
        model = create_vgg16().to(device)

    model.load_state_dict(torch.load(args.weight_path, map_location=device)['model'])

    dataset = DFGClassification(
        args.img_path,
        args.annot_path,
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

    loader = DataLoader(
        dataset, batch_size=args.batch_size, num_workers=args.num_workers
    )

    outs = []
    tgts = []

    with torch.no_grad():
        for img, tgt in tqdm(loader):
            img = img.to(device)
            tgt = F.one_hot(tgt, 200).to(torch.float32).to(device)

            tgts.append(tgt.cpu().numpy())

            out = model(img)

            outs.append(F.softmax(out, dim=1).cpu().numpy())

    outs = np.vstack(outs)
    tgts = np.vstack(tgts)
    accuracy = accuracy_score(tgts.argmax(1), outs.argmax(1))
    matrix = confusion_matrix(tgts.argmax(1), outs.argmax(1))
    precision = precision_score(tgts.argmax(1), outs.argmax(1), average="micro")
    recall = recall_score(tgts.argmax(1), outs.argmax(1), average="micro")

    torch.save(
        {
            "accuracy": accuracy,
            "confusion_matrix": matrix,
            "precision": precision,
            "recall": recall,
        },
        path.join(args.out_path, f"metrics-{args.comment}.pth"),
    )


if __name__ == "__main__":
    main()
