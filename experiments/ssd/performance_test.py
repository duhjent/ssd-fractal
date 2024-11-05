import argparse
import time

import numpy as np
import torch

from experiments.ssd.data import dfg
from models.ssd import build_ssd, build_ssd_fractal, build_ssd_resnet


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


def custom_collation(batch):
    return (torch.stack([x[0] for x in batch]), [x[1] for x in batch])


parser = argparse.ArgumentParser(description="Test SSD performance")

parser.add_argument("--batch_size", default=32, type=int, help="Batch size for eval")
parser.add_argument(
    "--cuda", default=True, type=str2bool, help="Use CUDA to train model"
)
parser.add_argument(
    "--model",
    default="vgg",
    choices=["vgg", "fractalnet", "resnet"],
    type=str,
    help="vgg, fractalnet or resnet backbone",
)
parser.add_argument(
    "--num_repetitions", type=int, help="How many times to run the test", required=True
)
parser.add_argument(
    "--deepest",
    default=False,
    type=str2bool,
    help="Use deepest column only for FractalNet",
)

args = parser.parse_args()


def main():
    print(args)
    device = "cuda" if args.cuda else "cpu"

    if args.model == "vgg":
        model = build_ssd("test", dfg, dfg["min_dim"], dfg["num_classes"])
    if args.model == "fractalnet":
        model = build_ssd_fractal("test", dfg, dfg["min_dim"], dfg["num_classes"])
    if args.model == "resnet":
        model = build_ssd_resnet("test", dfg, dfg["min_dim"], dfg["num_classes"])

    model = model.to(device)

    data = torch.randn((args.batch_size, 3, 300, 300), device=device)

    times = []

    with torch.no_grad():
        for _ in range(args.num_repetitions):
            if args.cuda:
                torch.cuda.synchronize()
            start = time.time()
            if args.model == "fractalnet":
                model(data, deepest=args.deepest)
            else:
                model(data)
            if args.cuda:
                torch.cuda.synchronize()
            end = time.time()
            times.append(end - start)

    times = np.array(times)
    print(times.mean(), times.var())


if __name__ == "__main__":
    main()
