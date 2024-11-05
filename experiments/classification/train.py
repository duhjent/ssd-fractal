import argparse
import os
from os import path

import torch
from torch import nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import v2 as transforms
from tqdm import tqdm

from experiments.classification.data import DFGClassification
from experiments.classification.model import create_fractalnet, create_vgg16

class_weights = torch.tensor(
    [
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.0018408,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        0.58537791,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        0.87552174,
        1.00685,
        1.01190955,
        1.01190955,
        1.00685,
        1.00685,
        1.00685,
        1.01190955,
        1.00685,
        1.00685,
        1.01190955,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.0170202,
        1.00685,
        1.00685,
        1.00685,
        1.01190955,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.01190955,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        0.95436019,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.0018408,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        0.95436019,
        1.00685,
        1.00685,
        1.00685,
        0.99688119,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        0.97280193,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.04336788,
        1.03266667,
        1.00685,
        1.00685,
        1.0018408,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        0.84609244,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.0018408,
        0.40928862,
        1.00685,
        1.00685,
        1.00685,
        1.00685,
        1.02218274,
        1.00685,
        1.00685,
        1.29083333,
        1.29083333,
        1.65057377,
        1.11872222,
        1.06544974,
        1.08848649,
        1.28261146,
        1.24302469,
        1.39840278,
        0.6580719,
        1.0018408,
        1.00685,
        1.00685,
    ],
    dtype=torch.float32,
)


def parse_array(s: str) -> list[int]:
    return [int(x) for x in s.split(",")]


parser = argparse.ArgumentParser()
parser.add_argument("--img_path", type=str, required=True)
parser.add_argument("--annot_path", type=str, required=True)
parser.add_argument("--model", type=str, default="fractal", choices=["fractal", "vgg"])
parser.add_argument("--num_epochs", type=int, required=True)
parser.add_argument("--num_workers", type=int, default=0)
parser.add_argument("--batch_size", type=int, required=True)
parser.add_argument("--lr", type=float, required=True)
parser.add_argument("--momentum", type=float, required=True)
parser.add_argument("--cuda", action="store_true")
parser.add_argument("--out_dir", type=str, default="./weights")
parser.add_argument("--train_ratio", type=float, default=0.8)
parser.add_argument("--val_ratio", type=float, default=0.2)
parser.add_argument("--comment", type=str, default="classification")
parser.add_argument("--continue_train", action="store_true")
parser.add_argument("--weights_file", type=str)
parser.add_argument("--lr_schedule", type=parse_array, default=[50, 65])
parser.add_argument("--start_epoch", type=int, default=0)

args = parser.parse_args()


def main():
    device = "cuda" if args.cuda else "cpu"

    if args.model == "fractal":
        model = create_fractalnet().to(device)
    elif args.model == "vgg":
        model = create_vgg16().to(device)

    if args.continue_train:
        assert args.weights_file is not None
        weights = torch.load(
            path.join(args.out_dir, args.weights_file), map_location=device
        )
        model.load_state_dict(weights["model"])

    writer = SummaryWriter(comment=args.comment)

    if not path.exists(args.out_dir):
        os.mkdir(args.out_dir)

    print(args)

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

    train_ds, val_ds = random_split(dataset, [args.train_ratio, args.val_ratio])

    train_dl = DataLoader(
        train_ds, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True
    )
    val_dl = DataLoader(
        val_ds, batch_size=args.batch_size, num_workers=args.num_workers
    )

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    if args.continue_train:
        optimizer.load_state_dict(weights["optimizer"])
    criterion = nn.CrossEntropyLoss(class_weights.to(device))

    scheduler = MultiStepLR(
        optimizer,
        args.lr_schedule,
        0.1,
        last_epoch=args.start_epoch if args.continue_train else -1,
    )

    global_iter = 0

    # img, tgt = next(iter(train_dl))
    # img = img.to(device)
    # tgt = F.one_hot(tgt, 200).to(torch.float32).to(device)
    for epoch in tqdm(
        range(args.start_epoch, args.num_epochs), desc="epochs", position=0
    ):
        # for epoch in range(args.num_epochs):
        model.train()
        running_loss = 0
        # for iter_num in range(1):
        for iter_num, (img, tgt) in tqdm(
            enumerate(train_dl),
            desc="iterations",
            position=1,
            leave=False,
            total=len(train_dl),
        ):
            img = img.to(device)
            tgt = F.one_hot(tgt, 200).to(torch.float32).to(device)

            optimizer.zero_grad()
            out = model(img)

            loss = criterion(out, tgt)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            writer.add_scalar("ImmediateLoss/train", loss.item(), global_iter)
            global_iter += 1

        epoch_loss = running_loss / (iter_num + 1)
        writer.add_scalar("Loss/train", epoch_loss, epoch)

        model.eval()
        val_running_loss = 0
        with torch.no_grad():
            for iter_num, (img, tgt) in enumerate(val_dl):
                img = img.to(device)
                tgt = F.one_hot(tgt, 200).to(torch.float32).to(device)

                out = model(img)
                loss = criterion(out, tgt)
                val_running_loss = loss.item()

        writer.add_scalar("Loss/val", val_running_loss / (iter_num + 1), epoch)
        torch.save(
            {"model": model.state_dict(), "optimizer": optimizer.state_dict()},
            path.join(args.out_dir, f"{args.model}-{args.comment}.pth"),
        )
        scheduler.step()


if __name__ == "__main__":
    main()
