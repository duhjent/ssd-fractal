import argparse
import os

import torch
from coco_eval import CocoEvaluator
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection as TvCocoDetection
from torchvision.datasets import wrap_dataset_for_transforms_v2
from torchvision.transforms import v2 as transforms
from tqdm import tqdm

from experiments.ssd.data import dfg
from models.ssd import build_ssd, build_ssd_fractal, build_ssd_resnet


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


def custom_collation(batch):
    return (torch.stack([x[0] for x in batch]), [x[1] for x in batch])


parser = argparse.ArgumentParser(description="Test SSD performance")

parser.add_argument(
    "--dataset_root",
    type=str,
    default=None,
    help="Dataset root directory path",
    required=True,
)
parser.add_argument("--batch_size", default=32, type=int, help="Batch size for eval")
parser.add_argument(
    "--num_workers", default=4, type=int, help="Number of workers used in dataloading"
)
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
    "--weights", default=None, required=True, type=str, help="Weight file to load"
)
parser.add_argument(
    "--conf_threshold",
    default=0.5,
    type=float,
    help="Confidence threshold to for detections to count into the results",
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
    dataset = wrap_dataset_for_transforms_v2(
        TvCocoDetection(
            os.path.join(args.dataset_root, "JPEGImages"),
            os.path.join(args.dataset_root, "test.json"),
            transforms=transforms.Compose(
                [
                    transforms.ToImage(),
                    transforms.ToDtype(torch.float32, scale=True),
                    transforms.Normalize(
                        [0.4649, 0.4758, 0.4479], [0.2797, 0.2809, 0.2897]
                    ),
                    transforms.Resize((300, 300)),
                ]
            ),
        ),
        target_keys=("boxes", "labels", "image_id"),
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=custom_collation,
    )

    evaluator = CocoEvaluator(coco_gt=dataset._dataset.coco, iou_types=["bbox"])

    device = "cuda" if args.cuda else "cpu"

    if args.model == "vgg":
        model = build_ssd("test", dfg, dfg["min_dim"], dfg["num_classes"])
    if args.model == "fractalnet":
        model = build_ssd_fractal("test", dfg, dfg["min_dim"], dfg["num_classes"])
    if args.model == "resnet":
        model = build_ssd_resnet("test", dfg, dfg["min_dim"], dfg["num_classes"])

    model = model.to(device)

    model.load_state_dict(torch.load(args.weights, map_location=device))

    model.eval()

    with torch.no_grad():
        # iterator = iter(loader)
        # for _ in range(1):
        for img, tgt in tqdm(loader):
            # img, tgt = next(iterator)
            img = img.to(device)
            if model == "fractalnet":
                detections = model(img, deepest=args.deepest)
            else:
                detections = model(img)
            outs = []
            for img_idx in range(img.size(0)):
                coco_img = dataset._dataset.coco.imgs[tgt[img_idx]["image_id"]]
                scale = torch.tensor(
                    [
                        coco_img["width"],
                        coco_img["height"],
                        coco_img["width"],
                        coco_img["height"],
                    ],
                    device=device,
                )
                for i in range(detections.size(1)):
                    j = 0
                    while detections[img_idx, i, j, 0] >= args.conf_threshold:
                        score = detections[img_idx, i, j, 0].item()
                        pt = detections[img_idx, i, j, 1:] * scale
                        pt = pt.tolist()
                        bbox = [pt[0], pt[1], pt[2] - pt[0] + 1, pt[3] - pt[1] + 1]
                        outs.append(
                            {
                                "image_id": tgt[img_idx]["image_id"],
                                "category_id": i - 1,
                                "bbox": bbox,
                                "score": score,
                            }
                        )
                        j += 1

    evaluator.update(outs)

    evaluator.synchronize_between_processes()
    evaluator.accumulate()
    evaluator.summarize()

    return


if __name__ == "__main__":
    main()
