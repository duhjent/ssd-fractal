import matplotlib.pyplot as plt
import PIL
import torch
from torchvision.datasets import CocoDetection, wrap_dataset_for_transforms_v2
from torchvision.transforms import v2
from torchvision.utils import draw_bounding_boxes


def main():
    dataset = wrap_dataset_for_transforms_v2(
        CocoDetection(
            "./data/dfg/JPEGImages",
            "./data/dfg/train.json",
            transforms=v2.Compose([v2.ToImage()]),
        ),
        target_keys=("boxes", "labels"),
    )

    img, tgt = dataset[76]

    img = draw_bounding_boxes(img, tgt["boxes"], width=5)

    img = v2.functional.to_pil_image(img)

    img.save("./outs/dfg-example.png")
    return


if __name__ == "__main__":
    main()
