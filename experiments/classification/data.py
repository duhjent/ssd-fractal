import json
from os import path

import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms.v2 import functional


class DFGClassification(Dataset):
    def __init__(
        self, img_path: str, annot_path: str, transform=None, target_transform=None
    ):
        self.img_path = img_path
        self.annot_path = annot_path
        with open(annot_path, "r") as annot_file:
            metadata = json.load(annot_file)
        self.images = {img["id"]: img["file_name"] for img in metadata["images"]}
        self.ids = [
            {
                "class_id": annot["category_id"],
                "file_name": self.images[annot["image_id"]],
                "bbox": annot["bbox"],
            }
            for annot in metadata["annotations"]
            if annot['area'] != 1 and annot['bbox'][2] > 0 and annot['bbox'][3] > 0
        ]

        self.tranform = transform
        self.target_transform = target_transform

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        annotation = self.ids[idx]
        img = read_image(path.join(self.img_path, annotation["file_name"]))
        img = functional.crop(
            img,
            annotation["bbox"][1],
            annotation["bbox"][0],
            annotation["bbox"][3],
            annotation["bbox"][2],
        )
        tgt = annotation["class_id"]

        if self.tranform is not None:
            img = self.tranform(img)
        if self.target_transform is not None:
            tgt = self.target_transform(tgt)

        return img, tgt
