from dataset import CocoWrapperDataset
from torchvision.datasets import CocoDetection as TvCocoDetection
from torchvision.datasets import wrap_dataset_for_transforms_v2
import torchvision.transforms.v2 as transforms
from torchvision.tv_tensors import BoundingBoxes
import torch
from torch.utils.data import DataLoader
import os
from experiments.ssd.data import detection_collate
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json

if __name__ == "__main__":
    dataset = TvCocoDetection(
        os.path.join("./data/dfg/", "JPEGImages"),
        os.path.join("./data/dfg/", "train.json"),
        # transforms=transforms.Compose(
        #     [
        #         transforms.ToImage(),
        #         transforms.ToDtype(torch.float32, scale=True),
        #         transforms.Normalize(
        #             [0.4649, 0.4758, 0.4479], [0.2797, 0.2809, 0.2897]
        #         ),
        #         transforms.Resize(300),
        #         ConditionalIoUCrop(),
        #         transforms.Resize((300, 300)),
        #         transforms.SanitizeBoundingBoxes(),
        #     ]
        # ),
    )
    # data_loader = DataLoader(
    #     dataset,
    #     4,
    #     num_workers=4,
    #     shuffle=True,
    #     collate_fn=detection_collate,
    #     # pin_memory=True,
    # )

    # plt.imshow(dataset._dataset._dataset[64][0])
    # plt.show()

    # dataset[80]

    # print(dataset._dataset._dataset[81])
    # print(dataset._dataset._dataset[82])
    # print(dataset._dataset._dataset[80])
    ds = []

    for idx in tqdm(range(len(dataset))):
        img, tgt = dataset[idx]
        if len(tgt) == 0:
            # ds.append({'id': dataset.coco.imgs[dataset.ids[idx]]['id']})
            ds.append(dataset.coco.imgs[dataset.ids[idx]])
            # print(dataset.coco.imgs[dataset.ids[idx]])
        # print(dataset.ids[idx])

    with open('./dump.json', 'a') as f:
        json.dump(ds, f)
