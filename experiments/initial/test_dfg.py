import torch
from torchvision.datasets import CocoDetection as TvCocoDetection
from torchvision.tv_tensors import BoundingBoxes
from torchvision.datasets import wrap_dataset_for_transforms_v2
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms import v2 as transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on
                                 0 dim
    """
    targets = []
    imgs = []
    for sample in batch:
        imgs.append(sample[0])
        targets.append(torch.FloatTensor(sample[1]))
    return torch.stack(imgs, 0), targets

class CocoWrapperDataset(Dataset):
    def __init__(self, dataset):
        self._dataset = dataset

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, idx):
        img, tgt = self._dataset[idx]
        return img, torch.cat((tgt['boxes']/300, tgt['labels'].unsqueeze(1)), dim=1)

transform = transforms.Compose(
    [
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize([0.4649, 0.4758, 0.4479], [0.2797, 0.2809, 0.2897]),
        transforms.Resize(300),
        transforms.RandomIoUCrop(),
        # transforms.RandomCrop((300, 300)),
        # transforms.RandomResizedCrop((300, 300)),
        transforms.Resize((300, 300)),
        transforms.SanitizeBoundingBoxes(),
    ]
)

ds = TvCocoDetection(
    "./data/dfg/JPEGImages/",
    "./data/dfg/train.json",
    transforms=transform,
)
ds = wrap_dataset_for_transforms_v2(ds, target_keys=("boxes", "labels"))
ds = CocoWrapperDataset(ds)

if __name__ == '__main__':
    data_loader = DataLoader(ds, 32, num_workers=1,
                                    shuffle=True, collate_fn=detection_collate,
                                    pin_memory=True)


    img, tgt = next(iter(data_loader))
    print(img.mean([0, 2, 3]), img.std([0, 2, 3]))
