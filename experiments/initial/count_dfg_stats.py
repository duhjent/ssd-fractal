from torchvision.datasets import CocoDetection as TvCocoDetection
from torchvision.transforms import v2 as transforms
from torch.utils.data import DataLoader, Dataset
import torch
from tqdm import tqdm
import math


class IgnoreTgtDataset(Dataset):
    def __init__(self, dataset):
        self._dataset = dataset

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, index):
        return self._dataset[index][0]


if __name__ == "__main__":
    ds = TvCocoDetection(
        "./data/dfg/JPEGImages/",
        "./data/dfg/train.json",
        transform=transforms.Compose(
            [
                transforms.ToImage(),
                transforms.ToDtype(torch.float32, scale=True),
            ]
        ),
    )
    ds = IgnoreTgtDataset(ds)

    dl = DataLoader(ds, batch_size=1)

    channels_sum, channels_squares_sum, num_batches = 0, 0, 0

    for img in tqdm(dl):
        img = img.float()
        channels_sum += torch.mean(img, dim=[0, 2, 3])
        channels_squares_sum += torch.mean(img**2, dim=[0, 2, 3])
        num_batches += 1

        # if num_batches == 2:
        #     break

    mean = channels_sum / num_batches
    std = torch.sqrt(channels_squares_sum / num_batches - mean**2)

    print(mean, std)
    print(mean * 255, std * 255)
