from torchvision.transforms import v2 as transforms

from dataset import MTSDDetectionDataset, visualize

if __name__ == "__main__":
    ds = MTSDDetectionDataset(
        "./data",
        "val",
        transform=transforms.Compose(
            [
                transforms.ToImage(),
                # transforms.Resize((800, 800))
                transforms.RandomCrop((800, 800)),
            ]
        ),
        skip_validation=True,
        objects_filter=lambda obj: obj["label"] != "other-sign",
        download=True,
    )

    visualize(*ds[0], width=2)
