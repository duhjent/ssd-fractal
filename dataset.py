import hashlib
import json
from os import path
from zipfile import ZipFile

import matplotlib.pyplot as plt
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import tv_tensors
from torchvision.transforms import v2 as transforms
from torchvision.utils import draw_bounding_boxes

checksums = {
    "annotations.zip": "99394f7890112823880d14525c54467a",
    "test.zip": "f111a735751470c098aca9bf4d721ccf",
    "train-0.zip": "982ea17dcb412f7fe57fa15a8cf91175",
    "train-1.zip": "008028e616f4bdd26cfcf802715f29eb",
    "train-2.zip": "48fd11f9bc1048b9ffa54a95605976b5",
    "val.zip": "f1be4cb09ffcbd7c2850f7ac2ed2760f",
}

splits_files = {
    "train": ["train-0.zip", "train-1.zip", "train-2.zip"],
    "test": ["test.zip"],
    "val": ["val.zip"],
}


class GTSDBDetectionDatase(Dataset):
    def __init__(
        self, base_path="./data", split="train", transform=None, download=False
    ):
        assert split in ["train", "test"]
        self.base_path = base_path
        self.split = split
        self.transform = transform if transform is not None else transforms.ToTensor()
        self.download = download
        if self.download:
            self.__download_files()

        self.__load_ids()
        self.__extract_files()

    def __download_files(self):
        import boto3
        import yaml

        with open("./s3.yaml", "r") as f:
            config = yaml.safe_load(f)
        session = boto3.session.Session()
        client = session.client("s3", **config["s3"])

        files_to_download = ["gt.txt", "FullIJCNN2013.zip"]
        for filename in files_to_download:
            if not path.exists(f"{self.base_path}/{filename}"):
                client.download_file(
                    "gtsdb", filename, path.join(self.base_path, filename)
                )

    def __load_ids(self):
        with open(
            f"{self.base_path}/gt.txt",
            "r",
        ) as split_ids:
            self.ids = [line[:-1] for line in split_ids.readlines()]

    def __extract_files(self):
        if not path.exists(f"{self.base_path}/FullIJCNN2013.zip"):
            raise Exception(f"FullIJCNN2013.zip file not found")

        if not path.exists(f"{self.base_path}/FullIJCNN2013"):
            with ZipFile(f"{self.base_path}/FullIJCNN2013.zip", "r") as split_zip:
                split_zip.extractall(f"{self.base_path}/FullIJCNN2013")


class MTSDDetectionDataset(Dataset):
    def __init__(
        self,
        base_path="./data",
        split="train",
        transform=None,
        skip_validation=False,
        objects_filter=None,
        download=False,
    ):
        assert split in ["train", "test", "val"]
        self.base_path = base_path
        self.split = split
        self.transform = transform if transform is not None else transforms.ToTensor()
        self.objects_filter = (
            objects_filter if objects_filter is not None else lambda x: True
        )
        self.download = download
        self.skip_validation = skip_validation
        if self.download:
            self.__download_files()

        if not self.skip_validation and not self.download:
            self.__check_files()
        self.__load_ids()
        self.__extract_files()

    def __check_file(self, filename):
        checksum = checksums[filename]
        with open(f"{self.base_path}/{filename}", "rb") as binary_file:
            data = binary_file.read()
            actual_checksum = hashlib.md5(data).hexdigest()
            if actual_checksum != checksum:
                raise Exception(f"File {filename} is incorrect!")

    def __download_files(self):
        import boto3
        import yaml

        with open("./s3.yaml", "r") as f:
            config = yaml.safe_load(f)
        session = boto3.session.Session()
        client = session.client("s3", **config["s3"])

        files_to_download = ["annotations.zip"] + splits_files[self.split]
        for filename in files_to_download:
            if not path.exists(f"{self.base_path}/{filename}"):
                client.download_file(
                    "data", filename, path.join(self.base_path, filename)
                )
            elif not self.skip_validation:
                self.__check_file(filename)

    def __check_files(self):
        files_to_check = ["annotations.zip"] + splits_files[self.split]
        for filename in files_to_check:
            self.__check_file(filename)

    def __load_ids(self):
        if not path.exists(f"{self.base_path}/annotations.zip"):
            raise Exception("annotations.zip file not found")
        if not path.exists(f"{self.base_path}/annotations"):
            with ZipFile(f"{self.base_path}/annotations.zip", "r") as annotations_zip:
                annotations_zip.extractall(f"{self.base_path}/annotations")
        with open(
            f"{self.base_path}/annotations/mtsd_v2_fully_annotated/splits/{self.split}.txt",
            "r",
        ) as split_ids:
            self.ids = [line[:-1] for line in split_ids.readlines()]

    def __extract_files(self):
        self.folders = []
        for filename in splits_files[self.split]:
            if not path.exists(f"{self.base_path}/{filename}"):
                raise Exception(f"{filename} file not found")

            folder_name = filename.split(".")[0]
            self.folders.append(folder_name)
            if not path.exists(f"{self.base_path}/{folder_name}"):
                with ZipFile(f"{self.base_path}/{filename}", "r") as split_zip:
                    split_zip.extractall(f"{self.base_path}/{folder_name}")

    def __load_annotation(self, id):
        annotation_path = f"{self.base_path}/annotations/mtsd_v2_fully_annotated/annotations/{id}.json"
        with open(annotation_path, "r") as f:
            annotations = json.load(f)
            return annotations

    def __getitem__(self, index) -> any:
        id = self.ids[index]
        annotations = self.__load_annotation(id)

        for folder in self.folders:
            image_path = f"{self.base_path}/{folder}/images/{id}.jpg"
            if path.exists(image_path):
                break

        image = Image.open(image_path)

        objects = [obj for obj in annotations["objects"] if self.objects_filter(obj)]
        bboxes = [
            [
                obj["bbox"]["xmin"],
                obj["bbox"]["ymin"],
                obj["bbox"]["xmax"],
                obj["bbox"]["ymax"],
            ]
            for obj in objects
        ]
        bboxes = tv_tensors.BoundingBoxes(
            bboxes, format="XYXY", canvas_size=(image.height, image.width)
        )
        label = {
            "boxes": bboxes,
            "labels": [obj["label"] for obj in objects],
            "id": id,
        }

        return self.transform(image, label)

    def __len__(self):
        return len(self.ids)


class CocoWrapperDataset(Dataset):
    def __init__(self, dataset):
        self._dataset = dataset
        self.name = "DFG"

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, idx):
        img, tgt = self._dataset[idx]
        return img, torch.cat((tgt["boxes"] / 300, tgt["labels"].unsqueeze(1)), dim=1)


def visualize(img, label, width=3):
    img_with_bboxes = draw_bounding_boxes(img, label["boxes"], width=width)

    plt.figure(figsize=(15, 15))
    plt.imshow(transforms.functional.to_pil_image(img_with_bboxes))
    plt.axis("off")
    plt.savefig("./out.png", bbox_inches="tight", pad_inches=0)
    # plt.show()
