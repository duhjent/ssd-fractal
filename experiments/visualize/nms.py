import os

import matplotlib.pyplot as plt
import torch
from torchvision.datasets import CocoDetection as TvCocoDetection
from torchvision.datasets import wrap_dataset_for_transforms_v2
from torchvision.transforms import v2 as transforms
from torchvision.transforms.v2 import functional as F_cv
from torchvision.utils import draw_bounding_boxes

from dataset import CocoWrapperDataset

dataset = CocoWrapperDataset(
    wrap_dataset_for_transforms_v2(
        TvCocoDetection(
            os.path.join("./data/dfg/", "JPEGImages"),
            os.path.join("./data/dfg/", "train.json"),
            transforms=transforms.Compose(
                [
                    transforms.ToImage(),
                    # transforms.ToDtype(torch.float32, scale=True),
                    # transforms.Normalize(
                    #     [0.4649, 0.4758, 0.4479], [0.2797, 0.2809, 0.2897]
                    # ),
                    # transforms.RandomIoUCrop(sampler_options=[0.1, 0.3, 0.5, 0.7, 0.9, 1.0]),
                    transforms.Resize((300, 300)),
                    transforms.SanitizeBoundingBoxes(),
                ]
            ),
        ),
        target_keys=("boxes", "labels"),
    )
)

img, bboxes = dataset[137]
bboxes = bboxes[:, :4]
bboxes = bboxes * 300

labels = [str(i) for i in range(bboxes.size(0))]

for bbox in bboxes:
    cnt = torch.randint(3, 10, (1,))
    corrections = torch.randn((cnt, 4)) * 3
    new_bboxes = bbox.unsqueeze(0).expand(cnt, -1) + corrections
    cur_bboxes = torch.cat((bbox.unsqueeze(0), new_bboxes), dim=0)
    img = draw_bounding_boxes(img, cur_bboxes)

# img = draw_bounding_boxes(img, bboxes)

img = F_cv.to_pil_image(img)
img.save("./outs/nms_case.png")
plt.imshow(img)
plt.axis("off")
plt.show()
