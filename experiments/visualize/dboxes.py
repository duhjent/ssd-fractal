import os

import matplotlib.pyplot as plt
import torch
from PIL import Image, ImageDraw
from torchvision.datasets import CocoDetection as TvCocoDetection
from torchvision.datasets import wrap_dataset_for_transforms_v2
from torchvision.transforms import v2 as transforms
from torchvision.transforms.v2 import functional as F_cv
from torchvision.utils import draw_bounding_boxes

from dataset import CocoWrapperDataset
from experiments.ssd.data import dfg
from models.ssd.layers.box_utils import point_form
from models.ssd.layers.functions import PriorBox

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

img, tgt = dataset[135]

img = F_cv.to_pil_image(img)

grid = Image.new("RGBA", img.size, (255, 255, 255, 0))
draw = ImageDraw.Draw(grid)

# Отримуємо розміри
width, height = img.size

# Малюємо вертикальні лінії
for x in range(0, width, 100):
    draw.line([(x, 0), (x, height)], fill=(0, 0, 0), width=2)

# Малюємо горизонтальні лінії
for y in range(0, height, 100):
    draw.line([(0, y), (width, y)], fill=(0, 0, 0), width=2)

f_k = 3
points = [
    ((i + 0.5) / f_k * 300, (j + 0.5) / f_k * 300)
    for i in range(f_k)
    for j in range(f_k)
]
point_size = 3

for x, y in points:
    # Малюємо коло з заливкою
    draw.ellipse(
        [(x - point_size, y - point_size), (x + point_size, y + point_size)], fill="red"
    )

# Накладаємо сітку на оригінальне зображення
result = Image.alpha_composite(img.convert("RGBA"), grid)
result.save("./outs/dboxes/locations.png")

plt.imshow(result)
plt.axis("off")
plt.show()

# dboxes = PriorBox(dfg).build()
# dboxes_point = point_form(dboxes) * 300.0
# dboxes_point.clamp_(0, 300)

# img_with_bboxes = draw_bounding_boxes(img, dboxes_point[-4:], colors=(0, 255, 20))
# img_with_bboxes = F_cv.to_pil_image(img_with_bboxes)
# img_with_bboxes.save('./outs/dboxes/layer6.png')
# plt.imshow(img_with_bboxes)
# plt.axis('off')
# plt.show()
