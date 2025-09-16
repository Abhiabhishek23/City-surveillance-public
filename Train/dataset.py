# train/dataset.py
import os
import json
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision
from torchvision.transforms import functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2

class ConstructionDataset(Dataset):
    def __init__(self, root_dir, transforms=None):
        """
        root_dir: contains folders 'images/' and 'annotations/'
        transforms: optional albumentations transform
        """
        self.root_dir = root_dir
        self.transforms = transforms
        self.imgs = sorted(os.listdir(os.path.join(root_dir, "images")))
        self.annots = sorted(os.listdir(os.path.join(root_dir, "annotations")))

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        # Load image
        img_path = os.path.join(self.root_dir, "images", self.imgs[idx])
        ann_path = os.path.join(self.root_dir, "annotations", self.annots[idx])
        img = Image.open(img_path).convert("RGB")
        img = np.array(img)

        # Load annotation
        with open(ann_path) as f:
            annot = json.load(f)

        boxes = []
        labels = []
        masks = []

        for obj in annot["objects"]:
            label = obj.get("label", "construction")
            poly = obj["polygon"]  # list of [x,y] points
            # bounding box
            x_coords = [p[0] for p in poly]
            y_coords = [p[1] for p in poly]
            x_min = min(x_coords)
            y_min = min(y_coords)
            x_max = max(x_coords)
            y_max = max(y_coords)
            boxes.append([x_min, y_min, x_max, y_max])
            labels.append(1)  # 1 for construction/building
            # mask
            mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
            poly_np = np.array([poly], dtype=np.int32)
            cv2.fillPoly(mask, poly_np, 1)
            masks.append(mask)

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = torch.tensor([idx])
        target["area"] = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        target["iscrowd"] = torch.zeros((len(labels),), dtype=torch.int64)

        # Albumentations transforms
        if self.transforms:
            transformed = self.transforms(image=img, masks=masks.numpy())
            img = transformed["image"]
            masks = torch.as_tensor(transformed["masks"], dtype=torch.uint8)
            target["masks"] = masks

        # Convert to tensor
        img = F.to_tensor(img)

        return img, target

def get_train_transform():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.RandomBrightnessContrast(p=0.2),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=10, p=0.5),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']), masks=True)

def get_valid_transform():
    return A.Compose([
        ToTensorV2()
    ])
