"""
Creates a Pytorch dataset to load the Pascal VOC dataset
"""

import torch
import os
import pandas as pd
from PIL import Image
import numpy as np


class BoneDataset(torch.utils.data.Dataset):
    def __init__(
        self, images, img_dir, label_dir, S=7, B=2, C=1, transform=None,
    ):
        self.images = images
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.S = S
        self.B = B
        self.C = C

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        label_path = os.path.join(self.label_dir, f"{self.images[index]}.txt")
        img_path = os.path.join(self.img_dir, f"{self.images[index]}.jpg")
        
        boxes = []
        class_labels = []
        
        with open(label_path) as f:
            for label in f.readlines():
                if label == '\n':
                    continue
                c, x, y, w, h = label.replace("\n", "").split()
                c = int(c)
                x, y, width, height = float(x), float(y), float(w), float(h)

                boxes.append([x, y, width, height])
                class_labels.append(c)

        image = Image.open(img_path).convert('L')
        boxes = torch.tensor(boxes)                # format: [[x, y, w, h],...]
        class_labels = torch.tensor(class_labels)  # format: [c1,....]

        if self.transform:
            transformed = self.transform(image=np.array(image), bboxes=boxes.tolist(), class_labels=class_labels.tolist())
            
            image = transformed['image']
            boxes = torch.tensor(transformed['bboxes'])
            class_labels = torch.tensor(transformed['class_labels'])
        
        boxes = torch.cat((class_labels.unsqueeze(-1), boxes), dim=1)
        
        # Convert To Cells
        label_matrix = torch.zeros((self.S, self.S, self.C + 5 * self.B))
        for box in boxes:
            class_label, x, y, width, height = box.tolist()
            class_label = int(class_label)

            # i,j represents the cell row and cell column
            i, j = int(self.S * y), int(self.S * x)
            x_cell, y_cell = self.S * x - j, self.S * y - i

            width_cell, height_cell = (
                width * self.S,
                height * self.S,
            )

            # If no object already found for specific cell i,j
            # Note: This means we restrict to ONE object
            # per cell!
            if label_matrix[i, j, self.C] == 0:
                # Set that there exists an object
                label_matrix[i, j, self.C] = 1

                # Box coordinates
                box_coordinates = torch.tensor(
                    [x_cell, y_cell, width_cell, height_cell]
                )

                label_matrix[i, j, (self.C + 1):(self.C + 5)] = box_coordinates

                # Set one hot encoding for class_label
                label_matrix[i, j, class_label] = 1
        
        return image, label_matrix