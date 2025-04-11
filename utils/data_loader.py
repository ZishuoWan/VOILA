import os
import numpy as np
import pandas as pd
import torch
from monai.transforms import (Compose, Resized, ToTensord, RandAxisFlipd,
                              RandAdjustContrastd, RandSpatialCropd, RandZoomd)
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, mode, data_path, basePath, image_shape=224, **kwargs):
        self.mode = mode
        self.kwargs = kwargs
        self.basePath = basePath
        self.datalist = pd.read_csv(data_path)
        self.transform = None
        self.image_shape = image_shape
        self.get_transform()
        

    def __len__(self):
        return self.datalist.shape[0]

    def __getitem__(self, idx):
        kwargs = {}
        filePath = self.datalist.at[idx, 'file']
        if filePath.endswith('.pt'):
            data = torch.load(os.path.join(self.basePath, filePath))
        elif filePath.endswith('.npz'):
            data = np.load(os.path.join(self.basePath, filePath))
        else:
            assert 0, "Unknown file."
        
        if self.mode == 'train':
            img = data['data']
            labels = data['seg']
            labels[labels<0] = 0
            if (img.max() - img.min()) > 100:
                img = normalizer(img)
            transform_input = {"image":img, 'labels':labels}
            transform_output = self.transform(transform_input)
            image = transform_output["image"]
            labels = transform_output["labels"].squeeze(0)
            return image.as_tensor(), labels.float(), kwargs

        elif self.mode == 'test':
            img = data['data']
            label = data['seg']
            if (img.max() - img.min()) > 100:
                img = normalizer(img)
            transform_input = {"image":img[np.newaxis,:], 'labels':label[np.newaxis,:]}
            transform_output = self.transform(transform_input)
            img = transform_output["image"].squeeze(0).float()
            label = transform_output["labels"].squeeze(0)
            return img, label, kwargs
        
    
    def get_transform(self):
        if self.mode == 'train':
            self.transform = Compose(
                    [
                        RandAxisFlipd(
                        keys=["image", "labels"],
                        prob=0.1,
                        ),
                        RandZoomd(
                            keys=["image", "labels"],
                            prob=0.1,
                            min_zoom=0.95,
                            max_zoom=1.05,
                            mode=["area", "nearest"]
                        ),
                        RandSpatialCropd(
                            keys=["image", "labels"],
                            roi_size=(self.image_shape,self.image_shape,self.image_shape),
                            random_size=False,
                        ),
                        RandAdjustContrastd(
                            keys=["image"],
                            prob=1,
                            gamma=(0.9, 1.1),
                        ),
                        Resized(
                            keys=["image", "labels"],
                            spatial_size=(self.image_shape,self.image_shape,self.image_shape),
                            mode=["area", "nearest"],
                        ),
                    ]
                )
        elif self.mode == 'test':
            self.transform = Compose(
                [
                    ToTensord(keys=["image", "labels"])
                ]
            )
        

def normalizer(image, seg = None):
    intensityproperties = {}
    intensityproperties['mean'] = -104.43730926513672
    intensityproperties['std'] = 505.3545227050781
    intensityproperties['percentile_00_5'] = -1003.0
    intensityproperties['percentile_99_5'] = 1546.0
    
    mean_intensity = intensityproperties['mean']
    std_intensity = intensityproperties['std']
    lower_bound = intensityproperties['percentile_00_5']
    upper_bound = intensityproperties['percentile_99_5']
    image = np.clip(image, lower_bound, upper_bound)
    image = (image - mean_intensity) / max(std_intensity, 1e-8)
    return image
