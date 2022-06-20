from tqdm import tqdm
from pathlib import Path
import pandas as pd
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets
import torchvision.datasets as datasets

datasetPath = Path('dataset')
clothMaskPath = datasetPath/'withMask/clothMask'
N95MaskPath = datasetPath/'withMask/N95MaskPath'
surgicalMaskPath = datasetPath/'withMask/surgicalMaskPath'
nonMaskPath = datasetPath/'withoutMask'
maskDF = pd.DataFrame()

for imgPath in tqdm(list(clothMaskPath.iterdir()), desc='clothMaskPath'):
    maskDF = maskDF.append({
        'image': str(imgPath),
        'mask': 1
    }, ignore_index=True)

for imgPath in tqdm(list(N95MaskPath.iterdir()), desc='N95MaskPath'):
    maskDF = maskDF.append({
        'image': str(imgPath),
        'mask': 2
    }, ignore_index=True)

for imgPath in tqdm(list(surgicalMaskPath.iterdir()), desc='surgicalMaskPath'):
    maskDF = maskDF.append({
        'image': str(imgPath),
        'mask': 3
    }, ignore_index=True)

for imgPath in tqdm(list(nonMaskPath.iterdir()), desc='nonMaskPath'):
    maskDF = maskDF.append({
        'image': str(imgPath),
        'mask': 0
    }, ignore_index=True)

test = maskDF
train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size,
                                               shuffle=True, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size,
                                              shuffle=shuffle_test, pin_memory=True)
    return train_loader, test_loader