# Mute warnings for torchvision v2
import torchvision
torchvision.disable_beta_transforms_warning()

from torch.utils.data import Dataset
from PIL import Image
import glob
from torchvision import datapoints
import os
import torchvision.transforms.v2 as F  
import torch


class FacialExpressionDataset(Dataset):

    def __init__(self, device, train_transform=None, classes=os.listdir("Data/"), root_dir="Data/", resolution=1024):
        
        self.train_transform = train_transform
        self.classes = classes
        self.labels = [*range(0, len(classes))]
        self.root_dir = root_dir
        self.preprocess = F.Compose([
                F.Resize(size=(256, 256), antialias=True),
                F.PILToTensor(),
                F.ConvertImageDtype(torch.float),
            ])
        self.filelist = glob.glob(root_dir + "*/*.jpg")
        self.resolution = resolution
        self.device = device

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, index):
        image = Image.open(self.filelist[index])
        image = datapoints.Image(image.convert("RGB")).to(self.device)
        image = self.preprocess(image)

        label = self.filelist[index].split("\\")[1]
        label = self.classes.index(label)

        if self.train_transform:
            image = self.train_transform(image)

        return image, label