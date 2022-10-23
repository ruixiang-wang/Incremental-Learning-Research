from torch.utils.data import Dataset
from PIL import Image
import torch
import numpy as np


mean = {'cifar100': (0.5071, 0.4867, 0.4408), 'TinyImageNet': (0.48149, 0.4577, 0.4082)}
std = {'cifar100': (0.2675, 0.2565, 0.2761), 'TinyImageNet': (0.2607, 0.2536, 0.2686)}

class BatchData(Dataset):
    def __init__(self, images, labels, input_transform=None):
        self.images = images
        self.labels = labels
        self.input_transform = input_transform

    def __getitem__(self, index):
        image = self.images[index]
        image = Image.fromarray(np.uint8(image))
        label = self.labels[index]
        if self.input_transform is not None:
            image = self.input_transform(image)
        label = torch.LongTensor([label])
        return image, label

    def __len__(self):
        return len(self.images)
