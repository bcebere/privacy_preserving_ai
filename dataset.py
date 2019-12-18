import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

caltech101_pickle = "caltech101_resnet_pretrained.pt"
caltech256_pickle = "caltech256_resnet_pretrained.pt"

classes_101 = 102
classes_256 = 257


class Caltech101Data(Dataset):
    """ Caltech101 dataset """

    def __init__(self, inpath, device, transform=None):
        self.transform = transform
        self.path = inpath
        self.category_list = [f for f in sorted(os.listdir(self.path))]
        self.image_list = []
        for catIdx in range(0, len(self.category_list)):
            category = self.category_list[catIdx]
            folder_path = os.path.join(self.path, category)
            image_names = [img for img in sorted(os.listdir(folder_path))]

            for j, image_name in enumerate(image_names):
                self.image_list.append(
                    (catIdx,
                     os.path.join(os.path.abspath(folder_path), image_name)))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        label, image_path = self.image_list[idx]
        image = Image.open(image_path)
        image = image.convert('RGB')

        extra_transforms = transforms.Compose([transforms.ToTensor()])
        if self.transform:
            image = self.transform(image)
        image = extra_transforms(image)
        sample = {'image': image, 'label': label}

        return sample


def load_caltech101_pretrained():
    return torch.load(caltech101_pickle)


def load_caltech256_pretrained():
    return torch.load(caltech256_pickle)


DATASETS = {
    101: {
        "data": load_caltech101_pretrained(),
        "classes": classes_101,
    },
    256: {
        "data": load_caltech256_pretrained(),
        "classes": classes_256,
    },
}
