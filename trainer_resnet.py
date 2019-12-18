import time
import os
import sys
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from dataset import Caltech101Data, caltech101_pickle, caltech256_pickle
from classifier import DIM, ClassifierResnetPretrain

output_pickle = caltech101_pickle


class Trainer():
    def __init__(self, loader, model, device):
        self.model = model
        self.loader = loader
        self.device = device
        self.model = model.to(self.device)

        if device == torch.device('cuda:0'):
            self.model.cuda()

    def train(self):
        out = []
        for batch_idx, data in enumerate(self.loader, 0):
            inputs, labels = data['image'], data['label']
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(inputs)
            for idx in range(outputs.shape[0]):
                print(outputs[idx].shape, labels[idx])
                out.append((outputs[idx], labels[idx]))
            print("Batch ", batch_idx, labels.shape)
        torch.save(out, output_pickle)


if os.path.exists(output_pickle):
    cached = torch.load(output_pickle)
    print("Found pickle ", len(cached))
    sys.exit(0)

tr = transforms.Compose([transforms.Resize((DIM, DIM))])

model = ClassifierResnetPretrain()
device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
cd = Caltech101Data('../datasets/101_ObjectCategories', device, tr)

loader = DataLoader(cd, batch_size=512, shuffle=True)
trainer = Trainer(loader, model, device)


def main():
    trainer.train()


if __name__ == '__main__':
    main()
