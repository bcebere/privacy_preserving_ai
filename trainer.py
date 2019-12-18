import time
import torch
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam, SGD
from torchvision import transforms

from dataset import Caltech101Data, load_caltech101_pretrained, load_caltech256_pretrained, classes_101, classes_256
from classifier import DIM, ClassifierResnetLight, classifier_bias_key, classifier_key

CLASSES = classes_256
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 1e-3


class Trainer():
    def __init__(self, loader, optimizer, loss_function, model, device):
        self.model = model
        self.loader = loader
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.device = device
        self.model = model.to(self.device)

        if device == torch.device('cuda:0'):
            self.model.cuda()

    def accuracy(self, output, labels):
        with torch.no_grad():
            output = torch.argmax(output.view(-1, CLASSES), -1)
            acc = torch.mean(torch.eq(output, labels).float())
        return acc.cpu().numpy()

    def train(self, epochs, loss=None):
        for epoch in range(epochs):
            mean_loss = 0.0
            mean_acc = 0.0
            start = time.time()
            for batch_idx, data in enumerate(self.loader['train'], 0):
                inputs, labels = data
                self.optimizer.zero_grad()

                outputs = self.model(inputs)

                loss = self.loss_function(outputs,
                                          labels) + 0.1 * self.model.l2_norm()
                loss.backward()
                self.optimizer.step()

                acc = self.accuracy(outputs, labels)

                mean_loss += loss.item()
                mean_acc += acc

            print('[Train {}, {}] loss: {} acc {} took {}'.format(
                epoch + 1, batch_idx + 1, mean_loss / (batch_idx + 1),
                mean_acc / (batch_idx + 1),
                time.time() - start))

            mean_loss = 0.0
            mean_acc = 0.0
            start = time.time()

            for batch_idx, data in enumerate(self.loader['val']):
                inputs, labels = data
                with torch.no_grad():
                    outputs = self.model(inputs)
                    loss = self.loss_function(outputs, labels)
                acc = self.accuracy(outputs, labels)
                mean_loss += loss.item()
                mean_acc += acc

            print('[Validate {}, {}] loss: {} acc {} took {}'.format(
                epoch + 1, batch_idx + 1, mean_loss / (batch_idx + 1),
                mean_acc / (batch_idx + 1),
                time.time() - start))


def main():
    model = ClassifierResnetLight(CLASSES)
    cd = load_caltech256_pretrained()

    optimizer = SGD(model.parameters(), lr=LEARNING_RATE)
    loss_function = torch.nn.CrossEntropyLoss()

    total = len(cd)
    dataset_size = {'train': int(0.9 * total)}
    dataset_size["val"] = total - dataset_size["train"]
    data = {}
    data['train'], data['val'] = random_split(
        cd, [dataset_size['train'], dataset_size['val']])
    loader = {
        phase: DataLoader(data[phase], batch_size=512, shuffle=True)
        for phase in ['train', 'val']
    }

    start = time.time()
    trainer = Trainer(loader, optimizer, loss_function, model, DEVICE)
    trainer.train(2000)

    print("Training took ", time.time() - start)


if __name__ == '__main__':
    main()
