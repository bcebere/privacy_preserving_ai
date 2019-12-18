import time
import copy
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam, SGD
from torchvision import transforms

from dataset import load_caltech101_pretrained, load_caltech256_pretrained, classes_101, classes_256
from classifier import DIM, DATASETS, ClassifierResnetLight, classifier_bias_key, classifier_key

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 1e-3


class Worker():
    def __init__(self, idx, loader, optimizer, loss_function, device, classes):
        self.idx = idx
        self.classes = classes
        self.worker_model = ClassifierResnetLight(classes).to(device)
        self.loader = loader
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.device = device

    def load_state_dict(self, state_dict):
        self.worker_model.load_state_dict(state_dict)
        own_state = self.worker_model.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                continue
            if isinstance(param, torch.nn.Parameter):
                param = param.data
            own_state[name].copy_(param)

        return own_state

    def update_state_dict(self, std_dict):
        self.load_state_dict(std_dict)
        self.optimizer = SGD(self.worker_model.parameters(), lr=LEARNING_RATE)

    def get_state_dict(self):
        return self.worker_model.state_dict()

    def train(self, epochs, loss=None):
        for epoch in range(epochs):
            mean_loss = 0.0
            start = time.time()
            for batch_idx, data in enumerate(self.loader, 0):
                inputs, labels = data
                self.optimizer.zero_grad()

                outputs = self.worker_model(inputs)
                loss = self.loss_function(
                    outputs, labels) + 0.1 * self.worker_model.l2_norm()

                loss.backward()
                self.optimizer.step()

                mean_loss += loss.item()

            #print('[Worker {} Train {}, {}] loss: {} took {}'.format(
            #    self.idx, epoch + 1, batch_idx + 1, mean_loss / (batch_idx + 1),
            #    time.time() - start))
            #sys.stdout.flush()


class Aggregator():
    def __init__(self, loaders, workers, optimizer, loss_function, model,
                 device, classes):
        self.model = model
        self.classes = classes
        self.pretrain_loader = loaders[0]
        self.validate_loader = loaders[1]
        self.workers = workers
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.device = device
        self.model = model.to(self.device)

        if device == torch.device('cuda:0'):
            self.model.cuda()

    def accuracy(self, output, labels):
        with torch.no_grad():
            output = torch.argmax(output.view(-1, self.classes), -1)
            acc = torch.mean(torch.eq(output, labels).float())
        return acc.cpu().numpy()

    def pretrain(self, epoch, loss=None):
        mean_loss = 0.0
        mean_acc = 0.0
        start = time.time()
        for batch_idx, data in enumerate(self.pretrain_loader, 0):
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

        #print('[Pretrain {}, {}] loss: {} acc {} took {}'.format(
        #    (-1) * (epoch + 1), batch_idx + 1, mean_loss / (batch_idx + 1),
        #    mean_acc / (batch_idx + 1),
        #    time.time() - start))

    def validate(self, epoch):
        mean_loss = 0.0
        mean_acc = 0.0
        start = time.time()
        for batch_idx, data in enumerate(self.validate_loader):
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
        sys.stdout.flush()

    def train(self, epochs, loss=None):

        for epoch in range(epochs):
            self.pretrain(epoch, loss)

            original_state_dict = self.model.state_dict()
            for worker in self.workers:
                worker.update_state_dict(original_state_dict)

            worker_weights = torch.zeros(
                self.model.classifier.weight.data.shape).to(self.device)
            worker_bias = torch.zeros(
                self.model.classifier.bias.data.shape).to(self.device)

            for worker in self.workers:
                worker.train(100, loss)

                worker_state_dict = worker.get_state_dict()
                worker_bias += worker_state_dict[classifier_bias_key]
                worker_weights += worker_state_dict[classifier_key]

            weights_mean = worker_weights / len(self.workers)
            bias_mean = worker_bias / len(self.workers)

            self.model.classifier.weight.data = weights_mean
            self.model.classifier.bias.data = bias_mean

            self.validate(epoch)


def dataset_sizes(total, workers_cnt):
    pretrain_size = int(0.05 * total)
    validate_size = int(0.1 * total)
    worker_total_size = total - pretrain_size - validate_size
    per_worker = int(worker_total_size / workers_cnt)

    sizes = [pretrain_size, validate_size]
    allocated = pretrain_size + validate_size

    for w in range(workers_cnt - 1):
        allocated += per_worker
        sizes.append(per_worker)

    sizes.append(total - allocated)

    return sizes


def main():
    #workers_cnt = WORKERS_CNT
    test = 256
    for workers_cnt in [10, 50, 100]:
        print("Training {} workers for {} testcase".format(workers_cnt, test))
        model = ClassifierResnetLight(DATASETS[test]["classes"])
        cd = DATASETS[test]["data"]

        optimizer = SGD(model.parameters(), lr=LEARNING_RATE)

        loss_function = torch.nn.CrossEntropyLoss()

        sizes = dataset_sizes(len(cd), workers_cnt)

        workers = []
        data = random_split(cd, sizes)

        aggregator_loaders = [
            DataLoader(data[0], batch_size=512, shuffle=True),
            DataLoader(data[1], batch_size=512, shuffle=True)
        ]

        for idx in range(workers_cnt):
            loader = DataLoader(data[idx + 2], batch_size=512, shuffle=True)
            worker = Worker(idx, loader, optimizer, loss_function, DEVICE,
                            DATASETS[test]["classes"])
            workers.append(worker)

        aggregator = Aggregator(aggregator_loaders, workers, optimizer,
                                loss_function, model, DEVICE,
                                DATASETS[test]["classes"])

        start = time.time()
        aggregator.train(50)
        print("Training with {} workers took {}".format(
            workers_cnt,
            time.time() - start))


if __name__ == '__main__':
    main()
