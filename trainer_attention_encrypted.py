import time
import copy
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam, SGD
from torch.autograd import Variable
from torchvision import transforms
from classifier import DIM, ClassifierResnetLight, ClassifierWithAttention, classifier_bias_key, classifier_key
from crypto import KeyMaster
from dataset import DATASETS

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 1e-3

DELTA_KEEP = 0.01


class Worker():
    def __init__(self, idx, loader, optimizer, loss_function, device,
                 key_master, classes):
        self.idx = idx
        self.classes = classes
        self.loader = loader
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.device = device
        self.key_master = key_master

    def load_state_dict(self, state_dict):
        dummy = ClassifierResnetLight(self.classes).to(self.device)
        dummy.load_state_dict(state_dict)

        self.worker_model = ClassifierWithAttention(
            self.classes, DELTA_KEEP, dummy.classifier.weight,
            dummy.classifier.bias, self.device)

    def update_state_dict(self, std_dict):
        self.load_state_dict(std_dict)
        self.optimizer = SGD(self.worker_model.get_params(), lr=LEARNING_RATE)

    def generate_encrypted_delta(self):
        delta = self.worker_model.delta

        tensor_size = list(delta.shape)
        topk, indices = torch.topk(delta, int(DELTA_KEEP * tensor_size[-1]))

        filtered_delta = Variable(torch.zeros(delta.shape)).to(self.device)
        filtered_delta = filtered_delta.scatter(1, indices, topk)

        sparse_delta = filtered_delta.to_sparse()

        delta_nz_value = sparse_delta.values()
        delta_nz_encrypted = self.key_master.encrypt_tensor_to_numpy(
            delta_nz_value)

        dummy = self.key_master.encrypt(0)
        encrypted_sparse = np.full(
            shape=delta.shape, fill_value=dummy, dtype=object)
        sparse_indices = sparse_delta.indices().cpu().detach().numpy()

        for i in range(sparse_indices.shape[1]):
            encrypted_sparse[sparse_indices[0][i]][sparse_indices[1][
                i]] = delta_nz_encrypted[i]

        return encrypted_sparse

    def get_state_dict(self):
        res = self.worker_model.state_dict()

        return {
            "secure": {
                "encrypted_delta": self.generate_encrypted_delta(),
            },
            "bias": res["bias"],
        }

    def train(self, epochs, loss=None):
        for epoch in range(epochs):
            mean_loss = 0.0
            start = time.time()
            for batch_idx, data in enumerate(self.loader, 0):
                inputs, labels = data
                self.optimizer.zero_grad()

                outputs = self.worker_model(inputs)
                loss = self.loss_function(
                    outputs, labels) + self.worker_model.elastic_net()

                loss.backward()
                self.optimizer.step()

                mean_loss += loss.item()


class Aggregator():
    def __init__(self, loaders, workers, optimizer, loss_function, model,
                 device, key_master, classes):
        self.model = model
        self.classes = classes
        self.pretrain_loader = loaders[0]
        self.validate_loader = loaders[1]
        self.workers = workers
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.device = device
        self.model = model.to(self.device)
        self.key_master = key_master

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
            start = time.time()
            self.pretrain(epoch, loss)

            original_state_dict = self.model.state_dict()
            for worker in self.workers:
                worker.update_state_dict(original_state_dict)

            worker_encrypted_deltas = np.zeros(
                self.model.classifier.weight.data.shape)
            worker_bias = torch.zeros(
                self.model.classifier.bias.data.shape).to(self.device)

            for worker in self.workers:
                worker.train(100, loss)

                worker_state_dict = worker.get_state_dict()
                worker_bias += worker_state_dict["bias"]
                worker_encrypted_deltas = np.add(
                    worker_encrypted_deltas,
                    worker_state_dict["secure"]["encrypted_delta"])

            decrypted_delta = self.key_master.decrypt_nparray(
                worker_encrypted_deltas)
            decrypted_delta_tensor = torch.from_numpy(decrypted_delta).to(
                self.device)
            decrypted_delta_tensor = decrypted_delta_tensor.float()

            decrypted_delta_mean = decrypted_delta_tensor / len(self.workers)
            decrypted_private_mean = decrypted_delta_mean + self.model.classifier.weight.data

            bias_mean = worker_bias / len(self.workers)

            self.model.classifier.bias.data = bias_mean
            self.model.classifier.weight.data = decrypted_private_mean

            self.validate(epoch)
            print("Aggregation epoch took ", time.time() - start)


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
    key_master = KeyMaster()
    #for test in [101, 256]:
    for test in [256]:
        for workers_cnt in [5]:
            print("Training {} workers for {} testcase".format(
                workers_cnt, test))
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
                loader = DataLoader(
                    data[idx + 2], batch_size=512, shuffle=True)
                worker = Worker(idx, loader, optimizer, loss_function, DEVICE,
                                key_master, DATASETS[test]["classes"])
                workers.append(worker)

            aggregator = Aggregator(aggregator_loaders, workers, optimizer,
                                    loss_function, model, DEVICE, key_master,
                                    DATASETS[test]["classes"])

            start = time.time()
            aggregator.train(100)
            print("Training with {}-{} workers took {}".format(
                test, workers_cnt,
                time.time() - start))


if __name__ == '__main__':
    main()
