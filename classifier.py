import torch
import time
from torch import nn
from torchvision.models import resnet
from torchvision import transforms
import torch.nn.functional as F

DIM = 140

LAMBDA = 0.1
ALPHA = 0.01


class ClassifierBasic(nn.Module):
    def __init__(self, num_classes):
        super(ClassifierBasic, self).__init__()
        self.num_classes = num_classes
        n_features = 3 * DIM * DIM
        hidden = 500

        self.fc = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(n_features, hidden),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, input_images):
        i = input_images.view(input_images.shape[0], -1)
        return self.fc(i)


classifier_key = "classifier.weight"
classifier_bias_key = "classifier.bias"


class ClassifierResnet(nn.Module):
    def __init__(self, num_classes):
        super(ClassifierResnet, self).__init__()
        self.num_classes = num_classes

        n_features = 3 * DIM * DIM
        interm_features = 2048

        resnet152 = resnet.resnet152(pretrained=True)
        self.resnet = nn.Sequential(*list(resnet152.children())[:-1])
        self.classifier = nn.Linear(interm_features, num_classes)

        for p in self.resnet.parameters():
            p.requires_grad = False

    def forward(self, input_images):
        #resnet
        interm = self.resnet(input_images)

        #normalize
        interm = torch.squeeze(interm)
        mean = interm.mean(-1).unsqueeze(dim=1)
        std = interm.std(-1).unsqueeze(dim=1)
        interm = (interm - mean) / std

        #classify
        interm = self.classifier(interm)

        return interm


class ClassifierResnetPretrain(nn.Module):
    def __init__(self):
        super(ClassifierResnetPretrain, self).__init__()
        self.out_features = 2048

        resnet152 = resnet.resnet152(pretrained=True)
        self.resnet = nn.Sequential(*list(resnet152.children())[:-1])

        for p in self.resnet.parameters():
            p.requires_grad = False

    def forward(self, input_images):
        #resnet
        interm = self.resnet(input_images)

        #normalize
        interm = torch.squeeze(interm)
        mean = interm.mean(-1).unsqueeze(dim=1)
        std = interm.std(-1).unsqueeze(dim=1)
        return (interm - mean) / std


class ClassifierResnetLight(nn.Module):
    def __init__(self, num_classes):
        super(ClassifierResnetLight, self).__init__()
        self.num_classes = num_classes
        interm_features = 2048
        self.classifier = nn.Linear(interm_features, num_classes)

    def l1_norm(self):
        return torch.norm(self.classifier.weight, p=1)

    def l2_norm(self):
        return torch.norm(self.classifier.weight, p=2)

    def elastic_net(self):
        return LAMBDA * (
            (1 - ALPHA) * 0.5 * self.l2_norm() + ALPHA * self.l1_norm())

    def forward(self, input_images):
        input_images = F.relu(input_images)
        return self.classifier(input_images)


class ClassifierWithAttention(nn.Module):
    def __init__(self, num_classes, keep_pct, weight0, bias, device):
        super(ClassifierWithAttention, self).__init__()
        self.num_classes = num_classes
        self.num_input = 2048
        self.num_total = self.num_classes * self.num_input
        self.num_params = int(keep_pct * self.num_total)

        self.weight0 = weight0
        self.bias = bias

        self.weight0.requires_grad = False

        self.delta = torch.zeros(
            self.num_classes,
            self.num_input,
            requires_grad=True,
            device=device)

    def get_params(self):
        return [self.delta]

    def l1_norm(self):
        return torch.norm(self.delta, p=1)

    def l2_norm(self):
        return torch.norm(self.delta, p=2)

    def elastic_net(self):
        return LAMBDA * (
            (1 - ALPHA) * 0.5 * self.l2_norm() + ALPHA * self.l1_norm())

    def current_delta(self):
        return self.delta

    def current_weight(self):
        return self.weight0 + self.current_delta()

    def forward(self, input_images):
        input_images = input_images.view(-1, self.num_input)
        input_images = F.relu(input_images)

        return F.linear(input_images, self.current_weight(), self.bias)
