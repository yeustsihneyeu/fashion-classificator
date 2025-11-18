import torch.nn as nn
import torch.nn.functional as F


class FashionCNN(nn.Module):
    def __init__(self):
        super(FashionCNN, self).__init__()
        self.con1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.poll = nn.MaxPool2d(2, 2)
        self.con2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.con3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.liner1 = nn.Linear(128 * 3 * 3, 256)
        self.liner2 = nn.Linear(256, 10)
        self.drop = nn.Dropout2d(0.20)

    def forward(self, x):
        x = self.poll(F.relu(self.bn1(self.con1(x))))
        x = self.poll(F.relu(self.bn2(self.con2(x))))
        x = self.poll(F.relu(self.bn3(self.con3(x))))
        x = x.view(-1, 128 * 3 * 3)
        x = self.drop(F.relu(self.liner1(x)))
        x = self.liner2(x)
        return x


CLASSES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]
