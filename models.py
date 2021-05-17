import torch.nn as nn

from neuron import SoftLIF


class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            # nn.Linear(256 * 2 * 2, 4096),
            nn.Linear(256, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        # x = x.view(x.size(0), 256 * 2 * 2)
        x = x.view(x.size(0), 256)
        x = self.classifier(x)
        return x


class ModifiedAlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ModifiedAlexNet, self).__init__()
        self.num_classes = num_classes
        self.extractor = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            SoftLIF(tau_ref=0.02, tau_rc=0.004, v_th=1, gamma=0.1),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            SoftLIF(tau_ref=0.02, tau_rc=0.004, v_th=1, gamma=0.1),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            SoftLIF(tau_ref=0.02, tau_rc=0.004, v_th=1, gamma=0.1),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            SoftLIF(tau_ref=0.02, tau_rc=0.004, v_th=1, gamma=0.1),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            SoftLIF(tau_ref=0.02, tau_rc=0.004, v_th=1, gamma=0.1),
            nn.AvgPool2d(kernel_size=2)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256, 4096),
            SoftLIF(tau_ref=0.02, tau_rc=0.004, v_th=1, gamma=0.1),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            SoftLIF(tau_ref=0.02, tau_rc=0.004, v_th=1, gamma=0.1),
            nn.Linear(4096, self.num_classes)
        )

    def forward(self, x):
        x = self.extractor(x)
        print(x)
        x = x.view(x.size(0), 256)
        x = self.classifier(x)
        print(x)
        return x


def alexnet(**kwargs):
    return AlexNet(**kwargs)


def modified_alexnet(**kwargs):
    return ModifiedAlexNet(**kwargs)
