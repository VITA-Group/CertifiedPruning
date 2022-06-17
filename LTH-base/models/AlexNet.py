import torch.nn as nn

'''
modified to fit dataset size
'''
NUM_CLASSES = 10


class AlexNet(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU( ),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU( ),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU( ),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU( ),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU( ),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 2 * 2, 4096),
            nn.ReLU( ),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
        )
        self.fc=nn.Linear(4096, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 2 * 2)
        x = self.classifier(x)
        x=self.fc(x)
        return x



def alexnet(num_classes=10):
    return AlexNet(num_classes)