#from turtle import forward
from numpy import block, identity
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchmetrics
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import seaborn as sn
import pytorch_lightning as pl#using pytorch lighting to shorten the code needed,
#specially the train method is very nice

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class KeyCallback(pl.callbacks.Callback):

    def on_exception(sel, trainer, pl_module):
        exit()

class block(pl.LightningModule):
    def __init__(self, in_channels, out_channels, id_downsample=None, stride=1):
        super(block, self).__init__()
        self.expansion = 4
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels*self.expansion, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(out_channels*self.expansion)
        self.relu = nn.ReLU()
        self.id_downsample = id_downsample
    
    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.conv3(x)
        x = self.bn3(x)

        if self.id_downsample is not None:
            identity = self.id_downsample(identity)
        x += identity
        x = self.relu(x)

        return x

class ResNet(pl.LightningModule):
    def __init__(self, block, layers, image_channels, classes):
        super(ResNet, self).__init__()
        self.accuracy = torchmetrics.Accuracy()
        self.valid_acc = torchmetrics.Accuracy()
        self.classes = classes
        self.num_classes = len(classes)
        self.conf_mat = torch.zeros((self.num_classes, self.num_classes), device=device)
        self.in_channels = 64
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        


        #ResNet layers
        self.layer1 = self._make_layer(block, layers[0], out_channels=64, stride=1)
        self.layer2 = self._make_layer(block, layers[1], out_channels=128, stride=2)
        self.layer3 = self._make_layer(block, layers[2], out_channels=256, stride=2)
        self.layer4 = self._make_layer(block, layers[3], out_channels=512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512*4, self.num_classes)

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        expected_shape = (batch_size, self.num_classes)
        assert x.shape == (batch_size, self.num_classes),\
            f"Expected output of forward pass to be: {expected_shape}, but got: {x.shape}"
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = F.softmax(logits, dim=1)
        self.log('train_acc_step', self.accuracy(preds, y))
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = F.cross_entropy(logits, y)
        preds = F.softmax(logits, dim=1)
        self.log('val_loss', loss)
        self.log('val_acc_step', self.accuracy(preds, y))

    def test_step(self, batch):
        x, y = batch
        logits = self.forward(x)
        loss = F.cross_entropy(logits, y)
        probs = F.softmax(logits, dim=1)
        preds = torch.min(probs, 1).indices
        self.log('test_loss', loss)
        self.log('test_acc_step', self.accuracy(preds, y))
        self.conf_mat += torchmetrics.functional.confusion_matrix(preds, y,
        num_classes=self.num_classes)

        return {"probs": probs, "labels": y}

    def test_epoch_end(self, outputs):
        print('testing epoch')
        probs = []
        y = []
        for output in outputs:
            probs.append(output["probs"])
            y.append(output["labels"])
        probs = torch.stack(probs)
        y = torch.stack(y)
        y = torch.flatten(y)
        probs = torch.reshape(probs, (y.shape[0], self.num_classes))
        precision, recall, _ = torchmetrics.functional.precision_recall_curve(probs, y,
        num_classes=self.num_classes)

        # Plot
        f, (ax1, ax2) = plt.subplots(2)
        ax1.set_title("Confusion matrix")
        sn.heatmap(self.conf_mat.detach().cpu().numpy(), annot=True, ax=ax1, fmt='.2f')
        for i, im_class in enumerate(self.classes):
            r = recall[i].detach().cpu()
            p = precision[i].detach().cpu()
            ax2.plot(r.numpy(), p.numpy(), lw=2, label='class {}'.format(im_class),)
            auprc = torchmetrics.functional.auc(r, p)
            print("AUPRC for class", i, "(" + classes[i]+"):", auprc)

        ax2.set_xlabel("recall")
        ax2.set_ylabel("precision")
        ax2.set_title("precision vs. recall curve")
        plt.show()

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.01, momentum=0.9)


    def _make_layer(self, block, num_res_blocks, out_channels, stride): #could maybe move identity downsample into the block
        identity_downsample = None
        layers = []
        
        if stride != 1 or self.in_channels != out_channels*4:
            identity_downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels*4, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels*4)
                )

        layers.append(block(self.in_channels, out_channels, identity_downsample, stride))
        self.in_channels = out_channels*4 #256
        for i in range(num_res_blocks - 1):
            layers.append(block(self.in_channels, out_channels)) #in channels 256 -> 64, 64*4 so 256 again

        return nn.Sequential(*layers)

def ResNet50(im_channels=3, classes=None):
    return ResNet(block, [3,4,6,3], im_channels, classes)
def ResNet101(im_channels=3, classes=None):
    return ResNet(block, [3,4,23,3], im_channels, classes)
def ResNet152(im_channels=3, classes=None):
    return ResNet(block, [3,8,36,3], im_channels, classes)


# class DNN(pl.LigthningModule):
#     def __init__(self, classes):
#         super(DNN, self).__init__()
#         self.classes = classes
#         self.num_classes = len(classes)
#         self.train_accuracy = torchmetrics.Accuracy()
#         self.val_accuracy = torchmetrics.Accuracy()
#         self.features = nn.ModuleList()
#         self.conv1 = nn.Conv2d(3, 6, 5)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(6, 16, 5)
#         self.fc1 = nn.Linear(16 * 5 * 5, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 10)

#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = x.view(-1, 16 * 5 * 5)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x

def load_dataset(batch_size=16, validation_fraction=0.1):
    transform = transforms.Compose([transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
    (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data',
    train=True,
    download=True,
    transform=transform)

    trainset, valset = random_split(trainset,
    [int(len(trainset)*(1-validation_fraction)),
    int(len(trainset)*validation_fraction)])

    trainloader = torch.utils.data.DataLoader(trainset,
    batch_size=batch_size,
    drop_last=True, num_workers=12)

    valloader = torch.utils.data.DataLoader(valset,
    batch_size=batch_size, num_workers=12)

    testset = torchvision.datasets.CIFAR10(root='./data',
    train=False,
    download=True,
    transform=transform)

    testloader = torch.utils.data.DataLoader(testset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=12)

    return trainloader, valloader, testloader

if __name__== '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    classes = ('plane', 'car', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck')
    #model = DNN(classes)
    trainloader, valloader, testloader = load_dataset()
    model = ResNet50(classes=classes)
    trainer = pl.Trainer(
    gpus=1,
    deterministic=True,
    max_epochs=5,
    precision=16,
    val_check_interval=0.5,
    callbacks=[KeyCallback()],
    )
    trainer.fit(model, trainloader, valloader)
    trainer.test(model, dataloaders=testloader)
    model.test_epoch_end()

