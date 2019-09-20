import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

def get_data():
	transform = transforms.Compose(
	    [transforms.ToTensor(),
	     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

	trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
	                                        download=True, transform=transform)
	trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
	                                          shuffle=True, num_workers=2)

	testset = torchvision.datasets.CIFAR10(root='./data', train=False,
	                                       download=True, transform=transform)
	testloader = torch.utils.data.DataLoader(testset, batch_size=4,
	                                         shuffle=False, num_workers=2)

	classes = ('plane', 'car', 'bird', 'cat',
	           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

class SelectiveNet(nn.Module):
    def __init__(self):
        super(SelectiveNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 42)
        
        self.classification_head1 = nn.Linear(42, 10)
        
        self.selection_head1 = nn.Linear(42, 42)
        self.selection_head2 = nn.Linear(42, 1)
        self.selection_head3 = nn.Sigmoid()
        
        self.auxiliary_head1 = nn.Linear(42, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        # calculate classification head
        f = self.classification_head1(x)
        # calculate selection head
        g = F.relu(self.selection_head1(x))
        g = self.selection_head2(g)
        g = self.selection_head3(g)
        # calculate auxiliary head              
        h = self.auxiliary_head1(x)
        return (f, g, h)

snet = SelectiveNet()
criterion1 = nn.CrossEntropyLoss()
criterion2 = nn.CrossEntropyLoss()
optimizer = optim.SGD(snet.parameters(), lr=0.001, momentum=0.9)

def train():
    running_loss = 0.0
    alpha = 0.5
    lambd = 32
    c = 0.8
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        f, g, h = snet(inputs)
        coverage = torch.mean(g)
        risk = torch.mean(criterion1(f, labels) * g) / coverage
        fgLoss = risk + lambd * (torch.max(c - coverage, 0))[0].pow(2)
        hLoss = criterion2(h, labels)
        
        loss = alpha * fgLoss + (1 - alpha) * hLoss
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0
