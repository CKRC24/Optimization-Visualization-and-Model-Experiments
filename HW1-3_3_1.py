
# coding: utf-8

# In[1]:


import os
import torch
import torch.utils.data
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

use_cuda = torch.cuda.is_available()
print(use_cuda)


# In[2]:


root = './data'
download = True

trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_set = datasets.MNIST(root=root, train=True, transform=trans, download=download)
test_set = datasets.MNIST(root=root, train=False, transform=trans)


# In[5]:


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=3)
        self.fc1 = nn.Linear(20*5*5, 32)
        self.fc2 = nn.Linear(32, 10)
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


# In[56]:


def train(epoch, model, optimizer):
    model.train()
    final_loss = 0.0
    for i in range(epoch):
        for batch_idx, (data, target) in enumerate(train_loader):
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
        print('Train Epoch: {} \tLoss: {:.6f}'.format(i+1, loss.data[0]))


# In[51]:


def get_accuracy_loss(loader, model):
    correct = 0
    total = 0
    loss = 0
    for data in loader:
        images, labels = data
        outputs = model(Variable(images))
        temp = F.nll_loss(outputs, Variable(labels))
        loss += temp.data[0]
        
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
    accuracy = 100 * correct / total
    entropy = loss / total
    return accuracy, entropy


# In[57]:


batch_list = [64, 1024]
for i in batch_list:
    batch_size = i

    train_loader = torch.utils.data.DataLoader(
                    dataset=train_set,
                    batch_size=batch_size,
                    shuffle=True)
    test_loader = torch.utils.data.DataLoader(
                    dataset=test_set,
                    batch_size=batch_size,
                    shuffle=False)
    model = Net()
    if use_cuda:
        model = model.cuda()
    optimizer = optim.Adam(model.parameters())
    train(20, model, optimizer)
    
    torch.save(model, "model_" + str(i))


# In[58]:


m1 = torch.load("model_64")
m2 = torch.load("model_1024")


# In[89]:


test_list = []
train_list = []
for alpha in np.arange(-1.0, 2.01, 0.05):
    t_model = Net()
    for param, p1, p2 in zip(t_model.parameters(), m1.parameters(), m2.parameters()):
        param.data = (1-alpha)*p1.data + alpha*p2.data
    train_list.append(get_accuracy_loss(train_loader, t_model))
    test_list.append(get_accuracy_loss(test_loader, t_model))
    print(alpha)


# In[125]:


plt_train_acc = [i[0] for i in train_list]
plt_train_loss = [np.log(i[1]) for i in train_list]
plt_test_acc = [i[0] for i in test_list]
plt_test_loss = [np.log(i[1]) for i in test_list]
alpha = np.arange(-1.0, 2.01, 0.05)
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot(alpha, plt_train_loss, color='b', label='train')
ax1.plot(alpha, plt_test_loss, 'b--', label='test')
ax1.set_xlabel('alpha')
ax1.set_ylabel('loss', color='b')
ax1.tick_params(axis='y', colors='b')

ax2.plot(alpha, plt_train_acc, color='r', label='train')
ax2.plot(alpha, plt_test_acc, 'r--', label='test')
ax2.set_ylabel('accuracy', color='r')
ax2.tick_params(axis='y', colors='r')
ax2.legend()
plt.savefig('sharpness.png', dpi=300)
plt.show()

