
# coding: utf-8

# In[39]:


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


# In[6]:


root = './data'
download = False

trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_set = datasets.MNIST(root=root, train=True, transform=trans, download=download)
test_set = datasets.MNIST(root=root, train=False, transform=trans)


# In[7]:


batch_size = 128

train_loader = torch.utils.data.DataLoader(
                dataset=train_set,
                batch_size=batch_size,
                shuffle=True)
test_loader = torch.utils.data.DataLoader(
                dataset=test_set,
                batch_size=batch_size,
                shuffle=False)


# In[12]:


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


# In[36]:


def train(epoch):
    model.train()
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

            # compute gradient norm
            grad_all = 0.0
            for p in model.parameters():
                grad = 0.0
                if p.grad is not None:
                    grad = (p.grad.cpu().data.numpy() ** 2).sum()
                grad_all += grad
            grad_norm = grad_all ** 0.5

            res_grad.append(grad_norm)
            res_loss.append(loss.data[0])
            if batch_idx % 20 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    i+1, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.data[0]))


# In[42]:


model = Net()
if use_cuda:
    model = model.cuda()
optimizer = optim.Adam(model.parameters())
res_grad = []
res_loss = []
train(20)


# In[52]:


plt.subplot(211)
plt.plot(res_grad)
plt.ylabel('grad')
plt.xlabel('iteration')
plt.subplot(212)
plt.plot(res_loss)
plt.ylabel('loss')
plt.xlabel('iteration')

plt.tight_layout()
plt.savefig('gradient_norm.png', dpi=300)
plt.show()

