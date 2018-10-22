
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


# In[3]:


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


# In[4]:


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


# In[5]:


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


# In[40]:


def get_norm(loader, model):
    for data in loader:
        images, labels = data
        inputs = Variable(images, requires_grad=True)
        outputs = model(inputs)
        loss = F.nll_loss(outputs, Variable(labels))
        loss.backward()
        grad = inputs.grad
        grad_all = 0.0
        for p in grad:
            temp = 0.0
            if p is not None:
                temp = (p.cpu().data.numpy() ** 2).sum()
            grad_all += temp
        grad_norm = grad_all ** 0.5
        return grad_norm


# In[46]:


batch_list = [64, 128, 256, 512, 1024]
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


# In[53]:


test_acc = []
train_acc = []
test_loss = []
train_loss = []
sen = []
for b in batch_list:
    model = torch.load("model_" + str(b))
    a_train, l_train = get_accuracy_loss(train_loader, model)
    a_test, l_test = get_accuracy_loss(test_loader, model)
    senn = get_norm(test_loader, model)
    test_loss.append(l_test)
    test_acc.append(a_test)
    train_loss.append(l_train)
    train_acc.append(a_train)
    sen.append(senn)


# In[66]:


batch = batch_list
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot(batch, train_loss, color='b', label='train')
ax1.plot(batch, test_loss, 'b--', label='test')
ax1.set_xlabel('batch size')
ax1.set_ylabel('loss', color='b')
ax1.tick_params(axis='y', colors='b')
ax1.legend(loc=2)

ax2.plot(batch, sen, color='r', label='sensitivity')
ax2.set_ylabel('sensitivity', color='r')
ax2.tick_params(axis='y', colors='r')
ax2.legend()
plt.savefig('sensitivity_loss.png', dpi=300)
plt.show()


# In[67]:


fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot(batch, train_acc, color='b', label='train')
ax1.plot(batch, test_acc, 'b--', label='test')
ax1.set_xlabel('batch size')
ax1.set_ylabel('accuracy', color='b')
ax1.tick_params(axis='y', colors='b')
ax1.legend(loc=2)

ax2.plot(batch, sen, color='r', label='sensitivity')
ax2.set_ylabel('sensitivity', color='r')
ax2.tick_params(axis='y', colors='r')
ax2.legend()
plt.savefig('sensitivity_acc.png', dpi=300)
plt.show()


# In[68]:


m1

