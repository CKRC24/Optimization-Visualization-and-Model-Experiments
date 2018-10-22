
# coding: utf-8

# In[1]:


import numpy as np
import math
import random
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable, grad
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as dsets
import torchvision.transforms as transforms


# In[2]:


# Hyper Parameters 
batch_size = 100
learning_rate = 0.001
# MNIST Dataset 
train_dataset = dsets.MNIST(root='./data', 
                            train=True, 
                            transform=transforms.ToTensor(),  
                            download=True)

test_dataset = dsets.MNIST(root='./data', 
                           train=False, 
                           transform=transforms.ToTensor())

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=batch_size, 
                                          shuffle=False)


# In[3]:


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
class Net(nn.Module):
    def __init__(self, hidden_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size) 
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, 10)  
        self.relu = nn.ReLU()
    def forward(self, x):
        out = self.relu(self.fc1(x))
        out = self.relu(self.fc2(out))
        out = self.relu(self.fc3(out))
        out = self.fc4(out)
        return out
dim = list(np.arange(10, 150, 7))
dim


# In[7]:


result = []
for dd in dim:
    model = Net(int(dd)).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  
    criterion = nn.CrossEntropyLoss()  
    print(model)
    params_num = count_parameters(model)
    cur_result = [params_num]
    print(params_num)
    min_loss = 1000
    for epoch in range(10):
        training_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):  
            # Convert torch tensor to Variable
            images = Variable(images.view(-1, 28*28).cuda())
            labels = Variable(labels.cuda())
            # Forward + Backward + Optimize
            optimizer.zero_grad()  # zero the gradient buffer
            outputs = model(images)
            loss = criterion(outputs, labels)
            training_loss += loss.data[0]
            loss.backward()
            optimizer.step()
        print ('[Epoch %d], Loss: %.4f' %(epoch+1, training_loss))
        if training_loss < min_loss:
            min_loss = training_loss
            best_params = model.state_dict()
    # load best model
    model.load_state_dict(best_params)
    # evaluate train and test data
    dsets = [train_loader, test_loader]
    for dset in dsets:
        correct = 0
        total = 0
        total_loss = 0.0
        for images, labels in dset:
            images = Variable(images.view(-1, 28*28).cuda())
            var_labels = Variable(labels.cuda())
            outputs = model(images)
            loss = criterion(outputs, var_labels)
            total_loss += loss.data[0]
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted.cpu() == labels).sum()
        acc = correct / total
        print(acc)
        cur_result += [total_loss, acc]
    result.append(cur_result)


# In[8]:


result = np.array(result)
# training loss
plt.scatter(result[:,0], result[:,1]/6, c="b")
# testing loss
plt.scatter(result[:,0], result[:,3], c="y")
plt.legend(['training loss','testing loss'])
plt.xlabel("# of params")
plt.ylabel("loss")
plt.savefig("./image/loss")
plt.show()
# training acc
plt.scatter(result[:,0], result[:,2], c="b")
# testing acc
plt.scatter(result[:,0], result[:,4], c="y")
plt.legend(['training accuracy','testing accuracy'])
plt.xlabel("# of params")
plt.ylabel("accuracy")
plt.savefig(".image/acc")
plt.show()

