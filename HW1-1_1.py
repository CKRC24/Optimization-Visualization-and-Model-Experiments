
# coding: utf-8

# In[1]:


import numpy as np
import math
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# In[2]:


def flatten_param(model):
    [param.grad for param in model.parameters()]
    return torch.cat([param.grad.data.view(-1) for param in model.parameters()], 0)
# flatten_param(FuncNet3())
# [p.grad.cpu().data.numpy() for p in FuncNet3().parameters()]


# In[3]:


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# In[10]:


class FuncNet1(nn.Module):
    def __init__(self):
        super(FuncNet1, self).__init__()
        self.fc1 = nn.Linear(1, 5)
        self.fc2 = nn.Linear(5, 8)
        self.fc3 = nn.Linear(8, 10)        
        self.fc4 = nn.Linear(10, 10)
        self.fc5 = nn.Linear(10, 10)        
        self.fc6 = nn.Linear(10, 8)
        self.fc7 = nn.Linear(8, 5)
        self.fc8 = nn.Linear(5, 1)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        x = self.fc8(x)
        return x

class FuncNet2(nn.Module):
    def __init__(self):
        super(FuncNet2, self).__init__()
        self.fc1 = nn.Linear(1, 8)
        self.fc2 = nn.Linear(8, 15)        
        self.fc3 = nn.Linear(15, 14)        
        self.fc4 = nn.Linear(14, 8)
        self.fc5 = nn.Linear(8, 1)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x

class FuncNet3(nn.Module):
    def __init__(self):
        super(FuncNet3, self).__init__()
        self.fc1 = nn.Linear(1, 21)
        self.fc2 = nn.Linear(21, 20)
        self.fc3 = nn.Linear(20, 1)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# In[113]:


print(count_parameters(FuncNet1()))
print(count_parameters(FuncNet2()))
print(count_parameters(FuncNet3()))


# In[130]:


def our_func(x):
#     return math.cos(2*math.pi*x) * math.pi*x
    return math.cos(2*math.pi*x) * math.sin(math.pi*x)
x = torch.from_numpy(np.arange(1.0, 4.0, 0.001))
y = torch.from_numpy(np.array([our_func(xx) for xx in x]))


# In[170]:


# plt.plot(x.numpy(), y.numpy())
# plt.show()


# In[133]:


losses = []
lines = []
funcNets = [FuncNet1(), FuncNet2(), FuncNet3()]
criterion = nn.MSELoss()

for funcNet in funcNets:
    print(funcNet)
#     optimizer = optim.SGD(funcNet.parameters(), lr=0.002, momentum=0.95)
    optimizer = optim.Adam(funcNet.parameters())
    aloss = []
    for epoch in range(10000):  # loop over the dataset multiple times
        running_loss = 0.0
        # wrap them in Variable
        inputs, labels = Variable(x.float().view(len(x),1)), Variable(y.float().view(len(x),1))
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = funcNet(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.data[0]
        aloss.append(running_loss)
        if epoch % 1000 == 999:    # print every 2000 mini-batches
            print('[%d] loss: %.3f' %
                  (epoch + 1, running_loss))
        running_loss = 0.0
    lines.append(outputs)
    losses.append(aloss)


# In[166]:


for i, cc in enumerate(["b", "g", "r"]):
    plt.plot(list(range(8000)), losses[i][:8000], c=cc)
plt.legend(['8 layer','5 layer','3 layer'])
plt.title("loss")
plt.savefig("image/loss")
plt.show()
# plt.savefig("loss")


# In[ ]:


plt.plot(x.numpy(), y.numpy())
for i, cc in enumerate(["r", "g", "y"]):
    plt.plot(x.numpy(), lines[i].data.numpy(), c=cc)
plt.legend(['our function','8 layer','5 layer', '3 layer'])
plt.title("simulate function")
plt.savefig("image/simulate_function")
plt.show()
# plt.savefig("simulate_function")

