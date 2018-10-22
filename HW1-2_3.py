
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


# In[133]:


# 生 training data, custom function: cos(2*pi*x) * sin(pi*x)
sampleSize = 2000
def our_func(sampleSize):
    sets = []
    x = np.random.normal(1, 0.3, sampleSize)
#     x = (np.random.random(sampleSize)*3)+1
    y = np.array([math.cos(2*math.pi*xx) * math.sin(math.pi*xx) for xx in x])
#     y = math.cos(2*math.pi*x) * math.sin(math.pi*x)
#     for i in range(sampleSize):
#         x = (random.normal()*3)+1
#         y = math.cos(2*math.pi*x) * math.sin(math.pi*x)
#         sets.append([x, y])
    return [x,y]

sets = our_func(sampleSize)
x = torch.from_numpy(sets[0])
y = torch.from_numpy(sets[1])

# 看一下 function 長相
plt.scatter(x, y)
plt.show()


# In[2]:


##### 定義 model
class FuncNet1(nn.Module):
    def __init__(self):
        super(FuncNet1, self).__init__()
        self.fc1 = nn.Linear(1, 6)
        self.fc2 = nn.Linear(6, 5)
        self.fc3 = nn.Linear(5, 3)
        self.fc4 = nn.Linear(3, 1)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x
print(FuncNet1())
# 計算 model 參數量
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# In[130]:


count_parameters(FuncNet1())


# In[17]:


def flat_grad(gradss):
    result = []
    for grads in gradss:
        this_grad = grads.contiguous().view(grads.numel())
        for grad in this_grad:
            result.append(grad.data[0])
    return result


# In[55]:


# grad_norm Loss function
from numpy import linalg as LA

def grad_norm(loss, model):
    loss_grad = grad(loss, model.parameters(), create_graph=True)
    return sum([grad.norm()**2 for grad in loss_grad]).sqrt()
def hessian(loss, model):
    loss_grad = grad(loss, model.parameters(), create_graph=True)
    hes = []
    for grads in loss_grad:
        this_grad = grads.contiguous().view(grads.numel())
        for g in this_grad:
            result = grad(g, model.parameters(), create_graph=True)
            flat = flat_grad(result)
    #         print(flat)
            hes.append(flat)
    hes = np.array(hes)
    eigvals = LA.eigvalsh(hes)
    minima_ratio = len([v for v in eigvals if v > 0]) / len(eigvals)
    return [loss.data[0], minima_ratio]


# In[147]:


import pdb
# training 參數
num_epoch = 20000
learning_rate = 0.001
path = './model' # 預設存model的資料夾
m_loss = []
counter = 0
result = []
criterion = nn.MSELoss()
# 開 train
for i in range(1000):
    model = FuncNet1()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    print("Start Training !")
    if counter >= 100:
        break
    for epoch in range(num_epoch):
        training_loss = 0.0
        inputs, labels = Variable(x.float().view(len(x),1)), Variable(y.float().view(len(x),1))
        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        if epoch > 4000:
            g_norm = grad_norm(loss, model)
            training_loss += g_norm
            optimizer.zero_grad()
            if g_norm.data[0] < 0.005 or epoch == num_epoch-1:
                counter += 1
                result.append(hessian(loss, model))
                break
            g_norm.backward()
        else:
            training_loss += loss.data[0]
            optimizer.zero_grad()
            loss.backward()
        optimizer.step()
        # print loss
        m_loss.append(training_loss)
        if epoch % 500 == 499:
            print('[Epoch: %d] Loss: %.3f' % (epoch + 1, training_loss))
    print('%d Finished Training !' % counter)


# In[141]:


xx = np.array(result)[:,0]
yy = np.array(result)[:,1]


# In[143]:


plt.scatter(yy,xx)
plt.xlabel("minima ratio")
plt.ylabel("loss")
plt.savefig("minima_ratio")
plt.show()


