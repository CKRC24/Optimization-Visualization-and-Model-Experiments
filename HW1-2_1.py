
# coding: utf-8

# In[96]:


# %matplotlib inline
import torch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import numpy as np
import Conv_Model as CNN
from sklearn.decomposition import PCA
import math


# In[ ]:


# preprocess
# 將PIL Image 轉成 Tensor 並 Normalize (mean: 0.5, std: 0.5) 
preprocess = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

# 準備 dataset, Train 有 shuffle, Test 無 shuffle
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=preprocess)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=100, shuffle=True, num_workers=2)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=preprocess)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=2)

# 定義 Image Class
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# In[ ]:


# 將 model 參數 flatten 成一個一維的 tensor
def flatten_param(model):
    return torch.cat([param.data.view(-1) for param in model.parameters()], 0).cpu().numpy()

# 將 model 中其中一層參數(第一層FC) flatten
def getLayerParam(model, num_layer=6):
    params = list(model.parameters())
    return params[6].data.view(-1).cpu().numpy()


# In[ ]:


# training 參數
train_event = 8
num_epoch = 45
stop_per_epoch = 3
learning_rate = 0.001
accuracy = []
# m_weights = []
# l_weights = []
criterion = nn.CrossEntropyLoss()


# In[ ]:


# 開 train
for i in range(train_event):
    print("Start Training Event[" + str(i) + "]")
    model = CNN.Conv_3()
    model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    m_acc = []  
    for epoch in range(num_epoch):
        ac = 0.0
        training_loss = 0.0
        correct = 0
        total = 50000
        for i, data in enumerate(train_loader, 0):
            # get inputs and wrap them into variable
            inputs, labels = data
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())

            # zero the gradient
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # 記錄多少 predict 對的
            
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels.data).sum()

            # 更新gradient
            loss.backward()
            optimizer.step()

            # print loss
            training_loss += loss.data[0]
            
        # print loss, accuracy per epoch
        training_loss = training_loss / 500
        ac = 100 * correct/float(total)
        print('[Epoch: %d] Loss: %.3f, Accuracy: %.3f%%' %(epoch + 1, training_loss, ac))
        
        # 每三個 epoch 存一次 weight & 其對應的 accuracy
        if epoch % 3 == 2:
            m_weights.append(flatten_param(model))
            l_weights.append(getLayerParam(model))
            accuracy.append(ac)

# 存 weight & accuracy
np.save('m_weights', np.array(m_weights))
np.save('l_weights', np.array(l_weights))
np.save('acc', np.array(accuracy))

# m_weights = np.load('./data/m_weights.npy')
# l_weights = np.load('./data/l_weights.npy')
# accuracy = np.load('./data/acc.npy')


# In[58]:


# PCA
cmap = ["#a37bd6", "#c38678", "#363b74", "#ecb939", "#cb2424", "#398564", "#5588ff", "#c6c386"]
pca = PCA(n_components=2)

# reduce dimension
rd_m_weight = pca.fit_transform(m_weights)
rd_l_weight = pca.fit_transform(l_weights)

# 畫圖 (whole model + 1-layer)
fig = plt.figure()
ax = plt.subplot(211)
ax.set_title("Whole Model")
ax.scatter(rd_m_weight[:, 0], rd_m_weight[:, 1], alpha=0)
for i, acc in enumerate(accuracy):
    c_index = int(i / 15)
    ax.text(rd_m_weight[i][0], rd_m_weight[i][1], acc, color=cmap[c_index])

ax = plt.subplot(212)
ax.set_title("Fully-Connected Layer")
ax.scatter(rd_l_weight[:, 0], rd_l_weight[:, 1], alpha=0)
for i, acc in enumerate(accuracy):
    c_index = int(i / 15)
    ax.text(rd_l_weight[i][0], rd_l_weight[i][1], acc, color=cmap[c_index])
plt.tight_layout()    
plt.savefig("./PCA")

