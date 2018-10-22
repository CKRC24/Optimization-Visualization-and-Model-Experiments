
# coding: utf-8

# In[ ]:


import torch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


# In[2]:


# preprocess
# 將PIL Image 轉成 Tensor 並 Normalize (mean: 0.5, std: 0.5) 
preprocess = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

# 準備 dataset, Train 有 shuffle, Test 無 shuffle
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=preprocess)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=100, shuffle=False, num_workers=2)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=preprocess)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=True, num_workers=2)

# 定義 Image Class
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# In[3]:


# 定義計算 model 參數的 function
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# In[4]:


class Conv_3(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 2 * 2, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 10)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 2 * 2)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# In[51]:


print(Conv_3())


# In[48]:


# training 參數
num_epoch = 1000
learning_rate = 0.00005
seed = 19

# 定義 loss function, optimizer
path = "./model"
losses = []
train_loss = []
test_loss = []
model = Conv_3().cuda()
criterion = nn.CrossEntropyLoss()

# 開 train
print('Start Training Model_' + model.__class__.__name__)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# fix random seed
torch.manual_seed(seed)
p_index = torch.randperm(100)


# In[ ]:


for epoch in range(num_epoch):
    training_loss = 0.0
    testing_loss = 0.0
    
    # train
    for i, (train_images, train_labels) in enumerate(train_loader):
        # get inputs and wrap them into variable
        train_labels = train_labels[p_index]
#         if i == 0:
#             print(train_labels[:5])
#         for k in range(100):
#             train_labels[k] = train_labels[(k+1)%100]
            
        train_images, train_labels = Variable(train_images.cuda()), Variable(train_labels.cuda())

        # zero the gradient
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(train_images)
        t_loss = criterion(outputs, train_labels)

        # 更新gradient
        t_loss.backward()
        optimizer.step()
        training_loss += t_loss.data[0]
        
    # calculate test loss
    for test_images, test_labels in test_loader:
        test_images, test_labels = Variable(test_images).cuda(), Variable(test_labels).cuda()
        pred = model(test_images)
        p_loss = criterion(pred, test_labels)
        testing_loss += p_loss.data[0]
        
    # print loss, accuracy per epoch
    training_loss = training_loss / 500
    testing_loss = testing_loss / 100
    print('[Epoch: %d] Training Loss: %.3f, Testing Loss: %.3f' %(epoch + 1, training_loss, testing_loss))

    # append loss for plot
    train_loss.append(training_loss)
    test_loss.append(testing_loss)

losses.append(train_loss)
losses.append(test_loss)
# np.save('./data/shuffle_loss', losses)
print('Finished Training !')


# In[50]:


# 畫 loss 的圖 
# losses = np.load('./data/shuffle_loss.npy')
for i, cc in enumerate(["#6CB3C9", "#FFAD32"]):
    plt.plot(list(range(num_epoch)), losses[i][:], c=cc)
plt.legend(['train','test'])
plt.xlabel('epoch', fontsize=12)
plt.ylabel('loss', fontsize=12)
plt.savefig("./CIFAR10_shuffle_loss")

