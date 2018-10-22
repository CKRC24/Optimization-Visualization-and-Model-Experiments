
# coding: utf-8

# In[1]:


# %matplotlib inline
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
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=100, shuffle=True, num_workers=2)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=preprocess)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=2)

# 定義 Image Class
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# In[3]:


# 定義計算 model 參數的 function
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# In[128]:


# 定義 model
class Conv_1(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(8 * 15 * 15, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(-1, 8 * 15 * 15)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class Conv_2(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 10, 3)
        self.conv2 = nn.Conv2d(10, 48, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(48 * 6 * 6, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 10)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 48 * 6 * 6)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class Conv_3(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 15, 3)
        self.conv2 = nn.Conv2d(15, 64, 3)
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 2 * 2, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 10)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 2 * 2)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# In[129]:


# 查看參數量
print(count_parameters(Model_1()))
print(count_parameters(Model_2()))
print(count_parameters(Model_3()))


# In[132]:


print(Model_3())


# In[ ]:


# training 參數
num_epoch = 20
learning_rate = 0.001

# 定義 loss function, optimizer
path = "./model"
losses = []
accuracy = []
lines = []
models = [Model_1(), Model_2(), Model_3()]
criterion = nn.CrossEntropyLoss()

# 開 train
for model in models:
    print('Start Training Model_' + model.__class__.__name__)
    model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    m_loss = []
    m_acc = []
    for epoch in range(num_epoch):
        ac = 0.0
        training_loss = 0.0
        correct = 0
        total = 50000
        for i, data in enumerate(train_loader, 0):
            # get inputs and wrap them into variable
            inputs, labels = data
#             inputs, labels = Variable(inputs), Variable(labels)
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
        m_loss.append(training_loss)
        m_ac.append(ac)
        print('[Epoch: %d] Loss: %.3f, Accuracy: %.3f%%' %(epoch + 1, training_loss, ac))
    
    lines.append(outputs)
    losses.append(m_loss)
    accuracy.append(m_ac)
    
    # save model
    torch.save(model.state_dict(), path + str(model.__class__.__name__) + ".pt")
print('Finished Training !')


# In[ ]:


# 計算testing accuracy
total = 0
correct = 0
for model in models:
    model.cuda()
    for test_images, labels in test_loader:
        test_images = Variable(test_images).cuda()
#         outputs = model(Variable(test_images))
        outputs = model(test_images) # GPU available

        # outputs 為每張圖10個class的energy分佈,dim: (n, 10)
        # 取最大的 index 作為output
        _, predictions = torch.max(outputs.data, 1) # 回傳(max, max_index)
        total += labels.size(0)
        correct += (predictions.cpu() == labels).sum()

    # print accuracy
    print(model.__class__.__name__ + ' Testing Accuracy: %d%%' % (100 * correct / total))


# In[ ]:


# 畫 loss 的圖 
for i, cc in enumerate(["b", "g", "r"]):
    plt.plot(list(range(20)), losses[i][:], c=cc)
plt.legend(['Conv2d_1','Conv2d_2','Con2d_3'])
plt.title("Training Loss")
# plt.show()
plt.savefig("loss")
plt.cla()


# In[ ]:


# 畫 accuracy 的圖
for i, cc in enumerate(["b", "g", "r"]):
    plt.plot(list(range(20)), accuracy[i][:], c=cc)
plt.legend(['Conv2d_1','Conv2d_2','Con2d_3'])
plt.title("Training Accuracy")
# plt.show()
plt.savefig("accuracy")

