
# coding: utf-8

# In[62]:


import numpy as np
import pandas as pd
from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, Dropout, Activation, Concatenate
from keras.utils import np_utils, to_categorical
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.datasets import mnist
import matplotlib.pyplot as plt
K.set_image_dim_ordering = "th"


# In[3]:


batch_size = 128
num_classes = 10

img_rows, img_cols = 28, 28
(x_train, y_train), (x_test, y_test) = mnist.load_data()


# In[7]:


if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)


# In[31]:


x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255


# In[12]:


y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)


# In[65]:


def create_model(num_hid, dim):
    model = Sequential()
    model.add(Conv2D(dim, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    for i in range(num_hid):
        model.add(Conv2D(dim, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    return model


# In[89]:


hidden_list = [0, 0, 0, 1, 2]
dim_list = [16, 32, 48, 16, 16]
epochs = 20


# In[90]:


plt.figure(figsize=(10,10))
for i in range(len(hidden_list)):
    num_hid = hidden_list[i]
    dim = dim_list[i]
    model = create_model(num_hid, dim)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    history = model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test))
    plt.subplot(211)
    plt.plot(history.history['loss'])
    plt.subplot(212)
    plt.plot(history.history['acc'])


# In[91]:


plt.subplot(211)
plt.title('Training Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['hidden_1_16', 'hidden_1_32', 'hidden_1_48', 'hidden_2_16', 
            'hidden_3_16'], loc='upper left')
plt.subplot(212)
plt.title('Training Accuracy')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(['hidden_1_16', 'hidden_1_32', 'hidden_1_48', 'hidden_2_16', 
            'hidden_3_16'], loc='upper left')
plt.tight_layout()
plt.savefig('test.png', dpi=300)

