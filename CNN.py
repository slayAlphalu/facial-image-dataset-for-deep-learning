# -- coding: utf-8 --
from __future__ import absolute_import
from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten,Dropout,BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.utils.np_utils import to_categorical
from keras.optimizers import SGD, Adadelta, Adagrad
from keras.utils import np_utils, generic_utils
from six.moves import range
from keras import losses
from keras import optimizers
import os
from PIL import Image
from keras.utils import plot_model
import numpy as np

seed = 7
np.random.seed(seed)

def load_data():
    data = np.empty((620,128,128,3),dtype="float32")
    label = np.empty((620,),dtype="uint8")
    #imgs = os.listdir("./resize_image")
    curwd = os.getcwd()
    os.chdir(curwd+"/resize_image")
    imgs = [item.strip() for item in os.popen("ls *.jpg|sort -n ").read().strip().split("\n")]
    os.chdir(curwd)
    num = len(imgs)
    for i in range(num):
        img = Image.open("./resize_image"+"/"+imgs[i])
        arr = np.asarray(img,dtype="float32")
        data[i,:,:,:] = arr
        label[i] = int(imgs[i].split('.')[0].split("-")[0])
    return data,label

def load_data_test():
    data_test=np.empty((62,128,128,3),dtype='float32')
    label_test=np.empty((62,),dtype='uint8')
    curwd = os.getcwd()
    os.chdir(curwd+"/resize_test")
    test = [item.strip() for item in os.popen("ls *.jpg|sort -n ").read().strip().split("\n")]
    os.chdir(curwd)
    #test = os.popen("ls [1-9]*.jpg").read().strip().split("\n")
    num=len(test)
    #imgs=os.listdir("./Users/Alpha/Desktop/my/resize_test")
    #num = len(imgs)
    for i in range(num):
     # img = Image.open("./Users/Alpha/Desktop/my/resize_test"+"/"+imgs[i])
        img = Image.open("./resize_test/"+test[i])
        arr = np.asarray(img,dtype="float32")
        data_test[i,:,:,:] = arr
        label_test[i] = int(test[i].split('-')[0])
    #os.chdir("../")
    return data_test,label_test

data, label = load_data()
print(data.shape[0], ' samples')
data_test,label_test=load_data_test()
print(data_test.shape[0], ' test')
#label为0~9共10个类别，keras要求格式为binary class matrices,
label = [item-1 for item in label]
label = np_utils.to_categorical(label, 10)
label_test = [item-1 for item in label_test]
label_test = np_utils.to_categorical(label_test, 10)

#生成一个model
model = Sequential()

#第1个卷积层
model.add(Conv2D(3, kernel_size=(3,3), strides=(2, 2), padding='same', activation='relu', input_shape=(128,128,3)))
model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding = 'same'))
#model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None))

#第2个卷积层
model.add(Conv2D(16, kernel_size=(3, 3), strides=(2,2),padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding = 'same'))
#model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None))
#第3个卷积层
model.add(Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu'))
#第4个卷积层
model.add(Conv2D(96, kernel_size=(3, 3), padding='same', activation='relu'))
#model.add(Conv2D(96, kernel_size=(3, 3), padding='same', activation='relu'))
#第5个卷积层
model.add(Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding = 'same'))
model.add(Dropout(0.1))

#全连接层，先将前一层输出的二维特征图flatten为一维的。
model.add(Flatten())
model.add(Dense(4096,activation='relu'))
#model.add(Dropout(0.25))
model.add(Dense(4096,activation='relu'))
#model.add(Dropout(0.25))

#Softmax分类，输出是10类别
model.add(Dense(10,activation='softmax')) #dense is the total number of labels, input_dim is the input dimension
sgd = optimizers.SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
#Adagrad = optimizers.Adagrad(lr=0.0001, decay=1e-6)
#Adam = optimizers.Adam(lr=0.0001, beta_1=0.9,beta_2=0.999,epsilon=1e-08)
#RMSprop=optimizers.RMSprop(lr=0.0001, epsilon=1e-6, rho=0.9)
model.compile(loss='categorical_crossentropy', optimizer='Adam',metrics=['accuracy'])
#model.compile(loss='mean_squared_error',optimizer='sgd',metrics=['mae', 'acc'])
#model.compile(loss='mean_absolute_percentage_error',optimizer='sgd')
#model.compile(loss='',optimizer='sgd',metrics=['mae', 'acc'])

#调用fit方法，就是一个训练过程.
model.fit(data,label,epochs=1,batch_size=620,verbose=1)
score= model.evaluate(data_test, label_test,verbose=1)

label_pred=model.predict(data_test)
label_class=label_pred.argmax(axis=-1)
labeltest=1+label_test.argmax(axis=-1).reshape((-1,1))
labelclass=1+label_class.reshape((-1,1))
import re

'''print(re.sub('[\[\]]', '',np.array_str(labeltest)))
print('---------')
print(re.sub('[\[\]]', '',np.array_str(labelclass)))
'''

file=open('data.txt','w')
file.write(str(re.sub('[\[\]]', '',np.array_str(labeltest))))
file.write("\n")
file.write("----------""\n")
file.write(str(re.sub('[\[\]]', '',np.array_str(labelclass))))
file.write(str(score[1]))
file.write("\n")
file.write(str(score[0]))
file.close()

#print('test acc=',score[1])
#print('test loss=',score[0])
