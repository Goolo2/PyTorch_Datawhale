# Training A Classifier

## 基础知识

### torchvision

#### 主要模块

1. torchvision.datasets 

   数据模块，包含cifar10，imagenet等图库

2. torchvision.models 

   模型模块

3. torchvision.transforms

   数据变换模块

4. torchvision.utils

   工具

### numpy.transpose

```python
>>> x = np.ones((1, 2, 3))
>>> np.transpose(x, (1, 0, 2)).shape
(2, 1, 3)
```

### transforms

#### 主要功能

- PIL.Image/numpy.ndarray与Tensor的相互转化；

  PIL.Image/numpy.ndarray转化为Tensor，常常用在训练模型阶段的数据读取，而Tensor转化为PIL.Image/numpy.ndarray则用在验证模型阶段的数据输出。

- 归一化；

- 对PIL.Image进行裁剪、缩放等操作。

#### 举例

1. transforms.ToTensor() 将 PIL.Image/numpy.ndarray 数据进转化为torch.FloadTensor，并归一化到[0, 1.0]：

   - 取值范围为[0, 255]的PIL.Image，转换成形状为[C, H, W]，取值范围是[0, 1.0]的torch.FloadTensor；
   - 形状为[H, W, C]的numpy.ndarray，转换成形状为[C, H, W]，取值范围是[0, 1.0]的torch.FloadTensor。

2. transforms.Compose就是将transforms组合在一起；

3. transforms.Normalize使用如下公式进行归一化：

   channel=（channel-mean）/std

```python
# transforms.ToTensor()
transform1 = transforms.Compose([
    transforms.ToTensor(), # range [0, 255] -> [0.0,1.0]
    ]
)
#归一化到【-1,1】
transform2 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))
    ]
)
```

### DataLoader

DataLoader是torch用来包装数据的工具，将自己的 (numpy array 或其他) 数据形式装换成 Tensor, 然后再放进这个包装器中，它帮你有效地迭代数据

```python
import torch
import torch.utils.data as Data
torch.manual_seed(1)    # reproducible

BATCH_SIZE = 5      # 批训练的数据个数

x = torch.linspace(1, 10, 10)       # x data (torch tensor)
y = torch.linspace(10, 1, 10)       # y data (torch tensor)

# 先转换成 torch 能识别的 Dataset
torch_dataset = Data.TensorDataset(data_tensor=x, target_tensor=y)

# 把 dataset 放入 DataLoader
loader = Data.DataLoader(
    dataset=torch_dataset,      # torch TensorDataset format
    batch_size=BATCH_SIZE,      # mini batch size
    shuffle=True,               # 要不要打乱数据 (打乱比较好)
    num_workers=2,              # 多线程来读数据
)

for epoch in range(3):   # 训练所有!整套!数据 3 次
    for step, (batch_x, batch_y) in enumerate(loader):  # 每一步 loader 释放一小批数据用来学习
        # 假设这里就是你训练的地方...

        # 打出来一些数据
        print('Epoch: ', epoch, '| Step: ', step, '| batch x: ',
              batch_x.numpy(), '| batch y: ', batch_y.numpy())

"""
Epoch:  0 | Step:  0 | batch x:  [ 6.  7.  2.  3.  1.] | batch y:  [  5.   4.   9.   8.  10.]
Epoch:  0 | Step:  1 | batch x:  [  9.  10.   4.   8.   5.] | batch y:  [ 2.  1.  7.  3.  6.]
Epoch:  1 | Step:  0 | batch x:  [  3.   4.   2.   9.  10.] | batch y:  [ 8.  7.  9.  2.  1.]
Epoch:  1 | Step:  1 | batch x:  [ 1.  7.  8.  5.  6.] | batch y:  [ 10.   4.   3.   6.   5.]
Epoch:  2 | Step:  0 | batch x:  [ 3.  9.  2.  6.  7.] | batch y:  [ 8.  2.  9.  5.  4.]
Epoch:  2 | Step:  1 | batch x:  [ 10.   4.   8.   1.   5.] | batch y:  [  1.   7.   3.  10.   6.]
"""
```

### Python类的继承

如果在子类中需要父类的构造方法就需要显式地调用父类的构造方法，或者不重写父类的构造方法。

子类不重写 **__init__**，实例化子类时，会自动调用父类定义的 **__init__**。

```python
class Father(object):
    def __init__(self, name):
        self.name=name
        print ( "name: %s" %( self.name) )
    def getName(self):
        return 'Father ' + self.name
 
class Son(Father):
    def getName(self):
        return 'Son '+self.name
 
if __name__=='__main__':
    son=Son('runoob')
    print ( son.getName() )
```

```
name: runoob
Son runoob
```

如果重写了**__init__** 时，实例化子类，就不会调用父类已经定义的 **__init__**

```python
class Father(object):
    def __init__(self, name):
        self.name=name
        print ( "name: %s" %( self.name) )
    def getName(self):
        return 'Father ' + self.name
 
class Son(Father):
    def __init__(self, name):
        print ( "hi" )
        self.name =  name
    def getName(self):
        return 'Son '+self.name
 
if __name__=='__main__':
    son=Son('runoob')
    print ( son.getName() )
```

```
hi
Son runoob
```

如果重写了**__init__** 时，要继承父类的构造方法，可以使用 **super** 关键字：

```
super(子类，self).__init__(参数1，参数2，....)
```

还有一种经典写法：

```
父类名称.__init__(self,参数1，参数2，...)
```

```python
class Father(object):
    def __init__(self, name):
        self.name=name
        print ( "name: %s" %( self.name))
    def getName(self):
        return 'Father ' + self.name
 
class Son(Father):
    def __init__(self, name):
        super(Son, self).__init__(name)
        print ("hi")
        self.name =  name
    def getName(self):
        return 'Son '+self.name
 
if __name__=='__main__':
    son=Son('runoob')
    print ( son.getName() )
```

```
name: runoob
hi
Son runoob
```

## 代码

```python
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 19:46:16 2019

@author: Goolo
"""

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#加载正则化的CIFAR10

#定义一个transform，把PIL image转换为Tensor
transform=transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

#transform参数表示使用我们上面定义的转换
trainset=torchvision.datasets.CIFAR10(root='./data',train=True,
                                      download=False,transform=transform)
# dataset torch能识别的dataset类型
#shuffle 是否打乱数据
#num_worker 多线程读取数据
trainloader=torch.utils.data.DataLoader(dataset=trainset,batch_size=4,shuffle=True,num_workers=2)

testset=torchvision.datasets.CIFAR10(root='./data',train=False,
                                      download=False,transform=transform)
trainloader=torch.utils.data.DataLoader(testset,batch_size=4,shuffle=False,num_workers=2)


    # 类别信息也是需要我们给定的
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


import matplotlib.pyplot as plt
import numpy as np

## 输出图像的函数
#def imshow(img):
#    img = img / 2 + 0.5     # unnormalize
#    npimg = img.numpy()
#    plt.imshow(np.transpose(npimg, (1, 2, 0)))
#    plt.show()
#
## 随机获取训练图片
#dataiter = iter(trainloader)
#images, labels = dataiter.next()
#
## 显示图片
#imshow(torchvision.utils.make_grid(images))
## 打印图片标签
#print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()   # 第二、三行都是python类继承的基本操作,此写法应该是python2.7的继承格式,但python3里写这个好像也可以
        self.conv1 = nn.Conv2d(3, 6, 5)       # 添加第一个卷积层,调用了nn里面的Conv2d（）
        self.pool = nn.MaxPool2d(2, 2)        # 最大池化层
        self.conv2 = nn.Conv2d(6, 16, 5)      # 同样是卷积层
        self.fc1 = nn.Linear(16 * 5 * 5, 120) # 接着三个全连接层
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
```

