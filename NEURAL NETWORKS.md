# NEURAL NETWORKS

## Network 训练流程

- Define the neural network that has some learnable parameters (or weights)
- Iterate over a dataset of inputs
- Process input through the network
- Compute the loss (how far is the output from being correct)
- Propagate gradients back into the network’s parameters
- Update the weights of the network, typically using a simple update rule: `weight = weight - learning_rate * gradient`

![1565161019099](D:\编程\DataCamp\datanote\pic\1565161019099.png)

这幅图片对应的网络结构是

```python
self.conv1=nn.Conv2d(1,6,5)
self.conv2=nn.Conv2d(6,16,5)
self.fc1=nn.Linear(16*5*5,120) 
```

16 * 5 * 5 中“5:”推导

（32-5+1）/2=14

（14-5+1）/2=5

## Conv2d详解

```python
import torch

x = torch.randn(2,1,7,3)
conv = torch.nn.conv2d(1,8,(2,3))
res = conv(x)

print(res.shape)    # shape = (2, 8, 6, 1)
```

  + **x**
    **[ batch_size, channels, height_1, width_1 ]**

|batch_size 一个batch中样例的个数       |2|
|---|---|
|channels 通道数，也就是当前层的深度 |1|
|height_1, 图片的高    |                             7|
|width_1, 图片的宽        |                          3|

  + **conv2d的参数**
    **[ channels, output, height_2, width_2 ]**

| channels, 通道数，和上面保持一致，也就是当前层的深度 | 1    |
| ---------------------------------------------------- | ---- |
| output 输出的深度                                    | 8    |
| height_2, 过滤器filter的高                           | 2    |

- **res**

  **[ batch_size,output, height_3, width_3 ]**

| batch_size, 一个batch中样例的个数，同上 | 2                                   |
| --------------------------------------- | ----------------------------------- |
| output 输出的深度                       | 8                                   |
| height_3, 卷积结果的高度                | 6 = height_1 - height_2 + 1 = 7-2+1 |
| width_3, 卷积结果的宽度                 | 1 = width_1 - width_2 +1 = 3-3+1    |

`torch.nn` only supports mini-batches. The entire `torch.nn` package only supports inputs that are a mini-batch of samples, and not a single sample.

For example, `nn.Conv2d` will take in a 4D Tensor of `nSamples x nChannels x Height x Width`.

If you have a single sample, just use `input.unsqueeze(0)` to add a fake batch dimension.

## 官网教程复刻

```python
#NEURAL NETWORKS
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1=nn.Conv2d(1,6,3)
        self.conv2=nn.Conv2d(6,16,3)
        self.fc1=nn.Linear(16*6*6,120) 
        self.fc2=nn.Linear(120,84)
        self.fc3=nn.Linear(84,10)
        
    def forward(self,x):
        x=F.max_pool2d(F.relu(self.conv1(x)),(2,2))
        x=F.max_pool2d(F.relu(self.conv2(x)),2)
    
        x = x.view(-1, self.num_flat_features(x))
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=self.fc3(x)
        return x
    
    def num_flat_features(self,x):
        size=x.size()[1:]
        num_features=1
        for s in size:
            num_features*=s
        return num_features
net=Net()
print(net)

params=list(net.parameters())
print(len(params))
print(params[0].size())
#注意这里的输入得是4维的才能和conv对应
input=t.randn(1,1,32,32)
out=net(input)
print(out)

net.zero_grad()
out.backward(t.randn(1,10))

output=net(input)
target=t.randn(10)
target=target.view(1,-1)
criterion=nn.MSELoss()

loss=criterion(output,target)
print(loss.grad_fn)
print(loss.grad_fn.next_functions[0][0])
print(loss.grad_fn.next_functions[0][0].next_functions[0][0])

net.zero_grad()

print('conv1.bias.grad before backward')
print(net.conv1.bias.grad)

loss.backward()

print('conv1.bias.grad after backward')
print(net.conv1.bias.grad)

#手动实现随机梯度下降法
#weight = weight - learning_rate * gradient
#data.sub_就是取出data减去（）中的值
learning_rate=0.01
for f in net.parameters():
    f.data.sub_(f.grad.data*learning_rate)
    
#也可以调用包来实现    
optimizer=optim.SGD(net.parameters(),lr=0.01)
optimizer.zero_grad()
output=net(input)
loss=criterion(output,target)
loss.backward()
optimizer.step() # Does the update
```

