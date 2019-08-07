# **PyTorch的基本概念**

### Pytorch基本概念

- Tensor和numpy对象共享内存，所以他们之间的转换很快，而且几乎不会消耗什么资源。但这也意味着，如果其中一个变了，另外一个也会随之改变。
- 想获取某一个元素的值，可以使用`scalar.item`。 直接`tensor[idx]`得到的还是一个tensor: 一个0-dim 的tensor，一般称为scalar.
- new_tensor = old_tensor.clone().detach()复制张量
- 需要注意的是，`t.tensor()`总是会进行数据拷贝，新tensor和原来的数据不再共享内存。所以如果你想共享内存的话，建议使用`torch.from_numpy()`或者`tensor.detach()`来新建一个tensor, 二者共享内存。  

### 官网教程复刻

```python
#pytorch基础练习
import torch as t
import numpy as np
print(t.__version__)

#1.矩阵基础---------------------------------
#构造随机的矩阵
x=t.zeros(5,3,dtype=t.long)
print(x)
x=t.empty(5,3)
print(x)
#构造确定的矩阵
y=t.Tensor([[1,2],[3,4]])
print(y)
#基于已存在的张量构造张量，下面两种方法共享输入张量的内存
x=x.new_ones(5,3,dtype=t.double)
print(x)
#生成与输入的size相同的，正态分布，均值0，方差1
x=t.randn_like(x,dtype=t.float)
print(x)
# 使用[0,1]均匀分布随机初始化二维数组
z=t.rand(5,3)
print(z)
#查看z的形状
print(z.size())
print(z.size()[0])
print(z.size()[1])
print(z.size(0))
print(z.size(1))

#2. 加法基础--------------------------------
#加法写法一
print(x+z)
#加法写法二
print(t.add(x,z))
#加法写法三：预先分配空间
result = t.Tensor(5,3) # 预先分配空间
t.add(x,z,out=result) 
print(result)


x=t.rand(5,3)
print("原来的z",z)
#add：不改变z
z.add(x)
print("不改变",z)
#add：改变z  //以`_`结尾的函数会修改自身
z.add_(x)
print("改变",z)


#reshape
c=t.randn(4,4)
d=c.view(16)
e=c.view(-1,8) # the size -1 is inferred from other dimensions
print(c.size(), d.size(), e.size())

#3. 数组和张量的转换
#张量和nd数组可以相互转换，除了charTensor
#Tensor -> Numpy
a = t.ones(5)
b=a.numpy()
print(type(a))
print(type(b))

# Numpy->Tensor
a=np.ones(5)
b=t.from_numpy(a)
print(type(a))
print(type(b))

#区分标量和张量的维度
scalar=b[0]
print(scalar)
print(scalar.size())#0-dim
#张量中只有一个元素，用item()获得值
print(scalar.item())

tensor=t.tensor([2])
print(tensor)
print(tensor.size())
print(tensor.item())
#张量的复制

tensor = t.tensor([3,4])
old_tensor = tensor
new_tensor = old_tensor.clone().detach()
new_tensor[0] = 1111
print(old_tensor, new_tensor)

# 在不支持CUDA的机器下，下一步还是在CPU上运行
device = t.device("cuda:0" if t.cuda.is_available() else "cpu")
y=t.ones_like(x,device=device)
x=x.to(device)
print(x)
z=x+y
print(z)
print(z.to("cpu",t.double))
```

