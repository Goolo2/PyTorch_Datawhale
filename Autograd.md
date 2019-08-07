# Autograd

PyTorch中所有神经网络的核心是`Autograd`包。让我们先来看看这个，然后我们会去训练我们的第一个神经网络。

`Autograd`包为张量的所有操作提供了自动微分。它是一个运行时定义(define-by-run)的框架，这意味着您的后向传播(backprop)是由您的代码如何运行来定义的，并且每一个迭代都可能是不同的。

让我们用更简单的术语和一些例子来学习这一点。

## 张量（Tensor）

- `torch.Tensor` 是Autograd包中的核心类。 如果您将某个tensor的属性`.requires_grad`设置为True，则这个tensor将开始跟踪针对它的所有操作。 当您完成计算时，您可以调用`.backward()`来自动计算所有梯度。这个tensor的梯度将会累计到 `.grad` 属性中去.

- 要阻止一个张量跟踪历史记录，可以调用`.detach()`将其从计算历史中分离出来，并防止将来的计算被跟踪。

- 为了防止跟踪历史记录(并使用内存)，还可以用`torch.no_grad()：`包裹代码块。在评估模型时，这可能特别有用，因为模型可能具有`Required_grad=True`的可训练参数，但对于这些参数，在评估阶段我们不需要计算梯度。
- 还有一个类对于Autograd的实现非常重要 - `Function`.
  - `Tensor` 和 `Function` 是互联的并且构成了一个无环计算图, 以此来实现对完整计算历程的编码. 每个Tensor有一个`.grad_fn`属性指向一个`Function`,正是这个`Function`创建了那个Tensor。 (由用户自己创建的张量Tensors的`grad_fn is None`).
- 如果你想计算导数(derivatives), 你可以调用 `Tensor`的`.backward()`。如果 `Tensor` 是一个标量 (i.e. it holds a one element data), 那么你不需要给`backward()`函数传递任何参数；然而，如果`Tensor` 是一个向量（有多个元素）那么你必须要传递一个参数 `gradient`
  而且这个参数的shape必须与待求导`Tensor`的shape相匹配。**因为Tensor无法对Tensor求导，需要gradient参数对y的各个分量加权求和得到标量后再来求导**

## L2 norm 正则化

在官网Autograd教程中，有以下代码：

```python
x = torch.randn(3, requires_grad=True)

y = x * 2
while y.data.norm() < 1000:
    y = y * 2

print(y)
```

```
tensor([-269.7014,   46.7115, 1120.6299], grad_fn=<MulBackward0>)
```

```python
In [15]: x = torch.randn(3, requires_grad=True)

In [16]: y = x * 2

In [17]: y.data
Out[17]: tensor([-1.2510, -0.6302,  1.2898])

In [18]: y.data.norm()
Out[18]: tensor(1.9041)

# computing the norm using elementary operations
In [19]: torch.sqrt(torch.sum(torch.pow(y, 2)))
Out[19]: tensor(1.9041)
```

对比两个输出的tensor可以看出y.data.norm到底做了什么

## backward()里参数的作用

Now in this case `y` is no longer a scalar. `torch.autograd` could not compute the full Jacobian directly, but if we just want the vector-Jacobian product, simply pass the vector to `backward` as argument:

```python
v = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float)
y.backward(v)

print(x.grad)
```

- <u>详细解释见下图</u>

![1565146695459](D:\编程\DataCamp\datanote\pic\1565146695459.png)



## 官网教程复刻

```python
import torch as t
x=t.ones(2,2,requires_grad=True)
print(x)

y=x+2
print(y)
print(y.grad_fn)

z=y*y*3
out=z.mean()
print(z,out)

a=t.randn(2,2)
a=(a*3)/(a-1)
print(a.requires_grad)
a.requires_grad=True
print(a.requires_grad)
b=(a*a).sum() #返回a*a中所有元素的和
print(b.grad_fn)

print(x.grad)
out.backward()
#上式等价于out.backward(t.tensor(1.))，因为out是标量
print(x.grad)

x=t.randn(3,requires_grad=True)
y=x*2
while y.data.norm()<1000:
    y=y*2
    
print(y)

gradients=t.tensor([0.1,1.0,0.0001],dtype=t.float)
y.backward(gradients)
print(x.grad)

print(x.requires_grad)
print((x**2).requires_grad) #requires_grad会传递
with t.no_grad():
    print((x**2).requires_grad)
```

