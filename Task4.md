# 损失函数CrossEntropyLoss详解

## 1.nn与nn.functional的区别

1. **torch.nn下的Conv1d**

   ```python
   import torch.nn.functional as F
   class Conv1d(_ConvNd):
       def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                    padding=0, dilation=1, groups=1, bias=True):
           kernel_size = _single(kernel_size)
           stride = _single(stride)
           padding = _single(padding)
           dilation = _single(dilation)
           super(Conv1d, self).__init__(
               in_channels, out_channels, kernel_size, stride, padding, dilation,
               False, _single(0), groups, bias)
   
           def forward(self, input):
               return F.conv1d(input, self.weight, self.bias, self.stride,
                               self.padding, self.dilation, self.groups)
   ```

2. **torch.nn.functional下的conv1d:**

   ```python
   def conv1d(input, weight, bias=None, stride=1, padding=0, dilation=1,
              groups=1):
       if input is not None and input.dim() != 3:
           raise ValueError("Expected 3D tensor as input, got {}D tensor instead.".format(input.dim()))
   
           f = ConvNd(_single(stride), _single(padding), _single(dilation), False,
                      _single(0), groups, torch.backends.cudnn.benchmark,
                      torch.backends.cudnn.deterministic, torch.backends.cudnn.enabled)
           return f(input, weight, bias)
   ```

3. **对比**

   可以看到torch.nn下的Conv1d类在forward时调用了nn.functional下的conv1d。

   这么设计是有其原因的。如果我们只保留nn.functional下的函数的话，在训练或者使用时，我们就要手动去维护weight, bias, stride这些中间量的值，这显然是给用户带来了不便。而如果我们只保留nn下的类的话，其实就牺牲了一部分灵活性，因为做一些简单的计算都需要创造一个类，这也与PyTorch的风格不符。

   - **调用方式对比**

     `nn.Xxx` 需要先实例化并传入参数，然后以函数调用的方式调用实例化的对象并传入输入数据。

     ```python
     inputs = torch.rand(64, 3, 244, 244)
     conv = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
     out = conv(inputs)
     ```

     `nn.functional.xxx`同时传入输入数据和weight, bias等其他参数 。

     ```python
     weight = torch.rand(64,3,3,3)
     bias = torch.rand(64) 
     out = nn.functional.conv2d(inputs, weight, bias, padding=1)
     ```

   - **nn.Xxx继承于nn.Module， 能够很好的与nn.Sequential结合使用， 而nn.functional.xxx无法与nn.Sequential结合使用。**

     ```python
     fm_layer = nn.Sequential(
                 nn.Conv2d(3, 64, kernel_size=3, padding=1),
                 nn.BatchNorm2d(num_features=64),
                 nn.ReLU(),
                 nn.MaxPool2d(kernel_size=2),
                 nn.Dropout(0.2)
       )
     ```

     

4. **举例**

   ```python
   import torch
   import torch.nn as nn
   import torch.nn.functional as F
   class Net(nn.Module):
       def __init__(self):
           super(Net, self).__init__()
           self.fc1 = nn.Linear(512, 256)
           self.fc2 = nn.Linear(256, 256)
           self.fc3 = nn.Linear(256, 2)
   
       def forward(self, x):
           x = F.relu(self.fc1(x))
           x = F.relu(F.dropout(self.fc2(x), 0.5))
           x = F.dropout(self.fc3(x), 0.5)
           return x
   ```
   
   以一个最简单的三层网络为例。需要维持状态的，主要是三个线性变换，所以在构造Module是，定义了三个nn.Linear对象，而在计算时，relu,dropout之类不需要保存状态的可以直接使用。

   问题回答：

   - 什么是维持状态？

     拿线性变化举例吧，其中的权重需要不停的更新，这就是需要维持的状态，而以激活函数来说，一般来说都是可以直接使用的，只要给定相应的输入，那么使用后就有确定的输出。

   - 像是relu这种东西，明摆着就是不用更新状态的，为啥nn里面还要给实现一次？

     还有一种利用Sequential进行模型搭建的方式，里面要求每个参数都是nn.Module，这时候就可以派上用场了。

     nn.ModuleList()也会用到

作者：蒲嘉宸
链接：https://www.zhihu.com/question/66782101/answer/246341271
来源：知乎
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。

## 2. Softmax详解

参考自https://www.cnblogs.com/marsggbo/p/10401215.html

### 基础概念

Softmax函数的输入是N维的随机真值向量，输出是另一个N维的真值向量， 
且值的范围是(0,1)(0,1)，和为1.0。即映射

![1565525600336](D:\编程\DataCamp\datanote\pic\1565525600336.png)

![1565525607664](D:\编程\DataCamp\datanote\pic\1565525607664.png)

其中每一个元素的公式为： 

![1565525660131](D:\编程\DataCamp\datanote\pic\1565525660131.png)

显然SjSj总是正的~(因为指数)；因为所有的SjSj的和为1，所以有Sj<1Sj<1，因此它的范围是(0,1)(0,1)。例如，一个含有三个元素的向量[1.0,2.0,3.0][1.0,2.0,3.0]被转化为[0.09,0.24,0.67][0.09,0.24,0.67]。 
转化后的元素与原始的对应的元素位置上保持一致，且和为1。我们将原始的向量拉伸为[1.0,2.0,5.0][1.0,2.0,5.0]，得到变换后的[0.02,0.05,0.93][0.02,0.05,0.93]，同样具有前面的性质。

注意此时因为最后一个元素(5.0)距离前面两个元素(1.0和2.0)较远，因此它的输出的softmax值占据了和1.0的大部分(0.93)。softmax并不是只选择一个最大元素，而是将向量分解为整个(1.0)的几部分，最大的输入元素得到一个比例较大的部分，但其他元素各自也获得对应的部分。

### 概率解释

softmax的性质(所有输出的值范围是(0,1)且和为1.0)使其在机器学习的概率解释中广泛使用。尤其是在多类别分类任务中，我们总是给输出结果对应的类别附上一个概率，即如果我们的输出类别有N种，我们就输出一个N维的概率向量且和为1.0。每一维的值对应一种类别的概率。我们可以将softmax解释如下： 

![1565525796724](D:\编程\DataCamp\datanote\pic\1565525796724.png)

其中，yy是输出的N个类别中的某个(取值为1...N1...N)。aa是任意一个N维向量。最常见的例子是多类别的逻辑斯谛回归，输入的向量xx乘以一个权重矩阵W，且该结果输入softmax函数以产生概率。我们在后面会探讨这个结构。事实证明，从概率的角度来看，softmax对于模型参数的最大似然估计是最优的。 不过，这超出了本文的范围。有关更多详细信息，请参阅“深度学习”一书的第5章(链接：www.deeplearningbook.org)。

### Softmax函数的导数

首先，明确我们计算的偏导数是什么

![1565533120501](D:\编程\DataCamp\datanote\pic\1565533120501.png)

这是==第i个输出关于第j个输入的偏导数==。我们使用一个更简洁的式子来表示：$D_jS_i$

因为softmax函数是一个$\mathbb{R}^{N}->\mathbb{R}^N$的函数，所以我们计算得到的导数是一个雅可比矩阵：（**列数由输入决定，行书由输出决定**）

![1565533408247](D:\编程\DataCamp\datanote\pic\1565533408247.png)

在机器学习的文献中，常常用术语梯度来表示通常所说的导数。严格来说，梯度只是为标量函数来定义的，例如机器学习中的损失函数；对于像softmax这样的向量函数，说是“梯度”是不准确的；==雅可比是一个向量函数的全部的导数，大多数情况下我们会说“导数”。==

对任意的ii和jj，让我们来计算$D_jS_i$

![1565573031346](D:\编程\DataCamp\datanote\pic\1565573031346.png)

我们将使用链式法则来计算导数，即对于$f(x)=g(x)/h(x)$

![1565573080090](D:\编程\DataCamp\datanote\pic\1565573080090.png)

在我们的情况下，有： 

![1565573095898](D:\编程\DataCamp\datanote\pic\1565573095898.png)

注意对于$h_i$，无论求其关于哪个$a_j$的导数，结果都是$e^{a_j}$，但是对于$g_i$就不同了，$g_i$关于$a_j$的导数是

$e^{a_j}$，当且仅当$i=j$，否则结果为0。

让我们回到$D_jS_i$；我们先考虑$i=j$的情况。根据链式法则我们有： 

![1565573652017](D:\编程\DataCamp\datanote\pic\1565573652017.png)

简单起见，我们使用$\sum_{}$表示$\sum_{n=1}^Ne^{a_k}$，继续简化如下

![1565573837347](D:\编程\DataCamp\datanote\pic\1565573837347.png)

类似的，考虑$i\not=j$的情况

![1565573900202](D:\编程\DataCamp\datanote\pic\1565573900202.png)

总结如下： 

![1565573914433](D:\编程\DataCamp\datanote\pic\1565573914433.png)

在文献中我们常常会见到各种各样的”浓缩的”公式，一个常见的例子是使用==克罗内克函数==： 

![1565573945730](D:\编程\DataCamp\datanote\pic\1565573945730.png)

于是我们有：

![1565573969344](D:\编程\DataCamp\datanote\pic\1565573969344.png)

**在文献中也有一些其它的表述：**

- 在雅可比矩阵中使用单位矩阵$I$来替换$\delta$，$I$使用元素的矩阵形式表示了δ。
- 使用”1”作为函数名而不是克罗内克δ，如下所示：$D_jS_i=S_i(1(i=j)-S_j)$。这里1($i=j$)意味着当$i=j$时值为1，否则为0。

### 数值稳定性（规范化）

对于一个给定的向量，利用Python来计算softmax的简单方法

```python
def softmax(x):
    """Compute the softmax of vector x."""
    exps = np.exp(x)
    return exps / np.sum(exps)
```

```
In [146]: softmax([1, 2, 3])
Out[146]: array([ 0.09003057, 0.24472847,  0.66524096])
```

然而当我们使用该函数计算较大的值时(或者大的负数时)，会出现一个问题：

```
In [148]: softmax([1000, 2000, 3000])
Out[148]: array([ nan,  nan,  nan])
```

Numpy使用的浮点数的数值范围是有限的。对于float64，最大可表示数字的大小为1030810308。 

因此我们需要通过规范输入使其输入不要太大或者太小，通过观察我们可以使用任意的常量C，如下所示： 

![1565579046188](D:\编程\DataCamp\datanote\pic\1565579046188.png)

然后将这个变量转换到指数上： 

![1565579057919](D:\编程\DataCamp\datanote\pic\1565579057919.png)

因为C是一个随机的常量，所以我们可以写为： 

![1565579081353](D:\编程\DataCamp\datanote\pic\1565579081353.png)

D也是一个任意常量。对任意D，这个公式等价于前面的式子，这让我们能够更好的进行计算。对于D，一个比较好的选择是所有输入的最大值的负数： 

![1565579113195](D:\编程\DataCamp\datanote\pic\1565579113195.png)

假定输入本身彼此相差不大，这会使输入转换到接近于0的范围。最重要的是，它将所有的输入转换为负数(除最大值外，最大值变为0)。很大的负指数结果会趋于0而不是无穷，这就让我们很好的避免了出现NaN的结果。

```python
def stablesoftmax(x):
    """Compute the softmax of vector x in a numerically
    stable way."""
    shiftx = x - np.max(x)
    exps = np.exp(shiftx)
    return exps / np.sum(exps)
```

```
In [150]: stablesoftmax([1000, 2000, 3000])
Out[150]: array([ 0.,  0.,  1.])
```

### 机器学习中的softmax层及其导数

softmax常用于机器学习中，特别是逻辑斯特回归：softmax层，其中我们将softmax应用于全连接层(矩阵乘法)的输出，如图所示。 

![img](https://img-blog.csdn.net/20180426095118859?watermark/2/text/Ly9ibG9nLmNzZG4ubmV0L2Nhc3NpZVB5dGhvbg==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

我们如何计算这个“softmax层”的导数(先进行全连接矩阵乘法，然后是softmax)？当然是使用链式规则！

在我们开始之前的一个重要的观点：**你可能会认为x是计算其导数的自然变量(natural variable)**。但事实并非如此。实际上，在机器学习中，我们通常希望找到最佳的权重矩阵W，因此我们希望用梯度下降的每一步来更新权重。**因此，实际上我们将计算该层的关于W的导数。**

我们首先将这个图改写为向量函数的组合。首先我们定义矩阵乘法`g(W)`，即映射：$\mathbb{R}^{NT}->\mathbb{R}^T$。因为输入(矩阵W)$N \times T$个元素，输出有T个元素。

接下来我们来考虑softmax，如果我们定义logits的向量是λ，我们有：$\mathbb{R}^{T}->\mathbb{R}^T$。总体来说，我们有：

![1565579770681](D:\编程\DataCamp\datanote\pic\1565579770681.png)

==这里的g(W)是指使W权重矩阵变为logits层的函数==

==S(W)是Softmax函数，使logtis层变为prob层==

==因此整个图可以表示为P(W)=S(g(W))==

使用多变量的链式法则，得到P(W)的雅可比矩阵： 

![1565579849106](D:\编程\DataCamp\datanote\pic\1565579849106.png)

> **Sog是复合函数的一种记法**
>
> 复合函数的链式求导举例：
> $$
> h(a)=f[g(a)]
> $$
> 则
> $$
> h'(a)=f'[g(x)]\cdot g'(x)
> $$
> 如
>
> $f(x)=3x,g(x)=3x+3,求h=g(f(x))的导数$
>
> 解：
>
> $令t=f(x)=3x$
>
> $那么h'(x)=g'(t)f'(x)=(3t+3)'(3x)'=3 \times 3=9$

这里注意正确计算相应的索引。因为g(W)：$\mathbb{R}^{NT}->\mathbb{R}^T$

![1565581084832](D:\编程\DataCamp\datanote\pic\1565581084832.png)

（**雅克比矩阵列数由输入决定，行数由输出决定**）

- <u>**先让我们回顾一下W**</u>
  $$
  \left[
  \begin{matrix}
   W_{11}      &  W_{12}       & \cdots &  W_{1N}      \\
    W_{21}      & W_{22}      & \cdots & W_{2N}    \\
   \vdots & \vdots & \ddots & \vdots \\
   W_{T1}     & W_{T2}     & \cdots & W_{TN}      \\
  \end{matrix}
  \right]
  $$
  我们可以看到输入是T×N的矩阵。

  而在上面的雅克比矩阵Dg中，在某种意义上，**权重矩阵W被“线性化”为长度为NT的向量**。 如果您熟悉多维数组的内存布局，应该很容易理解它是如何完成的。 （按照行主次序对其进行线性化处理，第一行是连续的，接着是第二行，等等。$W_{ij}$在雅可比矩阵中的列号是`(i-1)N`）

- <u>**再让我们回顾一下logits层**</u>
  $$
  \left[
  \begin{matrix}
   g_1          \\
   g_2           \\
   \vdots \\
   g_T           \\
  \end{matrix}
  \right]
  $$
  

  其中

  ![1565593006137](D:\编程\DataCamp\datanote\pic\1565593006137.png)



结合前面$D_jS_i$这种写法的定义（就是求偏导），我们可以类比得到
$$
D_1g_1=x_1\\
D_2g_1=x_2\\
\cdots\\
D_Ng_1=x_N\\
D_{N+1}g_1=0\\
\cdots\\
D_{NT}g_1=0
$$
我们使用同样的策略来计算g2⋯gT，我们可以得到雅可比矩阵： 

![1565593332504](D:\编程\DataCamp\datanote\pic\1565593332504.png)

从另一个角度来这个问题，如果我们将W的索引分解为i和j，可以得到

![1565593355749](D:\编程\DataCamp\datanote\pic\1565593355749.png)

最后，为了计算softmax层的完整的雅可比矩阵，我们只需要计算DSDS和DgDg间的乘积。注意P(W)：$\mathbb{R}^{NT}->\mathbb{R}^T$，因此雅可比矩阵的维度可以确定。因此DS是T×T，Dg是T×NT的，它们的乘积DP是T×NT的。 

在文献中，你会看到softmax层的导数大大减少了。因为涉及的两个函数很简单而且很常用。 如果我们仔细计算DS的行和Dg的列之间的乘积：

![1565593617650](D:\编程\DataCamp\datanote\pic\1565593617650.png)

Dg大多数为0，所以最终的结果很简单，仅当i=k时$D_{ij}g_k$不为0；然后它等于$x_j$。因此： 

![1565593734795](D:\编程\DataCamp\datanote\pic\1565593734795.png)

[克罗内克函数 $t=i$时 $\delta_{ti}=1$]

因此完全可以在没有实际雅可比矩阵乘法的情况下计算softmax层的导数; 这很好，因为矩阵乘法很耗时！由于全连接层的雅可比矩阵是稀疏的，我们可以避免大多数计算。

## 3. Softmax和交叉熵损失

我们刚刚看到softmax函数如何用作机器学习网络的一部分，以及如何使用多元链式规则计算它的导数。当我们处理这个问题的时候，经常看到损失函数和softmax一起使用来训练网络：交叉熵。

交叉熵有一个有趣的概率和信息理论解释，但在这里我只关注其使用机制。对于两个离散概率分布p和q，交叉熵函数定义为： 

![1565593961323](D:\编程\DataCamp\datanote\pic\1565593961323.png)

其中kk遍历分布定义的随机变量的所有的可能的值。具体而言，在我们的例子中有T个输出类别，所以k取值从1到T。
如果我们从softmax的输出P(一个概率分布)来考量。其它的概率分布是”正确的”类别输出，通常定义为Y，是一个大小为T的**one-hot编码**的向量，其中一个元素为1.0(该元素表示正确的类别)，其它为0。让我们重新定义该情况下的交叉熵公式:

![1565594038254](D:\编程\DataCamp\datanote\pic\1565594038254.png)

其中k遍历所有的输出类别，**P(k)是模型预测的类别的概率。Y(k)是数据集提供的真正的类别概率**。我们定义唯一的Y(k)=1.0的索引为y，因此对所有的k≠y，都有Y(k)=0，于是交叉熵公式可以简化为:

![1565594081511](D:\编程\DataCamp\datanote\pic\1565594081511.png)

实际上，我们把y当作一个常量，仅使用P来表示这个函数。进一步地，因为在我们的例子中P是一个向量，我们可以将P(y)表示为P的 第y个元素，即Py： 

![1565594103717](D:\编程\DataCamp\datanote\pic\1565594103717.png)

xent的雅可比矩阵是1×T的矩阵(一个行向量)。因为输出是一个标量且我们有T个输出(向量P有T个元素)： 

![1565594230530](D:\编程\DataCamp\datanote\pic\1565594230530.png)

现在回顾下P可以表示为输入为权值的函数：P(W)=S(g(W))。所以我们有另一个函数表示： 

![1565594313406](D:\编程\DataCamp\datanote\pic\1565594313406.png)

**【注意上式最左边的表示有歧义，以最右边的表示来理解】**

我们可以再次使用多元链式法则来计算xent关于W的梯度： 

![1565594436636](D:\编程\DataCamp\datanote\pic\1565594436636.png)

我们来检查一下雅可比行矩阵的维数。我们已经计算过了DP(W)，它是T×NT的。Dxent(P(W))是**1×T**的，所以得到的 雅可比矩阵Dxent(W)是**1×NT**的。这是有意义的，因为整个网络有一个输出(交叉熵损失，是一个标量)和NT个输入(权重)。 

同样的，有一个简单的方式来找到Dxent(W)的简单公式，因为矩阵乘法中的许多元素最终会被消除。注意到xent(P)只依赖于P的 
第y个元素。因此，在雅可比矩阵中，只有$D_yxent$是非0的： 

![1565594779407](D:\编程\DataCamp\datanote\pic\1565594779407.png)

其中，$D_yxent=-1/p_y$。回到整个的雅可比矩阵Dxent(W)，使Dxent(P)乘以D(P(W))的每一列，得到结果的行向量的每一个元素。回顾用行向量表示的按行优先的“线性化”的整个权重矩阵W。清晰起见，我们将使用i和j来索引它(DijDij)表示行向量的中的第iN+j个元素)：

![1565598146364](D:\编程\DataCamp\datanote\pic\1565598146364.png)

因为在$D_kxent(P)$中只有第y个元素是非0的，所以我们可以得到下式：

![1565598211936](D:\编程\DataCamp\datanote\pic\1565598211936.png)

根据我们的定义，$P_y=S_y$，所以可得：

![1565598241672](D:\编程\DataCamp\datanote\pic\1565598241672.png)

即使最终的结果很简洁清楚，但是我们不一定非要这样做。公式Dijxent(W)Dijxent(W)可能最终成为一个和的形式(或者某些和的和)。关于雅可比矩阵的这些技巧可能并没有太大意义，因为计算机可以完成所有的工作。我们需要做的就是计算出单个的雅矩阵，这通常毕竟容易，因为它们是更简单的非复合函数。这技术体现了多元链式法则的美妙和实用性。