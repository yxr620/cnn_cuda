# GPU 加速的一些结果

## 1. 卷积模型

### 1.1 模型结构

模型修改成Pytorch架构如下：

```python
class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        self.fc1 = nn.Linear(84, 16)
        self.fc2 = nn.Linear(16, 1)
        self.conv1 = nn.Conv1d(1, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(8, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=3)

    def forward(self, x):
        x = F.mish(self.fc1(x))
        x = x.view(-1, 1, 16)
        x = F.mish(self.conv1(x))
        x = self.pool(x)
        x = F.mish(self.conv2(x))
        x = torch.max(x, dim=2)[0]
        x = self.fc2(x)
        return x
```

此处模型的结构没有改变，始终是一维的模型，由全连接层、一维卷积层、池化层组成。模型的激活函数由之前的swish改变为了mish，因为pytorch没有原生的swish函数支持。

### 1.2 数据生成

由于实验数据为一维数据，因此直接使用下面代码生成随机数据，可以直接生成1e6个随机数据点。

```python
train_data = torch.randn(1000000, 84)
train_labels = torch.randn(1000000, 1)

val_data = torch.randn(2000, 84)
val_labels = torch.randn(2000, 1)
```

### 1.3 结果分析

将生成的数据放入模型中进行10个epoch的训练，结果如下：

| Batch | GPU     | CPU     |
| ----- | ------- | ------- |
| 16    | 1619.64 | 2002.33 |
| 512   | 170.05  | 237.91  |
| 1024  | 125.06  | 209.44  |
| 4096  | 105.79  | 161.89  |

此外如果使用的数据集大小较小（1e3~1e5），CPU的运行速度会比GPU更快。

从该模型的运行时间结果出发，对比之前使用GRU模型运行的时间我们可以得到如下两个结论：

1. 使用较少的数据集GPU对与模型加速不明显，甚至可能降低；
2. 一维模型的GPU提速效果远不如二维的模型。反映到结果是在一维数据集上的模型提升速度只有30%左右，但是在使用多特征GRU模型时GPU的对模型速度的提升有10000%。



## 2. 注意力模型TabNet

### 2.1 模型结构

TabNet的模型直接使用`pytorch-tabnet`实现，github的网址为[github](https://github.com/dreamquark-ai/tabnet)。模型的参数定义如下：

```python
model = TabNetRegressor(
    n_d=17,
    n_a=17,
    n_steps=1,
    gamma=1.5,
    n_independent=2,
    n_shared=2,
    seed=0,
    device_name=device_name
)
```
