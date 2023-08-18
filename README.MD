# GPU 加速的一些结果

## 1. 模型结构

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

## 2. 数据生成

由于实验数据为一维数据，因此直接使用下面代码生成随机数据，可以直接生成1e6个随机数据点。

```python
train_data = torch.randn(1000000, 84)
train_labels = torch.randn(1000000, 1)

val_data = torch.randn(2000, 84)
val_labels = torch.randn(2000, 1)
```

将生成的数据放入模型中进行10个epoch的训练，结果如下：

| Batch | GPU     | CPU     |
| ----- | ------- | ------- |
| 16    | 1619.64 | 2002.33 |
| 512   | 170.05  | 237.91  |
| 1024  | 125.06  | 209.44  |
| 4096  | 105.79  | 161.89  |
