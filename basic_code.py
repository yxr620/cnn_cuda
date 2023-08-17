import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import time

# 检查是否有可用的GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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


class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index]
        y = self.labels[index]
        return x, y


def train(model, train_loader, criterion, optimizer):
    model.train()
    train_loss = 0
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    return train_loss / len(train_loader)


def evaluate(model, val_loader, criterion):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
    return val_loss / len(val_loader)


def main():
    # 生成大量的训练数据和标签
    train_data = torch.randn(10000, 84)
    train_labels = torch.randn(10000, 1)

    # 生成大量的验证数据和标签
    val_data = torch.randn(2000, 84)
    val_labels = torch.randn(2000, 1)

    # 创建数据集和数据加载器
    train_dataset = CustomDataset(train_data, train_labels)
    val_dataset = CustomDataset(val_data, val_labels)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    # 创建模型实例并将其移动到设备（GPU或CPU）
    model = BaseModel().to(device)

    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练和验证循环
    num_epochs = 10

    # 在GPU上进行训练和验证
    start_time = time.time()
    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, criterion, optimizer)
        val_loss = evaluate(model, val_loader, criterion)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    gpu_time = time.time() - start_time

    # 将模型移动到CPU上
    model = model.to("cpu")

    # 创建新的数据加载器（在CPU上）
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    # 在CPU上进行训练和验证
    start_time = time.time()
    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, criterion, optimizer)
        val_loss = evaluate(model, val_loader, criterion)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    cpu_time = time.time() - start_time

    print(f"GPU Time: {gpu_time:.2f}s CPU Time: {cpu_time:.2f}s")

if __name__ == "__main__":
    main()