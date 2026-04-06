import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# ── 1. 数据 ──────────────────────────────────────────────
transform = transforms.Compose([
    transforms.ToTensor(),           # 图片 → Tensor，像素值从[0,255]变成[0,1]
    transforms.Normalize((0.1307,), (0.3081,))  # 标准化（MNIST的均值和标准差）
])

train_dataset = datasets.MNIST(root='./data', train=True,  download=True, transform=transform)
test_dataset  = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=64, shuffle=False)

print(f"训练集大小: {len(train_dataset)} 张")
print(f"测试集大小: {len(test_dataset)} 张")
print(f"图片尺寸: {train_dataset[0][0].shape}")  # [1, 28, 28]

# ── 2. 模型 ──────────────────────────────────────────────
class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),           # [1,28,28] → [784]
            nn.Linear(784, 256),    # 全连接层
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)      # 输出10个类别（0-9）的logits
        )

    def forward(self, x):
        return self.network(x)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n使用设备: {device}")

model = SimpleNN().to(device)
print(model)

# ── 3. 训练配置 ───────────────────────────────────────────
criterion = nn.CrossEntropyLoss()          # 多分类用交叉熵
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# ── 4. 训练循环 ───────────────────────────────────────────
def train(model, loader, optimizer, criterion, epoch):
    model.train()
    total_loss, correct = 0, 0

    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()          # 清空梯度
        outputs = model(images)        # 前向传播
        loss = criterion(outputs, labels)  # 计算loss
        loss.backward()                # 反向传播
        optimizer.step()               # 更新参数

        total_loss += loss.item()
        pred = outputs.argmax(dim=1)   # 取概率最大的类别
        correct += (pred == labels).sum().item()

        if batch_idx % 200 == 0:
            print(f"  Epoch {epoch} [{batch_idx*64}/{len(loader.dataset)}] "
                  f"Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(loader)
    accuracy = correct / len(loader.dataset) * 100
    return avg_loss, accuracy

def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct = 0, 0

    with torch.no_grad():              # 评估时不需要计算梯度
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            pred = outputs.argmax(dim=1)
            correct += (pred == labels).sum().item()

    avg_loss = total_loss / len(loader)
    accuracy = correct / len(loader.dataset) * 100
    return avg_loss, accuracy

# ── 5. 开始训练 ───────────────────────────────────────────
EPOCHS = 5
train_losses, test_accuracies = [], []

print("\n开始训练...\n")
for epoch in range(1, EPOCHS + 1):
    train_loss, train_acc = train(model, train_loader, optimizer, criterion, epoch)
    test_loss,  test_acc  = evaluate(model, test_loader, criterion)
    train_losses.append(train_loss)
    test_accuracies.append(test_acc)
    print(f"Epoch {epoch}: 训练Loss={train_loss:.4f}, 训练准确率={train_acc:.2f}%, 测试准确率={test_acc:.2f}%\n")

# ── 6. 可视化结果 ─────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Loss曲线
axes[0].plot(range(1, EPOCHS+1), train_losses, 'b-o')
axes[0].set_title('训练 Loss')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].grid(True)

# 准确率曲线
axes[1].plot(range(1, EPOCHS+1), test_accuracies, 'g-o')
axes[1].set_title('测试准确率 (%)')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy (%)')
axes[1].set_ylim([90, 100])
axes[1].grid(True)

plt.tight_layout()
plt.savefig('training_curves.png', dpi=150)
print("训练曲线已保存为 training_curves.png")

# ── 7. 看几个预测例子 ─────────────────────────────────────
model.eval()
images, labels = next(iter(test_loader))
with torch.no_grad():
    outputs = model(images.to(device))
    preds = outputs.argmax(dim=1).cpu()

fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for i, ax in enumerate(axes.flat):
    ax.imshow(images[i].squeeze(), cmap='gray')
    color = 'green' if preds[i] == labels[i] else 'red'
    ax.set_title(f"预测:{preds[i].item()}  真实:{labels[i].item()}", color=color)
    ax.axis('off')

plt.suptitle("测试集预测结果（绿色=正确，红色=错误）", fontsize=13)
plt.tight_layout()
plt.savefig('predictions.png', dpi=150)
print("预测示例已保存为 predictions.png")
print(f"\n最终测试准确率: {test_accuracies[-1]:.2f}%")
