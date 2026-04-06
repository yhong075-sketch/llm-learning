# MNIST 深度学习笔记

> 基于 `mnist_tutorial.py` 的学习记录，涵盖 PyTorch 神经网络训练完整流程。

---

## 目录

1. [数据处理](#1-数据处理)
2. [模型结构](#2-模型结构)
3. [损失函数](#3-损失函数)
4. [优化器](#4-优化器)
5. [训练循环（核心）](#5-训练循环核心)
6. [评估与预测](#6-评估与预测)

---

## 1. 数据处理

### MNIST 数据集

- 手写数字识别数据集（0–9 共 10 类）
- 训练集：60,000 张 / 测试集：10,000 张
- 每张图片尺寸：28 × 28 像素，灰度图

### 数据预处理

```python
transform = transforms.Compose([
    transforms.ToTensor(),                         # 像素 [0,255] → [0,1]
    transforms.Normalize((0.1307,), (0.3081,))    # 标准化（MNIST 的均值和标准差）
])
```

| 步骤 | 作用 |
|------|------|
| `ToTensor()` | 把图片转成 Tensor，像素值归一化到 [0,1] |
| `Normalize()` | 减均值除标准差，让数据均值≈0，方差≈1，训练更稳定 |

### DataLoader

```python
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
```

- **batch_size=64**：每次喂给网络 64 张图片（而非一次性全部），节省内存
- **shuffle=True**：每个 epoch 打乱顺序，防止模型记住顺序规律

---

## 2. 模型结构

### SimpleNN 网络

```python
class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),           # [1,28,28] → [784]
            nn.Linear(784, 256),    # 全连接层
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)      # 输出 10 个类别的 logits
        )
```

### 各层解析

| 层 | 输入 → 输出 | 说明 |
|----|-------------|------|
| `Flatten` | [1,28,28] → [784] | 把二维图片"拍平"成一维向量 |
| `Linear(784,256)` | 784 → 256 | 全连接层，学习特征 |
| `ReLU` | — | 激活函数：`max(0, x)` |
| `Linear(256,128)` | 256 → 128 | 进一步提取特征 |
| `ReLU` | — | 激活函数 |
| `Linear(128,10)` | 128 → 10 | 输出 10 个数字对应 0–9 的得分（logits） |

### 关键概念

- **全连接层（Linear）**：每个输入节点与每个输出节点都相连，参数量 = 输入 × 输出
- **ReLU 激活函数**：给网络引入非线性，公式 `f(x) = max(0, x)`，让网络能学习复杂规律
- **logits**：最后一层的原始输出，还不是概率，需经过 Softmax 才是概率

---

## 3. 损失函数

```python
criterion = nn.CrossEntropyLoss()
```

### 交叉熵损失（CrossEntropyLoss）

- 用于**多分类**任务的标准损失函数
- 内部自动包含 Softmax，不需要手动加
- loss 越小，代表模型预测越准确
- 公式：

$$L = -\sum_{i} y_i \log(\hat{y}_i)$$

其中 $y_i$ 是真实标签（one-hot），$\hat{y}_i$ 是预测概率。

---

## 4. 优化器

```python
optimizer = optim.Adam(model.parameters(), lr=1e-3)
```

### Adam 优化器

- 目前最常用的优化器，自动调整每个参数的学习率
- 比传统 SGD 收敛更快、更稳定

### 学习率（lr）

- `lr=1e-3` 即 0.001
- 控制每次参数更新的步长
- 太大 → 震荡不收敛 / 太小 → 训练极慢

---

## 5. 训练循环（核心）

### 完整流程（4 步）

```python
optimizer.zero_grad()              # ① 清空梯度
outputs = model(images)            # ② 前向传播
loss = criterion(outputs, labels)  # ③ 计算 loss
loss.backward()                    # ④ 反向传播（计算梯度）
optimizer.step()                   # ⑤ 更新参数
```

### 每步解释

| 步骤 | 作用 |
|------|------|
| `zero_grad()` | 清空上一步的梯度，否则梯度会累积出错 |
| 前向传播 | 图片经过网络，得到预测结果 `outputs` |
| 计算 loss | 衡量预测和真实标签的差距 |
| `backward()` | 自动计算每个参数对 loss 的"责任"（梯度） |
| `step()` | 按梯度方向更新参数，让下次预测更准 |

### Epoch

- 一个 **epoch** = 把训练集完整过一遍
- 本例训练了 5 个 epoch

---

## 6. 评估与预测

```python
model.eval()
with torch.no_grad():
    outputs = model(images)
    preds = outputs.argmax(dim=1)
```

### 关键点

| 代码 | 作用 |
|------|------|
| `model.eval()` | 切换到评估模式（Dropout、BatchNorm 行为改变） |
| `torch.no_grad()` | 关闭梯度计算，节省内存和时间 |
| `argmax(dim=1)` | 取概率最大的类别作为预测结果 |

### 准确率计算

```python
pred = outputs.argmax(dim=1)
correct += (pred == labels).sum().item()
accuracy = correct / total * 100
```

---

## 总结：训练流程图

```
数据集 (MNIST)
    ↓ DataLoader (batch_size=64)
输入图片 [64, 1, 28, 28]
    ↓ Flatten → Linear → ReLU → Linear → ReLU → Linear
模型输出 logits [64, 10]
    ↓ CrossEntropyLoss
loss（标量）
    ↓ backward()
梯度
    ↓ Adam.step()
更新参数 → 下一次更循环
```

## 最终效果

- 训练 5 个 epoch，测试准确率约 **97–98%**
- 模型参数量：784×256 + 256×128 + 128×10 ≈ **234,000 个参数**

---

*参考代码：`mnist_tutorial.py`*
