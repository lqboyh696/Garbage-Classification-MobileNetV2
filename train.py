import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
import os
import csv
import matplotlib.pyplot as plt

# =====================================================
# 1. 输出目录
# =====================================================

SAVE_DIR = "train_result"
os.makedirs(SAVE_DIR, exist_ok=True)

csv_path = os.path.join(SAVE_DIR, "train_log.csv")
loss_fig_path = os.path.join(SAVE_DIR, "loss_curve.png")
acc_fig_path = os.path.join(SAVE_DIR, "acc_curve.png")

# =====================================================
# 2. GPU
# =====================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 50)
print("设备:", device)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
print("=" * 50)

# =====================================================
# 3. 数据增强
# =====================================================

# 训练集：随机翻转 + 随机旋转，增加数据多样性，防止过拟合
train_transform = transforms.Compose([
    transforms.Resize((128, 128)),  # 缩小尺寸，加快训练速度
    transforms.RandomHorizontalFlip(),  # 随机水平翻转
    transforms.RandomRotation(15),  # 随机旋转15度
    transforms.ToTensor(),  # 转为张量(0~1)
    transforms.Normalize(  # 标准化（ImageNet均值/标准差）
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
])

# 验证集：只做基础处理，不做随机增强
val_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
])

# =====================================================
# 4. 数据集
# =====================================================

dataset_path = "garbage-classification"

# 用两个独立的dataset对象分别应用不同的预处理
# 注意：不能对random_split的结果直接设.dataset.transform，
# 因为两个subset共享同一个底层对象，后设的会覆盖先设的
train_data = datasets.ImageFolder(dataset_path, transform=train_transform)
val_data = datasets.ImageFolder(dataset_path, transform=val_transform)

class_names = train_data.classes
print("类别:", class_names)

# 按相同的随机种子划分索引，保证两个dataset划分一致
n_total = len(train_data)
n_train = int(n_total * 0.8)
n_val = n_total - n_train
indices = torch.randperm(n_total).tolist()
train_indices = indices[:n_train]
val_indices = indices[n_train:]

train_set = Subset(train_data, train_indices)
val_set = Subset(val_data, val_indices)

print(f"训练集: {n_train} 张，验证集: {n_val} 张")

# =====================================================
# 5. DataLoader
# =====================================================

train_loader = DataLoader(train_set, batch_size=16,
                          shuffle=True, num_workers=0)

val_loader = DataLoader(val_set, batch_size=16,
                        shuffle=False, num_workers=0)

# =====================================================
# 6. 模型
# =====================================================

# MobileNetV2是轻量级网络，适合快速训练
model = models.mobilenet_v2(weights="IMAGENET1K_V1")

# 冻结主干网络，只训练最后的分类头（加快训练速度）
for param in model.features.parameters():
    param.requires_grad = False

# 将原来的1000类分类头替换为实际类别数
model.classifier[1] = nn.Linear(model.last_channel, len(class_names))

model = model.to(device)
print("模型加载完成，开始训练...")

# =====================================================
# 7. 损失 & 优化器
# =====================================================

criterion = nn.CrossEntropyLoss()  # 多分类交叉熵损失
optimizer = torch.optim.Adam(model.classifier.parameters(), lr=0.001)

# =====================================================
# 8. CSV 初始化
# =====================================================

with open(csv_path, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["epoch", "loss", "train_acc", "val_acc"])

# =====================================================
# 9. 训练
# =====================================================

EPOCHS = 20
best_acc = 0.0

train_loss_list = []
train_acc_list = []
val_acc_list = []

print("\n开始训练...\n")

for epoch in range(EPOCHS):

    # ------------------
    # train
    # ------------------
    model.train()
    total_loss = 0
    correct = 0

    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()

    train_acc = correct / n_train
    avg_loss = total_loss / len(train_loader)

    # ------------------
    # val
    # ------------------
    model.eval()
    val_correct = 0

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            val_correct += (outputs.argmax(1) == labels).sum().item()

    val_acc = val_correct / n_val

    # ------------------
    # 保存日志
    # ------------------
    with open(csv_path, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([epoch+1, avg_loss, train_acc, val_acc])

    # 记录
    train_loss_list.append(avg_loss)
    train_acc_list.append(train_acc)
    val_acc_list.append(val_acc)

    print(f"Epoch[{epoch+1}/{EPOCHS}] "
          f"Loss:{avg_loss:.3f} "
          f"TrainAcc:{train_acc:.3f} "
          f"ValAcc:{val_acc:.3f}")

    # 保存最优模型权重
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(),
                   os.path.join(SAVE_DIR, "best_model.pth"))
        print(f"已保存最优权重 (ValAcc:{best_acc:.3f})")

print(f"\n训练完成！最优验证准确率: {best_acc:.3f}")
print(f"权重已保存至: {os.path.join(SAVE_DIR, 'best_model.pth')}")

# =====================================================
# 10. 画 Loss 曲线
# =====================================================

plt.figure()
plt.plot(train_loss_list)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.savefig(loss_fig_path)
plt.close()

# =====================================================
# 11. 画 Accuracy 曲线
# =====================================================

plt.figure()
plt.plot(train_acc_list, label="Train Acc")
plt.plot(val_acc_list, label="Val Acc")
plt.title("Accuracy Curve")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig(acc_fig_path)
plt.close()

print("结果已保存到:", SAVE_DIR)