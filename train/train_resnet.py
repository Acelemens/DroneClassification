import os
import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

# 1. 数据路径与模型保存路径
train_dir = r"/intel_dataset/train"
val_dir = r"/intel_dataset/val"
save_path = r"/models/resnet18_base.pth"
os.makedirs(os.path.dirname(save_path), exist_ok=True)

# 2. 数据预处理
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]),
}

train_dataset = datasets.ImageFolder(train_dir, transform=data_transforms['train'])
val_dataset = datasets.ImageFolder(val_dir, transform=data_transforms['val'])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# 3. 模型与设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

if torch.cuda.is_available():
    print(f"GPU数量: {torch.cuda.device_count()}")
    print(f"当前GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, 3)
model = model.to(device)

# 4. 损失函数 & 优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# 5. 早停验证配置
patience = 5  # 耐心值
best_val_loss = float('inf')
patience_counter = 0
best_model_state = None

# 6. 模型训练与验证
epochs = 50  # 增加最大训练轮数
for epoch in range(epochs):
    # 训练阶段
    model.train()
    running_loss = 0.0

    for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    train_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch+1} 训练损失: {train_loss:.4f}")

    # 验证阶段
    model.eval()
    val_loss = 0.0
    correct = total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_loss = val_loss / len(val_loader)
    acc = correct / total * 100
    print(f"验证损失: {val_loss:.4f}, 验证准确率: {acc:.2f}%")

    # 早停检查
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        best_model_state = model.state_dict().copy()
        print(f"新的最佳验证损失: {best_val_loss:.4f}")
    else:
        patience_counter += 1
        print(f"验证损失未改善，耐心计数: {patience_counter}/{patience}")
        
        if patience_counter >= patience:
            print(f"早停触发！在 Epoch {epoch+1} 停止训练")
            break

# 7. 保存最佳模型
if best_model_state is not None:
    model.load_state_dict(best_model_state)
    torch.save(best_model_state, save_path)
    print(f"最佳模型已保存为: {save_path}")
    print(f"最佳验证损失: {best_val_loss:.4f}")
else:
    torch.save(model.state_dict(), save_path)
    print(f"模型已保存为: {save_path}")
