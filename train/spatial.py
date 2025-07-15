import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, models, transforms
from torchvision.models.resnet import ResNet, BasicBlock
from torch.utils.data import DataLoader
from tqdm import tqdm

# 1. 数据路径配置
train_dir = r"/intel_dataset/train"
val_dir = r"/intel_dataset/val"
save_path = r"/models/resnet18_spatial.pth"
os.makedirs(os.path.dirname(save_path), exist_ok=True)

# 2. 数据增强与加载
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ]),
}

train_dataset = datasets.ImageFolder(train_dir, transform=data_transforms['train'])
val_dataset = datasets.ImageFolder(val_dir, transform=data_transforms['val'])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# 3. 空间注意力模块
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        return x * self.sigmoid(self.conv(x_cat))

# 4. 加入空间注意力的 ResNet18
class ResNetWithSpatialAttention(ResNet):
    def __init__(self):
        super().__init__(block=BasicBlock, layers=[2, 2, 2, 2])
        self.sa1 = SpatialAttention()
        self.sa2 = SpatialAttention()
        self.sa3 = SpatialAttention()
        self.sa4 = SpatialAttention()
        self.fc = nn.Linear(512, 3)  # 输出3类

    def _forward_impl(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x); x = self.sa1(x)
        x = self.layer2(x); x = self.sa2(x)
        x = self.layer3(x); x = self.sa3(x)
        x = self.layer4(x); x = self.sa4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# 5. 模型加载与设置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")
if torch.cuda.is_available():
    print(f"GPU数量: {torch.cuda.device_count()}")
    print(f"当前GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

model = ResNetWithSpatialAttention()
# 加载预训练参数（仅加载匹配部分）
pretrained_dict = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1).state_dict()
model_dict = model.state_dict()
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
model_dict.update(pretrained_dict)
model.load_state_dict(model_dict)
model = model.to(device)

# 6. 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# 7. 训练与验证
# 早停验证配置
patience = 5  # 耐心值
best_val_loss = float('inf')
patience_counter = 0
best_model_state = None

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
    print(f"[Epoch {epoch+1}] 训练损失: {train_loss:.4f}")

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
    print(f"验证损失: {val_loss:.4f}, 验证准确率：{acc:.2f}%")

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

# ============================== #
# 8. 保存最佳模型
# ============================== #
if best_model_state is not None:
    model.load_state_dict(best_model_state)
    torch.save(best_model_state, save_path)
    print(f"最佳模型已保存为 {save_path}")
    print(f"最佳验证损失: {best_val_loss:.4f}")
else:
    torch.save(model.state_dict(), save_path)
    print(f"模型已保存为 {save_path}")
