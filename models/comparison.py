import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
from tqdm import tqdm

# ============ 配置 ============ #
test_dir = r"D:\Python\PythonProject11\intel_dataset\test"
model_dir = r"D:\Python\PythonProject11\models"
class_names = ['buildings', 'forest', 'mountain']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============ 模型映射 ============ #
model_files = {
    "Base": "resnet18_base.pth",
    "Spatial": "resnet18_spatial.pth",
    "Channel": "resnet18_channel.pth",
    "SelfAtt": "resnet18_selfatt_light.pth",
    "CBAM": "resnet18_cbam.pth",
    "CA+SelfAtt": "resnet18_ca_sa.pth",
    "SA+SelfAtt": "resnet18_sa_sa.pth",
    "AllAttention": "resnet18_all_att.pth"
}

# ============ 数据加载 ============ #
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
test_dataset = datasets.ImageFolder(test_dir, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# ============ 导入模型结构 ============ #
print("正在导入所有模型结构...")
from models.model_architectures import (
    ResNetWithSpatialAttention,
    ResNetWithChannelAttention,
    ResNetWithSelfAttentionLite,
    ResNetWithCBAM,
    ResNetWithCA_SelfAttention,
    ResNetWithSA_SelfAttention,
    ResNetWithAllAttention
)

# ============ 获取模型结构 ============ #
def get_model(model_name):
    if model_name == "Base":
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 3)
        return model
    elif model_name == "Spatial":
        return ResNetWithSpatialAttention()
    elif model_name == "Channel":
        return ResNetWithChannelAttention()
    elif model_name == "SelfAtt":
        return ResNetWithSelfAttentionLite()
    elif model_name == "CBAM":
        return ResNetWithCBAM()
    elif model_name == "CA+SelfAtt":
        return ResNetWithCA_SelfAttention()
    elif model_name == "SA+SelfAtt":
        return ResNetWithSA_SelfAttention()
    elif model_name == "AllAttention":
        return ResNetWithAllAttention()
    else:
        raise ValueError(f"未知模型名：{model_name}")

# ============ 模型评估 ============ #
def evaluate_model(model_name, model_path):
    model = get_model(model_name)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    y_true, y_pred = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    return acc, prec, rec, f1

# ============ 主逻辑 ============ #
results = []
for name, filename in tqdm(model_files.items(), desc="评估模型中"):
    model_path = os.path.join(model_dir, filename)
    if not os.path.exists(model_path):
        print(f"模型文件不存在: {model_path}")
        continue
    acc, prec, rec, f1 = evaluate_model(name, model_path)
    results.append({
        "模型": name,
        "准确率": acc,
        "精确率": prec,
        "召回率": rec,
        "F1 分数": f1
    })

# ============ 输出结果表格 ============ #
df = pd.DataFrame(results)
df_sorted = df.sort_values("F1 分数", ascending=False)

df_formatted = df_sorted.copy()
for col in ["准确率", "精确率", "召回率", "F1 分数"]:
    df_formatted[col] = df_formatted[col].apply(lambda x: f"{x * 100:.2f}%")

print("\n各模型在测试集上的表现对比（按 F1 分数排序）：\n")
print(df_formatted.to_string(index=False))
