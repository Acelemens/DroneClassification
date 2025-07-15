import os
import torch
import random
import matplotlib.pyplot as plt
from torchvision import datasets, transforms, models
import matplotlib
matplotlib.rcParams['font.family'] = 'SimHei'

# 设置路径
test_dir = r"/intel_dataset/test"
model_path = r"../models/resnet18_base.pth"
class_names = ['buildings', 'forest', 'mountain']

# 图像预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# 加载数据
dataset = datasets.ImageFolder(test_dir, transform=transform)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载模型
model = models.resnet18()
model.fc = torch.nn.Linear(model.fc.in_features, 3)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# 随机抽取若干张图片进行可视化
num_images = 10
indices = random.sample(range(len(dataset)), num_images)
fig = plt.figure(figsize=(15, 10))

for i, idx in enumerate(indices):
    image_tensor, label = dataset[idx]
    image = image_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        _, pred = torch.max(output, 1)

    pred_label = class_names[pred.item()]
    true_label = class_names[label]

    # 显示图片
    img_np = image_tensor.permute(1, 2, 0).numpy()
    ax = fig.add_subplot(2, 5, i + 1)
    ax.imshow(img_np)
    ax.axis('off')
    title_color = 'green' if pred_label == true_label else 'red'
    ax.set_title(f"预测: {pred_label}\n实际: {true_label}", color=title_color, fontsize=10)

plt.tight_layout()
plt.show()
