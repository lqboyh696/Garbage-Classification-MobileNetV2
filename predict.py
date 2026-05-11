import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

matplotlib.rcParams['font.sans-serif'] = ['SimHei']

# =====================================================
# 1. 文件夹配置
# =====================================================

INPUT_DIR = "inputs"
OUTPUT_DIR = "results"

os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =====================================================
# 2. 设备
# =====================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("当前设备:", device)

# =====================================================
# 3. 类别
# =====================================================

class_names = [
    'cardboard（纸板）',
    'glass（玻璃）',
    'metal（金属）',
    'paper（纸张）',
    'plastic（塑料）',
    'trash（其他垃圾）'
]

# =====================================================
# 4. 加载模型
# =====================================================

model = models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(model.last_channel, len(class_names))

model.load_state_dict(torch.load("train_result/best_model.pth", map_location=device))

model = model.to(device)
model.eval()

print("模型加载成功")

# =====================================================
# 5. 图像预处理
# =====================================================

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# =====================================================
# 6. 支持的图片格式
# =====================================================

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")

# =====================================================
# 7. 批量推理
# =====================================================

images = [
    f for f in os.listdir(INPUT_DIR)
    if f.lower().endswith(IMG_EXTENSIONS)
]

if len(images) == 0:
    print("inputs 文件夹没有图片")
    exit()

print(f"\n发现 {len(images)} 张图片，开始推理...\n")

for img_name in images:

    img_path = os.path.join(INPUT_DIR, img_name)

    img = Image.open(img_path).convert("RGB")

    input_tensor = transform(img).unsqueeze(0).to(device)

    # ======================
    # 推理
    # ======================

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        pred = probs.argmax().item()

    label = class_names[pred]
    confidence = probs[pred].item()

    # ======================
    # 输出结果
    # ======================

    print(f"图片: {img_name}")
    print(f"预测: {label}")
    print(f"置信度: {confidence:.2%}")
    print("-" * 40)

    # ======================
    # 可视化并保存
    # ======================

    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.axis('off')

    plt.title(
        f"{label}\n{confidence:.2%}",
        fontsize=14
    )

    save_path = os.path.join(
        OUTPUT_DIR,
        f"result_{img_name}"
    )

    plt.savefig(save_path)
    plt.close()

print("\n全部推理完成！")
print(f"结果已保存到: {OUTPUT_DIR}")