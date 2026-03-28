"""
SceneRx Stage 2.5 - Calculator Layer
================================================
指标ID: IND_BRT
指标名称: Perceived Brightness Score (亮度感知评分)
类型: TYPE E (深度学习类)

说明:
使用Siamese CNN 模型输出的softmax置信度来量化场景的主观亮度感知。
常见设置为亮度等级分类（如 Dark / Medium / Bright），并用softmax概率作为置信度或得分。

由于需要预训练模型，此示例提供两种实现：
1. 完整实现（需要PyTorch和模型文件）
2. 占位符实现（基于简单亮度规则的估算，用于测试流程）

⚠️ 注意:
- 完整实现需要安装 PyTorch: pip install torch torchvision
- 需要预训练Siamese CNN模型文件
- 可能需要GPU支持
"""

import numpy as np
from PIL import Image
from typing import Dict
import os


# =============================================================================
# 指标定义
# =============================================================================
INDICATOR = {
    "id": "IND_BRT",
    "name": "Perceived Brightness Score",
    "unit": "confidence",
    "formula": "Softmax output of Siamese CNN",
    "target_direction": "INCREASE",
    "definition": "Subjective perceived brightness predicted by a Siamese CNN (softmax confidence)",
    "category": "CAT_COM",

    "calc_type": "deep_learning",

    # 模型配置（完整实现时使用）
    "model_config": {
        "model_type": "SiameseCNN",
        "model_path": "./models/brightness_siamese.pth",  # 需要预训练模型
        "input_size": (224, 224),
        "normalize": True,
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "num_classes": 3,
        "label_map": ["Dark", "Medium", "Bright"]
    },

    # 输出配置
    "output_type": "classification",
    "output_range": [0, 1],

    # 占位符模式（无模型时使用）
    "use_placeholder": True
}

print(f"\n✅ Calculator ready: {INDICATOR['id']} - {INDICATOR['name']}")
print(f"   Mode: {'Placeholder (rule-based)' if INDICATOR.get('use_placeholder', True) else 'Deep Learning'}")


# =============================================================================
# 检测深度学习环境
# =============================================================================
TORCH_AVAILABLE = False
try:
    import torch
    import torch.nn as nn
    import torchvision.transforms as transforms
    from torchvision import models
    TORCH_AVAILABLE = True
    print(f"   PyTorch: Available (version {torch.__version__})")
except ImportError:
    print(f"   PyTorch: Not installed")
    print(f"   To enable full DL mode: pip install torch torchvision")


# =============================================================================
# Siamese CNN (示例结构，占位)
# =============================================================================
class SiameseCNN(nn.Module):
    """
    Siamese CNN 示例结构：
    - 两个分支共享权重（这里以ResNet18 backbone 为例）
    - 输入为 (img, anchor) 或 (img1, img2)
    - 输出为分类logits（Bright/Dark/Medium）
    说明：实际训练结构可能不同，此处仅作为可加载权重的示例骨架。
    """
    def __init__(self, num_classes: int = 3):
        super().__init__()
        backbone = models.resnet18(pretrained=False)
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.classifier = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        feat = self.backbone(x)
        logits = self.classifier(feat)
        return logits


# =============================================================================
# 计算函数
# =============================================================================
def calculate_indicator(image_path: str) -> Dict:
    """
    计算 Perceived Brightness Score (亮度感知评分)

    TYPE E: 深度学习类

    根据配置选择实现方式:
    - 占位符模式: 基于平均亮度的简单估算（映射为0-1置信度）
    - 完整模式: Siamese CNN 分类softmax输出

    Args:
        image_path: 图片路径（建议使用原始图片）

    Returns:
        {
            'success': True/False,
            'value': float (0-1),
            'method': str,
            'confidence': float,
            'predicted_class': str/int,
            'softmax': dict (可选)
        }
    """
    use_placeholder = INDICATOR.get('use_placeholder', True)

    if use_placeholder or not TORCH_AVAILABLE:
        return calculate_placeholder(image_path)
    else:
        return calculate_deep_learning(image_path)


def calculate_placeholder(image_path: str) -> Dict:
    """
    占位符实现：基于平均亮度估算主观亮度感知

    算法:
    1. 转灰度并计算平均亮度 mean_gray (0-255)
    2. 将 mean_gray 线性映射到 0-1 作为亮度得分
    3. 构造一个伪softmax分布（Dark/Medium/Bright）用于流程测试

    注意: 仅用于测试流程，不是真正的深度学习预测
    """
    try:
        img = Image.open(image_path).convert('RGB')
        rgb = np.array(img, dtype=np.float64)

        gray = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
        mean_gray = float(np.mean(gray))

        # 0-1得分
        score = mean_gray / 255.0
        score = max(0.0, min(1.0, score))

        # 伪softmax（用三角形隶属函数）
        # Dark centered at 40, Medium at 128, Bright at 220
        def tri(x, c, w):
            d = abs(x - c)
            return max(0.0, 1.0 - d / w)

        p_dark = tri(mean_gray, 40, 80)
        p_med = tri(mean_gray, 128, 90)
        p_brt = tri(mean_gray, 220, 80)

        s = p_dark + p_med + p_brt
        if s <= 0:
            p_dark, p_med, p_brt = 1/3, 1/3, 1/3
        else:
            p_dark, p_med, p_brt = p_dark / s, p_med / s, p_brt / s

        probs = {"Dark": round(p_dark, 4), "Medium": round(p_med, 4), "Bright": round(p_brt, 4)}
        pred_class = max(probs, key=probs.get)
        confidence = float(probs[pred_class])

        return {
            'success': True,
            'value': round(float(score), 3),
            'method': 'placeholder_rule_based',
            'confidence': round(float(confidence), 3),
            'predicted_class': pred_class,
            'mean_gray': round(mean_gray, 3),
            'softmax': probs,
            'note': 'This is a placeholder estimation, not a deep learning prediction'
        }

    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'value': None,
            'method': 'placeholder_rule_based'
        }


def calculate_deep_learning(image_path: str) -> Dict:
    """
    完整实现：使用预训练 Siamese CNN 分类模型输出softmax置信度

    需要:
    - PyTorch
    - 预训练模型文件（输出 logits, shape [1, num_classes]）
    """
    try:
        model_config = INDICATOR.get('model_config', {})
        model_path = model_config.get('model_path', '')

        if not os.path.exists(model_path):
            return {
                'success': False,
                'error': f'Model file not found: {model_path}',
                'value': None,
                'method': 'deep_learning',
                'fallback': 'Run with use_placeholder=True or provide model file'
            }

        num_classes = int(model_config.get('num_classes', 3))
        label_map = model_config.get('label_map', None)

        model = SiameseCNN(num_classes=num_classes)
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)

        input_size = model_config.get('input_size', (224, 224))
        mean = model_config.get('mean', [0.485, 0.456, 0.406])
        std = model_config.get('std', [0.229, 0.224, 0.225])

        transform = transforms.Compose([
            transforms.Resize(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

        img = Image.open(image_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(img_tensor)  # [1, C]
            probs = torch.softmax(logits, dim=1).squeeze(0)  # [C]

        conf, pred = torch.max(probs, dim=0)
        conf_value = float(conf.cpu().numpy())
        pred_idx = int(pred.cpu().numpy())

        if isinstance(label_map, (list, tuple)) and pred_idx < len(label_map):
            pred_label = label_map[pred_idx]
        elif isinstance(label_map, dict) and pred_idx in label_map:
            pred_label = label_map[pred_idx]
        else:
            pred_label = pred_idx

        # softmax dict
        probs_list = probs.cpu().numpy().tolist()
        softmax = {}
        for i in range(num_classes):
            if isinstance(label_map, (list, tuple)) and i < len(label_map):
                k = label_map[i]
            elif isinstance(label_map, dict) and i in label_map:
                k = label_map[i]
            else:
                k = str(i)
            softmax[k] = round(float(probs_list[i]), 4)

        # 也可将“Brightness Score”定义为Bright类别的概率
        # 若无Bright类别，退回使用max conf
        if isinstance(label_map, (list, tuple)) and "Bright" in label_map:
            bright_idx = label_map.index("Bright")
            value = float(probs_list[bright_idx])
        else:
            value = conf_value

        return {
            'success': True,
            'value': round(float(value), 3),
            'method': 'deep_learning',
            'model_type': model_config.get('model_type', 'SiameseCNN'),
            'device': str(device),
            'confidence': round(float(conf_value), 3),
            'predicted_class': pred_label,
            'softmax': softmax
        }

    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'value': None,
            'method': 'deep_learning'
        }


# =============================================================================
# 辅助函数
# =============================================================================
def interpret_brightness(score: float) -> str:
    """解释亮度感知评分（0-1）"""
    if score < 0.2:
        return "Very dark appearance"
    elif score < 0.4:
        return "Dark appearance"
    elif score < 0.6:
        return "Moderate brightness"
    elif score < 0.8:
        return "Bright appearance"
    else:
        return "Very bright appearance"


# =============================================================================
# 测试代码
# =============================================================================
if __name__ == "__main__":
    print("\n🧪 Testing Perceived Brightness Score calculator...")

    dark = np.zeros((120, 120, 3), dtype=np.uint8) + 20
    mid = np.zeros((120, 120, 3), dtype=np.uint8) + 128
    bright = np.zeros((120, 120, 3), dtype=np.uint8) + 235

    for name, test_img in [('Dark', dark), ('Medium', mid), ('Bright', bright)]:
        test_path = f'/tmp/test_brt_{name}.png'
        Image.fromarray(test_img).save(test_path)

        result = calculate_indicator(test_path)

        print(f"\n   {name}:")
        print(f"      Score: {result['value']} (0-1)")
        print(f"      Method: {result['method']}")
        if 'mean_gray' in result:
            print(f"      Mean gray: {result['mean_gray']}")
        if 'predicted_class' in result:
            print(f"      Pred: {result['predicted_class']} (conf={result.get('confidence')})")
        print(f"      Interpretation: {interpret_brightness(result['value'])}")

        os.remove(test_path)
