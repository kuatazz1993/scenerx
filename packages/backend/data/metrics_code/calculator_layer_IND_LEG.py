"""
SceneRx Stage 2.5 - Calculator Layer
================================================
指标ID: IND_LEG
指标名称: Legibility Index (可读性指数)
类型: TYPE E (深度学习类)

说明:
使用深度学习分类模型的置信度来量化空间可读性（Legibility）。
通常认为：模型对场景类别判断越“自信”（softmax最大概率越高），空间越容易被识别与组织。

由于需要预训练模型，此示例提供两种实现：
1. 完整实现（需要PyTorch和模型文件）
2. 占位符实现（基于简单规则的估算，用于测试流程）

⚠️ 注意:
- 完整实现需要安装 PyTorch: pip install torch torchvision
- 需要预训练分类模型文件（输出类别 logits）
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
    "id": "IND_LEG",
    "name": "Legibility Index",
    "unit": "confidence",
    "formula": "Softmax probability vector output (max probability as confidence)",
    "target_direction": "INCREASE",
    "definition": "Legibility quantified by the confidence score of a deep learning classification model (softmax max-prob)",
    "category": "CAT_COM",

    "calc_type": "deep_learning",

    # 模型配置（完整实现时使用）
    "model_config": {
        "model_type": "ResNet50",
        "model_path": "./models/legibility_resnet50_cls.pth",  # 需要预训练模型（分类）
        "input_size": (224, 224),
        "normalize": True,
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "num_classes": 10,  # 需与你的训练类别数一致
        "label_map": None   # 可选：类别名称列表/字典
    },

    # 输出配置
    "output_type": "classification",
    "output_range": [0, 1],  # 置信度范围

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
    import torchvision.transforms as transforms
    from torchvision import models
    TORCH_AVAILABLE = True
    print(f"   PyTorch: Available (version {torch.__version__})")
except ImportError:
    print(f"   PyTorch: Not installed")
    print(f"   To enable full DL mode: pip install torch torchvision")


# =============================================================================
# 计算函数
# =============================================================================
def calculate_indicator(image_path: str) -> Dict:
    """
    计算 Legibility Index (可读性指数)

    TYPE E: 深度学习类

    根据配置选择实现方式:
    - 占位符模式: 基于图像信息密度（压缩比/边缘强度）的简单估算
    - 完整模式: 使用预训练分类CNN输出softmax置信度

    Args:
        image_path: 图片路径（建议使用原始图片而非mask）

    Returns:
        {
            'success': True/False,
            'value': float (可读性置信度 0-1),
            'method': str,
            'confidence': float,
            'predicted_class': int/str (如适用),
            'topk': list (如适用)
        }
    """
    use_placeholder = INDICATOR.get('use_placeholder', True)

    if use_placeholder or not TORCH_AVAILABLE:
        return calculate_placeholder(image_path)
    else:
        return calculate_deep_learning(image_path)


def calculate_placeholder(image_path: str) -> Dict:
    """
    占位符实现：基于规则的可读性估算

    思路:
    - 可读性高的空间往往具有更清晰的结构与更低的视觉噪声
    - 用两个可计算代理量进行估计：
      1) 边缘强度（Laplacian方差）越高 → 结构更明确（正向）
      2) JPEG压缩比越高 → 信息密度/复杂度越高（负向，用于惩罚）
    - 最终将得分映射到0-1范围

    注意: 仅用于测试流程，不是真正的深度学习预测
    """
    try:
        img = Image.open(image_path).convert('RGB')
        rgb = np.array(img, dtype=np.float64)

        h, w, _ = rgb.shape
        total_pixels = h * w

        # 1) 边缘强度：Laplacian方差（简单3x3核）
        gray = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]

        lap_kernel = np.array([[0, 1, 0],
                               [1, -4, 1],
                               [0, 1, 0]], dtype=np.float64)

        g = gray
        gp = np.pad(g, ((1, 1), (1, 1)), mode='edge')
        lap = (lap_kernel[0, 0] * gp[0:-2, 0:-2] + lap_kernel[0, 1] * gp[0:-2, 1:-1] + lap_kernel[0, 2] * gp[0:-2, 2:] +
               lap_kernel[1, 0] * gp[1:-1, 0:-2] + lap_kernel[1, 1] * gp[1:-1, 1:-1] + lap_kernel[1, 2] * gp[1:-1, 2:] +
               lap_kernel[2, 0] * gp[2:, 0:-2] + lap_kernel[2, 1] * gp[2:, 1:-1] + lap_kernel[2, 2] * gp[2:, 2:])

        edge_var = float(np.var(lap))

        # 2) 近似信息密度：随机子采样估算“可压缩性”（用灰度标准差代理压缩比惩罚）
        # 这里避免写临时JPEG文件（保持占位符轻量）
        gray_std = float(np.std(gray))

        # 3) 映射到0-1：结构清晰(+)与复杂噪声(-)
        # 经验性压缩：log尺度稳定
        edge_score = np.log1p(edge_var)  # 越大越清晰
        noise_penalty = np.log1p(gray_std)  # 越大越复杂

        raw = edge_score - 0.6 * noise_penalty

        # Sigmoid映射到0-1
        leg = 1.0 / (1.0 + np.exp(-raw))

        return {
            'success': True,
            'value': round(float(leg), 3),
            'method': 'placeholder_rule_based',
            'confidence': round(float(leg), 3),
            'edge_variance': round(edge_var, 3),
            'gray_std': round(gray_std, 3),
            'dimensions': {'height': int(h), 'width': int(w)},
            'total_pixels': int(total_pixels),
            'note': 'This is a placeholder estimation, not a deep learning prediction',
            'predicted_class': None,
            'topk': None
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
    完整实现：使用预训练深度学习分类模型输出softmax置信度

    需要:
    - PyTorch
    - 预训练分类模型文件（输出 logits, shape [1, num_classes]）
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

        model_type = model_config.get('model_type', 'ResNet50')
        num_classes = int(model_config.get('num_classes', 10))

        if model_type == 'ResNet50':
            model = models.resnet50(pretrained=False)
            model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

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

        # top-k
        k = min(3, num_classes)
        topk_probs, topk_idx = torch.topk(probs, k=k)
        topk_probs = topk_probs.cpu().numpy().tolist()
        topk_idx = topk_idx.cpu().numpy().tolist()

        label_map = model_config.get('label_map', None)
        if isinstance(label_map, (list, tuple)) and pred_idx < len(label_map):
            pred_label = label_map[pred_idx]
        elif isinstance(label_map, dict) and pred_idx in label_map:
            pred_label = label_map[pred_idx]
        else:
            pred_label = pred_idx

        topk = []
        for i in range(k):
            idx_i = int(topk_idx[i])
            prob_i = float(topk_probs[i])
            if isinstance(label_map, (list, tuple)) and idx_i < len(label_map):
                lab = label_map[idx_i]
            elif isinstance(label_map, dict) and idx_i in label_map:
                lab = label_map[idx_i]
            else:
                lab = idx_i
            topk.append({'class': lab, 'prob': round(prob_i, 4)})

        return {
            'success': True,
            'value': round(float(conf_value), 3),
            'method': 'deep_learning',
            'model_type': model_type,
            'device': str(device),
            'confidence': round(float(conf_value), 3),
            'predicted_class': pred_label,
            'topk': topk
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
def interpret_legibility(score: float) -> str:
    """解释可读性评分（0-1）"""
    if score < 0.2:
        return "Very low legibility: hard to identify/organize"
    elif score < 0.4:
        return "Low legibility"
    elif score < 0.6:
        return "Medium legibility"
    elif score < 0.8:
        return "High legibility"
    else:
        return "Very high legibility: easily identifiable and coherent"


# =============================================================================
# 测试代码
# =============================================================================
if __name__ == "__main__":
    print("\n🧪 Testing Legibility Index calculator...")

    # 简单测试：灰色块 vs 随机噪声（噪声可读性应更低）
    simple = np.full((120, 120, 3), 128, dtype=np.uint8)
    noisy = np.random.randint(0, 256, (120, 120, 3), dtype=np.uint8)

    for name, test_img in [('Simple', simple), ('Noisy', noisy)]:
        test_path = f'/tmp/test_leg_{name}.png'
        Image.fromarray(test_img).save(test_path)

        result = calculate_indicator(test_path)

        print(f"\n   {name}:")
        print(f"      Score: {result['value']} (0-1)")
        print(f"      Method: {result['method']}")
        if 'edge_variance' in result:
            print(f"      EdgeVar: {result['edge_variance']}, GrayStd: {result['gray_std']}")
        print(f"      Interpretation: {interpret_legibility(result['value'])}")

        os.remove(test_path)
