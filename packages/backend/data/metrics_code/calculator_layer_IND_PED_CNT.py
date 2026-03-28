"""
SceneRx Stage 2.5 - Calculator Layer
================================================
指标ID: IND_PED_CNT
指标名称: Pedestrian Count (行人数量)
类型: TYPE E (深度学习类)

说明:
使用实例分割模型（Mask R-CNN）检测图像中“people/pedestrian”实例数量，
输出行人/人的绝对计数（count）。

由于需要预训练模型，此示例提供两种实现：
1. 完整实现（需要PyTorch + torchvision，并使用预训练Mask R-CNN）
2. 占位符实现（用于测试流程：若输入为语义mask，则用person像素块近似计数）

⚠️ 注意:
- 完整实现需要安装 PyTorch: pip install torch torchvision
- 使用 torchvision 的预训练 Mask R-CNN（COCO）即可直接运行
- 需要原始RGB图像（不要用语义mask），否则检测效果会很差
"""

import numpy as np
from PIL import Image
from typing import Dict
import os


# =============================================================================
# 指标定义
# =============================================================================
INDICATOR = {
    "id": "IND_PED_CNT",
    "name": "Pedestrian Count",
    "unit": "count",
    "formula": "Count of instances identified as 'person' using Mask R-CNN",
    "target_direction": "NEUTRAL",
    "definition": "Absolute count of pedestrians/people detected in the scene using Mask R-CNN (He et al., 2017)",
    "category": "CAT_CMP",

    "calc_type": "deep_learning",

    # 模型配置
    "model_config": {
        "model_type": "MaskRCNN_ResNet50_FPN",
        "score_threshold": 0.5,    # 置信度阈值
        "nms_iou_threshold": 0.5,  # torchvision内部已做NMS，这里仅作为记录
        "max_detections": 100,
        "input_size": None,        # Mask R-CNN无需强制resize；可保持原始分辨率
        "person_class_id": 1       # COCO: person=1
    },

    # 占位符模式（无DL环境时使用）
    "use_placeholder": True
}

print(f"\n✅ Calculator ready: {INDICATOR['id']} - {INDICATOR['name']}")
print(f"   Mode: {'Placeholder (mask-based)' if INDICATOR.get('use_placeholder', True) else 'Deep Learning'}")


# =============================================================================
# 检测深度学习环境
# =============================================================================
TORCH_AVAILABLE = False
try:
    import torch
    import torchvision.transforms as transforms
    from torchvision.models.detection import maskrcnn_resnet50_fpn
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
    计算 Pedestrian Count (行人数量)

    TYPE E: 深度学习类

    根据配置选择实现方式:
    - 占位符模式: 若输入是语义分割mask，使用person像素连通域近似“实例数”
    - 完整模式: Mask R-CNN 实例分割检测 person 类别并计数

    Args:
        image_path: 图片路径（完整模式必须为原始RGB图像）

    Returns:
        {
            'success': True/False,
            'value': int (行人数),
            'method': str,
            'count': int,
            'confidence': dict/None,
            'detections': list (可选)
        }
    """
    use_placeholder = INDICATOR.get('use_placeholder', True)

    if use_placeholder or not TORCH_AVAILABLE:
        return calculate_placeholder(image_path)
    else:
        return calculate_deep_learning(image_path)


def calculate_placeholder(image_path: str) -> Dict:
    """
    占位符实现：基于语义mask的person连通域近似计数

    前提:
    - 输入为语义分割mask（RGB编码），且 semantic_colors 中存在 'person' 或 'person;individual;someone;somebody;...'
    - 用二值person像素掩膜的连通域数量近似“实例数”

    注意:
    - 这不是实例分割，连通域计数对遮挡/粘连很敏感，仅用于测试流程
    """
    try:
        img = Image.open(image_path).convert('RGB')
        pixels = np.array(img)
        h, w, _ = pixels.shape
        flat = pixels.reshape(-1, 3)

        # 尝试找person类
        person_keys = [
            "person",
            "person;individual;someone;somebody;...",
            "peoples",
            "pedestrian"
        ]

        person_rgb = None
        for k in person_keys:
            if 'semantic_colors' in globals() and k in semantic_colors:
                person_rgb = semantic_colors[k]
                break

        if person_rgb is None:
            return {
                'success': True,
                'value': 0,
                'count': 0,
                'method': 'placeholder_mask_based',
                'note': 'Person class not found in semantic_colors; placeholder returns 0'
            }

        mask = np.all(flat == person_rgb, axis=1).reshape(h, w).astype(np.uint8)

        # 连通域计数（8邻域）- 纯numpy BFS
        visited = np.zeros_like(mask, dtype=np.uint8)
        count = 0

        for y in range(h):
            for x in range(w):
                if mask[y, x] == 1 and visited[y, x] == 0:
                    count += 1
                    stack = [(y, x)]
                    visited[y, x] = 1
                    while stack:
                        cy, cx = stack.pop()
                        for dy in (-1, 0, 1):
                            for dx in (-1, 0, 1):
                                if dy == 0 and dx == 0:
                                    continue
                                ny, nx = cy + dy, cx + dx
                                if 0 <= ny < h and 0 <= nx < w:
                                    if mask[ny, nx] == 1 and visited[ny, nx] == 0:
                                        visited[ny, nx] = 1
                                        stack.append((ny, nx))

        return {
            'success': True,
            'value': int(count),
            'count': int(count),
            'method': 'placeholder_mask_based',
            'note': 'This is a placeholder estimation based on connected components, not Mask R-CNN'
        }

    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'value': None,
            'method': 'placeholder_mask_based'
        }


def calculate_deep_learning(image_path: str) -> Dict:
    """
    完整实现：使用 torchvision 预训练 Mask R-CNN (COCO) 检测 person 并计数

    输出:
    - count: 满足 score_threshold 的 person 检测实例数
    """
    try:
        cfg = INDICATOR.get('model_config', {})
        score_thr = float(cfg.get('score_threshold', 0.5))
        max_det = int(cfg.get('max_detections', 100))
        person_id = int(cfg.get('person_class_id', 1))

        # 加载模型（COCO预训练）
        model = maskrcnn_resnet50_fpn(pretrained=True)
        model.eval()

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)

        # 图像预处理
        img = Image.open(image_path).convert('RGB')
        img_tensor = transforms.ToTensor()(img).to(device)

        with torch.no_grad():
            outputs = model([img_tensor])[0]

        labels = outputs.get('labels', torch.tensor([])).detach().cpu().numpy().tolist()
        scores = outputs.get('scores', torch.tensor([])).detach().cpu().numpy().tolist()

        # 过滤 person & threshold
        detections = []
        for lab, sc in zip(labels, scores):
            if int(lab) == person_id and float(sc) >= score_thr:
                detections.append(float(sc))

        detections = sorted(detections, reverse=True)[:max_det]
        count = len(detections)

        return {
            'success': True,
            'value': int(count),
            'count': int(count),
            'method': 'deep_learning',
            'model_type': cfg.get('model_type', 'MaskRCNN_ResNet50_FPN'),
            'device': str(device),
            'score_threshold': score_thr,
            'confidence': {
                'mean_score': round(float(np.mean(detections)), 3) if count > 0 else 0,
                'max_score': round(float(np.max(detections)), 3) if count > 0 else 0
            },
            'detections': [{'score': round(float(s), 4)} for s in detections[:10]]  # 仅返回前10个得分
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
def interpret_ped_count(count: int) -> str:
    """解释行人数量"""
    if count == 0:
        return "No pedestrians detected"
    elif count <= 3:
        return "Few pedestrians"
    elif count <= 10:
        return "Moderate pedestrian presence"
    else:
        return "High pedestrian presence"


# =============================================================================
# 测试代码
# =============================================================================
if __name__ == "__main__":
    print("\n🧪 Testing Pedestrian Count calculator...")

    # 占位符模式无法在无person语义mask的情况下生成可靠测试
    # 这里用纯黑图进行流程测试（应返回0）
    test_img = np.zeros((128, 128, 3), dtype=np.uint8)
    test_path = "/tmp/test_ped_cnt.png"
    Image.fromarray(test_img).save(test_path)

    result = calculate_indicator(test_path)

    print(f"\n   Test: blank image")
    print(f"      Count: {result.get('value')}")
    print(f"      Method: {result.get('method')}")
    if result.get('success'):
        print(f"      Interpretation: {interpret_ped_count(int(result.get('value') or 0))}")

    os.remove(test_path)
