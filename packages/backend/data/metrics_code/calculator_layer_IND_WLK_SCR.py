"""
SceneRx Stage 2.5 - Calculator Layer
================================================
指标ID: IND_WLK_SCR
指标名称: Walkability Score (Tao et al.) / Walkability Ratio (步行性得分)
类型: TYPE B (two_class_ratio模式)

说明:
该指标用于表征“视觉步行性”，定义为：
行人要素（sidewalk, fence）与机动车道路要素（road）之间的比值。

注意：
- 这是“比值指标”，不同于常规百分比指标（不乘×100）。
- 当 Road 像素为 0 时，分母为 0，指标不可计算，返回 value=None。

公式:
IND_WLK_SCR = (Sidewalk + Fence) / Road
"""

import numpy as np
from PIL import Image
from typing import Dict

# semantic_colors 来自 input_layer.py（与其他指标文件保持一致）
from input_layer import semantic_colors


# =============================================================================
# 指标定义 - 【核心配置】
# =============================================================================
INDICATOR = {
    # 基本信息
    "id": "IND_WLK_SCR",
    "name": "Walkability Score (Tao et al.)",
    "unit": "ratio",  # ratio / %
    "formula": "(Sidewalk + Fence) / Road",
    "target_direction": "INCREASE",  # INCREASE / DECREASE / NEUTRAL
    "definition": "A visual perception score for walkability, defined as the ratio of pedestrian elements (sidewalk, fence) to vehicle road elements (road).",
    "category": "CAT_CFG",

    # TYPE B 配置（分子/分母类别）
    "calc_type": "two_class_ratio",  # ratio / inverse_ratio / two_class_ratio

    # 分子类别（行人要素）
    "numerator_classes": [
        "sidewalk",
        "fence",
    ],

    # 分母类别（道路要素）
    "denominator_classes": [
        "road",
    ]
}


# =============================================================================
# 构建颜色查找表
# =============================================================================
NUM_RGB = {}
DEN_RGB = {}

print(f"\n🎯 Building color lookup for {INDICATOR['id']}:")

print("   ▶ Numerator classes:")
for class_name in INDICATOR.get('numerator_classes', []):
    if class_name in semantic_colors:
        rgb = semantic_colors[class_name]
        NUM_RGB[rgb] = class_name
        print(f"     ✅ {class_name}: RGB{rgb}")
    else:
        print(f"     ⚠️ NOT FOUND: {class_name}")

print("   ▶ Denominator classes:")
for class_name in INDICATOR.get('denominator_classes', []):
    if class_name in semantic_colors:
        rgb = semantic_colors[class_name]
        DEN_RGB[rgb] = class_name
        print(f"     ✅ {class_name}: RGB{rgb}")
    else:
        print(f"     ⚠️ NOT FOUND: {class_name}")

print(
    f"\n✅ Calculator ready: {INDICATOR['id']} "
    f"(NUM={len(NUM_RGB)} classes matched, DEN={len(DEN_RGB)} classes matched)"
)


# =============================================================================
# 计算函数
# =============================================================================
def calculate_indicator(image_path: str) -> Dict:
    """
    计算 Walkability Score (IND_WLK_SCR)

    TYPE B - two_class_ratio模式:
        value = (numerator_pixels / total_pixels) / (denominator_pixels / total_pixels)
              = numerator_pixels / denominator_pixels

    Args:
        image_path: 语义分割mask图片路径

    Returns:
        {
            'success': True/False,
            'value': float or None (步行性比值),
            'numerator_pixels': int,
            'denominator_pixels': int,
            'total_pixels': int,
            'numerator_breakdown': dict,
            'denominator_breakdown': dict
        }
    """
    try:
        # Step 1: 加载图片
        img = Image.open(image_path).convert('RGB')
        pixels = np.array(img)
        h, w, _ = pixels.shape
        total_pixels = h * w

        flat_pixels = pixels.reshape(-1, 3)

        # Step 2: 统计分子像素
        numerator_count = 0
        numerator_counts = {}

        for rgb, class_name in NUM_RGB.items():
            mask = np.all(flat_pixels == rgb, axis=1)
            count = int(np.sum(mask))
            if count > 0:
                numerator_counts[class_name] = count
                numerator_count += count

        # Step 3: 统计分母像素（Road）
        denominator_count = 0
        denominator_counts = {}

        for rgb, class_name in DEN_RGB.items():
            mask = np.all(flat_pixels == rgb, axis=1)
            count = int(np.sum(mask))
            if count > 0:
                denominator_counts[class_name] = count
                denominator_count += count

        # Step 4: 计算比值
        if denominator_count == 0:
            value = None  # 分母为0不可计算
        else:
            value = numerator_count / denominator_count

        # Step 5: 返回结果
        return {
            'success': True,
            'value': None if value is None else round(float(value), 6),
            'numerator_pixels': int(numerator_count),
            'denominator_pixels': int(denominator_count),
            'total_pixels': int(total_pixels),
            'numerator_breakdown': numerator_counts,
            'denominator_breakdown': denominator_counts
        }

    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'value': None
        }


# =============================================================================
# 测试代码 (可选)
# =============================================================================
if __name__ == "__main__":
    print("\n🧪 Testing calculator...")

    test_img = np.zeros((100, 100, 3), dtype=np.uint8)

    # 30% sidewalk + 10% fence, 20% road
    if 'sidewalk' in semantic_colors:
        test_img[0:30, :] = semantic_colors['sidewalk']

    if 'fence' in semantic_colors:
        test_img[30:40, :] = semantic_colors['fence']

    if 'road' in semantic_colors:
        test_img[40:60, :] = semantic_colors['road']

    test_path = '/tmp/test_wlk_scr.png'
    Image.fromarray(test_img).save(test_path)

    result = calculate_indicator(test_path)
    print(f"   Result: {result}")

    import os
    os.remove(test_path)
