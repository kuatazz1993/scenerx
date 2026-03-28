"""
SceneRx Stage 2.5 - Calculator Layer
================================================
指标ID: IND_WLK_IDX_MA
指标名称: Walkability Index (Ma et al.) / Walkability (Ma) (步行性指数-马等)
类型: TYPE B (two_class_ratio模式)

说明:
该指标用于表征“视觉步行性”，定义为：
行人设施要素（fence + sidewalk）与机动车道路要素（road）之间的像素比值。

公式:
IND_WLK_IDX_MA = (P_Fence + P_Sidewalk) / P_Road
其中 P_x 为要素 x 的像素比例（在同一图像中比值等价于像素计数比值）。
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
    "id": "IND_WLK_IDX_MA",
    "name": "Walkability Index (Ma et al.)",
    "unit": "ratio",
    "formula": "(Fence + Sidewalk) / Road",
    "target_direction": "INCREASE",  # INCREASE / DECREASE / NEUTRAL
    "definition": "A visual walkability index calculated as the ratio of pedestrian facility pixels (sidewalks + fences) to road pixels.",
    "category": "CAT_CMP",

    # TYPE B 配置（分子/分母类别）
    "calc_type": "two_class_ratio",  # ratio / inverse_ratio / two_class_ratio

    # 分子类别（行人设施）
    "numerator_classes": [
        "fence",
        "sidewalk",
    ],

    # 分母类别（道路）
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
    计算 Walkability Index (Ma et al.) (IND_WLK_IDX_MA)

    TYPE B - two_class_ratio模式:
        value = numerator_pixels / denominator_pixels

    注意:
    - 当 road_pixels = 0 时，分母为0，value 返回 None（不可计算）。

    Returns:
        {
            'success': True/False,
            'value': float or None,
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

        # Step 2: 统计分子像素（fence + sidewalk）
        numerator_count = 0
        numerator_counts = {}

        for rgb, class_name in NUM_RGB.items():
            mask = np.all(flat_pixels == rgb, axis=1)
            count = int(np.sum(mask))
            if count > 0:
                numerator_counts[class_name] = count
                numerator_count += count

        # Step 3: 统计分母像素（road）
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
            value = None
        else:
            value = numerator_count / denominator_count

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

    # 10% fence + 30% sidewalk, 20% road  → ratio = (40) / (20) = 2.0
    if 'fence' in semantic_colors:
        test_img[0:10, :] = semantic_colors['fence']

    if 'sidewalk' in semantic_colors:
        test_img[10:40, :] = semantic_colors['sidewalk']

    if 'road' in semantic_colors:
        test_img[40:60, :] = semantic_colors['road']

    test_path = '/tmp/test_wlk_idx_ma.png'
    Image.fromarray(test_img).save(test_path)

    result = calculate_indicator(test_path)
    print(f"   Result: {result}")

    import os
    os.remove(test_path)
