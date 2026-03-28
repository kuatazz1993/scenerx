"""
SceneRx Stage 2.5 - Calculator Layer
================================================
指标ID: IND_VPI
指标名称: Visual Pavement Index (视觉铺装指数)
类型: TYPE A (ratio模式 / two_class_ratio变体)

说明:
计算可见铺装像素（sidewalk）在（铺装 + 道路）中的占比，
用于衡量行人空间相对于机动车空间的主导程度。

公式:
IND_VPI = Pn / (Pn + Rn) × 100
其中:
Pn = Pavement pixels (sidewalk)
Rn = Road pixels (road)
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
    "id": "IND_VPI",
    "name": "Visual Pavement Index",
    "unit": "%",
    "formula": "(Pn / (Pn + Rn)) × 100",
    "target_direction": "INCREASE",  # INCREASE / DECREASE / NEUTRAL
    "definition": "The ratio of visible pavement pixels to the sum of pavement and road pixels, indicating the dominance of pedestrian space relative to vehicle space.",
    "category": "CAT_CMP",

    # 计算类型（虽然不是传统 target/total，但仍是百分比型占比）
    "calc_type": "two_class_ratio",  # 复用该类型的“二类占比”思路

    # 分子（铺装）
    "numerator_classes": [
        "sidewalk",
    ],

    # 分母补集（道路）——用于构造 Pn + Rn
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
    计算 Visual Pavement Index (IND_VPI)

    逻辑:
        Pn = sidewalk pixels
        Rn = road pixels
        IND_VPI = Pn / (Pn + Rn) × 100

    Returns:
        {
            'success': True/False,
            'value': float (百分比) or None,
            'pavement_pixels': int,
            'road_pixels': int,
            'total_pnr_pixels': int,
            'pavement_breakdown': dict,
            'road_breakdown': dict
        }
    """
    try:
        # Step 1: 加载图片
        img = Image.open(image_path).convert('RGB')
        pixels = np.array(img)
        h, w, _ = pixels.shape

        flat_pixels = pixels.reshape(-1, 3)

        # Step 2: 统计铺装像素（Pn）
        pavement_count = 0
        pavement_counts = {}

        for rgb, class_name in NUM_RGB.items():
            mask = np.all(flat_pixels == rgb, axis=1)
            count = int(np.sum(mask))
            if count > 0:
                pavement_counts[class_name] = count
                pavement_count += count

        # Step 3: 统计道路像素（Rn）
        road_count = 0
        road_counts = {}

        for rgb, class_name in DEN_RGB.items():
            mask = np.all(flat_pixels == rgb, axis=1)
            count = int(np.sum(mask))
            if count > 0:
                road_counts[class_name] = count
                road_count += count

        # Step 4: 计算指标值
        total_pnr = pavement_count + road_count
        value = (pavement_count / total_pnr) * 100 if total_pnr > 0 else 0.0

        return {
            'success': True,
            'value': round(float(value), 3),
            'pavement_pixels': int(pavement_count),
            'road_pixels': int(road_count),
            'total_pnr_pixels': int(total_pnr),
            'pavement_breakdown': pavement_counts,
            'road_breakdown': road_counts
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

    # 40% sidewalk, 20% road
    if 'sidewalk' in semantic_colors:
        test_img[0:40, :] = semantic_colors['sidewalk']

    if 'road' in semantic_colors:
        test_img[40:60, :] = semantic_colors['road']

    test_path = '/tmp/test_vpi.png'
    Image.fromarray(test_img).save(test_path)

    result = calculate_indicator(test_path)
    print(f"   Result: {result}")

    import os
    os.remove(test_path)
