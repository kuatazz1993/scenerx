"""
SceneRx Stage 2.5 - Calculator Layer
================================================
指标ID: IND_WLK_IDX
指标名称: Walkability Index (Visual) / Walkability Index (步行性指数)
类型: TYPE A (ratio模式 / two_class_share)

说明:
计算街景图像中“人行道（sidewalk）”像素在（人行道 + 道路（road/driveway））中的占比，
用于衡量行人空间相对于机动车道路空间的主导程度。

公式:
IND_WLK_IDX = Area_sidewalk / (Area_sidewalk + Area_driveway) × 100
其中:
Area_sidewalk = sidewalk pixels
Area_driveway = road pixels
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
    "id": "IND_WLK_IDX",
    "name": "Walkability Index (Visual)",
    "unit": "%",
    "formula": "(Area_sidewalk / (Area_sidewalk + Area_driveway)) × 100",
    "target_direction": "INCREASE",  # INCREASE / DECREASE / NEUTRAL
    "definition": "The ratio of sidewalk pixels to the sum of sidewalk and driveway pixels in a street view image.",
    "category": "CAT_CMP",

    # 计算类型（占比类，但分母不是总像素）
    "calc_type": "two_class_ratio",

    # 分子：sidewalk
    "numerator_classes": [
        "sidewalk",
    ],

    # 分母补集：driveway（此处按你的类别定义使用 road）
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
    计算 Walkability Index (IND_WLK_IDX)

    逻辑:
        S = sidewalk pixels
        D = road (driveway) pixels
        IND_WLK_IDX = S / (S + D) × 100

    Returns:
        {
            'success': True/False,
            'value': float (百分比) or None,
            'sidewalk_pixels': int,
            'driveway_pixels': int,
            'total_sd_pixels': int,
            'sidewalk_breakdown': dict,
            'driveway_breakdown': dict
        }
    """
    try:
        # Step 1: 加载图片
        img = Image.open(image_path).convert('RGB')
        pixels = np.array(img)
        h, w, _ = pixels.shape

        flat_pixels = pixels.reshape(-1, 3)

        # Step 2: 统计 sidewalk 像素（S）
        sidewalk_count = 0
        sidewalk_counts = {}

        for rgb, class_name in NUM_RGB.items():
            mask = np.all(flat_pixels == rgb, axis=1)
            count = int(np.sum(mask))
            if count > 0:
                sidewalk_counts[class_name] = count
                sidewalk_count += count

        # Step 3: 统计 road/driveway 像素（D）
        driveway_count = 0
        driveway_counts = {}

        for rgb, class_name in DEN_RGB.items():
            mask = np.all(flat_pixels == rgb, axis=1)
            count = int(np.sum(mask))
            if count > 0:
                driveway_counts[class_name] = count
                driveway_count += count

        # Step 4: 计算指标值
        total_sd = sidewalk_count + driveway_count
        value = (sidewalk_count / total_sd) * 100 if total_sd > 0 else 0.0

        return {
            'success': True,
            'value': round(float(value), 3),
            'sidewalk_pixels': int(sidewalk_count),
            'driveway_pixels': int(driveway_count),
            'total_sd_pixels': int(total_sd),
            'sidewalk_breakdown': sidewalk_counts,
            'driveway_breakdown': driveway_counts
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

    # 40% sidewalk, 20% road -> 40/(40+20)=66.666%
    if 'sidewalk' in semantic_colors:
        test_img[0:40, :] = semantic_colors['sidewalk']

    if 'road' in semantic_colors:
        test_img[40:60, :] = semantic_colors['road']

    test_path = '/tmp/test_wlk_idx.png'
    Image.fromarray(test_img).save(test_path)

    result = calculate_indicator(test_path)
    print(f"   Result: {result}")

    import os
    os.remove(test_path)
