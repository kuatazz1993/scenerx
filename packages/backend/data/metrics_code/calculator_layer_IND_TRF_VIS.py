"""
SceneRx Stage 2.5 - Calculator Layer
================================================
指标ID: IND_TRF_VIS
指标名称: Visual Traffic Flow Index (视觉交通流指数)
类型: TYPE A (ratio模式 + 系数缩放)

说明:
统计图像中动态交通要素（车辆 car + 行人 pedestrian/person）的像素比例，
并乘以系数 0.25，用于近似表征视觉场中“交通流量”占用程度。

公式:
IND_TRF_VIS = 0.25 × (Car_pixels + Pedestrian_pixels) / Total_pixels × 100
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
    "id": "IND_TRF_VIS",
    "name": "Visual Traffic Flow Index",
    "unit": "%",  # 输出仍为百分比（与 GVI 等一致）
    "formula": "0.25 × (Sum(Car_pixels + Pedestrian_pixels) / Sum(Total_Pixels)) × 100",
    "target_direction": "INCREASE",  # INCREASE / DECREASE / NEUTRAL
    "definition": "The proportion of the visual field occupied by dynamic traffic elements (cars and pedestrians), scaled by 0.25.",
    "category": "CAT_CMP",

    # TYPE A 配置
    "calc_type": "ratio",

    # 系数缩放
    "scale": 0.25,

    # 目标语义类别 - 【必须与 Excel 的 Name 列完全一致】
    "target_classes": [
        "car",
        "person",  # pedestrian/person
    ]
}


# =============================================================================
# 构建颜色查找表 (从 input_layer.py 的 semantic_colors 获取)
# =============================================================================
TARGET_RGB = {}

print(f"\n🎯 Building color lookup for {INDICATOR['id']}:")
for class_name in INDICATOR.get('target_classes', []):
    if class_name in semantic_colors:
        rgb = semantic_colors[class_name]
        TARGET_RGB[rgb] = class_name
        print(f"   ✅ {class_name}: RGB{rgb}")
    else:
        print(f"   ⚠️ NOT FOUND: {class_name}")
        # 尝试部分匹配
        for name in semantic_colors.keys():
            if class_name.split(';')[0] in name or name.split(';')[0] in class_name:
                print(f"      💡 Did you mean: '{name}'?")
                break

print(f"\n✅ Calculator ready: {INDICATOR['id']} ({len(TARGET_RGB)} classes matched)")


# =============================================================================
# 计算函数
# =============================================================================
def calculate_indicator(image_path: str) -> Dict:
    """
    计算 Visual Traffic Flow Index (IND_TRF_VIS)

    TYPE A - ratio模式:
        base_ratio = target_pixels / total_pixels
        value = scale * base_ratio * 100

    Returns:
        {
            'success': True/False,
            'value': float (百分比, 已乘0.25缩放) or None,
            'target_pixels': int,
            'total_pixels': int,
            'class_breakdown': dict,
            'scale': float
        }
    """
    try:
        # Step 1: 加载图片
        img = Image.open(image_path).convert('RGB')
        pixels = np.array(img)
        h, w, _ = pixels.shape
        total_pixels = h * w

        flat_pixels = pixels.reshape(-1, 3)

        # Step 2: 统计目标类别像素
        target_count = 0
        class_counts = {}

        for rgb, class_name in TARGET_RGB.items():
            mask = np.all(flat_pixels == rgb, axis=1)
            count = int(np.sum(mask))

            if count > 0:
                class_counts[class_name] = count
                target_count += count

        # Step 3: 计算指标值
        scale = float(INDICATOR.get("scale", 1.0))
        base_ratio = (target_count / total_pixels) if total_pixels > 0 else 0.0
        value = scale * base_ratio * 100

        return {
            'success': True,
            'value': round(float(value), 3),
            'target_pixels': int(target_count),
            'total_pixels': int(total_pixels),
            'class_breakdown': class_counts,
            'scale': scale
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

    # 20% car + 20% person → base = 40% → scaled = 0.25*40 = 10%
    if 'car' in semantic_colors:
        test_img[0:20, :] = semantic_colors['car']

    if 'person' in semantic_colors:
        test_img[20:40, :] = semantic_colors['person']

    test_path = '/tmp/test_trf_vis.png'
    Image.fromarray(test_img).save(test_path)

    result = calculate_indicator(test_path)
    print(f"   Result: {result}")

    import os
    os.remove(test_path)
