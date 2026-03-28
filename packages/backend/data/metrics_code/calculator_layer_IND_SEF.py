"""
SceneRx Stage 2.5 - Calculator Layer
================================================
指标ID: IND_SEF
指标名称: Service Facility (服务设施占比)
类型: TYPE A (ratio模式)

说明:
统计图像中“服务设施/装饰设施”相关要素的像素比例，
包括座椅、路灯、标识、雨棚/遮阳棚、花盆、雕塑等，
计算其占总像素的百分比，用于衡量街景中的服务设施可视暴露程度。

公式:
IND_SEF = (bench + streetlight + signboard + rain_canopy + flowerpot + sculpture) / Total Pixels × 100
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
    "id": "IND_SEF",
    "name": "Service Facility",
    "unit": "%",
    "formula": "(Sum(Service_Facility_Pixels) / Sum(Total_Pixels)) × 100",
    "target_direction": "INCREASE",  # INCREASE / DECREASE / NEUTRAL
    "definition": "Sum of the area proportions of service facilities for use and decoration in street view imagery.",
    "category": "CAT_CMP",

    # TYPE A 配置
    "calc_type": "ratio",  # ratio / inverse_ratio / two_class_ratio

    # 目标语义类别 - 【必须与 Excel 的 Name 列完全一致】
    "target_classes": [
        "bench",
        "streetlight",
        "signboard",
        "rain_canopy",
        "flowerpot",
        "sculpture",
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
    计算 Service Facility (IND_SEF)

    TYPE A - ratio模式: 目标像素 / 总像素 × 100

    Returns:
        {
            'success': True/False,
            'value': float (百分比) or None,
            'target_pixels': int,
            'total_pixels': int,
            'class_breakdown': dict
        }
    """
    try:
        # Step 1: 加载图片
        img = Image.open(image_path).convert('RGB')
        pixels = np.array(img)
        h, w, _ = pixels.shape
        total_pixels = h * w

        # 展平像素数组以便快速比较
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
        value = (target_count / total_pixels) * 100 if total_pixels > 0 else 0.0

        return {
            'success': True,
            'value': round(float(value), 3),
            'target_pixels': int(target_count),
            'total_pixels': int(total_pixels),
            'class_breakdown': class_counts
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

    # 10% bench + 10% streetlight + 10% signboard = 30%（如果这些类在 semantic_colors 中存在）
    if 'bench' in semantic_colors:
        test_img[0:10, :] = semantic_colors['bench']

    if 'streetlight' in semantic_colors:
        test_img[10:20, :] = semantic_colors['streetlight']

    if 'signboard' in semantic_colors:
        test_img[20:30, :] = semantic_colors['signboard']

    test_path = '/tmp/test_sef.png'
    Image.fromarray(test_img).save(test_path)

    result = calculate_indicator(test_path)
    print(f"   Result: {result}")

    import os
    os.remove(test_path)
