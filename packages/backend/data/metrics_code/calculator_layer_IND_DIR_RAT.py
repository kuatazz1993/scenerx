"""
SceneRx Stage 2.5 - Calculator Layer
================================================
指标ID: IND_DIR_RAT
指标名称: Dirt Ratio (泥土占比)
类型: TYPE A (ratio模式)

说明:
统计图像中代表“泥土/裸地（dirt or bare ground）”的像素比例，
包括 earth 与 dirt 两类，计算其占总像素的百分比。

公式:
IND_DIR_RAT = Pixels_Dirt / Total_Pixels × 100
其中 Pixels_Dirt = Sum(earth + dirt pixels)
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
    "id": "IND_DIR_RAT",
    "name": "Dirt Ratio",
    "unit": "%",
    "formula": "(Sum(Dirt_Pixels) / Sum(Total_Pixels)) × 100",
    "target_direction": "INCREASE",  # INCREASE / DECREASE / NEUTRAL
    "definition": "The proportion of the visual field occupied by dirt or bare ground.",
    "category": "CAT_CMP",

    # TYPE A 配置
    "calc_type": "ratio",

    # 目标语义类别 - 【必须与 Excel 的 Name 列完全一致】
    "target_classes": [
        "earth",
        "dirt",
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
    计算 Dirt Ratio (IND_DIR_RAT)

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

    # 25% earth + 15% dirt → 40%
    if 'earth' in semantic_colors:
        test_img[0:25, :] = semantic_colors['earth']

    if 'dirt' in semantic_colors:
        test_img[25:40, :] = semantic_colors['dirt']

    test_path = '/tmp/test_dir_rat.png'
    Image.fromarray(test_img).save(test_path)

    result = calculate_indicator(test_path)
    print(f"   Result: {result}")

    import os
    os.remove(test_path)
