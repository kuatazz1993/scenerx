"""
SceneRx Stage 2.5 - Calculator Layer
================================================
指标ID: IND_RID_RAT
指标名称: Rider Ratio (骑行者占比)
类型: TYPE A (ratio模式)

说明:
统计图像中代表“骑行者（person on bike / on vehicle）”的像素比例，
计算其占总像素的百分比，用于衡量街景中骑行者（如自行车/摩托车骑乘者）的可视暴露水平。

公式:
IND_RID_RAT = Pixels_Rider / Total_Pixels × 100
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
    "id": "IND_RID_RAT",
    "name": "Rider Ratio",
    "unit": "%",
    "formula": "(Sum(Rider_Pixels) / Sum(Total_Pixels)) × 100",
    "target_direction": "INCREASE",  # INCREASE / DECREASE / NEUTRAL
    "definition": "The proportion of pixels representing riders (persons on vehicles such as bicycles or motorcycles) in street view imagery.",
    "category": "CAT_CMP",

    # TYPE A 配置
    "calc_type": "ratio",  # ratio / inverse_ratio / two_class_ratio

    # 目标语义类别 - 【必须与 Excel 的 Name 列完全一致】
    "target_classes": [
        "person (on bike)",
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
    计算 Rider Ratio (IND_RID_RAT)

    TYPE A - ratio模式: 目标像素 / 总像素 × 100

    Args:
        image_path: 语义分割mask图片路径

    Returns:
        {
            'success': True/False,
            'value': float (骑行者占比百分比) or None,
            'target_pixels': int (骑行者像素总数),
            'total_pixels': int (总像素数),
            'class_breakdown': dict (各类别像素数)
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
            # 查找与该RGB完全匹配的像素
            mask = np.all(flat_pixels == rgb, axis=1)
            count = np.sum(mask)

            if count > 0:
                class_counts[class_name] = int(count)
                target_count += count

        # Step 3: 计算指标值 (ratio模式)
        value = (target_count / total_pixels) * 100 if total_pixels > 0 else 0

        # Step 4: 返回结果
        return {
            'success': True,
            'value': round(value, 3),
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
    # 如果直接运行此文件，进行简单测试
    print("\n🧪 Testing calculator...")

    # 创建测试图片
    test_img = np.zeros((100, 100, 3), dtype=np.uint8)

    # 填充一些骑行者像素 (person (on bike) 的颜色)
    if 'person (on bike)' in semantic_colors:
        rid_rgb = semantic_colors['person (on bike)']
        test_img[0:20, 0:100] = rid_rgb  # 20% rider

    # 保存测试图片
    test_path = '/tmp/test_rid_rat.png'
    Image.fromarray(test_img).save(test_path)

    # 测试计算
    result = calculate_indicator(test_path)
    print(f"   Result: {result}")

    # 清理
    import os
    os.remove(test_path)
