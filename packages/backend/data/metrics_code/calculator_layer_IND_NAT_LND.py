"""
SceneRx Stage 2.5 - Calculator Layer
================================================
指标ID: IND_NAT_LND
指标名称: Natural Landscape Index (自然景观指数)
类型: TYPE A (ratio模式)

说明:
统计图像中代表“大尺度自然景观”的像素比例，
包括海洋（sea）、山体（mountain）和树木（tree），
计算其占总像素的百分比，用于衡量场景中连续自然景观要素的可视暴露水平。

公式:
IND_NAT_LND = (Sea + Mountain + Tree) / Total Pixels × 100
"""

import numpy as np
from PIL import Image
from typing import Dict

# semantic_colors 来自 input_layer.py（与其他指标文件保持一致）
from input_layer import semantic_colors


# =============================================================================
# 指标定义 - 【仅此处为核心配置】
# =============================================================================
INDICATOR = {
    # 基本信息
    "id": "IND_NAT_LND",
    "name": "Natural Landscape Index",
    "unit": "%",
    "formula": "(Sum(Sea + Mountain + Tree Pixels) / Sum(Total_Pixels)) × 100",
    "target_direction": "INCREASE",  # INCREASE / DECREASE / NEUTRAL
    "definition": "Proportion of pixels representing large natural landscape features such as sea, mountain, and tree.",
    "category": "CAT_CMP",

    # TYPE A 配置
    "calc_type": "ratio",  # ratio / inverse_ratio / two_class_ratio

    # 目标语义类别 - 【必须与 Excel 的 Name 列完全一致】
    "target_classes": [
        "sea",               # 海洋
        "mountain;mount",    # 山体
        "tree",              # 树木
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
        # 尝试部分匹配（与 GVI 文件完全一致）
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
    计算 Natural Landscape Index (自然景观指数)

    TYPE A - ratio模式: 目标像素 / 总像素 × 100
    """
    try:
        # Step 1: 加载图片
        img = Image.open(image_path).convert('RGB')
        pixels = np.array(img)
        h, w, _ = pixels.shape
        total_pixels = h * w

        # 展平像素数组
        flat_pixels = pixels.reshape(-1, 3)

        # Step 2: 统计目标类别像素
        target_count = 0
        class_counts = {}

        for rgb, class_name in TARGET_RGB.items():
            mask = np.all(flat_pixels == rgb, axis=1)
            count = np.sum(mask)

            if count > 0:
                class_counts[class_name] = int(count)
                target_count += count

        # Step 3: 计算指标值
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
    print("\n🧪 Testing calculator...")

    test_img = np.zeros((100, 100, 3), dtype=np.uint8)

    if 'sea' in semantic_colors:
        test_img[0:30, :] = semantic_colors['sea']      # 30%

    if 'mountain;mount' in semantic_colors:
        test_img[30:60, :] = semantic_colors['mountain;mount']  # 30%

    if 'tree' in semantic_colors:
        test_img[60:80, :] = semantic_colors['tree']    # 20%

    test_path = '/tmp/test_nat_lnd.png'
    Image.fromarray(test_img).save(test_path)

    result = calculate_indicator(test_path)
    print(f"   Result: {result}")

    import os
    os.remove(test_path)
