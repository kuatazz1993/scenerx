"""
SceneRx Stage 2.5 - Calculator Layer
================================================
指标ID: IND_WLD
指标名称: Wildness (野性指数)
类型: TYPE C (arctan-ratio 模式)

说明:
该指标用于刻画“城市野性（urban wildness）”，
通过比较自然生长型植被（flora，如灌木/野生植物）
相对于人工修整型自然要素（grass）及非自然要素
（如道路、建筑、围栏、车辆、行人等）的优势程度。

指标采用反正切函数（arctan）进行压缩，
避免极端比值对结果造成过度放大。

公式:
IND_WLD = arctan( Flora_pixels / (Grass_pixels + Non_natural_pixels) )

其中:
- Flora_pixels: 野生/自然生长植被像素
- Grass_pixels: 草地像素
- Non_natural_pixels: 人工/非自然要素像素
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
    "id": "IND_WLD",
    "name": "Wildness",
    "unit": "radian",  # arctan 输出为弧度
    "formula": "arctan(Flora / (Grass + Non_natural))",
    "target_direction": "INCREASE",
    "definition": (
        "An arctangent-transformed ratio of flora pixels to grass and "
        "non-natural elements, indicating the degree of urban wildness "
        "versus generic or artificial nature."
    ),
    "category": "CAT_CMP",

    # 计算类型（自定义）
    "calc_type": "atan_ratio",

    # 分子：野生 / 自然生长植被
    "numerator_classes": [
        "plant;flora;plant;life",
        "bush",
        "shrub",
    ],

    # 分母组成之一：人工修整型自然
    "grass_classes": [
        "grass",
    ],

    # 分母组成之二：非自然要素（人工环境）
    "non_natural_classes": [
        "road",
        "sidewalk",
        "building",
        "wall",
        "fence",
        "car",
        "person",
    ]
}


# =============================================================================
# 构建颜色查找表
# =============================================================================
NUM_RGB = {}
GRASS_RGB = {}
NONNAT_RGB = {}

print(f"\n🎯 Building color lookup for {INDICATOR['id']}:")

print("   ▶ Flora (numerator) classes:")
for class_name in INDICATOR["numerator_classes"]:
    if class_name in semantic_colors:
        NUM_RGB[semantic_colors[class_name]] = class_name
        print(f"     ✅ {class_name}")
    else:
        print(f"     ⚠️ NOT FOUND: {class_name}")

print("   ▶ Grass classes:")
for class_name in INDICATOR["grass_classes"]:
    if class_name in semantic_colors:
        GRASS_RGB[semantic_colors[class_name]] = class_name
        print(f"     ✅ {class_name}")
    else:
        print(f"     ⚠️ NOT FOUND: {class_name}")

print("   ▶ Non-natural classes:")
for class_name in INDICATOR["non_natural_classes"]:
    if class_name in semantic_colors:
        NONNAT_RGB[semantic_colors[class_name]] = class_name
        print(f"     ✅ {class_name}")
    else:
        print(f"     ⚠️ NOT FOUND: {class_name}")

print(
    f"\n✅ Calculator ready: {INDICATOR['id']} "
    f"(flora={len(NUM_RGB)}, grass={len(GRASS_RGB)}, non-natural={len(NONNAT_RGB)})"
)


# =============================================================================
# 计算函数
# =============================================================================
def calculate_indicator(image_path: str) -> Dict:
    """
    计算 Wildness Index (IND_WLD)

    逻辑:
        ratio = flora_pixels / (grass_pixels + non_natural_pixels)
        value = arctan(ratio)

    Returns:
        {
            'success': True/False,
            'value': float (弧度),
            'flora_pixels': int,
            'grass_pixels': int,
            'non_natural_pixels': int,
            'class_breakdown': dict
        }
    """
    try:
        img = Image.open(image_path).convert("RGB")
        pixels = np.array(img)
        h, w, _ = pixels.shape
        flat_pixels = pixels.reshape(-1, 3)

        flora_count = 0
        grass_count = 0
        nonnat_count = 0

        class_counts = {}

        # Flora
        for rgb, name in NUM_RGB.items():
            m = np.all(flat_pixels == rgb, axis=1)
            c = int(np.sum(m))
            if c > 0:
                flora_count += c
                class_counts[name] = c

        # Grass
        for rgb, name in GRASS_RGB.items():
            m = np.all(flat_pixels == rgb, axis=1)
            c = int(np.sum(m))
            if c > 0:
                grass_count += c
                class_counts[name] = c

        # Non-natural
        for rgb, name in NONNAT_RGB.items():
            m = np.all(flat_pixels == rgb, axis=1)
            c = int(np.sum(m))
            if c > 0:
                nonnat_count += c
                class_counts[name] = c

        denom = grass_count + nonnat_count
        ratio = flora_count / denom if denom > 0 else 0.0
        value = float(np.arctan(ratio))

        return {
            "success": True,
            "value": round(value, 6),
            "flora_pixels": flora_count,
            "grass_pixels": grass_count,
            "non_natural_pixels": nonnat_count,
            "class_breakdown": class_counts
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "value": None
        }


# =============================================================================
# 测试代码 (可选)
# =============================================================================
if __name__ == "__main__":
    print("\n🧪 Testing calculator...")

    test_img = np.zeros((100, 100, 3), dtype=np.uint8)

    # 20% flora, 20% grass, 40% non-natural
    if 'plant;flora;plant;life' in semantic_colors:
        test_img[0:20, :] = semantic_colors['plant;flora;plant;life']

    if 'grass' in semantic_colors:
        test_img[20:40, :] = semantic_colors['grass']

    if 'road' in semantic_colors:
        test_img[40:80, :] = semantic_colors['road']

    test_path = '/tmp/test_wld.png'
    Image.fromarray(test_img).save(test_path)

    result = calculate_indicator(test_path)
    print(f"   Result: {result}")

    import os
    os.remove(test_path)
