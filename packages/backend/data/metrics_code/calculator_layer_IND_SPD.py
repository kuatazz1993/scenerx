"""
SceneRx Stage 2.5 - Calculator Layer
================================================
指标ID: IND_SPD
指标名称: Spatial Division (空间分割度)
类型: TYPE D (组合类)

说明:
衡量活动空间中由分割性要素造成的空间破碎程度。
通过统计道路、路径、墙体、台阶、柱等“分割性元素”的面积占比并进行求和，
刻画原本连续、完整空间被切割的程度。

公式:
SPD = Sum(road + path + wall + stair + pillar)

单位: %
范围: 0 (空间高度连续) → 100 (空间高度破碎)
"""

import numpy as np
from PIL import Image
from typing import Dict


# =============================================================================
# 指标定义
# =============================================================================
INDICATOR = {
    "id": "IND_SPD",
    "name": "Spatial Division",
    "unit": "%",
    "formula": "Sum(road + path + wall + stair + pillar)",
    "target_direction": "NEUTRAL",
    "definition": "Proportion of space-dividing elements that fragment an otherwise coherent activity space",
    "category": "CAT_CFG",

    "calc_type": "composite",

    # 组成部分（按空间分割功能分组）
    "component_classes": {
        "roads": [
            "road"
        ],
        "paths": [
            "path"
        ],
        "walls": [
            "wall"
        ],
        "stairs": [
            "stair;stairs"
        ],
        "pillars": [
            "pillar;column"
        ]
    },

    # 聚合方式
    "aggregation": "sum"
}

print(f"\n✅ Calculator ready: {INDICATOR['id']} - {INDICATOR['name']}")
print(f"   Aggregation: {INDICATOR.get('aggregation', 'sum')}")


# =============================================================================
# 构建颜色查找表
# =============================================================================
COMPONENT_RGB = {}

print(f"\n🎯 Color lookup for spatial division components:")
for component_name, class_list in INDICATOR.get('component_classes', {}).items():
    COMPONENT_RGB[component_name] = {}
    print(f"\n   📦 {component_name}:")

    for class_name in class_list:
        if class_name in semantic_colors:
            rgb = semantic_colors[class_name]
            COMPONENT_RGB[component_name][rgb] = class_name
            print(f"      ✅ {class_name}: RGB{rgb}")
        else:
            print(f"      ⚠️ NOT FOUND: {class_name}")

print(f"\n✅ Components configured: {list(COMPONENT_RGB.keys())}")


# =============================================================================
# 计算函数
# =============================================================================
def calculate_indicator(image_path: str) -> Dict:
    """
    计算 Spatial Division (空间分割度)

    TYPE D: 组合类指标

    算法步骤:
    1. 分别统计各类分割性元素（road/path/wall/stair/pillar）的像素数
    2. 将所有分割性元素像素求和
    3. 计算其占总像素的比例（%）

    Args:
        image_path: 语义分割mask图片路径

    Returns:
        {
            'success': True/False,
            'value': float (SPD, %),
            'total_pixels': int,
            'division_pixels': int,
            'component_pixels': dict,
            'component_ratios': dict,
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

        component_counts = {}
        component_ratios = {}
        all_class_counts = {}

        # Step 2: 统计各组成部分
        for component_name, rgb_map in COMPONENT_RGB.items():
            component_total = 0

            for rgb, class_name in rgb_map.items():
                mask = np.all(flat_pixels == rgb, axis=1)
                count = np.sum(mask)
                if count > 0:
                    all_class_counts[class_name] = int(count)
                    component_total += count

            component_counts[component_name] = int(component_total)
            component_ratios[component_name] = round(
                (component_total / total_pixels) * 100, 3
            ) if total_pixels > 0 else 0

        # Step 3: 聚合
        division_pixels = sum(component_counts.values())
        value = (division_pixels / total_pixels) * 100 if total_pixels > 0 else 0

        division_level = interpret_spd(value)

        return {
            'success': True,
            'value': round(float(value), 3),
            'total_pixels': int(total_pixels),
            'division_pixels': int(division_pixels),
            'component_pixels': component_counts,
            'component_ratios': component_ratios,
            'class_breakdown': all_class_counts,
            'division_level': division_level
        }

    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'value': None
        }


# =============================================================================
# 辅助函数
# =============================================================================
def interpret_spd(value: float) -> str:
    """
    解释Spatial Division的含义
    """
    if value < 10:
        return "Very low division: highly continuous space"
    elif value < 25:
        return "Low division: mostly continuous space"
    elif value < 40:
        return "Moderate division: noticeable spatial segmentation"
    elif value < 60:
        return "High division: fragmented activity space"
    else:
        return "Very high division: highly fragmented space"


# =============================================================================
# 测试代码
# =============================================================================
if __name__ == "__main__":
    print("\n🧪 Testing Spatial Division calculator...")

    test_img = np.zeros((100, 100, 3), dtype=np.uint8)

    if 'road' in semantic_colors:
        test_img[0:30, :] = semantic_colors['road']     # 30% road
    if 'wall' in semantic_colors:
        test_img[30:45, :] = semantic_colors['wall']   # 15% wall
    if 'path' in semantic_colors:
        test_img[45:55, :] = semantic_colors['path']   # 10% path

    test_path = '/tmp/test_spd.png'
    Image.fromarray(test_img).save(test_path)

    result = calculate_indicator(test_path)

    print(f"\n   Test: 30% road + 15% wall + 10% path = 55% division")
    print(f"   Result: {result['value']}%")
    print(f"   Components: {result['component_ratios']}")
    print(f"   Level: {result['division_level']}")

    import os
    os.remove(test_path)
