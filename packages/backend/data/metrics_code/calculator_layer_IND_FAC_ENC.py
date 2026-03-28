"""
SceneRx Stage 2.5 - Calculator Layer
================================================
指标ID: IND_FAC_ENC
指标名称: Facade Enclosure Type (立面围合类型)
类型: TYPE D (组合类)

说明:
用于区分立面围合类型：封闭/不透水墙体（Sealed） vs 开放/栏杆式立面（Open/Railing）。
基于语义分割结果统计墙体类与栏杆/围栏类的像素占比，并进行类别判定。

分类规则:
- 若 sealed_ratio > open_ratio → Sealed
- 若 open_ratio > sealed_ratio → Open/Railing
- 若两者均为0或相等 → Mixed/Unknown

类别: Sealed / Open-Railing / Mixed-Unknown
"""

import numpy as np
from PIL import Image
from typing import Dict


# =============================================================================
# 指标定义
# =============================================================================
INDICATOR = {
    "id": "IND_FAC_ENC",
    "name": "Facade Enclosure Type",
    "unit": "category",
    "formula": "Category: Sealed vs Open/Railing",
    "target_direction": "NEUTRAL",
    "definition": "Categorical indicator distinguishing sealed/impervious walls vs open/railing facades",
    "category": "CAT_CFG",

    "calc_type": "composite",

    # 组成部分（按功能分组）
    "component_classes": {
        "sealed": [
            "wall"
        ],
        "open_railing": [
            "railing;rail",
            "fence;fencing"
        ]
    },

    # 聚合方式
    "aggregation": "categorical_compare",

    # 判定阈值（可选）
    "min_ratio_threshold": 0.5  # 低于该占比可认为不显著（%）
}

print(f"\n✅ Calculator ready: {INDICATOR['id']} - {INDICATOR['name']}")
print(f"   Aggregation: {INDICATOR.get('aggregation', 'categorical_compare')}")


# =============================================================================
# 构建颜色查找表
# =============================================================================
COMPONENT_RGB = {}

print(f"\n🎯 Facade enclosure color lookup:")
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
    计算 Facade Enclosure Type (立面围合类型)

    TYPE D: 组合类指标

    算法步骤:
    1. 分别统计 sealed（wall）与 open_railing（railing/fence）的像素数
    2. 转换为占总像素的比例（%）
    3. 基于比例大小进行类别判定

    Args:
        image_path: 语义分割mask图片路径

    Returns:
        {
            'success': True/False,
            'value': str (Category),
            'category_code': int,
            'sealed_ratio': float (%),
            'open_ratio': float (%),
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

        sealed_ratio = float(component_ratios.get("sealed", 0))
        open_ratio = float(component_ratios.get("open_railing", 0))

        category, code = classify_facade_enclosure(sealed_ratio, open_ratio)

        return {
            'success': True,
            'value': category,
            'category_code': int(code),
            'sealed_ratio': round(sealed_ratio, 3),
            'open_ratio': round(open_ratio, 3),
            'total_pixels': int(total_pixels),
            'component_pixels': component_counts,
            'component_ratios': component_ratios,
            'class_breakdown': all_class_counts
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
def classify_facade_enclosure(sealed_ratio: float, open_ratio: float) -> tuple:
    """
    立面围合类型判定

    Returns:
        (category_name, category_code)
        code: 1=Sealed, 2=Open/Railing, 0=Mixed/Unknown
    """
    thr = float(INDICATOR.get("min_ratio_threshold", 0.0))

    sealed_sig = sealed_ratio >= thr
    open_sig = open_ratio >= thr

    if (not sealed_sig) and (not open_sig):
        return ("Mixed/Unknown", 0)

    if sealed_ratio > open_ratio:
        return ("Sealed", 1)
    elif open_ratio > sealed_ratio:
        return ("Open/Railing", 2)
    else:
        return ("Mixed/Unknown", 0)


# =============================================================================
# 测试代码
# =============================================================================
if __name__ == "__main__":
    print("\n🧪 Testing Facade Enclosure Type calculator...")

    test_img = np.zeros((100, 100, 3), dtype=np.uint8)

    # Case 1: Sealed（墙为主）
    if 'wall' in semantic_colors:
        test_img[0:60, :] = semantic_colors['wall']  # 60% wall

    test_path = '/tmp/test_fac_enc.png'
    Image.fromarray(test_img).save(test_path)

    result = calculate_indicator(test_path)

    print(f"\n   Test: wall-dominant facade")
    print(f"   Category: {result['value']}")
    print(f"   Sealed ratio: {result['sealed_ratio']}%")
    print(f"   Open ratio: {result['open_ratio']}%")

    import os
    os.remove(test_path)
