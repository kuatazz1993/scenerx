"""
SceneRx Stage 2.5 - Calculator Layer
================================================
指标ID: IND_SFI
指标名称: Spatial Feasibility Index (行人空间可行性指数)
类型: TYPE D (组合类)

说明:
衡量街景中行人空间优先程度。
通过人行道像素与车行道（car lane）像素的比值来表征：
比值越高表示行人空间相对更充足、对机动车道依赖更低。

公式: SFI = Wn / Rn
其中:
- Wn: Sidewalk pixel count
- Rn: Car lane pixel count
"""

import numpy as np
from PIL import Image
from typing import Dict


# =============================================================================
# 指标定义
# =============================================================================
INDICATOR = {
    "id": "IND_SFI",
    "name": "Spatial Feasibility Index",
    "unit": "ratio",
    "formula": "SFI = Wn / Rn",
    "target_direction": "POSITIVE",
    "definition": "Ratio of sidewalk pixels to car lane pixels indicating pedestrian space priority",
    "category": "CAT_CFG",

    "calc_type": "composite",

    # 组成部分（按功能分组）
    "component_classes": {
        "Wn_sidewalk": [
            "sidewalk",           # 人行道
            "sidewalk;curb",      # 若存在复合标签（可选）
        ],
        "Rn_car_lane": [
            "road",               # 车行道（如无car lane细分，默认使用road）
            "lane",               # 车道（若存在）
            "car lane",           # 车行道（若存在）
            "driveway",           # 车行出入口/车道（可选）
        ]
    },

    # 聚合方式
    "aggregation": "ratio"
}

print(f"\n✅ Calculator ready: {INDICATOR['id']} - {INDICATOR['name']}")
print(f"   Aggregation: {INDICATOR.get('aggregation', 'ratio')}")


# =============================================================================
# 构建颜色查找表
# =============================================================================
COMPONENT_RGB = {}

print(f"\n🎯 Color lookup for components:")
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
    计算 Spatial Feasibility Index (行人空间可行性指数)

    TYPE D: 组合类指标

    算法步骤:
    1. 分别统计 Wn_sidewalk 与 Rn_car_lane 的像素数
    2. 计算 SFI = Wn / Rn
    3. 返回总值及分解值

    Args:
        image_path: 语义分割mask图片路径

    Returns:
        {
            'success': True/False,
            'value': float (SFI),
            'Wn': int,
            'Rn': int,
            'Wn_ratio': float (%),
            'Rn_ratio': float (%),
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

        # Step 2: 统计组成部分像素
        component_counts = {"Wn_sidewalk": 0, "Rn_car_lane": 0}
        component_ratios = {"Wn_sidewalk": 0, "Rn_car_lane": 0}
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

        Wn = int(component_counts.get("Wn_sidewalk", 0))
        Rn = int(component_counts.get("Rn_car_lane", 0))

        # Step 3: 计算比值
        if Rn == 0:
            value = 0
            note = "Rn (car lane pixels) is zero"
        else:
            value = Wn / Rn
            note = None

        # Step 4: 返回结果
        result = {
            'success': True,
            'value': round(float(value), 3),
            'total_pixels': int(total_pixels),
            'Wn': Wn,
            'Rn': Rn,
            'Wn_ratio': component_ratios.get("Wn_sidewalk", 0),
            'Rn_ratio': component_ratios.get("Rn_car_lane", 0),
            'component_pixels': component_counts,
            'component_ratios': component_ratios,
            'class_breakdown': all_class_counts
        }

        if note:
            result['note'] = note

        return result

    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'value': None
        }


# =============================================================================
# 辅助函数
# =============================================================================
def interpret_sfi(sfi: float) -> str:
    """
    解释SFI的含义
    """
    if sfi < 0.2:
        return "Very low pedestrian priority"
    elif sfi < 0.5:
        return "Low pedestrian priority"
    elif sfi < 1.0:
        return "Moderate pedestrian priority"
    else:
        return "High pedestrian priority"


# =============================================================================
# 测试代码
# =============================================================================
if __name__ == "__main__":
    print("\n🧪 Testing Spatial Feasibility Index calculator...")

    test_img = np.zeros((100, 100, 3), dtype=np.uint8)

    if ('sidewalk' in semantic_colors) and ('road' in semantic_colors):
        test_img[0:30, :] = semantic_colors['sidewalk']  # 30% sidewalk
        test_img[30:90, :] = semantic_colors['road']     # 60% road (car lane proxy)

        test_path = '/tmp/test_sfi.png'
        Image.fromarray(test_img).save(test_path)

        result = calculate_indicator(test_path)

        print(f"\n   Test: 30% sidewalk / 60% road => SFI ≈ 0.5")
        print(f"   Result: {result['value']}")
        print(f"   Wn: {result['Wn']}, Rn: {result['Rn']}")
        print(f"   Interpretation: {interpret_sfi(result['value'])}")

        import os
        os.remove(test_path)
    else:
        print("   ⚠️ Required classes not found in semantic_colors")

