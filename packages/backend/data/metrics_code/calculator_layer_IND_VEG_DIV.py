"""
SceneRx Stage 2.5 - Calculator Layer
================================================
指标ID: IND_VEG_DIV
指标名称: Vegetation Diversity Index (Vegetation Diversity / 植被多样性指数)
类型: TYPE B (数学公式类)

说明:
衡量Green View Index内部不同植被类型（tree, grass, plant）的多样性。
采用类似Simpson指数的形式：1 - Σ(pᵢ²)。
值越高表示植被类型越均衡、越多样；值越低表示被单一植被类型主导。

公式:
VEG_DIV = 1 - ( (Tn/GVIn)^2 + (Pn/GVIn)^2 + (Gn/GVIn)^2 )
其中:
- Tn: 树木占比（Percentage of trees）
- Pn: 植物占比（Percentage of plants）
- Gn: 草地占比（Percentage of grass）
- GVIn: Green View Index（总绿色占比 = Tn + Pn + Gn）

单位: 无量纲
范围: 0 (单一类型) 到 接近 2/3 (三类均匀分布时)
"""

import numpy as np
from PIL import Image
from typing import Dict


# =============================================================================
# 指标定义
# =============================================================================
INDICATOR = {
    "id": "IND_VEG_DIV",
    "name": "Vegetation Diversity Index",
    "unit": "dimensionless",
    "formula": "1 - ((Tn/GVIn)^2 + (Pn/GVIn)^2 + (Gn/GVIn)^2)",
    "target_direction": "POSITIVE",
    "definition": "Simpson-like diversity index of vegetation composition within GVI (tree/grass/plant)",
    "category": "CAT_CMP",

    "calc_type": "custom",

    "variables": {
        "Tn": "Percentage of trees",
        "Gn": "Percentage of grass",
        "Pn": "Percentage of plants",
        "GVIn": "Green View Index for the image (Tn + Pn + Gn)"
    }
}

print(f"\n✅ Calculator ready: {INDICATOR['id']} - {INDICATOR['name']}")
print(f"   Formula: {INDICATOR['formula']}")


# =============================================================================
# 计算函数
# =============================================================================
def calculate_indicator(image_path: str) -> Dict:
    """
    计算 Vegetation Diversity Index (植被多样性指数) - Simpson-like

    TYPE B: 自定义数学公式

    算法步骤:
    1. 统计 tree / grass / plant 三类像素数（或占比）
    2. 计算 GVIn = Tn + Pn + Gn
    3. 计算 VEG_DIV = 1 - Σ( (x/GVIn)^2 )

    Args:
        image_path: 语义分割mask图片路径

    Returns:
        {
            'success': True/False,
            'value': float (无量纲),
            'Tn': float,
            'Pn': float,
            'Gn': float,
            'GVIn': float,
            'total_pixels': int,
            'matched_pixels': int,
            'unmatched_pixels': int,
            'class_distribution': dict
        }
    """
    try:
        # Step 1: 加载图片
        img = Image.open(image_path).convert('RGB')
        pixels = np.array(img)
        h, w, _ = pixels.shape
        total_pixels = h * w
        flat_pixels = pixels.reshape(-1, 3)

        # Step 2: 统计 tree / grass / plant 像素数
        target_classes = ['tree', 'grass', 'plant']
        class_counts = {}

        for class_name in target_classes:
            if class_name not in semantic_colors:
                continue
            rgb = semantic_colors[class_name]
            mask = np.all(flat_pixels == rgb, axis=1)
            count = int(np.sum(mask))
            if count > 0:
                class_counts[class_name] = count

        matched_pixels = sum(class_counts.values())
        unmatched_pixels = total_pixels - matched_pixels

        # 若没有任何绿色植被类
        if matched_pixels == 0:
            return {
                'success': True,
                'value': 0,
                'Tn': 0,
                'Pn': 0,
                'Gn': 0,
                'GVIn': 0,
                'total_pixels': int(total_pixels),
                'matched_pixels': 0,
                'unmatched_pixels': int(unmatched_pixels),
                'class_distribution': {},
                'note': 'No vegetation classes detected in image'
            }

        # Step 3: 计算 Tn/Pn/Gn（以“绿色内部占比”为口径）
        Tn = class_counts.get('tree', 0)
        Pn = class_counts.get('plant', 0)
        Gn = class_counts.get('grass', 0)

        GVIn = Tn + Pn + Gn

        if GVIn == 0:
            return {
                'success': True,
                'value': 0,
                'Tn': 0,
                'Pn': 0,
                'Gn': 0,
                'GVIn': 0,
                'total_pixels': int(total_pixels),
                'matched_pixels': int(matched_pixels),
                'unmatched_pixels': int(unmatched_pixels),
                'class_distribution': class_counts,
                'note': 'GVIn is zero'
            }

        # Step 4: 计算 VEG_DIV = 1 - Σ(pᵢ²)
        pt = Tn / GVIn
        pp = Pn / GVIn
        pg = Gn / GVIn

        veg_div = 1 - (pt ** 2 + pp ** 2 + pg ** 2)

        # Step 5: 返回结果
        return {
            'success': True,
            'value': round(float(veg_div), 3),
            'Tn': round(float(pt), 3),
            'Pn': round(float(pp), 3),
            'Gn': round(float(pg), 3),
            'GVIn': round(float(GVIn / total_pixels), 3),
            'total_pixels': int(total_pixels),
            'matched_pixels': int(matched_pixels),
            'unmatched_pixels': int(unmatched_pixels),
            'class_distribution': class_counts
        }

    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'value': None
        }


# =============================================================================
# 辅助函数：指数解释
# =============================================================================
def interpret_veg_div(value: float) -> str:
    """
    解释植被多样性指数的含义
    """
    if value < 0.2:
        return "Low diversity: dominated by one vegetation type"
    elif value < 0.4:
        return "Moderate diversity: some balance among vegetation types"
    else:
        return "High diversity: vegetation types are well balanced"


# =============================================================================
# 测试代码
# =============================================================================
if __name__ == "__main__":
    print("\n🧪 Testing Vegetation Diversity calculator...")

    # 创建测试图片 - tree/grass/plant 均等分布
    test_img = np.zeros((90, 90, 3), dtype=np.uint8)

    required = all(k in semantic_colors for k in ['tree', 'grass', 'plant'])
    if required:
        test_img[0:30, :] = semantic_colors['tree']
        test_img[30:60, :] = semantic_colors['grass']
        test_img[60:90, :] = semantic_colors['plant']

        test_path = '/tmp/test_veg_div.png'
        Image.fromarray(test_img).save(test_path)

        result = calculate_indicator(test_path)

        print("   Test: 1/3 tree + 1/3 grass + 1/3 plant")
        print("   Expected: 1 - 3*(1/3^2) = 0.667")
        print(f"   Result: {result['value']}")
        print(f"   Interpretation: {interpret_veg_div(result['value'])}")

        import os
        os.remove(test_path)
    else:
        print("   ⚠️ Required classes not found in semantic_colors")
