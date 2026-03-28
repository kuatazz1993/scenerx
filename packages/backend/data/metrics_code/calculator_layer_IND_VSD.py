"""
SceneRx Stage 2.5 - Calculator Layer
================================================
指标ID: IND_VSD
指标名称: Vegetation Structural Diversity (植被结构多样性)
类型: TYPE B (数学公式类)

说明:
衡量植被垂直层次（tree, shrub, herb）的结构多样性。
基于语义分割像素统计，采用Shannon-Wiener指数计算：
层次类型越丰富、分布越均匀，指数越高。

公式:
VSD = - Σ(pᵢ × ln(pᵢ))
其中 pᵢ = 某类植被像素数 / 植被总像素数

单位: 无量纲
范围: 0 (单一层次) 到 ln(n) (n个层次均匀分布)
"""

import numpy as np
from PIL import Image
from typing import Dict


# =============================================================================
# 指标定义
# =============================================================================
INDICATOR = {
    "id": "IND_VSD",
    "name": "Vegetation Structural Diversity",
    "unit": "dimensionless",
    "formula": "VSD = -Σ(pᵢ × ln(pᵢ))",
    "target_direction": "POSITIVE",
    "definition": "Shannon-Wiener index measuring diversity of vegetation vertical layers (tree/shrub/herb)",
    "category": "CAT_CFG",

    "calc_type": "custom",

    "variables": {
        "pi": "Proportion of specific vegetation type pixels relative to total vegetation pixels",
        "VSD": "Vegetation Structural Diversity (Shannon-Wiener index)",
        "n": "Number of unique vegetation layers detected"
    }
}

print(f"\n✅ Calculator ready: {INDICATOR['id']} - {INDICATOR['name']}")
print(f"   Formula: {INDICATOR['formula']}")


# =============================================================================
# 计算函数
# =============================================================================
def calculate_indicator(image_path: str) -> Dict:
    """
    计算 Vegetation Structural Diversity (植被结构多样性) - Shannon-Wiener

    TYPE B: 自定义数学公式

    算法步骤:
    1. 统计图像中 tree / shrub / herb 的像素数
    2. 计算每个类别概率 pᵢ = count_i / total_veg
    3. 计算 VSD = -Σ(pᵢ × ln(pᵢ))

    Args:
        image_path: 语义分割mask图片路径

    Returns:
        {
            'success': True/False,
            'value': float (VSD),
            'n_layers': int,
            'max_possible_entropy': float (理论最大值 ln(n)),
            'normalized_entropy': float (归一化 0-1),
            'total_pixels': int,
            'matched_pixels': int,
            'unmatched_pixels': int,
            'layer_distribution': dict
        }
    """
    try:
        # Step 1: 加载图片
        img = Image.open(image_path).convert('RGB')
        pixels = np.array(img)
        h, w, _ = pixels.shape
        total_pixels = h * w
        flat_pixels = pixels.reshape(-1, 3)

        # Step 2: 统计 tree / shrub / herb 像素数
        target_layers = ['tree', 'shrub', 'herb']
        layer_counts = {}

        for layer in target_layers:
            if layer not in semantic_colors:
                continue
            rgb = semantic_colors[layer]
            mask = np.all(flat_pixels == rgb, axis=1)
            count = int(np.sum(mask))
            if count > 0:
                layer_counts[layer] = count

        total_veg = sum(layer_counts.values())
        unmatched_pixels = total_pixels - total_veg

        if total_veg == 0:
            return {
                'success': True,
                'value': 0,
                'n_layers': 0,
                'max_possible_entropy': 0,
                'normalized_entropy': 0,
                'total_pixels': int(total_pixels),
                'matched_pixels': 0,
                'unmatched_pixels': int(unmatched_pixels),
                'layer_distribution': {},
                'note': 'No vegetation layers detected in image'
            }

        # Step 3: 计算概率分布
        probabilities = [count / total_veg for count in layer_counts.values()]

        # Step 4: 计算 Shannon-Wiener 指数（自然对数 ln）
        vsd = 0.0
        for p in probabilities:
            if p > 0:
                vsd -= p * np.log(p)

        # Step 5: 计算额外指标
        n_layers = len(layer_counts)
        max_entropy = np.log(n_layers) if n_layers > 1 else 0
        normalized_entropy = vsd / max_entropy if max_entropy > 0 else 0

        return {
            'success': True,
            'value': round(float(vsd), 3),
            'n_layers': int(n_layers),
            'max_possible_entropy': round(float(max_entropy), 3),
            'normalized_entropy': round(float(normalized_entropy), 3),
            'total_pixels': int(total_pixels),
            'matched_pixels': int(total_veg),
            'unmatched_pixels': int(unmatched_pixels),
            'layer_distribution': layer_counts,
            'top_layers': dict(sorted(layer_counts.items(),
                                      key=lambda x: x[1],
                                      reverse=True)[:3])
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
def interpret_vsd(vsd: float, n_layers: int) -> str:
    """
    解释VSD的含义
    """
    if n_layers <= 1:
        return "Very low diversity: dominated by a single vegetation layer"

    max_h = np.log(n_layers)
    ratio = vsd / max_h if max_h > 0 else 0

    if ratio < 0.3:
        return "Low diversity: dominated by one layer"
    elif ratio < 0.6:
        return "Medium diversity: moderate layer balance"
    elif ratio < 0.8:
        return "High diversity: diverse layer distribution"
    else:
        return "Very high diversity: nearly uniform layer distribution"


# =============================================================================
# 测试代码
# =============================================================================
if __name__ == "__main__":
    print("\n🧪 Testing Vegetation Structural Diversity calculator...")

    # 创建测试图片 - 1/3 tree, 1/3 shrub, 1/3 herb
    test_img = np.zeros((90, 90, 3), dtype=np.uint8)

    if all(k in semantic_colors for k in ['tree', 'shrub', 'herb']):
        test_img[0:30, :] = semantic_colors['tree']
        test_img[30:60, :] = semantic_colors['shrub']
        test_img[60:90, :] = semantic_colors['herb']

        test_path = '/tmp/test_vsd.png'
        Image.fromarray(test_img).save(test_path)

        result = calculate_indicator(test_path)

        print("   Test: 1/3 tree + 1/3 shrub + 1/3 herb")
        print("   Expected: ln(3) ≈ 1.099 (uniform distribution)")
        print(f"   Result: {result['value']}")
        print(f"   Layers: {result['n_layers']}")
        print(f"   Interpretation: {interpret_vsd(result['value'], result['n_layers'])}")

        import os
        os.remove(test_path)
    else:
        print("   ⚠️ Required classes not found in semantic_colors")
