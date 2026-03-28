"""
SceneRx Stage 2.5 - Calculator Layer
================================================
指标ID: IND_SQI
指标名称: Street Quality Index (Composite) (街道空间质量指数)
类型: TYPE D (组合类)

说明:
基于多项街景视觉指标构建城市街道总体空间质量综合指数。
权重由AHP确定，并按线性加权模型进行聚合。

公式:
SQI = 1.507*GVI + 0.692*SVI + 0.447*EVI - 2.412*MVI - 0.011*SFI - 0.800*PCI
      - 0.060*VII + 1.414*ITI - 0.167*CCI + 0.208*SWI + 7.798*VEI - 6.344*IRI

其中:
- GVI: Green View Index
- SVI: Sky View Index
- EVI: Enclosure Visual Index
- MVI: Motorization Visual Index
- SFI: Spatial Feasibility Index
- PCI: (Perceptual/Physical) Crowding Index
- VII: (Visual) Infrastructure Index
- ITI: Interestingness Index
- CCI: (Color/Complexity) Crowding Index
- SWI: Sidewalk Index
- VEI: Visual Entropy
- IRI: (Imageability) Risk/Irregularity Index
"""

import numpy as np
from typing import Dict


# =============================================================================
# 指标定义
# =============================================================================
INDICATOR = {
    "id": "IND_SQI",
    "name": "Street Quality Index (Composite)",
    "unit": "score",
    "formula": "SQI = 1.507*GVI + 0.692*SVI + 0.447*EVI - 2.412*MVI - 0.011*SFI - 0.800*PCI - 0.060*VII + 1.414*ITI - 0.167*CCI + 0.208*SWI + 7.798*VEI - 6.344*IRI",
    "target_direction": "POSITIVE",
    "definition": "AHP-weighted composite index measuring overall visual spatial quality of an urban street",
    "category": "CAT_COM",

    "calc_type": "composite",

    # 组成部分（按模型项）
    "components": [
        "GVI",
        "SVI",
        "EVI",
        "MVI",
        "SFI",
        "PCI",
        "VII",
        "ITI",
        "CCI",
        "SWI",
        "VEI",
        "IRI"
    ],

    # 聚合方式
    "aggregation": "weighted_sum",

    # AHP权重（线性模型系数）
    "weights": {
        "GVI": 1.507,
        "SVI": 0.692,
        "EVI": 0.447,
        "MVI": -2.412,
        "SFI": -0.011,
        "PCI": -0.800,
        "VII": -0.060,
        "ITI": 1.414,
        "CCI": -0.167,
        "SWI": 0.208,
        "VEI": 7.798,
        "IRI": -6.344
    }
}

print(f"\n✅ Calculator ready: {INDICATOR['id']} - {INDICATOR['name']}")
print(f"   Aggregation: {INDICATOR.get('aggregation', 'weighted_sum')}")


# =============================================================================
# 计算函数
# =============================================================================
def calculate_indicator(values: Dict[str, float]) -> Dict:
    """
    计算 Street Quality Index (Composite) (街道空间质量指数)

    TYPE D: 组合类指标

    算法步骤:
    1. 输入各子指标值（建议为同量纲/已标准化后的值）
    2. 按线性权重系数计算加权和
    3. 返回总值与各项贡献分解

    Args:
        values: dict, 包含GVI/SVI/EVI/MVI/SFI/PCI/VII/ITI/CCI/SWI/VEI/IRI

    Returns:
        {
            'success': True/False,
            'value': float (SQI),
            'aggregation_method': str,
            'weights': dict,
            'components': dict,
            'contributions': dict
        }
    """
    try:
        comps = INDICATOR.get('components', [])
        w = INDICATOR.get('weights', {})

        component_values = {}
        contributions = {}
        total = 0.0

        for k in comps:
            x = float(values.get(k, 0))
            wk = float(w.get(k, 0))
            component_values[k] = round(x, 3)
            contrib = wk * x
            contributions[k] = round(float(contrib), 3)
            total += contrib

        return {
            'success': True,
            'value': round(float(total), 3),
            'aggregation_method': INDICATOR.get('aggregation', 'weighted_sum'),
            'weights': {k: round(float(w.get(k, 0)), 3) for k in comps},
            'components': component_values,
            'contributions': contributions
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
def interpret_sqi(score: float) -> str:
    """
    解释SQI的含义
    """
    if score < -1:
        return "Very low street quality"
    elif score < 0:
        return "Low street quality"
    elif score < 1:
        return "Medium street quality"
    elif score < 2:
        return "High street quality"
    else:
        return "Very high street quality"


# =============================================================================
# 测试代码
# =============================================================================
if __name__ == "__main__":
    print("\n🧪 Testing Street Quality Index calculator...")

    test_values = {
        "GVI": 0.35,
        "SVI": 0.40,
        "EVI": 0.30,
        "MVI": 0.20,
        "SFI": 0.50,
        "PCI": 0.25,
        "VII": 0.30,
        "ITI": 0.45,
        "CCI": 0.20,
        "SWI": 0.55,
        "VEI": 0.60,
        "IRI": 0.25
    }

    result = calculate_indicator(test_values)

    print("\n   Test inputs:")
    print(f"      Components: {result['components']}")
    print(f"      Contributions: {result['contributions']}")
    print(f"      SQI: {result['value']}")
    print(f"      Level: {interpret_sqi(result['value'])}")
