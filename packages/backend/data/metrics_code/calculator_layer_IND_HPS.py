"""
SceneRx Stage 2.5 - Calculator Layer
================================================
指标ID: IND_HPS
指标名称: Human Perception Score (人类感知综合得分)
类型: TYPE D (组合类)

说明:
基于Greenness, Openness, Enclosure, Walkability, Imageability五个维度构建综合得分。
权重通过熵权法（Entropy Weighting Method）确定，然后进行加权求和。

公式: HPS = Σ (w_k × x_k)
其中:
- x_k: 标准化后的子指标值（Greenness, Openness, Enclosure, Walkability, Imageability）
- w_k: 熵权法得到的权重（Σw_k = 1）
"""

import numpy as np
from typing import Dict


# =============================================================================
# 指标定义
# =============================================================================
INDICATOR = {
    "id": "IND_HPS",
    "name": "Human Perception Score",
    "unit": "score",
    "formula": "HPS = Σ (w_k × x_k)",
    "target_direction": "POSITIVE",
    "definition": "Composite score of human perception dimensions using entropy-derived weights",
    "category": "CAT_COM",

    "calc_type": "composite",

    # 组成部分（按功能分组）
    "components": [
        "Greenness",
        "Openness",
        "Enclosure",
        "Walkability",
        "Imageability"
    ],

    # 聚合方式
    "aggregation": "weighted_sum",

    # 权重（由熵权法确定；可在外部更新/传入）
    "weights": {
        "Greenness": 0.2,
        "Openness": 0.2,
        "Enclosure": 0.2,
        "Walkability": 0.2,
        "Imageability": 0.2
    }
}

print(f"\n✅ Calculator ready: {INDICATOR['id']} - {INDICATOR['name']}")
print(f"   Aggregation: {INDICATOR.get('aggregation', 'weighted_sum')}")


# =============================================================================
# 计算函数
# =============================================================================
def calculate_indicator(values: Dict[str, float], weights: Dict[str, float] = None) -> Dict:
    """
    计算 Human Perception Score (人类感知综合得分)

    TYPE D: 组合类指标

    算法步骤:
    1. 输入五个子指标值（建议为标准化后的值）
    2. 使用熵权法得到的权重（可传入；否则使用INDICATOR默认权重）
    3. 计算加权和 HPS = Σ(w_k × x_k)
    4. 返回总值及各分量贡献

    Args:
        values: {
            'Greenness': float,
            'Openness': float,
            'Enclosure': float,
            'Walkability': float,
            'Imageability': float
        }
        weights: 可选，熵权法得到的权重dict（Σw_k应为1）

    Returns:
        {
            'success': True/False,
            'value': float (HPS),
            'weights': dict,
            'components': dict (输入值),
            'contributions': dict (w_k * x_k),
            'aggregation_method': str
        }
    """
    try:
        comps = INDICATOR.get('components', [])
        w = weights if weights is not None else INDICATOR.get('weights', {})

        # Step 1: 提取并校验输入
        x = {}
        for k in comps:
            x[k] = float(values.get(k, 0))

        # Step 2: 权重归一化（防止Σw≠1）
        w_vec = np.array([float(w.get(k, 0)) for k in comps], dtype=float)
        w_sum = float(np.sum(w_vec))
        if w_sum > 0:
            w_vec = w_vec / w_sum
        else:
            w_vec = np.ones(len(comps), dtype=float) / len(comps)

        w_norm = {k: round(float(w_vec[i]), 3) for i, k in enumerate(comps)}

        # Step 3: 计算加权和与贡献
        contributions = {}
        total = 0.0
        for i, k in enumerate(comps):
            c = float(w_vec[i] * x[k])
            contributions[k] = round(c, 3)
            total += c

        return {
            'success': True,
            'value': round(float(total), 3),
            'aggregation_method': INDICATOR.get('aggregation', 'weighted_sum'),
            'weights': w_norm,
            'components': {k: round(float(x[k]), 3) for k in comps},
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
def interpret_hps(score: float) -> str:
    """
    解释HPS的含义
    """
    if score < 0.2:
        return "Very low perceived quality"
    elif score < 0.4:
        return "Low perceived quality"
    elif score < 0.6:
        return "Medium perceived quality"
    elif score < 0.8:
        return "High perceived quality"
    else:
        return "Very high perceived quality"


# =============================================================================
# 测试代码
# =============================================================================
if __name__ == "__main__":
    print("\n🧪 Testing Human Perception Score calculator...")

    test_values = {
        "Greenness": 0.70,
        "Openness": 0.55,
        "Enclosure": 0.40,
        "Walkability": 0.60,
        "Imageability": 0.50
    }

    test_weights = {
        "Greenness": 0.25,
        "Openness": 0.20,
        "Enclosure": 0.15,
        "Walkability": 0.25,
        "Imageability": 0.15
    }

    result = calculate_indicator(test_values, test_weights)

    print("\n   Test inputs:")
    print(f"      Values: {result['components']}")
    print(f"      Weights: {result['weights']}")
    print(f"      Contributions: {result['contributions']}")
    print(f"      HPS: {result['value']}")
    print(f"      Level: {interpret_hps(result['value'])}")
