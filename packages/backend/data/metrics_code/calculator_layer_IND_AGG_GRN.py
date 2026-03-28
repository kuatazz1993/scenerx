"""
SceneRx Stage 2.5 - Calculator Layer
================================================
指标ID: IND_AGG_GRN
指标名称: Aggregation of Greenery Perception (Greenery Aggregation / 绿视聚合指数)
类型: TYPE B (数学公式类)

说明:
衡量人类感知到的绿化在空间单元之间的聚合/不均衡程度。
当绿化感知在不同位置分布越不均匀，聚合指数越高；
当各位置绿化感知均匀分布时，聚合指数接近 0。

公式:
AI_g = Σ | (P_{gi} / ΣP_g) - (1/n) | / 2

其中:
- P_{gi} 为位置 i 的绿化感知值
- ΣP_g 为研究区域内总绿化感知值
- n 为位置（样本点）数量

单位: 无量纲
范围: 0 (完全均匀) → 接近 1 (高度聚合)
"""

import numpy as np
from typing import Dict, List


# =============================================================================
# 指标定义
# =============================================================================
INDICATOR = {
    "id": "IND_AGG_GRN",
    "name": "Aggregation of Greenery Perception",
    "unit": "dimensionless",
    "formula": "AI_g = Σ | (P_{gi} / ΣP_g) - (1/n) | / 2",
    "target_direction": "NEUTRAL",
    "definition": "Inequality of spatial distribution of perceived greenery across locations",
    "category": "CAT_CFG",

    "calc_type": "custom",

    "variables": {
        "P_{gi}": "Greenery perception value at location i",
        "P_{g}": "Total greenery perception in the area",
        "n": "Total number of locations",
        "AI_g": "Aggregation index of greenery perception"
    }
}

print(f"\n✅ Calculator ready: {INDICATOR['id']} - {INDICATOR['name']}")
print(f"   Formula: {INDICATOR['formula']}")


# =============================================================================
# 计算函数
# =============================================================================
def calculate_indicator(P_gi: List[float]) -> Dict:
    """
    计算 Aggregation of Greenery Perception (绿视聚合指数)

    TYPE B: 自定义数学公式

    算法步骤:
    1. 输入各空间位置的绿化感知值 P_{gi}
    2. 计算总绿化感知 ΣP_g
    3. 计算每个位置的占比 (P_{gi} / ΣP_g)
    4. 计算与完全均匀分布 (1/n) 的偏差
    5. 按公式求和并除以 2

    Args:
        P_gi: list，每个空间位置的绿化感知值（如 GVI、SceneRx 等）

    Returns:
        {
            'success': True/False,
            'value': float (AI_g),
            'n_locations': int,
            'total_greenery': float,
            'mean_share': float,
            'distribution': list
        }
    """
    try:
        values = np.array(P_gi, dtype=float)
        n = len(values)

        if n == 0:
            return {
                'success': True,
                'value': 0,
                'n_locations': 0,
                'total_greenery': 0,
                'mean_share': 0,
                'distribution': [],
                'note': 'No locations provided'
            }

        total_greenery = values.sum()

        if total_greenery <= 0:
            return {
                'success': True,
                'value': 0,
                'n_locations': n,
                'total_greenery': float(total_greenery),
                'mean_share': 1 / n,
                'distribution': values.tolist(),
                'note': 'Total greenery perception is zero'
            }

        proportions = values / total_greenery
        uniform_share = 1 / n

        agg_index = np.sum(np.abs(proportions - uniform_share)) / 2

        return {
            'success': True,
            'value': round(float(agg_index), 3),
            'n_locations': int(n),
            'total_greenery': round(float(total_greenery), 3),
            'mean_share': round(float(uniform_share), 3),
            'distribution': proportions.round(3).tolist()
        }

    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'value': None
        }


# =============================================================================
# 辅助函数：聚合指数解释
# =============================================================================
def interpret_aggregation(value: float) -> str:
    """
    解释绿视聚合指数的含义
    """
    if value < 0.1:
        return "Very low aggregation: greenery is evenly distributed"
    elif value < 0.3:
        return "Low aggregation: slight spatial concentration"
    elif value < 0.6:
        return "Moderate aggregation: noticeable clustering of greenery"
    else:
        return "High aggregation: greenery perception is highly concentrated"


# =============================================================================
# 测试代码
# =============================================================================
if __name__ == "__main__":
    print("\n🧪 Testing Greenery Aggregation calculator...")

    # Case 1: 完全均匀分布
    test_uniform = [10, 10, 10, 10]
    res1 = calculate_indicator(test_uniform)
    print("   Test 1: Uniform distribution")
    print("   Value:", res1['value'])
    print("   Interpretation:", interpret_aggregation(res1['value']))

    # Case 2: 高度聚合
    test_agg = [35, 5, 5, 5]
    res2 = calculate_indicator(test_agg)
    print("   Test 2: Aggregated distribution")
    print("   Value:", res2['value'])
    print("   Interpretation:", interpret_aggregation(res2['value']))
