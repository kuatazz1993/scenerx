"""
SceneRx Stage 2.5 - Calculator Layer
================================================
指标ID: IND_VIS_QUA
指标名称: Visual Quality Index (视觉环境质量指数)
类型: TYPE D (组合类)

说明:
将多个感知维度（如安全、活力、美观、财富、愉悦、有趣等）组合为一个视觉环境质量综合指标。
采用简单求和方式进行聚合，反映总体视觉质量水平。

公式: VIS_QUA = Σ(Component Scores)
其中:
- Component Scores: 各感知维度得分（Safety, Liveliness, Beauty, Wealth, Cheerfulness, Interestingness）
"""

import numpy as np
from typing import Dict


# =============================================================================
# 指标定义
# =============================================================================
INDICATOR = {
    "id": "IND_VIS_QUA",
    "name": "Visual Quality Index",
    "unit": "score",
    "formula": "Sum(Safety, Liveliness, Beauty, Wealth, Cheerfulness, Interestingness)",
    "target_direction": "POSITIVE",
    "definition": "Composite visual environmental quality index aggregated from multiple perceptual dimensions",
    "category": "CAT_COM",

    "calc_type": "composite",

    # 组成部分（按功能分组）
    "components": [
        "Safety",
        "Liveliness",
        "Beauty",
        "Wealth",
        "Cheerfulness",
        "Interestingness"
    ],

    # 聚合方式
    "aggregation": "sum"
}

print(f"\n✅ Calculator ready: {INDICATOR['id']} - {INDICATOR['name']}")
print(f"   Aggregation: {INDICATOR.get('aggregation', 'sum')}")


# =============================================================================
# 计算函数
# =============================================================================
def calculate_indicator(values: Dict[str, float]) -> Dict:
    """
    计算 Visual Quality Index (视觉环境质量指数)

    TYPE D: 组合类指标

    算法步骤:
    1. 输入各感知维度得分
    2. 按组件列表进行求和
    3. 返回总值及各组件贡献

    Args:
        values: {
            'Safety': float,
            'Liveliness': float,
            'Beauty': float,
            'Wealth': float,
            'Cheerfulness': float,
            'Interestingness': float
        }

    Returns:
        {
            'success': True/False,
            'value': float (VIS_QUA),
            'aggregation_method': str,
            'components': dict,
            'contributions': dict
        }
    """
    try:
        comps = INDICATOR.get('components', [])

        component_values = {}
        contributions = {}
        total = 0.0

        for k in comps:
            v = float(values.get(k, 0))
            component_values[k] = round(v, 3)
            contributions[k] = round(v, 3)
            total += v

        return {
            'success': True,
            'value': round(float(total), 3),
            'aggregation_method': INDICATOR.get('aggregation', 'sum'),
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
def interpret_vis_qua(score: float) -> str:
    """
    解释VIS_QUA的含义
    """
    if score < 1:
        return "Very low visual quality"
    elif score < 2:
        return "Low visual quality"
    elif score < 3:
        return "Medium visual quality"
    elif score < 4:
        return "High visual quality"
    else:
        return "Very high visual quality"


# =============================================================================
# 测试代码
# =============================================================================
if __name__ == "__main__":
    print("\n🧪 Testing Visual Quality Index calculator...")

    test_values = {
        "Safety": 0.70,
        "Liveliness": 0.55,
        "Beauty": 0.60,
        "Wealth": 0.50,
        "Cheerfulness": 0.45,
        "Interestingness": 0.65
    }

    result = calculate_indicator(test_values)

    print("\n   Test inputs:")
    print(f"      Components: {result['components']}")
    print(f"      VIS_QUA: {result['value']}")
    print(f"      Level: {interpret_vis_qua(result['value'])}")
