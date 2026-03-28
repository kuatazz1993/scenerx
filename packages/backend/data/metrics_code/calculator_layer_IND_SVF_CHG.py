"""
SceneRx Stage 2.5 - Calculator Layer
================================================
指标ID: IND_SVF_CHG
指标名称: Sky View Factor Change (SVF变化)
类型: TYPE D (组合类)

说明:
计算相邻两个测量点（通常间隔10m）之间的Sky View Factor（SVF）变化幅度。
用于反映天空开敞度在空间上的突变程度。

公式: SVF_CHG = |SVF_t - SVF_{t-1}|
其中:
- SVF_t: 当前点SVF
- SVF_{t-1}: 前一点SVF
"""

import numpy as np
from PIL import Image
from typing import Dict


# =============================================================================
# 指标定义
# =============================================================================
INDICATOR = {
    "id": "IND_SVF_CHG",
    "name": "Sky View Factor Change",
    "unit": "ratio",
    "formula": "|SVF_t - SVF_{t-1}|",
    "target_direction": "NEUTRAL",
    "definition": "Absolute change in Sky View Factor between two adjacent points",
    "category": "CAT_CFG",

    "calc_type": "composite",

    # 组成部分（按功能分组）
    "component_sources": {
        "current": "SVF_t",
        "previous": "SVF_{t-1}"
    },

    # 聚合方式
    "aggregation": "absolute_difference"
}

print(f"\n✅ Calculator ready: {INDICATOR['id']} - {INDICATOR['name']}")
print(f"   Aggregation: {INDICATOR.get('aggregation', 'absolute_difference')}")


# =============================================================================
# 计算函数
# =============================================================================
def calculate_indicator(SVF_t: float, SVF_t_minus_1: float) -> Dict:
    """
    计算 Sky View Factor Change (SVF变化)

    TYPE D: 组合类指标

    算法步骤:
    1. 输入相邻两点SVF值（SVF_t, SVF_{t-1}）
    2. 计算绝对差值 |SVF_t - SVF_{t-1}|
    3. 返回总值及分解值

    Args:
        SVF_t: 当前点SVF
        SVF_t_minus_1: 前一点SVF

    Returns:
        {
            'success': True/False,
            'value': float (SVF变化幅度),
            'SVF_t': float,
            'SVF_{t-1}': float,
            'aggregation_method': str
        }
    """
    try:
        svf_current = float(SVF_t)
        svf_prev = float(SVF_t_minus_1)

        value = abs(svf_current - svf_prev)

        change_level = interpret_svf_change(value)

        return {
            'success': True,
            'value': round(float(value), 3),
            'aggregation_method': INDICATOR.get('aggregation', 'absolute_difference'),
            'SVF_t': round(svf_current, 3),
            'SVF_{t-1}': round(svf_prev, 3),
            'change_level': change_level
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
def interpret_svf_change(change: float) -> str:
    """
    解释SVF变化幅度的含义

    Args:
        change: |SVF_t - SVF_{t-1}|

    Returns:
        str: 变化幅度级别描述
    """
    if change < 0.05:
        return "Very stable: minimal SVF change"
    elif change < 0.15:
        return "Stable: small SVF change"
    elif change < 0.30:
        return "Moderate change: noticeable SVF variation"
    else:
        return "High change: strong SVF variation"


# =============================================================================
# 测试代码
# =============================================================================
if __name__ == "__main__":
    print("\n🧪 Testing SVF Change calculator...")

    # 测试样例
    tests = [
        ("Very small change", 0.62, 0.60),
        ("Moderate change", 0.62, 0.40),
        ("High change", 0.85, 0.20),
    ]

    for name, svf_t, svf_prev in tests:
        result = calculate_indicator(svf_t, svf_prev)

        print(f"\n   {name}:")
        print(f"      SVF_t: {result['SVF_t']}")
        print(f"      SVF_{'{t-1}'}: {result['SVF_{t-1}']}")
        print(f"      SVF_CHG: {result['value']}")
        print(f"      Level: {result['change_level']}")
