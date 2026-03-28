"""
SceneRx Stage 2.5 - Calculator Layer
================================================
指标ID: IND_VSG_BLK
指标名称: Visible Street Greenery (Block Level) (街区可见街道绿量)
类型: TYPE D (组合类)

说明:
计算街区周边街道街景绿量的加权平均值，并引入距离衰减。
权重由街道中心线长度与距离衰减项共同决定：
距离越近、街道越长，对街区绿量贡献越大。

公式:
Gn = Σ(Gi * Li * Di^α) / Σ(Li * Di^α)

其中:
- Gn: 街区可见绿量
- Gi: 周边街道街景可见绿量（如GVI）
- Li: 街道中心线长度
- Di: 街道到街区的距离
- α: 距离衰减参数（通常为负值表示随距离衰减）
"""

import numpy as np
from typing import Dict, List


# =============================================================================
# 指标定义
# =============================================================================
INDICATOR = {
    "id": "IND_VSG_BLK",
    "name": "Visible Street Greenery (Block Level)",
    "unit": "ratio",
    "formula": "Gn = Σ(Gi * Li * Di^α) / Σ(Li * Di^α)",
    "target_direction": "POSITIVE",
    "definition": "Distance-decay weighted average of visible street greenery around a block",
    "category": "CAT_CMP",

    "calc_type": "composite",

    # 组成部分（按功能分组）
    "components": [
        "Gi",
        "Li",
        "Di"
    ],

    # 聚合方式
    "aggregation": "distance_decay_weighted_mean",

    # 距离衰减参数
    "alpha": -1.0
}

print(f"\n✅ Calculator ready: {INDICATOR['id']} - {INDICATOR['name']}")
print(f"   Aggregation: {INDICATOR.get('aggregation', 'distance_decay_weighted_mean')}")
print(f"   Alpha: {INDICATOR.get('alpha', -1.0)}")


# =============================================================================
# 计算函数
# =============================================================================
def calculate_indicator(Gi: List[float], Li: List[float], Di: List[float], alpha: float = None) -> Dict:
    """
    计算 Visible Street Greenery (Block Level) (街区可见街道绿量)

    TYPE D: 组合类指标

    算法步骤:
    1. 输入周边街道的Gi、Li、Di数组
    2. 按权重 w_i = Li * Di^α 计算加权
    3. 计算 Gn = Σ(Gi * w_i) / Σ(w_i)

    Args:
        Gi: list，周边街道可见绿量值
        Li: list，周边街道中心线长度
        Di: list，街道到街区距离
        alpha: 距离衰减参数（默认使用INDICATOR['alpha']）

    Returns:
        {
            'success': True/False,
            'value': float (Gn),
            'alpha': float,
            'n_streets': int,
            'weights': list,
            'weighted_components': dict
        }
    """
    try:
        a = float(alpha) if alpha is not None else float(INDICATOR.get('alpha', -1.0))

        Gi_arr = np.array(Gi, dtype=float)
        Li_arr = np.array(Li, dtype=float)
        Di_arr = np.array(Di, dtype=float)

        n = len(Gi_arr)
        if n == 0 or len(Li_arr) != n or len(Di_arr) != n:
            return {
                'success': True,
                'value': 0,
                'alpha': a,
                'n_streets': int(n),
                'note': 'Input lists must have the same non-zero length'
            }

        # 避免距离为0导致Di^α异常
        Di_safe = np.where(Di_arr <= 0, np.nan, Di_arr)

        weights = Li_arr * (Di_safe ** a)

        # 剔除无效权重（Di<=0 或 产生nan/inf）
        valid = np.isfinite(weights) & np.isfinite(Gi_arr) & np.isfinite(Li_arr) & np.isfinite(Di_arr)
        Gi_v = Gi_arr[valid]
        Li_v = Li_arr[valid]
        Di_v = Di_arr[valid]
        w_v = weights[valid]

        denom = float(np.sum(w_v))
        if denom <= 0:
            return {
                'success': True,
                'value': 0,
                'alpha': a,
                'n_streets': int(n),
                'note': 'Sum of weights is zero'
            }

        numer = float(np.sum(Gi_v * w_v))
        Gn = numer / denom

        weighted_components = {
            'numerator': round(numer, 6),
            'denominator': round(denom, 6)
        }

        return {
            'success': True,
            'value': round(float(Gn), 3),
            'alpha': round(float(a), 3),
            'n_streets': int(len(Gi_v)),
            'weights': np.round(w_v, 6).tolist(),
            'weighted_components': weighted_components,
            'inputs_summary': {
                'Gi_mean': round(float(np.mean(Gi_v)) if len(Gi_v) > 0 else 0, 3),
                'Li_sum': round(float(np.sum(Li_v)) if len(Li_v) > 0 else 0, 3),
                'Di_mean': round(float(np.mean(Di_v)) if len(Di_v) > 0 else 0, 3)
            }
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
def interpret_vsg_blk(value: float) -> str:
    """
    解释Gn的含义
    """
    if value < 0.1:
        return "Very low visible greenery around block"
    elif value < 0.25:
        return "Low visible greenery around block"
    elif value < 0.5:
        return "Medium visible greenery around block"
    else:
        return "High visible greenery around block"


# =============================================================================
# 测试代码
# =============================================================================
if __name__ == "__main__":
    print("\n🧪 Testing Visible Street Greenery (Block Level) calculator...")

    # 三条街道示例
    Gi_test = [0.30, 0.50, 0.10]
    Li_test = [120, 80, 200]
    Di_test = [30, 60, 15]
    alpha_test = -1.0

    result = calculate_indicator(Gi_test, Li_test, Di_test, alpha_test)

    print("\n   Test inputs:")
    print(f"      Gi: {Gi_test}")
    print(f"      Li: {Li_test}")
    print(f"      Di: {Di_test}")
    print(f"      alpha: {alpha_test}")
    print(f"      Gn: {result['value']}")
    print(f"      Interpretation: {interpret_vsg_blk(result['value'])}")
