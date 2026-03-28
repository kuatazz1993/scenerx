"""
SceneRx Stage 2.5 - Calculator Layer
================================================
指标ID: IND_TVW
指标名称: Type of Visual Walkability (视觉步行性类型)
类型: TYPE E (深度学习类)

说明:
基于视觉步行性相关指标（如 greenery, openness, pavement, crowdedness）对街段进行类型划分。
通过K-means聚类将样本分配到不同步行性类型（cluster），输出类别标签及聚类置信度（可选）。

由于聚类属于无监督学习，此示例提供两种实现：
1. 完整实现（使用sklearn进行K-means聚类，需要预先fit并保存模型）
2. 占位符实现（基于简单规则分段，用于测试流程）

⚠️ 注意:
- 完整实现需要安装 scikit-learn: pip install scikit-learn
- 需要预先fit好的KMeans模型文件（或在代码中提供训练数据进行fit）
- 输入应为指标向量而不是单张图片
"""

import numpy as np
from typing import Dict
import os
import pickle


# =============================================================================
# 指标定义
# =============================================================================
INDICATOR = {
    "id": "IND_TVW",
    "name": "Type of Visual Walkability",
    "unit": "category",
    "formula": "K-means Clustering(Gi, Si, Di, Ni)",
    "target_direction": "NEUTRAL",
    "definition": "Categorical classification of street segments via K-means clustering of visual walkability indicators",
    "category": "CAT_COM",

    "calc_type": "deep_learning",

    # 模型配置（完整实现时使用）
    "model_config": {
        "model_type": "KMeans",
        "model_path": "./models/tvw_kmeans.pkl",   # 需要预训练聚类模型
        "n_clusters": 5,
        "feature_order": ["Gi", "Si", "Di", "Ni"], # greenery, openness, pavement, crowdedness
        "standardize": True,                       # 是否需要标准化（若训练时做了StandardScaler）
        "scaler_path": "./models/tvw_scaler.pkl"   # 可选：StandardScaler
    },

    # 输出配置
    "output_type": "classification",

    # 占位符模式（无模型时使用）
    "use_placeholder": True
}

print(f"\n✅ Calculator ready: {INDICATOR['id']} - {INDICATOR['name']}")
print(f"   Mode: {'Placeholder (rule-based)' if INDICATOR.get('use_placeholder', True) else 'K-means'}")


# =============================================================================
# 检测sklearn环境
# =============================================================================
SKLEARN_AVAILABLE = False
try:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
    print(f"   scikit-learn: Available")
except ImportError:
    print(f"   scikit-learn: Not installed")
    print(f"   To enable full K-means mode: pip install scikit-learn")


# =============================================================================
# 计算函数
# =============================================================================
def calculate_indicator(values: Dict[str, float]) -> Dict:
    """
    计算 Type of Visual Walkability (视觉步行性类型)

    TYPE E: 深度学习/机器学习类（无监督聚类）

    根据配置选择实现方式:
    - 占位符模式: 基于简单规则对(Gi,Si,Di,Ni)分段归类
    - 完整模式: 加载KMeans模型，对输入向量进行聚类预测

    Args:
        values: dict, 至少包含 Gi, Si, Di, Ni

    Returns:
        {
            'success': True/False,
            'value': int/str (cluster label),
            'method': str,
            'cluster': int,
            'distance_to_centroids': list (可选),
            'features_used': dict
        }
    """
    use_placeholder = INDICATOR.get('use_placeholder', True)

    if use_placeholder or not SKLEARN_AVAILABLE:
        return calculate_placeholder(values)
    else:
        return calculate_kmeans(values)


def calculate_placeholder(values: Dict[str, float]) -> Dict:
    """
    占位符实现：基于规则的步行性类型划分

    规则示例（仅用于流程测试）:
    - 高绿量(Gi) + 高开敞(Si) + 高铺装(Di) + 低拥挤(Ni) -> Type 1
    - 低绿量 + 低开敞 + 高拥挤 -> Type 2
    - 其他 -> Type 0

    注意: 仅用于测试流程，不代表真实聚类结果
    """
    try:
        Gi = float(values.get("Gi", 0))
        Si = float(values.get("Si", 0))
        Di = float(values.get("Di", 0))
        Ni = float(values.get("Ni", 0))

        # 简单阈值（假定指标已0-1归一化）
        if (Gi >= 0.5) and (Si >= 0.5) and (Di >= 0.5) and (Ni <= 0.3):
            cluster = 1
            label = "High walkability (green-open-paved, low crowdedness)"
        elif (Gi <= 0.3) and (Si <= 0.3) and (Ni >= 0.6):
            cluster = 2
            label = "Low walkability (low green-open, high crowdedness)"
        else:
            cluster = 0
            label = "Mixed walkability"

        return {
            'success': True,
            'value': cluster,
            'cluster': int(cluster),
            'method': 'placeholder_rule_based',
            'features_used': {
                'Gi': round(Gi, 3),
                'Si': round(Si, 3),
                'Di': round(Di, 3),
                'Ni': round(Ni, 3)
            },
            'label': label,
            'note': 'This is a placeholder rule-based classification, not K-means clustering'
        }

    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'value': None,
            'method': 'placeholder_rule_based'
        }


def calculate_kmeans(values: Dict[str, float]) -> Dict:
    """
    完整实现：加载KMeans模型进行聚类预测

    需要:
    - scikit-learn
    - 预训练KMeans模型文件（pkl）
    - 如训练时使用StandardScaler，则提供scaler_path
    """
    try:
        cfg = INDICATOR.get('model_config', {})
        model_path = cfg.get('model_path', '')
        scaler_path = cfg.get('scaler_path', None)
        feature_order = cfg.get('feature_order', ["Gi", "Si", "Di", "Ni"])
        use_scaler = bool(cfg.get('standardize', True))

        if not os.path.exists(model_path):
            return {
                'success': False,
                'error': f'Model file not found: {model_path}',
                'value': None,
                'method': 'kmeans',
                'fallback': 'Run with use_placeholder=True or provide kmeans model file'
            }

        with open(model_path, "rb") as f:
            kmeans = pickle.load(f)

        x = np.array([[float(values.get(k, 0)) for k in feature_order]], dtype=float)

        scaler = None
        if use_scaler and scaler_path and os.path.exists(scaler_path):
            with open(scaler_path, "rb") as f:
                scaler = pickle.load(f)
            x_in = scaler.transform(x)
        else:
            x_in = x

        cluster = int(kmeans.predict(x_in)[0])

        # 到各中心的距离（可做“置信度”代理：越近越确定）
        if hasattr(kmeans, "transform"):
            dists = kmeans.transform(x_in).reshape(-1).tolist()
            dists = [round(float(d), 6) for d in dists]
        else:
            dists = None

        return {
            'success': True,
            'value': cluster,
            'cluster': cluster,
            'method': 'kmeans',
            'features_used': {k: round(float(values.get(k, 0)), 3) for k in feature_order},
            'distance_to_centroids': dists
        }

    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'value': None,
            'method': 'kmeans'
        }


# =============================================================================
# 辅助函数
# =============================================================================
def interpret_tvw(cluster: int) -> str:
    """
    解释聚类类别（示例占位）
    """
    mapping = {
        0: "Mixed/Transitional walkability type",
        1: "High walkability type",
        2: "Low walkability type",
        3: "Crowded but active walkability type",
        4: "Open but low greenery walkability type"
    }
    return mapping.get(int(cluster), "Unknown walkability type")


# =============================================================================
# 测试代码
# =============================================================================
if __name__ == "__main__":
    print("\n🧪 Testing Type of Visual Walkability calculator...")

    tests = [
        ("High walkability", {"Gi": 0.7, "Si": 0.6, "Di": 0.8, "Ni": 0.2}),
        ("Low walkability", {"Gi": 0.2, "Si": 0.2, "Di": 0.5, "Ni": 0.8}),
        ("Mixed", {"Gi": 0.4, "Si": 0.5, "Di": 0.3, "Ni": 0.4})
    ]

    for name, vals in tests:
        result = calculate_indicator(vals)
        print(f"\n   {name}:")
        print(f"      Cluster: {result.get('value')}")
        print(f"      Method: {result.get('method')}")
        print(f"      Interpretation: {interpret_tvw(int(result.get('value') or 0))}")
