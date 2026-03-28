"""
SceneRx Stage 2.5 - Calculator Layer
================================================
指标ID: IND_GRC_CON
指标名称: Grayscale Contrast (GLCM) (灰度对比度-GLCM)
类型: TYPE C (图像处理类)

说明:
基于灰度共生矩阵（GLCM）的纹理对比度指标，用于表征相邻像素灰度差异程度。
对比度越大表示邻域灰度变化越强、纹理越粗糙。

公式:
GRC_CON = Σ (i - j)^2 × P(i, j)
其中:
- P(i, j): 灰度值 i 与 j 在给定位移关系下的共生概率

单位: 无量纲
范围: >= 0
"""

import numpy as np
from PIL import Image
from typing import Dict
import os


# =============================================================================
# 指标定义
# =============================================================================
INDICATOR = {
    "id": "IND_GRC_CON",
    "name": "Grayscale Contrast (GLCM)",
    "unit": "dimensionless",
    "formula": "Σ (i-j)^2 × P(i,j)",
    "target_direction": "NEUTRAL",
    "definition": "GLCM contrast measuring intensity difference between neighboring pixels in grayscale",
    "category": "CAT_CMP",

    "calc_type": "custom",

    "variables": {
        "P(i,j)": "Gray Level Co-occurrence probability matrix",
        "i,j": "Gray levels",
        "d": "Pixel offset distance",
        "θ": "Direction of offset"
    },

    # TYPE C 特殊配置
    "use_original_image": False,
    "original_image_path": None,

    # GLCM配置
    "levels": 32,
    "distance": 1,
    "angles": [0, 45, 90, 135]  # degrees
}

print(f"\n✅ Calculator ready: {INDICATOR['id']} - {INDICATOR['name']}")
print(f"   Formula: {INDICATOR['formula']}")
print(f"   Use original image: {INDICATOR.get('use_original_image', False)}")


# =============================================================================
# 计算函数
# =============================================================================
def calculate_indicator(image_path: str) -> Dict:
    """
    计算 Grayscale Contrast (GLCM) (灰度对比度-GLCM)

    TYPE C: 图像处理类

    算法步骤:
    1. 加载图像（mask或原始图像）并转灰度
    2. 量化灰度级（levels）
    3. 构建GLCM并归一化得到P(i,j)
    4. 计算对比度 Σ(i-j)^2 × P(i,j)
    5. 多方向取平均（可选）

    Args:
        image_path: 图片路径

    Returns:
        {
            'success': True/False,
            'value': float (GRC_CON),
            'levels': int,
            'distance': int,
            'angles': list,
            'per_angle_contrast': dict,
            'dimensions': dict
        }
    """
    try:
        # Step 1: 确定实际图片路径
        actual_path = image_path
        image_source = 'mask'

        if INDICATOR.get('use_original_image', False):
            original_base = INDICATOR.get('original_image_path')
            if original_base:
                if 'mask' in image_path:
                    relative = image_path.split('mask')[-1]
                    for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
                        test_path = original_base + relative.rsplit('.', 1)[0] + ext
                        if os.path.exists(test_path):
                            actual_path = test_path
                            image_source = 'original'
                            break

        # Step 2: 加载并转灰度
        img = Image.open(actual_path).convert('RGB')
        rgb = np.array(img, dtype=np.float64)
        h, w, _ = rgb.shape

        gray = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
        gray = np.clip(gray, 0, 255)

        # Step 3: 量化灰度级
        levels = int(INDICATOR.get('levels', 32))
        if levels < 2:
            levels = 2

        q = np.floor(gray / 256.0 * levels).astype(np.int32)
        q[q == levels] = levels - 1

        # Step 4: 构建GLCM并计算对比度
        d = int(INDICATOR.get('distance', 1))
        angles = INDICATOR.get('angles', [0, 45, 90, 135])

        def _offset_for_angle(deg: int) -> tuple:
            if deg == 0:
                return (0, d)
            if deg == 45:
                return (-d, d)
            if deg == 90:
                return (-d, 0)
            if deg == 135:
                return (-d, -d)
            return (0, d)

        def _glcm_contrast(q_img: np.ndarray, dy: int, dx: int, levels: int) -> float:
            H, W = q_img.shape
            glcm = np.zeros((levels, levels), dtype=np.float64)

            if dy >= 0:
                y1a, y1b = 0, H - dy
                y2a, y2b = dy, H
            else:
                y1a, y1b = -dy, H
                y2a, y2b = 0, H + dy

            if dx >= 0:
                x1a, x1b = 0, W - dx
                x2a, x2b = dx, W
            else:
                x1a, x1b = -dx, W
                x2a, x2b = 0, W + dx

            a = q_img[y1a:y1b, x1a:x1b].ravel()
            b = q_img[y2a:y2b, x2a:x2b].ravel()

            if a.size == 0:
                return 0.0

            idx = a * levels + b
            counts = np.bincount(idx, minlength=levels * levels).astype(np.float64)
            glcm = counts.reshape(levels, levels)

            total = glcm.sum()
            if total <= 0:
                return 0.0

            P = glcm / total

            i = np.arange(levels).reshape(-1, 1)
            j = np.arange(levels).reshape(1, -1)
            contrast = np.sum(((i - j) ** 2) * P)
            return float(contrast)

        per_angle = {}
        values = []

        for ang in angles:
            dy, dx = _offset_for_angle(int(ang))
            c = _glcm_contrast(q, dy, dx, levels)
            per_angle[str(ang)] = round(float(c), 3)
            values.append(c)

        mean_contrast = float(np.mean(values)) if len(values) > 0 else 0.0

        return {
            'success': True,
            'value': round(mean_contrast, 3),
            'levels': levels,
            'distance': d,
            'angles': angles,
            'per_angle_contrast': per_angle,
            'dimensions': {'height': int(h), 'width': int(w)},
            'image_source': image_source,
            'actual_path': actual_path
        }

    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'value': None
        }


# =============================================================================
# 辅助函数：对比度解释
# =============================================================================
def interpret_grc_con(value: float) -> str:
    """
    解释GRC_CON的含义
    """
    if value < 1:
        return "Very low texture contrast: smooth grayscale surface"
    elif value < 5:
        return "Low texture contrast: subtle intensity differences"
    elif value < 15:
        return "Medium texture contrast: noticeable texture"
    else:
        return "High texture contrast: strong grayscale variations"


# =============================================================================
# 测试代码
# =============================================================================
if __name__ == "__main__":
    print("\n🧪 Testing Grayscale Contrast (GLCM) calculator...")

    # 平滑图像：低对比度纹理
    smooth = np.full((128, 128, 3), 128, dtype=np.uint8)

    # 棋盘格：高对比度纹理
    checker = np.zeros((128, 128, 3), dtype=np.uint8)
    block = 8
    for i in range(0, 128, block):
        for j in range(0, 128, block):
            val = 255 if ((i // block + j // block) % 2 == 0) else 0
            checker[i:i+block, j:j+block] = val

    for name, test_img in [('Smooth', smooth), ('Checker', checker)]:
        test_path = f'/tmp/test_grc_con_{name}.png'
        Image.fromarray(test_img).save(test_path)

        result = calculate_indicator(test_path)

        print(f"\n   {name}:")
        print(f"      GRC_CON: {result['value']}")
        print(f"      Per-angle: {result['per_angle_contrast']}")
        print(f"      Interpretation: {interpret_grc_con(result['value'])}")

        os.remove(test_path)
