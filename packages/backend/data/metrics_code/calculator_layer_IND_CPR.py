"""
SceneRx Stage 2.5 - Calculator Layer
================================================
指标ID: IND_CPR
指标名称: Compression Ratio (图像压缩比)
类型: TYPE C (图像处理类)

说明:
通过计算图像在JPEG压缩后的文件大小与原始未压缩图像大小之比，
作为图像信息密度（information density）的代理指标。
压缩比越高，通常表示图像中可压缩冗余较少、结构/纹理更复杂；
压缩比越低，表示图像更规则、信息冗余度更高。

公式:
Compression Ratio = Size_compressed / Size_original

其中:
- Size_compressed: JPEG压缩后的图像文件大小
- Size_original: 原始未压缩图像大小（以RGB原始数据量估计）

单位: 无量纲
范围: (0, 1]（实际应用中通常远小于1）
"""

import numpy as np
from PIL import Image
from typing import Dict
import os
import tempfile


# =============================================================================
# 指标定义
# =============================================================================
INDICATOR = {
    "id": "IND_CPR",
    "name": "Compression Ratio",
    "unit": "ratio",
    "formula": "Compression Ratio = Size_compressed / Size_original",
    "target_direction": "NEUTRAL",
    "definition": "Ratio between compressed JPEG size and original uncompressed image size as a proxy for information density",
    "category": "CAT_CMP",

    "calc_type": "custom",

    "variables": {
        "Size_{compressed}": "Size of the compressed image (JPEG)",
        "Size_{original}": "Original uncompressed image size (RGB)"
    },

    # TYPE C 特殊配置
    "jpeg_quality": 75  # JPEG压缩质量（1-95，数值越高压缩越弱）
}

print(f"\n✅ Calculator ready: {INDICATOR['id']} - {INDICATOR['name']}")
print(f"   Formula: {INDICATOR['formula']}")
print(f"   JPEG quality: {INDICATOR.get('jpeg_quality')}")


# =============================================================================
# 计算函数
# =============================================================================
def calculate_indicator(image_path: str) -> Dict:
    """
    计算 Compression Ratio (图像压缩比)

    TYPE C: 图像处理类

    算法步骤:
    1. 加载原始图像
    2. 估算未压缩图像大小（RGB: height × width × 3 bytes）
    3. 以指定JPEG质量进行压缩并保存为临时文件
    4. 读取压缩后文件大小
    5. 计算 Compression Ratio

    Args:
        image_path: 原始图像路径

    Returns:
        {
            'success': True/False,
            'value': float (Compression Ratio),
            'size_original_bytes': int,
            'size_compressed_bytes': int,
            'jpeg_quality': int,
            'dimensions': dict
        }
    """
    try:
        # Step 1: 加载图像
        img = Image.open(image_path).convert('RGB')
        pixels = np.array(img)

        h, w, c = pixels.shape

        # Step 2: 原始未压缩大小（RGB，1 byte/channel）
        size_original = h * w * c  # bytes

        # Step 3: JPEG压缩（临时文件）
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            tmp_path = tmp.name

        img.save(
            tmp_path,
            format='JPEG',
            quality=INDICATOR.get('jpeg_quality', 75),
            optimize=True
        )

        # Step 4: 读取压缩后大小
        size_compressed = os.path.getsize(tmp_path)

        # Step 5: 计算压缩比
        compression_ratio = size_compressed / size_original if size_original > 0 else 0

        # 清理临时文件
        os.remove(tmp_path)

        return {
            'success': True,
            'value': round(float(compression_ratio), 4),
            'size_original_bytes': int(size_original),
            'size_compressed_bytes': int(size_compressed),
            'jpeg_quality': INDICATOR.get('jpeg_quality'),
            'dimensions': {'height': h, 'width': w}
        }

    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'value': None
        }


# =============================================================================
# 辅助函数：压缩比解释
# =============================================================================
def interpret_cpr(value: float) -> str:
    """
    解释Compression Ratio的含义
    """
    if value < 0.05:
        return "Very low ratio: highly regular or smooth image"
    elif value < 0.10:
        return "Low ratio: relatively simple visual structure"
    elif value < 0.20:
        return "Medium ratio: moderate information density"
    else:
        return "High ratio: complex texture and rich visual information"


# =============================================================================
# 测试代码
# =============================================================================
if __name__ == "__main__":
    print("\n🧪 Testing Compression Ratio calculator...")

    # 创建简单图像（低信息密度）
    simple_img = np.full((200, 200, 3), 128, dtype=np.uint8)

    # 创建复杂图像（高信息密度）
    complex_img = np.random.randint(0, 256, (200, 200, 3), dtype=np.uint8)

    for name, test_img in [('Simple', simple_img), ('Complex', complex_img)]:
        test_path = f'/tmp/test_cpr_{name}.png'
        Image.fromarray(test_img).save(test_path)

        result = calculate_indicator(test_path)

        print(f"\n   {name}:")
        print(f"      Compression Ratio: {result['value']}")
        print(f"      Interpretation: {interpret_cpr(result['value'])}")

        os.remove(test_path)
