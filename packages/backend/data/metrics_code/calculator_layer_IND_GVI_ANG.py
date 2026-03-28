"""
SceneRx Stage 2.5 - Calculator Layer
================================================
指标ID: IND_GVI_ANG
指标名称: Green View Index by Vertical Angle (分角度绿视率)
类型: TYPE A (ratio模式 / 分区统计)

说明:
在鱼眼（Orthographic / Fisheye）投影的天空图中，以画面中心（天顶/zenith）为圆心，
按垂直角（从天顶向地平线的夹角，0°→90°）划分多个纬向环带（latitudinal zones），
分别计算每个角度区间内的植被像素占比：
    Zone GVI = Vegetation Pixels in Zone / Total Pixels in Zone × 100

分区（degrees）:
0–22.5, 22.5–45, 45–67.5, 67.5–90

公式:
IND_GVI_ANG(zone) = (Sum(Veg Pixels in zone) / Sum(Total Pixels in zone)) × 100
"""

import numpy as np
from PIL import Image
from typing import Dict, List, Tuple

# semantic_colors 来自 input_layer.py（与其他指标文件保持一致）
from input_layer import semantic_colors


# =============================================================================
# 指标定义 - 【核心配置】
# =============================================================================
INDICATOR = {
    # 基本信息
    "id": "IND_GVI_ANG",
    "name": "Green View Index by Vertical Angle",
    "unit": "%",
    "formula": "GVI(zone) = (Sum(Veg Pixels in zone) / Sum(Total Pixels in zone)) × 100",
    "target_direction": "INCREASE",  # INCREASE / DECREASE / NEUTRAL
    "definition": "Proportion of vegetation pixels within specific vertical angular (latitudinal) zones of a fisheye-projected sky map centered on zenith.",
    "category": "CAT_CMP",

    # 配置（鱼眼分区）
    "projection": "Orthographic (Fisheye)",
    "zones_deg": [0.0, 22.5, 45.0, 67.5, 90.0],  # 4个区间：0-22.5, 22.5-45, 45-67.5, 67.5-90

    # 目标语义类别 - 【必须与 Excel 的 Name 列完全一致】
    "target_classes": [
        "grass",
        "tree",
        "plant;flora;plant;life",
    ]
}


# =============================================================================
# 构建颜色查找表 (从 input_layer.py 的 semantic_colors 获取)
# =============================================================================
TARGET_RGB = {}

print(f"\n🎯 Building color lookup for {INDICATOR['id']}:")
for class_name in INDICATOR.get('target_classes', []):
    if class_name in semantic_colors:
        rgb = semantic_colors[class_name]
        TARGET_RGB[rgb] = class_name
        print(f"   ✅ {class_name}: RGB{rgb}")
    else:
        print(f"   ⚠️ NOT FOUND: {class_name}")
        # 尝试部分匹配（与其他指标文件保持一致）
        for name in semantic_colors.keys():
            if class_name.split(';')[0] in name or name.split(';')[0] in class_name:
                print(f"      💡 Did you mean: '{name}'?")
                break

print(f"\n✅ Calculator ready: {INDICATOR['id']} ({len(TARGET_RGB)} classes matched)")


# =============================================================================
# 工具函数：角度分区与鱼眼角度计算（Orthographic projection）
# =============================================================================
def _build_zone_pairs(z_edges: List[float]) -> List[Tuple[float, float]]:
    """把 [0,22.5,45,67.5,90] 转成 [(0,22.5),(22.5,45),(45,67.5),(67.5,90)]"""
    pairs = []
    for i in range(len(z_edges) - 1):
        pairs.append((float(z_edges[i]), float(z_edges[i + 1])))
    return pairs


def _compute_zenith_angles_deg(h: int, w: int) -> np.ndarray:
    """
    计算每个像素相对于天顶（图像中心）的垂直角 theta（单位：deg），范围 [0, 90]。
    Orthographic fisheye: r = R * sin(theta)  => theta = arcsin(r/R)
    - center = (w-1)/2, (h-1)/2
    - R 取到图像边界的最小半径（保证在画面内）
    超出圆盘（r>R）的像素设为 NaN（不计入任何区间）。
    """
    cy = (h - 1) / 2.0
    cx = (w - 1) / 2.0
    R = min(cx, cy)

    yy, xx = np.indices((h, w))
    dx = xx - cx
    dy = yy - cy
    r = np.sqrt(dx * dx + dy * dy)

    # 圆盘外不参与统计
    inside = r <= R
    theta = np.full((h, w), np.nan, dtype=np.float32)

    # 避免浮点误差导致 r/R > 1
    rr = np.clip(r[inside] / R, 0.0, 1.0)
    theta[inside] = np.degrees(np.arcsin(rr)).astype(np.float32)
    return theta


def _zone_label(z0: float, z1: float) -> str:
    """区间标签"""
    return f"{z0:g}-{z1:g}"


# =============================================================================
# 计算函数
# =============================================================================
def calculate_indicator(image_path: str) -> Dict:
    """
    计算 IND_GVI_ANG (分角度绿视率)

    Returns:
        {
            'success': True/False,
            'value': dict { '0-22.5': float, '22.5-45': float, ... } (每个角度区间的GVI百分比),
            'zone_pixels': dict {zone: total_pixels_in_zone},
            'zone_target_pixels': dict {zone: veg_pixels_in_zone},
            'class_breakdown': dict (整体各类别像素数),
            'zone_class_breakdown': dict {zone: {class: count}},
            'meta': dict (projection, zones)
        }
    """
    try:
        # Step 1: 加载图片
        img = Image.open(image_path).convert('RGB')
        pixels = np.array(img)
        h, w, _ = pixels.shape

        # Step 2: 计算每个像素的垂直角 theta（deg），并构建分区
        theta_deg = _compute_zenith_angles_deg(h, w)
        zone_pairs = _build_zone_pairs(INDICATOR["zones_deg"])

        # Step 3: 预先计算植被掩膜（按RGB精确匹配）
        flat_pixels = pixels.reshape(-1, 3)
        veg_mask_flat = np.zeros(flat_pixels.shape[0], dtype=bool)

        class_counts = {}
        for rgb, class_name in TARGET_RGB.items():
            m = np.all(flat_pixels == rgb, axis=1)
            c = int(np.sum(m))
            if c > 0:
                class_counts[class_name] = c
                veg_mask_flat |= m

        veg_mask = veg_mask_flat.reshape(h, w)

        # Step 4: 分区统计
        zone_values = {}
        zone_pixels = {}
        zone_target_pixels = {}
        zone_class_breakdown = {}

        # 方便按类统计（每一类单独mask）
        class_masks = {}
        for rgb, class_name in TARGET_RGB.items():
            class_masks[class_name] = np.all(pixels == rgb, axis=-1)

        for z0, z1 in zone_pairs:
            label = _zone_label(z0, z1)

            # 注意：左闭右开，最后一个区间右端包含90（避免边界丢失）
            if z1 >= 90.0:
                in_zone = (theta_deg >= z0) & (theta_deg <= z1)
            else:
                in_zone = (theta_deg >= z0) & (theta_deg < z1)

            # 圆盘外 theta 为 NaN，会自动为 False
            total_in_zone = int(np.sum(in_zone))
            veg_in_zone = int(np.sum(veg_mask & in_zone))

            zone_pixels[label] = total_in_zone
            zone_target_pixels[label] = veg_in_zone

            val = (veg_in_zone / total_in_zone) * 100 if total_in_zone > 0 else 0.0
            zone_values[label] = round(float(val), 3)

            # 每区间内各植被类别分解
            zcb = {}
            for cls, cmask in class_masks.items():
                cnt = int(np.sum(cmask & in_zone))
                if cnt > 0:
                    zcb[cls] = cnt
            zone_class_breakdown[label] = zcb

        return {
            'success': True,
            'value': zone_values,
            'zone_pixels': zone_pixels,
            'zone_target_pixels': zone_target_pixels,
            'class_breakdown': class_counts,
            'zone_class_breakdown': zone_class_breakdown,
            'meta': {
                'projection': INDICATOR.get('projection'),
                'zones_deg': INDICATOR.get('zones_deg')
            }
        }

    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'value': None
        }


# =============================================================================
# 测试代码 (可选)
# =============================================================================
if __name__ == "__main__":
    print("\n🧪 Testing calculator...")

    # 创建一个简单的“鱼眼圆盘”测试图：中心草地，外圈树
    h, w = 300, 300
    test_img = np.zeros((h, w, 3), dtype=np.uint8)

    cy = (h - 1) / 2.0
    cx = (w - 1) / 2.0
    R = min(cx, cy)

    yy, xx = np.indices((h, w))
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    inside = r <= R

    # 默认把圆盘内设为 tree，中心小圆设为 grass
    if 'tree' in semantic_colors:
        test_img[inside] = semantic_colors['tree']

    if 'grass' in semantic_colors:
        center = r <= (0.35 * R)
        test_img[center] = semantic_colors['grass']

    # 保存测试图片
    test_path = '/tmp/test_gvi_ang.png'
    Image.fromarray(test_img).save(test_path)

    # 测试计算
    result = calculate_indicator(test_path)
    print(f"   Result: {result}")

    # 清理
    import os
    os.remove(test_path)
