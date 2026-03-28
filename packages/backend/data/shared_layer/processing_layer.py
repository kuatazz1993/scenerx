"""
SceneRx Stage 2.5 - PROCESSING LAYER（处理层）
================================================
🔒 完全统一，所有指标共用，无需修改

更新: 新增 image_id 推导和 lat/lng 元数据注入

功能:
1. process_zone() - 处理单个区域的所有图片
2. calculate_statistics() - 计算描述性统计量
3. Main Processing Loop - 遍历所有区域并处理

依赖变量（来自 INPUT LAYER）:
- query_data: 项目和区域信息
- zone_image_map: 各区域各图层的图片列表
- image_metadata: {image_id: {lat, lng, ...}} 图片元数据（可选）
- PATHS: 路径配置
- LAYERS: 图层列表

依赖变量（来自 CALCULATOR LAYER）:
- INDICATOR: 指标定义字典
- calculate_indicator(): 指标计算函数

输出变量（供 OUTPUT LAYER 使用）:
- all_zone_results: 所有区域的处理结果
- all_values: 所有有效值的列表
- all_values_by_layer: 按图层分组的值列表
"""

import os
import numpy as np
from typing import Dict, List, Any


# =============================================================================
# 1. PROCESS ZONE FUNCTION
# =============================================================================
def process_zone(zone: Dict, zone_images: Dict[str, List[str]], 
                 base_path: str, calculator_func,
                 image_metadata: Dict[str, Dict] = None) -> Dict:
    """
    处理单个区域的所有图片。
    
    遍历该区域所有图层的所有图片，调用calculator_func计算指标值，
    并汇总统计结果。
    
    Args:
        zone: 区域字典，包含 id, name, area_sqm, status
        zone_images: {layer: [filenames]}，该区域各图层的图片列表
        base_path: mask文件夹根路径
        calculator_func: calculate_indicator 函数引用
        image_metadata: {image_id: {lat, lng, ...}} 图片元数据（可选）
        
    Returns:
        区域处理结果字典，包含：
        - zone_id, zone_name, area_sqm, status: 区域基本信息
        - layers: 各图层的详细结果
        - all_values: 该区域所有有效值
        - values_by_layer: 按图层分组的值
        - images_processed: 成功处理的图片数
        - images_failed: 处理失败的图片数
        - images_no_data: 无有效数据的图片数
        
    Example:
        >>> result = process_zone(zone, zone_images, '/path/to/mask', 
        ...                       calculate_indicator, image_metadata)
        >>> print(f"Processed {result['images_processed']} images")
    """
    if image_metadata is None:
        image_metadata = {}
    
    zone_id = zone['id']
    
    results = {
        'zone_id': zone_id,
        'zone_name': zone['name'],
        'area_sqm': zone.get('area_sqm', 0),
        'status': zone.get('status', 'unknown'),
        'layers': {},
        'all_values': [],
        'values_by_layer': {},
        'images_processed': 0,
        'images_failed': 0,
        'images_no_data': 0
    }
    
    for layer, filenames in zone_images.items():
        layer_results = {
            'images': [],
            'values': [],
            'statistics': {}
        }
        
        for filename in filenames:
            image_path = os.path.join(base_path, zone_id, layer, filename)
            
            # 从文件名推导 image_id（去掉扩展名）
            image_id = os.path.splitext(filename)[0]
            
            # 调用 CALCULATOR 层的计算函数
            result = calculator_func(image_path)
            
            if result['success']:
                # 构建图片结果：以 image_id 为首要标识
                image_data = {
                    'image_id': image_id,
                    'filename': filename,
                    'value': result['value']
                }
                
                # 注入经纬度等元数据（如果有）
                meta = image_metadata.get(image_id, {})
                for meta_key, meta_val in meta.items():
                    image_data[meta_key] = meta_val
                
                # 添加 calculator 返回的额外字段
                # （如 target_pixels, total_pixels, class_breakdown 等）
                for key, val in result.items():
                    if key not in ['success', 'value', 'error']:
                        image_data[key] = val
                
                layer_results['images'].append(image_data)
                
                # 收集有效值（value 不为 None）
                if result['value'] is not None:
                    layer_results['values'].append(result['value'])
                    results['all_values'].append(result['value'])
                else:
                    results['images_no_data'] += 1
                
                results['images_processed'] += 1
            else:
                results['images_failed'] += 1
        
        # 计算该图层的统计量
        if layer_results['values']:
            arr = np.array(layer_results['values'])
            layer_results['statistics'] = {
                'N': len(arr),
                'Mean': round(float(np.mean(arr)), 3),
                'Std': round(float(np.std(arr)), 3),
                'Min': round(float(np.min(arr)), 3),
                'Max': round(float(np.max(arr)), 3),
                'Median': round(float(np.median(arr)), 3)
            }
        
        results['layers'][layer] = layer_results
        results['values_by_layer'][layer] = layer_results['values']
    
    return results


# =============================================================================
# 2. CALCULATE STATISTICS FUNCTION
# =============================================================================
def calculate_statistics(values: List[float]) -> Dict:
    """
    计算描述性统计量。
    
    Args:
        values: 数值列表
        
    Returns:
        统计量字典，包含：
        - N: 样本数量
        - Mean: 均值
        - Std: 标准差
        - Min, Q1, Median, Q3, Max: 五数概括
        - Range, IQR: 极差和四分位距
        - Variance: 方差
        - CV(%): 变异系数
        
    Example:
        >>> stats = calculate_statistics([10, 20, 30, 40, 50])
        >>> print(f"Mean: {stats['Mean']}, Std: {stats['Std']}")
    """
    if not values:
        return {}
    
    arr = np.array(values)
    q1, q3 = np.percentile(arr, 25), np.percentile(arr, 75)
    mean_val = float(np.mean(arr))
    std_val = float(np.std(arr))
    
    return {
        'N': len(values),
        'Mean': round(mean_val, 3),
        'Std': round(std_val, 3),
        'Min': round(float(np.min(arr)), 3),
        'Q1': round(float(q1), 3),
        'Median': round(float(np.median(arr)), 3),
        'Q3': round(float(q3), 3),
        'Max': round(float(np.max(arr)), 3),
        'Range': round(float(np.max(arr) - np.min(arr)), 3),
        'IQR': round(float(q3 - q1), 3),
        'Variance': round(float(np.var(arr)), 3),
        'CV(%)': round(float(std_val / mean_val * 100), 2) if mean_val != 0 else 0
    }


# =============================================================================
# 3. MAIN PROCESSING LOOP
# =============================================================================
print("\n" + "=" * 70)
print(f"🔄 PROCESSING: {INDICATOR['id']} - {INDICATOR['name']}")
print(f"   Layers: {', '.join(LAYERS)}")
if image_metadata:
    print(f"   Image metadata: {len(image_metadata)} entries loaded")
else:
    print(f"   Image metadata: not available (output will not include lat/lng)")
print("=" * 70)

# 初始化结果容器
all_zone_results = []
all_values = []
all_values_by_layer = {layer: [] for layer in LAYERS}

# 遍历所有区域
for zone in query_data['zones']:
    zone_id = zone['id']
    zone_images = zone_image_map.get(zone_id, {})
    total_zone_images = sum(len(files) for files in zone_images.values())
    
    print(f"\n📄 Processing: {zone['name']} ({total_zone_images} images)...")
    
    # 处理该区域（传入 image_metadata）
    result = process_zone(
        zone=zone,
        zone_images=zone_images,
        base_path=PATHS['image_base_path'],
        calculator_func=calculate_indicator,
        image_metadata=image_metadata
    )
    
    all_zone_results.append(result)
    all_values.extend(result['all_values'])
    
    # 按图层收集值
    for layer in LAYERS:
        all_values_by_layer[layer].extend(result['values_by_layer'].get(layer, []))
    
    # 打印进度
    print(f"   ✅ Processed: {result['images_processed']}")
    if result['images_failed'] > 0:
        print(f"   ⚠️ Failed: {result['images_failed']}")
    if result['images_no_data'] > 0:
        print(f"   ℹ️ No data: {result['images_no_data']}")
    if result['all_values']:
        mean_val = np.mean(result['all_values'])
        print(f"   📊 Zone Mean: {mean_val:.2f}{INDICATOR['unit']}")

# 打印总结
print("\n" + "=" * 70)
total_processed = sum(r['images_processed'] for r in all_zone_results)
total_failed = sum(r['images_failed'] for r in all_zone_results)
total_no_data = sum(r.get('images_no_data', 0) for r in all_zone_results)

print(f"✅ PROCESSING COMPLETE")
print(f"   Total processed: {total_processed}")
if total_failed > 0:
    print(f"   Total failed: {total_failed}")
if total_no_data > 0:
    print(f"   Total no data: {total_no_data}")

if all_values:
    print(f"\n📊 OVERALL Mean: {np.mean(all_values):.2f}{INDICATOR['unit']}")

# 打印按图层统计
print(f"\n📊 BY LAYER:")
for layer in LAYERS:
    if all_values_by_layer[layer]:
        mean_val = np.mean(all_values_by_layer[layer])
        n_val = len(all_values_by_layer[layer])
        print(f"   {layer}: Mean={mean_val:.2f}{INDICATOR['unit']}, N={n_val}")
    else:
        print(f"   {layer}: No data")

print("=" * 70)
