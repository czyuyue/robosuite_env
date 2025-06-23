from curses import keyname
import numpy as np
import torch
import os
import os.path as osp
import ssl
import pdb
import sys
import urllib
import h5py
from typing import Optional, Union, Tuple
import subprocess
import tempfile
from PIL import Image
import glob
def read_16bit_avi(avi_path):
    """
    Reads a 16-bit grayscale AVI file using ffmpeg to preserve uint16 format.
    Returns a numpy array of shape (num_frames, H, W) with dtype uint16.
    """
    if not os.path.exists(avi_path):
        raise IOError(f"Cannot find video file: {avi_path}")
    
    # Create temporary directory for extracted frames
    with tempfile.TemporaryDirectory() as temp_dir:
        # Use ffmpeg to extract frames as 16-bit PNG files
        frame_pattern = os.path.join(temp_dir, "frame_%04d.png")
        cmd = [
            "ffmpeg", 
            "-i", avi_path,
            "-pix_fmt", "gray16le",
            "-vsync", "0",
            frame_pattern,
            "-y"  # Overwrite output files
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"ffmpeg failed: {e.stderr}")
        except FileNotFoundError:
            raise RuntimeError("ffmpeg not found. Please install ffmpeg.")
        
        # Get list of extracted frame files
        frame_files = sorted(glob.glob(os.path.join(temp_dir, "frame_*.png")))
        
        if len(frame_files) == 0:
            raise RuntimeError("No frames extracted from video.")
        
        # Load frames into numpy array
        frames = []
        for frame_file in frame_files:
            # Load 16-bit PNG using PIL
            img = Image.open(frame_file)
            # Convert to numpy array and ensure uint16 dtype
            frame = np.array(img, dtype=np.uint16)
            # print(f"Frame shape: {frame.shape}, dtype: {frame.dtype}, min: {frame.min()}, max: {frame.max()}")
            frames.append(frame)
        
        return np.stack(frames, axis=0)
def voxel_downsample(
    points: np.ndarray,
    voxel_size: Union[float, np.ndarray],
    method: str = "centroid",
    return_indices: bool = False,
    features: Optional[np.ndarray] = None
) -> Union[np.ndarray, Tuple[np.ndarray, ...]]:
    """
    对点云进行体素下采样
    
    Args:
        points: 输入点云，形状为 (N, 3) 或 (N, D) 其中D>=3
        voxel_size: 体素大小，可以是标量或长度为3的数组 [x_size, y_size, z_size]
        method: 下采样方法
            - "centroid": 使用每个体素内点的重心 (默认)
            - "random": 从每个体素随机选择一个点
            - "average": 对每个体素内的所有点求平均
        return_indices: 是否返回原始点云中被选中点的索引
        features: 可选的特征数组，形状为 (N, F)，会根据选择的点进行相应的处理
        
    Returns:
        如果 return_indices=False:
            - downsampled_points: 下采样后的点云
            - downsampled_features: 如果提供了features，返回对应的特征
        如果 return_indices=True:
            额外返回 selected_indices: 被选中点的原始索引
    """
    if points.shape[0] == 0:
        return points if features is None else (points, features)
    
    # 确保输入是numpy数组
    points = np.asarray(points)
    if features is not None:
        features = np.asarray(features)
        assert features.shape[0] == points.shape[0], "点数和特征数必须相同"
    
    # 处理voxel_size参数
    if np.isscalar(voxel_size):
        voxel_size = np.array([voxel_size, voxel_size, voxel_size])
    else:
        voxel_size = np.asarray(voxel_size)
        if voxel_size.shape[0] != 3:
            raise ValueError("voxel_size必须是标量或长度为3的数组")
    
    # 计算体素坐标
    xyz = points[:, :3]
    min_coords = np.min(xyz, axis=0)
    voxel_coords = np.floor((xyz - min_coords) / voxel_size).astype(np.int32)
    
    # 创建体素索引
    # 使用字典来存储每个体素中的点
    voxel_dict = {}
    for i, voxel_coord in enumerate(voxel_coords):
        key = tuple(voxel_coord)
        if key not in voxel_dict:
            voxel_dict[key] = []
        voxel_dict[key].append(i)
    
    # 根据不同方法处理每个体素
    selected_indices = []
    downsampled_points = []
    downsampled_features = [] if features is not None else None
    
    for voxel_indices in voxel_dict.values():
        voxel_points = points[voxel_indices]
        
        if method == "centroid":
            # 计算重心
            centroid = np.mean(voxel_points, axis=0)
            downsampled_points.append(centroid)
            # 找到最接近重心的点的索引
            distances = np.linalg.norm(voxel_points[:, :3] - centroid[:3], axis=1)
            closest_idx = voxel_indices[np.argmin(distances)]
            selected_indices.append(closest_idx)
            
            if features is not None:
                # 对特征也计算平均值
                avg_features = np.mean(features[voxel_indices], axis=0)
                downsampled_features.append(avg_features)
                
        elif method == "random":
            # 随机选择一个点
            random_idx = np.random.choice(len(voxel_indices))
            selected_idx = voxel_indices[random_idx]
            selected_indices.append(selected_idx)
            downsampled_points.append(points[selected_idx])
            
            if features is not None:
                downsampled_features.append(features[selected_idx])
                
        elif method == "average":
            # 计算平均值
            averaged_point = np.mean(voxel_points, axis=0)
            downsampled_points.append(averaged_point)
            # 对于average方法，选择第一个点的索引作为代表
            selected_indices.append(voxel_indices[0])
            
            if features is not None:
                avg_features = np.mean(features[voxel_indices], axis=0)
                downsampled_features.append(avg_features)
                
        else:
            raise ValueError(f"不支持的方法: {method}. 支持的方法: 'centroid', 'random', 'average'")
    
    # 转换为numpy数组
    downsampled_points = np.array(downsampled_points) if downsampled_points else np.empty((0, points.shape[1]))
    selected_indices = np.array(selected_indices) if selected_indices else np.array([], dtype=int)
    
    if downsampled_features is not None:
        downsampled_features = np.array(downsampled_features) if downsampled_features else np.empty((0, features.shape[1]))
    
    # 构建返回值
    results = [downsampled_points]
    if features is not None:
        results.append(downsampled_features)
    if return_indices:
        results.append(selected_indices)
    
    return results[0] if len(results) == 1 else tuple(results)


def adaptive_voxel_downsample(
    points: np.ndarray,
    target_num_points: int,
    min_voxel_size: float = 0.001,
    max_voxel_size: float = 1.0,
    max_iterations: int = 20
) -> Tuple[np.ndarray, float]:
    """
    自适应体素下采样，自动调整体素大小以达到目标点数
    
    Args:
        points: 输入点云，形状为 (N, 3) 或 (N, D)
        target_num_points: 目标点数
        min_voxel_size: 最小体素大小
        max_voxel_size: 最大体素大小
        max_iterations: 最大迭代次数
        
    Returns:
        (downsampled_points, final_voxel_size): 下采样后的点云和最终使用的体素大小
    """
    if points.shape[0] <= target_num_points:
        return points, min_voxel_size
    
    low, high = min_voxel_size, max_voxel_size
    best_points = points
    best_voxel_size = min_voxel_size
    
    for _ in range(max_iterations):
        mid_voxel_size = (low + high) / 2
        downsampled = voxel_downsample(points, mid_voxel_size, method="centroid")
        num_points = len(downsampled)
        
        if abs(num_points - target_num_points) < target_num_points * 0.1:  # 10%的容差
            return downsampled, mid_voxel_size
        elif num_points > target_num_points:
            low = mid_voxel_size
        else:
            high = mid_voxel_size
            best_points = downsampled
            best_voxel_size = mid_voxel_size
    
    return best_points, best_voxel_size


def uniform_voxel_grid_downsample(
    points: np.ndarray,
    grid_size: Union[int, Tuple[int, int, int]],
    method: str = "centroid"
) -> np.ndarray:
    """
    使用均匀网格进行体素下采样
    
    Args:
        points: 输入点云，形状为 (N, 3) 或 (N, D)
        grid_size: 网格大小，可以是整数（立方体网格）或三元组 (nx, ny, nz)
        method: 下采样方法，同 voxel_downsample
        
    Returns:
        下采样后的点云
    """
    if points.shape[0] == 0:
        return points
    
    xyz = points[:, :3]
    min_coords = np.min(xyz, axis=0)
    max_coords = np.max(xyz, axis=0)
    
    if isinstance(grid_size, int):
        grid_size = (grid_size, grid_size, grid_size)
    
    # 计算每个维度的体素大小
    ranges = max_coords - min_coords
    voxel_size = ranges / np.array(grid_size)
    
    return voxel_downsample(points, voxel_size, method=method)


# 使用示例函数
def demo_voxel_downsampling():
    """
    体素下采样的使用示例
    """
    # 创建示例点云
    np.random.seed(42)
    points = np.random.rand(10000, 3) * 10  # 10000个随机点在[0,10]^3范围内
    colors = np.random.rand(10000, 3)  # 对应的颜色特征
    
    print(f"原始点云: {points.shape[0]} 个点")
    
    # 基本体素下采样
    downsampled1 = voxel_downsample(points, voxel_size=0.5, method="centroid")
    print(f"体素大小0.5下采样后: {downsampled1.shape[0]} 个点")
    
    # 带特征的下采样
    downsampled2, downsampled_colors = voxel_downsample(
        points, voxel_size=0.5, method="centroid", features=colors
    )
    print(f"带颜色特征的下采样: {downsampled2.shape[0]} 个点, {downsampled_colors.shape[0]} 个颜色")
    
    # 自适应下采样
    adaptive_points, used_voxel_size = adaptive_voxel_downsample(points, target_num_points=1000)
    print(f"自适应下采样到1000个点: {adaptive_points.shape[0]} 个点, 体素大小: {used_voxel_size:.4f}")
    
    # 均匀网格下采样
    grid_points = uniform_voxel_grid_downsample(points, grid_size=20)
    print(f"20x20x20网格下采样: {grid_points.shape[0]} 个点")


if __name__ == "__main__":
    demo_voxel_downsampling()

