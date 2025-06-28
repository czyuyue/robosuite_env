"""
轨迹修复Transformer模型使用示例 - 带条件特征
本示例展示如何使用修改后的TrajectoryInpaintingTransformer，支持每点条件特征
"""

import torch
import torch.nn as nn
from pcd_transformer import (
    create_trajectory_inpainting_model, 
    DDPMScheduler, 
    training_step, 
    generate_trajectory_mask
)

class ConditionFeatureExtractor(nn.Module):
    """
    示例条件特征提取器
    根据你的具体任务，这可以是任何特征提取网络
    """
    def __init__(self, input_dim=3, output_dim=64):
        super().__init__()
        self.feature_net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
            nn.LayerNorm(output_dim)
        )
    
    def forward(self, point_cloud):
        """
        Args:
            point_cloud: [B, Q, 3] 输入点云
        Returns:
            [B, Q, output_dim] 每个点的条件特征
        """
        return self.feature_net(point_cloud)

def example_usage():
    """完整的使用示例"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 设置参数
    T = 50      # 轨迹时间步数
    Q = 256     # 点云点数
    c_dim = 64  # 条件特征维度
    batch_size = 2
    
    # 1. 创建带条件的模型
    print("1. Creating model with condition support...")
    model = create_trajectory_inpainting_model(T=T, Q=Q, c_dim=c_dim).to(device)
    scheduler = DDPMScheduler()
    condition_extractor = ConditionFeatureExtractor(input_dim=3, output_dim=c_dim).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 2. 准备数据
    print("\n2. Preparing data...")
    
    # 模拟完整轨迹数据
    full_trajectory = torch.randn(batch_size, T, Q, 3).to(device)
    
    # 生成掩码（哪些部分需要修复）
    mask = generate_trajectory_mask(T, Q, mask_ratio=0.4, temporal_coherent=True)
    mask = mask.unsqueeze(0).repeat(batch_size, 1, 1, 1).to(device)
    
    # 已知轨迹部分（掩码区域置零）
    known_trajectory = full_trajectory * mask
    
    # 提取条件特征（基于第一帧点云）
    first_frame = full_trajectory[:, 0, :, :]  # [B, Q, 3]
    condition_features = condition_extractor(first_frame)  # [B, Q, c_dim]
    
    print(f"Trajectory shape: {full_trajectory.shape}")
    print(f"Mask shape (1=known, 0=inpaint): {mask.shape}")
    print(f"Condition features shape: {condition_features.shape}")
    print(f"Known ratio: {mask.float().mean().item():.2%}")
    
    # 3. 训练示例
    print("\n3. Training example...")
    
    model.train()
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(condition_extractor.parameters()), 
        lr=1e-4
    )
    
    for step in range(5):  # 简单训练5步作为示例
        optimizer.zero_grad()
        
        # 重新提取条件特征（如果需要梯度更新）
        condition_features = condition_extractor(first_frame)
        
        # 准备训练数据
        batch = {
            'trajectory': full_trajectory,
            'mask': mask,
            'known_trajectory': known_trajectory,
            'condition': condition_features
        }
        
        # 计算损失
        loss = training_step(model, scheduler, batch, device)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        print(f"Step {step+1}, Loss: {loss.item():.6f}")
    
    # 4. 推理示例
    print("\n4. Inference example...")
    
    model.eval()
    with torch.no_grad():
        # 模拟推理过程（简化版DDPM采样）
        # 这里只展示单步推理，完整的采样需要多步去噪
        
        # 添加噪声到轨迹
        timestep = torch.randint(100, 200, (batch_size,)).to(device)
        noise = torch.randn_like(full_trajectory)
        noisy_trajectory = scheduler.add_noise(full_trajectory, noise, timestep)
        
        # 模型预测噪声
        predicted_noise = model(
            noisy_trajectory, 
            timestep, 
            mask, 
            known_trajectory, 
            condition_features
        )
        
        print(f"Predicted noise shape: {predicted_noise.shape}")
        print(f"Noise prediction MSE: {nn.MSELoss()(predicted_noise, noise).item():.6f}")
    
    # 5. 不同条件的对比
    print("\n5. Testing different conditions...")
    
    with torch.no_grad():
        # 创建不同的条件特征
        condition_A = torch.randn(batch_size, Q, c_dim).to(device)  # 条件A
        condition_B = torch.zeros(batch_size, Q, c_dim).to(device)  # 条件B（零向量）
        
        # 使用相同的噪声轨迹和时间步
        test_timestep = torch.full((batch_size,), 500).to(device)
        test_noisy = scheduler.add_noise(full_trajectory, noise, test_timestep)
        
        # 不同条件下的预测
        pred_A = model(test_noisy, test_timestep, mask, known_trajectory, condition_A)
        pred_B = model(test_noisy, test_timestep, mask, known_trajectory, condition_B)
        
        # 计算预测差异
        diff = torch.norm(pred_A - pred_B, dim=-1).mean()
        print(f"Prediction difference between conditions: {diff.item():.6f}")
        print("Note: Larger difference indicates stronger condition influence")

def integration_tips():
    """集成建议"""
    print("""
    
=== 集成建议 ===

1. 条件特征设计：
   - c_dim建议设置为64-256之间
   - 条件特征应该包含对轨迹修复有用的信息
   - 可以是几何特征、语义特征、任务相关特征等

2. 训练策略：
   - 可以先预训练条件特征提取器
   - 然后联合训练或分阶段训练
   - 使用不同的学习率可能有帮助

3. 条件特征示例：
   - 点云法向量、曲率等几何特征
   - 点云语义分割结果
   - 机器人任务相关的特征（如可操作性、距离障碍物等）
   - 历史轨迹统计特征

4. 推理时：
   - 确保条件特征与训练时保持一致
   - 可以通过调整条件特征来控制生成结果
   - 条件特征的质量直接影响修复效果

5. 兼容性：
   - 设置c_dim=0可以禁用条件特征
   - 向后兼容原有的无条件模型
   - 可以在推理时传入None来使用零条件
    """)

if __name__ == "__main__":
    # 运行示例
    example_usage()
    
    # 显示集成建议
    integration_tips() 