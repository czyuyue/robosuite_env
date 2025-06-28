import copy
import pdb
from typing import List
import torch
import torch.nn as nn
from robosuite.modules.build import MODELS, build_model_from_cfg
from robosuite.modules.diffusion.pcd_transformer import TrajectoryInpaintingTransformer

from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler



class KPDiff(nn.Module):
    def __init__(self,
                 model: TrajectoryInpaintingTransformer,
                 loss_args=None,
                 encoder_args=None,
                 decoder_args=None,
                 noise_scheduler=None,
                 **kwargs):
        super().__init__()

        self.loss_args = loss_args
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.query_embedding = torch.nn.Parameter(torch.randn(1, 1, 6), requires_grad=True)
        self.encoder = build_model_from_cfg(encoder_args)
        self.decoder = build_model_from_cfg(decoder_args)

    def get_context(self, pos, feat, query, n_b, n_q, text_emb=None, return_all=False):
        query_pos = torch.concat([pos, query], axis=-2)                                       # (B, N+Q, 3)
        query_feat = torch.concat([feat, self.query_embedding.repeat(n_b, n_q, 1)], axis=-2)  # (B, N+Q, 6)
        # if self.clip_fusion in ['early', 'both']:
        #     query_feat = torch.concat([query_feat, text_emb.unsqueeze(-2).repeat(1, query_feat.shape[1], 1)], axis=-1)          # (B, N+Q, 6+F_aligned_dim)
        p, f = self.encoder({'pos': query_pos, 'x': query_feat.transpose(1, 2)})
        if self.decoder is not None:
            f = self.decoder(p, f).squeeze(-1)   # (B, F, N+Q)

        if return_all is True:
            return f.transpose(1, 2)                 # (B, N+Q, F)
        else: 
            return f[:, :, -n_q:].transpose(1, 2)    # (B, Q, F)

    def forward(self, pos, feat, dtraj, query_mask, pack=None):
        n_b = dtraj.shape[0] # batch size
        n_q = dtraj.shape[1] # number of keypoints
        n_t = dtraj.shape[2] # trajectory length
        
        keypoints = dtraj[:, :, 0, :] # (B, Q, 3)
        context = self.get_context(pos, feat, keypoints, n_b, n_q) # (B, Q, F)
        
        # 转换维度以匹配TrajectoryInpaintingTransformer的输入格式
        # dtraj: (B, Q, T, 3) -> (B, T, Q, 3)
        trajectory = dtraj.permute(0, 2, 1, 3)  # (B, T, Q, 3)
        
        # 创建轨迹掩码：1表示已知，0表示需要inpaint
        # query_mask: (B, Q) -> trajectory_mask: (B, T, Q, 1)
        non_query_mask = (1 - query_mask).unsqueeze(1).unsqueeze(-1)  # (B, 1, Q, 1)
        trajectory_mask = non_query_mask.expand(n_b, n_t, n_q, 1)  # (B, T, Q, 1)
        
        # 已知轨迹部分（掩码区域保留原值，查询区域置零）
        known_trajectory = trajectory * trajectory_mask  # (B, T, Q, 3)
        
        # 随机采样扩散时间步
        device = trajectory.device
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (n_b,), device=device
        ).long()
        
        # 生成噪声
        noise = torch.randn_like(trajectory)  # (B, T, Q, 3)
        
        # 使用DDPM调度器添加噪声
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise, timesteps
        )  # (B, T, Q, 3)
        
        # 调用TrajectoryInpaintingTransformer预测噪声
        # condition使用context特征: (B, Q, F)
        noise_pred = self.model(
            noisy_trajectory=noisy_trajectory,    # (B, T, Q, 3)
            timesteps=timesteps,                  # (B,)
            mask=trajectory_mask,                 # (B, T, Q, 1)
            known_trajectory=known_trajectory,    # (B, T, Q, 3)
            condition=context                     # (B, Q, F)
        )  # (B, T, Q, 3)
        
        # 计算损失 - 只在查询区域（需要预测的区域）
        query_mask_expanded = query_mask.unsqueeze(1).unsqueeze(-1).expand(n_b, n_t, n_q, 3)  # (B, T, Q, 3)
        
        loss = F.mse_loss(
            noise_pred * query_mask_expanded,     # 预测噪声（仅查询区域）
            noise * query_mask_expanded           # 真实噪声（仅查询区域）
        )
        
        return {
            'loss': loss,
            'noise_pred': noise_pred,
            'noise_target': noise,
            'trajectory_mask': trajectory_mask,
            'context': context
        }

    def inference(self, pos, feat, query, num_sample=10, trajectory_length=None, guidance_scale=1.0):
        """
        推理函数：生成关键点轨迹
        
        Args:
            pos: 点云位置 (B, N, 3)
            feat: 点云特征 (B, N, F)
            query: 查询关键点 (B, Q, 3)
            num_sample: 采样步数，默认使用全部步数
            trajectory_length: 轨迹长度，如果为None则使用模型默认值
            guidance_scale: 引导强度，用于条件生成
            
        Returns:
            生成的轨迹 (B, Q, T, 3)
        """
        device = pos.device
        n_b = pos.shape[0]
        n_q = query.shape[1]
        
        # 获取轨迹长度
        if trajectory_length is None:
            trajectory_length = self.model.T
        n_t = trajectory_length
        
        # 获取上下文特征作为条件
        context = self.get_context(pos, feat, query, n_b, n_q)  # (B, Q, F)
        
        # 设置采样调度器
        self.noise_scheduler.set_timesteps(num_sample, device=device)
        timesteps = self.noise_scheduler.timesteps
        
        # 从纯噪声开始
        trajectory = torch.randn(n_b, n_t, n_q, 3, device=device)  # (B, T, Q, 3)
        
        # 创建掩码：在推理时，所有位置都需要生成（全为查询区域）
        trajectory_mask = torch.zeros(n_b, n_t, n_q, 1, device=device)  # 全0表示全部需要生成
        known_trajectory = torch.zeros_like(trajectory)  # 没有已知轨迹
        
        # 逐步去噪
        for i, t in enumerate(timesteps):
            # 扩展时间步到batch维度
            t_batch = t.unsqueeze(0).repeat(n_b).to(device)
            
            # 预测噪声
            with torch.no_grad():
                noise_pred = self.model(
                    noisy_trajectory=trajectory,
                    timesteps=t_batch,
                    mask=trajectory_mask,
                    known_trajectory=known_trajectory,
                    condition=context
                )
                
                # 可选：添加无条件生成用于引导
                if guidance_scale != 1.0:
                    # 无条件预测（使用零条件）
                    noise_pred_uncond = self.model(
                        noisy_trajectory=trajectory,
                        timesteps=t_batch,
                        mask=trajectory_mask,
                        known_trajectory=known_trajectory,
                        condition=torch.zeros_like(context)
                    )
                    
                    # 引导公式：noise_pred = uncond + guidance_scale * (cond - uncond)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred - noise_pred_uncond)
            
            # 使用调度器更新轨迹
            trajectory = self.noise_scheduler.step(
                noise_pred, t, trajectory, return_dict=False
            )[0]
        
        # 转换回原始维度格式: (B, T, Q, 3) -> (B, Q, T, 3)
        trajectory = trajectory.permute(0, 2, 1, 3)
        
        return trajectory
    
    def inference_with_partial_trajectory(self, pos, feat, partial_trajectory, trajectory_mask, num_sample=10, guidance_scale=1.0):
        """
        基于部分已知轨迹的推理函数（轨迹修复）
        
        Args:
            pos: 点云位置 (B, N, 3)
            feat: 点云特征 (B, N, F)
            partial_trajectory: 部分轨迹 (B, Q, T, 3)
            trajectory_mask: 轨迹掩码 (B, Q, T, 1)，1表示已知，0表示需要生成
            num_sample: 采样步数
            guidance_scale: 引导强度
            
        Returns:
            修复后的完整轨迹 (B, Q, T, 3)
        """
        device = pos.device
        n_b, n_q, n_t, _ = partial_trajectory.shape
        
        # 获取初始关键点（使用第一帧）
        initial_keypoints = partial_trajectory[:, :, 0, :]  # (B, Q, 3)
        
        # 获取上下文特征
        context = self.get_context(pos, feat, initial_keypoints, n_b, n_q)  # (B, Q, F)
        
        # 转换维度格式
        trajectory = partial_trajectory.permute(0, 2, 1, 3)  # (B, T, Q, 3)
        mask = trajectory_mask.permute(0, 2, 1, 3)  # (B, T, Q, 1)
        
        # 已知轨迹部分
        known_trajectory = trajectory * mask
        
        # 设置采样调度器
        self.noise_scheduler.set_timesteps(num_sample, device=device)
        timesteps = self.noise_scheduler.timesteps
        
        # 在未知区域添加噪声
        noise = torch.randn_like(trajectory)
        # 只在未知区域(mask=0)添加噪声，已知区域保持不变
        unknown_mask = 1 - mask
        trajectory = trajectory * mask + noise * unknown_mask
        
        # 逐步去噪
        for i, t in enumerate(timesteps):
            t_batch = t.unsqueeze(0).repeat(n_b).to(device)
            
            with torch.no_grad():
                # 预测噪声
                noise_pred = self.model(
                    noisy_trajectory=trajectory,
                    timesteps=t_batch,
                    mask=mask,
                    known_trajectory=known_trajectory,
                    condition=context
                )
                
                # 条件引导
                if guidance_scale != 1.0:
                    noise_pred_uncond = self.model(
                        noisy_trajectory=trajectory,
                        timesteps=t_batch,
                        mask=mask,
                        known_trajectory=known_trajectory,
                        condition=torch.zeros_like(context)
                    )
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred - noise_pred_uncond)
            
            # 更新轨迹
            trajectory_new = self.noise_scheduler.step(
                noise_pred, t, trajectory, return_dict=False
            )[0]
            
            # 保持已知区域不变，只更新未知区域
            trajectory = trajectory_new * unknown_mask + known_trajectory * mask
        
        # 转换回原始维度格式
        trajectory = trajectory.permute(0, 2, 1, 3)  # (B, Q, T, 3)
        
        return trajectory

if __name__ == "__main__":
    # KPDiff模型使用示例
    import numpy as np
    from robosuite.modules.diffusion.pcd_transformer import create_trajectory_inpainting_model
    
    print("=== KPDiff模型使用示例 ===")
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 模型参数
    T = 50        # 轨迹时间步数
    Q = 8         # 关键点数量
    N = 1024      # 点云点数
    c_dim = 128   # 条件特征维度
    batch_size = 2
    
    print(f"轨迹长度: {T}, 关键点数量: {Q}, 点云点数: {N}")
    
    # 1. 创建TrajectoryInpaintingTransformer
    print("\n1. 创建TrajectoryInpaintingTransformer...")
    transformer_model = create_trajectory_inpainting_model(
        T=T, Q=Q, c_dim=c_dim
    ).to(device)
    
    # 2. 创建DDPMScheduler
    print("2. 创建DDPMScheduler...")
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="linear",
        prediction_type="epsilon"  # 预测噪声
    )
    
    # 3. 创建简单的encoder和decoder配置
    print("3. 创建encoder和decoder配置...")
    encoder_args = {
        'NAME': 'PointNet',  # 这需要根据你的实际encoder来配置
        # 其他encoder参数
    }
    decoder_args = {
        'NAME': 'MLP',  # 这需要根据你的实际decoder来配置
        # 其他decoder参数
    }
    
    # 由于没有具体的encoder/decoder实现，我们创建简单的替代品
    class SimpleEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv1d(6, 128, 1)
            
        def forward(self, data):
            pos = data['pos']  # (B, N+Q, 3)
            x = data['x']      # (B, 6, N+Q)
            feat = self.conv(x)  # (B, 128, N+Q)
            return pos, feat
    
    class SimpleDecoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = nn.Conv1d(128, 128, 1)
            
        def forward(self, pos, feat):
            return self.mlp(feat)
    
    # 4. 创建KPDiff模型
    print("4. 创建KPDiff模型...")
    kpdiff_model = KPDiff(
        model=transformer_model,
        loss_args=None,
        encoder_args=encoder_args,
        decoder_args=decoder_args,
        noise_scheduler=noise_scheduler
    ).to(device)
    
    # 替换encoder和decoder为简单版本
    kpdiff_model.encoder = SimpleEncoder().to(device)
    kpdiff_model.decoder = SimpleDecoder().to(device)
    
    print(f"KPDiff模型参数数量: {sum(p.numel() for p in kpdiff_model.parameters()):,}")
    
    # 5. 准备测试数据
    print("\n5. 准备测试数据...")
    
    # 点云数据
    pos = torch.randn(batch_size, N, 3).to(device)           # 点云位置
    feat = torch.randn(batch_size, N, 6).to(device)          # 点云特征
    
    # 关键点轨迹数据
    dtraj = torch.randn(batch_size, Q, T, 3).to(device)      # 完整轨迹
    query_mask = torch.zeros(batch_size, Q).to(device)       # 查询掩码
    query_mask[:, :Q//2] = 1  # 前一半是查询点，后一半是已知点
    
    print(f"点云形状: {pos.shape}, 特征形状: {feat.shape}")
    print(f"轨迹形状: {dtraj.shape}, 查询掩码形状: {query_mask.shape}")
    print(f"查询点比例: {query_mask.mean().item():.2%}")
    
    # 6. 训练示例
    print("\n6. 训练示例...")
    
    kpdiff_model.train()
    optimizer = torch.optim.AdamW(kpdiff_model.parameters(), lr=1e-4)
    
    for step in range(3):  # 训练3步作为演示
        optimizer.zero_grad()
        
        # 前向传播
        outputs = kpdiff_model(pos, feat, dtraj, query_mask)
        loss = outputs['loss']
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        print(f"训练步骤 {step+1}, 损失: {loss.item():.6f}")
    
    # 7. 推理示例
    print("\n7. 推理示例...")
    
    kpdiff_model.eval()
    
    # 7.1 完全生成
    print("7.1 完全生成轨迹...")
    query_points = dtraj[:, :, 0, :].clone()  # 使用初始关键点作为query
    
    with torch.no_grad():
        generated_trajectory = kpdiff_model.inference(
            pos=pos,
            feat=feat,
            query=query_points,
            num_sample=20,  # 使用20步采样
            trajectory_length=T
        )
    
    print(f"生成轨迹形状: {generated_trajectory.shape}")
    print(f"生成轨迹范围: [{generated_trajectory.min().item():.3f}, {generated_trajectory.max().item():.3f}]")
    
    # 7.2 轨迹修复
    print("\n7.2 轨迹修复...")
    
    # 创建部分轨迹（模拟缺失数据）
    partial_trajectory = dtraj.clone()
    trajectory_mask = torch.ones_like(dtraj[:, :, :, :1])  # (B, Q, T, 1)
    
    # 将中间部分设为未知
    mask_start = T // 3
    mask_end = 2 * T // 3
    trajectory_mask[:, :, mask_start:mask_end, :] = 0
    partial_trajectory[:, :, mask_start:mask_end, :] = 0
    
    print(f"掩码范围: {mask_start} 到 {mask_end}")
    print(f"已知比例: {trajectory_mask.mean().item():.2%}")
    
    with torch.no_grad():
        repaired_trajectory = kpdiff_model.inference_with_partial_trajectory(
            pos=pos,
            feat=feat,
            partial_trajectory=partial_trajectory,
            trajectory_mask=trajectory_mask,
            num_sample=20
        )
    
    print(f"修复轨迹形状: {repaired_trajectory.shape}")
    
    # 计算修复精度（在掩码区域）
    unknown_mask = (1 - trajectory_mask)
    repair_error = F.mse_loss(
        repaired_trajectory * unknown_mask,
        dtraj * unknown_mask
    )
    print(f"修复误差 (MSE): {repair_error.item():.6f}")
    
    # 8. 模型保存示例
    print("\n8. 模型保存示例...")
    
    # 保存模型状态
    model_state = {
        'kpdiff_state_dict': kpdiff_model.state_dict(),
        'transformer_config': {
            'T': T,
            'Q': Q,
            'c_dim': c_dim,
            'd_model': 256,
            'nhead': 8,
            'num_layers': 6
        },
        'scheduler_config': noise_scheduler.config,
        'optimizer_state_dict': optimizer.state_dict()
    }
    
    # torch.save(model_state, 'kpdiff_model.pth')
    print("模型状态已准备保存（已注释掉实际保存操作）")
    
    print("\n=== 示例完成 ===")
    print("\n使用建议：")
    print("1. 根据具体任务调整T、Q、c_dim等参数")
    print("2. 实现适合的encoder和decoder")
    print("3. 准备合适的点云和轨迹数据")
    print("4. 调整采样步数和引导强度以获得最佳效果")
    print("5. 在训练时使用合适的数据增强和正则化")
