import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class TemporalPositionEmbedding(nn.Module):
    """为时序数据添加位置编码"""
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [B, T, d_model]
        return x + self.pe[:x.size(1)].unsqueeze(0)

class PointCloudAttention(nn.Module):
    """点云内的自注意力机制"""
    def __init__(self, d_model, nhead=8, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.attention = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        # x: [B, T, Q, d_model]
        B, T, Q, d_model = x.shape
        
        # 重塑为 [B*T, Q, d_model] 对每个时间步的点云做attention
        x_flat = x.view(B * T, Q, d_model)
        attn_out, _ = self.attention(x_flat, x_flat, x_flat)
        attn_out = attn_out.view(B, T, Q, d_model)
        
        return self.norm(x + attn_out)

class TemporalAttention(nn.Module):
    """时间维度的自注意力机制"""
    def __init__(self, d_model, nhead=8, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        # x: [B, T, Q, d_model]
        B, T, Q, d_model = x.shape
        
        # 重塑为 [B*Q, T, d_model] 对每个点的时序做attention
        x_flat = x.permute(0, 2, 1, 3).contiguous().view(B * Q, T, d_model)
        attn_out, _ = self.attention(x_flat, x_flat, x_flat)
        attn_out = attn_out.view(B, Q, T, d_model).permute(0, 2, 1, 3)
        
        return self.norm(x + attn_out)

class ResidualBlock(nn.Module):
    """残差块，包含时间嵌入"""
    def __init__(self, d_model, time_emb_dim, dropout=0.1):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, d_model),
            nn.SiLU()
        )
        
        self.block1 = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 2),
            nn.SiLU(),
            nn.Dropout(dropout)
        )
        
        self.block2 = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout)
        )
        
    def forward(self, x, time_emb):
        # x: [B, T, Q, d_model]
        # time_emb: [B, time_emb_dim]
        
        time_emb = self.time_mlp(time_emb)  # [B, d_model]
        time_emb = time_emb.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, d_model]
        
        h = self.block1(x)
        h = h + time_emb  # 广播加法
        h = self.block2(h)
        
        return x + h

class TransformerBlock(nn.Module):
    """结合时间和空间注意力的Transformer块"""
    def __init__(self, d_model, nhead=8, time_emb_dim=256, dropout=0.1):
        super().__init__()
        self.point_attn = PointCloudAttention(d_model, nhead, dropout)
        self.temporal_attn = TemporalAttention(d_model, nhead, dropout)
        self.residual = ResidualBlock(d_model, time_emb_dim, dropout)
        
    def forward(self, x, time_emb):
        # 先做点云内注意力，再做时序注意力
        x = self.point_attn(x)
        x = self.temporal_attn(x)
        x = self.residual(x, time_emb)
        return x

class TrajectoryInpaintingTransformer(nn.Module):
    def __init__(
        self,
        T=100,                    # 时间步数
        Q=1024,                   # 点云数量
        input_dim=3,              # 输入维度 (x,y,z)
        d_model=256,              # 模型维度
        nhead=8,                  # 注意力头数
        num_layers=6,             # Transformer层数
        time_embed_dim=256,       # 时间嵌入维度
        c_dim=0,                  # condition维度，0表示不使用condition
        dropout=0.1
    ):
        super().__init__()
        
        self.T = T
        self.Q = Q
        self.input_dim = input_dim
        self.d_model = d_model
        self.c_dim = c_dim
        
        # 时间嵌入
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbeddings(128),
            nn.Linear(128, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim)
        )
        
        # 输入投影
        # 输入: noisy_trajectory (T,Q,3) + mask (T,Q,1) + known_trajectory (T,Q,3) + condition (T,Q,c_dim)
        # 总计: (T,Q,7+c_dim) -> (T,Q,d_model)
        input_proj_dim = input_dim * 2 + 1 + c_dim
        self.input_proj = nn.Linear(input_proj_dim, d_model)
        
        # condition投影层 (可选，用于更好地处理condition)
        if c_dim > 0:
            self.condition_proj = nn.Sequential(
                nn.Linear(c_dim, c_dim),
                nn.SiLU(),
                nn.Linear(c_dim, c_dim)
            )
        
        # 位置编码
        self.temporal_pos_emb = TemporalPositionEmbedding(d_model, max_len=T)
        
        # 点云位置编码 (可学习的)
        self.point_pos_emb = nn.Parameter(torch.randn(1, 1, Q, d_model) * 0.02)
        
        # Transformer层
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, nhead, time_embed_dim, dropout)
            for _ in range(num_layers)
        ])
        
        # 输出投影
        self.output_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, input_dim)  # 输出预测的噪声 (T,Q,3)
        )
        
    def forward(self, noisy_trajectory, timesteps, mask, known_trajectory, condition=None):
        """
        Args:
            noisy_trajectory: [B, T, Q, 3] 带噪声的轨迹
            timesteps: [B] 扩散时间步
            mask: [B, T, Q, 1] 掩码，1表示已知，0表示需要inpaint
            known_trajectory: [B, T, Q, 3] 已知轨迹部分
            condition: [B, Q, c_dim] 每个点的条件特征，可选
        Returns:
            [B, T, Q, 3] 预测的噪声
        """
        B, T, Q, _ = noisy_trajectory.shape
        
        # 时间嵌入
        time_emb = self.time_embed(timesteps)  # [B, time_embed_dim]
        
        # 掩码化已知轨迹 (在已知区域保留原值，未知区域置零)
        masked_known = known_trajectory * mask
        
        # 拼接输入特征
        x = torch.cat([noisy_trajectory, mask, masked_known], dim=-1)  # [B, T, Q, 7]
        
        # 处理condition
        if condition is not None and self.c_dim > 0:
            # 检查condition的维度
            assert condition.shape == (B, Q, self.c_dim), f"Expected condition shape [B, Q, c_dim] = [{B}, {Q}, {self.c_dim}], got {condition.shape}"
            
            # 通过投影层处理condition
            condition_processed = self.condition_proj(condition)  # [B, Q, c_dim]
            
            # 广播condition到时间维度
            condition_expanded = condition_processed.unsqueeze(1).expand(B, T, Q, self.c_dim)  # [B, T, Q, c_dim]
            
            # 拼接condition
            x = torch.cat([x, condition_expanded], dim=-1)  # [B, T, Q, 7+c_dim]
        elif self.c_dim > 0:
            # 如果模型需要condition但没有提供，则用零填充
            zero_condition = torch.zeros(B, T, Q, self.c_dim, device=x.device, dtype=x.dtype)
            x = torch.cat([x, zero_condition], dim=-1)  # [B, T, Q, 7+c_dim]
        
        # 输入投影
        x = self.input_proj(x)  # [B, T, Q, d_model]
        
        # 添加位置编码
        # 时间位置编码
        x_temp = x.mean(dim=2)  # [B, T, d_model] 平均池化点云维度
        x_temp = self.temporal_pos_emb(x_temp)  # [B, T, d_model]
        x = x + x_temp.unsqueeze(2)  # 广播到 [B, T, Q, d_model]
        
        # 点云位置编码
        x = x + self.point_pos_emb  # [B, T, Q, d_model]
        
        # Transformer处理
        for layer in self.layers:
            x = layer(x, time_emb)
        
        # 输出投影
        noise_pred = self.output_proj(x)  # [B, T, Q, 3]
        
        return noise_pred

class DDPMScheduler:
    """DDPM调度器，适配轨迹数据"""
    def __init__(self, num_timesteps=1000, beta_start=0.0001, beta_end=0.02):
        self.num_timesteps = num_timesteps
        
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
        self.sqrt_alphas_cumprod = self.alphas_cumprod.sqrt()
        self.sqrt_one_minus_alphas_cumprod = (1. - self.alphas_cumprod).sqrt()
    
    def add_noise(self, original, noise, timesteps):
        """为轨迹添加噪声"""
        # original: [B, T, Q, 3]
        # noise: [B, T, Q, 3] 
        # timesteps: [B]
        
        sqrt_alpha_prod = self.sqrt_alphas_cumprod[timesteps]  # [B]
        sqrt_one_minus_alpha_prod = self.sqrt_one_minus_alphas_cumprod[timesteps]  # [B]
        
        # 广播到轨迹维度
        sqrt_alpha_prod = sqrt_alpha_prod.view(-1, 1, 1, 1)  # [B, 1, 1, 1]
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.view(-1, 1, 1, 1)
        
        return (
            sqrt_alpha_prod * original +
            sqrt_one_minus_alpha_prod * noise
        )

def create_trajectory_inpainting_model(T=100, Q=1024, c_dim=0):
    """创建轨迹修复模型"""
    model = TrajectoryInpaintingTransformer(
        T=T,
        Q=Q,
        input_dim=3,
        d_model=256,
        nhead=8,
        num_layers=6,
        time_embed_dim=256,
        c_dim=c_dim,
        dropout=0.1
    )
    return model

def training_step(model, scheduler, batch, device):
    """训练步骤"""
    trajectories = batch['trajectory'].to(device)      # [B, T, Q, 3] 完整轨迹
    masks = batch['mask'].to(device)                   # [B, T, Q, 1] 掩码
    known_trajectories = batch['known_trajectory'].to(device)  # [B, T, Q, 3] 已知部分
    
    # condition可能存在也可能不存在
    condition = batch.get('condition', None)
    if condition is not None:
        condition = condition.to(device)  # [B, Q, c_dim]
    
    batch_size = trajectories.shape[0]
    
    # 随机采样时间步
    timesteps = torch.randint(0, scheduler.num_timesteps, (batch_size,), device=device)
    
    # 生成噪声并添加到轨迹
    noise = torch.randn_like(trajectories)
    noisy_trajectories = scheduler.add_noise(trajectories, noise, timesteps)
    
    # 模型预测噪声
    noise_pred = model(noisy_trajectories, timesteps, masks, known_trajectories, condition)
    
    # 计算损失 - 只在需要inpaint的区域 (mask=0)
    inpaint_mask = (1 - masks)  # [B, T, Q, 1]
    loss = F.mse_loss(
        noise_pred * inpaint_mask,    # 预测噪声(仅inpaint区域)
        noise * inpaint_mask          # 真实噪声(仅inpaint区域)
    )
    
    return loss

# 生成mask的辅助函数
def generate_trajectory_mask(T, Q, mask_ratio=0.3, temporal_coherent=True):
    """
    生成轨迹mask
    Args:
        T: 时间步数
        Q: 点数
        mask_ratio: 需要inpaint的比例
        temporal_coherent: 是否保持时间连贯性
    """
    if temporal_coherent:
        # 随机选择一些时间段进行mask
        mask = torch.ones(T, Q, 1)
        num_masked_steps = int(T * mask_ratio)
        
        # 随机选择连续的时间段
        start_idx = torch.randint(0, T - num_masked_steps + 1, (1,)).item()
        mask[start_idx:start_idx + num_masked_steps] = 0
        
    else:
        # 随机mask一些点
        mask = torch.ones(T, Q, 1)
        total_points = T * Q
        num_masked = int(total_points * mask_ratio)
        
        # 随机选择要mask的位置
        flat_mask = mask.view(-1)
        masked_indices = torch.randperm(total_points)[:num_masked]
        flat_mask[masked_indices] = 0
        mask = flat_mask.view(T, Q, 1)
    
    return mask

if __name__ == "__main__":
    # 测试模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 模型参数
    T, Q = 100, 512  # 100时间步，512个点
    c_dim = 64       # condition维度
    
    # 测试两种情况：有condition和无condition
    print("Testing without condition:")
    model_no_cond = create_trajectory_inpainting_model(T=T, Q=Q, c_dim=0).to(device)
    
    print("Testing with condition:")
    model_with_cond = create_trajectory_inpainting_model(T=T, Q=Q, c_dim=c_dim).to(device)
    
    scheduler = DDPMScheduler()
    
    # 模拟数据
    batch_size = 4
    trajectory = torch.randn(batch_size, T, Q, 3).to(device)
    timesteps = torch.randint(0, 1000, (batch_size,)).to(device)
    
    # 生成mask
    mask = generate_trajectory_mask(T, Q, mask_ratio=0.3, temporal_coherent=True)
    mask = mask.unsqueeze(0).repeat(batch_size, 1, 1, 1).to(device)
    
    known_trajectory = trajectory * mask  # 已知部分
    
    # 生成condition特征
    condition = torch.randn(batch_size, Q, c_dim).to(device)  # [B, Q, c_dim]
    
    # 测试无condition模型
    print("\n=== Testing model without condition ===")
    noise_pred_no_cond = model_no_cond(trajectory, timesteps, mask, known_trajectory)
    
    print(f"Trajectory shape: {trajectory.shape}")
    print(f"Mask shape: {mask.shape}")
    print(f"Noise prediction shape (no condition): {noise_pred_no_cond.shape}")
    print(f"Model parameters (no condition): {sum(p.numel() for p in model_no_cond.parameters()):,}")
    
    # 测试有condition模型
    print("\n=== Testing model with condition ===")
    noise_pred_with_cond = model_with_cond(trajectory, timesteps, mask, known_trajectory, condition)
    
    print(f"Condition shape: {condition.shape}")
    print(f"Noise prediction shape (with condition): {noise_pred_with_cond.shape}")
    print(f"Model parameters (with condition): {sum(p.numel() for p in model_with_cond.parameters()):,}")
    
    # 模拟训练（无condition）
    print("\n=== Training without condition ===")
    batch_no_cond = {
        'trajectory': trajectory,
        'mask': mask,
        'known_trajectory': known_trajectory
    }
    
    loss_no_cond = training_step(model_no_cond, scheduler, batch_no_cond, device)
    print(f"Training loss (no condition): {loss_no_cond.item():.6f}")
    
    # 模拟训练（有condition）
    print("\n=== Training with condition ===")
    batch_with_cond = {
        'trajectory': trajectory,
        'mask': mask,
        'known_trajectory': known_trajectory,
        'condition': condition
    }
    
    loss_with_cond = training_step(model_with_cond, scheduler, batch_with_cond, device)
    print(f"Training loss (with condition): {loss_with_cond.item():.6f}")
    
    # 测试condition缺失时的处理（模型需要condition但没有提供）
    print("\n=== Testing condition missing handling ===")
    try:
        noise_pred_missing = model_with_cond(trajectory, timesteps, mask, known_trajectory, condition=None)
        print(f"Successfully handled missing condition, output shape: {noise_pred_missing.shape}")
    except Exception as e:
        print(f"Error when condition is missing: {e}")
        
    print("\n=== All tests completed ===")