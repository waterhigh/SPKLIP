import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

class ExpertAdapter(nn.Module):
    def __init__(self, in_channels, adapter_channels, expertise="semantic"):
        super().__init__()
        self.expertise = expertise
        
        # 核心适配器（所有专家共享）
        self.core_adapter = nn.Sequential(
            nn.Linear(in_channels, adapter_channels),
            nn.GELU(),
            nn.Linear(adapter_channels, in_channels)
        )
        
        # 特化模块（按专家类型初始化）
        if self.expertise == "temporal":
            self.temporal_attn = nn.MultiheadAttention(
                embed_dim=in_channels, 
                num_heads=1, 
                batch_first=True
            )
        elif self.expertise == "syntax":
            # LSTM的输出维度是adapter_channels，需映射回in_channels
            self.lstm = nn.LSTM(
                input_size=in_channels,
                hidden_size=adapter_channels,
                batch_first=True
            )
            self.lstm_proj = nn.Linear(adapter_channels, in_channels)  # 新增投影层
        
    def forward(self, x):
        # 核心适配器
        x = self.core_adapter(x)  # 输出维度: [batch, seq_len, in_channels]
        
        # 特化模块
        if self.expertise == "temporal":
            x, _ = self.temporal_attn(x, x, x)  # 输出维度不变
        elif self.expertise == "syntax":
            x, _ = self.lstm(x)          # LSTM输出维度: [batch, seq_len, adapter_channels]
            x = self.lstm_proj(x)        # 投影到 [batch, seq_len, in_channels]
        
        return x  # 保证所有专家输出维度一致

class MoETextAdapter(nn.Module):
    def __init__(self, in_channels, adapter_channels, num_experts=3):
        super().__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleDict({
            "semantic": ExpertAdapter(in_channels, adapter_channels, "semantic"),
            "temporal": ExpertAdapter(in_channels, adapter_channels, "temporal"),
            "syntax": ExpertAdapter(in_channels, adapter_channels, "syntax")
        })
        self.gate = nn.Sequential(
            nn.Linear(in_channels, num_experts),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, x):
        # 确保输入是3D（batch_first=True）
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [batch, 1, in_channels]
        batch_size, seq_len, feat_dim = x.shape
        
        # 计算门控权重
        gate_weights = self.gate(x.mean(dim=1))  # [batch, num_experts]
        topk_weights, topk_indices = gate_weights.topk(2, dim=-1)  # [batch, 2]
        
        # 预计算所有专家输出 ------------------------------------------------------
        # 将输入复制为 [batch, num_experts, seq_len, feat_dim]
        expanded_x = x.unsqueeze(1).expand(-1, self.num_experts, -1, -1)
        
        # 并行计算所有专家输出 [batch, num_experts, seq_len, feat_dim]
        expert_outputs = torch.stack([
            self.experts["semantic"](expanded_x[:,0]),
            self.experts["temporal"](expanded_x[:,1]),
            self.experts["syntax"](expanded_x[:,2])
        ], dim=1)
        
        # 动态选择Top-K专家 ------------------------------------------------------
        # 索引格式转换：topk_indices [batch, 2] → [batch, 2, 1, 1] 用于gather
        indices = topk_indices.view(batch_size, 2, 1, 1).expand(-1, -1, seq_len, feat_dim)
        
        # 收集选中专家的输出 [batch, 2, seq_len, feat_dim]
        selected_experts = expert_outputs.gather(
            dim=1, 
            index=indices
        )
        
        # 权重计算 [batch, 2, 1, 1] → 广播到所有维度
        weights = topk_weights.view(batch_size, 2, 1, 1)
        
        # 加权求和 [batch, seq_len, feat_dim]
        weighted_out = (selected_experts * weights).sum(dim=1)
        
        # 残差连接
        return x + weighted_out
