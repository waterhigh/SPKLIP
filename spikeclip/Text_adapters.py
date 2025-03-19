import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

class ExpertAdapter(nn.Module):
    def __init__(self, in_channels, adapter_channels, expertise="semantic"):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Linear(in_channels, adapter_channels),
            nn.GELU(),
            nn.Linear(adapter_channels, in_channels)
        )
        # 专家特化设计（根据需求扩展）
        if expertise == "temporal":
            self.adapter.add_module("temporal_attn", nn.MultiheadAttention(in_channels, 1))  # 时序专家
        elif expertise == "syntax":
            self.adapter.add_module("lstm", nn.LSTM(in_channels, adapter_channels))  # 语法专家
        # 初始化保持一致性 
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0.)

class MoETextAdapter(nn.Module):
    def __init__(self, in_channels, adapter_channels, num_experts=3):
        super().__init__()
        self.experts = nn.ModuleList([
            ExpertAdapter(in_channels, adapter_channels, expertise)
            for expertise in ["semantic", "temporal", "syntax"]  # 定义不同专家类型
        ])
        self.gate = nn.Sequential(
            nn.Linear(in_channels, num_experts),
            nn.Softmax(dim=-1)  # 门控网络
        )
    
    def forward(self, x):
        # 门控选择专家（稀疏激活）
        gate_weights = self.gate(x.mean(dim=1))  # [batch, num_experts]
        topk_weights, topk_indices = gate_weights.topk(2, dim=-1)  # Top-2激活
        
        # 动态融合专家输出
        expert_outs = []
        for i in range(x.size(0)):
            selected_experts = [self.experts[idx] for idx in topk_indices[i]]
            weighted_out = sum(w * exp(x[i].unsqueeze(0)) for w, exp in zip(topk_weights[i], selected_experts))
            expert_outs.append(weighted_out)
        
        x = x + torch.cat(expert_outs, dim=0)
        return x
