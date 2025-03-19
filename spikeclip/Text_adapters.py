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
        
        # 特化模块
        if expertise == "temporal":
            self.temporal_attn = nn.MultiheadAttention(in_channels, num_heads=1)
        elif expertise == "syntax":
            self.lstm = nn.LSTM(input_size=in_channels, hidden_size=adapter_channels, batch_first=True)
        
        # 初始化
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0.)

    def forward(self, x):
        # 先通过核心适配器
        x = self.core_adapter(x)
        
        # 根据专家类型应用特化模块
        if self.expertise == "temporal":
            # 多头注意力需要三维输入（假设x是二维的，需增加序列维度）
            x = x.unsqueeze(0)  # [1, batch, in_channels]
            x, _ = self.temporal_attn(x, x, x)  # 自注意力
            x = x.squeeze(0)  # 恢复二维
        elif self.expertise == "syntax":
            # LSTM需要序列输入（假设x是 [batch, in_channels]，需视为长度为1的序列）
            x = x.unsqueeze(1)  # [batch, 1, in_channels]
            x, _ = self.lstm(x)
            x = x.squeeze(1)  # 恢复二维
        
        return x

class LoRAAdapter(nn.Module):
    def __init__(self, in_channels, adapter_channels, rank=16):
        super().__init__()
        # 原始适配器（冻结）
        self.original_fc1 = nn.Linear(in_channels, adapter_channels)
        self.original_fc2 = nn.Linear(adapter_channels, in_channels)
        # LoRA增量参数（训练）
        self.lora_A = nn.Linear(in_channels, rank, bias=False)
        self.lora_B = nn.Linear(rank, in_channels, bias=False)
        # 初始化LoRA参数为0
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
    
    def forward(self, x):
        # 原始适配器路径（冻结）
        original_out = self.original_fc2(self.original_fc1(x))
        # LoRA路径（低秩增量）
        lora_out = self.lora_B(self.lora_A(x))
        # 残差连接
        x = x + original_out + lora_out
        return x

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


class LoRAMoETextAdapter(nn.Module):
    def __init__(self, in_channels, adapter_channels, num_experts=3, rank=16):
        super().__init__()
        self.experts = nn.ModuleList([
            LoRAAdapter(in_channels, adapter_channels, rank)  # 每个专家使用LoRA
            for _ in range(num_experts)
        ])
        self.gate = nn.Sequential(
            nn.Linear(in_channels, num_experts),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, x):
        # 门控选择专家（同前）
        gate_weights = self.gate(x.mean(dim=1))
        topk_weights, topk_indices = gate_weights.topk(2, dim=-1)
        expert_outs = []
        for i in range(x.size(0)):
            selected_experts = [self.experts[idx] for idx in topk_indices[i]]
            weighted_out = sum(w * exp(x[i].unsqueeze(0)) for w, exp in zip(topk_weights[i], selected_experts))
            expert_outs.append(weighted_out)
        x = x + torch.cat(expert_outs, dim=0)
        return x
