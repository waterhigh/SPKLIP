import torch
import torchinfo
import torch.nn as nn
from timm.models.layers import to_2tuple, trunc_normal_, DropPath
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
from einops.layers.torch import Rearrange
import torch.nn.functional as F
from functools import partial

class Quant(torch.autograd.Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, i, min_value, max_value):
        ctx.min = min_value
        ctx.max = max_value
        ctx.save_for_backward(i)
        return torch.round(torch.clamp(i, min=min_value, max=max_value))

    @staticmethod
    @torch.cuda.amp.custom_fwd
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        i, = ctx.saved_tensors
        grad_input[i < ctx.min] = 0
        grad_input[i > ctx.max] = 0
        return grad_input, None, None
    
class MultiSpike(nn.Module):
    def __init__(
        self,
        min_value=0,
        max_value=4,
        Norm=None,
        ):
        super().__init__()
        if Norm == None:
            self.Norm = max_value
        else:
            self.Norm = Norm
        self.min_value = min_value
        self.max_value = max_value
    
    @staticmethod
    def spike_function(x, min_value, max_value):
        return Quant.apply(x, min_value, max_value)
        
    def __repr__(self):
        return f"MultiSpike(Max_Value={self.max_value}, Min_Value={self.min_value}, Norm={self.Norm})"     

    def forward(self, x): # B C H W
        return self.spike_function(x, min_value=self.min_value, max_value=self.max_value) / (self.Norm)

class BNAndPadLayer(nn.Module):
    def __init__(
        self,
        pad_pixels,
        num_features,
        eps=1e-5,
        momentum=0.1,
        affine=True,
        track_running_stats=True,
    ):
        super(BNAndPadLayer, self).__init__()
        self.bn = nn.BatchNorm2d(
            num_features, eps, momentum, affine, track_running_stats
        )
        self.pad_pixels = pad_pixels

    def forward(self, input):
        output = self.bn(input)
        if self.pad_pixels > 0:
            if self.bn.affine:
                pad_values = (
                    self.bn.bias.detach()
                    - self.bn.running_mean
                    * self.bn.weight.detach()
                    / torch.sqrt(self.bn.running_var + self.bn.eps)
                )
            else:
                pad_values = -self.bn.running_mean / torch.sqrt(
                    self.bn.running_var + self.bn.eps
                )
            output = F.pad(output, [self.pad_pixels] * 4)
            pad_values = pad_values.view(1, -1, 1, 1)
            output[:, :, 0 : self.pad_pixels, :] = pad_values
            output[:, :, -self.pad_pixels :, :] = pad_values
            output[:, :, :, 0 : self.pad_pixels] = pad_values
            output[:, :, :, -self.pad_pixels :] = pad_values
        return output

    @property
    def weight(self):
        return self.bn.weight

    @property
    def bias(self):
        return self.bn.bias

    @property
    def running_mean(self):
        return self.bn.running_mean

    @property
    def running_var(self):
        return self.bn.running_var

    @property
    def eps(self):
        return self.bn.eps


class TemporalPositionEncoder(nn.Module):
    def __init__(self, embed_dim, max_len=512):
        super().__init__()
        # 可学习相对位置编码
        self.pos_emb = nn.Parameter(torch.randn(max_len, 1, embed_dim))
        self.spike = MultiSpike(max_value=4)
        
    def forward(self, x):  # x: [T, B, D]
        T = x.size(0)
        pos = self.pos_emb[:T]  # [T, 1, D]
        return self.spike(x + pos)
    
class SpikingMHA(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # 脉冲驱动线性变换
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.spike = MultiSpike(max_value=4)
        
    def forward(self, x):  # x: [T, B, D]
        B, T, D = x.shape[1], x.shape[0], x.shape[2]
        
        # 脉冲驱动投影
        q = self.q_proj(x).view(T, B, self.num_heads, self.head_dim).permute(1,2,0,3)  # [B, H, T, d]
        k = self.k_proj(x).view(T, B, self.num_heads, self.head_dim).permute(1,2,0,3)
        v = self.v_proj(x).view(T, B, self.num_heads, self.head_dim).permute(1,2,0,3)
        
        # 脉冲注意力计算
        attn = (q @ k.transpose(-2,-1)) / (self.head_dim**0.5)  # [B, H, T, T]
        attn = self.spike(attn).softmax(dim=-1)
        x = (attn @ v).permute(2,0,1,3).reshape(T, B, D)  # [T, B, D]
        
        return x
    

class TemporalFusionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.mha = SpikingMHA(embed_dim, num_heads)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 4*embed_dim),
            MultiSpike(max_value=4),
            nn.Linear(4*embed_dim, embed_dim),
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
    def forward(self, x):  # x: [T, B, D]
        # 多头注意力分支
        attn_out = self.mha(self.norm1(x))
        x = x + attn_out
        
        # MLP增强分支
        mlp_out = self.mlp(self.norm2(x))
        x = x + mlp_out
        return x
    
class PureSpikeTransformer(nn.Module):
    def __init__(self, input_dim, num_heads=8, num_layers=6):
        super().__init__()
        self.pos_encoder = TemporalPositionEncoder(input_dim)
        self.blocks = nn.ModuleList([
            TemporalFusionBlock(input_dim, num_heads)
            for _ in range(num_layers)
        ])
        
    def forward(self, x):  # x: [T, B, D]
        # 位置编码
        x = self.pos_encoder(x)  # [T, B, D]
        
        # 时序特征融合
        for block in self.blocks:
            x = block(x)
            
        return x  # [T, B, D]
    
if __name__ == "__main__":
    
    # 初始化模型
    model = PureSpikeTransformer(
        input_dim=128,  # 输入特征维度
        num_heads=8,    # 注意力头数（建议D可被头数整除）
        num_layers=6     # 融合层数
    )

    # 示例输入
    x = torch.randn(50, 32, 128)  # [T=50, B=32, D=128]

    # 前向传播
    output = model(x)  # 保持[T, B, D]维度

    print(output.shape)
