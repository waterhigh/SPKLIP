from collections import OrderedDict
from typing import Tuple, Union
# import math
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torchinfo
# from align_arch import *

thresh = 0.5  # neuronal threshold
lens = 0.5  # hyper-parameters of approximate function
decay = 0.25  # decay constants
num_classes = 1000
time_window = 6
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# define approximate firing function
class ActFun(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.gt(thresh).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad = grad_output.clone()
        
        # 改进后的梯度计算（避免梯度爆炸）
        temp = torch.sigmoid(5*(thresh - input))  # 使用sigmoid替代三角波
        return grad * temp


act_fun = ActFun.apply
# membrane potential update


class mem_update(nn.Module):
    def __init__(self):
        super().__init__()
        self.decay = 0.25
        self.register_buffer('mem', None)  # 正确注册缓存

    def forward(self, x):
        """
        输入 x 维度: [B, C, H, W]
        输出维度: [B, C, H, W]
        """
        B, C, H, W = x.shape
        
        # 初始化膜电位（动态适配输入维度）
        if self.mem is None or self.mem.shape != (B, C, H, W):
            self.mem = torch.zeros(B, C, H, W, device=x.device)
        
        # 膜电位更新
        self.mem = self.decay * self.mem + x
        
        # 脉冲生成
        spike = ActFun.apply(self.mem)
        self.mem = self.mem * (1 - spike.detach())
        
        return spike

class BatchNorm2d1(nn.BatchNorm2d):
    """阈值初始化版本"""
    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            nn.init.constant_(self.weight, thresh)  # 使用预设阈值初始化
            nn.init.zeros_(self.bias)

class BatchNorm2d2(nn.BatchNorm2d):
    """零初始化版本""" 
    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            nn.init.constant_(self.weight, 0)  # 权重初始化为0
            nn.init.zeros_(self.bias)

class batch_norm_2d(nn.Module):
    """适配脉冲网络的二维BN"""
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.bn = BatchNorm2d1(num_features, eps=eps, momentum=momentum)

    def forward(self, x):
        # 直接处理BCHW输入 [batch, channel, height, width]
        return self.bn(x)

class batch_norm_2d1(nn.Module):
    """零初始化变体"""
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.bn = BatchNorm2d2(num_features, eps=eps, momentum=momentum)

    def forward(self, x):
        # 保持维度不变
        return self.bn(x)

class Snn_Conv2d(nn.Conv2d):
    def __init__(self, 
                 in_channels, 
                 out_channels,
                 kernel_size, 
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 padding_mode='zeros',
                 thresh=0.5,
                 decay=0.25):
        super().__init__(
            in_channels, out_channels, kernel_size, 
            stride, padding, dilation, groups, bias, padding_mode
        )
        
        # 注册为持久化缓冲区
        self.register_buffer('mem', None)  # 关键修改
        self.thresh = thresh
        self.decay = decay

    def forward(self, x):
        # 执行卷积
        conv_out = F.conv2d(
            x, self.weight, self.bias, 
            self.stride, self.padding,
            self.dilation, self.groups
        )
        
        # 动态初始化膜电位
        if self.mem is None or self.mem.shape != conv_out.shape:
            self.mem = torch.zeros_like(conv_out, device=x.device)
            self.register_buffer('mem', self.mem)  # 重新注册更新后的缓冲区
        
        # 膜电位更新
        self.mem = self.decay * self.mem + conv_out
        
        # 脉冲生成
        spike = ActFun.apply(self.mem)  # 关键修改点
        self.mem = self.mem * (1 - spike.detach())
        
        return spike
    
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()
        # 卷积层替换为脉冲版本
        self.conv1 = Snn_Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = batch_norm_2d(planes)
        
        self.conv2 = Snn_Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = batch_norm_2d(planes)
        
        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()
        
        self.conv3 = Snn_Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = batch_norm_2d(planes * self.expansion)
        
        # 膜电位管理器
        self.mem_manager = mem_update()
        
        # 下采样调整
        self.downsample = None
        self.stride = stride
        if stride > 1 or inplanes != planes * self.expansion:
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", Snn_Conv2d(inplanes, planes * self.expansion, 1, bias=False)),
                ("1", batch_norm_2d(planes * self.expansion))
            ]))

    def forward(self, x):
        identity = x
        
        # 脉冲特征提取
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.mem_manager(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.mem_manager(out)
        out = self.avgpool(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        # 残差连接
        if self.downsample is not None:
            identity = self.downsample(x)
        
        # 膜电位相加
        out += identity
        return self.mem_manager(out)

class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim, embed_dim, num_heads, output_dim=None):
        super().__init__()
        self.spacial_size = spacial_dim ** 2
        self.embed_dim = embed_dim
        
        # 位置编码初始化 (保持4D结构)
        self.positional_embedding = nn.Parameter(
            torch.randn(1, embed_dim, spacial_dim, spacial_dim)  # [1, D, H, W]
        )
        
        # 位置编码处理 (保持空间结构)
        self.pos_encoder = nn.Sequential(
            Snn_Conv2d(embed_dim, embed_dim, 3, padding=1),
            mem_update()
        )
        
        # 注意力模块 (兼容空间结构)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=0.1)
        self.proj = Snn_Conv2d(embed_dim, output_dim or embed_dim, 1)
        self.mem = mem_update()

    def forward(self, x):
        """
        输入形状: [B, C, H, W]
        输出形状: [B, output_dim]
        """
        # 保留原始空间维度
        B, C, H, W = x.shape
        
        # 位置编码融合 (保持4D)
        pos_emb = self.pos_encoder(self.positional_embedding)
        x = x + pos_emb  # [B, C, H, W] + [1, D, H, W]
        
        # 空间维度展平 (不改变张量维度数)
        x_flat = x.flatten(2)  # [B, C, H*W]
        
        # 添加分类token
        cls_token = x.mean(dim=(2,3), keepdim=True)  # [B, C, 1, 1]
        x = torch.cat([cls_token, x_flat], dim=2)     # [B, C, H*W+1]
        
        # 调整维度为注意力输入格式 [Seq, B, Dim]
        x = x.permute(2, 0, 1)  # [H*W+1, B, C]
        
        # 脉冲注意力计算
        attn_output, _ = self.attn(
            query=x[:1],  # 仅用cls_token作为query
            key=x,
            value=x
        )
        
        # 投影输出
        output = self.proj(attn_output.permute(1, 2, 0).unsqueeze(-1))  # [B, D, 1, 1]
        return self.mem(output.squeeze(-1).squeeze(-1))  # [B, output_dim]


# 修改，不使用attentionpooling
class ModifiedResNet(nn.Module):
    def __init__(self, layers, output_dim, heads, input_resolution, width, input_channels=320):
        super().__init__()
        # 初始化卷积层
        self.conv1 = Snn_Conv2d(input_channels, width//2, kernel_size=3, stride=2, padding=1)
        self.bn1 = batch_norm_2d(width//2)
        
        self.conv2 = Snn_Conv2d(width//2, width//2, kernel_size=3, padding=1)
        self.bn2 = batch_norm_2d(width//2)
        
        self.conv3 = Snn_Conv2d(width//2, width, kernel_size=3, padding=1)
        self.bn3 = batch_norm_2d(width)
        
        # 下采样组件
        self.avgpool = nn.AvgPool2d(2)
        self.mem_pool = mem_update()
        
        # 残差层配置
        self._inplanes = width
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width*2, layers[1], stride=2)
        self.layer3 = self._make_layer(width*4, layers[2], stride=2)
        self.layer4 = self._make_layer(width*8, layers[3], stride=2)
        
        # 最终池化层（关键修正部分）
        self.final_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # [B, C, 1, 1]
            Snn_Conv2d(
                in_channels=width * 32,  # 计算方式：width*8 * Bottleneck.expansion(4)
                out_channels=output_dim,
                kernel_size=1
            ),  # [B, output_dim, 1, 1]
            mem_update(),
            nn.Flatten(start_dim=1)  # [B, output_dim]

        )

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]
        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        # 添加维度校验点
        def validate_shape(tensor, expected_dims):
            if tensor.dim() != expected_dims:
                raise ValueError(f"维度错误: 期望{expected_dims}D，实际得到{tensor.dim()}D张量")

        # 预处理阶段
        def stem(x):
            x = self.conv1(x)
            validate_shape(x, 4)
            x = self.bn1(x)
            x = self.mem_pool(x)
            
            x = self.conv2(x)
            x = self.bn2(x)
            x = self.mem_pool(x)
            
            x = self.conv3(x)
            x = self.bn3(x)
            x = self.avgpool(x)
            return self.mem_pool(x)
        
        # 主流程
        x = stem(x)
        validate_shape(x, 4)
        
        x = self.layer1(x)
        # print("Layer1输出形状:", x.shape)
        validate_shape(x, 4)
        
        x = self.layer2(x)
        # print("Layer2输出形状:", x.shape)
        validate_shape(x, 4)
        
        x = self.layer3(x)
        validate_shape(x, 4)
        
        x = self.layer4(x)
        # print("Layer4输出形状:", x.shape)
        validate_shape(x, 4)
        
        return self.final_pool(x)
    
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 确保输出通道数一致
        self.conv1 = Snn_Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.mem_update1 = mem_update()
        
        self.conv2 = Snn_Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.mem_update2 = mem_update()
        
        # 残差路径通道对齐
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                Snn_Conv2d(in_channels, out_channels, kernel_size=1),
                mem_update()
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.mem_update1(out)
        
        out = self.conv2(out)
        out = self.mem_update2(out)
        
        # 确保维度匹配
        assert out.shape == identity.shape, f"维度不匹配: {out.shape} vs {identity.shape}"
        return out + identity

class SurrogateSigmoid(torch.autograd.Function):
    """使用替代梯度保持可微性"""
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x.gt(0).float()  # 硬Sigmoid

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        grad = 1 / (1 + torch.abs(x))  # 替代梯度
        return grad_output * grad

# 修正后的CALayer2实现
class CALayer2(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.ca_block = nn.Sequential(
            Snn_Conv2d(in_channels, in_channels*2, 3, padding=1, bias=True),
            mem_update(),
            Snn_Conv2d(in_channels*2, in_channels, 3, padding=1, bias=True),
            SurrogateSigmoidWrapper()  # 使用包装后的模块
        )

    def forward(self, x):
        weight = self.ca_block(x)
        return weight

# 将Function包装成Module子类
class SurrogateSigmoidWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return SurrogateSigmoid.apply(x)  # 调用Function的apply方法
    

class FeatureExtractor(nn.Module):
    def __init__(
        self, in_channels, features, out_channels, channel_step, num_of_layers=16
    ):
        super(FeatureExtractor, self).__init__()
        self.channel_step = channel_step
        self.conv0_0 = Snn_Conv2d(
            in_channels=in_channels, out_channels=16, kernel_size=3, padding=1
        )
        self.conv0_1 = Snn_Conv2d(
            in_channels=in_channels - 2 * channel_step,
            out_channels=16, kernel_size=3, padding=1
        )
        self.conv0_2 = Snn_Conv2d(
            in_channels=in_channels - 4 * channel_step,
            out_channels=16, kernel_size=3, padding=1
        )
        self.conv0_3 = Snn_Conv2d(
            in_channels=in_channels - 6 * channel_step,
            out_channels=16, kernel_size=3, padding=1
        )
        self.conv1_0 = Snn_Conv2d(
            in_channels=16, out_channels=1, kernel_size=3, padding=1
        )
        self.conv1_1 = Snn_Conv2d(
            in_channels=16, out_channels=1, kernel_size=3, padding=1
        )
        self.conv1_2 = Snn_Conv2d(
            in_channels=16, out_channels=1, kernel_size=3, padding=1
        )
        self.conv1_3 = Snn_Conv2d(
            in_channels=16, out_channels=1, kernel_size=3, padding=1
        )
        self.ca = CALayer2(in_channels=4)
        self.conv = Snn_Conv2d(
            in_channels=4, out_channels=features, kernel_size=3, padding=1
        )
        self.relu = mem_update()
        layers = []
        for _ in range(num_of_layers - 2):
            layers.append(BasicBlock(in_channels=features, out_channels=features))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        out_0 = self.conv1_0(self.relu(self.conv0_0(x)))
        out_1 = self.conv1_1(
            self.relu(self.conv0_1(x[:, self.channel_step : -self.channel_step, :, :]))
        )
        out_2 = self.conv1_2(
            self.relu(
                self.conv0_2(x[:, 2 * self.channel_step : -2 * self.channel_step, :, :])
            )
        )
        out_3 = self.conv1_3(
            self.relu(
                self.conv0_3(x[:, 3 * self.channel_step : -3 * self.channel_step, :, :])
            )
        )
        out = torch.cat((out_0, out_1), 1)
        out = torch.cat((out, out_2), 1)
        out = torch.cat((out, out_3), 1)
        est = out
        weight = self.ca(out)
        out = weight * out
        out = self.conv(out)
        out = self.relu(out)
        tmp = out
        out = self.net(out)
        return out + tmp, est


import torch
from torch import nn

class ResNetWithTransformer(nn.Module):
    def __init__(self, 
                 resnet_layers, 
                 input_channels, 
                 frame_feature_dim, 
                 num_frames, 
                 transformer_layers, 
                 transformer_heads):
        super().__init__()
        # ResNet 部分
        self.resnet = ModifiedResNet(
            layers=resnet_layers,
            output_dim=frame_feature_dim,
            heads=transformer_heads,
            input_resolution=(240, 320),  # 输入图像的分辨率
            width=64,
            input_channels=input_channels
        )
        
        # Transformer 部分
        self.transformer = TransformerEncoder(
            TransformerEncoderLayer(
                d_model=frame_feature_dim,  # 帧级特征的维度
                nhead=transformer_heads,
                dim_feedforward=frame_feature_dim * 4,  # 前馈网络的维度
                dropout=0.1
            ),
            num_layers=transformer_layers
        )

        # Feature Extractor 部分
        self.extractor = FeatureExtractor(
            in_channels=61,  
            features=64, 
            out_channels=64, 
            channel_step=1, 
            num_of_layers=8
        )
        self.win_r = 30  # 窗口半径
        self.win_step = 45  # 窗口步长

        # self.ted_adapter = TEDAdapter(d_model=frame_feature_dim)


    def forward(self, video_frames):
        """
        video_frames: 输入视频序列，形状为 [B, T, C, H, W]
        B: 批次大小
        T: 时间帧数（25）
        C: 通道数（10）
        H, W: 帧的分辨率（240, 320）
        """
        B, T, C, H, W = video_frames.shape
        # Step 1: 对每帧应用窗口处理并提取特征
        processed_blocks = []
        for b in range(B):
            batch_frames = video_frames[b]  # [T, C, H, W]
            batch_frames = batch_frames.view(T * C, H, W)  # 展开为 [250, H, W]
            # 提取五个窗口
            block0 = batch_frames[0 : 2 * self.win_r + 1].unsqueeze(0)  # [1, 61, H, W]
            block1 = batch_frames[self.win_step : self.win_step + 2 * self.win_r + 1].unsqueeze(0)
            block2 = batch_frames[2 * self.win_step : 2 * self.win_step + 2 * self.win_r + 1].unsqueeze(0)
            block3 = batch_frames[3 * self.win_step : 3 * self.win_step + 2 * self.win_r + 1].unsqueeze(0)
            block4 = batch_frames[4 * self.win_step : 4 * self.win_step + 2 * self.win_r + 1].unsqueeze(0)
            # 应用特征提取器
            block0_out, est0 = self.extractor(block0)
            block1_out, est1 = self.extractor(block1)
            block2_out, est2 = self.extractor(block2)
            block3_out, est3 = self.extractor(block3)
            block4_out, est4 = self.extractor(block4)
            block_out = torch.stack([block0_out, block1_out, block2_out, block3_out, block4_out], dim=0)  # [5, 64, H, W]
            block_out = block_out.squeeze(1) 
            ##block_out = block_out.view(1, 5 * 64, H, W)
            processed_blocks.append(block_out.unsqueeze(0))  # [1, 5,64, H, W]
        processed_frames = torch.cat(processed_blocks, dim=0)  # [B, 5,64, H, W]
        processed_frames =processed_frames.squeeze(1) 
        B, T, C, H, W = processed_frames.shape
        frame_features = []
        for t in range(T):
            frame_feature = self.resnet(processed_frames[:, t])  # [B, D]
            frame_features.append(frame_feature)

        # 经过ResNet处理后得到frame_features [B, T, D]
        frame_features = torch.stack(frame_features, dim=1)  # [B, T, D]
    
        # 应用TED-Adapter
        # adapted_features = self.ted_adapter(frame_features)  # [B, T, D]
        
        # Step 2: Transformer 融合时间信息
        # Transformer 输入需要形状 [T, B, D]
        frame_features = frame_features.permute(1, 0, 2)  # [T, B, D]
        temporal_features = self.transformer(frame_features)  # [T, B, D]
        
        # Step 3: 全局特征池化
        global_feature = temporal_features.mean(dim=0)  # [T, B, D] -> [B, D]
        
        return global_feature

class TextAdapter(nn.Module):

    def __init__(self, in_channels, adapter_channels):
        super().__init__()
        self.textad_fc1 = nn.Linear(in_channels, adapter_channels)
        self.textad_gelu = nn.GELU()
        self.textad_fc2 = nn.Linear(adapter_channels, in_channels)
        nn.init.constant_(self.textad_fc1.bias, 0.)
        nn.init.constant_(self.textad_fc2.bias, 0.)

    def forward(self, x):
        # pdb.set_trace()
        x1 = self.textad_fc1(x)
        x1 = self.textad_gelu(x1)
        x1 = self.textad_fc2(x1)
        x = x + x1
        return x

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)
class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class ResidualAttentionBlock(nn.Module):
    def __init__(
        self, 
        d_model: int, 
        n_head: int, 
        attn_mask: torch.Tensor = None, 
        adapter_channels: int = 64  # 新增适配器参数
    ):
        super().__init__()
        
        # 原始组件
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask
        
        # 新增TextAdapter（插入在FFN之后）
        self.text_adapter = TextAdapter(
            in_channels=d_model, 
            adapter_channels=adapter_channels
        )

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        # 原始流程
        x = x + self.attention(self.ln_1(x))  # 自注意力 + 残差
        x = x + self.mlp(self.ln_2(x))         # FFN + 残差
        
        # 插入TextAdapter（残差连接已包含在TextAdapter内部）
        x = self.text_adapter(x)               # 文本适配器
        
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)

class CLIP(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 context_length: int,
                 embed_dim: int,
                 # vision
                 image_resolution: tuple,
                 vision_layers: Tuple[int, int, int, int],
                 vision_width: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int,
                 input_channels: int = 64,  # 默认为 3 通道，可以设置为 50
                 # text
                ):
        """
        CLIP Model with ResNetWithTransformer as vision encoder.
        """
        super().__init__()

        self.context_length = context_length

        # Vision Encoder: ResNetWithTransformer
        self.visual = ResNetWithTransformer(
            resnet_layers=vision_layers,
            input_channels=input_channels,
            frame_feature_dim=embed_dim,
            num_frames=64,  # 假设每个视频有10帧
            transformer_layers=transformer_layers,
            transformer_heads=transformer_heads
        )

        # Text Encoder: Transformer
        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.initialize_parameters()

    def initialize_parameters(self):
        # 初始化文本编码器
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        # 文本自回归掩码
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # 保留上三角部分
        return mask

    @property
    def dtype(self):
        return self.visual.resnet.conv1.weight.dtype

    def encode_image(self, video_frames):
        """
        Encode video frames using ResNetWithTransformer.
        Args:
            video_frames: Input tensor of shape [B, T, C, H, W]
        Returns:
            Feature embeddings of shape [B, embed_dim]
        """
        return self.visual(video_frames)

    def encode_text(self, text):
        """
        Encode text using Transformer.
        Args:
            text: Input tokenized tensor of shape [B, context_length]
        Returns:
            Feature embeddings of shape [B, embed_dim]
        """
        x = self.token_embedding(text).type(self.dtype)  # [B, context_length, transformer_width]
        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # [N, L, D] -> [L, N, D]
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # [L, N, D] -> [N, L, D]
        x = self.ln_final(x).type(self.dtype)

        # Take features from [EOS] token
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        return x

    def forward(self, video_frames, text):
        """
        Forward pass for CLIP: compute video and text embeddings and logits.
        Args:
            video_frames: Tensor of shape [B, T, C, H, W]
            text: Tokenized text tensor of shape [B, context_length]
        Returns:
            logits_per_video: [B, B]
            logits_per_text: [B, B]
        """
        video_features = self.encode_image(video_frames)
        text_features = self.encode_text(text)

        # Normalize features
        video_features = video_features / video_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # Compute similarity logits
        logit_scale = self.logit_scale.exp()
        logits_per_video = logit_scale * video_features @ text_features.t()
        logits_per_text = logits_per_video.t()

        return logits_per_video, logits_per_text

    
def build_model(state_dict: dict):
    """
    Build CLIP model using ModifiedResNet for visual encoding.
    Args:
        state_dict: Pre-trained state dictionary.
    Returns:
        CLIP model.
    """
    counts = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
    vision_layers = tuple(counts)
    vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
    output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
    vision_patch_size = None
    assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
    image_resolution = (output_width * 32, output_width * 32)  # Assume square input

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith("transformer.resblocks")))
    # image_resolution = (240, 320)  # 输入图像的分辨率 (高, 宽)
    model = CLIP(
        embed_dim=embed_dim,
        image_resolution=image_resolution,
        vision_layers=vision_layers,
        vision_width=vision_width,
        input_channels=50,  # For 50-channel input
        context_length=context_length,
        vocab_size=vocab_size,
        transformer_width=transformer_width,
        transformer_heads=transformer_heads,
        transformer_layers=transformer_layers,
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    model.load_state_dict(state_dict)
    return model.eval()

############################################################################################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 假设你有一个定义好的模型，比如 ResNetWithTransformer 或 CLIP

if __name__ == "__main__":
    # 测试输入
    batch_size = 1
    input_channels = 10
    num_frames = 25      # 每个视频的帧数
    image_height = 240
    image_width = 320
    context_length = 77  # 文本的最大长度
    vocab_size = 49408   # 词汇表大小

    # 模拟视频和文本输入
    video_tensor = torch.randn(batch_size, num_frames, input_channels, image_height, image_width).to(device)  # 移动到 GPU
    text_tensor = torch.randint(0, vocab_size, (batch_size, context_length)).to(device)  # 移动到 GPU

    # 模型参数
    embed_dim = 512  # 嵌入维度
    vision_layers = (3, 4, 6, 3)  # ResNet 的层数配置
    vision_width = 64
    transformer_width = 256
    transformer_heads = 4
    transformer_layers = 8

    # 初始化 CLIP 模型（使用 ResNetWithTransformer 作为视觉编码器）
    model = CLIP(
        embed_dim=embed_dim,
        image_resolution=(image_height, image_width),
        vision_layers=vision_layers,
        vision_width=vision_width,
        input_channels=64,
        context_length=context_length,
        vocab_size=vocab_size,
        transformer_width=transformer_width,
        transformer_heads=transformer_heads,
        transformer_layers=transformer_layers
    ).to(device)  # 将模型移到 GPU

    # 运行模型
    print("Running model with:")
    print(f"Video input shape: {video_tensor.shape}")  # 预期形状: [B, T, C, H, W]
    print(f"Text input shape: {text_tensor.shape}")  # 预期形状: [B, context_length]

    # 获取视频和文本的对比学习结果
    logits_per_video, logits_per_text = model(video_tensor, text_tensor)

    print("Output dimensions:")
    print(f"logits_per_video shape: {logits_per_video.shape}")  # 预期形状: [B, B] 代表视频与文本之间的相似度
    print(f"logits_per_text shape: {logits_per_text.shape}")    # 预期形状: [B, B] 代表文本与视频之间的相似度

    # 提取视频特征和文本特征
    video_features = model.encode_image(video_tensor)
    text_features = model.encode_text(text_tensor)

    print("Intermediate feature dimensions:")
    print(f"Video features shape: {video_features.shape}")  # 预期形状: [B, embed_dim]
    print(f"Text features shape: {text_features.shape}")    # 预期形状: [B, embed_dim]

    # 计算视频和文本之间的余弦相似度
    cosine_similarity = torch.nn.functional.cosine_similarity(video_features, text_features)
    print("Cosine similarity between video features and text features:")
    print(cosine_similarity)  # 输出每个视频和文本之间的相似度得分

    torchinfo.summary(
    model,
    input_data=(video_tensor, text_tensor),  # 直接传递真实输入张量
    col_names=["input_size", "output_size", "num_params", "params_percent"],
    depth=3,
    device=device
)
