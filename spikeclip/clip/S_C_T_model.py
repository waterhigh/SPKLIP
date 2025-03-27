from collections import OrderedDict
from typing import Tuple, Union
# import math
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torchinfo
from S_T import *
# from align_arch import *

thresh = 0.5  # neuronal threshold
lens = 0.5  # hyper-parameters of approximate function
decay = 0.25  # decay constants
time_window = 2
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
        grad_input = grad_output.clone()
        temp = abs(input - thresh) < lens
        temp = temp / (2 * lens)
        return grad_input * temp.float()


act_fun = ActFun.apply
# membrane potential update


class mem_update(nn.Module):

    def __init__(self):
        super(mem_update, self).__init__()

    def forward(self, x):
        mem = torch.zeros_like(x[0]).to(device)
        spike = torch.zeros_like(x[0]).to(device)
        output = torch.zeros_like(x)
        mem_old = 0
        for i in range(time_window):
            if i >= 1:
                mem = mem_old * decay * (1 - spike.detach()) + x[i]
            else:
                mem = x[i]
            spike = act_fun(mem)
            mem_old = mem.clone()
            output[i] = spike
        return output


class batch_norm_2d(nn.Module):
    """TDBN"""
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(batch_norm_2d, self).__init__()
        self.bn = BatchNorm3d1(num_features)

    def forward(self, input):
        y = input.transpose(0, 2).contiguous().transpose(0, 1).contiguous()
        y = self.bn(y)
        return y.contiguous().transpose(0, 1).contiguous().transpose(0, 2)


class batch_norm_2d1(nn.Module):
    """TDBN-Zero init"""
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(batch_norm_2d1, self).__init__()
        self.bn = BatchNorm3d2(num_features)

    def forward(self, input):
        y = input.transpose(0, 2).contiguous().transpose(0, 1).contiguous()
        y = self.bn(y)
        return y.contiguous().transpose(0, 1).contiguous().transpose(0, 2)


class BatchNorm3d1(torch.nn.BatchNorm3d):

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            nn.init.constant_(self.weight, thresh)
            nn.init.zeros_(self.bias)


class BatchNorm3d2(torch.nn.BatchNorm3d):

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            nn.init.constant_(self.weight, 0)
            nn.init.zeros_(self.bias)


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
                 marker='b'):
        super(Snn_Conv2d,
              self).__init__(in_channels, out_channels, kernel_size, stride,
                             padding, dilation, groups, bias, padding_mode)
        self.marker = marker

    def forward(self, input):
        weight = self.weight
        h = (input.size()[3] - self.kernel_size[0] +
             2 * self.padding[0]) // self.stride[0] + 1
        w = (input.size()[4] - self.kernel_size[0] +
             2 * self.padding[0]) // self.stride[0] + 1
        c1 = torch.zeros(time_window,
                         input.size()[1],
                         self.out_channels,
                         h,
                         w,
                         device=input.device)
        for i in range(time_window):
            c1[i] = F.conv2d(input[i], weight, self.bias, self.stride,
                             self.padding, self.dilation, self.groups)
        return c1
    
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()
        # 主路径卷积全部使用 stride=1，下采样改由 avgpool 完成
        self.conv1 = Snn_Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False)
        self.bn1 = batch_norm_2d(planes)
        
        self.conv2 = Snn_Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = batch_norm_2d(planes)
        
        self.avgpool = nn.AvgPool3d(kernel_size=(1, stride, stride)) if stride > 1 else nn.Identity()
        
        self.conv3 = Snn_Conv2d(planes, planes * self.expansion, kernel_size=1, stride=1, bias=False)
        self.bn3 = batch_norm_2d(planes * self.expansion)
        
        self.mem_update = mem_update()

        # 下采样路径同样使用 avgpool 下采样
        self.downsample = None
        if stride > 1 or inplanes != planes * self.expansion:
            self.downsample = nn.Sequential(
                Snn_Conv2d(inplanes, planes * self.expansion, kernel_size=1, stride=1, bias=False),
                nn.AvgPool3d(kernel_size=(1, stride, stride)),
                batch_norm_2d(planes * self.expansion)
            )

    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.avgpool(out)  # 主路径下采样
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.mem_update(out)

        if self.downsample is not None:
            identity = self.downsample(x)  # 下采样路径也通过 avgpool

        out += identity  # 此时维度一致
        return out

class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        # 初始化 positional_embedding
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        # Flatten the input tensor: (B, C, H, W) -> (HW, B, C)
        x = x.flatten(start_dim=2).permute(2, 0, 1)  # (HW, B, C)
        
        # Concatenate mean value as the first token
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1, B, C)

        # Get the spatial size
        spatial_dim = x.shape[0]  # This is the number of tokens (HW+1)

        # Ensure positional embedding matches the spatial dimension of the input
        positional_embedding = self.positional_embedding[:spatial_dim, :]

        # Add positional embedding to the input
        x = x + positional_embedding[:, None, :].to(x.dtype)

        # Apply multi-head attention
        x, _ = F.multi_head_attention_forward(
            query=x[:1], key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0.1,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        return x.squeeze(0)



class ModifiedResNet(nn.Module):
    def __init__(self, layers, output_dim, heads, input_resolution, width, input_channels=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # 修改1：替换为SNN兼容的3-layer stem
        self.conv1 = Snn_Conv2d(input_channels, width//2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = batch_norm_2d(width//2)  # TDBN
        
        self.conv2 = Snn_Conv2d(width//2, width//2, kernel_size=3, padding=1, bias=False)
        self.bn2 = batch_norm_2d(width//2)
        
        self.conv3 = Snn_Conv2d(width//2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = batch_norm_2d(width)

        self.mem_update_stem = mem_update()  # 添加脉冲激活
        self.avgpool = nn.AvgPool3d(kernel_size=(1,2,2))  # 修改为3D池化

        # 修改2：残差层适配SNN
        self._inplanes = width
        self.layer1 = self._make_snn_layer(width, layers[0])
        self.layer2 = self._make_snn_layer(width*2, layers[1], stride=2)
        self.layer3 = self._make_snn_layer(width*4, layers[2], stride=2)
        self.layer4 = self._make_snn_layer(width*8, layers[3], stride=2)

        # 修改3：SNN兼容的注意力池化
        h, w = input_resolution
        spatial_dim_h = h // 32
        spatial_dim_w = w // 32
        embed_dim = width * 32

        self.final_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # 全局平均池化 (也可换成nn.AdaptiveMaxPool2d)
            nn.Flatten(),
            nn.Linear(width * 8 * Bottleneck.expansion, output_dim)  # 注意输入维度匹配
        )

    def _make_snn_layer(self, planes, blocks, stride=1):
        """创建SNN兼容的残差层"""
        layers = [Bottleneck(self._inplanes, planes, stride)]
        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        # 输入维度扩展：添加时间窗口 [T,B,C,H,W]
        x = x.unsqueeze(0).expand(time_window, *(-1,)*4)  # [T,B,C,H,W]
        
        def stem(x):
            x = self.conv1(x)       # [T,B,C,H,W]
            x = self.bn1(x)         # TDBN处理
            x = self.conv2(x)       # [T,B,C,H,W]
            x = self.bn2(x)
            x = self.conv3(x)       # [T,B,C,H,W]
            x = self.bn3(x)
            x = self.avgpool(x)     # 时间维度保留
            return self.mem_update_stem(x)  # 脉冲激活

        x = x.type(self.conv1.weight.dtype)  # 确保数据类型匹配
        x = stem(x)                 # [T,B,C,H,W]
        x = self.layer1(x)          # [T,B,C,H,W]
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # 时空特征聚合
        x = x.mean(dim=0)           # 时间维度平均 [B,output_dim]
        x = self.final_pool(x)        # [B,output_dim]
           
        return x
    
class BasicBlock(nn.Module):
    def __init__(self, features):
        super().__init__()
        # 所有卷积替换为SNN兼容版本
        self.conv1 = Snn_Conv2d(features, features, kernel_size=3, padding=1, bias=True)
        self.bn1 = batch_norm_2d(features)
        
        self.conv2 = Snn_Conv2d(features, features, kernel_size=3, padding=1, bias=True)
        self.bn2 = batch_norm_2d(features)
        
        self.conv3 = Snn_Conv2d(features, features, kernel_size=3, padding=1, bias=True)
        self.bn3 = batch_norm_2d(features)
        
        self.mem_update = mem_update()  # 脉冲激活层

    def forward(self, x):
        # 输入形状: [T=6, B, C, H, W]
        identity = x
        
        out = self.conv1(x)       # [6,B,C,H,W]
        out = self.bn1(out)
        out = self.mem_update(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.mem_update(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        return self.mem_update(out + identity)

class CALayer2(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        # 注意力模块SNN化
        self.ca_conv1 = Snn_Conv2d(in_channels, in_channels*2, 3, padding=1)
        self.bn1 = batch_norm_2d(in_channels*2)
        self.ca_conv2 = Snn_Conv2d(in_channels*2, in_channels, 3, padding=1)
        self.bn2 = batch_norm_2d(in_channels)
        self.mem_update_sigmoid = mem_update()  # 用脉冲近似Sigmoid

    def forward(self, x):
        # x形状: [T,B,C,H,W]
        T, B, C, H, W = x.shape
        # 时间维度平均聚合
        x_mean = x.mean(dim=0)  # [B,C,H,W]
        x_mean = x_mean.unsqueeze(0).expand(T, B, C, H, W)  # 广播回[T,B,C,H,W]
        
        weight = self.ca_conv1(x_mean)
        weight = self.bn1(weight)
        weight = self.mem_update_sigmoid(weight)
        
        weight = self.ca_conv2(weight)
        weight = self.bn2(weight)
        return self.mem_update_sigmoid(weight)  # 模拟Sigmoid激活
    

class FeatureExtractor(nn.Module):
    def __init__(self, in_channels, features, out_channels, channel_step, num_of_layers=16):
        super().__init__()
        self.channel_step = channel_step
        self.time_window = 6  # 新增时间维度参数
        
        # 初始卷积组SNN化
        self.conv0_0 = Snn_Conv2d(in_channels, 16, kernel_size=3, padding=1)
        self.bn0_0 = batch_norm_2d(16)
        self.mem_update_conv0_0 = mem_update()
        
        self.conv0_1 = Snn_Conv2d(in_channels-2*channel_step, 16, kernel_size=3, padding=1)
        self.bn0_1 = batch_norm_2d(16)
        self.mem_update_conv0_1 = mem_update()
        
        self.conv0_2 = Snn_Conv2d(in_channels-4*channel_step, 16, kernel_size=3, padding=1)
        self.bn0_2 = batch_norm_2d(16)
        self.mem_update_conv0_2 = mem_update()
        
        self.conv0_3 = Snn_Conv2d(in_channels-6*channel_step, 16, kernel_size=3, padding=1)
        self.bn0_3 = batch_norm_2d(16)
        self.mem_update_conv0_3 = mem_update()
        
        # 第二层卷积SNN化
        self.conv1_0 = Snn_Conv2d(16, 1, kernel_size=3, padding=1)
        self.conv1_1 = Snn_Conv2d(16, 1, kernel_size=3, padding=1)
        self.conv1_2 = Snn_Conv2d(16, 1, kernel_size=3, padding=1)
        self.conv1_3 = Snn_Conv2d(16, 1, kernel_size=3, padding=1)
        
        self.ca = CALayer2(in_channels=4)
        self.conv = Snn_Conv2d(4, features, kernel_size=3, padding=1)
        self.bn_conv = batch_norm_2d(features)
        self.mem_update_main = mem_update()
        
        # 构建SNN残差块
        layers = []
        for _ in range(num_of_layers - 2):
            layers.append(BasicBlock(features))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # 输入维度扩展: [1,61,240,320] -> [6,1,61,240,320]
        x = x.unsqueeze(0).expand(self.time_window, -1, -1, -1, -1)
        
        # 第一组卷积处理
        out0 = self.mem_update_conv0_0(self.bn0_0(self.conv0_0(x)))
        out1 = self.mem_update_conv0_1(self.bn0_1(self.conv0_1(x[:, :, self.channel_step:-self.channel_step])))
        out2 = self.mem_update_conv0_2(self.bn0_2(self.conv0_2(x[:, :, 2*self.channel_step:-2*self.channel_step])))
        out3 = self.mem_update_conv0_3(self.bn0_3(self.conv0_3(x[:, :, 3*self.channel_step:-3*self.channel_step])))
        
        # 第二层卷积
        out0 = self.conv1_0(out0)
        out1 = self.conv1_1(out1)
        out2 = self.conv1_2(out2)
        out3 = self.conv1_3(out3)
        
        # 拼接与注意力
        out = torch.cat([out0, out1, out2, out3], dim=2)  # 在通道维度拼接
        est = out
        weight = self.ca(out)
        out = weight * out  # 注意力加权
        
        # 主路径处理
        out = self.mem_update_main(self.bn_conv(self.conv(out)))
        tmp = out
        out = self.net(out)

        out = out + tmp
        out = out.mean(dim=0)  # [1,64,240,320]
        est = est.mean(dim=0)

        return out, est


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
        
        # # Transformer 部分
        # self.transformer = TransformerEncoder(
        #     TransformerEncoderLayer(
        #         d_model=frame_feature_dim,  # 帧级特征的维度
        #         nhead=transformer_heads,
        #         dim_feedforward=frame_feature_dim * 4,  # 前馈网络的维度
        #         dropout=0.1
        #     ),
        #     num_layers=transformer_layers
        # )

        # spiking-transformer
        self.transformer = PureSpikeTransformer(
                input_dim=frame_feature_dim, 
                num_heads=transformer_heads,  
                num_layers=transformer_layers   
            )
        

        # Feature Extractor 部分
        self.extractor = FeatureExtractor(
            in_channels=61,  
            features=64, 
            out_channels=64, 
            channel_step=1, 
            num_of_layers=3
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
            # block_out = block_out.view(1, 5 * 64, H, W)
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
    embed_dim = 256  # 嵌入维度
    vision_layers = (2, 2, 2, 2)  # ResNet 的层数配置
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


    # 节省40%显存，精度损失<0.5%（网页5的推荐方案）

    from torch.cuda.amp import autocast
    with autocast():
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
        
        max_mem = torch.cuda.max_memory_allocated() / 1024**2  
        print(f"最大显存占用：{max_mem:.2f} MB")
