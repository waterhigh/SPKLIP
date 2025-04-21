# -*- coding: UTF-8 -*-
from collections import OrderedDict
from typing import Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torchinfo
import torchvision

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu3 = nn.ReLU(inplace=True)

        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu3(out)
        return out


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.flatten(start_dim=2).permute(2, 0, 1)
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)
        spatial_dim = x.shape[0]
        positional_embedding = self.positional_embedding[:spatial_dim, :]
        x = x + positional_embedding[:, None, :].to(x.dtype)
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
    def __init__(self, layers, output_dim, heads, input_resolution, width, input_channels=320):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        self.conv1 = nn.Conv2d(input_channels, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.relu3 = nn.ReLU(inplace=True)

        self.avgpool = nn.AvgPool2d(2)

        self._inplanes = width
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        h, w = input_resolution
        spatial_dim_h = h // 32
        spatial_dim_w = w // 32
        embed_dim = width * 32
        self.attnpool = AttentionPool2d(spatial_dim_h * spatial_dim_w, embed_dim, heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]
        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        def stem(x):
            x = self.relu1(self.bn1(self.conv1(x)))
            x = self.relu2(self.bn2(self.conv2(x)))
            x = self.relu3(self.bn3(self.conv3(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)

        return x


class BasicBlock(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=features,
            out_channels=features,
            kernel_size=3,
            padding=1,
            bias=True,
        )
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(
            in_channels=features,
            out_channels=features,
            kernel_size=3,
            padding=1,
            bias=True,
        )
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(
            in_channels=features,
            out_channels=features,
            kernel_size=3,
            padding=1,
            bias=True,
        )
        self.relu3 = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.conv3(out)
        return self.relu3(x + out)


class CALayer2(nn.Module):
    def __init__(self, in_channels):
        super(CALayer2, self).__init__()
        self.ca_block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * 2, 3, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(in_channels * 2, in_channels, 3, padding=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        weight = self.ca_block(x)
        return weight


class TEDAdapter(nn.Module):
    def __init__(self, d_model, reduction=4):
        super().__init__()
        self.d_model = d_model
        self.reduction = reduction
        self.te_conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=3,
            padding=1,
            groups=d_model
        )
        self.td_conv = nn.Conv2d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=3,
            padding=1,
            groups=d_model
        )
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(d_model, d_model // reduction, 1),
            nn.ReLU(),
            nn.Conv2d(d_model // reduction, d_model, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, T, D = x.shape
        te_out = self.te_conv(x.permute(0, 2, 1)).permute(0, 2, 1)
        x_diff = x[:, 1:] - x[:, :-1]
        x_diff = F.pad(x_diff, (0, 0, 0, 1))
        td_out = self.td_conv(
            x_diff.view(B * T, D, 1, 1)
        ).view(B, T, D)
        attn = self.ca(
            (te_out + td_out).view(B * T, D, 1, 1)
        ).view(B, T, D)
        return x + attn * (te_out + td_out)


class FeatureExtractor(nn.Module):
    def __init__(
            self, in_channels, features, out_channels, channel_step, num_of_layers=16
    ):
        super(FeatureExtractor, self).__init__()
        self.channel_step = channel_step
        self.conv0_0 = nn.Conv2d(
            in_channels=in_channels, out_channels=16, kernel_size=3, padding=1
        )
        self.conv0_1 = nn.Conv2d(
            in_channels=in_channels - 2 * channel_step,
            out_channels=16, kernel_size=3, padding=1
        )
        self.conv0_2 = nn.Conv2d(
            in_channels=in_channels - 4 * channel_step,
            out_channels=16, kernel_size=3, padding=1
        )
        self.conv0_3 = nn.Conv2d(
            in_channels=in_channels - 6 * channel_step,
            out_channels=16, kernel_size=3, padding=1
        )
        self.conv1_0 = nn.Conv2d(
            in_channels=16, out_channels=1, kernel_size=3, padding=1
        )
        self.conv1_1 = nn.Conv2d(
            in_channels=16, out_channels=1, kernel_size=3, padding=1
        )
        self.conv1_2 = nn.Conv2d(
            in_channels=16, out_channels=1, kernel_size=3, padding=1
        )
        self.conv1_3 = nn.Conv2d(
            in_channels=16, out_channels=1, kernel_size=3, padding=1
        )
        self.ca = CALayer2(in_channels=4)
        self.conv = nn.Conv2d(
            in_channels=4, out_channels=features, kernel_size=3, padding=1
        )
        self.relu = nn.ReLU()
        layers = []
        for _ in range(num_of_layers - 2):
            layers.append(BasicBlock(features=features))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        out_0 = self.conv1_0(self.relu(self.conv0_0(x)))
        out_1 = self.conv1_1(
            self.relu(self.conv0_1(x[:, self.channel_step:-self.channel_step, :, :]))
        )
        out_2 = self.conv1_2(
            self.relu(
                self.conv0_2(x[:, 2 * self.channel_step:-2 * self.channel_step, :, :])
            )
        )
        out_3 = self.conv1_3(
            self.relu(
                self.conv0_3(x[:, 3 * self.channel_step:-3 * self.channel_step, :, :])
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
        # self.resnet = ModifiedResNet(
        #     layers=resnet_layers,
        #     output_dim=frame_feature_dim,
        #     heads=transformer_heads,
        #     input_resolution=(240, 320),
        #     width=64,
        #     input_channels=input_channels
        # )
        self.transformer = TransformerEncoder(
            TransformerEncoderLayer(
                d_model=frame_feature_dim,
                nhead=transformer_heads,
                dim_feedforward=frame_feature_dim * 4,
                dropout=0.1
            ),
            num_layers=transformer_layers
        )
        self.extractor = FeatureExtractor(
            in_channels=61,
            features=64,
            out_channels=64,
            channel_step=1,
            num_of_layers=8
        )

        self.resnet = torchvision.models.resnet18(pretrained=False)
        # 修改输入通道数
        self.resnet.conv1 = nn.Conv2d(64, 64, kernel_size=7, stride=2, padding=3, bias=False)  # 输入通道改为64
        # 修改输出维度
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, frame_feature_dim)

        self.win_r = 30
        self.win_step = 45
        

    def forward(self, video_frames):
        B, T, C, H, W = video_frames.shape
        processed_blocks = []
        for b in range(B):
            batch_frames = video_frames[b]
            batch_frames = batch_frames.view(T * C, H, W)
            block0 = batch_frames[0: 2 * self.win_r + 1].unsqueeze(0)
            block1 = batch_frames[self.win_step: self.win_step + 2 * self.win_r + 1].unsqueeze(0)
            block2 = batch_frames[2 * self.win_step: 2 * self.win_step + 2 * self.win_r + 1].unsqueeze(0)
            block3 = batch_frames[3 * self.win_step: 3 * self.win_step + 2 * self.win_r + 1].unsqueeze(0)
            block4 = batch_frames[4 * self.win_step: 4 * self.win_step + 2 * self.win_r + 1].unsqueeze(0)
            block0_out, est0 = self.extractor(block0)
            block1_out, est1 = self.extractor(block1)
            block2_out, est2 = self.extractor(block2)
            block3_out, est3 = self.extractor(block3)
            block4_out, est4 = self.extractor(block4)
            block_out = torch.stack([block0_out, block1_out, block2_out, block3_out, block4_out], dim=0)
            block_out = block_out.squeeze(1)
            processed_blocks.append(block_out.unsqueeze(0))
        processed_frames = torch.cat(processed_blocks, dim=0)
        processed_frames = processed_frames.squeeze(1)
        B, T, C, H, W = processed_frames.shape
        frame_features = []
        for t in range(T):
            frame_feature = self.resnet(processed_frames[:, t])
            frame_features.append(frame_feature)
        frame_features = torch.stack(frame_features, dim=1)
        frame_features = frame_features.permute(1, 0, 2)
        # temporal_features = self.transformer(frame_features)
        global_feature = frame_features.mean(dim=0)
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
        x1 = self.textad_fc1(x)
        x1 = self.textad_gelu(x1)
        x1 = self.textad_fc2(x1)
        x = x + x1
        return x


class LayerNorm(nn.LayerNorm):
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
            adapter_channels: int = 64
    ):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask
        self.text_adapter = TextAdapter(
            in_channels=d_model,
            adapter_channels=adapter_channels
        )

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        x = self.text_adapter(x)
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
                 image_resolution: tuple,
                 vision_layers: Tuple[int, int, int, int],
                 vision_width: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int,
                 input_channels: int = 64
                 ):
        super().__init__()
        self.context_length = context_length
        self.visual = ResNetWithTransformer(
            resnet_layers=vision_layers,
            input_channels=input_channels,
            frame_feature_dim=embed_dim,
            num_frames=64,
            transformer_layers=transformer_layers,
            transformer_heads=transformer_heads
        )
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
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)
        return mask

    @property
    def dtype(self):
        return self.visual.resnet.conv1.weight.dtype

    def encode_image(self, video_frames):
        return self.visual(video_frames)

    def encode_text(self, text):
        x = self.token_embedding(text).type(self.dtype)
        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        return x

    def forward(self, video_frames, text):
        video_features = self.encode_image(video_frames)
        text_features = self.encode_text(text)
        video_features = video_features / video_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        logit_scale = self.logit_scale.exp()
        logits_per_video = logit_scale * video_features @ text_features.t()
        logits_per_text = logits_per_video.t()
        return logits_per_video, logits_per_text


def build_model(state_dict: dict):
    counts = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
    vision_layers = tuple(counts)
    vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
    output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
    assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
    image_resolution = (output_width * 32, output_width * 32)
    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith("transformer.resblocks")))
    model = CLIP(
        embed_dim=embed_dim,
        image_resolution=image_resolution,
        vision_layers=vision_layers,
        vision_width=vision_width,
        input_channels=50,
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    batch_size = 1
    input_channels = 10
    num_frames = 25
    image_height = 240
    image_width = 320
    context_length = 77
    vocab_size = 49408
    video_tensor = torch.randn(batch_size, num_frames, input_channels, image_height, image_width).to(device)
    text_tensor = torch.randint(0, vocab_size, (batch_size, context_length)).to(device)
    embed_dim = 512
    vision_layers = (3, 4, 6, 3)
    vision_width = 64
    transformer_width = 256
    transformer_heads = 4
    transformer_layers = 8
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
    ).to(device)
    logits_per_video, logits_per_text = model(video_tensor, text_tensor)
    video_features = model.encode_image(video_tensor)
    text_features = model.encode_text(text_tensor)
    cosine_similarity = torch.nn.functional.cosine_similarity(video_features, text_features)
    torchinfo.summary(
        model,
        input_data=(video_tensor, text_tensor),
        col_names=["input_size", "output_size", "num_params", "params_percent"],
        depth=3,
        device=device
    )
    cosine_similarity = torch.nn.functional.cosine_similarity(video_features, text_features)
    print("Cosine similarity between video features and text features:")
    print(cosine_similarity)    
