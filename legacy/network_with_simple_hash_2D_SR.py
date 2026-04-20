import torch
import torch.nn as nn
import tinycudann as tcnn
from siren_pytorch import Sine


class SuperResolutionNet(nn.Module):
    """
    轻量级单通道超分辨率网络
    基于ESPCN (Efficient Sub-Pixel CNN) 架构，适合单通道输入
    支持任意整数倍上采样（通过子像素卷积实现）
    """
    def __init__(self, scale_factor=2, num_channels=1, num_feat=64, num_block=4):
        """
        Args:
            scale_factor: 上采样倍数（必须是整数）
            num_channels: 输入通道数（默认1，单通道）
            num_feat: 特征通道数
            num_block: 残差块数量
        """
        super().__init__()
        self.scale_factor = scale_factor
        
        # 特征提取
        self.conv_first = nn.Conv2d(num_channels, num_feat, 3, 1, 1)
        
        # 残差块
        body = []
        for _ in range(num_block):
            body.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1))
            body.append(nn.BatchNorm2d(num_feat))
            body.append(nn.ReLU(inplace=True))
        self.body = nn.Sequential(*body)
        
        # 上采样层（使用子像素卷积）
        self.conv_up = nn.Conv2d(num_feat, num_channels * (scale_factor ** 2), 3, 1, 1)
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Args:
            x: 输入张量，形状为 (B, C, H, W) 或 (B, H, W) 或 (H, W)
        Returns:
            上采样后的张量，形状为 (B, C, H*scale, W*scale) 或 (H*scale, W*scale)
        """
        # 处理不同的输入形状
        original_shape = x.shape
        if len(original_shape) == 2:
            # (H, W) -> (1, 1, H, W)
            x = x.unsqueeze(0).unsqueeze(0)
            need_squeeze = True
        elif len(original_shape) == 3:
            # (B, H, W) -> (B, 1, H, W)
            x = x.unsqueeze(1)
            need_squeeze = True
        else:
            # (B, C, H, W)
            need_squeeze = False
        
        # 前向传播
        feat = self.conv_first(x)
        feat = self.body(feat)
        out = self.conv_up(feat)
        out = self.pixel_shuffle(out)
        
        # 恢复原始形状
        if need_squeeze:
            if len(original_shape) == 2:
                out = out.squeeze(0).squeeze(0)  # (H*scale, W*scale)
            else:
                out = out.squeeze(1)  # (B, H*scale, W*scale)
        
        return out


class MLPRenderer(nn.Module):
    """
    Multi-Layer Perceptron (MLP) Renderer for mapping input feature vectors to outputs (e.g., pixel values).
    """
    def __init__(
        self, 
        in_dim=32, 
        hidden_dim=64, 
        num_layers=2, 
        out_dim=1, 
        use_layer_norm=True
    ):
        super().__init__()
        # activation = nn.ReLU()
        # activation = nn.LeakyReLU(0.2)
        activation = Sine()
        layers = []
        layers.append(nn.Linear(in_dim, hidden_dim))
        if use_layer_norm:
            layers.append(nn.LayerNorm(hidden_dim, elementwise_affine=False))
        layers.append(activation)

        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim, elementwise_affine=False))
            layers.append(activation)

        layers.append(nn.Linear(hidden_dim, out_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class Implicit2D(nn.Module):
    """
    2D Implicit Feature Representation:
    - Contains only a 2D feature grid (xy_features)
    - No temporal dimension
    - Directly outputs image/field distribution through MLP renderer
    """
    def __init__(
        self, 
        use_layer_norm=False,
        n_levels=16,              # 多分辨率哈希编码的层数
        n_features_per_level=2,   # 每层的特征数
        log2_hashmap_size=19,     # 哈希表大小 2^19
        base_resolution=16,       # 最粗糙层的分辨率
        per_level_scale=1.5,      # 每层分辨率增长因子
        use_tcnn_mlp=True         # 是否使用 tcnn.Network 作为渲染器
    ):
        super().__init__()
        
        # 特征维度由哈希编码参数自动计算
        self.num_features = n_levels * n_features_per_level
        
        self.xy_hash_encoder = tcnn.Encoding(
            n_input_dims=2,  # 输入是 2D 坐标 (x, y)
            encoding_config={
                "otype": "HashGrid",
                "n_levels": n_levels,
                "n_features_per_level": n_features_per_level,
                "log2_hashmap_size": log2_hashmap_size,
                "base_resolution": base_resolution,
                "per_level_scale": per_level_scale
            }
        )

        # Renderer: 选择使用 tcnn.Network 或 PyTorch MLPRenderer
        if use_tcnn_mlp:
            # 使用 CUDA 优化的 Fully Fused MLP（更快）
            self.mlp_renderer = tcnn.Network(
                n_input_dims=self.num_features,
                n_output_dims=1,
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": 32,
                    "n_hidden_layers": 2
                }
            )
        else:
            # 使用 PyTorch MLPRenderer（更灵活，支持 LayerNorm）
            self.mlp_renderer = MLPRenderer(in_dim=self.num_features, use_layer_norm=use_layer_norm)

        # Will initialize normalized coordinates for hash encoding later
        self.xy_coords = None

    def init_2d_coords(self, output_width, output_height):
        """
        Initialize normalized coordinates for hash encoding.
        坐标已经归一化到 [0, 1] 范围。
        """
        half_dx = 0.5 / output_width
        half_dy = 0.5 / output_height
        xs = torch.linspace(half_dx, 1 - half_dx, output_width)
        ys = torch.linspace(half_dy, 1 - half_dy, output_height)
        xv, yv = torch.meshgrid([xs, ys], indexing="ij")
        coords = torch.stack((yv.flatten(), xv.flatten())).t()  # [N, 2], 归一化到 [0, 1]

        self.output_width = output_width
        self.output_height = output_height
        self.xy_coords = nn.Parameter(coords[None], requires_grad=False)  # [1, N, 2]

    def forward(self):
        """
        Forward propagation: directly render 2D features through MLP
        Output shape: (output_width, output_height)
        """
        xy_coords_normalized = self.xy_coords.squeeze(0)  # [N, 2]
        
        # 确保坐标在 CUDA 上（tinycudann 要求输入在 CUDA）
        if not xy_coords_normalized.is_cuda:
            xy_coords_normalized = xy_coords_normalized.cuda()
        
        # 通过哈希编码器获取多尺度空间特征
        spatial_features = self.xy_hash_encoder(xy_coords_normalized)  # [N, num_features]
        
        # 转换为 float32 以确保与 MLP 兼容（修复 autocast 时的 dtype 不匹配）
        spatial_features = spatial_features.float()
        
        # 通过 MLP 渲染器生成输出
        if isinstance(self.mlp_renderer, tcnn.Network):
            output = self.mlp_renderer(spatial_features)  # [N, 1]
            output = output.reshape(self.output_width, self.output_height, 1)
        else:
            output = self.mlp_renderer(spatial_features)  # [N, 1]
            output = output.reshape(self.output_width, self.output_height, 1)
        
        output = output.squeeze(-1)  # [output_width, output_height]
        return output


class ComplexINRModel2D(nn.Module):
    """
    2D Complex Implicit Neural Representation Model:
    Simultaneously generates real and imaginary parts (or amplitude and phase) of images.
    """
    def __init__(
        self, 
        output_width, 
        output_height, 
        downsample_factor, 
        use_layer_norm,
        update_probe=True,
        probe_width=None,  # probe的宽度
        probe_height=None,  # probe的高度
        probe_pattern=None,  # 初始probe pattern（复数张量）
        share_spatial_features=False,  # 是否让实部和虚部共享同一个哈希编码器
        # Instant NGP 哈希编码参数
        n_levels=16,
        n_features_per_level=2,
        log2_hashmap_size=19,
        base_resolution=16,
        per_level_scale=1.5,
        use_tcnn_mlp=True
    ):
        super().__init__()
        self.update_probe = update_probe
        self.share_spatial_features = share_spatial_features
        
        # 如果没有指定probe尺寸，使用默认值
        self.probe_width = probe_width if probe_width is not None else output_width
        self.probe_height = probe_height if probe_height is not None else output_height
        
        # 初始化probe pattern和缩放参数
        if update_probe:
            if probe_pattern is not None:
                # 有初始probe pattern：使用 k*network_output + probe_pattern 的形式
                self.probe_pattern = probe_pattern
                # Probe的标量缩放参数
                self.k_real = nn.Parameter(torch.tensor(1e-10))  # 实部缩放
                self.k_imag = nn.Parameter(torch.tensor(1e-10))  # 虚部缩放
            else:
                # 没有初始probe pattern：直接输出网络生成的实部和虚部
                self.probe_pattern = None
                self.k_real = None
                self.k_imag = None
        else:
            self.probe_pattern = None
            self.k_real = None
            self.k_imag = None
        
        # 哈希编码参数字典
        hash_params = {
            'n_levels': n_levels,
            'n_features_per_level': n_features_per_level,
            'log2_hashmap_size': log2_hashmap_size,
            'base_resolution': base_resolution,
            'per_level_scale': per_level_scale,
            'use_tcnn_mlp': use_tcnn_mlp
        }
        
        self.num_features = n_levels * n_features_per_level
        
        if share_spatial_features:
            # 共享模式：实部和虚部共享同一个哈希编码器，但有各自的 MLP renderer
            # Object 的共享哈希编码器
            self.object_shared_encoder = tcnn.Encoding(
                n_input_dims=2,
                encoding_config={
                    "otype": "HashGrid",
                    "n_levels": n_levels,
                    "n_features_per_level": n_features_per_level,
                    "log2_hashmap_size": log2_hashmap_size,
                    "base_resolution": base_resolution,
                    "per_level_scale": per_level_scale
                }
            )
            
            # 为实部和虚部各创建独立的 MLP renderer
            if use_tcnn_mlp:
                self.object_real_mlp = tcnn.Network(
                    n_input_dims=self.num_features,
                    n_output_dims=1,
                    network_config={
                        "otype": "FullyFusedMLP",
                        "activation": "ReLU",
                        "output_activation": "None",
                        "n_neurons": 32,
                        "n_hidden_layers": 2
                    }
                )
                self.object_imag_mlp = tcnn.Network(
                    n_input_dims=self.num_features,
                    n_output_dims=1,
                    network_config={
                        "otype": "FullyFusedMLP",
                        "activation": "ReLU",
                        "output_activation": "None",
                        "n_neurons": 32,
                        "n_hidden_layers": 2
                    }
                )
            else:
                self.object_real_mlp = MLPRenderer(in_dim=self.num_features, use_layer_norm=use_layer_norm)
                self.object_imag_mlp = MLPRenderer(in_dim=self.num_features, use_layer_norm=use_layer_norm)
            
            self.object_real_part_generator = None
            self.object_imag_part_generator = None
            
            # Probe 的共享哈希编码器（如果需要）
            if update_probe:
                self.probe_shared_encoder = tcnn.Encoding(
                    n_input_dims=2,
                    encoding_config={
                        "otype": "HashGrid",
                        "n_levels": n_levels,
                        "n_features_per_level": n_features_per_level,
                        "log2_hashmap_size": log2_hashmap_size,
                        "base_resolution": base_resolution,
                        "per_level_scale": per_level_scale
                    }
                )
                
                if use_tcnn_mlp:
                    self.probe_real_mlp = tcnn.Network(
                        n_input_dims=self.num_features,
                        n_output_dims=1,
                        network_config={
                            "otype": "FullyFusedMLP",
                            "activation": "ReLU",
                            "output_activation": "None",
                            "n_neurons": 32,
                            "n_hidden_layers": 2
                        }
                    )
                    self.probe_imag_mlp = tcnn.Network(
                        n_input_dims=self.num_features,
                        n_output_dims=1,
                        network_config={
                            "otype": "FullyFusedMLP",
                            "activation": "ReLU",
                            "output_activation": "None",
                            "n_neurons": 32,
                            "n_hidden_layers": 2
                        }
                    )
                else:
                    self.probe_real_mlp = MLPRenderer(in_dim=self.num_features, use_layer_norm=use_layer_norm)
                    self.probe_imag_mlp = MLPRenderer(in_dim=self.num_features, use_layer_norm=use_layer_norm)
                
                self.probe_real_part_generator = None
                self.probe_imag_part_generator = None
            else:
                self.probe_shared_encoder = None
                self.probe_real_mlp = None
                self.probe_imag_mlp = None
                self.probe_real_part_generator = None
                self.probe_imag_part_generator = None
        else:
            # 独立模式：实部和虚部使用完全独立的 Implicit2D
            self.object_real_part_generator = Implicit2D(
                use_layer_norm=use_layer_norm,
                **hash_params
            )
            self.object_imag_part_generator = Implicit2D(
                use_layer_norm=use_layer_norm,
                **hash_params
            )
            
            self.object_shared_encoder = None
            self.object_real_mlp = None
            self.object_imag_mlp = None
            
            # Probe feature generators (only create if update_probe is True)
            if update_probe:
                self.probe_real_part_generator = Implicit2D(
                    use_layer_norm=use_layer_norm,
                    **hash_params
                )
                self.probe_imag_part_generator = Implicit2D(
                    use_layer_norm=use_layer_norm,
                    **hash_params
                )
                self.probe_shared_encoder = None
                self.probe_real_mlp = None
                self.probe_imag_mlp = None
            else:
                self.probe_real_part_generator = None
                self.probe_imag_part_generator = None
                self.probe_shared_encoder = None
                self.probe_real_mlp = None
                self.probe_imag_mlp = None

        self.output_width = output_width
        self.output_height = output_height

        self.init_scale_grids(downsample_factor)

    def init_scale_grids(self, downsample_factor):
        """
        Initialize downsampling/upsampling grid coordinates and create upsampling module.
        """
        downsampled_width = self.output_width // downsample_factor
        downsampled_height = self.output_height // downsample_factor
        
        if self.share_spatial_features:
            # 共享模式：初始化坐标参数供共享编码器使用
            # Object 坐标
            half_dx = 0.5 / downsampled_width
            half_dy = 0.5 / downsampled_height
            xs = torch.linspace(half_dx, 1 - half_dx, downsampled_width)
            ys = torch.linspace(half_dy, 1 - half_dy, downsampled_height)
            xv, yv = torch.meshgrid([xs, ys], indexing="ij")
            coords = torch.stack((yv.flatten(), xv.flatten())).t()  # [N, 2]
            
            self.downsampled_width = downsampled_width
            self.downsampled_height = downsampled_height
            self.xy_coords = nn.Parameter(coords, requires_grad=False)  # [N, 2]
            
            # Probe 坐标（使用指定的probe尺寸）
            if self.update_probe:
                half_dx_probe = 0.5 / self.probe_width
                half_dy_probe = 0.5 / self.probe_height
                xs_probe = torch.linspace(half_dx_probe, 1 - half_dx_probe, self.probe_width)
                ys_probe = torch.linspace(half_dy_probe, 1 - half_dy_probe, self.probe_height)
                xv_probe, yv_probe = torch.meshgrid([xs_probe, ys_probe], indexing="ij")
                probe_coords = torch.stack((yv_probe.flatten(), xv_probe.flatten())).t()  # [N_probe, 2]
                self.probe_coords = nn.Parameter(probe_coords, requires_grad=False)  # [N_probe, 2]
            else:
                self.probe_coords = None
        else:
            # 独立模式：使用原有的 Implicit2D 初始化方式
            self.object_real_part_generator.init_2d_coords(
                output_width=downsampled_width,
                output_height=downsampled_height
            )
            self.object_imag_part_generator.init_2d_coords(
                output_width=downsampled_width,
                output_height=downsampled_height
            )

            # Initialize probe grids (only if probe generators exist)
            if self.update_probe:
                self.probe_real_part_generator.init_2d_coords(
                    output_width=self.probe_width,
                    output_height=self.probe_height
                )
                self.probe_imag_part_generator.init_2d_coords(
                    output_width=self.probe_width,
                    output_height=self.probe_height
                )
            
            self.downsampled_width = downsampled_width
            self.downsampled_height = downsampled_height
            self.xy_coords = None
            self.probe_coords = None
            
        self.downsample_factor = downsample_factor
        # 使用超分辨率网络替代简单的双线性上采样
        self.upsample = SuperResolutionNet(
            scale_factor=downsample_factor,
            num_channels=1,
            num_feat=64,
            num_block=4
        )

    def forward(self):
        """
        Forward propagation:
        1. Generate real and imaginary parts at low-resolution through generators
        2. Upsample to original resolution
        
        Returns:
            Tuple of (object_real, object_imag, probe_real, probe_imag)
            Shape: (W, H) for object, (probe_width, probe_height) for probe
            If update_probe=False, probe_real and probe_imag will be None
        """
        if self.share_spatial_features:
            # 共享模式：使用共享的编码器 + 独立的 MLP
            xy_coords = self.xy_coords
            if not xy_coords.is_cuda:
                xy_coords = xy_coords.cuda()
            
            # Object: 共享编码器，独立 MLP
            spatial_features = self.object_shared_encoder(xy_coords)  # [N, num_features]
            # 转换为 float32 以确保与 MLP 兼容（修复 autocast 时的 dtype 不匹配）
            spatial_features = spatial_features.float()
            
            if isinstance(self.object_real_mlp, tcnn.Network):
                object_real_part = self.object_real_mlp(spatial_features)  # [N, 1]
                object_imag_part = self.object_imag_mlp(spatial_features)  # [N, 1]
                object_real_part = object_real_part.reshape(self.downsampled_width, self.downsampled_height)
                object_imag_part = object_imag_part.reshape(self.downsampled_width, self.downsampled_height)
            else:
                object_real_part = self.object_real_mlp(spatial_features)  # [N, 1]
                object_imag_part = self.object_imag_mlp(spatial_features)  # [N, 1]
                object_real_part = object_real_part.reshape(self.downsampled_width, self.downsampled_height, 1).squeeze(-1)
                object_imag_part = object_imag_part.reshape(self.downsampled_width, self.downsampled_height, 1).squeeze(-1)
            
            # Upsample to full resolution (超分辨率网络会自动处理形状)
            object_real_part = self.upsample(object_real_part)  # [W, H]
            object_imag_part = self.upsample(object_imag_part)  # [W, H]
            
            # Probe: 共享编码器，独立 MLP（如果启用）
            if self.update_probe:
                probe_coords_local = self.probe_coords
                if not probe_coords_local.is_cuda:
                    probe_coords_local = probe_coords_local.cuda()
                
                probe_features = self.probe_shared_encoder(probe_coords_local)  # [N_probe, num_features]
                # 转换为 float32 以确保与 MLP 兼容（修复 autocast 时的 dtype 不匹配）
                probe_features = probe_features.float()
                
                if isinstance(self.probe_real_mlp, tcnn.Network):
                    probe_real_part = self.probe_real_mlp(probe_features)
                    probe_imag_part = self.probe_imag_mlp(probe_features)
                    probe_real_part = probe_real_part.reshape(self.probe_width, self.probe_height)
                    probe_imag_part = probe_imag_part.reshape(self.probe_width, self.probe_height)
                else:
                    probe_real_part = self.probe_real_mlp(probe_features)
                    probe_imag_part = self.probe_imag_mlp(probe_features)
                    probe_real_part = probe_real_part.reshape(self.probe_width, self.probe_height, 1).squeeze(-1)
                    probe_imag_part = probe_imag_part.reshape(self.probe_width, self.probe_height, 1).squeeze(-1)
                
                # 如果有初始probe pattern，叠加之（带缩放）；否则直接输出网络结果
                if self.probe_pattern is not None:
                    probe_real_part = self.k_real * probe_real_part + self.probe_pattern.real.to(probe_real_part.device)
                    probe_imag_part = self.k_imag * probe_imag_part + self.probe_pattern.imag.to(probe_imag_part.device)
            else:
                probe_real_part = None
                probe_imag_part = None
        else:
            # 独立模式：使用完全独立的 Implicit2D
            object_real_part = self.object_real_part_generator()  # [W_ds, H_ds]
            object_imag_part = self.object_imag_part_generator()  # [W_ds, H_ds]
            
            # Upsample to full resolution (超分辨率网络会自动处理形状)
            object_real_part = self.upsample(object_real_part)  # [W, H]
            object_imag_part = self.upsample(object_imag_part)  # [W, H]

            # Generate probe fields (only if enabled)
            if self.update_probe:
                probe_real_part = self.probe_real_part_generator()  # [probe_width, probe_height]
                probe_imag_part = self.probe_imag_part_generator()  # [probe_width, probe_height]
                
                # 如果有初始probe pattern，叠加之（带缩放）；否则直接输出网络结果
                if self.probe_pattern is not None:
                    probe_real_part = self.k_real * probe_real_part + self.probe_pattern.real.to(probe_real_part.device)
                    probe_imag_part = self.k_imag * probe_imag_part + self.probe_pattern.imag.to(probe_imag_part.device)
            else:
                probe_real_part = None
                probe_imag_part = None

        return object_real_part, object_imag_part, probe_real_part, probe_imag_part

if __name__ == "__main__":
    # 检查 CUDA 是否可用
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("=== Test 1: 2D Model with custom probe pattern (使用 k*x+b 模式) ===")
    # 创建一个自定义的probe pattern
    probe_width, probe_height = 128, 128
    custom_probe_real = torch.randn(probe_width, probe_height) * 0.1
    custom_probe_imag = torch.randn(probe_width, probe_height) * 0.1
    custom_probe_pattern = custom_probe_real + 1j * custom_probe_imag
    
    model_with_probe = ComplexINRModel2D(
        output_width=256,
        output_height=512,
        downsample_factor=2,
        use_layer_norm=False,
        update_probe=True,
        probe_width=probe_width,  # 指定probe的宽度
        probe_height=probe_height,  # 指定probe的高度
        probe_pattern=custom_probe_pattern,  # 使用自定义probe pattern
        # Instant NGP 参数（默认：16×2=32维特征）
        n_levels=16,
        n_features_per_level=2,
        use_tcnn_mlp=True  # 使用 CUDA 优化的 MLP 渲染器
    )
    print("Renderer type: tcnn.Network (Fully Fused MLP)")
    print("Probe mode: k*network_output + probe_pattern (有初始pattern)")
    
    # 将模型的 PyTorch 参数移到 CUDA
    model_with_probe = model_with_probe.to(device)
    
    object_real, object_imag, probe_real, probe_imag = model_with_probe()
    
    print(f"Object real part shape: {object_real.shape} (batch维已去除)")
    print(f"Object imag part shape: {object_imag.shape} (batch维已去除)")
    print(f"Probe real part shape: {probe_real.shape if probe_real is not None else 'None'}")
    print(f"Probe imag part shape: {probe_imag.shape if probe_imag is not None else 'None'}")
    print(f"Probe scaling parameters - k_real: {model_with_probe.k_real.item():.2e}, k_imag: {model_with_probe.k_imag.item():.2e}")
    
    trainable_params = sum(p.numel() for p in model_with_probe.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model_with_probe.parameters())
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Total parameters: {total_params:,}")
    
    print("\n=== Test 2: 2D Model with PyTorch MLPRenderer (more flexible) ===")
    model_without_probe = ComplexINRModel2D(
        output_width=256,
        output_height=512,
        downsample_factor=2,
        use_layer_norm=True,    # LayerNorm 只在 PyTorch MLPRenderer 中可用
        update_probe=False,  # Probe disabled
        # Instant NGP 参数（默认：16×2=32维特征）
        n_levels=16,
        n_features_per_level=2,
        use_tcnn_mlp=False      # 使用 PyTorch MLPRenderer
    )
    print("Renderer type: PyTorch MLPRenderer (with LayerNorm)")
    
    # 将模型移到 CUDA
    model_without_probe = model_without_probe.to(device)
    
    object_real, object_imag, probe_real, probe_imag = model_without_probe()
    
    print(f"Object real part shape: {object_real.shape} (batch维已去除)")
    print(f"Object imag part shape: {object_imag.shape} (batch维已去除)")
    print(f"Probe real part: {probe_real if probe_real is not None else 'None'}")
    print(f"Probe imag part: {probe_imag if probe_imag is not None else 'None'}")
    
    trainable_params = sum(p.numel() for p in model_without_probe.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model_without_probe.parameters())
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Total parameters: {total_params:,}")
    
    print("\n=== Test 3: 2D Model without probe pattern (直接输出网络结果) ===")
    # probe_pattern=None: 网络直接输出实部和虚部，不使用 k*x+b 的形式
    model_shared = ComplexINRModel2D(
        output_width=256,
        output_height=512,
        downsample_factor=2,
        use_layer_norm=False,
        update_probe=True,
        probe_width=128,  # 指定probe的宽度
        probe_height=128,  # 指定probe的高度
        probe_pattern=None,  # 不传入初始pattern: 直接输出网络生成的实部和虚部
        share_spatial_features=True,  # 共享空间特征编码器
        # Instant NGP 参数（默认：16×2=32维特征）
        n_levels=16,
        n_features_per_level=2,
        use_tcnn_mlp=True
    )
    print("Feature sharing: Enabled (real and imag parts share the same hash encoder)")
    print("Probe mode: Direct network output (no k*x+b, since probe_pattern=None)")
    print(f"Has k_real/k_imag: {model_shared.k_real is not None}")
    
    # 将模型移到 CUDA
    model_shared = model_shared.to(device)
    
    object_real, object_imag, probe_real, probe_imag = model_shared()
    
    print(f"Object real part shape: {object_real.shape} (batch维已去除)")
    print(f"Object imag part shape: {object_imag.shape} (batch维已去除)")
    print(f"Probe real part shape: {probe_real.shape if probe_real is not None else 'None'}")
    print(f"Probe imag part shape: {probe_imag.shape if probe_imag is not None else 'None'}")
    
    trainable_params = sum(p.numel() for p in model_shared.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model_shared.parameters())
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Total parameters: {total_params:,}")
    
    print("\n=== Parameter Comparison ===")
    params_test1 = sum(p.numel() for p in model_with_probe.parameters() if p.requires_grad)
    params_test3 = sum(p.numel() for p in model_shared.parameters() if p.requires_grad)
    print(f"Independent mode (Test 1): {params_test1:,} parameters")
    print(f"Shared mode (Test 3):      {params_test3:,} parameters")
    print(f"Parameter reduction:        {params_test1 - params_test3:,} ({100*(params_test1-params_test3)/params_test1:.1f}%)")
