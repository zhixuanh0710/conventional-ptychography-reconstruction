import torch
import torch.nn as nn

class MLPRenderer(nn.Module):
    """
    多层感知机 (MLP) 渲染器，用于将输入特征向量映射到输出（如像素值）。
    """
    def __init__(
        self, 
        in_dim=32, 
        hidden_dim=32, 
        num_layers=2, 
        out_dim=1, 
        use_layer_norm=True
    ):
        super().__init__()
        # 可根据需要切换不同激活函数
        activation = nn.ReLU()
        # activation = Sine()
        # activation = nn.LeakyReLU(0.2)
        # activation = nn.Sigmoid()

        layers = []
        # 第1层：输入 -> hidden_dim
        layers.append(nn.Linear(in_dim, hidden_dim))
        if use_layer_norm:
            layers.append(nn.LayerNorm(hidden_dim, elementwise_affine=False))
        layers.append(activation)

        # 中间层：重复 num_layers - 1 次
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim, elementwise_affine=False))
            layers.append(activation)

        # 最后一层：hidden_dim -> out_dim
        layers.append(nn.Linear(hidden_dim, out_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class FeatureGrid2D(nn.Module):
    """
    二维特征网格，用于在 (x, y) 平面上存储一个低分辨率的多通道特征张量，
    并支持通过双线性插值在连续坐标上采样特征。
    """
    def __init__(self, x_dim, y_dim, num_features=32, downsample_factor=1):
        super().__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.num_features = num_features

        # 低分辨率网格尺寸
        x_mode = x_dim // downsample_factor
        y_mode = y_dim // downsample_factor

        # 初始化存储的特征网格
        self.xy_features = nn.Parameter(
            2e-4 * torch.rand((x_mode, y_mode, num_features)) - 1e-4, 
            requires_grad=True
        )

        # 预计算插值索引和权重
        half_dx = 0.5 / x_dim
        half_dy = 0.5 / y_dim
        xs = torch.linspace(half_dx, 1 - half_dx, x_dim)
        ys = torch.linspace(half_dy, 1 - half_dy, y_dim)
        xv, yv = torch.meshgrid([xs, ys], indexing="ij")
        xy_coords = torch.stack((yv.flatten(), xv.flatten())).t()  # [N, 2]

        # 将 [0,1] 坐标映射到 [0, x_mode] 和 [0, y_mode]
        scaled_coords = xy_coords * torch.tensor([x_mode, y_mode], device=xs.device).float()
        indices = scaled_coords.long()  # 整数部分作为索引
        self.lerp_weights = nn.Parameter(
            scaled_coords - indices.float(),
            requires_grad=False
        )

        # 四个顶点索引：x0, y0, x1, y1
        self.x0 = nn.Parameter(indices[:, 0].clamp(min=0, max=x_mode - 1), requires_grad=False)
        self.y0 = nn.Parameter(indices[:, 1].clamp(min=0, max=y_mode - 1), requires_grad=False)
        self.x1 = nn.Parameter((self.x0 + 1).clamp(max=x_mode - 1), requires_grad=False)
        self.y1 = nn.Parameter((self.y0 + 1).clamp(max=y_mode - 1), requires_grad=False)

    def interpolate_2d_features(self):
        """
        利用双线性插值，从低分辨率特征网格中采样得到 (x_dim * y_dim) 个特征向量。
        """
        return (
            self.xy_features[self.y0, self.x0]
            * (1.0 - self.lerp_weights[:, 0:1])
            * (1.0 - self.lerp_weights[:, 1:2])
            + self.xy_features[self.y0, self.x1]
            * self.lerp_weights[:, 0:1]
            * (1.0 - self.lerp_weights[:, 1:2])
            + self.xy_features[self.y1, self.x0]
            * (1.0 - self.lerp_weights[:, 0:1])
            * self.lerp_weights[:, 1:2]
            + self.xy_features[self.y1, self.x1]
            * self.lerp_weights[:, 0:1]
            * self.lerp_weights[:, 1:2]
        )

    def forward(self):
        return self.interpolate_2d_features()


class Implicit2D(FeatureGrid2D):
    """
    继承自 FeatureGrid2D，在二维特征网格基础上增加一个 MLP 渲染器，
    将插值得到的特征映射为 2D 图像输出。
    """
    def __init__(self, phasee_size, num_features=32, downsample_factor=1):
        super().__init__(
            x_dim=phasee_size, 
            y_dim=phasee_size, 
            num_features=num_features, 
            downsample_factor=downsample_factor
        )
        self.mlp_renderer = MLPRenderer(in_dim=num_features)

    def forward(self):
        features = self.interpolate_2d_features()
        return self.mlp_renderer(features)


class Implicit3D(nn.Module):
    """
    三维特征隐式表示：
    - 包含一个 2D 特征网格 (xy_features)
    - 以及一个 1D z_features，用于在 z 轴上进行线性插值
    - 通过 Hadamard 积将 xy 特征与 z 特征融合
    - 最后通过 MLP 渲染器输出图像/场分布
    """
    def __init__(
        self, 
        x_mode, 
        y_mode, 
        num_z_slices, 
        z_min, 
        z_max, 
        num_features=32, 
        use_layer_norm=False
    ):
        super().__init__()
        self.x_mode = x_mode
        self.y_mode = y_mode
        self.num_features = num_features

        # 2D 特征网格 (低分辨率)
        self.xy_features = nn.Parameter(
            2e-4 * torch.randn((x_mode, y_mode, num_features)),
            requires_grad=True
        )

        # 1D z 方向特征
        self.z_features = nn.Parameter(
            torch.randn((num_z_slices, num_features)),
            requires_grad=True
        )
        self.z_min = z_min
        self.z_max = z_max
        self.num_z_slices = num_z_slices

        # 渲染器
        self.mlp_renderer = MLPRenderer(in_dim=num_features, use_layer_norm=use_layer_norm)

        # 后续将根据实际的 x_dim, y_dim 初始化插值坐标
        self.x0 = None
        self.xy_coords = None

    def init_2d_coords(self, x_dim, y_dim, x_max, y_max):
        """
        初始化 (x_dim, y_dim) 尺寸下的插值坐标，映射到 [0, x_max] 和 [0, y_max]。
        """
        half_dx = 0.5 / x_dim
        half_dy = 0.5 / y_dim
        xs = torch.linspace(half_dx, 1 - half_dx, x_dim)
        ys = torch.linspace(half_dy, 1 - half_dy, y_dim)
        xv, yv = torch.meshgrid([xs, ys], indexing="ij")
        coords = torch.stack((yv.flatten(), xv.flatten())).t()  # [N, 2]
        scaled_coords = coords * torch.tensor([x_max, y_max], device=xs.device).float()
        indices = scaled_coords.long()

        self.x_dim = x_dim
        self.y_dim = y_dim
        self.xy_coords = nn.Parameter(coords[None], requires_grad=False)

        if self.x0 is not None:
            # 如果之前已经创建过，就更新
            device = self.x0.device
            self.x0.data = indices[:, 0].clamp(0, x_max - 1).to(device)
            self.y0.data = indices[:, 1].clamp(0, y_max - 1).to(device)
            self.x1.data = (self.x0 + 1).clamp(max=x_max - 1).to(device)
            self.y1.data = (self.y0 + 1).clamp(max=y_max - 1).to(device)
            self.lerp_weights.data = (scaled_coords - indices.float()).to(device)
        else:
            self.x0 = nn.Parameter(indices[:, 0].clamp(0, x_max - 1), requires_grad=False)
            self.y0 = nn.Parameter(indices[:, 1].clamp(0, y_max - 1), requires_grad=False)
            self.x1 = nn.Parameter((self.x0 + 1).clamp(max=x_max - 1), requires_grad=False)
            self.y1 = nn.Parameter((self.y0 + 1).clamp(max=y_max - 1), requires_grad=False)
            self.lerp_weights = nn.Parameter(scaled_coords - indices.float(), requires_grad=False)

    def normalize_z(self, z):
        """
        将 z 坐标从 [z_min, z_max] 线性映射到 [0, num_z_slices - 1]，便于插值。
        """
        return (self.num_z_slices - 1) * (z - self.z_min) / (self.z_max - self.z_min)

    def interpolate_3d_features(self, z):
        """
        根据输入的 z，分别对 xy 平面和 z 方向进行插值，然后通过 Hadamard 积融合特征。
        """
        # 1. z 插值
        z_norm = self.normalize_z(z)
        z0 = z_norm.long().clamp(min=0, max=self.num_z_slices - 1)
        z1 = (z0 + 1).clamp(max=self.num_z_slices - 1)
        z_lerp_weights = (z_norm - z_norm.long().float())[:, None]

        # 2. xy 平面双线性插值
        xy_feat = (
            self.xy_features[self.y0, self.x0]
            * (1.0 - self.lerp_weights[:, 0:1])
            * (1.0 - self.lerp_weights[:, 1:2])
            + self.xy_features[self.y0, self.x1]
            * self.lerp_weights[:, 0:1]
            * (1.0 - self.lerp_weights[:, 1:2])
            + self.xy_features[self.y1, self.x0]
            * (1.0 - self.lerp_weights[:, 0:1])
            * self.lerp_weights[:, 1:2]
            + self.xy_features[self.y1, self.x1]
            * self.lerp_weights[:, 0:1]
            * self.lerp_weights[:, 1:2]
        )

        # 3. z 特征线性插值
        z_feat = (
            self.z_features[z0] * (1.0 - z_lerp_weights) 
            + self.z_features[z1] * z_lerp_weights
        )
        z_feat = z_feat[:, None].repeat(1, xy_feat.shape[0], 1)

        # 4. xy_feat 与 z_feat 逐元素乘法融合
        feat = xy_feat[None].repeat(z.shape[0], 1, 1) * z_feat

        return feat

    def forward(self, z):
        """
        给定 z（可以是 batch 维度），先融合得到 3D 特征，再通过 MLP 渲染得到最终输出。
        输入形状: (batch_size, z_dim)。
        输出形状: (batch_size, 1, x_dim, y_dim)。
        """
        feat = self.interpolate_3d_features(z)
        out = self.mlp_renderer(feat)

        b = z.shape[0]
        w, h = self.x_dim, self.y_dim
        out = out.view(b, 1, w, h)
        return out


class ComplexINRModel(nn.Module):
    """
    综合模型：同时生成图像的实部与虚部（或其他双通道），
    通过两个 Implicit3D 分别负责实部和虚部的特征插值与渲染。
    """
    def __init__(
        self, 
        width, 
        height, 
        num_features, 
        x_mode, 
        y_mode, 
        z_min, 
        z_max, 
        downsample_factor, 
        use_layer_norm,
        diffuser = None
    ):
        super().__init__()
        # 分别创建 3D 隐式表示用于生成实部和虚部
        self.ampli_generator = Implicit3D(
            x_mode=x_mode,
            y_mode=y_mode,
            num_z_slices=1,
            z_min=z_min,
            z_max=z_max,
            num_features=num_features,
            use_layer_norm=use_layer_norm
        )
        self.phase_generator = Implicit3D(
            x_mode=x_mode,
            y_mode=y_mode,
            num_z_slices=1,
            z_min=z_min,
            z_max=z_max,
            num_features=num_features,
            use_layer_norm=use_layer_norm
        )
        # 新增的漫射器特征生成器

        if diffuser is not None:
            # 该参数可训练，以输入的diffuser作为初始值
            # self.diffuser_ampli = nn.Parameter(torch.abs(diffuser).float(),   requires_grad=True)
            # self.diffuser_phase = nn.Parameter(torch.angle(diffuser).float(), requires_grad=True)
            # 假设 diffuser 是一个 complex 张量
            real_part = diffuser.real
            imag_part = diffuser.imag

            # 将实部和虚部各自注册为参数
            self.diffuser_real = nn.Parameter(real_part.float(), requires_grad=True)
            self.diffuser_imag = nn.Parameter(imag_part.float(), requires_grad=True)

        else:
            self.diffuser_ampli = nn.Parameter(torch.randn((512,512), requires_grad=True))
            self.diffuser_phase = nn.Parameter(torch.randn((512,512), requires_grad=True))
        self.width = width
        self.height = height

        self.init_scale_grids(downsample_factor)

    def init_scale_grids(self, downsample_factor):
        """
        初始化下采样 / 上采样网格坐标，并创建上采样模块。
        """
        self.ampli_generator.init_2d_coords(
            x_dim=self.width // downsample_factor,
            y_dim=self.height // downsample_factor,
            x_max=self.ampli_generator.x_mode,
            y_max=self.ampli_generator.y_mode
        )
        self.phase_generator.init_2d_coords(
            x_dim=self.width // downsample_factor,
            y_dim=self.height // downsample_factor,
            x_max=self.phase_generator.x_mode,
            y_max=self.phase_generator.y_mode
        )
        self.downsample_factor = downsample_factor
        self.upsample = nn.Upsample(scale_factor=downsample_factor, mode="bilinear")

    def forward(self, z_input):
        """
        前向传播：
        1. 通过 ampli_generator 和 phase_generator 分别得到实部与虚部的低分辨率图像
        2. 上采样到原始分辨率
        """
        object_ampli = self.ampli_generator(z_input)
        object_phase = self.phase_generator(z_input)

        object_ampli = self.upsample(object_ampli).squeeze(1)  # (B, W, H)
        object_phase = self.upsample(object_phase).squeeze(1)  # (B, W, H)

        # diffuser_ampli = self.diffuser_ampli
        # diffuser_phase = self.diffuser_phase
        # 计算复数形式的 diffuser
        diffuser_real = self.diffuser_real
        diffuser_imag = self.diffuser_imag

        return object_ampli, object_phase, diffuser_real, diffuser_imag # diffuser_ampli, diffuser_phase
