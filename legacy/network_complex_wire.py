import torch
import torch.nn as nn
import torch.nn.functional as F
import tinycudann as tcnn
from wire2d import INR


class ComplexINRModel2D(nn.Module):
    """
    Conventional ptychography 风格的 2D 复数隐式模型（wire2d INR + Gabor 激活）。
    
    特点:
    - 独立哈希编码器，分别生成 Object 与 Probe 的复数场。
    - 可选残差：output = k * network_output + initial_pattern。
    - Object 先在低分辨率生成，再通过频域上采样恢复至原尺寸。
    """
    def __init__(
        self, 
        output_width, 
        output_height, 
        downsample_factor, 
        update_probe=True,
        probe_width=None,
        probe_height=None,
        # 哈希编码参数
        n_levels=16,
        n_features_per_level=2,
        log2_hashmap_size=19,
        base_resolution=16,
        per_level_scale=1.5,
        # wire2d INR 参数
        first_omega_0=10.0,
        hidden_omega_0=10.0,
        scale=10.0,
        hidden_features=64,
        hidden_layers=2,
        trainable_omega0=False,  # 是否让 omega 参数可训练
        trainable_scale0=False,  # 是否让 scale/sigma 参数可训练
        # 残差连接参数
        use_residual=False,
        object_initial=None,  # 复数 tensor [H, W] 或 None
        probe_initial=None    # 复数 tensor [H_p, W_p] 或 None
    ):
        super().__init__()
        self.update_probe = update_probe
        self.use_residual = use_residual
        self.output_width = output_width
        self.output_height = output_height
        self.probe_width = probe_width if probe_width is not None else output_width
        self.probe_height = probe_height if probe_height is not None else output_height
        
        # 哈希编码器 (输出实数特征)
        self.num_features = n_levels * n_features_per_level
        encoding_config = {
            "otype": "HashGrid",
            "n_levels": n_levels,
            "n_features_per_level": n_features_per_level,
            "log2_hashmap_size": log2_hashmap_size,
            "base_resolution": base_resolution,
            "per_level_scale": per_level_scale
        }
        self.xy_hash_encoder = tcnn.Encoding(
            n_input_dims=2,
            encoding_config=encoding_config
        )
        self.probe_hash_encoder = (
            tcnn.Encoding(n_input_dims=2, encoding_config=encoding_config.copy())
            if update_probe else None
        )
        
        # Object MLP (使用 wire2d 的 INR)
        # 输入: num_features (实数) -> 输出: 1 (复数)
        self.object_complex_mlp = INR(
            in_features=self.num_features,
            hidden_features=hidden_features,
            hidden_layers=hidden_layers,
            out_features=1,
            first_omega_0=first_omega_0,
            hidden_omega_0=hidden_omega_0,
            scale=scale,
            trainable_omega0=trainable_omega0,
            trainable_scale0=trainable_scale0,
            outermost_linear=True
        )
        
        # Probe MLP (使用 wire2d 的 INR) - 可选
        if update_probe:
            self.probe_complex_mlp = INR(
                in_features=self.num_features,
                hidden_features=hidden_features,
                hidden_layers=hidden_layers,
                out_features=1,
                first_omega_0=first_omega_0,
                hidden_omega_0=hidden_omega_0,
                scale=scale,
                trainable_omega0=trainable_omega0,
                trainable_scale0=trainable_scale0,
                outermost_linear=True
            )
        else:
            self.probe_complex_mlp = None

        self.init_scale_grids(downsample_factor)
        
        # 初始化残差连接参数
        if self.use_residual:
            self.init_residual_params(object_initial, probe_initial)

    def init_scale_grids(self, downsample_factor):
        # Object 低分辨率网格
        downsampled_width = self.output_width // downsample_factor
        downsampled_height = self.output_height // downsample_factor
        
        half_dx = 0.5 / downsampled_width
        half_dy = 0.5 / downsampled_height
        xs = torch.linspace(half_dx, 1 - half_dx, downsampled_width)
        ys = torch.linspace(half_dy, 1 - half_dy, downsampled_height)
        xv, yv = torch.meshgrid([xs, ys], indexing="ij")
        coords = torch.stack((yv.flatten(), xv.flatten())).t()  # [N, 2]
        
        self.downsampled_width = downsampled_width
        self.downsampled_height = downsampled_height
        self.xy_coords = nn.Parameter(coords, requires_grad=False)
        
        # Probe 网格（不下采样）
        if self.update_probe:
            half_dx_probe = 0.5 / self.probe_width
            half_dy_probe = 0.5 / self.probe_height
            xs_probe = torch.linspace(half_dx_probe, 1 - half_dx_probe, self.probe_width)
            ys_probe = torch.linspace(half_dy_probe, 1 - half_dy_probe, self.probe_height)
            xv_probe, yv_probe = torch.meshgrid([xs_probe, ys_probe], indexing="ij")
            probe_coords = torch.stack((yv_probe.flatten(), xv_probe.flatten())).t()
            self.probe_coords = nn.Parameter(probe_coords, requires_grad=False)
        else:
            self.probe_coords = None
        
        self.downsample_factor = downsample_factor

    def init_residual_params(self, object_initial, probe_initial):
        """
        初始化残差连接参数
        输出 = k * MLP_output + initial_value （仅当提供 initial 时）
        
        注意：
        - 如果提供了 initial_value：使用残差连接 k * MLP + initial
        - 如果没有提供 initial_value：直接使用 MLP 输出，不使用残差
        
        训练策略（当有 initial 时）：
        - 训练初期: k≈0.01，output ≈ initial_value（主要依赖初始值）
        - 训练后期: k增大，output = k*MLP_output + initial（网络逐渐起作用）
        """
        # Object 残差参数
        if object_initial is not None:
            # 提供了初始值，使用残差连接
            if not torch.is_complex(object_initial):
                raise ValueError("object_initial must be a complex tensor")
            self.register_buffer('object_initial', object_initial.clone())
            # 创建可学习的缩放因子 k
            self.k_object = nn.Parameter(torch.tensor(0.01 + 0j, dtype=torch.complex64))
            self.has_object_residual = True
        else:
            # 没有初始值，不使用残差连接
            self.has_object_residual = False
        
        # Probe 残差参数
        if self.update_probe:
            if probe_initial is not None:
                # 提供了初始值，使用残差连接
                if not torch.is_complex(probe_initial):
                    raise ValueError("probe_initial must be a complex tensor")
                self.register_buffer('probe_initial', probe_initial.clone())
                # 创建可学习的缩放因子 k
                self.k_probe = nn.Parameter(torch.tensor(0.01 + 0j, dtype=torch.complex64))
                self.has_probe_residual = True
            else:
                # 没有初始值，不使用残差连接
                self.has_probe_residual = False
        else:
            self.has_probe_residual = False

    def fourier_upsample(self, x, scale_factor):
        # x 是复数 [B, C, H, W]
        # 直接对复数进行 FFT
        B, C, H, W = x.shape
        
        # FFT
        x_fft = torch.fft.fft2(x, dim=(-2, -1))
        x_fft_shifted = torch.fft.fftshift(x_fft, dim=(-2, -1))
        
        # Pad
        new_H, new_W = int(H * scale_factor), int(W * scale_factor)
        pad_h, pad_w = (new_H - H) // 2, (new_W - W) // 2
        x_fft_padded = F.pad(x_fft_shifted, (pad_w, new_W - W - pad_w, pad_h, new_H - H - pad_h), mode='constant', value=0)
        
        # IFFT
        x_fft_unshifted = torch.fft.ifftshift(x_fft_padded, dim=(-2, -1))
        x_upsampled = torch.fft.ifft2(x_fft_unshifted, dim=(-2, -1))
        
        # Scale amplitude
        return x_upsampled * (scale_factor ** 2)

    def forward(self):
        xy_coords = self.xy_coords
        if not xy_coords.is_cuda:
            xy_coords = xy_coords.cuda()
            
        # 1. 获取哈希特征 (实数)
        spatial_features = self.xy_hash_encoder(xy_coords).float() # [N, num_features]
        
        # 2. Object: wire2d INR -> 复数输出
        # 注意: wire2d.INR 的 forward 默认返回 output.real
        # 我们需要获取完整的复数输出
        object_complex = self._get_complex_output(self.object_complex_mlp, spatial_features)
        object_complex = object_complex.reshape(self.downsampled_width, self.downsampled_height)
        
        # 3. Upsample Object
        # [W_ds, H_ds] -> [1, 1, W_ds, H_ds] -> Upsample -> [W, H]
        object_complex = self.fourier_upsample(object_complex.unsqueeze(0).unsqueeze(0), self.downsample_factor).squeeze(0).squeeze(0)
        
        # 4. Probe (可选，直接生成到目标尺寸)
        if self.update_probe:
            probe_coords = self.probe_coords
            if not probe_coords.is_cuda:
                probe_coords = probe_coords.cuda()
            probe_features = self.probe_hash_encoder(probe_coords).float()
            probe_complex = self._get_complex_output(self.probe_complex_mlp, probe_features)
            probe_complex = probe_complex.reshape(self.probe_width, self.probe_height)
        else:
            probe_complex = None
        
        # 5. 应用残差连接 (如果启用)
        if self.use_residual:
            object_complex = self.apply_residual(object_complex, is_object=True)
            if self.update_probe and probe_complex is not None:
                probe_complex = self.apply_residual(probe_complex, is_object=False)

        return object_complex, probe_complex
    
    def _get_complex_output(self, model, coords):
        """
        从 wire2d.INR 模型获取完整的复数输出
        wire2d.INR 的 forward 默认返回 output.real，这里我们直接调用网络获取复数
        """
        output = model.net(coords)  # 直接调用 net，绕过 forward 中的 .real 操作
        return output  # 返回完整的复数 tensor
    
    def apply_residual(self, complex_tensor, is_object=True):
        """
        应用残差连接: output = k * input + initial （仅当有 initial 时）
        如果没有 initial，直接返回网络输出
        """
        if is_object:
            if self.has_object_residual:
                # 有初始值，使用残差连接
                return self.k_object * complex_tensor + self.object_initial
            else:
                # 没有初始值，直接返回网络输出
                return complex_tensor
        else:
            if self.has_probe_residual:
                # 有初始值，使用残差连接
                return self.k_probe * complex_tensor + self.probe_initial
            else:
                # 没有初始值，直接返回网络输出
                return complex_tensor

def check_nan_inf(tensor, name="Tensor"):
    """检查张量中是否有 NaN 或 Inf"""
    if torch.is_complex(tensor):
        real_nan = torch.isnan(tensor.real).any()
        imag_nan = torch.isnan(tensor.imag).any()
        real_inf = torch.isinf(tensor.real).any()
        imag_inf = torch.isinf(tensor.imag).any()
        
        has_issue = real_nan or imag_nan or real_inf or imag_inf
        if has_issue:
            print(f"❌ {name}: Real NaN={real_nan}, Imag NaN={imag_nan}, Real Inf={real_inf}, Imag Inf={imag_inf}")
            print(f"   Real range: [{tensor.real.min():.6f}, {tensor.real.max():.6f}]")
            print(f"   Imag range: [{tensor.imag.min():.6f}, {tensor.imag.max():.6f}]")
            return True
        else:
            print(f"✓ {name}: 无 NaN/Inf")
            print(f"   Real range: [{tensor.real.min():.6f}, {tensor.real.max():.6f}]")
            print(f"   Imag range: [{tensor.imag.min():.6f}, {tensor.imag.max():.6f}]")
            return False
    else:
        has_nan = torch.isnan(tensor).any()
        has_inf = torch.isinf(tensor).any()
        has_issue = has_nan or has_inf
        if has_issue:
            print(f"❌ {name}: NaN={has_nan}, Inf={has_inf}")
            print(f"   Range: [{tensor.min():.6f}, {tensor.max():.6f}]")
            return True
        else:
            print(f"✓ {name}: 无 NaN/Inf, Range: [{tensor.min():.6f}, {tensor.max():.6f}]")
            return False


if __name__ == "__main__":
    # Simple test
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("\n" + "="*60)
    print("=== Test 1: 基础模型 (无残差，检查 NaN) ===")
    print("="*60)
    
    # 使用较小的参数避免数值溢出
    model = ComplexINRModel2D(
        output_width=128,
        output_height=128,
        downsample_factor=2,
        update_probe=True,
        first_omega_0=1.0,      # 降低频率参数，避免数值爆炸
        hidden_omega_0=1.0,
        scale=1.0,              # 降低 scale，避免 Gaussian 项过大
        hidden_features=32,     # 减小网络规模
        hidden_layers=1,        # 减少层数
        n_levels=8,             # 减少哈希层级
        n_features_per_level=2,
        log2_hashmap_size=16
    ).to(device)
    
    print(f"\n模型参数总数: {sum(p.numel() for p in model.parameters()):,}")
    
    # 检查模型权重初始化
    print("\n检查模型初始化:")
    for name, param in model.named_parameters():
        if param.requires_grad and len(param.shape) > 0:
            if torch.is_complex(param):
                print(f"  {name}: shape={param.shape}, real=[{param.real.min():.4f}, {param.real.max():.4f}], imag=[{param.imag.min():.4f}, {param.imag.max():.4f}]")
            else:
                print(f"  {name}: shape={param.shape}, range=[{param.min():.4f}, {param.max():.4f}]")
    
    print("\n执行前向传播...")
    obj, probe = model()
    
    print(f"\nObject shape: {obj.shape}, dtype: {obj.dtype}")
    print(f"Probe shape: {probe.shape if probe is not None else 'None'}, dtype: {probe.dtype if probe is not None else 'None'}")
    
    # 检查输出是否有 NaN 或 Inf
    print("\n检查输出数值:")
    obj_has_issue = check_nan_inf(obj, "Object")
    probe_has_issue = check_nan_inf(probe, "Probe") if probe is not None else False
    
    # Check if output is complex
    assert torch.is_complex(obj), "Object 应该是复数类型"
    assert not obj_has_issue, "Object 不应该包含 NaN 或 Inf"
    if probe is not None:
        assert torch.is_complex(probe), "Probe 应该是复数类型"
        assert not probe_has_issue, "Probe 不应该包含 NaN 或 Inf"
    
    print("\n✓ Test 1 passed: 输出是复数且无 NaN/Inf!")
    
    print("\n" + "="*60)
    print("=== Test 2: 带残差连接的模型 (检查 NaN) ===")
    print("="*60)
    
    # 创建一个初始复数图像（避免过大的值）
    obj_init = (torch.randn(128, 128, device=device) + 
                1j * torch.randn(128, 128, device=device)) * 0.01  # 使用很小的初始值
    probe_init = (torch.randn(128, 128, device=device) + 
                  1j * torch.randn(128, 128, device=device)) * 0.01
    
    model_res = ComplexINRModel2D(
        output_width=128,
        output_height=128,
        downsample_factor=2,
        update_probe=True,
        first_omega_0=1.0,
        hidden_omega_0=1.0,
        scale=1.0,
        hidden_features=32,
        hidden_layers=1,
        n_levels=8,
        n_features_per_level=2,
        log2_hashmap_size=16,
        use_residual=True,
        object_initial=obj_init,
        probe_initial=probe_init
    ).to(device)
    
    print(f"\n模型参数总数: {sum(p.numel() for p in model_res.parameters()):,}")
    print(f"k_object 初始值: {model_res.k_object.item()}")
    print(f"k_probe 初始值: {model_res.k_probe.item()}")
    
    print("\n执行前向传播...")
    obj_res, probe_res = model_res()
    
    print(f"\nObject shape: {obj_res.shape}, dtype: {obj_res.dtype}")
    print(f"Probe shape: {probe_res.shape if probe_res is not None else 'None'}, dtype: {probe_res.dtype if probe_res is not None else 'None'}")
    
    # 检查输出是否有 NaN 或 Inf
    print("\n检查输出数值:")
    obj_res_has_issue = check_nan_inf(obj_res, "Object (residual)")
    probe_res_has_issue = check_nan_inf(probe_res, "Probe (residual)") if probe_res is not None else False
    
    # 验证哪些参数是可训练的
    print("\n可训练参数检查:")
    trainable_params = [name for name, param in model_res.named_parameters() if param.requires_grad]
    print(f"可训练参数数量: {len(trainable_params)}")
    
    # 检查 initial 是否可训练（应该是False）
    buffers = [name for name, _ in model_res.named_buffers()]
    print(f"不可训练的 buffers (包含initial): {[b for b in buffers if 'initial' in b]}")
    
    # 检查 k 是否可训练（应该是True）
    k_params = [name for name in trainable_params if 'k_' in name]
    print(f"可训练的 k 参数: {k_params}")
    
    # 检查残差连接标志
    print(f"has_object_residual: {model_res.has_object_residual}")
    print(f"has_probe_residual: {model_res.has_probe_residual}")
    
    assert torch.is_complex(obj_res), "Object 应该是复数类型"
    assert not obj_res_has_issue, "Object 不应该包含 NaN 或 Inf"
    if probe_res is not None:
        assert torch.is_complex(probe_res), "Probe 应该是复数类型"
        assert not probe_res_has_issue, "Probe 不应该包含 NaN 或 Inf"
    
    # 由于提供了 initial，应该有残差连接
    assert model_res.has_object_residual, "提供了 object_initial，应该有残差连接"
    assert model_res.has_probe_residual, "提供了 probe_initial，应该有残差连接"
    assert 'object_initial' in buffers, "object_initial 应该是 buffer（不可训练）"
    assert 'k_object' in trainable_params, "k_object 应该是可训练的"
    assert torch.is_complex(model_res.k_object), "k_object 应该是复数"
    assert abs(model_res.k_object.item() - (0.01 + 0j)) < 1e-6, "k_object 应该初始化为0.01+0j"
    if model_res.has_probe_residual:
        assert 'probe_initial' in buffers, "probe_initial 应该是 buffer（不可训练）"
        assert 'k_probe' in trainable_params, "k_probe 应该是可训练的"
        assert torch.is_complex(model_res.k_probe), "k_probe 应该是复数"
        assert abs(model_res.k_probe.item() - (0.01 + 0j)) < 1e-6, "k_probe 应该初始化为0.01+0j"
    
    print("\n" + "="*60)
    print("=== Test 3: 无 initial 值的残差模型 (直接用网络输出) ===")
    print("="*60)
    
    # 创建没有 initial 的残差模型
    model_no_init = ComplexINRModel2D(
        output_width=128,
        output_height=128,
        downsample_factor=2,
        update_probe=True,
        first_omega_0=1.0,
        hidden_omega_0=1.0,
        scale=1.0,
        hidden_features=32,
        hidden_layers=1,
        n_levels=8,
        n_features_per_level=2,
        log2_hashmap_size=16,
        use_residual=True,
        object_initial=None,      # ⭐ 没有提供 initial
        probe_initial=None     # ⭐ 没有提供 initial
    ).to(device)
    
    print(f"\n模型参数总数: {sum(p.numel() for p in model_no_init.parameters()):,}")
    print(f"has_object_residual: {model_no_init.has_object_residual}")
    print(f"has_probe_residual: {model_no_init.has_probe_residual}")
    
    print("\n执行前向传播...")
    obj_no_init, probe_no_init = model_no_init()
    
    print(f"\nObject shape: {obj_no_init.shape}, dtype: {obj_no_init.dtype}")
    print(f"Probe shape: {probe_no_init.shape if probe_no_init is not None else 'None'}, dtype: {probe_no_init.dtype if probe_no_init is not None else 'None'}")
    
    # 检查输出
    print("\n检查输出数值:")
    obj_no_init_has_issue = check_nan_inf(obj_no_init, "Object (no initial)")
    probe_no_init_has_issue = check_nan_inf(probe_no_init, "Probe (no initial)") if probe_no_init is not None else False
    
    # 验证没有创建 k 参数和 initial buffer
    trainable_params_no_init = [name for name, param in model_no_init.named_parameters() if param.requires_grad]
    buffers_no_init = [name for name, _ in model_no_init.named_buffers()]
    k_params_no_init = [name for name in trainable_params_no_init if 'k_' in name]
    initial_buffers = [name for name in buffers_no_init if 'initial' in name]
    
    print(f"\n参数检查:")
    print(f"  k 参数: {k_params_no_init} (应该为空)")
    print(f"  initial buffers: {initial_buffers} (应该为空)")
    
    assert torch.is_complex(obj_no_init), "Object 应该是复数类型"
    assert not obj_no_init_has_issue, "Object 不应该包含 NaN 或 Inf"
    if probe_no_init is not None:
        assert torch.is_complex(probe_no_init), "Probe 应该是复数类型"
        assert not probe_no_init_has_issue, "Probe 不应该包含 NaN 或 Inf"
    assert not model_no_init.has_object_residual, "没有 initial，不应该有残差连接"
    assert not model_no_init.has_probe_residual, "没有 initial，不应该有残差连接"
    assert len(k_params_no_init) == 0, "没有 initial 时不应该创建 k 参数"
    assert len(initial_buffers) == 0, "没有 initial 时不应该创建 initial buffer"
    
    print("\n✓ Test 3 passed: 没有 initial 时直接使用网络输出!")
    
    print("\n" + "="*60)
    print("✅ 所有测试通过!")
    print("="*60)
    print("✓ 输出是复数张量")
    print("✓ 输出无 NaN/Inf")
    print("✓ 提供 initial 时: 使用残差连接 k*MLP + initial")
    print("✓ 不提供 initial 时: 直接使用网络输出")
    print("✓ Initial values 已冻结（不可训练）")
    print("✓ 缩放因子 k 仅在有 initial 时创建")
    print("✓ 使用 wire2d INR with ComplexGaborLayer2D 激活函数!")
    print("="*60)
