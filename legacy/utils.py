import numpy as np
import matplotlib as mpl
from matplotlib.colors import ListedColormap
import torch
from scipy.io import loadmat
from math import pi

def save_model_with_required_grad(model, save_path):
    tensors_to_save = []
    
    # Traverse through model parameters and append tensors with require_grad=True to the list
    for param_name, param_tensor in model.named_parameters():
        if param_tensor.requires_grad:
            tensors_to_save.append(param_tensor)
    
    # Save the list of tensors
    torch.save(tensors_to_save, save_path)

def load_model_with_required_grad(model, load_path):
    # Load the list of tensors
    tensors_to_load = torch.load(load_path)
    
    # Traverse through model parameters and load tensors from the list
    for param_name, param_tensor in model.named_parameters():
        if param_tensor.requires_grad:
            param_tensor.data = tensors_to_load.pop(0).data

newcolors = np.vstack(
    (
        np.flipud(mpl.colormaps['magma'](np.linspace(0, 1, 128))),
        mpl.colormaps['magma'](np.linspace(0, 1, 128)),
    )
)
newcmp = ListedColormap(newcolors, name='magma_cyclic')

def load_matlab_file(file_path, key_name):
    """
    加载MATLAB文件，自动处理普通格式和7.3版本(HDF5格式)
    
    Parameters:
    -----------
    file_path : str
        MATLAB文件路径
    key_name : str or None
        要加载的特定键名，None表示加载所有键
        
    Returns:
    --------
    dict : 包含MATLAB数据的字典
    """
    try:
        # 尝试使用普通方法加载MATLAB文件
        data = loadmat(file_path)
        print(f"成功使用scipy.io.loadmat加载文件: {file_path}")
        return data
    except Exception as e:
        # 如果是MATLAB 7.3版本（HDF5格式），使用h5py加载
        print(f"scipy.io.loadmat加载失败，尝试使用h5py加载MATLAB 7.3格式文件: {file_path}")
        print(f"错误信息: {e}")
        try:
            import h5py
            with h5py.File(file_path, 'r') as f:
                # 打印所有可用的键
                print("HDF5文件中的键:", list(f.keys()))
                data = {}
                
                # 如果指定了键名，尝试加载
                if key_name and key_name in f.keys():
                    data[key_name] = np.array(f[key_name]).T  # HDF5通常需要转置
                    print(f"成功加载键 '{key_name}'")
                else:
                    # 加载所有数据键（跳过元数据键）
                    available_keys = list(f.keys())
                    data_keys = [k for k in available_keys if not k.startswith('#')]
                    for key in data_keys:
                        data[key] = np.array(f[key]).T
                        print(f"加载键 '{key}', 形状: {data[key].shape}")
                        
            print("成功使用h5py加载MATLAB 7.3格式文件")
            return data
        except ImportError:
            print("未安装h5py库，请运行: pip install h5py")
            raise
        except Exception as h5_error:
            print(f"h5py加载也失败: {h5_error}")
            raise

def propagate(object, pixelSizeObject, wavelength, distance):
    M, N = object.shape
    k0 = 2 * pi / wavelength
    kmax = pi / pixelSizeObject
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    kxm0 = torch.linspace(-kmax, kmax, M, device=device)
    kym0 = torch.linspace(-kmax, kmax, N, device=device)
    kxm, kym = torch.meshgrid(kxm0, kym0, indexing='ij')

    realPart = k0**2 - kxm**2 - kym**2
    kzm = torch.sqrt(torch.complex(realPart, torch.zeros_like(realPart)))
    Hz = torch.exp(1j * distance * torch.real(kzm)) * torch.exp(-abs(distance) * torch.abs(torch.imag(kzm))) * ((realPart >= 0).type(torch.complex64))
    
    objectProp = torch.fft.ifft2(torch.fft.ifftshift(Hz * torch.fft.fftshift(torch.fft.fft2(object))))
    return objectProp

def subPixelShift(image, xShift, yShift, mag):
    m, n = image.shape

    # Create frequency coordinates along the two dimensions.
    fy = torch.linspace(-torch.floor(torch.tensor(m/2, dtype=torch.float64)),
                        torch.ceil(torch.tensor(m/2, dtype=torch.float64)) - 1,
                        m, device=image.device, dtype=image.dtype)
    fy = torch.fft.ifftshift(fy)
    
    fx = torch.linspace(-torch.floor(torch.tensor(n/2, dtype=torch.float64)),
                        torch.ceil(torch.tensor(n/2, dtype=torch.float64)) - 1,
                        n, device=image.device, dtype=image.dtype)
    fx = torch.fft.ifftshift(fx)

    # Generate meshgrid with shape (m, n)
    FY, FX = torch.meshgrid(fy, fx, indexing='ij')

    Hs = torch.exp(-1j * 2 * pi * (FX * -xShift / n * mag + FY * -yShift / m * mag))
    outputImage = torch.fft.ifft2(torch.fft.fft2(image) * Hs)
    return outputImage
