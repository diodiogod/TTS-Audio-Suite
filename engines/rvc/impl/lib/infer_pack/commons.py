import math
import numpy as np
import torch
from torch.nn import functional as F

def median_pool1d(signal, kernel_size, dim=-1, stride=1):
    """
    Apply a 1D median filter along a specified dimension for a 3D tensor.

    Args:
        signal (torch.Tensor): The input tensor of shape (batch_size, F, T).
        kernel_size (int): The size of the median filter kernel.
        dim (int): The dimension along which to apply the median filter.
        stride (int): The stride of the sliding window.

    Returns:
        torch.Tensor: The filtered tensor.
    """
    # Ensure the input tensor is 3D (batch_size, F, T)
    if signal.dim() != 3:
        raise ValueError(f"Input tensor {signal.shape} must be 3D (batch_size, F, T)")

    # Adjust kernel size if larger than input dimension
    input_size = signal.size(dim)
    if kernel_size > input_size:
        kernel_size = input_size

    # Pad the input to handle edge cases
    padding = (kernel_size - 1) // 2
    signal = F.pad(signal, (padding, padding), mode='reflect')
    if dim == -1:
        signal = F.pad(signal, (padding, padding), mode='reflect')
    elif dim == -2:
        signal = F.pad(signal, (0, 0, padding, padding), mode="reflect")

    # Unfold the input to create sliding windows
    unfolded = signal.unfold(dim, kernel_size, stride)

    # Compute the median along the last dimension
    median = unfolded.median(dim=-1).values

    return median

def autocorrelation1d(log_magnitude, max_lag=None):
    """
    Computes the autocorrelation of log-magnitude spectrogram.
    :param log_magnitude: Log magnitude spectrogram (batch, channels, time)
    :param max_lag: Maximum lag for which to compute autocorrelation.
    :return: Autocorrelation of the log-magnitude spectrogram.
    """
    if max_lag is None:
        max_lag = log_magnitude.size(-1) // 2  # Default to half the signal length

    # Ensure the signal is zero-mean
    log_magnitude = log_magnitude - log_magnitude.mean(dim=-1, keepdim=True)
    
    # Compute autocorrelation using FFT-based convolution
    padded_log_magnitude = F.pad(log_magnitude, (max_lag, 0), mode='reflect')
    
    # Ensure the kernel has the correct shape for conv1d
    kernel = log_magnitude
    padded_log_magnitude = padded_log_magnitude
    
    autocorr = F.conv1d(padded_log_magnitude, kernel, groups=log_magnitude.size(1))
    
    # Take only the relevant part
    autocorr = F.normalize(autocorr[:,:max_lag + 1], p=2, dim=-1)
    
    return autocorr.nan_to_num(0)

def minmax_scale(tensor: torch.Tensor, eps=1e-8):
    tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min() + eps)
    return tensor

def compute_correlation(x: torch.Tensor, y: torch.Tensor, eps=1e-8):
    """
    Calculate the Pearson correlation coefficient between two batched vectors.
    
    Args:
        x (torch.Tensor): A tensor of shape (batch_size, num_features).
        y (torch.Tensor): A tensor of shape (batch_size, num_features).
        
    Returns:
        torch.Tensor: A tensor of shape (batch_size,) containing the correlation coefficients.
    """
    # Ensure the inputs are float tensors
    x = x.float()
    y = y.float()
    
    # Calculate the mean of each batch
    mean_x = torch.mean(x, dim=-1, keepdim=True)
    mean_y = torch.mean(y, dim=-1, keepdim=True)
    
    # Subtract the mean from each element
    x_centered = x - mean_x
    y_centered = y - mean_y
    
    # Calculate the covariance
    covariance = torch.sum(x_centered * y_centered, dim=-1)
    
    # Calculate the standard deviations
    std_x = torch.sqrt(torch.sum(x_centered ** 2, dim=-1)+eps).nan_to_num(eps)
    std_y = torch.sqrt(torch.sum(y_centered ** 2, dim=-1)+eps).nan_to_num(eps)
    
    # Calculate the correlation coefficient
    correlation = covariance / (std_x * std_y + eps)
    
    return correlation.nan_to_num(eps)

def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


def convert_pad_shape(pad_shape):
    l = pad_shape[::-1]
    pad_shape = [item for sublist in l for item in sublist]
    return pad_shape

def intersperse(lst, item):
  result = [item] * (len(lst) * 2 + 1)
  result[1::2] = lst
  return result

def kl_divergence(m_p, logs_p, m_q, logs_q):
    """KL(P||Q)"""
    kl = (logs_q - logs_p) - 0.5
    kl += (
        0.5 * (torch.exp(2.0 * logs_p) + ((m_p - m_q) ** 2)) * torch.exp(-2.0 * logs_q)
    )
    return kl


def rand_gumbel(shape):
    """Sample from the Gumbel distribution, protect from overflows."""
    uniform_samples = torch.rand(shape) * 0.99998 + 0.00001
    return -torch.log(-torch.log(uniform_samples))


def rand_gumbel_like(x):
    g = rand_gumbel(x.size()).to(dtype=x.dtype, device=x.device)
    return g


def slice_segments(x, ids_str, segment_size=4):
    ret = torch.zeros_like(x[:, :, :segment_size])
    for i in range(x.size(0)):
        idx_str = ids_str[i]
        idx_end = idx_str + segment_size
        ret[i] = x[i, :, idx_str:idx_end]
    return ret


def slice_segments2(x, ids_str, segment_size=4):
    ret = torch.zeros_like(x[:, :segment_size])
    for i in range(x.size(0)):
        idx_str = ids_str[i]
        idx_end = idx_str + segment_size
        ret[i] = x[i, idx_str:idx_end]
    return ret


def rand_slice_segments(x, x_lengths=None, segment_size=4):
    b, d, t = x.size()
    if x_lengths is None:
        x_lengths = t
    ids_str_max = x_lengths - segment_size + 1
    ids_str = (torch.rand([b]).to(device=x.device) * ids_str_max).to(dtype=torch.long)
    ret = slice_segments(x, ids_str, segment_size)
    return ret, ids_str


def get_timing_signal_1d(length, channels, min_timescale=1.0, max_timescale=1.0e4):
    position = torch.arange(length, dtype=torch.float)
    num_timescales = channels // 2
    log_timescale_increment = math.log(float(max_timescale) / float(min_timescale)) / (
        num_timescales - 1
    )
    inv_timescales = min_timescale * torch.exp(
        torch.arange(num_timescales, dtype=torch.float) * -log_timescale_increment
    )
    scaled_time = position.unsqueeze(0) * inv_timescales.unsqueeze(1)
    signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], 0)
    signal = F.pad(signal, [0, 0, 0, channels % 2])
    signal = signal.view(1, channels, length)
    return signal


def add_timing_signal_1d(x, min_timescale=1.0, max_timescale=1.0e4):
    b, channels, length = x.size()
    signal = get_timing_signal_1d(length, channels, min_timescale, max_timescale)
    return x + signal.to(dtype=x.dtype, device=x.device)


def cat_timing_signal_1d(x, min_timescale=1.0, max_timescale=1.0e4, axis=1):
    b, channels, length = x.size()
    signal = get_timing_signal_1d(length, channels, min_timescale, max_timescale)
    return torch.cat([x, signal.to(dtype=x.dtype, device=x.device)], axis)


def subsequent_mask(length):
    mask = torch.tril(torch.ones(length, length)).unsqueeze(0).unsqueeze(0)
    return mask


@torch.jit.script
def fused_add_tanh_sigmoid_multiply(input_a, input_b, n_channels):
    n_channels_int = n_channels[0]
    in_act = input_a + input_b
    t_act = torch.tanh(in_act[:, :n_channels_int, :])
    s_act = torch.sigmoid(in_act[:, n_channels_int:, :])
    acts = t_act * s_act
    return acts


def convert_pad_shape(pad_shape):
    l = pad_shape[::-1]
    pad_shape = [item for sublist in l for item in sublist]
    return pad_shape


def shift_1d(x):
    x = F.pad(x, convert_pad_shape([[0, 0], [0, 0], [1, 0]]))[:, :, :-1]
    return x


def sequence_mask(length, max_length=None):
    if max_length is None:
        max_length = length.max()
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)


def generate_path(duration, mask):
    """
    duration: [b, 1, t_x]
    mask: [b, 1, t_y, t_x]
    """
    device = duration.device

    b, _, t_y, t_x = mask.shape
    cum_duration = torch.cumsum(duration, -1)

    cum_duration_flat = cum_duration.view(b * t_x)
    path = sequence_mask(cum_duration_flat, t_y).to(mask.dtype)
    path = path.view(b, t_x, t_y)
    path = path - F.pad(path, convert_pad_shape([[0, 0], [1, 0], [0, 0]]))[:, :-1]
    path = path.unsqueeze(1).transpose(2, 3) * mask
    return path


def clip_grad_value_(parameters, clip_value, norm_type=2, batch_size=1):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    if clip_value is not None:
        clip_value = float(clip_value)

    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)/batch_size
        total_norm += param_norm.item() ** norm_type
        if clip_value is not None:
            p.grad.data.clamp_(min=-clip_value, max=clip_value)
    total_norm = total_norm ** (1.0 / norm_type)
    return total_norm

def serialize_tensor(tensor):
    if hasattr(tensor,"item"): return tensor.item()
    else: return float(f"{float(tensor):.3f}")

def sigmoid_value(step, total_steps=100, start_value=0, end_value=1, midpoint=.5):
    """
    Calculate the scaled sigmoid function.

    Parameters:
    step (int): The current training epoch.
    total_steps (int): The total number of training epochs.
    start_value (float): The starting value of the clipping range.
    end_value (float): The ending value of the clipping range.
    midpoint (float): Percentage where sigmoid function is 50% between start and end value
    steepness (int): How fast the values change per step

    Returns:
    float: The calculated scaled sigmoid value.
    """
    # Sigmoid function parameters
    

    # Calculate the sigmoid value
    sigmoid_value = 1 / (1 + np.power(10,-np.log10(total_steps) * (step / total_steps - midpoint)))

    # Scale the sigmoid output to the desired range
    value = start_value + (end_value - start_value) * sigmoid_value
    return value