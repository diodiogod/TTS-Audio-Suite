"""
GGUF Utilities for VibeVoice - Complete dequantization implementation
Copied and adapted from ComfyUI-GGUF reference implementation
"""

import gguf
import torch
import numpy as np
from tqdm import tqdm
from typing import Dict, Any, Optional


class GGMLTensor(torch.Tensor):
    """
    A proper PyTorch Tensor subclass for GGUF quantized tensors that keeps 
    quantized data in GPU memory for real VRAM savings.
    """
    
    def __new__(cls, data, tensor_type, tensor_shape, device='cpu', dtype=torch.float16):
        """
        Create a new GGMLTensor that behaves like a regular PyTorch tensor
        but stores ONLY the compressed quantized data to save memory.
        """
        # Create a tiny dummy tensor to satisfy PyTorch's tensor creation
        # The actual quantized data is stored separately
        dummy_tensor = torch.zeros(1, device=device, dtype=torch.uint8)  # Minimal memory
        obj = torch.Tensor._make_subclass(cls, dummy_tensor, require_grad=False)
        
        # Store ONLY the compressed quantized data and metadata
        # DO NOT store the full expanded tensor - this causes massive RAM usage
        if hasattr(data, 'numpy'):
            # Convert to minimal numpy array to save memory
            obj.gguf_data = data.numpy().copy()  # Copy to avoid mmap issues
        else:
            obj.gguf_data = data
            
        obj.tensor_type = tensor_type           # GGUF quantization type  
        obj.tensor_shape = tensor_shape         # Original unquantized shape
        obj._device = device                    # Track actual device
        obj._dtype = dtype                      # Target dtype when dequantized
        obj._dequantized_cache = None           # Cache for performance
        
# Removed verbose tensor creation logs
        
        return obj
    
    @property
    def shape(self):
        """Return the original tensor shape (not the dummy tensor shape)"""
        return self.tensor_shape
    
    @property  
    def device(self):
        """Return the actual device where quantized data lives"""
        return self._device
    
    @property
    def dtype(self):
        """Return expected dtype when dequantized"""
        return self._dtype
    
    def dequantize_for_computation(self):
        """
        Dequantize tensor for immediate computation (real GGUF approach).
        This is called automatically during forward passes.
        """
        # Cache the dequantized tensor briefly for performance
        if self._dequantized_cache is None:
            print(f"🔄 Dequantizing {self.tensor_shape} from compressed data...")
            
            # Convert numpy data back to torch tensor for dequantization
            if isinstance(self.gguf_data, np.ndarray):
                torch_data = torch.from_numpy(self.gguf_data)
            else:
                torch_data = self.gguf_data
                
            # Create temporary tensor with correct metadata for dequantization
            temp_tensor = type('TempTensor', (), {
                'data': torch_data,
                'tensor_type': self.tensor_type, 
                'tensor_shape': self.tensor_shape
            })()
            
            self._dequantized_cache = dequantize_tensor(temp_tensor, dtype=self._dtype)
            
            # Move dequantized tensor to target device
            if self._device != 'cpu':
                self._dequantized_cache = self._dequantized_cache.to(self._device)
                
            print(f"✅ Dequantized to {self._dequantized_cache.shape} {self._dequantized_cache.dtype} on {self._dequantized_cache.device}")
        return self._dequantized_cache
    
    def clear_dequant_cache(self):
        """Clear dequantized cache to save VRAM"""
        self._dequantized_cache = None
    
    def to(self, device):
        """Move quantized data to device - this is crucial for VRAM savings"""
        if isinstance(device, str):
            device = torch.device(device)
        
        print(f"🔧 Moving GGMLTensor {self.tensor_shape} from {self._device} to {device}")
        
        if device == self._device:
            return self
            
        # For numpy data, we keep it on CPU and move during dequantization
        # This avoids the massive memory spike during loading
        if isinstance(self.gguf_data, np.ndarray):
            print(f"✅ Keeping quantized numpy data on CPU, will move to {device} during dequantization")
            new_data = self.gguf_data  # Keep numpy on CPU
        elif hasattr(self.gguf_data, 'to'):
            new_data = self.gguf_data.to(device)
            print(f"✅ Moved tensor data to {device}: {new_data.device}")
        else:
            new_data = self.gguf_data
            print(f"⚠️ Keeping data as-is")
        
        # Create new tensor with target device
        new_tensor = GGMLTensor(
            new_data, 
            self.tensor_type, 
            self.tensor_shape, 
            device=device,
            dtype=self._dtype
        )
        return new_tensor
    
    def cuda(self, device=None):
        """Move to CUDA device - this is where VRAM savings happen"""
        if device is None:
            device = torch.cuda.current_device()
        return self.to(f'cuda:{device}')
    
    def cpu(self):
        """Move to CPU device"""
        return self.to('cpu')
    
    def size(self, dim=None):
        """Return tensor size"""
        if dim is None:
            return self.tensor_shape
        return self.tensor_shape[dim] if dim < len(self.tensor_shape) else 1
    
    def dim(self):
        """Return number of dimensions"""
        return len(self.tensor_shape)
    
    def numel(self):
        """Return number of elements"""
        return torch.Size(self.tensor_shape).numel()
    
    def __str__(self):
        return f"GGMLTensor(shape={self.tensor_shape}, type={self.tensor_type}, device={self._device})"
    
    def __repr__(self):
        return self.__str__()

# Compatible tensor types (no dequantization needed)
TORCH_COMPATIBLE_QTYPES = (None, gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16)

def is_torch_compatible(tensor):
    return tensor is None or getattr(tensor, "tensor_type", None) in TORCH_COMPATIBLE_QTYPES

def is_quantized(tensor):
    return not is_torch_compatible(tensor)

def dequantize_tensor(tensor, dtype=None, dequant_dtype=None):
    qtype = getattr(tensor, "tensor_type", None)
    oshape = getattr(tensor, "tensor_shape", tensor.shape)

    if qtype in TORCH_COMPATIBLE_QTYPES:
        return tensor.to(dtype)
    elif qtype in dequantize_functions:
        dequant_dtype = dtype if dequant_dtype == "target" else dequant_dtype
        return dequantize(tensor.data, qtype, oshape, dtype=dequant_dtype).to(dtype)
    else:
        # this is incredibly slow
        tqdm.write(f"Falling back to numpy dequant for qtype: {qtype}")
        new = gguf.quants.dequantize(tensor.cpu().numpy(), qtype)
        return torch.from_numpy(new).to(tensor.device, dtype=dtype)

def dequantize(data, qtype, oshape, dtype=None):
    """
    Dequantize tensor back to usable shape/dtype
    """
    block_size, type_size = gguf.GGML_QUANT_SIZES[qtype]
    dequantize_blocks = dequantize_functions[qtype]

    rows = data.reshape(
        (-1, data.shape[-1])
    ).view(torch.uint8)

    n_blocks = rows.numel() // type_size
    blocks = rows.reshape((n_blocks, type_size))
    blocks = dequantize_blocks(blocks, block_size, type_size, dtype)
    return blocks.reshape(oshape)

def to_uint32(x):
    # no uint32 :(
    x = x.view(torch.uint8).to(torch.int32)
    return (x[:, 0] | x[:, 1] << 8 | x[:, 2] << 16 | x[:, 3] << 24).unsqueeze(1)

def split_block_dims(blocks, *args):
    n_max = blocks.shape[1]
    dims = list(args) + [n_max - sum(args)]
    return torch.split(blocks, dims, dim=1)

# Full weights #
def dequantize_blocks_BF16(blocks, block_size, type_size, dtype=None):
    return (blocks.view(torch.int16).to(torch.int32) << 16).view(torch.float32)

# Legacy Quants #
def dequantize_blocks_Q8_0(blocks, block_size, type_size, dtype=None):
    d, x = split_block_dims(blocks, 2)
    d = d.view(torch.float16).to(dtype)
    x = x.view(torch.int8)
    return (d * x)

def dequantize_blocks_Q5_1(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]

    d, m, qh, qs = split_block_dims(blocks, 2, 2, 4)
    d = d.view(torch.float16).to(dtype)
    m = m.view(torch.float16).to(dtype)
    qh = to_uint32(qh)

    qh = qh.reshape((n_blocks, 1)) >> torch.arange(32, device=d.device, dtype=torch.int32).reshape(1, 32)
    ql = qs.reshape((n_blocks, -1, 1, block_size // 2)) >> torch.tensor([0, 4], device=d.device, dtype=torch.uint8).reshape(1, 1, 2, 1)
    qh = (qh & 1).to(torch.uint8)
    ql = (ql & 0x0F).reshape((n_blocks, -1))

    qs = (ql | (qh << 4))
    return (d * qs) + m

def dequantize_blocks_Q5_0(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]

    d, qh, qs = split_block_dims(blocks, 2, 4)
    d  = d.view(torch.float16).to(dtype)
    qh = to_uint32(qh)

    qh = qh.reshape(n_blocks, 1) >> torch.arange(32, device=d.device, dtype=torch.int32).reshape(1, 32)
    ql = qs.reshape(n_blocks, -1, 1, block_size // 2) >> torch.tensor([0, 4], device=d.device, dtype=torch.uint8).reshape(1, 1, 2, 1)

    qh = (qh & 1).to(torch.uint8)
    ql = (ql & 0x0F).reshape(n_blocks, -1)

    qs = (ql | (qh << 4)).to(torch.int8) - 16
    return (d * qs)

def dequantize_blocks_Q4_1(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]

    d, m, qs = split_block_dims(blocks, 2, 2)
    d = d.view(torch.float16).to(dtype)
    m = m.view(torch.float16).to(dtype)

    qs = qs.reshape((n_blocks, -1, 1, block_size // 2)) >> torch.tensor([0, 4], device=d.device, dtype=torch.uint8).reshape(1, 1, 2, 1)
    qs = (qs & 0x0F).reshape(n_blocks, -1)

    return (d * qs) + m

def dequantize_blocks_Q4_0(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]

    d, qs = split_block_dims(blocks, 2)
    d  = d.view(torch.float16).to(dtype)

    qs = qs.reshape((n_blocks, -1, 1, block_size // 2)) >> torch.tensor([0, 4], device=d.device, dtype=torch.uint8).reshape((1, 1, 2, 1))
    qs = (qs & 0x0F).reshape((n_blocks, -1)).to(torch.int8) - 8
    return (d * qs)

# K Quants #
QK_K = 256
K_SCALE_SIZE = 12

def get_scale_min(scales):
    n_blocks = scales.shape[0]
    scales = scales.view(torch.uint8)
    scales = scales.reshape((n_blocks, 3, 4))

    d, m, m_d = torch.split(scales, scales.shape[-2] // 3, dim=-2)

    sc = torch.cat([d & 0x3F, (m_d & 0x0F) | ((d >> 2) & 0x30)], dim=-1)
    min = torch.cat([m & 0x3F, (m_d >> 4) | ((m >> 2) & 0x30)], dim=-1)

    return (sc.reshape((n_blocks, 8)), min.reshape((n_blocks, 8)))

def dequantize_blocks_Q6_K(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]

    ql, qh, scales, d, = split_block_dims(blocks, QK_K // 2, QK_K // 4, QK_K // 16)

    scales = scales.view(torch.int8).to(dtype)
    d = d.view(torch.float16).to(dtype)
    d = (d * scales).reshape((n_blocks, QK_K // 16, 1))

    ql = ql.reshape((n_blocks, -1, 1, 64)) >> torch.tensor([0, 4], device=d.device, dtype=torch.uint8).reshape((1, 1, 2, 1))
    ql = (ql & 0x0F).reshape((n_blocks, -1, 32))
    qh = qh.reshape((n_blocks, -1, 1, 32)) >> torch.tensor([0, 2, 4, 6], device=d.device, dtype=torch.uint8).reshape((1, 1, 4, 1))
    qh = (qh & 0x03).reshape((n_blocks, -1, 32))
    q = (ql | (qh << 4)).to(torch.int8) - 32
    q = q.reshape((n_blocks, QK_K // 16, -1))

    return (d * q).reshape((n_blocks, QK_K))

def dequantize_blocks_Q5_K(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]

    d, dmin, scales, qh, qs = split_block_dims(blocks, 2, 2, K_SCALE_SIZE, QK_K // 8)

    d = d.view(torch.float16).to(dtype)
    dmin = dmin.view(torch.float16).to(dtype)

    sc, m = get_scale_min(scales)

    d = (d * sc).reshape((n_blocks, -1, 1))
    dm = (dmin * m).reshape((n_blocks, -1, 1))

    ql = qs.reshape((n_blocks, -1, 1, 32)) >> torch.tensor([0, 4], device=d.device, dtype=torch.uint8).reshape((1, 1, 2, 1))
    qh = qh.reshape((n_blocks, -1, 1, 32)) >> torch.tensor([i for i in range(8)], device=d.device, dtype=torch.uint8).reshape((1, 1, 8, 1))
    ql = (ql & 0x0F).reshape((n_blocks, -1, 32))
    qh = (qh & 0x01).reshape((n_blocks, -1, 32))
    q = (ql | (qh << 4))

    return (d * q - dm).reshape((n_blocks, QK_K))

def dequantize_blocks_Q4_K(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]

    d, dmin, scales, qs = split_block_dims(blocks, 2, 2, K_SCALE_SIZE)
    d = d.view(torch.float16).to(dtype)
    dmin = dmin.view(torch.float16).to(dtype)

    sc, m = get_scale_min(scales)

    d = (d * sc).reshape((n_blocks, -1, 1))
    dm = (dmin * m).reshape((n_blocks, -1, 1))

    qs = qs.reshape((n_blocks, -1, 1, 32)) >> torch.tensor([0, 4], device=d.device, dtype=torch.uint8).reshape((1, 1, 2, 1))
    qs = (qs & 0x0F).reshape((n_blocks, -1, 32))

    return (d * qs - dm).reshape((n_blocks, QK_K))

def dequantize_blocks_Q3_K(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]

    hmask, qs, scales, d = split_block_dims(blocks, QK_K // 8, QK_K // 4, 12)
    d = d.view(torch.float16).to(dtype)

    lscales, hscales = scales[:, :8], scales[:, 8:]
    lscales = lscales.reshape((n_blocks, 1, 8)) >> torch.tensor([0, 4], device=d.device, dtype=torch.uint8).reshape((1, 2, 1))
    lscales = lscales.reshape((n_blocks, 16))
    hscales = hscales.reshape((n_blocks, 1, 4)) >> torch.tensor([0, 2, 4, 6], device=d.device, dtype=torch.uint8).reshape((1, 4, 1))
    hscales = hscales.reshape((n_blocks, 16))
    scales = (lscales & 0x0F) | ((hscales & 0x03) << 4)
    scales = (scales.to(torch.int8) - 32)

    dl = (d * scales).reshape((n_blocks, 16, 1))

    ql = qs.reshape((n_blocks, -1, 1, 32)) >> torch.tensor([0, 2, 4, 6], device=d.device, dtype=torch.uint8).reshape((1, 1, 4, 1))
    qh = hmask.reshape(n_blocks, -1, 1, 32) >> torch.tensor([i for i in range(8)], device=d.device, dtype=torch.uint8).reshape((1, 1, 8, 1))
    ql = ql.reshape((n_blocks, 16, QK_K // 16)) & 3
    qh = (qh.reshape((n_blocks, 16, QK_K // 16)) & 1) ^ 1
    q = (ql.to(torch.int8) - (qh << 2).to(torch.int8))

    return (dl * q).reshape((n_blocks, QK_K))

def dequantize_blocks_Q2_K(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]

    scales, qs, d, dmin = split_block_dims(blocks, QK_K // 16, QK_K // 4, 2)
    d = d.view(torch.float16).to(dtype)
    dmin = dmin.view(torch.float16).to(dtype)

    # (n_blocks, 16, 1)
    dl = (d * (scales & 0xF)).reshape((n_blocks, QK_K // 16, 1))
    ml = (dmin * (scales >> 4)).reshape((n_blocks, QK_K // 16, 1))

    shift = torch.tensor([0, 2, 4, 6], device=d.device, dtype=torch.uint8).reshape((1, 1, 4, 1))

    qs = (qs.reshape((n_blocks, -1, 1, 32)) >> shift) & 3
    qs = qs.reshape((n_blocks, QK_K // 16, 16))
    qs = dl * qs - ml

    return qs.reshape((n_blocks, -1))

# Mapping of quantization types to their dequantization functions
dequantize_functions = {
    gguf.GGMLQuantizationType.BF16: dequantize_blocks_BF16,
    gguf.GGMLQuantizationType.Q8_0: dequantize_blocks_Q8_0,
    gguf.GGMLQuantizationType.Q5_1: dequantize_blocks_Q5_1,
    gguf.GGMLQuantizationType.Q5_0: dequantize_blocks_Q5_0,
    gguf.GGMLQuantizationType.Q4_1: dequantize_blocks_Q4_1,
    gguf.GGMLQuantizationType.Q4_0: dequantize_blocks_Q4_0,
    gguf.GGMLQuantizationType.Q6_K: dequantize_blocks_Q6_K,
    gguf.GGMLQuantizationType.Q5_K: dequantize_blocks_Q5_K,
    gguf.GGMLQuantizationType.Q4_K: dequantize_blocks_Q4_K,
    gguf.GGMLQuantizationType.Q3_K: dequantize_blocks_Q3_K,
    gguf.GGMLQuantizationType.Q2_K: dequantize_blocks_Q2_K,
}

def load_gguf_state_dict(gguf_path: str) -> Dict[str, torch.Tensor]:
    """
    Load GGUF file and return state dict with tensor objects
    Simplified loader adapted from reference implementation
    """
    import warnings
    
    reader = gguf.GGUFReader(gguf_path)
    
    # Load all tensors
    state_dict = {}
    for tensor in reader.tensors:
        tensor_name = tensor.name
        
        # NOTE: line below from reference to avoid persistent numpy warning about mmap
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="The given NumPy array is not writable")
            torch_tensor = torch.from_numpy(tensor.data) # mmap
        
        # Get original shape from GGUF metadata or infer from tensor
        shape = torch.Size(tuple(int(v) for v in reversed(tensor.shape)))
        
        # Only log issues, not every tensor
        # print(f"🐛 GGUF DEBUG: Processing tensor '{tensor_name}'")
        
        # Create tensor with GGUF metadata for quantization detection
        if tensor.tensor_type in TORCH_COMPATIBLE_QTYPES:
            # Standard tensor - just reshape
            if shape != torch_tensor.shape:
                if len(shape) == 0:
                    print(f"❌ GGUF: Empty shape detected for {tensor_name}! Skipping reshape.")
                    state_dict[tensor_name] = torch_tensor
                else:
                    torch_tensor = torch_tensor.view(*shape)
                    state_dict[tensor_name] = torch_tensor
            else:
                state_dict[tensor_name] = torch_tensor
        else:
            # Quantized tensor - keep quantized for VRAM savings
            state_dict[tensor_name] = GGMLTensor(torch_tensor, tensor.tensor_type, shape)
    
    return state_dict