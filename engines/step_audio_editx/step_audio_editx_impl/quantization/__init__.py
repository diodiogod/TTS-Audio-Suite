"""
量化工具包

提供各种模型量化方法，包括：
- AWQ量化 (llmcompressor)
- 与现有BitsAndBytes量化的集成

使用方法:
    from quantization import awq_quantize
"""

# 导入主要的量化模块
from .awq_quantize import quantize_model

__all__ = [
    "quantize_model",
]