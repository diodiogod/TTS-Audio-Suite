"""
Base class for engine-specific handlers
"""

from abc import ABC, abstractmethod
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..base_wrapper import ComfyUIModelWrapper


class BaseEngineHandler(ABC):
    """
    Base class for engine-specific model handling.
    
    Provides interface for engine-specific unloading and memory management.
    """
    
    @abstractmethod
    def partially_unload(self, wrapper: 'ComfyUIModelWrapper', device: str, memory_to_free: int) -> int:
        """
        Partially unload the model to free memory.
        
        Args:
            wrapper: The ComfyUI model wrapper
            device: Target device to move to (usually 'cpu')
            memory_to_free: Amount of memory to free in bytes
            
        Returns:
            Amount of memory actually freed in bytes
        """
        pass
    
    @abstractmethod 
    def model_unload(self, wrapper: 'ComfyUIModelWrapper', memory_to_free: Optional[int], unpatch_weights: bool) -> bool:
        """
        Fully unload the model from GPU memory.
        
        Args:
            wrapper: The ComfyUI model wrapper
            memory_to_free: Amount of memory to free (ignored for full unload)
            unpatch_weights: Whether to unpatch weights (TTS models don't use this)
            
        Returns:
            True if model was unloaded, False otherwise
        """
        pass