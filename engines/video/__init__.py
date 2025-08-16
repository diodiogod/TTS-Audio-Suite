"""
Video analysis engines for mouth movement detection
"""

from typing import Dict, Type, Optional
import logging

logger = logging.getLogger(__name__)

# Provider registry
AVAILABLE_PROVIDERS: Dict[str, Type] = {}


def register_provider(name: str, provider_class: Type):
    """Register a video analysis provider"""
    AVAILABLE_PROVIDERS[name] = provider_class
    logger.info(f"Registered video analysis provider: {name}")


def get_provider(name: str) -> Optional[Type]:
    """Get a registered provider by name"""
    return AVAILABLE_PROVIDERS.get(name)


# Auto-discover and register providers
def discover_providers():
    """Discover and register available providers"""
    
    # Try to import MediaPipe provider
    try:
        from .providers.mediapipe_provider import MediaPipeProvider
        register_provider("MediaPipe", MediaPipeProvider)
    except ImportError as e:
        logger.debug(f"MediaPipe provider not available: {e}")
    
    # Try to import OpenSeeFace provider
    try:
        from .providers.openseeface_provider import OpenSeeFaceProvider
        register_provider("OpenSeeFace", OpenSeeFaceProvider)
    except ImportError as e:
        logger.debug(f"OpenSeeFace provider not available: {e}")
    
    # Try to import dlib provider
    try:
        from .providers.dlib_provider import DlibProvider
        register_provider("dlib", DlibProvider)
    except ImportError as e:
        logger.debug(f"dlib provider not available: {e}")


# Run discovery on import
discover_providers()