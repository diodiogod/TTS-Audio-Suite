"""
Voice Discovery Cache Manager
Persistent caching system for voice character discovery to speed up initialization.
Cache is stored in .cache/ folder (gitignored) with background refresh.
"""

import os
import json
import time
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set, Callable
from datetime import datetime, timedelta


class VoiceDiscoveryCacheManager:
    """Manage persistent cache for voice discovery results with background refresh."""

    CACHE_VERSION = "1.0"

    def __init__(self, node_root_dir: Optional[str] = None):
        """
        Initialize cache manager.

        Args:
            node_root_dir: Root directory of the TTS Audio Suite node (auto-detected if None)
        """
        self.node_root_dir = node_root_dir or self._get_node_root()
        self.cache_dir = os.path.join(self.node_root_dir, ".cache")
        self.cache_file = os.path.join(self.cache_dir, "voice_discovery.json")
        self.metadata_file = os.path.join(self.cache_dir, "cache_metadata.json")

        # Background refresh tracking
        self._refresh_thread = None
        self._refresh_callback = None
        self._stop_refresh = False
        self._refresh_lock = threading.Lock()

        # Ensure cache directory exists
        os.makedirs(self.cache_dir, exist_ok=True)

    def _get_node_root(self) -> str:
        """Get the root directory of the TTS Audio Suite node."""
        # This file is at utils/voice/cache_manager.py, so go up 3 levels to get node root
        return os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

    def is_cache_valid(self) -> bool:
        """
        Check if cache file exists and is readable.
        No expiration check - cache lives until ComfyUI refresh or manually invalidated.

        Returns:
            True if cache file exists and can be loaded
        """
        if not os.path.exists(self.cache_file):
            return False

        try:
            with open(self.cache_file, 'r', encoding='utf-8') as f:
                json.load(f)  # Validate JSON is readable
            return True
        except Exception:
            return False

    def load_cache(self) -> Optional[Dict]:
        """
        Load voice discovery cache from disk.

        Returns:
            Cached data or None if cache doesn't exist/is invalid
        """
        if not os.path.exists(self.cache_file):
            return None

        try:
            with open(self.cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            # Silently fail - cache is just an optimization
            return None

    def save_cache(self, cache_data: Dict):
        """
        Save voice discovery cache to disk.

        Args:
            cache_data: Voice discovery data to cache
        """
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, indent=2)
        except Exception:
            # Silently fail - cache is just an optimization
            pass

    def invalidate_cache(self):
        """Manually invalidate the cache."""
        try:
            if os.path.exists(self.cache_file):
                os.remove(self.cache_file)
            if os.path.exists(self.metadata_file):
                os.remove(self.metadata_file)
        except Exception:
            pass

    def start_background_refresh(self, refresh_callback: Callable[[], Tuple[Dict, bool]]):
        """
        Start background refresh thread. Scans directories once after ComfyUI loads.

        Args:
            refresh_callback: Callable that returns (cache_data, was_updated) tuple
        """
        if self._refresh_thread is not None and self._refresh_thread.is_alive():
            return  # Already running

        self._refresh_callback = refresh_callback
        self._stop_refresh = False
        self._refresh_thread = threading.Thread(target=self._background_refresh_worker, daemon=True)
        self._refresh_thread.start()

    def _background_refresh_worker(self):
        """Background thread worker - performs one cache update after startup."""
        try:
            # Give ComfyUI time to fully load before starting scan
            time.sleep(2)

            if self._stop_refresh:
                return

            # Call the refresh callback to get updated data
            if self._refresh_callback:
                with self._refresh_lock:
                    result = self._refresh_callback()
                    if isinstance(result, tuple):
                        updated_data, was_updated = result
                    else:
                        updated_data = result
                        was_updated = False

                    if updated_data:
                        self.save_cache(updated_data)
                        # Return whether cache was actually updated
                        return was_updated
        except Exception:
            pass  # Silently fail - background refresh is optional

    def stop_background_refresh(self):
        """Stop the background refresh thread."""
        self._stop_refresh = True
        if self._refresh_thread and self._refresh_thread.is_alive():
            self._refresh_thread.join(timeout=1)
