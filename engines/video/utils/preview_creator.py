"""
Unified Preview Video Creation Utility
Handles preview video generation for all video analysis providers
"""

import os
import cv2
import time
import logging
from typing import List, Optional
import numpy as np

logger = logging.getLogger(__name__)


class PreviewVideoCreator:
    """
    Unified preview video creation utility for all video analysis providers.
    Uses the same proven logic as MediaPipe for consistency and reliability.
    """
    
    @staticmethod
    def create_preview_video(frames: List[np.ndarray], fps: float, width: int, height: int, provider_name: str = "Unknown") -> Optional[str]:
        """
        Create preview video with annotations using proven MediaPipe logic
        
        Args:
            frames: List of annotated video frames
            fps: Video frame rate
            width: Video width
            height: Video height  
            provider_name: Name of the provider creating the preview
            
        Returns:
            Path to created preview video file, or None if creation failed
        """
        try:
            if not frames:
                logger.warning("No frames available for preview video creation")
                return None
                
            logger.info(f"Creating {provider_name} preview video with {len(frames)} frames, {width}x{height} at {fps} FPS")
            
            # Create output path in ComfyUI output directory like Save Video does
            import folder_paths
            
            output_dir = folder_paths.get_output_directory()
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate unique filename
            timestamp = int(time.time())
            filename_mp4 = f"mouth_preview_{provider_name.lower()}_{timestamp}.mp4"
            filename_webp = f"mouth_preview_{provider_name.lower()}_{timestamp}.webp"
            output_path_mp4 = os.path.join(output_dir, filename_mp4)
            output_path_webp = os.path.join(output_dir, filename_webp)
            
            # Create MP4 video
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path_mp4, fourcc, fps, (width, height))
            
            if not out.isOpened():
                logger.error(f"Failed to open video writer for {output_path_mp4}")
                return None
            
            frames_written = 0
            for i, frame in enumerate(frames):
                try:
                    # Ensure frame dimensions match
                    if frame.shape[:2] != (height, width):
                        frame = cv2.resize(frame, (width, height))
                    out.write(frame)
                    frames_written += 1
                except Exception as e:
                    logger.warning(f"Failed to write frame {i}: {e}")
            
            out.release()
            logger.info(f"Successfully wrote {frames_written}/{len(frames)} frames to MP4")
            
            # Create WEBM video for native ComfyUI display (like SaveWEBM - better performance)
            webm_path = PreviewVideoCreator._create_webm_preview(frames, fps, width, height, output_path_webp)
            
            if webm_path and os.path.exists(webm_path):
                logger.info(f"Created {provider_name} WebM preview: {webm_path}")
                # Clean up MP4 if WebM was successful
                try:
                    os.unlink(output_path_mp4)
                    return webm_path
                except:
                    pass
            
            logger.info(f"Created {provider_name} MP4 preview: {output_path_mp4}")
            return output_path_mp4
            
        except Exception as e:
            logger.error(f"Failed to create {provider_name} preview video: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    @staticmethod
    def _create_webm_preview(frames: List[np.ndarray], fps: float, width: int, height: int, output_path_webp: str) -> Optional[str]:
        """
        Create WebM preview using av library (like SaveWEBM) - optimized for fast preview
        """
        try:
            import av
            from fractions import Fraction
            
            # Convert BGR frames to RGB
            rgb_frames = []
            for frame in frames:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb_frames.append(rgb_frame)
            
            # Create WEBM using av library like SaveWEBM does - optimized for fast preview
            webm_path = output_path_webp.replace('.webp', '.webm')
            container = av.open(webm_path, mode="w")
            stream = container.add_stream("libvpx-vp9", rate=Fraction(round(fps * 1000), 1000))
            stream.width = width
            stream.height = height
            stream.pix_fmt = "yuv420p"
            
            # Fast encoding options for preview (lower quality but much faster)
            stream.options = {
                "crf": "45",  # Higher CRF = lower quality but faster encoding
                "speed": "8",  # Fastest encoding speed
                "cpu-used": "8"  # Maximum CPU efficiency mode
            }
            
            for rgb_frame in rgb_frames:
                av_frame = av.VideoFrame.from_ndarray(rgb_frame, format="rgb24")
                for packet in stream.encode(av_frame):
                    container.mux(packet)
            
            # Flush encoder
            for packet in stream.encode():
                container.mux(packet)
            
            container.close()
            
            if os.path.exists(webm_path) and os.path.getsize(webm_path) > 1000:  # Sanity check
                return webm_path
            else:
                logger.warning("WebM creation failed or file too small")
                return None
                
        except Exception as e:
            logger.warning(f"WebM creation failed: {e}")
            return None