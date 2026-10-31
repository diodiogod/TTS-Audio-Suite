"""Browser-playable preview encoding shared by mouth analysis providers."""
import logging
import os
import uuid
from fractions import Fraction

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class PreviewVideoCreator:
    @staticmethod
    def create_preview_video(frames, fps, width, height, provider_name="Unknown"):
        if not frames or not np.isfinite(fps) or fps <= 0:
            logger.warning("Preview needs video frames and a valid frame rate.")
            return None
        import folder_paths
        # The node serves previews with ComfyUI's type=temp file route.
        output_dir = folder_paths.get_temp_directory()
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, f"mouth_preview_{uuid.uuid4().hex}.webm")
        try:
            import av
            with av.open(path, mode="w") as container:
                stream = container.add_stream("libvpx-vp9", rate=Fraction(str(fps)).limit_denominator(100000))
                stream.width, stream.height = width, height
                stream.pix_fmt = "yuv420p"
                stream.options = {"crf": "45", "deadline": "realtime", "cpu-used": "8"}
                for frame in frames:
                    if frame.shape[:2] != (height, width):
                        frame = cv2.resize(frame, (width, height))
                    video_frame = av.VideoFrame.from_ndarray(frame, format="bgr24")
                    for packet in stream.encode(video_frame):
                        container.mux(packet)
                for packet in stream.encode():
                    container.mux(packet)
            logger.info(f"Created {provider_name} preview: {path}")
            return path
        except Exception as exc:
            logger.warning(f"Could not encode preview video: {exc}. Timing results are still available.")
            if os.path.exists(path):
                os.unlink(path)
            return None
