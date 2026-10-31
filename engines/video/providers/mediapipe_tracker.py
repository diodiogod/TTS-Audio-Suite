"""MediaPipe API compatibility and per-video face tracking lifecycle."""
import os


def mediapipe_backend():
    """Return the usable API, preserving existing legacy installations."""
    import mediapipe as mp
    if hasattr(getattr(mp, "solutions", None), "face_mesh"):
        return mp, "legacy"
    if hasattr(getattr(getattr(mp, "tasks", None), "vision", None), "FaceLandmarker"):
        return mp, "tasks"
    raise ImportError("MediaPipe has no supported face tracking API. Reinstall MediaPipe or select OpenSeeFace.")


def mediapipe_available():
    try:
        import cv2
        mediapipe_backend()
        return True
    except (ImportError, OSError, AttributeError):
        return False


class FaceTracker:
    """Expose normalized landmarks from legacy FaceMesh or Tasks VIDEO mode."""

    def __init__(self):
        self.mp, self.backend = mediapipe_backend()
        self.detector = None
        self.last_timestamp = -1

    def __enter__(self):
        if self.backend == "legacy":
            self.detector = self.mp.solutions.face_mesh.FaceMesh(
                max_num_faces=1, refine_landmarks=True,
                min_detection_confidence=0.5, min_tracking_confidence=0.5)
        else:
            from utils.downloads.unified_downloader import UnifiedDownloader
            downloader = UnifiedDownloader()
            filename = "face_landmarker.task"
            path = downloader.check_existing_model("mediapipe", filename)
            if path is None:
                path = os.path.join(downloader.get_organized_path("mediapipe"), filename)
                url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
                if not downloader.download_file(url, path, "MediaPipe Face Landmarker"):
                    raise RuntimeError("Could not download the MediaPipe Face Landmarker model.")
            vision = self.mp.tasks.vision
            options = vision.FaceLandmarkerOptions(
                base_options=self.mp.tasks.BaseOptions(model_asset_path=path),
                running_mode=vision.RunningMode.VIDEO, num_faces=1,
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5, min_tracking_confidence=0.5)
            self.detector = vision.FaceLandmarker.create_from_options(options)
        return self

    def landmarks(self, rgb_frame, timestamp_ms):
        if self.backend == "legacy":
            faces = self.detector.process(rgb_frame).multi_face_landmarks
            return faces[0].landmark if faces else None
        timestamp_ms = max(self.last_timestamp + 1, round(timestamp_ms))
        image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb_frame)
        result = self.detector.detect_for_video(image, timestamp_ms)
        self.last_timestamp = timestamp_ms
        return result.face_landmarks[0] if result.face_landmarks else None

    def __exit__(self, *_):
        if self.detector is not None:
            self.detector.close()
            self.detector = None
