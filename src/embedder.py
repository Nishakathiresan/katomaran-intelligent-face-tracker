import cv2
import numpy as np
from insightface.app import FaceAnalysis


class FaceEmbedder:
    def __init__(self):
        self.app = FaceAnalysis(
            name="buffalo_l",
            providers=["CPUExecutionProvider"]
        )

        self.app.prepare(
            ctx_id=0,
            det_size=(640, 640)
        )

    def get_embedding(self, frame, box):
        x1, y1, x2, y2 = map(int, box)

        width = x2 - x1
        height = y2 - y1

        # Add generous padding around YOLO detection
        pad_x = int(width * 0.75)
        pad_y = int(height * 0.75)

        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(frame.shape[1], x2 + pad_x)
        y2 = min(frame.shape[0], y2 + pad_y)

        crop = frame[y1:y2, x1:x2]

        if crop.size == 0:
            return None

        faces = self.app.get(crop)

        if not faces:
            return None

        face = max(
            faces,
            key=lambda f:
            (f.bbox[2] - f.bbox[0]) *
            (f.bbox[3] - f.bbox[1])
        )

        return np.asarray(
            face.normed_embedding,
            dtype=np.float32
        )