import cv2
import json

from src.detector import FaceDetector
from src.embedder import FaceEmbedder
from src.tracker import FaceTracker
from src.database import FaceDatabase
from src.logger import AppLogger


class FaceTrackingPipeline:
    def __init__(self, config_path="config.json"):
        # Load configuration
        with open(config_path, "r", encoding="utf-8") as file:
            self.config = json.load(file)

        # Create components
        self.detector = FaceDetector(
            self.config["detector_weights"],
            self.config["det_conf"]
        )

        self.embedder = FaceEmbedder()

        self.tracker = FaceTracker(
            self.config["exit_timeout_frames"]
        )

        self.database = FaceDatabase(
            self.config["db_path"]
        )

        self.logger = AppLogger(
            self.config["log_dir"]
        )

        # Load already registered faces
        self.gallery = self.database.get_faces()

        # Track information
        self.track_to_face = {}
        self.last_seen = {}
        self.last_crop = {}

        # Active recognized faces
        self.active_faces = {}

    def find_matching_face(self, embedding):
        best_id = None
        best_similarity = -1.0

        for face_id, stored_embedding in self.gallery.items():
            similarity = float(
                embedding @ stored_embedding
            )

            if similarity > best_similarity:
                best_similarity = similarity
                best_id = face_id

        if (
            best_id is not None
            and best_similarity >= self.config["similarity_threshold"]
        ):
            return best_id

        return None

    def process_video(self):
        video_path = self.config["video_source"]

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise RuntimeError(
                f"Could not open video: {video_path}"
            )

        frame_number = 0

        self.logger.info(
            f"Starting video processing: {video_path}"
        )

        while True:
            success, frame = cap.read()

            if not success:
                break

            frame_number += 1

            # Process only selected frames
            if (
                frame_number
                % self.config["detection_skip_frames"]
                != 0
            ):
                continue

            # Face detection
            result = self.detector.detect(frame)

            detections = self._convert_detections(result)

            # Tracking
            tracked = self.tracker.update(detections)

            if tracked.tracker_id is None:
                continue

            for box, track_id in zip(
                tracked.xyxy,
                tracked.tracker_id
            ):
                track_id = int(track_id)

                x1, y1, x2, y2 = map(
                    int,
                    box
                )

                width = x2 - x1
                height = y2 - y1

                # Ignore very small faces
                if (
                    min(width, height)
                    < self.config["min_face_size"]
                ):
                    continue

                # Keep coordinates inside frame
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(frame.shape[1], x2)
                y2 = min(frame.shape[0], y2)

                crop = frame[y1:y2, x1:x2]

                if crop.size == 0:
                    continue

                self.last_seen[track_id] = frame_number
                self.last_crop[track_id] = crop

                # New track
                if track_id not in self.track_to_face:

                    embedding = self.embedder.get_embedding(
                        frame,
                        box
                    )

                    if embedding is None:
                        continue

                    face_id = self.find_matching_face(
                        embedding
                    )

                    # New person
                    if face_id is None:

                        face_id = self.database.register_face(
                            embedding
                        )

                        self.gallery[face_id] = embedding

                        self.logger.info(
                            f"New face registered: id={face_id}"
                        )

                    # Existing person
                    else:
                        self.logger.info(
                            f"Face recognized: id={face_id}"
                        )

                    self.track_to_face[
                        track_id
                    ] = face_id

                    # Add this tracking ID to the active face
                    if face_id not in self.active_faces:
                        self.active_faces[face_id] = set()

                    self.active_faces[face_id].add(track_id)

                    # Log entry only when this face was not already active
                    if len(self.active_faces[face_id]) == 1:
                        self.database.log_event(
                            face_id,
                            "entry"
                        )

            # Check for people who have left
            self._check_exits(frame_number)

        cap.release()

        # Mark remaining tracked people as exited
        self._flush_remaining_tracks()

        self.database.close()

        self.logger.info(
            "Video processing completed."
        )

    def _convert_detections(self, result):
        """
        Convert Ultralytics result to supervision Detections.
        """
        import supervision as sv

        return sv.Detections.from_ultralytics(
            result
        )

    def _check_exits(self, current_frame):
        timeout = self.config[
            "exit_timeout_frames"
        ]

        expired_tracks = []

        for track_id, last_frame in self.last_seen.items():

            if (
                current_frame - last_frame
                > timeout
            ):
                expired_tracks.append(track_id)

        for track_id in expired_tracks:

            face_id = self.track_to_face.pop(
                track_id,
                None
            )

            self.last_seen.pop(
                track_id,
                None
            )

            self.last_crop.pop(
                track_id,
                None
            )

            if face_id is not None:

                # Remove this tracking ID from the active face
                if face_id in self.active_faces:
                    self.active_faces[face_id].discard(track_id)

                    # Remove face only when no tracking IDs remain
                    if not self.active_faces[face_id]:
                        self.active_faces.pop(face_id)

                        self.database.log_event(
                            face_id,
                            "exit"
                        )

                        self.logger.info(
                            f"Face exited: id={face_id}"
                        )

    def _flush_remaining_tracks(self):
        for track_id, face_id in list(
            self.track_to_face.items()
        ):

            # Remove this tracking ID from the active face
            if face_id in self.active_faces:
                self.active_faces[face_id].discard(track_id)

                # Log only one exit for each active face
                if not self.active_faces[face_id]:
                    self.active_faces.pop(face_id)

                    self.database.log_event(
                        face_id,
                        "exit"
                    )

                    self.logger.info(
                        f"Face exited at end of video: id={face_id}"
                    )

        self.track_to_face.clear()
        self.last_seen.clear()
        self.last_crop.clear()
        self.active_faces.clear()


if __name__ == "__main__":
    pipeline = FaceTrackingPipeline()
    pipeline.process_video()