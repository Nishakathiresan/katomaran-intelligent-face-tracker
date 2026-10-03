import supervision as sv


class FaceTracker:
    def __init__(self, lost_track_buffer=30):
        self.tracker = sv.ByteTrack(
            lost_track_buffer=lost_track_buffer
        )

    def update(self, detections):
        return self.tracker.update_with_detections(detections)