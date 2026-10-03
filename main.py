from src.pipeline import FaceTrackingPipeline


def main():
    pipeline = FaceTrackingPipeline()
    pipeline.process_video()


if __name__ == "__main__":
    main()