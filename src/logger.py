import logging
import os


class AppLogger:
    def __init__(self, log_dir="logs"):
        os.makedirs(log_dir, exist_ok=True)

        log_file = os.path.join(log_dir, "face_tracker.log")

        logging.basicConfig(
            filename=log_file,
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s"
        )

        self.logger = logging.getLogger("face_tracker")

    def info(self, message):
        self.logger.info(message)

    def warning(self, message):
        self.logger.warning(message)

    def error(self, message):
        self.logger.error(message)

    def debug(self, message):
        self.logger.debug(message)