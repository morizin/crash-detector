import logging
import os
import sys
from datetime import datetime


def setup_logging(run_id: str) -> None:
    root_logger = logging.getLogger()

    # Prevent duplicate handlers
    if root_logger.handlers:
        return

    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    log_format = "%(asctime)s [%(levelname)s] : %(name)s - %(message)s"
    formatter = logging.Formatter(log_format)

    file_handler = logging.FileHandler(os.path.join(log_dir, f"{run_id}.log"))
    stream_handler = logging.StreamHandler(sys.stdout)

    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(stream_handler)


if __name__ == "__main__":
    run_id = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
    os.environ["CRASH_DETECTION_RUN_ID"] = run_id
    setup_logging(run_id)

    from src.crash_detection.pipelines.base import BasePipeline

    pipeline = BasePipeline()
    pipeline.kickoff()
