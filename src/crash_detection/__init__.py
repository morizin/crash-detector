import logging
from dotenv import load_dotenv
from datetime import datetime
import os
import sys

load_dotenv()
PROJECT_NAME = __name__.split(".")[-1]
__version__ = "0.1.0"

logging_str = "%(asctime)s [%(levelname)s] : %(module)s - %(message)s"

TIMESTAMP = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format=logging_str,
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, f"{TIMESTAMP}.log")),
        logging.StreamHandler(sys.stdout),
    ],
)

logger = logging.getLogger("CrashDetectionLogger")
