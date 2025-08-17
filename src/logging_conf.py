import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from .config import PATHS

def setup_logging():
    PATHS.reports_dir.mkdir(parents=True, exist_ok=True)
    log_file = PATHS.reports_dir / "pipeline.log"

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))

 
    fh = RotatingFileHandler(log_file, maxBytes=5_000_000, backupCount=3)
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))

    
    logger.handlers.clear()
    logger.addHandler(ch)
    logger.addHandler(fh)
