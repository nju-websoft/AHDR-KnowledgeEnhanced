import logging
import functools
import os
import sys
from pathlib import Path
from datetime import datetime

def get_logger(file_name, parent_log_dir):
    now = datetime.now()
    time_ymd = now.strftime("%Y-%m-%d")
    time_hms = now.strftime("%H-%M-%S")
    # home = str(Path.home())
    # log_dir = os.path.join(home, "Scripts-for-DPR", "log", time_ymd, time_hms)
    log_dir = os.path.join(parent_log_dir, time_ymd, time_hms)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # create logger for prd_ci
    log = logging.getLogger(file_name)
    log.setLevel(level=logging.INFO)

    # create formatter and add it to the handlers
    # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # create file handler for logger.
    fh = logging.FileHandler(os.path.join(log_dir, file_name), encoding='utf-8')
    fh.setLevel(level=logging.DEBUG)
    fh.setFormatter(formatter)

    # create console handler for logger.
    ch = logging.StreamHandler()
    ch.setLevel(level=logging.DEBUG)
    ch.setFormatter(formatter)

    # add handlers to logger.
    log.addHandler(fh)
    log.addHandler(ch)

    return log

py_fname = sys.argv[0].split('/')[-1]
home_path = '/home/zyzhang'
log = get_logger(f"{py_fname}.log", os.path.join(home_path, "code", "keds", "logs"))

def Log(func):
    functools.wraps(func)
    def wrapper(*args, **kw):
        log.info(f"start: {func.__name__}")
        ret = func(*args, **kw)
        log.info(f"end: {func.__name__}")
        return ret
    return wrapper



if __name__ == "__main__":
    log = get_logger("my_logger.log")
    log.info("hello, world!")
