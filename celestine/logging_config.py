import logging
from datetime import datetime
from pathlib import Path

# decorator to time a function
def time_function(func):
    def wrapper(*args, **kwargs):
        start_time = datetime.now()
        return_value = func(*args, **kwargs)
        end_time = datetime.now()
        duration = end_time - start_time
        return return_value, duration

    return wrapper


def get_string_current_time():
    return datetime.now().strftime("%Y-%m-%d-%H:%M:%S")


def log_time(logger, duration, message=""):
    logger.info(
        f"{message}{duration.seconds//3600}:{(duration.seconds//60)%60}:{duration.seconds//60}:{duration.microseconds}"
    )


# create directory for logs if it doesn't exist and name log file current datetime
log_directory = Path("logs")
log_directory.mkdir(exist_ok=True)
log_file = log_directory / Path(f"{get_string_current_time()}.log")

logging.basicConfig(
    filename=log_file,
    level=logging.DEBUG,
    format="%(asctime)s:%(levelname)s: %(message)s",
)
logger = logging.getLogger("celestine")

# suppress the logs from matplotlib
logging.getLogger("matplotlib").setLevel(logging.ERROR)
