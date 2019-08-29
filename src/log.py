from functools import wraps
import logging

log_format = "%(message)s"

logging.basicConfig(filename="../out/out.log",
                    filemode="w",
                    level=logging.DEBUG,
                    format=log_format)

logger = logging.getLogger('root')

def log(func):
    @wraps(func)
    def inner(*args, **kwargs):
        result = func(*args, **kwargs)
        l_msg = f'func:{func.__name__}:args:{args}:kwargs:{kwargs}:result:{result}'
        logger.debug(l_msg)
        return result
    return inner
