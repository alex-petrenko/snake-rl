"""
Logging utilities.

"""


import sys
import logging


LOGGING_INITIALIZED = False


def init_logger():
    """Initialize logging facilities for particular script."""
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format='%(asctime)s.%(msecs)03d %(name)s:%(lineno)d %(levelname)s %(message)s',
        datefmt='%m-%d %H:%M:%S',
    )


def get_test_logger():
    global LOGGING_INITIALIZED
    if LOGGING_INITIALIZED is False:
        init_logger()
        LOGGING_INITIALIZED = True

    return logging.getLogger('test')
