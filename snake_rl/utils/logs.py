"""
Logging utilities.

"""


import sys
import logging


def init_logger():
    """Initialize logging facilities for particular script."""
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format='%(asctime)s.%(msecs)03d %(name)s:%(lineno)d %(levelname)s %(message)s',
        datefmt='%m-%d %H:%M:%S',
    )
