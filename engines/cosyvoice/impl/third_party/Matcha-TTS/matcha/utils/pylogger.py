import logging

# Patched for inference: removed lightning dependency
# from lightning.pytorch.utilities import rank_zero_only

# Simple stub for rank_zero_only (multi-GPU decorator not needed for inference)
def rank_zero_only(func):
    """Stub decorator that just returns the function as-is."""
    return func


def get_pylogger(name: str = __name__) -> logging.Logger:
    """Initializes a multi-GPU-friendly python command line logger.

    :param name: The name of the logger, defaults to ``__name__``.

    :return: A logger object.
    """
    logger = logging.getLogger(name)

    # this ensures all logging levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU setup
    logging_levels = ("debug", "info", "warning", "error", "exception", "fatal", "critical")
    for level in logging_levels:
        setattr(logger, level, rank_zero_only(getattr(logger, level)))

    return logger
