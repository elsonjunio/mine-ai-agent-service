import logging
from pythonjsonlogger import jsonlogger


def setup_logger(level: str = 'INFO'):
    """Configures the global application logger in JSON format.

    Args:
        level (str): "INFO", "DEBUG", "ERROR"

    Returns:
        None
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    handler = logging.StreamHandler()

    formatter = jsonlogger.JsonFormatter(
        '%(asctime)s %(levelname)s %(name)s %(message)s'
    )

    handler.setFormatter(formatter)

    logging.basicConfig(level=numeric_level, handlers=[handler])

    logging.getLogger().info('Logger initialized', extra={'level': level})
