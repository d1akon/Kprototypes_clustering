import logging

def setup_logger(name="KPrototypesLogger", log_file="data/output/train.log", level=logging.INFO):
    """
    Sets up a logger with an output file and standard format.
    :param name: Logger name.
    :param log_file: Path to the log file.
    :param level: Logging level.
    :return: Configured logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    #----- Log format
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    #----- Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    #----- File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)

    #----- Add handlers to the logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger
