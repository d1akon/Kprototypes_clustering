import logging
import yaml

def setup_logger(config_path="config/logging_config.yaml"):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    logging.config.dictConfig(config)
    return logging.getLogger(__name__)
