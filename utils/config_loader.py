import yaml

def load_config(config_path="config/config.yaml"):
    """
    Loads configurations from a YAML conf file.
    :param config_path: Path to the configuration file.
    :return: Dictionary with the loaded configurations.
    """
    with open(config_path, "r") as file:
        return yaml.safe_load(file)
