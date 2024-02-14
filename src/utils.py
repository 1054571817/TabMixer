import yaml
from types import SimpleNamespace


class YamlObject:
    def __init__(self, yaml_data):
        self._data = yaml_data
        self._cache = {}

    def __getattr__(self, item):
        if item not in self._data:
            raise AttributeError(f"No attribute '{item}' found in YAML data")

        value = self._data[item]
        self._cache[item] = value  # Cache the value for subsequent access
        return value

    def __setattr__(self, key, value):
        self._data[key] = value

    def _dict_to_object(self, data):
        if isinstance(data, dict):
            return SimpleNamespace(**{key: self._dict_to_object(value) for key, value in data.items()})
        elif isinstance(data, list):
            return [self._dict_to_object(item) for item in data]
        else:
            return data


def dict_to_object(data):
    if isinstance(data, dict):
        return SimpleNamespace(**{key: dict_to_object(value) for key, value in data.items()})
    elif isinstance(data, list):
        return [dict_to_object(item) for item in data]
    else:
        return data


def load_yaml(yaml_file):
    with open(yaml_file, 'r') as f:
        yaml_data = yaml.safe_load(f)

    yaml_object = dict_to_object(yaml_data)
    return yaml_object
