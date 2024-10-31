import json
from typing import Dict, Any

class Config:
    def __init__(self, config_file: str = 'config.json'):
        self.config_file = config_file
        self.config = self.load_config()

    def load_config(self) -> Dict[str, Any]:
        try:
            with open(self.config_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return self.default_config()

    def save_config(self):
        with open(self.config_file, 'w') as f:
            json.dump(self.config, f, indent=2)

    def default_config(self) -> Dict[str, Any]:
        return {
            'max_line_length': 79,
            'check_function_names': True,
            'check_variable_names': True,
            'check_class_names': True,
            'check_constant_names': True,
            'check_import_order': True,
            'check_spacing': True,
            'complexity_threshold': 10,
            'similarity_threshold': 0.8
        }

    def get(self, key: str, default: Any = None) -> Any:
        return self.config.get(key, default)

    def set(self, key: str, value: Any):
        self.config[key] = value
        self.save_config()
