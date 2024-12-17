import os
import json
import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, List, Callable, Union
import yaml
from cryptography.fernet import Fernet

logger = logging.getLogger(__name__)


class ConfigEnvironment(Enum):
    DEVELOPMENT = "development"
    TESTING = "testing"
    PRODUCTION = "production"


class ConfigValueType(Enum):
    STRING = str
    INTEGER = int
    FLOAT = float
    BOOLEAN = bool
    PATH = Path
    LIST = list
    DICT = dict


@dataclass
class ConfigValue:
    """Represents a single configuration value with metadata and validation."""
    name: str
    value: Any
    value_type: ConfigValueType
    description: str = ""
    secret: bool = False
    required: bool = True
    default: Any = None
    validators: List[Callable[[Any], bool]] = None

    def __post_init__(self):
        self.validators = self.validators or []
        if self.value is None and self.default is not None:
            self.value = self.default
        self.validate()

    def validate(self) -> None:
        """Validate the configuration value."""
        if self.required and self.value is None:
            raise ValueError(f"Required configuration value '{self.name}' is missing")

        if self.value is not None:
            try:
                self.value = self.value_type.value(self.value)
            except (ValueError, TypeError) as e:
                raise ValueError(f"Invalid type for {self.name}: expected {self.value_type.name}, got {type(self.value)}") from e

            for validator in self.validators:
                if not validator(self.value):
                    raise ValueError(f"Validation failed for {self.name}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "value": str(self.value) if self.secret else self.value,
            "type": self.value_type.name,
            "description": self.description,
            "required": self.required,
            "secret": self.secret
        }


class ConfigSection:
    """Represents a group of related configuration values."""
    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self.values: Dict[str, ConfigValue] = {}
        self.sections: Dict[str, 'ConfigSection'] = {}

    def add_value(self, value: ConfigValue) -> None:
        self.values[value.name] = value

    def add_section(self, section: 'ConfigSection') -> None:
        self.sections[section.name] = section

    def get_value(self, path: str) -> Any:
        parts = path.split('.')
        if len(parts) == 1:
            return self.values.get(parts[0]).value
        return self.sections[parts[0]].get_value('.'.join(parts[1:]))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "values": {k: v.to_dict() for k, v in self.values.items()},
            "sections": {k: v.to_dict() for k, v in self.sections.items()}
        }


class ConfigManager:
    """Manages configuration loading, validation, and access."""
    def __init__(self, environment: ConfigEnvironment = ConfigEnvironment.DEVELOPMENT):
        self.environment = environment
        self.root_section = ConfigSection("root", "Root configuration section")
        self._crypto = None
        self._setup_encryption()

    def _setup_encryption(self) -> None:
        key = os.environ.get('CONFIG_ENCRYPTION_KEY')
        if key and len(key) == 44:  # Fernet-nycklar är 44 tecken långa
            self._crypto = Fernet(key.encode())
        else:
            logger.warning("Invalid or missing CONFIG_ENCRYPTION_KEY")

    def load_file(self, file_path: Union[str, Path]) -> None:
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {file_path}")

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                if file_path.suffix in {'.json', '.yaml', '.yml'}:
                    config_data = yaml.safe_load(f) if file_path.suffix in {'.yaml', '.yml'} else json.load(f)
                else:
                    raise ValueError(f"Unsupported configuration file format: {file_path.suffix}")
            self._load_config_data(config_data)
        except Exception as e:
            logger.error(f"Error loading configuration file {file_path}: {e}")
            raise

    def _load_config_data(self, data: Dict[str, Any]) -> None:
        for key, value in data.items():
            if isinstance(value, dict) and '_metadata' not in value:
                new_section = ConfigSection(key)
                self.root_section.add_section(new_section)
                self._load_config_data(value)
            else:
                metadata = value.get('_metadata', {}) if isinstance(value, dict) else {}
                actual_value = value if not isinstance(value, dict) else value.get('value')
                config_value = ConfigValue(
                    name=key,
                    value=actual_value,
                    value_type=ConfigValueType[metadata.get('type', 'STRING')],
                    description=metadata.get('description', ''),
                    secret=metadata.get('secret', False),
                    required=metadata.get('required', True),
                    default=metadata.get('default')
                )
                self.root_section.add_value(config_value)

    def get_value(self, path: str, default: Any = None) -> Any:
        try:
            return self.root_section.get_value(path)
        except (KeyError, AttributeError):
            return default

    def save_file(self, file_path: Union[str, Path]) -> None:
        file_path = Path(file_path)
        data = self.root_section.to_dict()
        with open(file_path, 'w', encoding='utf-8') as f:
            if file_path.suffix in {'.json', '.yaml', '.yml'}:
                yaml.safe_dump(data, f) if file_path.suffix in {'.yaml', '.yml'} else json.dump(data, f, indent=2)
            else:
                raise ValueError(f"Unsupported configuration file format: {file_path.suffix}")
