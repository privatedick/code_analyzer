import os
import json
import logging
import typing
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, List, Union, Callable
from datetime import datetime
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
    description: str
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
            raise ValueError(f"Required configuration value {self.name} is missing")

        if self.value is not None:
            try:
                self.value = self.value_type.value(self.value)
            except (ValueError, TypeError) as e:
                raise ValueError(f"Invalid type for {self.name}: expected {self.value_type.name}") from e

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
        """Add a configuration value to the section."""
        self.values[value.name] = value

    def add_section(self, section: 'ConfigSection') -> None:
        """Add a subsection to the section."""
        self.sections[section.name] = section

    def get_value(self, path: str) -> Any:
        """Get a configuration value using dot notation path."""
        parts = path.split('.')
        if len(parts) == 1:
            return self.values[parts[0]].value
        return self.sections[parts[0]].get_value('.'.join(parts[1:]))

    def to_dict(self) -> Dict[str, Any]:
        """Convert section to dictionary representation."""
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
        self._load_default_config()
        self._setup_logging()

    def _setup_encryption(self) -> None:
        """Initialize encryption for sensitive values."""
        key = os.environ.get('CONFIG_ENCRYPTION_KEY')
        if key:
            self._crypto = Fernet(key.encode())

    def _setup_logging(self) -> None:
        """Configure logging for the configuration system."""
        logging.basicConfig(
            level=self.get_value('logging.level', logging.INFO),
            format=self.get_value(
                'logging.format', 
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        )

    def _load_default_config(self) -> None:
        """Load default configuration values."""
        self.add_section(ConfigSection("logging", "Logging configuration"))
        self.add_value(
            "logging.level",
            logging.INFO,
            ConfigValueType.INTEGER,
            "Logging level"
        )
        self.add_value(
            "logging.format",
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            ConfigValueType.STRING,
            "Logging format string"
        )

    def load_file(self, file_path: Union[str, Path]) -> None:
        """Load configuration from a file."""
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {file_path}")

        try:
            with open(file_path, 'r') as f:
                if file_path.suffix == '.json':
                    config_data = json.load(f)
                elif file_path.suffix in {'.yml', '.yaml'}:
                    config_data = yaml.safe_load(f)
                else:
                    raise ValueError("Unsupported configuration file format")

            self._load_config_data(config_data)
            logger.info(f"Loaded configuration from {file_path}")
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
            raise

    def _load_config_data(self, data: Dict[str, Any], section: Optional[ConfigSection] = None) -> None:
        """Recursively load configuration data into sections."""
        section = section or self.root_section
        
        for key, value in data.items():
            if isinstance(value, dict) and '_metadata' not in value:
                new_section = ConfigSection(key)
                section.add_section(new_section)
                self._load_config_data(value, new_section)
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
                section.add_value(config_value)

    def add_section(self, section: ConfigSection) -> None:
        """Add a configuration section."""
        self.root_section.add_section(section)

    def add_value(
        self, 
        path: str, 
        value: Any, 
        value_type: ConfigValueType,
        description: str,
        secret: bool = False,
        required: bool = True,
        default: Any = None,
        validators: List[Callable[[Any], bool]] = None
    ) -> None:
        """Add a configuration value using dot notation path."""
        parts = path.split('.')
        section = self.root_section
        
        for part in parts[:-1]:
            if part not in section.sections:
                section.add_section(ConfigSection(part))
            section = section.sections[part]

        config_value = ConfigValue(
            name=parts[-1],
            value=value,
            value_type=value_type,
            description=description,
            secret=secret,
            required=required,
            default=default,
            validators=validators
        )
        section.add_value(config_value)

    def get_value(self, path: str, default: Any = None) -> Any:
        """Get a configuration value using dot notation path."""
        try:
            return self.root_section.get_value(path)
        except (KeyError, AttributeError):
            return default

    def save_file(self, file_path: Union[str, Path]) -> None:
        """Save configuration to a file."""
        file_path = Path(file_path)
        data = self.root_section.to_dict()
        
        try:
            with open(file_path, 'w') as f:
                if file_path.suffix == '.json':
                    json.dump(data, f, indent=2)
                elif file_path.suffix in {'.yml', '.yaml'}:
                    yaml.safe_dump(data, f)
                else:
                    raise ValueError("Unsupported configuration file format")
                
            logger.info(f"Saved configuration to {file_path}")
        except Exception as e:
            logger.error(f"Error saving configuration: {str(e)}")
            raise

    def validate_all(self) -> List[str]:
        """Validate all configuration values and return any errors."""
        errors = []
        
        def validate_section(section: ConfigSection, path: str = "") -> None:
            for name, value in section.values.items():
                full_path = f"{path}.{name}" if path else name
                try:
                    value.validate()
                except ValueError as e:
                    errors.append(f"{full_path}: {str(e)}")
            
            for name, subsection in section.sections.items():
                new_path = f"{path}.{name}" if path else name
                validate_section(subsection, new_path)

        validate_section(self.root_section)
        return errors

    def get_environment_override(self, key: str) -> Optional[str]:
        """Get environment variable override for a configuration key."""
        env_key = f"CONFIG_{key.upper().replace('.', '_')}"
        return os.environ.get(env_key)

    def to_dict(self) -> Dict[str, Any]:
        """Convert entire configuration to dictionary."""
        return self.root_section.to_dict()

    def __enter__(self) -> 'ConfigManager':
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Cleanup resources when used as context manager."""
        pass  # Add cleanup if needed

class Config:
    """Main configuration interface for backward compatibility."""
    def __init__(self, config_file: Optional[str] = None):
        self.manager = ConfigManager()
        if config_file:
            self.manager.load_file(config_file)

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        return self.manager.get_value(key, default)

    def set(self, key: str, value: Any) -> None:
        """Set configuration value."""
        self.manager.add_value(
            key,
            value,
            ConfigValueType.STRING,  # Default type for backward compatibility
            f"Value for {key}"  # Default description
        )

    def save_config(self) -> None:
        """Save configuration to file."""
        if hasattr(self, 'config_file'):
            self.manager.save_file(self.config_file)
