from unittest.mock import patch, mock_open
import pytest
import os
import json
import yaml
from src.code_analyzer.config import Config  # Uppdaterad import

@pytest.fixture
def test_config_file():
    """Skapar och rensar en testfil för konfiguration."""
    filename = 'test_config.json'
    yield filename
    if os.path.exists(filename):
        os.remove(filename)

def test_default_config():
    """Verifierar att standardkonfigurationen har rätt värden."""
    config = Config()
    default_config = config.default_config()
    assert 'max_line_length' in default_config
    assert 'check_function_names' in default_config
    assert default_config['max_line_length'] == 79
    assert default_config['check_function_names'] is True

def test_load_config(test_config_file):
    """Verifierar att konfigurationen kan laddas från en JSON-fil."""
    test_config = {'max_line_length': 100, 'check_function_names': False}
    with open(test_config_file, 'w') as f:
        json.dump(test_config, f)
    
    config = Config(test_config_file)
    assert config.get('max_line_length') == 100
    assert config.get('check_function_names') is False

def test_load_yaml_config(test_config_file):
    """Verifierar att konfigurationen kan laddas från en YAML-fil."""
    test_config_file = 'test_config.yaml'
    test_config = {'max_line_length': 88, 'check_function_names': True}
    with open(test_config_file, 'w') as f:
        yaml.dump(test_config, f)
    
    config = Config(test_config_file)
    assert config.get('max_line_length') == 88
    assert config.get('check_function_names') is True

def test_save_config(test_config_file):
    """Verifierar att konfigurationen kan sparas till en fil."""
    config = Config(test_config_file)
    config.set('max_line_length', 120)
    config.save_config()

    with open(test_config_file, 'r') as f:
        saved_config = json.load(f)
    
    assert saved_config['max_line_length'] == 120

def test_get_existing_key():
    """Verifierar att get() returnerar rätt värde för en befintlig nyckel."""
    config = Config()
    config.set('custom_key', 'custom_value')
    assert config.get('custom_key') == 'custom_value'

def test_get_nonexistent_key():
    """Verifierar att get() returnerar standardvärdet när en nyckel inte finns."""
    config = Config()
    assert config.get('nonexistent_key') is None
    assert config.get('nonexistent_key', 'default_value') == 'default_value'

def test_set_value():
    """Verifierar att set() ändrar konfigurationsvärdet korrekt."""
    config = Config()
    config.set('max_line_length', 99)
    assert config.get('max_line_length') == 99

def test_save_and_load_config(test_config_file):
    """Verifierar att konfigurationen kan sparas och laddas korrekt."""
    config = Config(test_config_file)
    config.set('max_line_length', 100)
    config.set('check_function_names', False)
    config.save_config()

    loaded_config = Config(test_config_file)
    assert loaded_config.get('max_line_length') == 100
    assert loaded_config.get('check_function_names') is False

def test_invalid_file_format():
    """Verifierar att ett fel kastas när filformatet är ogiltigt."""
    with pytest.raises(ValueError, match="Unsupported configuration file format"):
        config = Config('test_config.txt')

def test_missing_file():
    """Verifierar att ett FileNotFoundError kastas för en saknad fil."""
    with pytest.raises(FileNotFoundError):
        Config('nonexistent_file.json')

def test_invalid_type_handling():
    """Verifierar att felaktiga datatyper hanteras korrekt."""
    with pytest.raises(ValueError):
        ConfigValue(name="test", value="not_an_int", value_type=ConfigValueType.INTEGER)
