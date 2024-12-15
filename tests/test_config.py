from unittest.mock import Mock, patch
import pytest
import os
import json
from code_analyzer.config import Config

@pytest.fixture
def test_config_file():
    filename = 'test_config.json'
    yield filename
    if os.path.exists(filename):
        os.remove(filename)

def test_default_config():
    config = Config()
    default_config = config.default_config()
    assert default_config['max_line_length'] == 79
    assert default_config['check_function_names'] is True

def test_load_config(test_config_file):
    test_config = {'max_line_length': 100, 'check_function_names': False}
    with open(test_config_file, 'w') as f:
        json.dump(test_config, f)
    
    config = Config(test_config_file)
    assert config.get('max_line_length') == 100
    assert config.get('check_function_names') is False

def test_save_config(test_config_file):
    config = Config(test_config_file)
    config.set('max_line_length', 120)
    config.save_config()

    with open(test_config_file, 'r') as f:
        saved_config = json.load(f)
    
    assert saved_config['max_line_length'] == 120

def test_get_nonexistent_key():
    config = Config()
    assert config.get('nonexistent_key') is None
    assert config.get('nonexistent_key', 'default') == 'default'
