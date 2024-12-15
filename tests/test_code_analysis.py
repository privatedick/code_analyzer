from unittest.mock import Mock, patch
import pytest
import ast
import logging
from code_analyzer.config import Config
from code_analyzer.code_analysis import CodeAnalysis

@pytest.fixture
def code_analysis():
    config = Config()
    return CodeAnalysis(config)

def test_extract_functions(code_analysis):
    code = """
def func1(a, b):
    return a + b

def func2(x):
    if x > 0:
        return x
    else:
        return -x
"""
    tree = ast.parse(code)
    functions = code_analysis.extract_functions(tree)
    assert len(functions) == 2
    assert functions[0]['name'] == 'func1'
    assert functions[0]['args'] == ['a', 'b']
    assert functions[0]['complexity'] == 1
    assert functions[1]['name'] == 'func2'
    assert functions[1]['args'] == ['x']
    assert functions[1]['complexity'] == 2

def test_calculate_complexity(code_analysis):
    code = """
def complex_func(x):
    if x > 0:
        for i in range(x):
            if i % 2 == 0:
                print(i)
    else:
        while x < 0:
            x += 1
    return x
"""
    tree = ast.parse(code)
    complexity = code_analysis.calculate_complexity(tree)
    assert complexity == 4  # 1 (base) + 1 (if) + 1 (for) + 1 (if inside for)

def test_check_style(code_analysis):
    code = """
import sys
import os

def badFunctionName(x):
    VeryLongVariableName = x + 1
    return VeryLongVariableName

class lowercase_class:
    pass

print("This is a very long line that definitely exceeds the maximum line length set in the configuration")
"""
    tree = ast.parse(code)
    issues = code_analysis.check_style(code, tree)
    assert 'line_length' in issues
    assert 'function_names' in issues
    assert 'variable_names' in issues
    assert 'class_names' in issues
    assert len(issues['line_length']) == 1
    assert len(issues['function_names']) == 1
    assert len(issues['variable_names']) == 1
    assert len(issues['class_names']) == 1

def test_extract_imports(code_analysis):
    code = """
import os
import sys
from code_analyzer.datetime import datetime
from code_analyzer.math import sqrt, pi
"""
    tree = ast.parse(code)
    imports = code_analysis.extract_imports(tree)
    assert len(imports) == 4
    assert 'os' in imports
    assert 'sys' in imports
    assert 'from code_analyzer.datetime import datetime' in imports
    assert 'from code_analyzer.math import sqrt, pi' in imports

def test_logger_setup(code_analysis):
    assert code_analysis.logger.level == logging.INFO
    assert len(code_analysis.logger.handlers) == 1
    assert isinstance(code_analysis.logger.handlers[0], logging.StreamHandler)
