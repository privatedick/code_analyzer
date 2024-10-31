import pytest
from unittest.mock import Mock, patch
from config import Config
from project_comparison import ProjectComparison

@pytest.fixture
def project_comparison():
    config = Config()
    return ProjectComparison(config)

@pytest.fixture
def mock_projects():
    project1 = {
        'files': [
            {
                'file': 'file1.py',
                'functions': [{'name': 'func1'}, {'name': 'func2'}],
                'complexity': 5,
                'style_issues': {'line_length': ['issue1'], 'function_names': ['issue2']},
                'imports': ['import os', 'import sys']
            }
        ]
    }
    project2 = {
        'files': [
            {
                'file': 'file1.py',
                'functions': [{'name': 'func1'}, {'name': 'func3'}],
                'complexity': 7,
                'style_issues': {'line_length': ['issue1', 'issue2']},
                'imports': ['import os', 'from datetime import datetime']
            },
            {
                'file': 'file2.py',
                'functions': [{'name': 'func4'}],
                'complexity': 3,
                'style_issues': {},
                'imports': ['import pandas']
            }
        ]
    }
    return project1, project2

def test_compare_projects(project_comparison, mock_projects):
    with patch('project_comparison.CodeAnalysis') as MockCodeAnalysis:
        MockCodeAnalysis.return_value.analyze_project.side_effect = mock_projects
        result = project_comparison.compare_projects('path1', 'path2')
        
        assert result['file_count_diff'] == 1
        assert result['function_count_diff'] == 1
        assert result['complexity_diff'] == 5.0
        assert result['style_issues_diff'] == {'line_length': 1, 'function_names': -1}
        assert len(result['similar_functions']) == 1
        assert result['unique_imports'] == {
            'unique_to_project1': ['import sys'],
            'unique_to_project2': ['from datetime import datetime', 'import pandas']
        }

def test_compare_function_counts(project_comparison, mock_projects):
    project1, project2 = mock_projects
    diff = project_comparison._compare_function_counts(project1, project2)
    assert diff == 1

def test_compare_complexity(project_comparison, mock_projects):
    project1, project2 = mock_projects
    diff = project_comparison._compare_complexity(project1, project2)
    assert diff == 5.0

def test_compare_style_issues(project_comparison, mock_projects):
    project1, project2 = mock_projects
    diff = project_comparison._compare_style_issues(project1, project2)
    assert diff == {'line_length': 1, 'function_names': -1}

def test_find_similar_functions(project_comparison, mock_projects):
    project1, project2 = mock_projects
    similar = project_comparison._find_similar_functions(project1, project2)
    assert len(similar) == 1
    assert similar[0]['function1'] == 'file1.py:func1'
    assert similar[0]['function2'] == 'file1.py:func1'
    assert similar[0]['similarity'] == 1.0

def test_compare_imports(project_comparison, mock_projects):
    project1, project2 = mock_projects
    diff = project_comparison._compare_imports(project1, project2)
    assert diff == {
        'unique_to_project1': ['import sys'],
        'unique_to_project2': ['from datetime import datetime', 'import pandas']
    }
