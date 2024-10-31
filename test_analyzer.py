import pytest
import logging
from unittest.mock import Mock, patch
from config import Config
from analyzer import CodeAnalyzer

@pytest.fixture
def analyzer():
    config = Config()
    return CodeAnalyzer(config)

def test_analyze_project(analyzer):
    with patch('analyzer.CodeAnalysis') as mock_code_analysis:
        mock_code_analysis.return_value.analyze_project.return_value = {'mock': 'result'}
        result = analyzer.analyze_project('/fake/path')
        assert result == {'mock': 'result'}
        mock_code_analysis.return_value.analyze_project.assert_called_once_with('/fake/path')

def test_compare_projects(analyzer):
    with patch('analyzer.ProjectComparison') as mock_project_comparison:
        mock_project_comparison.return_value.compare_projects.return_value = {'mock': 'comparison'}
        result = analyzer.compare_projects('/fake/path1', '/fake/path2')
        assert result == {'mock': 'comparison'}
        mock_project_comparison.return_value.compare_projects.assert_called_once_with('/fake/path1', '/fake/path2')

def test_visualize_results(analyzer):
    with patch('analyzer.Visualization') as mock_visualization:
        mock_results = {'mock': 'results'}
        analyzer.visualize_results(mock_results)
        mock_visualization.return_value.create_visualizations.assert_called_once_with(mock_results)

def test_logger_setup(analyzer):
    assert analyzer.logger.level == logging.INFO
    assert len(analyzer.logger.handlers) == 1
    assert isinstance(analyzer.logger.handlers[0], logging.StreamHandler)
