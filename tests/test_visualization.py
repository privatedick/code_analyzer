from unittest.mock import Mock, patch
import pytest
from unittest.mock import patch, MagicMock
from src.code_analyzer.visualization import Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

@pytest.fixture
def mock_analysis_results():
    return {
        'files': [
            {
                'file': 'file1.py',
                'functions': [
                    {'name': 'func1', 'complexity': 3},
                    {'name': 'func2', 'complexity': 5}
                ],
                'style_issues': {
                    'line_length': ['issue1'],
                    'function_names': ['issue2', 'issue3']
                }
            },
            {
                'file': 'file2.py',
                'functions': [
                    {'name': 'func3', 'complexity': 2},
                    {'name': 'func4', 'complexity': 4},
                    {'name': 'func5', 'complexity': 1}
                ],
                'style_issues': {
                    'line_length': ['issue4', 'issue5'],
                    'variable_names': ['issue6']
                }
            }
        ]
    }

@pytest.fixture
def visualization():
    return Visualization()

def test_create_visualizations(visualization, mock_analysis_results):
    with patch.object(visualization, '_create_complexity_bar_chart') as mock_bar, \
         patch.object(visualization, '_create_style_issues_pie_chart') as mock_pie, \
         patch.object(visualization, '_create_function_count_histogram') as mock_hist, \
         patch.object(visualization, '_create_complexity_heatmap') as mock_heatmap:
        
        visualization.create_visualizations(mock_analysis_results)
        
        mock_bar.assert_called_once_with(mock_analysis_results)
        mock_pie.assert_called_once_with(mock_analysis_results)
        mock_hist.assert_called_once_with(mock_analysis_results)
        mock_heatmap.assert_called_once_with(mock_analysis_results)

@patch('matplotlib.pyplot.savefig')
@patch('matplotlib.pyplot.close')
def test_create_complexity_bar_chart(mock_close, mock_savefig, visualization, mock_analysis_results):
    visualization._create_complexity_bar_chart(mock_analysis_results)
    mock_savefig.assert_called_once_with('complexity_bar_chart.png')
    mock_close.assert_called_once()

@patch('matplotlib.pyplot.savefig')
@patch('matplotlib.pyplot.close')
def test_create_style_issues_pie_chart(mock_close, mock_savefig, visualization, mock_analysis_results):
    visualization._create_style_issues_pie_chart(mock_analysis_results)
    mock_savefig.assert_called_once_with('style_issues_pie_chart.png')
    mock_close.assert_called_once()

@patch('matplotlib.pyplot.savefig')
@patch('matplotlib.pyplot.close')
def test_create_function_count_histogram(mock_close, mock_savefig, visualization, mock_analysis_results):
    visualization._create_function_count_histogram(mock_analysis_results)
    mock_savefig.assert_called_once_with('function_count_histogram.png')
    mock_close.assert_called_once()

@patch('matplotlib.pyplot.savefig')
@patch('matplotlib.pyplot.close')
@patch('seaborn.heatmap')
def test_create_complexity_heatmap(mock_heatmap, mock_close, mock_savefig, visualization, mock_analysis_results):
    visualization._create_complexity_heatmap(mock_analysis_results)
    mock_savefig.assert_called_once_with('complexity_heatmap.png')
    mock_close.assert_called_once()
    mock_heatmap.assert_called_once()
