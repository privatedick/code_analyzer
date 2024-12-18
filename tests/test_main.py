from unittest.mock import Mock, patch
import pytest
from unittest.mock import patch, MagicMock
from src.code_analyzer.main import main

def test_main_analyze():
    with patch('argparse.ArgumentParser.parse_args') as mock_args, \
         patch('config.Config') as MockConfig, \
         patch('analyzer.Analyzer') as MockCodeAnalyzer:
        
        mock_args.return_value = MagicMock(
            project_path='/fake/path',
            compare=None,
            visualize=False,
            config='config.json'
        )
        mock_analyzer = MockCodeAnalyzer.return_value
        mock_analyzer.analyze_project.return_value = {'fake': 'results'}

        main()

        MockConfig.assert_called_once_with('config.json')
        mock_analyzer.analyze_project.assert_called_once_with('/fake/path')

def test_main_compare():
    with patch('argparse.ArgumentParser.parse_args') as mock_args, \
         patch('config.Config') as MockConfig, \
         patch('analyzer.Analyzer') as MockCodeAnalyzer:
        
        mock_args.return_value = MagicMock(
            project_path='/fake/path1',
            compare='/fake/path2',
            visualize=False,
            config='config.json'
        )
        mock_analyzer = MockCodeAnalyzer.return_value
        mock_analyzer.compare_projects.return_value = {'fake': 'comparison'}

        main()

        MockConfig.assert_called_once_with('config.json')
        mock_analyzer.compare_projects.assert_called_once_with('/fake/path1', '/fake/path2')

def test_main_visualize():
    with patch('argparse.ArgumentParser.parse_args') as mock_args, \
         patch('config.Config') as MockConfig, \
         patch('analyzer.Analyzer') as MockCodeAnalyzer:
        
        mock_args.return_value = MagicMock(
            project_path='/fake/path',
            compare=None,
            visualize=True,
            config='config.json'
        )
        mock_analyzer = MockCodeAnalyzer.return_value
        mock_analyzer.analyze_project.return_value = {'fake': 'results'}

        main()

        MockConfig.assert_called_once_with('config.json')
        mock_analyzer.analyze_project.assert_called_once_with('/fake/path')
        mock_analyzer.visualize_results.assert_called_once_with({'fake': 'results'})
