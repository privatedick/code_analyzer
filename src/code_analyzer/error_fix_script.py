import os
import re

def fix_imports(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
    
    # Add missing imports
    if 'import pytest' not in content:
        content = 'import pytest\n' + content
    if 'from unittest.mock import Mock, patch' not in content:
        content = 'from unittest.mock import Mock, patch\n' + content
    
    # Replace Analyzer with CodeAnalyzer
    content = content.replace('from analyzer import Analyzer', 'from analyzer import CodeAnalyzer')
    content = content.replace('MockAnalyzer', 'MockCodeAnalyzer')
    
    with open(file_path, 'w') as file:
        file.write(content)

def fix_function_signatures(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
    
    # Update function signatures to include 'self' if it's a method in a class
    content = re.sub(r'(class\s+\w+\([^)]*\):\s*\n(?:\s+[^\n]+\n)*?\s+)def ([a-zA-Z_]+)\((?!self)', r'\1def \2(self, ', content)
    
    with open(file_path, 'w') as file:
        file.write(content)

def update_analyzer_file(file_path):
    content = '''
import logging
from typing import Dict, Any
from config import Config
from code_analysis import CodeAnalysis
from project_comparison import ProjectComparison
from visualization import Visualization

class CodeAnalyzer:
    def __init__(self, config: Config):
        self.config = config
        self.logger = self._setup_logger()
        self.code_analysis = CodeAnalysis(config)
        self.project_comparison = ProjectComparison(config)
        self.visualization = Visualization()

    def _setup_logger(self):
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def analyze_project(self, project_path: str) -> Dict[str, Any]:
        try:
            self.logger.info(f"Starting analysis of project: {project_path}")
            analysis_results = self.code_analysis.analyze_project(project_path)
            self.logger.info("Analysis completed successfully")
            return analysis_results
        except Exception as e:
            self.logger.error(f"Error during project analysis: {str(e)}")
            raise

    def compare_projects(self, project1_path: str, project2_path: str) -> Dict[str, Any]:
        try:
            self.logger.info(f"Comparing projects: {project1_path} and {project2_path}")
            comparison_results = self.project_comparison.compare_projects(project1_path, project2_path)
            self.logger.info("Project comparison completed successfully")
            return comparison_results
        except Exception as e:
            self.logger.error(f"Error during project comparison: {str(e)}")
            raise

    def visualize_results(self, analysis_results: Dict[str, Any]) -> None:
        try:
            self.logger.info("Generating visualizations")
            self.visualization.create_visualizations(analysis_results)
            self.logger.info("Visualizations generated successfully")
        except Exception as e:
            self.logger.error(f"Error generating visualizations: {str(e)}")
            raise
'''
    with open(file_path, 'w') as file:
        file.write(content)

def main():
    # Fix imports and function signatures in test files
    test_files = [
        'test_analyzer.py',
        'test_code_analyzis.py',  # Note the spelling
        'test_config.py',
        'test_main.py',
        'test_project_comparison.py',
        'test_visualization.py'
    ]
    for file in test_files:
        if os.path.exists(file):
            fix_imports(file)
            fix_function_signatures(file)
            print(f"Fixed {file}")
        else:
            print(f"Warning: {file} not found")
    
    # Update analyzer.py
    if os.path.exists('analyzer.py'):
        update_analyzer_file('analyzer.py')
        print("Updated analyzer.py")
    else:
        print("Warning: analyzer.py not found")
    
    print("All fixes applied. Please run the tests again.")

if __name__ == "__main__":
    main()
