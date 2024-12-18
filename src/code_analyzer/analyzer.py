
import logging
from typing import Dict, Any
from src.code_analyzer.config import Config
from src.code_analyzer.code_analysis import CodeAnalysis
from src.code_analyzer.project_comparison import ProjectComparison
from src.code_analyzer.visualization import Visualization

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
