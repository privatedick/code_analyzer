import os
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib.figure import Figure
from matplotlib.axes import Axes


logger = logging.getLogger(__name__)

@dataclass
class VisualizationConfig:
    """Configuration for visualization settings."""
    output_dir: Path
    color_scheme: Dict[str, str]
    figure_size: Tuple[int, int]
    dpi: int
    output_format: str
    interactive: bool
    max_points: int = 1000
    font_size: int = 10
    show_grid: bool = True
    timestamp_format: str = "%Y%m%d_%H%M%S"

    def __post_init__(self):
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

class VisualizationBase(ABC):
    """Base class for all visualizations."""

    def __init__(self, config: VisualizationConfig):
        self.config = config
        self._setup_style()

    def _setup_style(self) -> None:
        """Configure matplotlib style settings."""
        plt.style.use('seaborn')
        sns.set_palette(list(self.config.color_scheme.values()))
        plt.rcParams.update({
            'font.size': self.config.font_size,
            'figure.figsize': self.config.figure_size,
            'figure.dpi': self.config.dpi
        })

    def _create_figure(self) -> Tuple[Figure, Axes]:
        """Create a new figure and axes with configured settings."""
        fig, ax = plt.subplots(figsize=self.config.figure_size)
        if self.config.show_grid:
            ax.grid(True, alpha=0.3)
        return fig, ax

    def _save_figure(self, fig: Figure, name: str) -> Path:
        """Save figure with configured settings."""
        timestamp = datetime.now().strftime(self.config.timestamp_format)
        filename = f"{name}_{timestamp}.{self.config.output_format}"
        output_path = self.config.output_dir / filename

        try:
            fig.savefig(output_path, dpi=self.config.dpi, bbox_inches='tight')
            logger.info(f"Saved visualization to {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Failed to save visualization: {str(e)}")
            raise
        finally:
            plt.close(fig)

    @abstractmethod
    def create(self, data: Any) -> Path:
        """Create and save the visualization."""
        pass

class ComplexityVisualization(VisualizationBase):
    """Visualization for code complexity metrics."""

    def create(self, data: Dict[str, Any]) -> Path:
        fig, ax = self._create_figure()

        try:
            self._plot_complexity_data(data, ax)
            return self._save_figure(fig, "complexity_analysis")
        except Exception as e:
            logger.error(f"Failed to create complexity visualization: {str(e)}")
            plt.close(fig)
            raise

    def _plot_complexity_data(self, data: Dict[str, Any], ax: Axes) -> None:
        """Plot complexity metrics."""
        df = self._prepare_complexity_data(data)

        sns.barplot(
            data=df,
            x='file',
            y='complexity',
            hue='type',
            ax=ax
        )

        ax.set_title('Code Complexity Analysis')
        ax.set_xlabel('Files')
        ax.set_ylabel('Complexity Score')
        ax.tick_params(axis='x', rotation=45)

    def _prepare_complexity_data(self, data: Dict[str, Any]) -> pd.DataFrame:
        """Prepare complexity data for visualization."""
        records = []
        for file_data in data['results']:
            file_name = Path(file_data['file_path']).name
            records.append({
                'file': file_name,
                'complexity': file_data['complexity']['total'],
                'type': 'Total'
            })
            if 'average_per_function' in file_data['complexity']:
                records.append({
                    'file': file_name,
                    'complexity': file_data['complexity']['average_per_function'],
                    'type': 'Per Function'
                })

        return pd.DataFrame(records)

class StyleIssuesVisualization(VisualizationBase):
    """Visualization for code style issues."""

    def create(self, data: Dict[str, Any]) -> Path:
        fig, ax = self._create_figure()

        try:
            self._plot_style_issues(data, ax)
            return self._save_figure(fig, "style_issues")
        except Exception as e:
            logger.error(f"Failed to create style issues visualization: {str(e)}")
            plt.close(fig)
            raise

    def _plot_style_issues(self, data: Dict[str, Any], ax: Axes) -> None:
        """Plot style issues distribution."""
        df = self._prepare_style_data(data)

        sns.heatmap(
            data=df,
            annot=True,
            fmt='d',
            cmap='YlOrRd',
            ax=ax
        )

        ax.set_title('Code Style Issues Distribution')
        ax.set_xlabel('Issue Type')
        ax.set_ylabel('File')

    def _prepare_style_data(self, data: Dict[str, Any]) -> pd.DataFrame:
        """Prepare style issues data for visualization."""
        records = []
        for file_data in data['results']:
            file_name = Path(file_data['file_path']).name
            issues_dict = {
                'file': file_name,
                **{k: len(v) for k, v in file_data['style_issues'].items()}
            }
            records.append(issues_dict)

        df = pd.DataFrame(records)
        return df.set_index('file')

class FunctionMetricsVisualization(VisualizationBase):
    """Visualization for function-level metrics."""

    def create(self, data: Dict[str, Any]) -> Path:
        fig, ax = self._create_figure()

        try:
            self._plot_function_metrics(data, ax)
            return self._save_figure(fig, "function_metrics")
        except Exception as e:
            logger.error(f"Failed to create function metrics visualization: {str(e)}")
            plt.close(fig)
            raise

    def _plot_function_metrics(self, data: Dict[str, Any], ax: Axes) -> None:
        """Plot function-level metrics."""
        df = self._prepare_function_data(data)

        sns.scatterplot(
            data=df,
            x='complexity',
            y='args_count',
            size='lines',
            hue='is_async',
            ax=ax
        )

        ax.set_title('Function Metrics Analysis')
        ax.set_xlabel('Complexity')
        ax.set_ylabel('Number of Arguments')

    def _prepare_function_data(self, data: Dict[str, Any]) -> pd.DataFrame:
        """Prepare function-level data for visualization."""
        records = []
        for file_data in data['results']:
            for func in file_data['functions']:
                records.append({
                    'file': Path(file_data['file_path']).name,
                    'function': func['name'],
                    'complexity': func['complexity'],
                    'args_count': len(func['args']),
                    'is_async': func['is_async'],
                    'lines': func.get('line_count', 10)  # Default size if not available
                })

        return pd.DataFrame(records)

class Visualization:
    """Main visualization coordinator class."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = VisualizationConfig(
            output_dir=Path(config.get('output_dir', 'visualizations')),
            color_scheme={
                'primary': '#2196F3',
                'secondary': '#FF9800',
                'error': '#F44336',
                'success': '#4CAF50'
            },
            figure_size=(12, 8),
            dpi=100,
            output_format='png',
            interactive=config.get('interactive', False)
        )

        self.visualizers = {
            'complexity': ComplexityVisualization(self.config),
            'style': StyleIssuesVisualization(self.config),
            'functions': FunctionMetricsVisualization(self.config)
        }

    def create_visualizations(self, analysis_results: Dict[str, Any]) -> Dict[str, Path]:
        """Create all visualizations for the analysis results."""
        outputs = {}

        for name, visualizer in self.visualizers.items():
            logger.info(f"Creating {name} visualization...")
            outputs[name] = visualizer.create(analysis_results)

        logger.info("Successfully created all visualizations")
        return outputs
