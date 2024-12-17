# src/code_analyzer/analyzer.py

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

# End of src/code_analyzer/analyzer.py

# src/code_analyzer/code_analysis.py
import os
import ast
import re
import sys
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Any, Optional

# Anpassad konfigurationsklass
class Config:
    def __init__(self, **kwargs):
        self._config = kwargs

    def get(self, key, default=None):
        return self._config.get(key, default)

# Resultat för analys av filer
@dataclass
class AnalysisResult:
    file_path: Path
    functions: List[Dict[str, Any]]
    complexity: Dict[str, Any]
    style_issues: Dict[str, List[str]]
    imports: Dict[str, Any]
    line_count: int
    error: Optional[str] = None

class CodeAnalysis:
    def __init__(self, config: Config):
        self.config = config
        self._executor = ThreadPoolExecutor(max_workers=os.cpu_count())
        self._setup_logging()

    def _setup_logging(self) -> None:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(self.config.get('log_level', logging.INFO))

    async def analyze_project(self, project_path: str) -> Dict[str, Any]:
        try:
            project_path = Path(project_path)
            if not project_path.is_dir():
                raise ValueError(f"Invalid project path: {project_path}")

            python_files = list(project_path.rglob("*.py"))
            logger.info(f"Found {len(python_files)} Python files to analyze")

            tasks = [self.analyze_file(file) for file in python_files]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            successful_results = [
                r for r in results
                if isinstance(r, AnalysisResult) and not r.error
            ]
            failed_files = [
                str(python_files[i]) for i, r in enumerate(results)
                if isinstance(r, Exception) or getattr(r, 'error', None)
            ]

            if failed_files:
                logger.warning(f"Failed to analyze {len(failed_files)} files")

            return {
                'results': successful_results,
                'failed_files': failed_files,
                'statistics': self._calculate_project_statistics(successful_results)
            }

        except Exception as e:
            logger.error(f"Project analysis failed: {str(e)}")
            raise

    async def analyze_file(self, file_path: Path) -> AnalysisResult:
        try:
            content = file_path.read_text(encoding='utf-8')
            tree = ast.parse(content)

            tasks = [
                self._analyze_functions(tree),
                self._analyze_complexity(tree),
                self._analyze_style(content, tree),
                self._analyze_imports(tree)
            ]

            functions, complexity, style_issues, imports = await asyncio.gather(*tasks)

            return AnalysisResult(
                file_path=file_path,
                functions=functions,
                complexity=complexity,
                style_issues=style_issues,
                imports=imports,
                line_count=len(content.splitlines())
            )

        except Exception as e:
            logger.error(f"Error analyzing file {file_path}: {str(e)}")
            return AnalysisResult(
                file_path=file_path,
                functions=[],
                complexity={},
                style_issues={},
                imports={},
                line_count=0,
                error=str(e)
            )

    async def _analyze_functions(self, tree: ast.AST) -> List[Dict[str, Any]]:
        functions = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                complexity = await self._calculate_node_complexity(node)
                functions.append({
                    'name': node.name,
                    'args': [arg.arg for arg in node.args.args],
                    'returns': self._get_return_annotation(node),
                    'complexity': complexity,
                    'is_async': isinstance(node, ast.AsyncFunctionDef),
                    'line_number': node.lineno,
                    'docstring': ast.get_docstring(node),
                    'decorator_list': [ast.unparse(d) for d in node.decorator_list]
                })
        return functions

    @lru_cache(maxsize=128)
    async def _calculate_node_complexity(self, node: ast.AST) -> int:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            self._calculate_complexity_sync,
            node
        )

    def _calculate_complexity_sync(self, node: ast.AST) -> int:
        complexity = 1
        for child in ast.walk(node):
            if isinstance(child, (
                ast.If, ast.For, ast.While, ast.AsyncFor,
                ast.AsyncWith, ast.Try, ast.ExceptHandler
            )):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
        return complexity

    async def _analyze_complexity(self, tree: ast.AST) -> Dict[str, Any]:
        """Analysera övergripande komplexitetsmetrik."""
        complexity = await self._calculate_node_complexity(tree)
        functions = [
            n for n in ast.walk(tree)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]

        return {
            'total': complexity,
            'average_per_function': (
                complexity / len(functions) if functions else 0
            ),
            'max_allowed': self.config.get('max_complexity', 10)
        }

    async def _analyze_style(self, content: str, tree: ast.AST) -> Dict[str, List[str]]:
        """Analysera kodstil och konventioner."""
        issues = {
            'line_length': [],
            'naming': [],
            'whitespace': [],
            'docstring': []
        }

        await asyncio.gather(
            self._check_line_issues(content, issues),
            self._check_naming_issues(tree, issues),
            self._check_docstring_issues(tree, issues)
        )

        return issues

    async def _analyze_imports(self, tree: ast.AST) -> Dict[str, Any]:
        """Analysera import-satser och deras organisation."""
        imports = {'standard': [], 'third_party': [], 'local': []}
        import_lines = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for name in node.names:
                    import_type = self._classify_import(name.name)
                    imports[import_type].append({
                        'name': name.name,
                        'alias': name.asname,
                        'line': node.lineno
                    })
                    import_lines.append(node.lineno)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    import_type = self._classify_import(node.module)
                    imports[import_type].append({
                        'from': node.module,
                        'names': [(n.name, n.asname) for n in node.names],
                        'line': node.lineno
                    })
                    import_lines.append(node.lineno)

        return {
            'imports': imports,
            'ordered': import_lines == sorted(import_lines),
            'total_count': len(import_lines)
        }

    def _classify_import(self, module_name: str) -> str:
        """Klassificera en import som standardbibliotek, tredjeparts eller lokal."""
        stdlib_modules = set(sys.stdlib_module_names)
        first_part = module_name.split('.')[0]
        if first_part in stdlib_modules:
            return 'standard'
        elif '.' in module_name or first_part in {'config', 'utils', 'core'}:
            return 'local'
        return 'third_party'

    async def _check_line_issues(
        self, content: str, issues: Dict[str, List[str]]
    ) -> None:
        """Kontrollera linjer för längd- och whitespaceproblem."""
        max_length = self.config.get('max_line_length', 79)

        for i, line in enumerate(content.splitlines(), 1):
            line = line.rstrip('\n\r')
            if len(line) > max_length:
                issues['line_length'].append(
                    f"Line {i}: Exceeds {max_length} characters"
                )
            if line.endswith(' '):
                issues['whitespace'].append(
                    f"Line {i}: Contains trailing whitespace"
                )

    async def _check_naming_issues(
        self, tree: ast.AST, issues: Dict[str, List[str]]
    ) -> None:
        """Kontrollera namngivningskonventioner."""
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if not re.match(r'^[a-z_][a-z0-9_]*$', node.name):
                    issues['naming'].append(
                        f"Line {node.lineno}: Function '{node.name}' "
                        "should use snake_case"
                    )
            elif isinstance(node, ast.ClassDef):
                if not re.match(r'^[A-Z][a-zA-Z0-9]*$', node.name):
                    issues['naming'].append(
                        f"Line {node.lineno}: Class '{node.name}' "
                        "should use PascalCase"
                    )

    async def _check_docstring_issues(
        self, tree: ast.AST, issues: Dict[str, List[str]]
    ) -> None:
        """Kontrollera docstrings för funktioner och klasser."""
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if not ast.get_docstring(node):
                    kind = 'Function' if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) else 'Class'
                    issues['docstring'].append(
                        f"Line {node.lineno}: {kind} '{node.name}' lacks a docstring"
                    )

    def _get_return_annotation(self, node: ast.FunctionDef) -> str:
        """Hämta returtyp-annotering från en funktion."""
        if node.returns:
            return ast.unparse(node.returns)
        return 'Any'

    def _calculate_project_statistics(
        self, results: List[AnalysisResult]
    ) -> Dict[str, Any]:
        """Beräkna övergripande statistik för projektet."""
        if not results:
            return {}

        total_complexity = sum(r.complexity.get('total', 0) for r in results)
        total_functions = sum(len(r.functions) for r in results)
        total_lines = sum(r.line_count for r in results)

        return {
            'file_count': len(results),
            'total_lines': total_lines,
            'total_functions': total_functions,
            'average_file_length': total_lines / len(results),
            'average_complexity': total_complexity / len(results),
            'functions_per_file': total_functions / len(results),
            'style_issues_per_file': sum(
                sum(len(issues) for issues in r.style_issues.values())
                for r in results
            ) / len(results) if results else 0
        }

# Setup för att använda logging korrekt
logger = logging.getLogger(__name__)

# End of src/code_analyzer/code_analysis.py

# src/code_analyzer/config.py
import os
import json
import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, List, Callable, Union
import yaml
from cryptography.fernet import Fernet

logger = logging.getLogger(__name__)


class ConfigEnvironment(Enum):
    DEVELOPMENT = "development"
    TESTING = "testing"
    PRODUCTION = "production"


class ConfigValueType(Enum):
    STRING = str
    INTEGER = int
    FLOAT = float
    BOOLEAN = bool
    PATH = Path
    LIST = list
    DICT = dict


@dataclass
class ConfigValue:
    """Represents a single configuration value with metadata and validation."""
    name: str
    value: Any
    value_type: ConfigValueType
    description: str = ""
    secret: bool = False
    required: bool = True
    default: Any = None
    validators: List[Callable[[Any], bool]] = None

    def __post_init__(self):
        self.validators = self.validators or []
        if self.value is None and self.default is not None:
            self.value = self.default
        self.validate()

    def validate(self) -> None:
        """Validate the configuration value."""
        if self.required and self.value is None:
            raise ValueError(f"Required configuration value '{self.name}' is missing")

        if self.value is not None:
            try:
                self.value = self.value_type.value(self.value)
            except (ValueError, TypeError) as e:
                raise ValueError(f"Invalid type for {self.name}: expected {self.value_type.name}, got {type(self.value)}") from e

            for validator in self.validators:
                if not validator(self.value):
                    raise ValueError(f"Validation failed for {self.name}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "value": str(self.value) if self.secret else self.value,
            "type": self.value_type.name,
            "description": self.description,
            "required": self.required,
            "secret": self.secret
        }


class ConfigSection:
    """Represents a group of related configuration values."""
    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self.values: Dict[str, ConfigValue] = {}
        self.sections: Dict[str, 'ConfigSection'] = {}

    def add_value(self, value: ConfigValue) -> None:
        self.values[value.name] = value

    def add_section(self, section: 'ConfigSection') -> None:
        self.sections[section.name] = section

    def get_value(self, path: str) -> Any:
        parts = path.split('.')
        if len(parts) == 1:
            return self.values.get(parts[0]).value
        return self.sections[parts[0]].get_value('.'.join(parts[1:]))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "values": {k: v.to_dict() for k, v in self.values.items()},
            "sections": {k: v.to_dict() for k, v in self.sections.items()}
        }


class ConfigManager:
    """Manages configuration loading, validation, and access."""
    def __init__(self, environment: ConfigEnvironment = ConfigEnvironment.DEVELOPMENT):
        self.environment = environment
        self.root_section = ConfigSection("root", "Root configuration section")
        self._crypto = None
        self._setup_encryption()

    def _setup_encryption(self) -> None:
        key = os.environ.get('CONFIG_ENCRYPTION_KEY')
        if key and len(key) == 44:  # Fernet-nycklar är 44 tecken långa
            self._crypto = Fernet(key.encode())
        else:
            logger.warning("Invalid or missing CONFIG_ENCRYPTION_KEY")

    def load_file(self, file_path: Union[str, Path]) -> None:
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {file_path}")

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                if file_path.suffix in {'.json', '.yaml', '.yml'}:
                    config_data = yaml.safe_load(f) if file_path.suffix in {'.yaml', '.yml'} else json.load(f)
                else:
                    raise ValueError(f"Unsupported configuration file format: {file_path.suffix}")
            self._load_config_data(config_data)
        except Exception as e:
            logger.error(f"Error loading configuration file {file_path}: {e}")
            raise

    def _load_config_data(self, data: Dict[str, Any]) -> None:
        for key, value in data.items():
            if isinstance(value, dict) and '_metadata' not in value:
                new_section = ConfigSection(key)
                self.root_section.add_section(new_section)
                self._load_config_data(value)
            else:
                metadata = value.get('_metadata', {}) if isinstance(value, dict) else {}
                actual_value = value if not isinstance(value, dict) else value.get('value')
                config_value = ConfigValue(
                    name=key,
                    value=actual_value,
                    value_type=ConfigValueType[metadata.get('type', 'STRING')],
                    description=metadata.get('description', ''),
                    secret=metadata.get('secret', False),
                    required=metadata.get('required', True),
                    default=metadata.get('default')
                )
                self.root_section.add_value(config_value)

    def get_value(self, path: str, default: Any = None) -> Any:
        try:
            return self.root_section.get_value(path)
        except (KeyError, AttributeError):
            return default

    def save_file(self, file_path: Union[str, Path]) -> None:
        file_path = Path(file_path)
        data = self.root_section.to_dict()
        with open(file_path, 'w', encoding='utf-8') as f:
            if file_path.suffix in {'.json', '.yaml', '.yml'}:
                yaml.safe_dump(data, f) if file_path.suffix in {'.yaml', '.yml'} else json.dump(data, f, indent=2)
            else:
                raise ValueError(f"Unsupported configuration file format: {file_path.suffix}")

# End of src/code_analyzer/config.py

# src/code_analyzer/main.py
import argparse
from config import Config
from analyzer import CodeAnalyzer

def main():
    # Skapa en parser för kommandoradsargument
    parser = argparse.ArgumentParser(description="Code Analysis Tool")
    parser.add_argument('project_path', type=str, help='Path to the project to analyze')
    parser.add_argument('--compare', type=str, help='Path to another project to compare with', default=None)
    parser.add_argument('--visualize', action='store_true', help='Generate visualizations')
    parser.add_argument('--config', type=str, help='Path to custom config file', default='config.json')
    
    # Parsar argumenten från kommandoraden
    args = parser.parse_args()

    # Ladda konfigurationen
    config = Config(args.config)
    analyzer = CodeAnalyzer(config)

    # Jämför projekt om det anges
    if args.compare:
        results = analyzer.compare_projects(args.project_path, args.compare)
        print("Comparison Results:")
        print(results)
    else:
        # Utför kodanalys om ingen jämförelse är angiven
        results = analyzer.analyze_project(args.project_path)
        print("Analysis Results:")
        print(results)

    # Om visualisering ska genereras
    if args.visualize:
        analyzer.visualize_results(results)
        print("Visualizations have been generated and saved as PNG files.")

# Huvudprogrammet startar här
if __name__ == "__main__":
    main()

# End of src/code_analyzer/main.py

# src/code_analyzer/project_comparison.py
from typing import Dict, Any, List
import difflib
from src.code_analyzer.config import Config
from src.code_analyzer.code_analysis import CodeAnalysis

class ProjectComparison:
    def __init__(self, config: Config):
        self.config = config
        self.code_analysis = CodeAnalysis(config)

    def compare_projects(self, project1_path: str, project2_path: str) -> Dict[str, Any]:
        project1 = self.code_analysis.analyze_project(project1_path)
        project2 = self.code_analysis.analyze_project(project2_path)

        return {
            'file_count_diff': len(project2['files']) - len(project1['files']),
            'function_count_diff': self._compare_function_counts(project1, project2),
            'complexity_diff': self._compare_complexity(project1, project2),
            'style_issues_diff': self._compare_style_issues(project1, project2),
            'similar_functions': self._find_similar_functions(project1, project2),
            'unique_imports': self._compare_imports(project1, project2),
        }

    def _compare_function_counts(self, project1: Dict[str, Any], project2: Dict[str, Any]) -> int:
        count1 = sum(len(file['functions']) for file in project1['files'])
        count2 = sum(len(file['functions']) for file in project2['files'])
        return count2 - count1

    def _compare_complexity(self, project1: Dict[str, Any], project2: Dict[str, Any]) -> float:
        complexity1 = sum(file['complexity'] for file in project1['files'])
        complexity2 = sum(file['complexity'] for file in project2['files'])
        return (complexity2 - complexity1) / len(project1['files'])

    def _compare_style_issues(self, project1: Dict[str, Any], project2: Dict[str, Any]) -> Dict[str, int]:
        issues1 = self._count_style_issues(project1)
        issues2 = self._count_style_issues(project2)
        return {key: issues2.get(key, 0) - issues1.get(key, 0) for key in set(issues1) | set(issues2)}

    def _count_style_issues(self, project: Dict[str, Any]) -> Dict[str, int]:
        counts = {}
        for file in project['files']:
            for issue_type, issues in file['style_issues'].items():
                counts[issue_type] = counts.get(issue_type, 0) + len(issues)
        return counts

    def _find_similar_functions(self, project1: Dict[str, Any], project2: Dict[str, Any]) -> List[Dict[str, Any]]:
        similar_functions = []
        for file1 in project1['files']:
            for func1 in file1['functions']:
                for file2 in project2['files']:
                    for func2 in file2['functions']:
                        similarity = difflib.SequenceMatcher(None, func1['name'], func2['name']).ratio()
                        if similarity > self.config.get('similarity_threshold', 0.8):
                            similar_functions.append({
                                'function1': f"{file1['file']}:{func1['name']}",
                                'function2': f"{file2['file']}:{func2['name']}",
                                'similarity': similarity
                            })
        return similar_functions

    def _compare_imports(self, project1: Dict[str, Any], project2: Dict[str, Any]) -> Dict[str, List[str]]:
        imports1 = set(import_stmt for file in project1['files'] for import_stmt in file['imports'])
        imports2 = set(import_stmt for file in project2['files'] for import_stmt in file['imports'])
        return {
            'unique_to_project1': list(imports1 - imports2),
            'unique_to_project2': list(imports2 - imports1)
        }

# End of src/code_analyzer/project_comparison.py

# src/code_analyzer/visualization.py
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

# End of src/code_analyzer/visualization.py

# src/code_analyzer/__init__.py

# End of src/code_analyzer/__init__.py

# tests/test_analyzer.py
import pytest
import logging
from unittest.mock import Mock, patch
from code_analyzer.config import Config
from code_analyzer.analyzer import CodeAnalyzer

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

# End of tests/test_analyzer.py

# tests/test_code_analysis.py
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

# End of tests/test_code_analysis.py

# tests/test_config.py
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

# End of tests/test_config.py

# tests/test_main.py
from unittest.mock import Mock, patch
import pytest
from unittest.mock import patch, MagicMock
from code_analyzer.main import main

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

# End of tests/test_main.py

# tests/test_project_comparison.py
import pytest
from unittest.mock import Mock, patch
from code_analyzer.config import Config
from code_analyzer.project_comparison import ProjectComparison

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
                'imports': ['import os', 'from code_analyzer.datetime import datetime']
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
            'unique_to_project2': ['from code_analyzer.datetime import datetime', 'import pandas']
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
        'unique_to_project2': ['from code_analyzer.datetime import datetime', 'import pandas']
    }

# End of tests/test_project_comparison.py

# tests/test_visualization.py
from unittest.mock import Mock, patch
import pytest
from unittest.mock import patch, MagicMock
from code_analyzer.visualization import Visualization
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

# End of tests/test_visualization.py

# tests/__init__.py

# End of tests/__init__.py

# pyproject.toml
[tool.poetry]
name = "code_analyzer"
version = "0.1.0"
description = "A tool for analyzing and visualizing code."
authors = ["Your Name <you@example.com>"]
license = "MIT"
readme = "README.md"

# Här definieras vilka kataloger som inkluderas som paket
packages = [
    { include = "code_analyzer", from = "src" }
]

[tool.poetry.dependencies]
python = "^3.11"
matplotlib = "^3.10.0"
pyyaml = "^6.0.2"
cryptography = "^44.0.0"
seaborn = "^0.13.2"

# pytest är inte en direkt "dependency" utan en "dev-dependency"
loguru = "^0.7.3"
[tool.poetry.dev-dependencies]
pytest = "^8.3.4"

[build-system]
requires = ["poetry-core"]
build-backend = "poetry.core.masonry.api"

# End of pyproject.toml

